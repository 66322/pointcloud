"""Shared decoded raw frames on a local filesystem, before stochastic transforms.

Array files use the kernel page cache across workers and GPUs. Each consumer
copies a frame before augmentation. No tensors, CUDA objects or stochastic
training outputs are persisted here.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys

import numpy as np

from .common import fresh_dir, git_state, read_json, timestamp, write_json

_dataset = None
_output = None


def dataset_identity(dataset):
    return {"dataset_class": type(dataset).__module__ + "." + type(dataset).__name__,
            "data_root": str(Path(dataset.data_root).resolve()),
            "with_images": bool(dataset.with_images), "ignore_index": int(dataset.ignore_index),
            "learning_map": {str(k): int(v) for k, v in dataset.learning_map.items()}}


class CachedRawReader:
    def __init__(self, dataset, root):
        self.root = Path(root).resolve()
        manifest = read_json(self.root / "manifest.json")
        if manifest.get("status") != "ready" or manifest.get("schema") != 1:
            raise ValueError("Raw cache requires a completed schema-1 manifest")
        if manifest["dataset"] != dataset_identity(dataset):
            raise ValueError("Raw cache dataset identity differs from the requested dataset")
        self.entries = manifest["entries"]
        # A saved bound get_data would unpickle via getattr(dataset, "get_data")
        # after that attribute was replaced by this reader, causing recursion.
        self.dataset = dataset
        self.original, self.name = type(dataset).get_data, type(dataset).get_data_name
        self.hits = self.misses = 0

    def __call__(self, index):
        token = str(self.name(self.dataset, index))
        entry = self.entries.get(token)
        if entry is None:
            self.misses += 1
            return self.original(self.dataset, index)
        folder = (self.root / "frames" / entry).resolve()
        if not folder.is_relative_to(self.root / "frames"):
            raise ValueError("Raw cache entry escapes the frame directory")
        metadata = read_json(folder / "metadata.json")
        if metadata["name"] != token:
            raise ValueError("Raw cache frame identity differs")
        data = dict(metadata["scalars"])
        for key in metadata["arrays"]:
            # The OS shares cached file pages. Writable arrays belong to this
            # request, so in-place transforms cannot modify another epoch.
            data[key] = np.load(folder / (key + ".npy"), mmap_mode="r", allow_pickle=False).copy()
        self.hits += 1
        return data


def attach_raw_cache(dataset, root):
    if root:
        reader = CachedRawReader(dataset, root)
        dataset.get_data = reader
        return reader
    return None


def _setup(paths, split, output):
    global _dataset, _output
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"
    sys.path.insert(0, paths["ditr_root"])
    from pointcept.engines.defaults import default_config_parser
    from pointcept.datasets import build_dataset
    import torch
    torch.set_num_threads(1)
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass
    cfg = default_config_parser(paths["config"], {})
    config = cfg.data[split]
    config.data_root = paths["data_root"]
    _dataset, _output = build_dataset(config), Path(output)
    if type(_dataset).__name__ != "NuScenesDataset":
        raise ValueError("Decoded raw cache currently supports upstream NuScenesDataset")
    if len(_dataset) != len(_dataset.data_list):
        raise ValueError("Build the raw cache from a dataset with loop=1")


def write_frame(data, directory):
    directory.mkdir()
    arrays, scalars = [], {}
    for key, value in data.items():
        if not key.isidentifier():
            raise ValueError("Raw cache field must be a simple identifier")
        if isinstance(value, np.ndarray):
            np.save(directory / (key + ".npy"), value, allow_pickle=False)
            arrays.append(key)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            scalars[key] = value
        else:
            raise ValueError("Unsupported raw field " + key)
    write_json(directory / "metadata.json", {
        "name": str(data["name"]), "arrays": arrays, "scalars": scalars})
    return sum(file.stat().st_size for file in directory.iterdir())


def _build_frame(index):
    data = _dataset.get_data(index)
    entry = f"{index:08d}"
    size = write_frame(data, _output / "frames" / entry)
    return str(data["name"]), entry, size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="configs/paths.local.json")
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-gib", type=float, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.max_gib <= 0 or args.workers < 1:
        parser.error("Positive cache budget and worker count required")
    paths = read_json(args.paths)
    output = fresh_dir(args.out)
    (output / "frames").mkdir()
    write_json(output / "build_status.json", {"status": "building", "started_at": timestamp()})
    try:
        _setup(paths, args.split, output)
        if not len(_dataset):
            raise ValueError("Empty dataset")
        budget = int(args.max_gib * 1024**3)
        token, entry, used = _build_frame(0)
        if used > budget:
            raise ValueError("A single decoded frame exceeds the cache budget")
        entries = {token: entry}
        print(f"RAW CACHE: {_dataset.__class__.__name__}, {len(_dataset)} frames, "
              f"{args.workers} processes, budget {args.max_gib:.2f} GiB, "
              f"first frame {used / 1024**2:.2f} MiB", flush=True)
        cursor, stopped = 1, False
        # At most workers candidate frames are being decoded/written at once.
        # Temporary disk use can exceed the published budget by those frames.
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                 initializer=_setup, initargs=(paths, args.split, str(output))) as pool:
            pending = {}
            while pending or (cursor < len(_dataset) and not stopped):
                while not stopped and cursor < len(_dataset) and len(pending) < args.workers:
                    pending[pool.submit(_build_frame, cursor)] = cursor
                    cursor += 1
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    token, entry, size = future.result()
                    if used + size <= budget:
                        entries[token] = entry
                        used += size
                    else:
                        stopped = True
                        # This exact directory was created in this fresh cache.
                        shutil.rmtree(output / "frames" / entry)
                if len(entries) % 100 == 0:
                    print(f"Decoded {len(entries)}/{len(_dataset)}; {used / 1024**3:.2f} GiB", flush=True)
        manifest = {"schema": 1, "status": "ready", "created_at": timestamp(),
                    "split": args.split, "dataset": dataset_identity(_dataset),
                    "upstream_git": git_state(paths["ditr_root"]), "entries": entries,
                    "cached_frames": len(entries), "total_frames": len(_dataset),
                    "bytes": used, "max_bytes": budget,
                    "semantics": "raw get_data only; independent writable copy before transforms"}
        write_json(output / "manifest.json.tmp", manifest)
        os.replace(output / "manifest.json.tmp", output / "manifest.json")
        write_json(output / "build_status.json", {"status": "ready", "ended_at": timestamp()})
        print(f"Cache ready: {len(entries)}/{len(_dataset)} frames; {used / 1024**3:.2f} GiB", flush=True)
    except BaseException as error:
        write_json(output / "build_status.json", {"status": "failed", "error": str(error), "ended_at": timestamp()})
        raise


if __name__ == "__main__":
    main()
