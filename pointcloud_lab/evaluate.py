"""Run a fresh, single-GPU upstream tester with an isolated runtime intervention."""
import argparse
import importlib
import os
import sys
from pathlib import Path

from .common import fresh_dir, git_state, read_json, select_infos, sha256, timestamp, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu", required=True, choices=("0", "1"))
    parser.add_argument("--tokens", required=True, help="JSON list; same for every paired condition")
    parser.add_argument("--expected-frames", type=int, default=100)
    parser.add_argument("--allow-upstream-drift", action="store_true")
    args = parser.parse_args()
    # Must happen before ANY import of torch, timm, or upstream pointcept.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    paths, experiment, tokens = read_json(args.paths), read_json(args.experiment), read_json(args.tokens)
    if len(tokens) != args.expected_frames:
        raise ValueError("Token count differs from --expected-frames")
    allowed = {"id", "assignment", "camera_order", "dx", "dy", "seed", "description"}
    if set(experiment) - allowed:
        raise ValueError("Unrecognized experiment keys")
    mode = experiment.get("assignment", "official")
    if mode not in {"official", "last", "first", "mean", "zero_features"}:
        raise ValueError("Unsupported assignment")
    if mode == "official" and any(experiment.get(k, v) != v
                                 for k, v in (("camera_order", "normal"), ("dx", 0), ("dy", 0))):
        raise ValueError("Official assignment cannot silently apply an intervention")
    upstream = git_state(paths["ditr_root"])
    if not args.allow_upstream_drift and upstream["sha"] != paths["expected_upstream_commit"]:
        raise ValueError("Upstream SHA differs; inspect before --allow-upstream-drift")
    output = fresh_dir(args.out)
    write_json(output / "status.json", {"status": "running", "started_at": timestamp()})
    try:
        os.environ.setdefault("HF_HOME", paths["hf_cache"])
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        sys.path.insert(0, paths["ditr_root"])
        # Preserve all relative paths expected by author's saved configuration.
        os.chdir(paths["ditr_root"])
        import torch
        from pointcept.engines.defaults import default_config_parser, default_setup
        from pointcept.engines.test import SemSegTester
        from pointcept.datasets import build_dataset
        from .interventions import assign_variant

        cfg = default_config_parser(paths["config"], {
            "save_path": output.as_posix(), "weight": paths["weight"],
            "seed": experiment.get("seed", 20261101),
            "model.backbone.dinov2_frozen": False,
        })
        cfg.data.test.data_root = paths["data_root"]
        cfg.resume = False
        cfg = default_setup(cfg)
        # Save EFFECTIVE config after all overrides, including data root.
        cfg.dump(str(output / "config.py"))
        original = importlib.import_module(
            "pointcept.models.point_transformer_v3.point_transformer_v3m1_image"
        )
        if mode != "official":
            def intervention(data, feats, patch, missing=None):
                return assign_variant(
                    data, feats, patch, missing, mode=mode,
                    order=experiment.get("camera_order", "normal"),
                    dx=float(experiment.get("dx", 0)), dy=float(experiment.get("dy", 0)),
                )
            original.assign_image_feat = intervention
        audit = {}

        class LabTester(SemSegTester):
            def build_model(self):
                # Upstream loads checkpoint with strict=True. No conversion or skipped keys here.
                model = super().build_model()
                backbone = model.module.backbone if hasattr(model, "module") else model.backbone
                backbone.img_enc.requires_grad_(False)
                backbone.img_enc.eval()
                params = list(backbone.img_enc.parameters())
                audit.update({
                    "load_state_dict_strict": True,
                    "model_state_dict_entries": len(model.state_dict()),
                    "dino_parameter_elements": sum(x.numel() for x in params),
                    "dino_trainable_elements": sum(x.numel() for x in params if x.requires_grad),
                    "dino_training_flag": backbone.img_enc.training,
                    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                    "test_aug_count": len(cfg.data.test.test_cfg.aug_transform),
                })
                write_json(output / "model_audit.json", audit)
                return model

            def build_test_loader(self):
                dataset = build_dataset(cfg.data.test)
                dataset.data_list = select_infos(dataset.data_list, tokens=tokens)
                return torch.utils.data.DataLoader(
                    dataset, batch_size=1, shuffle=False, num_workers=0,
                    collate_fn=self.__class__.collate_fn, pin_memory=True,
                )

        write_json(output / "provenance.json", {
            "created_at": timestamp(), "experiment": experiment, "paths": paths,
            "lab_git": git_state(Path(__file__).resolve().parents[1]), "upstream_git": upstream,
            "weight_sha256": sha256(paths["weight"]),
            "config_sha256": sha256(paths["config"]), "tokens": tokens,
            "upstream_source_sha256": {
                name: sha256(Path(paths["ditr_root"]) / name)
                for name in (
                    "pointcept/datasets/nuscenes.py", "pointcept/datasets/utils.py",
                    "pointcept/models/point_transformer_v3/utils.py",
                    "pointcept/models/point_transformer_v3/point_transformer_v3m1_image.py",
                    "pointcept/engines/test.py",
                )
            },
            "torch": torch.__version__, "physical_gpu_requested": args.gpu,
            "kind": "evaluation_intervention_not_retrained_method",
        })
        tester = LabTester(cfg=cfg)
        if audit["dino_trainable_elements"] or audit["dino_training_flag"]:
            raise RuntimeError("DINO freeze audit failed")
        torch.cuda.reset_peak_memory_stats()
        import time
        torch.cuda.synchronize()
        started = time.perf_counter()
        tester.test()
        torch.cuda.synchronize()
        duration = time.perf_counter() - started
        prediction_count = sum(1 for _ in (output / "result").glob("*_pred.npy"))
        if prediction_count != len(tokens):
            raise RuntimeError("Incomplete output prediction cache")
        write_json(output / "runtime.json", {
            "wall_seconds_with_io_tta_logging": duration,
            "frames": prediction_count,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "timing_note": "End-to-end tester wall time, not isolated model latency",
        })
        write_json(output / "status.json", {"status": "completed", "ended_at": timestamp()})
    except Exception as error:
        write_json(output / "status.json", {
            "status": "failed", "ended_at": timestamp(), "error": str(error),
        })
        raise


if __name__ == "__main__":
    main()
