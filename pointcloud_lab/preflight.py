"""Read-only server inventory. Does not load a model or allocate a GPU."""
import argparse
import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path

from .common import fresh_dir, git_state, read_json, sha256, timestamp, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--query-allowed-gpus", action="store_true")
    args = parser.parse_args()
    paths = read_json(args.paths)
    output = fresh_dir(args.out)
    facts = {}
    for key in ("ditr_root", "data_root", "info", "baseline_predictions", "baseline_log",
                "config", "weight", "hf_cache"):
        path = Path(paths[key])
        facts[key] = {"exists": path.exists(), "path": str(path),
                      "size_bytes": path.stat().st_size if path.is_file() else None}
    for key in ("info", "config", "baseline_log"):
        if Path(paths[key]).is_file():
            facts[key]["sha256"] = sha256(paths[key])
    versions = {}
    for name in ("numpy", "Pillow", "torch", "timm", "flash-attn", "spconv-cu124"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    source = git_state(paths["ditr_root"])
    count = sum(1 for _ in Path(paths["baseline_predictions"]).glob("*_pred.npy"))
    report = {
        "kind": "read_only_preflight", "created_at": timestamp(),
        "python": platform.python_version(), "versions": versions,
        "upstream_git": source, "lab_git": git_state(Path(__file__).resolve().parents[1]),
        "expected_upstream_commit_matches": source["sha"] == paths["expected_upstream_commit"],
        "files": facts, "baseline_prediction_count": count,
        "strict_checkpoint_loading": "NOT_CHECKED; requires the GPU evaluation smoke run",
        "actual_dino_freeze": "NOT_CHECKED; checked by eval runner after model construction",
        "CUDA_VISIBLE_DEVICES_before_run": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if Path(paths["baseline_log"]).is_file():
        lines = Path(paths["baseline_log"]).read_text(encoding="utf-8", errors="replace").splitlines()
        report["baseline_result_lines"] = [x for x in lines if "Val result:" in x][-3:]
    if args.query_allowed_gpus:
        result = subprocess.run(
            ["nvidia-smi", "-i", "0,1", "--query-gpu=index,name,memory.total,memory.used",
             "--format=csv"], capture_output=True, text=True,
        )
        report["allowed_gpu_query"] = {"returncode": result.returncode, "stdout": result.stdout,
                                       "stderr": result.stderr}
    write_json(output / "preflight.json", report)
    missing = [key for key, value in facts.items() if not value["exists"]]
    write_json(output / "status.json", {
        "status": "completed" if not missing else "needs_attention", "missing": missing,
    })
    print(f"Saved {output / 'preflight.json'}; missing={missing}")
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
