"""Export compact, per-job proposal evidence with the configuration and execution commit."""
import argparse
from collections import Counter
import os
from pathlib import Path
import shutil
import tempfile

from .common import read_json, timestamp, write_json


def snapshot(source, destination):
    source, destination = Path(source), Path(destination)
    if read_json(source / "plan.json").get("schema") != "proposal_v1":
        raise ValueError("Expected a proposal campaign")
    queue = read_json(source / "queue.json")
    fields = ("id", "seed", "method", "condition", "status", "gpu", "pid", "started_at", "ended_at")
    jobs = []
    for job in queue["jobs"]:
        row = {k: job[k] for k in fields if k in job}
        if job["status"] == "completed":
            summary = job["summary"]
            row["summary"] = {k: summary[k] for k in ("miou_pct", "macc_pct", "oa_pct",
                                                       "optimizer_updates", "amp_skipped_steps") if k in summary}
        jobs.append(row)
    payload = {"kind": "server_queue_snapshot", "run": source.name, "captured_at": timestamp(),
               "queue_updated_at": queue["updated_at"], "execution": read_json(source / "provenance.json"),
               "counts": dict(Counter(j["status"] for j in jobs)), "jobs": jobs}
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a complete snapshot and keep every earlier observation.
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        ready = Path(tmp) / "snapshot.json"
        write_json(ready, payload)
        os.link(ready, destination)


def export_proposal(source, target):
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    queue = read_json(source / "queue.json")["jobs"]
    kept_train = ("epochs", "attempts", "optimizer_updates", "amp_skipped_steps",
                  "wall_seconds_this_invocation", "peak_allocated_bytes", "first_step_passed")
    kept_env = ("upstream_git", "trainable_parameter_elements", "dino_trainable_elements",
                "train_frames_per_epoch", "steps_per_epoch", "total_attempted_steps", "amp", "torch",
                "gpu_model", "physical_gpu", "physical_gpu_requested", "test_aug_count",
                "initial_weight_name", "initial_weight_bytes", "weight_bytes", "upstream_config_name")
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        folder = Path(tmp) / "record"
        folder.mkdir()
        for name in ("plan.json", "selection.json", "provenance.json", "status.json", "summary.json",
                     "tokens.json", "stf_tokens.json", "comparison.csv", "seed_deltas.csv", "contrasts.csv"):
            if (source / name).is_file():
                shutil.copyfile(source / name, folder / name)
        for path in source.glob("seed_*_protocol.json"):
            shutil.copyfile(path, folder / path.name)
        jobs_dir = source / "jobs"
        for job in queue:
            job_dir = Path(job["output"]).resolve()
            if job_dir.parent != jobs_dir.resolve() or job_dir.name != job["id"]:
                raise ValueError(f"Unexpected proposal job directory: {job['id']}")
            record = {key: job.get(key) for key in ("id", "seed", "method", "kind", "condition", "status",
                                                      "gpu", "started_at", "ended_at", "exit_code")}
            state = read_json(job_dir / "status.json") if (job_dir / "status.json").exists() else {}
            if job["status"] == "completed":
                if state.get("status") != "completed":
                    raise ValueError(f"Inconsistent job status: {job['id']}")
                summary = read_json(job_dir / "summary.json")
                record["metrics"] = (summary if job["kind"] == "eval" else
                                     {k: summary.get(k) for k in kept_train})
                provenance = read_json(job_dir / "provenance.json")
                record["environment"] = {k: provenance[k] for k in kept_env if k in provenance}
                if (job_dir / "runtime.json").exists():
                    runtime = read_json(job_dir / "runtime.json")
                    record["runtime"] = {k: runtime[k] for k in (
                        "wall_seconds_with_io_tta_logging", "frames", "peak_allocated_bytes")}
                if (job_dir / "model_audit.json").exists():
                    record["model_audit"] = read_json(job_dir / "model_audit.json")
            else:
                record["error"] = (state.get("error") or job.get("error", "No completion receipt"))[-4000:]
            (folder / "jobs").mkdir(exist_ok=True)
            write_json(folder / "jobs" / (job["id"] + ".json"), record)
        stamp = read_json(source / "provenance.json")
        status = read_json(source / "status.json")["status"]
        (folder / "README.md").write_text(
            f"# {source.name}\n\n执行提交 `{stamp['source_git']['sha']}`。队列状态 `{status}`。\n\n"
            "本记录取自服务器保存的队列与各任务终态。comparison.csv 给出各条件结果，"
            "seed_deltas.csv 保留逐种子配对差，contrasts.csv 给出跨种子汇总。"
            "jobs/ 保留逐类、分层、运行成本及失败原因。\n", encoding="utf-8")
        os.rename(folder, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture one immutable saved-queue observation")
    parser.add_argument("--run", required=True)
    parser.add_argument("--snapshot", required=True, help="New small JSON file in the experiment record")
    args = parser.parse_args()
    snapshot(args.run, args.snapshot)
