"""Two-GPU queue: paired fine-tuning followed by each method's final full validation."""
import argparse
import csv
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .common import fresh_dir, git_state, load_infos, read_json, timestamp, write_json
from .training_protocol import validate_protocol

ROOT = Path(__file__).resolve().parents[1]


def training_command(paths, protocol, method, gpu, output):
    return [sys.executable, "-u", "-m", "pointcloud_lab.train",
            "--paths", str(paths), "--protocol", str(protocol), "--method", method,
            "--gpu", str(gpu), "--out", str(output)]


def validation_command(paths, training, tokens, gpu, output, count, experiment=None):
    return [sys.executable, "-u", "-m", "pointcloud_lab.evaluate",
            "--paths", str(paths), "--experiment",
            str(experiment or ROOT / "configs/experiments/default_assignment.json"),
            "--trained-run", str(training), "--tokens", str(tokens),
            "--gpu", str(gpu), "--out", str(output), "--expected-frames", str(count)]


def write_report(output, rows):
    ordered = sorted(rows, key=lambda row: row["method"] != "control")
    control = next((row["miou_pct"] for row in rows if row["method"] == "control"), None)
    for row in ordered:
        row["delta_vs_trained_control_pp"] = round(row["miou_pct"] - control, 4) if control is not None else None
        row["delta_vs_existing_82_48_pp"] = round(row["miou_pct"] - 82.48, 4)
    if ordered:
        with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(ordered[0]))
            writer.writeheader()
            writer.writerows(ordered)
    write_json(output / "summary.json", {
        "kind": "paired_finetuning_then_full_validation", "results": ordered,
        "primary_comparison": "same-budget trained control, not pretrained 82.48 alone",
    })
    lines = ["# Same-budget training experiments", "",
             "| Method | mIoU % | Delta vs trained control pp | Optimizer updates |",
             "|---|---:|---:|---:|"]
    for row in ordered:
        delta = row["delta_vs_trained_control_pp"]
        lines.append(f"| {row['method']} | {row['miou_pct']:.2f} | "
                     f"{delta if delta is not None else 'pending'} | {row['optimizer_updates']} |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="configs/paths.local.json")
    parser.add_argument("--protocol", default="configs/training/stage1.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    paths_file, protocol_file = Path(args.paths).resolve(), Path(args.protocol).resolve()
    protocol = validate_protocol(read_json(protocol_file))
    paths = read_json(paths_file)
    output = fresh_dir(args.out)
    write_json(output / "status.json", {"status": "running", "started_at": timestamp()})
    write_json(output / "protocol.json", protocol)
    write_json(output / "provenance.json", {"lab_git": git_state(ROOT), "paths": paths})
    active, streams, results = {}, [], []
    waiting = list(protocol["methods"])
    code_root = output / "code"
    paths_snapshot = output / "paths.json"

    def start(method, gpu, phase):
        method_root = output / method
        if phase == "train":
            method_root.mkdir()
            write_json(method_root / "status.json", {"status": "running", "stage": "train"})
            command = training_command(paths_snapshot, output / "protocol.json", method, gpu, method_root / "train")
        else:
            write_json(method_root / "status.json", {"status": "running", "stage": "full_validation"})
            command = validation_command(paths_snapshot, method_root / "train", output / "tokens.json",
                                         gpu, method_root / "val", len(tokens),
                                         code_root / "configs/experiments/default_assignment.json")
        log = output / f"{method}_{phase}_console.log"
        stream = log.open("w", encoding="utf-8")
        streams.append(stream)
        process = subprocess.Popen(command, cwd=code_root, stdout=stream, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL)
        active[gpu] = {"method": method, "phase": phase, "process": process, "log": log}
        print(f"{method}: {phase}, GPU {gpu}, PID {process.pid}, log={log}", flush=True)

    try:
        # Small source/config snapshots keep queued jobs stable across later Git pulls.
        # No repository clone, file hashes, dataset copies or checkpoint copies.
        shutil.copytree(ROOT / "pointcloud_lab", code_root / "pointcloud_lab",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (code_root / "configs/experiments").mkdir(parents=True)
        shutil.copy2(ROOT / "configs/experiments/default_assignment.json",
                     code_root / "configs/experiments/default_assignment.json")
        write_json(code_root / "snapshot.json", {"source_git": git_state(ROOT), "created_at": timestamp()})
        write_json(paths_snapshot, paths)
        infos = load_infos(paths["info"])
        tokens = [str(item["lidar_token"]) for item in infos]
        del infos
        write_json(output / "tokens.json", tokens)
        for gpu in ("0", "1"):
            if waiting:
                start(waiting.pop(0), gpu, "train")
        while active:
            for gpu, job in list(active.items()):
                process, method, phase = job["process"], job["method"], job["phase"]
                if process.poll() is None:
                    continue
                if process.returncode:
                    tail = "\n".join(job["log"].read_text(encoding="utf-8", errors="replace").splitlines()[-30:])
                    write_json(output / method / "status.json", {
                        "status": "failed", "stage": phase, "error": tail,
                    })
                    raise RuntimeError(f"{method} {phase} failed; {job['log']}\n{tail}")
                if phase == "train":
                    start(method, gpu, "val")
                else:
                    metrics = read_json(output / method / "val/summary.json")
                    training = read_json(output / method / "train/summary.json")
                    results.append({
                        "method": method, "gpu": gpu, "miou_pct": metrics["miou_pct"],
                        "macc_pct": metrics["macc_pct"], "oa_pct": metrics["oa_pct"],
                        "optimizer_updates": training["optimizer_updates"],
                        "amp_skipped_steps": training["amp_skipped_steps"],
                    })
                    write_report(output, results)
                    write_json(output / method / "status.json", {"status": "completed", "ended_at": timestamp()})
                    del active[gpu]
                    if waiting:
                        start(waiting.pop(0), gpu, "train")
            if active:
                time.sleep(1)
        write_json(output / "status.json", {"status": "completed", "ended_at": timestamp()})
    except BaseException as error:
        for job in active.values():
            process = job["process"]
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        write_json(output / "status.json", {"status": "failed", "ended_at": timestamp(), "error": str(error)})
        raise
    finally:
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    main()
