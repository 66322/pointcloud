"""Persistent two-GPU continuation queue, with independent failure isolation.

No Torch import in the controller. The older frozen queue finishes normally;
completed same-budget work is reused. Every new GPU job runs from this queue's
own code/config snapshot and has an independent output directory.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .common import fresh_dir, git_state, load_infos, read_json, timestamp, write_json
from .eval_conditions import condition_description
from .run_training import training_command, validation_command
from .training_protocol import validate_protocol

ROOT = Path(__file__).resolve().parents[1]
DONE = {"completed", "reused"}
TERMINAL = DONE | {"failed", "blocked"}


def atomic_json(path, payload):
    path = Path(path)
    write_json(str(path) + ".tmp", payload)
    os.replace(str(path) + ".tmp", path)


def optional_json(path):
    try:
        return read_json(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def validate_plan(plan):
    required = {"id", "screen_protocol", "medium_epochs", "medium_methods", "robust_methods",
                "robust_camera_counts", "projection_methods", "projection_shift_x"}
    if set(plan) != required:
        raise ValueError(f"Queue plan keys must be {sorted(required)}")
    protocol = validate_protocol(plan["screen_protocol"])
    for key in ("medium_methods", "robust_methods", "projection_methods"):
        methods = plan[key]
        if len(methods) != len(set(methods)) or set(methods) - set(protocol["methods"]):
            raise ValueError(f"{key} requires unique screened methods")
        if methods and "control" not in methods:
            raise ValueError(f"{key} must include its matched control")
    if (not isinstance(plan["medium_epochs"], int)
            or plan["medium_epochs"] <= protocol["epochs"]):
        raise ValueError("Medium budget must exceed screen epochs")
    counts = plan["robust_camera_counts"]
    if len(counts) != len(set(counts)) or set(counts) - {0, 1, 3}:
        raise ValueError("Unique missing-camera counts from 0,1,3 required")
    return plan


def make_jobs(plan, output):
    jobs = []
    def add(name, kind, phase, method, dependencies=(), **condition):
        jobs.append(dict(id=name, kind=kind, phase=phase, method=method,
                         dependencies=list(dependencies), status="pending",
                         output=str(output / "jobs" / name), **condition))
    for method in plan["screen_protocol"]["methods"]:
        train = f"screen_{method}_train"
        add(train, "train", "screen", method)
        add(f"screen_{method}_clean", "eval", "screen", method, [train],
            train_job=train, condition="clean", camera_count=6, shift_x=0.0)
    for method in plan["robust_methods"]:
        train = f"screen_{method}_train"
        for count in plan["robust_camera_counts"]:
            add(f"robust_{method}_cam{count}", "eval", "robust", method, [train],
                train_job=train, condition=f"camera_{count}", camera_count=count, shift_x=0.0)
    for method in plan["projection_methods"]:
        train = f"screen_{method}_train"
        add(f"projection_{method}_shift", "eval", "projection", method, [train],
            train_job=train, condition="pixel_shift_x", camera_count=6,
            shift_x=plan["projection_shift_x"])
    for method in plan["medium_methods"]:
        train = f"medium_{method}_train"
        add(train, "train", "medium", method, [f"screen_{method}_clean"])
        add(f"medium_{method}_clean", "eval", "medium", method, [train],
            train_job=train, condition="clean", camera_count=6, shift_x=0.0)
    return jobs


def dependency_state(job, by_id):
    states = [by_id[key]["status"] for key in job["dependencies"]]
    if any(state in {"failed", "blocked"} for state in states):
        return "blocked"
    return "ready" if all(state in DONE for state in states) else "waiting"


def same_budget(left, right):
    return ({k: v for k, v in left.items() if k not in {"id", "methods"}}
            == {k: v for k, v in right.items() if k not in {"id", "methods"}})


def reuse_prior(jobs, prior, protocol, paths, tokens):
    """Only reuse completed artifacts with matching budget, machine paths/tokens."""
    reused = [job["id"] for job in jobs if job["status"] == "reused"]
    prior_protocol = optional_json(prior / "protocol.json")
    prior_paths = optional_json(prior / "paths.json")
    prior_tokens = optional_json(prior / "tokens.json")
    if (prior_protocol is None or not same_budget(protocol, prior_protocol)
            or prior_paths != paths or prior_tokens != tokens):
        return {"reused": reused, "reason": "prior common budget/paths/tokens do not match; retained saved reuse"}
    for job in jobs:
        if job["phase"] != "screen" or job["status"] != "pending":
            continue
        candidate = prior / job["method"] / ("train" if job["kind"] == "train" else "val")
        status = optional_json(candidate / "status.json")
        summary = optional_json(candidate / "summary.json")
        if not status or status.get("status") != "completed" or summary is None:
            continue
        training = prior / job["method"] / "train"
        saved = optional_json(training / "experiment.json")
        if (not saved or saved.get("method") != job["method"]
                or not same_budget(saved["protocol"], protocol)):
            continue
        if not (training / "model/model_last.pth").is_file():
            continue
        if job["kind"] == "eval":
            provenance = optional_json(candidate / "provenance.json")
            if (not provenance or provenance.get("tokens") != tokens
                    or provenance.get("experiment", {}).get("assignment", "official") != "official"):
                continue
            condition = provenance.get("evaluation_condition")
            if condition and (condition["camera_count"] != 6 or condition["pixel_shift_x"] != 0):
                continue
            if not {"miou_pct", "macc_pct", "oa_pct"}.issubset(summary):
                continue
        job.update(status="reused", output=str(candidate), summary=summary,
                   reused_from=str(candidate), ended_at=timestamp())
        reused.append(job["id"])
    return {"reused": reused, "reason": "completed same-budget prior artifacts"}


def gpu_idle(gpu):
    result = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=True)
    return not result.stdout.strip()


def process_alive(pid, output=None):
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True
    proc = Path(f"/proc/{pid}/cmdline")
    if output is not None and proc.is_file():
        try:
            return str(output).encode() in proc.read_bytes()
        except FileNotFoundError:
            return False
    return True


def controller_lock(output):
    """An OS-held lock prevents two resumed controllers from sharing a queue."""
    stream = (output / "controller.lock").open("a+b")
    if stream.tell() == 0:
        stream.write(b"0")
        stream.flush()
    stream.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise RuntimeError("A controller is already running for this queue")
    return stream


class AdoptedProcess:
    """Observe, without killing, a job from an interrupted controller."""
    def __init__(self, job):
        self.pid, self.output = job["pid"], Path(job["output"])
        self.returncode = None

    def poll(self):
        if process_alive(self.pid, self.output):
            return None
        status = optional_json(self.output / "status.json")
        self.returncode = 0 if status and status.get("status") == "completed" else 1
        return self.returncode


def job_command(job, output, by_id, frame_count):
    if job["kind"] == "train":
        return training_command(
            output / "paths.json", output / f"{job['phase']}_protocol.json",
            job["method"], job["gpu"], job["output"])
    command = validation_command(
        output / "paths.json", by_id[job["train_job"]]["output"], output / "tokens.json",
        job["gpu"], job["output"], frame_count,
        output / "code/configs/experiments/default_assignment.json")
    return command + ["--camera-count", str(job["camera_count"]),
                      "--pixel-shift-x", str(job["shift_x"])]


def write_reports(output, jobs):
    rows = []
    controls = {(job["phase"], job.get("condition")): job["summary"]["miou_pct"]
                for job in jobs if job["kind"] == "eval" and job["method"] == "control"
                and job["status"] in DONE}
    for job in jobs:
        if job["kind"] != "eval":
            continue
        metrics = job.get("summary", {})
        miou = metrics.get("miou_pct")
        control = controls.get((job["phase"], job["condition"]))
        rows.append(dict(
            id=job["id"], phase=job["phase"], method=job["method"],
            condition=job["condition"], status=job["status"], miou_pct=miou,
            delta_vs_matched_control_pp=(
                round(miou - control, 4) if miou is not None and control is not None else None),
            macc_pct=metrics.get("macc_pct"), oa_pct=metrics.get("oa_pct"),
            source=job["output"]))
    atomic_json(output / "summary.json", {"kind": "multi_route_queue", "results": rows,
                "note": "Screen and medium budgets use separate controls; see protocol snapshots"})
    if rows:
        with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# Multi-route queue", "",
             "| Phase | Method | Condition | Status | mIoU % | Delta vs matched control pp |",
             "|---|---|---|---|---:|---:|"]
    for row in rows:
        miou, delta = row["miou_pct"], row["delta_vs_matched_control_pp"]
        lines.append(f"| {row['phase']} | {row['method']} | {row['condition']} | {row['status']} | "
                     f"{miou if miou is not None else ''} | {delta if delta is not None else ''} |")
    failed = [job for job in jobs if job["status"] in {"failed", "blocked"}]
    if failed:
        lines += ["", "## Failed or dependent tasks", ""]
        lines += [f"- {job['id']}: {job['status']}; {job.get('error', '')}" for job in failed]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute_queue(output, jobs, frame_count, *, poll_seconds=15):
    by_id, active = {job["id"]: job for job in jobs}, {}
    for job in jobs:
        if job["status"] == "running":
            gpu = job["gpu"]
            if gpu in active:
                raise ValueError("Saved queue has multiple active jobs on one GPU")
            active[gpu] = (job, AdoptedProcess(job), None)
    def save():
        atomic_json(output / "queue.json", {"jobs": jobs, "updated_at": timestamp()})
        write_reports(output, jobs)
    try:
        save()
        while any(job["status"] not in TERMINAL for job in jobs):
            for gpu, (job, process, stream) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                if stream is not None:
                    stream.close()
                result = optional_json(Path(job["output"]) / "summary.json")
                status = optional_json(Path(job["output"]) / "status.json")
                valid_result = (isinstance(result, dict) and (
                    {"miou_pct", "macc_pct", "oa_pct"}.issubset(result)
                    if job["kind"] == "eval" else "optimizer_updates" in result))
                if (code == 0 and valid_result and status
                        and status.get("status") == "completed"):
                    job.update(status="completed", summary=result)
                    print(f"DONE {job['id']}: {job['output']}", flush=True)
                else:
                    log = Path(job.get("log", ""))
                    tail = ("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:])
                            if log.is_file() else "No final completed status/summary")
                    job.update(status="failed", error=f"exit={code}; {tail}")
                    print(f"FAILED {job['id']}: independent tasks will continue; {log}", flush=True)
                job["ended_at"], job["exit_code"] = timestamp(), code
                del active[gpu]
                save()
            for job in jobs:
                if job["status"] == "pending" and dependency_state(job, by_id) == "blocked":
                    job.update(status="blocked", error="A required training/validation job failed")
                    save()
            for gpu in ("0", "1"):
                if gpu in active:
                    continue
                ready = next((job for job in jobs if job["status"] == "pending"
                              and dependency_state(job, by_id) == "ready"), None)
                if ready is None:
                    continue
                try:
                    if not gpu_idle(gpu):
                        continue
                except (OSError, subprocess.SubprocessError) as error:
                    message = f"Cannot inspect GPU {gpu}: {error}"
                    if ready.get("gpu_probe_error") != message:
                        ready["gpu_probe_error"] = message
                        print(f"WAIT {message}; retrying without starting a GPU job", flush=True)
                        save()
                    continue
                ready.pop("gpu_probe_error", None)
                ready["gpu"] = gpu
                command = job_command(ready, output, by_id, frame_count)
                ready["log"] = str(output / "logs" / f"{ready['id']}.log")
                stream = Path(ready["log"]).open("w", encoding="utf-8")
                try:
                    process = subprocess.Popen(command, cwd=output / "code",
                                               stdout=stream, stderr=subprocess.STDOUT,
                                               stdin=subprocess.DEVNULL)
                except OSError as error:
                    stream.close()
                    ready.update(status="failed", error=str(error), ended_at=timestamp())
                    save()
                    continue
                ready.update(status="running", pid=process.pid, started_at=timestamp(), command=command)
                active[gpu] = (ready, process, stream)
                save()
                print(f"START {ready['id']}: GPU {gpu}, PID {process.pid}, log={ready['log']}", flush=True)
            if any(job["status"] not in TERMINAL for job in jobs):
                time.sleep(poll_seconds)
        save()
    except BaseException:
        save()  # Leave running PIDs available for --resume adoption.
        raise
    finally:
        for _, _, stream in active.values():
            if stream is not None:
                stream.close()
    return "completed_with_failures" if any(job["status"] in {"failed", "blocked"} for job in jobs) else "completed"


def wait_for_prior(prior, output, poll_seconds=15):
    if not prior.is_dir():
        raise ValueError(f"Prior batch directory missing: {prior}")
    last = None
    while True:
        state = optional_json(prior / "status.json")
        status = state.get("status") if state else "status_not_ready"
        if status in {"completed", "failed", "completed_with_failures"}:
            return
        atomic_json(output / "status.json", {
            "status": "waiting_for_prior", "prior_batch": str(prior),
            "prior_status": status, "updated_at": timestamp()})
        if status != last:
            print(f"WAIT {prior}: status={status}; no GPU jobs launched", flush=True)
            last = status
        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="configs/paths.local.json")
    parser.add_argument("--plan", default="configs/training/multi_route_queue.json")
    parser.add_argument("--after", help="Wait for this existing batch; reuse completed same-budget work")
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = Path(args.out).resolve()
    if args.resume:
        plan = validate_plan(read_json(output / "plan.json"))
        paths = read_json(output / "paths.json")
        prior_value = read_json(output / "provenance.json")["prior_batch"]
        prior = Path(prior_value) if prior_value else None
        jobs = read_json(output / "queue.json")["jobs"]
    else:
        plan = validate_plan(read_json(args.plan))
        paths = read_json(args.paths)
        prior = Path(args.after).resolve() if args.after else None
        output = fresh_dir(output)
        (output / "jobs").mkdir()
        (output / "logs").mkdir()
        shutil.copytree(ROOT / "pointcloud_lab", output / "code/pointcloud_lab",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (output / "code/configs/experiments").mkdir(parents=True)
        shutil.copy2(ROOT / "configs/experiments/default_assignment.json",
                     output / "code/configs/experiments/default_assignment.json")
        write_json(output / "code/snapshot.json", {
            "source_git": git_state(ROOT), "created_at": timestamp()})
        write_json(output / "plan.json", plan)
        write_json(output / "paths.json", paths)
        write_json(output / "screen_protocol.json", plan["screen_protocol"])
        write_json(output / "medium_protocol.json", dict(
            plan["screen_protocol"], id=plan["id"] + "_medium", epochs=plan["medium_epochs"]))
        write_json(output / "provenance.json", {
            "lab_git": git_state(ROOT), "created_at": timestamp(),
            "prior_batch": str(prior) if prior else None,
            "medium_initialization": "fresh original pretrained checkpoint; not resume of 2 epochs",
            "promotion_policy": "predeclared medium methods, not filtered by screen score",
            "robust_conditions": [condition_description(n) for n in plan["robust_camera_counts"]],
            "projection_condition": condition_description(6, plan["projection_shift_x"])})
        jobs = make_jobs(plan, output)
        atomic_json(output / "queue.json", {"jobs": jobs, "updated_at": timestamp()})
    lock = controller_lock(output)
    atomic_json(output / "controller.json", {"pid": os.getpid(), "started_at": timestamp()})
    try:
        if prior:
            wait_for_prior(prior, output)
        tokens_file = output / "tokens.json"
        if tokens_file.is_file():
            tokens = read_json(tokens_file)
        else:
            tokens = [str(item["lidar_token"]) for item in load_infos(paths["info"])]
            write_json(tokens_file, tokens)
        if prior:
            reuse = reuse_prior(jobs, prior, plan["screen_protocol"], paths, tokens)
            atomic_json(output / "queue.json", {"jobs": jobs, "updated_at": timestamp()})
            atomic_json(output / "reuse.json", reuse)
            print(f"REUSE {len(reuse['reused'])} completed jobs; {reuse['reason']}", flush=True)
        atomic_json(output / "status.json", {"status": "running", "started_at": timestamp()})
        final = execute_queue(output, jobs, len(tokens))
        atomic_json(output / "status.json", {"status": final, "ended_at": timestamp()})
    except BaseException as error:
        atomic_json(output / "status.json", {
            "status": "failed", "ended_at": timestamp(), "error": str(error),
            "recovery": "resume this campaign; recorded running children are observed, not relaunched"})
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
