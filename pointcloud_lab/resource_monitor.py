"""Read-only host/container inventory and two-GPU telemetry, without Torch."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .common import timestamp, write_json


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, PermissionError):
        return None


def memory_info():
    text = read_text("/proc/meminfo") or ""
    info = {}
    for line in text.splitlines():
        key, value = line.split(":", 1)
        info[key] = int(value.split()[0]) * 1024
    limit_text = read_text("/sys/fs/cgroup/memory.max")
    current_text = read_text("/sys/fs/cgroup/memory.current")
    # cgroup v1 fallback for older container images.
    if limit_text is None:
        limit_text = read_text("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        current_text = read_text("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    limit = int(limit_text) if limit_text and limit_text != "max" else None
    current = int(current_text) if current_text else None
    available = info.get("MemAvailable")
    if limit is not None and current is not None:
        headroom = max(0, limit - current)
        available = min(available, headroom) if available is not None else headroom
    return {"host_total_bytes": info.get("MemTotal"), "host_available_bytes": info.get("MemAvailable"),
            "cgroup_limit_bytes": limit, "cgroup_current_bytes": current,
            "conservative_available_bytes": available, "host_cached_bytes": info.get("Cached")}


def cpu_count():
    count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    quota = read_text("/sys/fs/cgroup/cpu.max")
    if quota and quota.split()[0] != "max":
        amount, period = map(int, quota.split())
        count = min(count, max(1, amount // period))
    return count


def gpu_info():
    fields = "index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu"
    try:
        result = subprocess.run(["nvidia-smi", "--id=0,1", "--query-gpu=" + fields,
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        return {"fields": fields.split(","), "rows": result.stdout.strip().splitlines(),
                "error": result.stderr.strip() if result.returncode else None}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def inventory():
    memory, cores = memory_info(), cpu_count()
    loader_workers = 4 if cores >= 12 else 2 if cores >= 6 else 1
    idle_cores = max(0, cores - 2*loader_workers - 4)
    available = memory["conservative_available_bytes"]
    gib = 1024**3
    reserve = max(16*gib, available*.25) if available is not None else None
    cache = max(0, min(available*.5, available-reserve)) if available is not None else None
    return {"at": timestamp(), "cpu_budget": cores, "memory": memory, "gpus": gpu_info(),
            "shm": dict(zip(("total", "used", "free"), shutil.disk_usage("/dev/shm")))
                if Path("/dev/shm").exists() else None,
            "recommendation": {
                "gpu_jobs": 2, "loader_workers_per_gpu": loader_workers,
                "loader_prefetch_factor": 1, "cpu_threads_per_worker": 1,
                "cache_build_processes_when_idle": min(4, max(1, cores // 4)),
                "cache_build_processes_during_training": min(4, idle_cores),
                "combined_decoded_cache_target_bytes": int(cache) if cache is not None else None,
                "reserve_bytes": int(reserve) if reserve is not None else None,
                "note": "Starting budget only; inspect actual data wait, CPU quota, memory pressure and throughput"},
            "loadavg": read_text("/proc/loadavg"),
            "io_pressure": read_text("/proc/pressure/io"),
            "memory_pressure": read_text("/proc/pressure/memory"),
            "cpu_stat": (read_text("/proc/stat") or "").splitlines()[:1],
            "vm_stat": [line for line in (read_text("/proc/vmstat") or "").splitlines()
                        if line.split()[0] in {"pgmajfault", "pswpin", "pswpout", "pgpgin", "pgpgout"}]}


def process_identity(pid):
    text = read_text(f"/proc/{pid}/stat")
    if text:
        # comm is parenthesized and can contain spaces or parentheses.
        fields = text.rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    try:
        os.kill(pid, 0)
        return str(pid)
    except ProcessLookupError:
        return None
    except PermissionError:
        return str(pid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--interval", type=float, default=10)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("Positive sampling interval required")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.pid is None:
        write_json(output, inventory())
        print(output)
        return
    identity = process_identity(args.pid)
    with output.open("x", encoding="utf-8", buffering=1) as stream:
        while identity is not None and process_identity(args.pid) == identity:
            stream.write(json.dumps(inventory(), ensure_ascii=False) + "\n")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
