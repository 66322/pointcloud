"""One smoke frame, then two full-val view-assignment experiments on GPU 0/1."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import fresh_dir, git_state, load_infos, read_json, timestamp, write_json
from .results import read_test_metrics

LAB_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (("mean_visible_cameras", "0"), ("first_visible_camera", "1"))


def evaluation_command(paths_file, variant, tokens_file, gpu, out, count):
    return [
        sys.executable, "-m", "pointcloud_lab.evaluate",
        "--paths", str(paths_file), "--experiment",
        str(LAB_ROOT / "configs" / "experiments" / f"{variant}.json"),
        "--tokens", str(tokens_file), "--expected-frames", str(count),
        "--gpu", gpu, "--out", str(out),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="configs/paths.local.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Use only when this exact runner has already passed one frame")
    parser.add_argument("--baseline-miou", type=float, default=82.48)
    args = parser.parse_args()
    paths_file = Path(args.paths).resolve()
    paths = read_json(paths_file)
    infos = load_infos(paths["info"])
    tokens = [str(x["lidar_token"]) for x in infos]
    if not tokens:
        raise ValueError("Empty validation index")
    output = fresh_dir(args.out)
    write_json(output / "status.json", {"status": "running", "stage": "starting"})
    write_json(output / "tokens.json", tokens)
    write_json(output / "smoke_tokens.json", tokens[:1])
    write_json(output / "provenance.json", {
        "created_at": timestamp(), "frames": len(tokens), "paths": paths,
        "lab_git": git_state(LAB_ROOT),
        "baseline_miou_pct": args.baseline_miou,
        "baseline_source": "existing user-validated full-val run; not rerun",
        "variants": [{"name": name, "physical_gpu": gpu} for name, gpu in VARIANTS],
    })
    processes, handles, rows = [], [], []
    try:
        if not args.skip_smoke:
            print("One-frame check: mean_visible_cameras on GPU 0", flush=True)
            command = evaluation_command(
                paths_file, "mean_visible_cameras", output / "smoke_tokens.json",
                "0", output / "smoke_mean_one_frame", 1,
            )
            with (output / "smoke_console.log").open("w", encoding="utf-8") as stream:
                result = subprocess.run(command, cwd=LAB_ROOT, stdout=stream,
                                        stderr=subprocess.STDOUT)
            if result.returncode:
                log = output / "smoke_console.log"
                tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-30:])
                raise RuntimeError(f"One-frame run failed; see {log}\n\n{tail}")
        print(f"Starting full-val experiments directly: {len(tokens)} frames per method", flush=True)
        write_json(output / "status.json", {"status": "running", "stage": "full_validation"})
        for variant, gpu in VARIANTS:
            command = evaluation_command(
                paths_file, variant, output / "tokens.json", gpu, output / variant, len(tokens),
            )
            stream = (output / f"{variant}_console.log").open("w", encoding="utf-8")
            handles.append(stream)
            child_env = os.environ.copy()
            child_env["CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.Popen(command, cwd=LAB_ROOT, stdout=stream,
                                       stderr=subprocess.STDOUT, env=child_env)
            processes.append((variant, gpu, process))
            print(f"{variant}: GPU {gpu}, log={output / (variant + '_console.log')}", flush=True)
        pending = {name for name, _, _ in processes}
        while pending:
            for variant, gpu, process in processes:
                if variant not in pending or process.poll() is None:
                    continue
                pending.remove(variant)
                row = {"method": variant, "gpu": gpu, "status": "failed",
                       "miou_pct": None, "delta_miou_pp": None,
                       "macc_pct": None, "oa_pct": None}
                if process.returncode == 0:
                    metrics = read_test_metrics(output / variant / "test.log")
                    row.update(
                        status="completed", miou_pct=metrics["miou_pct"],
                        delta_miou_pp=metrics["miou_pct"] - args.baseline_miou,
                        macc_pct=metrics["macc_pct"], oa_pct=metrics["oa_pct"],
                    )
                rows.append(row)
                print(f"{variant}: {row['status']}, mIoU={row['miou_pct']}", flush=True)
            if pending:
                time.sleep(1)
        with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(output / "summary.json", {
            "kind": "full_validation_inference_intervention",
            "frames_per_method": len(tokens), "baseline_miou_pct": args.baseline_miou,
            "results": rows,
            "note": "Parameter-free inference interventions; no new training. "
                    "Report negative results. A one-shot drop does not rule out retraining.",
        })
        lines = [
            "# Full-validation view-assignment experiments", "",
            f"Existing baseline: {args.baseline_miou:.2f}% mIoU; frames per method: {len(tokens)}.",
            "", "| Method | Status | mIoU % | Delta pp |",
            "|---|---|---:|---:|",
        ]
        lines += [f"| {r['method']} | {r['status']} | {r['miou_pct']} | "
                  f"{r['delta_miou_pp']} |" for r in rows]
        (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        failed = any(r["status"] != "completed" for r in rows)
        write_json(output / "status.json", {
            "status": "failed" if failed else "completed", "ended_at": timestamp(),
        })
        if failed:
            raise SystemExit(1)
    except BaseException as error:
        # Only terminate child processes created by this batch, never unrelated GPU work.
        for _, _, process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if not isinstance(error, SystemExit):
            write_json(output / "status.json", {
                "status": "failed", "ended_at": timestamp(), "error": str(error),
            })
        raise
    finally:
        for stream in handles:
            stream.close()


if __name__ == "__main__":
    main()
