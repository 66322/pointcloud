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


def evaluation_command(paths_file, variant, tokens_file, gpu, out, count,
                       io_workers=2, prefetch_frames=1, image_cache=True):
    command = [
        sys.executable, "-m", "pointcloud_lab.evaluate",
        "--paths", str(paths_file), "--experiment",
        str(LAB_ROOT / "configs" / "experiments" / f"{variant}.json"),
        "--tokens", str(tokens_file), "--expected-frames", str(count),
        "--gpu", gpu, "--out", str(out),
        "--io-workers", str(io_workers), "--prefetch-frames", str(prefetch_frames),
    ]
    if not image_cache:
        command.append("--no-image-cache")
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="configs/paths.local.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Use only when this exact runner has already passed one frame")
    parser.add_argument("--baseline-miou", type=float, default=82.48)
    parser.add_argument("--io-workers", type=int, default=2)
    parser.add_argument("--prefetch-frames", type=int, default=1)
    parser.add_argument("--image-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reference-smoke",
                        help="Optional old smoke result directory: compare the SAME single-frame prediction")
    args = parser.parse_args()
    if args.io_workers < 0 or args.prefetch_frames < 0:
        parser.error("I/O workers and prefetch frames must be nonnegative")
    if args.skip_smoke and args.reference_smoke:
        parser.error("--reference-smoke requires the single smoke frame")
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
        "execution": {"raw_io_threads": args.io_workers,
                      "prefetch_frames": args.prefetch_frames, "image_cache": args.image_cache},
    })
    processes, handles, rows = [], [], []
    try:
        if not args.skip_smoke:
            print("One-frame check: mean_visible_cameras on GPU 0", flush=True)
            command = evaluation_command(
                paths_file, "mean_visible_cameras", output / "smoke_tokens.json",
                "0", output / "smoke_mean_one_frame", 1,
                args.io_workers, args.prefetch_frames, args.image_cache,
            )
            with (output / "smoke_console.log").open("w", encoding="utf-8") as stream:
                result = subprocess.run(command, cwd=LAB_ROOT, stdout=stream,
                                        stderr=subprocess.STDOUT)
            if result.returncode:
                log = output / "smoke_console.log"
                tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-30:])
                raise RuntimeError(f"One-frame run failed; see {log}\n\n{tail}")
            if args.reference_smoke:
                import numpy as np
                token = tokens[0]
                old = Path(args.reference_smoke) / f"{token}_pred.npy"
                new = output / "smoke_mean_one_frame" / "result" / f"{token}_pred.npy"
                same = np.array_equal(np.load(old, allow_pickle=False), np.load(new, allow_pickle=False))
                write_json(output / "smoke_equivalence.json", {
                    "token": token, "identical_point_predictions": same,
                    "reference": str(old), "kind": "same single frame, no extra inference pass",
                })
                if not same:
                    raise RuntimeError("Fast single-frame prediction differs from reference; inspect before full run")
        print(f"Starting full-val experiments directly: {len(tokens)} frames per method", flush=True)
        write_json(output / "status.json", {"status": "running", "stage": "full_validation"})
        for variant, gpu in VARIANTS:
            command = evaluation_command(
                paths_file, variant, output / "tokens.json", gpu, output / variant, len(tokens),
                args.io_workers, args.prefetch_frames, args.image_cache,
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
                        delta_miou_pp=round(metrics["miou_pct"] - args.baseline_miou, 4),
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
