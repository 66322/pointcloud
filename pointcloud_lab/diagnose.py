"""Read existing final predictions in ORIGINAL LiDAR point order; no model/GPU."""
import argparse
import csv
import platform
import socket
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .common import (
    fresh_dir, git_state, load_infos, read_json, resolve_raw, select_infos,
    timestamp, write_json,
)
from .geometry import group_masks, project
from .metrics import (
    CLASS_NAMES, LEARNING_MAP, LABEL_MAPPING_VERSION, bootstrap_scene_miou, confusion, scores,
)


def diagnose(paths, out, limit=100, seed=20261101, tokens=None, scene_map=None,
             alignment_confirmed=False, bootstrap=500):
    if not alignment_confirmed:
        raise ValueError("Require explicit confirmation that final caches use original point order")
    selected = select_infos(load_infos(paths["info"]), limit=limit, seed=seed, tokens=tokens)
    output = fresh_dir(out)
    write_json(output / "status.json", {"status": "running", "started_at": timestamp()})
    matrices, scene_matrices = {}, defaultdict(lambda: np.zeros((16, 16), dtype=np.int64))
    manifest, point_counts, relationship_counts = [], defaultdict(int), defaultdict(int)
    try:
        for frame_index, info in enumerate(selected):
            token = str(info["lidar_token"])
            lidar_path = resolve_raw(paths["data_root"], info["lidar_path"])
            gt_path = resolve_raw(paths["data_root"], info["gt_segment_path"])
            raw = np.fromfile(lidar_path, dtype=np.float32)
            if raw.size % 5:
                raise ValueError(f"{token}: LiDAR shape not divisible by 5")
            points = raw.reshape(-1, 5)[:, :3]
            if not np.isfinite(points).all():
                raise ValueError(f"{token}: nonfinite coordinates")
            raw_gt = np.fromfile(gt_path, dtype=np.uint8)
            pred_path = Path(paths["baseline_predictions"]) / f"{token}_pred.npy"
            pred = np.load(pred_path, allow_pickle=False)
            if len(raw_gt) != len(points) or pred.shape != (len(points),):
                raise ValueError(f"{token}: original LiDAR/GT/cache length mismatch")
            if np.any(raw_gt >= len(LEARNING_MAP)):
                raise ValueError(f"{token}: unknown raw GT class")
            target = LEARNING_MAP[raw_gt]
            confusion(pred, target)  # Validate IDs even if a later group is empty.
            visibility = []
            for camera in info["cams"].values():
                image_path = resolve_raw(paths["data_root"], camera["data_path"])
                with Image.open(image_path) as image:
                    w, h = image.size
                _, mask = project(points, camera, (h, w))
                visibility.append(mask)
            if len(visibility) != 6:
                raise ValueError(f"{token}: expected 6 cameras, got {len(visibility)}")
            visible = np.stack(visibility, axis=1)
            groups = group_masks(points, visible)
            for name, mask in groups.items():
                matrices.setdefault(name, np.zeros((16, 16), dtype=np.int64))
                matrices[name] += confusion(pred[mask], target[mask])
                point_counts[name] += int(mask.sum())
                relationship_counts[name] += int(visible[mask].sum())
            frame_cm = confusion(pred, target)
            if scene_map is not None:
                if token not in scene_map:
                    raise ValueError(f"{token}: missing scene mapping")
                scene_matrices[str(scene_map[token])] += frame_cm
            manifest.append({
                "token": token, "points": len(points),
                "labelled_points": int((target != -1).sum()),
                "covered_distinct_points": int(visible.any(1).sum()),
                "point_camera_relationships": int(visible.sum()),
            })
            if (frame_index + 1) % 20 == 0:
                print(f"diagnosed {frame_index + 1}/{len(selected)}", flush=True)
        summary = {name: {
            **scores(cm), "all_points": point_counts[name],
            "point_camera_relationships": relationship_counts[name],
        } for name, cm in matrices.items()}
        ci = bootstrap_scene_miou(list(scene_matrices.values()), bootstrap, seed) if scene_map else None
        off_diagonal = matrices["all"].copy()
        np.fill_diagonal(off_diagonal, 0)
        pairs = []
        for flat in np.argsort(off_diagonal, axis=None)[::-1][:10]:
            truth, prediction = np.unravel_index(flat, off_diagonal.shape)
            if off_diagonal[truth, prediction] > 0:
                pairs.append({"true_class": CLASS_NAMES[truth],
                              "predicted_class": CLASS_NAMES[prediction],
                              "points": int(off_diagonal[truth, prediction])})
        write_json(output / "summary.json", {
            "kind": "existing_prediction_diagnostic", "frames": len(selected),
            "groups": summary, "scene_cluster_ci95_miou_fixed16_pct": ci,
            "top_confusion_pairs_by_point_count": pairs,
            "ci_note": "scene bootstrap" if ci else "not estimated; need >=2 scenes and scene map",
            "visibility_definition": "geometric frustum coverage, NOT occlusion-verified visibility",
            "alignment_evidence": "existing verified cache + token/count/range checks; "
                                  "shape alone does NOT prove semantic order",
        })
        with (output / "per_class.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["group", "class_id", "class_name", "target_points", "union", "iou_pct"])
            for group, result in summary.items():
                for i, name in enumerate(CLASS_NAMES):
                    writer.writerow([group, i, name, result["target_support"][i],
                                     result["union"][i], result["iou_pct"][i]])
        write_json(output / "frame_manifest.json", manifest)
        write_json(output / "tokens.json", [x["lidar_token"] for x in selected])
        write_json(output / "confusions.json", {k: x.tolist() for k, x in matrices.items()})
        write_json(output / "provenance.json", {
            "created_at": timestamp(), "seed": seed,
            "label_mapping_version": LABEL_MAPPING_VERSION,
            "lab_git": git_state(Path(__file__).resolve().parents[1]),
            "upstream_git": git_state(paths["ditr_root"]),
            "python": platform.python_version(), "host": socket.gethostname(),
            "paths": paths, "selected_tokens": [x["lidar_token"] for x in selected],
        })
        overall = summary["all"]
        lines = [
            "# Existing-cache diagnostic", "",
            f"Frames: {len(selected)}; labelled points: {overall['labelled_points']}.",
            f"Fixed-16 mIoU: {overall['miou_fixed16_pct']:.4f}%; "
            f"present-union classes: {overall['present_union_classes']}/16.",
            "Existing-cache analysis; this is not a new-model improvement.", "",
            "| Group | Points (incl ignored) | Labelled | mIoU fixed16 % | OA % |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, result in summary.items():
            lines.append(f"| {name} | {result['all_points']} | {result['labelled_points']} | "
                         f"{result['miou_fixed16_pct']:.4f} | {result['oa_pct']} |")
        lines += ["", "Frustum coverage is not physical visibility.",
                  "Per-class absent unions are null in JSON/CSV; fixed16 matches upstream zero convention.",
                  "Distance × class counts/IoUs are in per_class.csv for each range group."]
        (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_json(output / "status.json", {"status": "completed", "ended_at": timestamp()})
    except Exception as error:
        write_json(output / "status.json", {
            "status": "failed", "ended_at": timestamp(), "error": str(error),
        })
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--tokens", help="JSON array of exact lidar tokens")
    parser.add_argument("--scene-map", help="Optional JSON lidar_token -> scene_token")
    parser.add_argument("--predictions", help="Override cache source for a NEW inference run")
    parser.add_argument("--confirm-original-point-order", action="store_true")
    args = parser.parse_args()
    paths = read_json(args.paths)
    if args.predictions:
        paths["baseline_predictions"] = args.predictions
    diagnose(paths, args.out, None if args.full else args.limit, args.seed,
             read_json(args.tokens) if args.tokens else None,
             read_json(args.scene_map) if args.scene_map else None,
             args.confirm_original_point_order)


if __name__ == "__main__":
    main()
