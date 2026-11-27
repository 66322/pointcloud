"""Measure candidate image coverage of current lidar points; no network or labels."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .common import fresh_dir, git_state, load_infos, read_json, resolve_raw, select_infos, write_json
from .geometry import project
from .proposal_data import MISSING, RING


def previous_coordinates(xyz, current, previous):
    # Both matrices map global coordinates into the corresponding keyframe lidar.
    global_to_current = np.asarray(current["ref_from_car"]) @ np.asarray(current["car_from_global"])
    global_to_previous = np.asarray(previous["ref_from_car"]) @ np.asarray(previous["car_from_global"])
    transform = global_to_previous @ np.linalg.inv(global_to_current)
    return xyz @ transform[:3, :3].T + transform[:3, 3]


def projected_views(xyz, cameras, names, data_root):
    masks, distances = [], []
    for name in names:
        cam = cameras[name]
        with Image.open(resolve_raw(data_root, cam["data_path"])) as image:
            w, h = image.size
        uv, valid = project(xyz, cam, (h, w))
        radius = (((uv + .5) / [w, h] - .5) ** 2).sum(1)
        masks.append(valid)
        distances.append(np.where(valid, radius, np.inf))
    return np.column_stack(masks), np.column_stack(distances)


def coverage_counts(visible, center_distances, keep, history):
    full = visible.any(axis=1)
    surviving = visible[:, keep].any(axis=1)
    primary = center_distances.argmin(axis=1)
    counts = dict(points=len(full), current_full=int(full.sum()), current_surviving=int(surviving.sum()),
                  current_overlap=int((visible.sum(axis=1) >= 2).sum()),
                  recovered_from_other_current=int((surviving & ~keep[primary]).sum()))
    if history is not None:
        uncovered = ~surviving
        recovered = uncovered & history
        counts.update(paired_points=len(full), paired_current_uncovered=int(uncovered.sum()),
                      history_new=int(recovered.sum()), history_new_after_dropout=int((recovered & full).sum()),
                      history_new_outside_full=int((recovered & ~full).sum()),
                      paired_still_uncovered=int((uncovered & ~history).sum()))
    return counts


def audit(paths, samples, condition, outage, limit, tokens=None):
    infos = load_infos(paths["info"])
    by_sample = {info["token"]: info for info in infos}
    if len(by_sample) != len(infos):
        raise ValueError("Expected one lidar keyframe per sample")
    meta = {sample["token"]: sample for sample in samples}
    selected = select_infos(infos, limit=limit, tokens=tokens)
    names = MISSING.get(condition, RING)
    totals = Counter()
    history_gaps = []
    frames = Counter()
    for info in selected:
        xyz = np.fromfile(resolve_raw(paths["data_root"], info["lidar_path"]), np.float32).reshape(-1, 5)[:, :3]
        camera_order = tuple(info["cams"])
        if len(camera_order) != len(RING) or set(camera_order) != set(RING):
            raise ValueError("Expected the six named nuScenes cameras")
        keep = np.array([name in names for name in camera_order], dtype=bool)
        visible, distances = projected_views(xyz, info["cams"], camera_order, paths["data_root"])
        sample = meta[info["token"]]
        prev = sample["prev"]
        history = None
        if not prev:
            frames["scene_start"] += 1
        elif prev not in by_sample:
            frames["previous_outside_index"] += 1
        else:
            previous = by_sample[prev]
            if meta[prev]["scene_token"] != sample["scene_token"] or previous["timestamp"] >= info["timestamp"]:
                raise ValueError("Expected an earlier sample in the same scene")
            cutoff_us = round(info["timestamp"] * 1_000_000)
            usable = [name for name in (names if outage == "persistent" else RING)
                      if previous["cams"][name]["timestamp"] < cutoff_us]
            history = np.zeros(len(xyz), dtype=bool)
            if usable:
                past_xyz = previous_coordinates(xyz, info, previous)
                past_visible, _ = projected_views(past_xyz, previous["cams"], usable, paths["data_root"])
                history = past_visible.any(axis=1)
                history_gaps.extend((cutoff_us - previous["cams"][n]["timestamp"]) / 1000 for n in usable)
            frames["paired"] += 1
        totals.update(coverage_counts(visible, distances, keep, history))
    # Include zero counts for an empty intersection or for a selection containing only scene starts.
    keys = ("points", "current_full", "current_surviving", "current_overlap", "recovered_from_other_current",
            "paired_points", "paired_current_uncovered", "history_new", "history_new_after_dropout",
            "history_new_outside_full", "paired_still_uncovered")
    denom = totals["paired_current_uncovered"]
    return dict(condition=condition, outage=outage, retained_current_names=list(names),
                frames=dict(selected=len(selected), scene_start=frames["scene_start"],
                            previous_outside_index=frames["previous_outside_index"], paired=frames["paired"]),
                counts={key: totals[key] for key in keys},
                history_new_pct_of_paired_current_uncovered=(100 * totals["history_new"] / denom if denom else None),
                historical_camera_age_ms_range=[min(history_gaps), max(history_gaps)] if history_gaps else None,
                definition="Geometric frustum coverage of raw current points under a static-world transform.",
                split_index=Path(paths["info"]).name,
                selected_lidar_tokens=[info["lidar_token"] for info in selected])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="configs/paths.local.json")
    parser.add_argument("--samples", help="nuScenes sample.json; defaults to data_root/raw/v1.0-trainval/sample.json")
    parser.add_argument("--condition", choices=["clean", *MISSING], default="triple_a")
    parser.add_argument("--outage", choices=["persistent", "current-only"], default="persistent")
    parser.add_argument("--tokens", help="Optional existing JSON list of current lidar tokens")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", required=True, help="New output directory")
    args = parser.parse_args()
    paths = read_json(args.paths)
    source = args.samples or resolve_raw(paths["data_root"], "v1.0-trainval/sample.json")
    result = audit(paths, read_json(source), args.condition, args.outage, args.limit,
                   read_json(args.tokens) if args.tokens else None)
    result["execution"] = git_state(Path(__file__).resolve().parents[1])
    out = fresh_dir(args.out)
    write_json(out / "coverage.json", result)
    print(out / "coverage.json")


if __name__ == "__main__":
    main()
