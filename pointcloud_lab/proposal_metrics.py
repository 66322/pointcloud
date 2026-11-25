"""Original-point scores, named routing strata and true-weather transfer evidence."""
from collections import Counter
from pathlib import Path

import numpy as np

from .common import load_infos, resolve_raw, select_infos
from .metrics import CLASS_NAMES, LEARNING_MAP, confusion
from .proposal_data import camera_geometry, retained_indices, route_groups
from . import semanticstf


def describe(cm, classes):
    ids = np.asarray(classes)
    truth = cm.sum(1)[ids]
    correct = cm.diagonal()[ids]
    union = (cm.sum(0) + cm.sum(1) - cm.diagonal())[ids]
    iou = np.divide(100. * correct, union, out=np.zeros(len(ids)), where=union != 0)
    acc = np.divide(100. * correct, truth, out=np.zeros(len(ids)), where=truth != 0)
    return {"miou_pct": float(iou.mean()) if truth.sum() else None,
            "miou_present_pct": float(iou[union > 0].mean()) if union.any() else None,
            "macc_pct": float(acc.mean()) if truth.sum() else None,
            "oa_pct": float(100. * correct.sum() / truth.sum()) if truth.sum() else None,
            "labelled_points": int(truth.sum()), "classes_with_union": int((union > 0).sum()),
            "per_class": {CLASS_NAMES[c]: {"iou_pct": float(iou[k]) if union[k] else None,
                                          "accuracy_pct": float(acc[k]) if truth[k] else None,
                                          "support": int(truth[k]), "union": int(union[k])}
                          for k, c in enumerate(ids)}}


def evaluate_predictions(paths, tokens, predictions, condition):
    from PIL import Image
    external = condition == "stf_val"
    classes = sorted(semanticstf.COMMON.values()) if external else range(16)
    sums, frame_counts = {}, Counter()
    if external:
        weather = semanticstf.manifest(paths["semanticstf_root"])
        entries = [(token, None) for token in tokens]
    else:
        infos = select_infos(load_infos(paths["info"]), tokens=tokens)
        entries = [(str(info["lidar_token"]), info) for info in infos]
    for token, info in entries:
        pred = np.load(Path(predictions) / (token + "_pred.npy"), allow_pickle=False)
        if external:
            label = semanticstf.labels(paths["semanticstf_root"], token)
            groups = {weather[token]: slice(None)}
        else:
            raw = paths["data_root"]
            all_xyz = np.fromfile(resolve_raw(raw, info["lidar_path"]), np.float32).reshape(-1, 5)[:, :3]
            raw_labels = np.fromfile(resolve_raw(raw, info["gt_segment_path"]), np.uint8)
            if len(raw_labels) != len(all_xyz):
                raise ValueError(f"Original point/label mismatch in {token}")
            keep = retained_indices(all_xyz, token, condition)
            xyz, label = all_xyz[keep], LEARNING_MAP[raw_labels[keep]]
            first = next(iter(info["cams"].values()))
            with Image.open(resolve_raw(raw, first["data_path"])) as im:
                geom = camera_geometry(xyz, info["cams"], im.size[::-1], condition)
            groups = route_groups(geom)
            distance = np.linalg.norm(xyz, axis=1)
            groups.update(range_lt20=distance < 20, range_20_40=(distance >= 20) & (distance < 40),
                          range_ge40=distance >= 40)
            if condition == "clean":
                groups["sparse_support"] = retained_indices(xyz, token, "sparse")
        if pred.shape != label.shape:
            raise ValueError(f"Prediction count differs from scored original points in {token}")
        for group, mask in {"all": slice(None), **groups}.items():
            cm = confusion(pred[mask], label[mask])
            if group not in sums:
                sums[group] = np.zeros((16, 16), dtype=np.int64)
            sums[group] += cm
            frame_counts[group] += 1
    result = {name: dict(describe(cm, classes), frames=frame_counts[name]) for name, cm in sums.items()}
    return {**result.pop("all"), "groups": result,
            "scope": "SemanticSTF common9 val" if external else "nuScenes lidarseg16 val",
            "source": "original-point confusion counts; absent union contributes zero to fixed-class mIoU"}
