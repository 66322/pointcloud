import numpy as np

CLASS_NAMES = (
    "barrier", "bicycle", "bus", "car", "construction_vehicle", "motorcycle",
    "pedestrian", "traffic_cone", "trailer", "truck", "driveable_surface",
    "other_flat", "sidewalk", "terrain", "manmade", "vegetation",
)
# nuScenes raw 32-class IDs -> the pinned DITR 16-class train IDs.
LEARNING_MAP = np.array([
    -1, -1, 6, 6, 6, -1, 6, -1, -1, 0, -1, -1, 7, -1, 1, 2,
    2, 3, 4, -1, 5, 5, 8, 9, 10, 11, 12, 13, 14, -1, 15, -1,
], dtype=np.int64)


def confusion(pred, target, classes=16):
    pred, target = np.asarray(pred), np.asarray(target)
    if pred.ndim != 1 or pred.shape != target.shape:
        raise ValueError("Predictions and labels must be aligned 1-D arrays")
    if not np.issubdtype(pred.dtype, np.integer) or not np.issubdtype(target.dtype, np.integer):
        raise ValueError("Class arrays must contain integer IDs")
    if np.any((pred < 0) | (pred >= classes)):
        raise ValueError("Prediction IDs outside train-ID range; no automatic remapping")
    if np.any((target < -1) | (target >= classes)):
        raise ValueError("Target IDs outside supported range")
    mask = target != -1
    return np.bincount(
        classes * target[mask].astype(np.int64) + pred[mask],
        minlength=classes * classes,
    ).reshape(classes, classes)


def scores(cm):
    cm = np.asarray(cm)
    support = cm.sum(1)
    intersection = cm.diagonal()
    union = support + cm.sum(0) - intersection
    iou = np.divide(intersection, union, out=np.zeros(len(cm), float), where=union != 0)
    acc = np.divide(intersection, support, out=np.zeros(len(cm), float), where=support != 0)
    present = union > 0
    return {
        "miou_fixed16_pct": float(100 * iou.mean()),
        "miou_present_pct": float(100 * iou[present].mean()) if present.any() else None,
        "macc_fixed16_pct": float(100 * acc.mean()),
        "oa_pct": float(100 * intersection.sum() / support.sum()) if support.sum() else None,
        "present_union_classes": int(present.sum()),
        "labelled_points": int(support.sum()),
        "iou_pct": [float(100 * x) if u else None for x, u in zip(iou, union)],
        "target_support": support.tolist(),
        "union": union.tolist(),
    }


def bootstrap_scene_miou(scene_confusions, repeats=500, seed=20261101):
    """Cluster bootstrap over scenes, never individual points/adjacent frames."""
    arrays = np.asarray(scene_confusions)
    if len(arrays) < 2 or repeats < 1:
        return None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        cm = arrays[rng.integers(0, len(arrays), len(arrays))].sum(0)
        values.append(scores(cm)["miou_fixed16_pct"])
    return np.percentile(values, [2.5, 97.5]).tolist()
