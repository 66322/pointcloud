"""Physical projection errors, named view loss and paired raw-frame corruptions."""
from functools import partial
import zlib

import numpy as np

RING = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK",
        "CAM_BACK_LEFT", "CAM_FRONT_LEFT")
ALIGN = {"yaw_m1": -1., "yaw_p1": 1., "yaw_m3": -3., "yaw_p3": 3., "tx_020": 0.}
QUALITY = ("dark", "blur", "sparse", "dark_sparse")
MISSING = {"triple_a": RING[::2], "triple_b": RING[1::2], "none": (),
           **{"only_" + c[4:].lower(): (c,) for c in RING}}
CONDITIONS = ("clean", *ALIGN, *QUALITY, *MISSING)
POINT_KEYS = ("lab_depth", "lab_primary")
FRAME_KEYS = ("lab_keep", "lab_pose", "lab_depth_ok")


def configure_fields(split):
    """Carry point metadata through *both* upstream voxel samplers, frame data through Collect."""
    transforms = list(split["transform"])
    if split.get("test_mode"):
        test = split["test_cfg"]
        transforms += [test["voxelize"], *test["post_transform"]]
    for step in transforms:
        if step["type"] in {"GridSample", "Collect"}:
            extra = POINT_KEYS + (FRAME_KEYS if step["type"] == "Collect" else ())
            step["keys"] = tuple(dict.fromkeys((*step["keys"], *extra)))


def training_condition(distribution="mixed"):
    # PairedDataset seeds NumPy by (seed, epoch, row) in each worker.
    u = np.random.rand()
    if u < .5:
        return "clean", None
    if distribution == "alignment":
        return ("yaw_m3" if u < .75 else "yaw_p3"), None
    if u < .65:
        return np.random.choice(list(ALIGN)), None
    if u < .8:
        return np.random.choice(QUALITY), None
    k = np.random.randint(6)
    return "random_missing", tuple(np.random.choice(RING, k, replace=False))


def retained_indices(xyz, name, condition):
    if condition not in {"sparse", "dark_sparse"}:
        return slice(None)
    # Use the identical draw in the sparse and joint conditions, independent of training seed.
    rng = np.random.default_rng(zlib.crc32(name.encode("utf-8")))
    azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
    sector = (np.abs(azimuth) < np.pi / 3) & (np.linalg.norm(xyz, axis=1) >= 20.)
    keep = ~sector | (rng.random(len(xyz)) < .25)
    if len(keep) and not keep.any():
        keep[0] = True
    return np.flatnonzero(keep)


def corrupt_image(images, condition):
    if condition not in {"dark", "dark_sparse", "blur"}:
        return images
    import cv2
    result = images.copy()
    _, h, w, _ = result.shape
    for image in result:
        region = image[h // 4:3 * h // 4, w // 4:3 * w // 4]
        if condition == "blur":
            region[:] = cv2.GaussianBlur(region, (31, 31), 8)
        else:
            region[:] = (region.astype(np.float32) * .2).astype(region.dtype)
    return result


def camera_geometry(xyz, cameras, hw, condition, keep_names=None):
    names = tuple(cameras)
    if not names or (len(names) > 1 and set(names) != set(RING)):
        raise ValueError(f"Expected the six named nuScenes cameras, got {names}")
    h, w = hw
    yaw = np.deg2rad(ALIGN.get(condition, 0.))
    cs, sn = np.cos(yaw), np.sin(yaw)
    error_rot = np.array([[cs, 0, sn], [0, 1, 0], [-sn, 0, cs]])
    uvs, depths, masks, axes, centers = [], [], [], [], []
    for cam in cameras.values():
        rotation = np.asarray(cam["sensor2lidar_rotation"], dtype=np.float64)
        center = np.asarray(cam["sensor2lidar_translation"], dtype=np.float64)
        q = ((xyz - center) @ rotation) @ error_rot.T
        if condition == "tx_020":
            q[:, 0] += .20
        z = q[:, 2]
        uvw = q @ np.asarray(cam["camera_intrinsics"]).T
        uv = np.divide(uvw[:, :2], uvw[:, 2:], out=np.zeros_like(uvw[:, :2]),
                       where=np.abs(uvw[:, 2:]) > 1e-8)
        uv = uv.astype(np.float32)
        valid = ((z > 1e-8) & np.isfinite(uv).all(1) & (uv[:, 0] >= 0)
                 & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h))
        uvs.append(uv)
        masks.append(valid)
        depths.append(z.astype(np.float32))
        axes.append(np.arctan2(rotation[1, 2], rotation[0, 2]))
        centers.append(center)
    uv = np.stack(uvs, axis=1)
    mask = np.stack(masks, axis=1)
    radius = np.square((uv + .5) / [w, h] - .5).sum(-1)
    primary = np.where(mask, radius, np.inf).argmin(1)
    allowed = MISSING.get(condition, names) if keep_names is None else keep_names
    keep = np.asarray([n in allowed for n in names], dtype=bool)
    pose = np.zeros((len(names), len(names), 4), dtype=np.float32)
    for i, src in enumerate(names):
        for j, dst in enumerate(names):
            adjacent = len(names) > 1 and (RING.index(src) - RING.index(dst)) % 6 in (1, 5)
            angle = axes[j] - axes[i]
            pose[i, j] = adjacent, np.sin(angle), np.cos(angle), np.linalg.norm(centers[j] - centers[i])
    return dict(image_coord=uv, image_mask=mask, lab_depth=np.stack(depths, axis=1),
                lab_primary=primary, lab_keep=keep[None], lab_pose=pose[None])


def prepare_frame(raw, cameras, condition, keep_names=None):
    selected = retained_indices(raw["coord"], str(raw["name"]), condition)
    frame = {k: raw[k][selected] for k in ("coord", "strength", "segment")}
    frame["name"] = raw["name"]
    frame["image"] = corrupt_image(raw["image"], condition)
    frame.update(camera_geometry(frame["coord"], cameras, frame["image"].shape[1:3],
                                 condition, keep_names))
    frame["lab_depth_ok"] = np.array([condition not in ALIGN], dtype=bool)
    return frame


class ProposalReader:
    """Keep upstream raw reads/cache usable with spawn and ordered eval prefetch."""
    def __init__(self, dataset, condition, cached=None, training_distribution="mixed"):
        self.dataset = dataset
        self.reader = cached if cached is not None else partial(type(dataset).get_data, dataset)
        self.condition = condition
        self.training_distribution = training_distribution

    def __call__(self, index):
        raw = self.reader(index)
        info = self.dataset.data_list[index % len(self.dataset.data_list)]
        condition, keep = (training_condition(self.training_distribution) if self.condition == "train"
                           else (self.condition, None))
        return prepare_frame(raw, info["cams"], condition, keep)


def route_groups(geometry):
    mask = geometry["image_mask"] & geometry["lab_keep"]
    primary = geometry["lab_primary"]
    direct = mask[np.arange(len(mask)), primary]
    adjacent = geometry["lab_pose"][0, primary, :, 0].astype(bool)
    neighbor = ~direct & (mask & adjacent).any(1)
    fallback = ~(direct | neighbor)
    any_current = mask.any(axis=1)
    # Preserve the historical restricted router group and expose its two distinct cases.
    return {"primary": direct, "neighbor_only": neighbor, "fallback": fallback,
            "other_current_only": fallback & any_current, "no_current_view": ~any_current}
