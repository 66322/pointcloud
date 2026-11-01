import numpy as np


def project(points, camera, image_hw):
    rotation = np.asarray(camera["sensor2lidar_rotation"], dtype=np.float64)
    translation = np.asarray(camera["sensor2lidar_translation"], dtype=np.float64)
    # Row-vector inverse of p_lidar = R @ p_camera + t.
    q = (np.asarray(points) - translation) @ rotation
    homogeneous = q @ np.asarray(camera["camera_intrinsics"]).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = homogeneous[:, :2] / homogeneous[:, 2:3]
    h, w = image_hw
    mask = (
        (q[:, 2] > 0) & np.isfinite(uv).all(1)
        & (uv[:, 0] >= 0) & (uv[:, 0] < w)
        & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    )
    return uv, mask


def group_masks(points, visible):
    counts = visible.sum(1)
    distance = np.linalg.norm(points[:, :3], axis=1)
    return {
        "all": np.ones(len(points), dtype=bool),
        "camera_0": counts == 0, "camera_1": counts == 1, "camera_ge2": counts >= 2,
        "range_0_20m": distance < 20,
        "range_20_40m": (distance >= 20) & (distance < 40),
        "range_ge40m": distance >= 40,
    }
