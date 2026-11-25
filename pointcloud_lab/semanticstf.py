"""Read the rectified left-camera SemanticSTF distribution in original point order."""
import csv
from pathlib import Path

import numpy as np

from .common import read_json
from .proposal_data import prepare_frame

# Raw SemanticSTF -> pinned nuScenes train ID. All other GT is excluded.
COMMON = {1: 3, 2: 1, 3: 5, 4: 9, 6: 6, 9: 10, 11: 12, 15: 15, 17: 13}
WEATHER = ("snow", "rain", "light_fog", "dense_fog")


def manifest(root):
    with (Path(root) / "val/val.txt").open(encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    result = {}
    for name, weather in rows:
        if name in result or Path(name).name != name or weather not in WEATHER:
            raise ValueError(f"Invalid or duplicate SemanticSTF entry: {name}, {weather}")
        result[name] = weather
    return result


def calibration(root):
    from scipy.spatial.transform import Rotation
    folder = Path(root) / "calib"
    intr = read_json(folder / "calib_cam_stereo_left.json")
    transforms = {row["child_frame_id"]: row for row in read_json(folder / "calib_tf_tree_full.json")}

    def to_body(frame):
        row = transforms[frame]
        if row["header"]["frame_id"] != "body":
            raise ValueError("Expected camera and lidar calibrations in the vehicle body frame")
        pose = row["transform"]
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat([pose["rotation"][k] for k in "xyzw"]).as_matrix()
        matrix[:3, 3] = [pose["translation"][k] for k in "xyz"]
        return matrix

    rect = np.eye(4)
    rect[:3, :3] = np.array(intr["R"]).reshape(3, 3)
    projection = np.array(intr["P"]).reshape(3, 4)
    if np.any(projection[:, 3]):
        raise ValueError("Expected rectified left camera without stereo baseline in P")
    lidar_to_rect = rect @ np.linalg.solve(to_body("cam_stereo_left_optical"), to_body("lidar_hdl64_s3_roof"))
    rect_to_lidar = np.linalg.inv(lidar_to_rect)
    return {"CAM_FRONT": {"sensor2lidar_rotation": lidar_to_rect[:3, :3].T,
                          "sensor2lidar_translation": rect_to_lidar[:3, 3],
                          "camera_intrinsics": projection[:, :3]}}, (intr["height"], intr["width"])


def labels(root, name):
    raw = np.fromfile(Path(root) / "val/labels" / (name + ".label"), dtype="<u4") & 0xffff
    if raw.size and raw.max() > 20:
        raise ValueError("SemanticSTF label outside the supplied 0–20 dictionary")
    table = np.full(21, -1, dtype=np.int64)
    for src, dst in COMMON.items():
        table[src] = dst
    return table[raw]


def read_frame(root, name, cameras, hw):
    from PIL import Image
    folder = Path(root) / "val"
    points = np.fromfile(folder / "velodyne" / (name + ".bin"), dtype="<f4").reshape(-1, 5)
    segment = labels(root, name)
    if len(points) != len(segment) or not len(points):
        raise ValueError("SemanticSTF point/label length mismatch or empty frame")
    with Image.open(folder / "images" / (name + ".png")) as im:
        if im.size != hw[::-1]:
            raise ValueError("Image dimensions differ from the left-camera calibration")
        image = np.asarray(im.convert("RGB"))[None]
    return prepare_frame({"name": name, "coord": points[:, :3], "strength": points[:, 3:4] / 255.,
                          "segment": segment, "image": image}, cameras, "clean")


def heading_alignment(stf_front, source_front):
    def angle(camera):
        axis = np.asarray(camera["sensor2lidar_rotation"])[:2, 2]
        return np.arctan2(axis[1], axis[0])
    delta = angle(source_front) - angle(stf_front)
    c, s = np.cos(delta), np.sin(delta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def build_dataset(config, root, tokens, source_front):
    from pointcept.datasets.defaults import DefaultDataset
    names = manifest(root)
    if len(set(tokens)) != len(tokens) or any(name not in names for name in tokens):
        raise ValueError("Requested frames must be unique members of SemanticSTF val")
    cameras, hw = calibration(root)
    align = heading_alignment(cameras["CAM_FRONT"], source_front)

    class LeftCameraDataset(DefaultDataset):
        def get_data_list(self):
            return list(tokens)

        def get_data(self, idx):
            frame = read_frame(root, self.data_list[idx], cameras, hw)
            # Perform projection in the native calibrated frame before rotating the PTv3 inputs.
            frame["coord"] = frame["coord"] @ align.T
            return frame

        def get_data_name(self, idx):
            return self.data_list[idx]

    return LeftCameraDataset(split="val", data_root=str(root), test_mode=True,
                             transform=config.transform, test_cfg=config.test_cfg, ignore_index=-1)
