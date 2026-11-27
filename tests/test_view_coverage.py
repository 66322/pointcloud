import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pointcloud_lab.proposal_data import RING
from pointcloud_lab.view_coverage import audit, coverage_counts, previous_coordinates


class CoverageTests(unittest.TestCase):
    def test_primary_fallback_and_incremental_coverage_have_different_denominators(self):
        visible = np.array([[1, 1], [1, 0], [0, 0], [0, 1]], bool)
        distances = np.where(visible, [[1, 2], [1, 2], [1, 2], [1, 2]], np.inf)
        values = coverage_counts(visible, distances, np.array([False, True]),
                                 np.array([True, True, False, True]))
        self.assertEqual(values["recovered_from_other_current"], 1)
        self.assertEqual(values["paired_current_uncovered"], 2)
        self.assertEqual(values["history_new"], 1)
        self.assertEqual(values["paired_still_uncovered"], 1)

    def test_transform_from_current_lidar_to_previous_lidar(self):
        current = {"car_from_global": np.eye(4), "ref_from_car": np.eye(4)}
        current["car_from_global"][0, 3] = -4
        previous = {"car_from_global": np.eye(4), "ref_from_car": np.eye(4)}
        previous["ref_from_car"][1, 3] = 1
        np.testing.assert_allclose(previous_coordinates(np.array([[2., 0., 1.]]), current, previous),
                                   [[6., 1., 1.]])

    def test_previous_keyframe_respects_scene_boundaries_outage_and_camera_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            Image.new("RGB", (4, 4)).save(raw / "image.png")
            np.array([[0., 0., 1., 0., 0.], [5., 0., 1., 0., 0.]], np.float32).tofile(raw / "points.bin")
            samples = [{"token": "a", "scene_token": "s", "prev": ""},
                       {"token": "b", "scene_token": "s", "prev": "a"}]
            frames = []
            for index, sample in enumerate(samples, start=1):
                cams = {name: dict(data_path="image.png", camera_intrinsics=[[1, 0, 2], [0, 1, 2], [0, 0, 1]],
                                   sensor2lidar_rotation=np.eye(3), timestamp=index * 1_000_000,
                                   sensor2lidar_translation=[0 if name == RING[0] else 100, 0, 0])
                        for name in RING}
                frames.append(dict(token=sample["token"], lidar_token="lidar_" + sample["token"], cams=cams,
                                   timestamp=float(index), ref_from_car=np.eye(4), car_from_global=np.eye(4),
                                   lidar_path="points.bin"))
            info = root / "val.pkl"
            paths = {"info": str(info), "data_root": str(root)}
            info.write_bytes(pickle.dumps(frames))
            persistent = audit(paths, samples, "none", "persistent", None)
            self.assertEqual(persistent["counts"]["history_new"], 0)
            current_only = audit(paths, samples, "none", "current-only", None)
            self.assertEqual(current_only["frames"], dict(selected=2, paired=1, scene_start=1, previous_outside_index=0))
            self.assertEqual(current_only["history_new_pct_of_paired_current_uncovered"], 50.)
            frames[0]["cams"][RING[0]]["timestamp"] = 2_000_000
            info.write_bytes(pickle.dumps(frames))
            self.assertEqual(audit(paths, samples, "none", "current-only", None)["counts"]["history_new"], 0)
            samples[0]["scene_token"] = "other"
            with self.assertRaisesRegex(ValueError, "same scene"):
                audit(paths, samples, "none", "current-only", None)
