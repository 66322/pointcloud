import tempfile
import unittest
from pathlib import Path
import pickle

import numpy as np
from PIL import Image

from pointcloud_lab.common import read_json, select_infos
from pointcloud_lab.compare import compare
from pointcloud_lab.diagnose import diagnose
from pointcloud_lab.export_record import export
from pointcloud_lab.geometry import project
from pointcloud_lab.metrics import LEARNING_MAP, bootstrap_scene_miou, confusion, scores


class MetricTests(unittest.TestCase):
    def test_ignore_and_absent_classes(self):
        cm = confusion(np.array([0, 1, 0]), np.array([0, 0, -1]))
        result = scores(cm)
        self.assertEqual(result["labelled_points"], 2)
        self.assertEqual(result["iou_pct"][0], 50)
        self.assertEqual(result["iou_pct"][1], 0)
        self.assertIsNone(result["iou_pct"][2])
        self.assertEqual(result["oa_pct"], 50)

    def test_reject_remapped_ids_or_shape(self):
        with self.assertRaises(ValueError):
            confusion(np.array([16]), np.array([0]))
        with self.assertRaises(ValueError):
            confusion(np.array([[0]]), np.array([0]))
        with self.assertRaises(ValueError):
            confusion(np.array([0.0]), np.array([0]))
        self.assertEqual(LEARNING_MAP[9], 0)
        self.assertEqual(LEARNING_MAP[30], 15)

    def test_scene_bootstrap_not_point_bootstrap(self):
        self.assertIsNone(bootstrap_scene_miou([np.eye(16)], 20))
        ci = bootstrap_scene_miou([np.eye(16), np.eye(16)], 20)
        self.assertEqual(ci, [100, 100])

    def test_deterministic_nested_selection(self):
        infos = [{"lidar_token": str(i)} for i in range(10)]
        small = select_infos(infos, 2)
        large = select_infos(list(reversed(infos)), 8)
        self.assertEqual(small, large[:2])
        with self.assertRaises(ValueError):
            select_infos(infos, tokens=["not-there"])

    def test_projection_inverse_and_frustum(self):
        cam = {"sensor2lidar_rotation": np.eye(3), "sensor2lidar_translation": [1, 0, 0],
               "camera_intrinsics": np.eye(3)}
        uv, mask = project(np.array([[2, 1, 1], [2, 1, -1], [5, 1, 1]]), cam, (3, 3))
        np.testing.assert_array_equal(mask, [True, False, False])
        np.testing.assert_allclose(uv[0], [1, 1])


class IntegrationTests(unittest.TestCase):
    def fixture(self, root):
        raw = root / "data" / "raw"
        raw.mkdir(parents=True)
        np.array([[1, 1, 1, 10, 0], [30, 0, 1, 10, 0],
                  [50, 0, -1, 10, 0]], np.float32).tofile(raw / "points.bin")
        np.array([9, 14, 0], np.uint8).tofile(raw / "labels.bin")
        Image.new("RGB", (10, 10)).save(raw / "image.jpg")
        camera = {"sensor2lidar_rotation": np.eye(3),
                  "sensor2lidar_translation": np.zeros(3), "camera_intrinsics": np.eye(3),
                  "data_path": "image.jpg"}
        info = [{"lidar_token": "frame", "lidar_path": "points.bin",
                 "gt_segment_path": "labels.bin",
                 "cams": {str(i): camera for i in range(6)}}]
        with (root / "info.pkl").open("wb") as stream:
            pickle.dump(info, stream)
        predictions = root / "pred"
        predictions.mkdir()
        np.save(predictions / "frame_pred.npy", np.array([0, 1, 0]))
        return {"data_root": str(root / "data"), "info": str(root / "info.pkl"),
                "baseline_predictions": str(predictions), "ditr_root": str(root)}

    def test_full_readonly_diagnostic_and_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.fixture(root)
            original = (root / "pred/frame_pred.npy").read_bytes()
            output = root / "analysis"
            diagnose(paths, output, 1, alignment_confirmed=True)
            report = read_json(output / "summary.json")
            self.assertEqual(report["groups"]["all"]["labelled_points"], 2)
            self.assertEqual(report["groups"]["camera_ge2"]["all_points"], 1)
            self.assertEqual(report["groups"]["all"]["point_camera_relationships"], 6)
            self.assertEqual(read_json(output / "tokens.json"), ["frame"])
            self.assertEqual(original, (root / "pred/frame_pred.npy").read_bytes())
            with self.assertRaises(FileExistsError):
                diagnose(paths, output, 1, alignment_confirmed=True)
            export(output, root / "record")
            self.assertTrue((root / "record/summary.json").is_file())
            self.assertFalse(any((root / "record").glob("*.npy")))
            second = root / "second"
            np.save(root / "pred/frame_pred.npy", np.array([1, 1, 0]))
            diagnose(paths, second, 1, alignment_confirmed=True)
            compared = compare(output, second, root / "comparison")
            self.assertLess(read_json(compared / "summary.json")["groups"][0]["delta_miou_pp"], 0)
            # Identical lengths with different raw point order must fail paired comparison.
            raw_path = root / "data/raw/points.bin"
            points = np.fromfile(raw_path, np.float32).reshape(-1, 5)
            points[[0, 1]] = points[[1, 0]]
            points.tofile(raw_path)
            diagnose(paths, root / "changed_input", 1, alignment_confirmed=True)
            with self.assertRaisesRegex(ValueError, "input_array_set_sha256"):
                compare(output, root / "changed_input", root / "should_not_exist")

    def test_alignment_rejected_before_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.fixture(root)
            with self.assertRaises(ValueError):
                diagnose(paths, root / "unconfirmed", 1)
            np.save(root / "pred/frame_pred.npy", np.array([0, 1]))
            with self.assertRaises(ValueError):
                diagnose(paths, root / "bad", 1, alignment_confirmed=True)
            self.assertEqual(read_json(root / "bad/status.json")["status"], "failed")


if __name__ == "__main__":
    unittest.main()
