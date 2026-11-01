import tempfile
import unittest
from pathlib import Path
import pickle
from unittest.mock import patch
from types import SimpleNamespace

from pointcloud_lab.results import read_test_metrics
from pointcloud_lab.run_views import evaluation_command, main
from pointcloud_lab.common import write_json, read_json


class ResultTests(unittest.TestCase):
    def test_parse_upstream_final_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "test.log"
            log.write_text(
                "Val result: mIoU/mAcc/allAcc 0.8248/0.9003/0.9463\n"
                "Class_5 - motorcycle Result: iou/accuracy 0.9263/0.9300\n"
            )
            result = read_test_metrics(log)
            self.assertAlmostEqual(result["miou_pct"], 82.48)
            self.assertAlmostEqual(result["per_class"]["motorcycle"]["iou_pct"], 92.63)
            log.write_text("Test: frame 1\n")
            with self.assertRaises(ValueError):
                read_test_metrics(log)

    def test_batch_commands_use_full_token_count_and_explicit_gpu(self):
        command = evaluation_command(
            Path("paths.json"), "mean_visible_cameras", Path("tokens.json"),
            "1", Path("fresh_run"), 6019,
        )
        self.assertEqual(command[command.index("--gpu") + 1], "1")
        self.assertEqual(command[command.index("--expected-frames") + 1], "6019")

    def setup_batch(self, root):
        index = root / "info.pkl"
        with index.open("wb") as stream:
            pickle.dump([{"lidar_token": f"token_{i}"} for i in range(3)], stream)
        paths = root / "paths.json"
        write_json(paths, {"info": str(index)})
        return paths

    def test_failed_smoke_prevents_full_gpu_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.setup_batch(root)
            argv = ["run_views", "--paths", str(paths), "--out", str(root / "run")]
            with patch("sys.argv", argv), patch(
                "pointcloud_lab.run_views.git_state", return_value={"sha": "test", "dirty": ""}
            ), patch(
                "pointcloud_lab.run_views.subprocess.run",
                return_value=SimpleNamespace(returncode=1)
            ), patch("pointcloud_lab.run_views.subprocess.Popen") as launch:
                with self.assertRaisesRegex(RuntimeError, "One-frame run failed"):
                    main()
                launch.assert_not_called()
            self.assertEqual(read_json(root / "run/status.json")["status"], "failed")

    def test_two_full_conditions_without_pilot_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.setup_batch(root)
            devices = []

            def fake_launch(command, **kwargs):
                gpu = command[command.index("--gpu") + 1]
                devices.append(gpu)
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], gpu)
                self.assertEqual(command[command.index("--expected-frames") + 1], "3")
                out = Path(command[command.index("--out") + 1])
                out.mkdir()
                (out / "test.log").write_text(
                    "Val result: mIoU/mAcc/allAcc 0.8300/0.9000/0.9500\n")
                return SimpleNamespace(poll=lambda: 0, returncode=0)

            argv = ["run_views", "--paths", str(paths), "--out", str(root / "run"), "--skip-smoke"]
            with patch("sys.argv", argv), patch(
                "pointcloud_lab.run_views.git_state", return_value={"sha": "test", "dirty": ""}
            ), patch(
                "pointcloud_lab.run_views.subprocess.Popen", side_effect=fake_launch
            ):
                main()
            self.assertEqual(devices, ["0", "1"])
            summary = read_json(root / "run/summary.json")
            self.assertEqual(summary["frames_per_method"], 3)
            self.assertEqual(len(summary["results"]), 2)
            self.assertAlmostEqual(summary["results"][0]["delta_miou_pp"], 0.52)
            self.assertEqual(read_json(root / "run/status.json")["status"], "completed")


if __name__ == "__main__":
    unittest.main()
