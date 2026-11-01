import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from pointcloud_lab.campaign import (
    controller_lock, execute_queue, make_jobs, reuse_prior,
    validate_plan, wait_for_prior, write_reports, main as campaign_main,
)
from pointcloud_lab.common import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def plan():
    return validate_plan(read_json(ROOT / "configs/training/multi_route_queue.json"))


class CampaignTests(unittest.TestCase):
    def test_default_has_nine_new_short_six_medium_and_twelve_stress_evaluations(self):
        jobs = make_jobs(plan(), Path("/run"))
        self.assertEqual(len(jobs), 50)
        self.assertEqual(sum(j["kind"] == "train" for j in jobs), 19)
        self.assertEqual(sum(j["kind"] == "eval" for j in jobs), 31)
        self.assertEqual(sum(j["phase"] == "medium" and j["kind"] == "train" for j in jobs), 6)
        self.assertEqual(sum(j["phase"] in {"robust", "projection"} for j in jobs), 12)
        self.assertEqual(len({j["id"] for j in jobs}), len(jobs))
        invalid = copy.deepcopy(plan())
        invalid["medium_methods"].remove("control")
        with self.assertRaises(ValueError):
            validate_plan(invalid)

    def test_waits_without_gpu_launch_and_reuses_only_completed_matching_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prior, out = root / "prior", root / "out"
            prior.mkdir()
            out.mkdir()
            write_json(prior / "status.json", {"status": "running"})
            def finish(_):
                write_json(prior / "status.json", {"status": "completed"})
            with patch("pointcloud_lab.campaign.time.sleep", side_effect=finish):
                wait_for_prior(prior, out, 0)
            self.assertEqual(read_json(out / "status.json")["status"], "waiting_for_prior")
            protocol = plan()["screen_protocol"]
            write_json(prior / "protocol.json", dict(protocol, id="old", methods=["control"]))
            write_json(prior / "paths.json", {"info": "same"})
            write_json(prior / "tokens.json", ["a"])
            train = prior / "control/train"
            train.mkdir(parents=True)
            (train / "model").mkdir()
            (train / "model/model_last.pth").touch()
            write_json(train / "status.json", {"status": "completed"})
            write_json(train / "summary.json", {"optimizer_updates": 9376})
            write_json(train / "experiment.json", {"method": "control", "protocol": protocol})
            val = prior / "control/val"
            val.mkdir()
            write_json(val / "status.json", {"status": "completed"})
            write_json(val / "summary.json", {"miou_pct": 82., "macc_pct": 90., "oa_pct": 94.})
            write_json(val / "provenance.json", {"tokens": ["a"], "experiment": {"assignment": "official"}})
            jobs = make_jobs(plan(), out)
            result = reuse_prior(jobs, prior, protocol, {"info": "same"}, ["a"])
            self.assertEqual(len(result["reused"]), 2)
            self.assertEqual(jobs[0]["output"], str(train))
            self.assertEqual(len(reuse_prior(jobs, prior, protocol, {"info": "same"}, ["a"])["reused"]), 2)
            mismatch = reuse_prior(make_jobs(plan(), out), prior,
                                   dict(protocol, epochs=10), {"info": "same"}, ["a"])
            self.assertEqual(mismatch["reused"], [])
            write_json(val / "status.json", {"status": "running"})
            jobs = make_jobs(plan(), out)
            self.assertEqual(len(reuse_prior(jobs, prior, protocol, {"info": "same"}, ["a"])["reused"]), 1)

    def test_two_gpu_dependencies_failure_isolation_and_completed_not_relaunched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "logs").mkdir()
            (root / "jobs").mkdir()
            protocol = plan()
            protocol["screen_protocol"]["methods"] = ["control", "bilinear", "local_attention"]
            protocol.update(medium_methods=["control"], robust_methods=[], projection_methods=[])
            jobs = make_jobs(protocol, root)
            jobs[0].update(status="reused", output=str(root / "existing_control"),
                           summary={"optimizer_updates": 9376})
            jobs[1].update(status="reused", summary={"miou_pct": 82., "macc_pct": 90., "oa_pct": 94.})
            running, commands = {}, []
            def launch(command, **kwargs):
                gpu = command[command.index("--gpu") + 1]
                self.assertIn(gpu, ("0", "1"))
                self.assertNotIn(gpu, running)
                self.assertEqual(kwargs["stdin"], -3)  # subprocess.DEVNULL
                commands.append(command)
                path = Path(command[command.index("--out") + 1])
                path.mkdir()
                failed = "screen_bilinear_train" == path.name
                summary = ({"optimizer_updates": 9376} if "pointcloud_lab.train" in command else
                           {"miou_pct": 83., "macc_pct": 90., "oa_pct": 94.})
                write_json(path / "summary.json", summary)
                write_json(path / "status.json", {"status": "failed" if failed else "completed"})
                process = SimpleNamespace(pid=100 + len(commands), returncode=None)
                def poll():
                    process.returncode = 1 if failed else 0
                    running.pop(gpu, None)
                    return process.returncode
                process.poll = poll
                running[gpu] = process
                return process
            with patch("pointcloud_lab.campaign.subprocess.Popen", side_effect=launch), patch(
                "pointcloud_lab.campaign.gpu_idle", side_effect=lambda gpu: gpu not in running
            ), patch("pointcloud_lab.campaign.time.sleep"):
                status = execute_queue(root, jobs, 6019, poll_seconds=0)
                self.assertEqual(len(commands), 5)
                # A completed queue can be resumed without repeating its work.
                execute_queue(root, jobs, 6019, poll_seconds=0)
                self.assertEqual(len(commands), 5)
            self.assertEqual(status, "completed_with_failures")
            by_id = {j["id"]: j for j in jobs}
            self.assertEqual(by_id["screen_bilinear_clean"]["status"], "blocked")
            self.assertEqual(by_id["screen_local_attention_clean"]["status"], "completed")
            self.assertEqual(by_id["medium_control_clean"]["status"], "completed")
            for command in commands:
                if "pointcloud_lab.evaluate" in command:
                    self.assertEqual(command[command.index("--expected-frames") + 1], "6019")
                    self.assertEqual(command[command.index("--camera-count") + 1], "6")
            rows = read_json(root / "summary.json")["results"]
            self.assertEqual(next(r for r in rows if r["method"] == "local_attention")[
                "delta_vs_matched_control_pp"], 1.)

    def test_resume_adopts_live_job_and_does_not_launch_a_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "logs").mkdir()
            output = root / "train"
            output.mkdir()
            write_json(output / "status.json", {"status": "completed"})
            write_json(output / "summary.json", {"optimizer_updates": 10})
            jobs = [dict(id="old", kind="train", phase="screen", method="control",
                         dependencies=[], output=str(output), status="running", gpu="0", pid=123)]
            with patch("pointcloud_lab.campaign.process_alive", side_effect=[True, False]), patch(
                "pointcloud_lab.campaign.subprocess.Popen"
            ) as launch, patch("pointcloud_lab.campaign.time.sleep"):
                self.assertEqual(execute_queue(root, jobs, 1, poll_seconds=0), "completed")
                launch.assert_not_called()

    def test_controller_lock_rejects_duplicate_and_releases_on_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = controller_lock(Path(tmp))
            try:
                with self.assertRaises(RuntimeError):
                    controller_lock(Path(tmp))
            finally:
                first.close()
            controller_lock(Path(tmp)).close()

    def test_reports_never_mix_screen_medium_or_missing_camera_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs = []
            for phase, condition, method, score in (
                ("screen", "clean", "control", 82), ("medium", "clean", "control", 83),
                ("medium", "clean", "local_attention", 84),
                ("robust", "camera_0", "control", 70), ("robust", "camera_0", "camera_dropout", 75),
            ):
                jobs.append(dict(id=f"{phase}_{method}", kind="eval", phase=phase,
                                 condition=condition, method=method, status="completed", output="test",
                                 summary={"miou_pct": score, "macc_pct": 90., "oa_pct": 94.}))
            write_reports(Path(tmp), jobs)
            rows = read_json(Path(tmp) / "summary.json")["results"]
            self.assertEqual(rows[2]["delta_vs_matched_control_pp"], 1)
            self.assertEqual(rows[4]["delta_vs_matched_control_pp"], 5)

    def test_completed_campaign_exports_new_jobs_and_reuse_references_without_large_artifacts(self):
        from pointcloud_lab.export_record import export
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "run", Path(tmp) / "record"
            source.mkdir()
            write_json(source / "status.json", {"status": "completed_with_failures"})
            write_json(source / "queue.json", {"jobs": [
                {"id": "new", "status": "completed"}, {"id": "old", "status": "reused"},
                {"id": "bad", "status": "failed", "error": "test failure"}]})
            write_json(source / "reuse.json", {"reused": ["old"], "source": "/existing/run"})
            child = source / "jobs/new"
            child.mkdir(parents=True)
            write_json(child / "status.json", {"status": "completed"})
            write_json(child / "summary.json", {"miou_pct": 83.})
            (child / "model.pth").write_bytes(b"do not publish")
            (child / "frame_pred.npy").write_bytes(b"do not publish")
            export(source, target)
            self.assertTrue((target / "jobs/new/summary.json").is_file())
            self.assertEqual(read_json(target / "reuse.json")["reused"], ["old"])
            self.assertEqual(list(target.rglob("*.pth")), [])
            self.assertEqual(list(target.rglob("*.npy")), [])
            write_json(source / "status.json", {"status": "waiting_for_prior"})
            with self.assertRaises(ValueError):
                export(source, Path(tmp) / "partial")

    def test_main_snapshots_runs_full_dag_and_resume_does_not_repeat_jobs(self):
        import pickle
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = root / "val.pkl"
            with index.open("wb") as stream:
                pickle.dump([{"lidar_token": "a"}, {"lidar_token": "b"}], stream)
            paths = root / "paths.json"
            write_json(paths, {"info": str(index)})
            definition = plan()
            definition["screen_protocol"]["methods"] = ["control", "center_select"]
            definition.update(medium_methods=["control"], robust_methods=[], projection_methods=[])
            plan_file = root / "plan.json"
            write_json(plan_file, definition)
            commands = []
            def launch(command, **kwargs):
                commands.append(command)
                output = Path(command[command.index("--out") + 1])
                output.mkdir()
                summary = ({"optimizer_updates": 10} if "pointcloud_lab.train" in command else
                           {"miou_pct": 83., "macc_pct": 90., "oa_pct": 94.})
                write_json(output / "status.json", {"status": "completed"})
                write_json(output / "summary.json", summary)
                return SimpleNamespace(pid=123, poll=lambda: 0)
            argv = ["campaign", "--paths", str(paths), "--plan", str(plan_file),
                    "--out", str(root / "run")]
            with patch("sys.argv", argv), patch("pointcloud_lab.campaign.gpu_idle", return_value=True), patch(
                "pointcloud_lab.campaign.subprocess.Popen", side_effect=launch
            ), patch("pointcloud_lab.campaign.time.sleep"), patch(
                "pointcloud_lab.campaign.git_state", return_value={"sha": "test"}
            ):
                campaign_main()
                self.assertEqual(len(commands), 6)
                with patch("sys.argv", ["campaign", "--out", str(root / "run"), "--resume"]):
                    campaign_main()
                self.assertEqual(len(commands), 6)
            self.assertEqual(read_json(root / "run/status.json")["status"], "completed")
            self.assertTrue((root / "run/code/pointcloud_lab/sampling.py").is_file())
            self.assertEqual(read_json(root / "run/medium_protocol.json")["epochs"], 10)
            self.assertEqual(read_json(root / "run/tokens.json"), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
