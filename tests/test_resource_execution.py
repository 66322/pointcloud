import copy
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pointcloud_lab.common import read_json, write_json
from pointcloud_lab.campaign import make_jobs, reuse_campaign_results, validate_plan
from pointcloud_lab.export_record import export
from pointcloud_lab.raw_cache import CachedRawReader, dataset_identity, write_frame
from pointcloud_lab.resource_monitor import memory_info
from pointcloud_lab.training_models import AssignmentPolicy

ROOT = Path(__file__).resolve().parents[1]


class RawDataset:
    data_root = "/raw/example"
    with_images = True
    ignore_index = -1
    learning_map = {0: -1, 20: -1, 24: 10}

    def get_data_name(self, index):
        return ["a", "b"][index]

    def get_data(self, index):
        return {"name": self.get_data_name(index), "segment": np.array([-1, 10], dtype=np.int64),
                "image": np.arange(24, dtype=np.uint8).reshape(2, 2, 2, 3)}


def initialize_cache_fixture(output):
    from pointcloud_lab import raw_cache
    raw_cache._dataset = RawDataset()
    raw_cache._output = Path(output)


def read_cache_fixture(dataset, index):
    return dataset.get_data(index)


class ResourceExecutionTests(unittest.TestCase):
    def test_cache_construction_uses_spawn_workers_without_transferring_arrays(self):
        from pointcloud_lab.raw_cache import _build_frame
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "frames").mkdir()
            with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn"),
                                     initializer=initialize_cache_fixture, initargs=(str(root),)) as pool:
                results = list(pool.map(_build_frame, [0, 1]))
            self.assertEqual({item[0] for item in results}, {"a", "b"})
            for token, entry, size in results:
                self.assertGreater(size, 0)
                self.assertEqual(read_json(root / "frames" / entry / "metadata.json")["name"], token)
                self.assertTrue((root / "frames" / entry / "image.npy").is_file())

    def test_cache_is_exact_before_augmentation_and_each_consumer_can_mutate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "frames").mkdir()
            dataset = RawDataset()
            expected = dataset.get_data(0)
            write_frame(expected, root / "frames" / "00000000")
            write_json(root / "manifest.json", {
                "schema": 1, "status": "ready", "dataset": dataset_identity(dataset),
                "entries": {"a": "00000000"}})
            reader = CachedRawReader(dataset, root)
            first = reader(0)
            self.assertEqual(first["image"].dtype, np.uint8)
            np.testing.assert_array_equal(first["image"], expected["image"])
            first["image"][:] = 0
            first["segment"][:] = 7
            np.testing.assert_array_equal(reader(0)["image"], expected["image"])
            np.testing.assert_array_equal(reader(0)["segment"], expected["segment"])
            self.assertEqual(reader(1)["name"], "b")
            self.assertEqual((reader.hits, reader.misses), (3, 1))
            dataset.get_data = reader
            with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
                hit = pool.submit(read_cache_fixture, dataset, 0).result(timeout=30)
                miss = pool.submit(read_cache_fixture, dataset, 1).result(timeout=30)
            np.testing.assert_array_equal(hit["image"], expected["image"])
            self.assertEqual(miss["name"], "b")
            dataset.learning_map = {0: -1, 20: 3}
            with self.assertRaises(ValueError):
                CachedRawReader(dataset, root)

    def test_new_dropout_methods_control_retention_and_leave_global_rng_untouched(self):
        features = torch.zeros(2000, 6, 2, 1, 1)
        data = {"offset": torch.arange(1, 2001), "image_mask": torch.ones(2000, 6, dtype=torch.bool)}
        masks = []
        original = lambda d, *args: masks.append(d["image_mask"].clone())
        protocol = {"camera_dropout": .2, "seed": 20261101}
        state = torch.get_rng_state()
        for method in ("camera_dropout_half", "uniform_camera_dropout"):
            policy = AssignmentPolicy(SimpleNamespace(training=True), original, method, protocol)
            policy(data, features, 14)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            self.assertAlmostEqual(float(masks[-1].sum(1).float().mean()), 3, delta=.15)
        self.assertEqual(set(masks[-1].sum(1).tolist()), set(range(7)))
        self.assertTrue(data["image_mask"].all())

    def test_cgroup_limit_constrains_cache_budget_even_if_host_ram_is_large(self):
        values = {"/proc/meminfo": "MemTotal: 1048576000 kB\nMemAvailable: 838860800 kB\nCached: 100 kB",
                  "/sys/fs/cgroup/memory.max": str(32*1024**3),
                  "/sys/fs/cgroup/memory.current": str(20*1024**3)}
        with patch("pointcloud_lab.resource_monitor.read_text", side_effect=values.get):
            self.assertEqual(memory_info()["conservative_available_bytes"], 12*1024**3)

    def test_first_wave_has_five_training_jobs_and_twenty_evaluations(self):
        plan = validate_plan(read_json(ROOT / "configs/training/resource_stage1.json"))
        jobs = make_jobs(plan, Path("/run"))
        self.assertEqual(sum(j["kind"] == "train" for j in jobs), 5)
        self.assertEqual(sum(j["kind"] == "eval" for j in jobs), 20)
        self.assertFalse(any(j["phase"] == "medium" for j in jobs))

    def test_reuses_completed_ten_epoch_campaign_node_but_never_two_epoch_node(self):
        plan = validate_plan(read_json(ROOT / "configs/training/resource_stage1.json"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prior = root / "prior"
            train = prior / "jobs" / "medium_control_train"
            (train / "model").mkdir(parents=True)
            (train / "model/model_last.pth").touch()
            protocol = copy.deepcopy(plan["screen_protocol"])
            protocol.pop("profile_steps")
            write_json(train / "status.json", {"status": "completed"})
            write_json(train / "summary.json", {"optimizer_updates": 46880})
            write_json(train / "experiment.json", {"method": "control", "protocol": protocol})
            write_json(prior / "paths.json", {"info": "same"})
            write_json(prior / "tokens.json", ["a"])
            write_json(prior / "queue.json", {"jobs": [{
                "id": "medium_control_train", "kind": "train", "method": "control",
                "status": "completed", "output": str(train)}]})
            jobs = make_jobs(plan, root / "new")
            result = reuse_campaign_results(jobs, prior, plan["screen_protocol"], {"info": "same"}, ["a"])
            self.assertEqual(result, ["screen_control_train"])
            self.assertEqual(jobs[0]["output"], str(train))
            protocol["epochs"] = 2
            write_json(train / "experiment.json", {"method": "control", "protocol": protocol})
            jobs = make_jobs(plan, root / "new")
            self.assertEqual(reuse_campaign_results(jobs, prior, plan["screen_protocol"], {"info": "same"}, ["a"]), [])

    def test_export_does_not_visit_interrupted_archives_with_stale_running_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "run"
            current = source / "jobs" / "eval_a"
            archive = source / "jobs" / "eval_a.interrupted_20261102"
            current.mkdir(parents=True)
            archive.mkdir()
            write_json(source / "status.json", {"status": "completed"})
            write_json(source / "queue.json", {"jobs": [{
                "id": "eval_a", "status": "completed", "output": str(current)}]})
            write_json(current / "status.json", {"status": "completed"})
            write_json(archive / "status.json", {"status": "running"})
            export(source, root / "record")
            self.assertTrue((root / "record/jobs/eval_a/status.json").is_file())
            self.assertFalse((root / "record/jobs" / archive.name).exists())


if __name__ == "__main__":
    unittest.main()
