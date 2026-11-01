import importlib.util
import json
import multiprocessing as mp
import pickle
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pointcloud_lab.common import read_json, write_json
from pointcloud_lab.training_protocol import sample_seed, validate_protocol
from pointcloud_lab.run_training import main as queue_main

HAS_TORCH = importlib.util.find_spec("torch") is not None


class ToyDataset:
    transform = SimpleNamespace()

    def __len__(self):
        return 8

    def __getitem__(self, index):
        import torch
        return {"value": torch.tensor([index, random.random(), np.random.rand(), float(torch.rand(()))])}


def toy_collate(batch, mix_prob=0):
    import torch
    return {"value": torch.stack([row["value"] for row in batch]),
            "mix": torch.tensor([random.random() < mix_prob])}


class ProtocolTests(unittest.TestCase):
    def test_budget_validation_and_seed_keys(self):
        protocol = json.loads((Path(__file__).resolve().parents[1] / "configs/training/stage1.json").read_text())
        validate_protocol(protocol)
        self.assertEqual(protocol["epochs"], 2)
        self.assertNotEqual(sample_seed(42, 0, 1), sample_seed(42, 1, 1))
        self.assertNotEqual(sample_seed(42, 0, 1), sample_seed(42, 0, 2))
        with self.assertRaises(ValueError):
            validate_protocol(dict(protocol, batch_size=0))

    def test_queue_runs_training_then_validation_on_only_two_gpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = root / "val.pkl"
            with index.open("wb") as stream:
                pickle.dump([{"lidar_token": "a"}, {"lidar_token": "b"}], stream)
            paths = root / "paths.json"
            write_json(paths, {"info": str(index)})
            protocol = Path(__file__).resolve().parents[1] / "configs/training/stage1.json"
            calls = []

            def launch(command, **kwargs):
                output = Path(command[command.index("--out") + 1])
                output.mkdir()
                gpu = command[command.index("--gpu") + 1]
                self.assertIn(gpu, ("0", "1"))
                calls.append(command)
                if "pointcloud_lab.train" in command:
                    self.assertEqual(command[command.index("--protocol") + 1], str(root / "run/protocol.json"))
                    method = command[command.index("--method") + 1]
                    write_json(output / "summary.json", {"optimizer_updates": 10, "amp_skipped_steps": 0})
                else:
                    method = Path(command[command.index("--trained-run") + 1]).parent.name
                    self.assertEqual(command[command.index("--expected-frames") + 1], "2")
                    write_json(output / "summary.json",
                               {"miou_pct": 83.0 if method == "control" else 83.2,
                                "macc_pct": 90.0, "oa_pct": 94.0})
                return SimpleNamespace(pid=123, poll=lambda: 0, returncode=0)

            argv = ["queue", "--paths", str(paths), "--protocol", str(protocol), "--out", str(root / "run")]
            with patch("sys.argv", argv), patch(
                "pointcloud_lab.run_training.git_state", return_value={"sha": "test"}
            ), patch("pointcloud_lab.run_training.subprocess.Popen", side_effect=launch), patch(
                "pointcloud_lab.run_training.time.sleep"
            ):
                queue_main()
            self.assertEqual(len(calls), 8)
            result = read_json(root / "run/summary.json")
            self.assertEqual(result["results"][0]["method"], "control")
            self.assertEqual(result["results"][1]["delta_vs_trained_control_pp"], 0.2)
            self.assertEqual(read_json(root / "run/status.json")["status"], "completed")


@unittest.skipUnless(HAS_TORCH, "Torch required for training-core tests")
class TrainingCoreTests(unittest.TestCase):
    def test_parallel_loader_keeps_samples_augmentations_and_mix_decisions(self):
        import torch
        from pointcloud_lab.training_data import PairedDataset, EpochSampler, PairedCollator, initialize_worker
        ctx = mp.get_context("spawn")

        def collect(workers, epoch_number):
            epoch = ctx.Value("q", epoch_number)
            dataset = PairedDataset(ToyDataset(), epoch, 42)
            options = dict(dataset=dataset, sampler=EpochSampler(dataset, epoch, 42),
                           batch_size=2, num_workers=workers,
                           collate_fn=PairedCollator(toy_collate, 0.8, 42),
                           generator=torch.Generator().manual_seed(42))
            if workers:
                options.update(multiprocessing_context=ctx, prefetch_factor=1, worker_init_fn=initialize_worker)
            return [item.tensors() for item in torch.utils.data.DataLoader(**options)]

        serial, parallel, next_epoch = collect(0, 0), collect(2, 0), collect(0, 1)
        for old, fast in zip(serial, parallel):
            torch.testing.assert_close(old["value"], fast["value"], rtol=0, atol=0)
            self.assertTrue(torch.equal(old["mix"], fast["mix"]))
        self.assertFalse(torch.equal(serial[0]["value"], next_epoch[0]["value"]))

    def test_numpy_transport_preserves_nested_values_and_tensor_dtypes(self):
        import torch
        from pointcloud_lab.training_data import PipeBatch
        original = {"image": torch.randn(2, 3, 4), "offset": torch.tensor([2, 8], dtype=torch.int32),
                    "mask": [torch.tensor([True, False])], "name": ["a", "b"]}
        restored = pickle.loads(pickle.dumps(PipeBatch(original))).tensors()
        for key in ("image", "offset"):
            self.assertTrue(torch.equal(original[key], restored[key]))
            self.assertEqual(original[key].dtype, restored[key].dtype)
        self.assertTrue(torch.equal(original["mask"][0], restored["mask"][0]))
        self.assertEqual(original["name"], restored["name"])

    def test_seed_context_does_not_advance_the_model_cpu_rng(self):
        import torch
        from pointcloud_lab.training_data import isolated_seed
        before = torch.get_rng_state()
        with isolated_seed(3):
            torch.rand(15)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_identity_gates_strict_checkpoint_and_parent_train_freeze(self):
        import torch
        from torch import nn
        from pointcloud_lab.training_models import install_treatment, load_full_state

        class Point(dict):
            def __getattr__(self, key):
                return self[key]
            def __setattr__(self, key, value):
                self[key] = value

        class Projection(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 3)
            def forward(self, point):
                point.feat = self.linear(point.feat)
                return point

        class Up(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj, self.proj_skip = Projection(), Projection()
                self.proj_image_feat = nn.Linear(5, 3)
                self.traceable = True
            def forward(self, point):
                parent, inverse = point.pop("pooling_parent"), point.pop("pooling_inverse")
                point, parent = self.proj(point), self.proj_skip(parent)
                parent.feat = parent.feat + point.feat[inverse] + self.proj_image_feat(parent.image_feat)
                parent["unpooling_parent"] = point
                return parent

        def make():
            model = nn.Module()
            model.backbone = nn.Module()
            model.backbone.img_enc = nn.Sequential(nn.Linear(2, 2), nn.Dropout())
            model.backbone.stages = nn.ModuleList([Up() for _ in range(4)])
            return model

        original, gated = make(), make()
        gated.load_state_dict(original.state_dict(), strict=True)
        image_module = SimpleNamespace(SerializedUnpooling=Up, assign_image_feat=lambda *args: None)
        rng = torch.get_rng_state()
        install_treatment(gated, image_module, "gate", {"seed": 42, "gate_hidden": 4, "camera_dropout": 0.2})
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        coarse, skip, image = torch.randn(2, 3), torch.randn(3, 3), torch.randn(3, 5)
        def inputs():
            return Point(feat=coarse.clone(), pooling_inverse=torch.tensor([0, 0, 1]),
                         pooling_parent=Point(feat=skip.clone(), image_feat=image.clone()))
        expected = original.backbone.stages[0](inputs()).feat
        actual = gated.backbone.stages[0](inputs()).feat
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().sum().backward()
        self.assertGreater(gated.backbone.stages[0].lab_gate.net[-1].weight.grad.abs().sum(), 0)
        gated.train()
        self.assertFalse(gated.backbone.img_enc.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in gated.backbone.img_enc.parameters()))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pth"
            torch.save({"state_dict": {"module." + key: value for key, value in gated.state_dict().items()}}, path)
            load_full_state(gated, path)
            with self.assertRaises(RuntimeError):
                load_full_state(original, path)

    def test_camera_dropout_is_private_rng_supports_mix3d_and_disables_at_eval(self):
        import torch
        from pointcloud_lab.interventions import assign_variant
        from pointcloud_lab.training_models import AssignmentPolicy
        backbone = torch.nn.Linear(1, 1).train()
        original = lambda data, features, patch, missing: assign_variant(
            data, features, patch, missing, mode="last")
        policy = AssignmentPolicy(backbone, original, "camera_dropout", {"camera_dropout": 0.2, "seed": 42})
        policy.probability = 1.0
        data = {"offset": torch.tensor([2]), "unmix3d_offset": torch.tensor([1, 2]),
                "image_mask": torch.ones(2, 2, dtype=torch.bool), "image_coord": torch.zeros(2, 2, 2)}
        features = torch.tensor([[[[[2.]]], [[[6.]]]], [[[[3.]]], [[[7.]]]]])
        rng = torch.get_rng_state()
        self.assertTrue(torch.equal(policy(data, features, 14), torch.zeros(2, 1)))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(data["image_mask"].all())
        backbone.eval()
        torch.testing.assert_close(policy(data, features, 14)[:, 0], torch.tensor([6., 7.]))


if __name__ == "__main__":
    unittest.main()
