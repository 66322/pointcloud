import csv
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from pointcloud_lab.common import read_json, write_json
from pointcloud_lab.export_record import export
from pointcloud_lab.metrics import confusion
from pointcloud_lab.proposal_data import (
    RING, camera_geometry, configure_fields, retained_indices, route_groups,
)
from pointcloud_lab.proposal_metrics import describe, evaluate_predictions
from pointcloud_lab.proposal_model import ProposalFusion
from pointcloud_lab.proposal_model import encode_views
from pointcloud_lab.proposal_record import snapshot
from pointcloud_lab.proposal_queue import DEFAULT_PLAN, command, report, schedule
from pointcloud_lab.semanticstf import COMMON, heading_alignment
from pointcloud_lab.training_models import install_treatment
from pointcloud_lab.training_protocol import PROPOSAL_METHODS

ROOT = Path(__file__).resolve().parents[1]


def tensor_batch():
    torch.manual_seed(83)
    n, v = 12, 6
    mask = torch.zeros(n, v, dtype=torch.bool)
    mask[:, 0] = True
    mask[:4, 1] = True
    mask[8:, 0] = False
    pose = torch.zeros(1, v, v, 4)
    pose[0, 0, 1] = torch.tensor([1., .8, .6, 1.2])
    uv = torch.rand(n, v, 2) * 12
    uv[1, 0] = torch.tensor([15.99, 15.99])
    return dict(coord=torch.rand(n, 3), grid_coord=torch.arange(n)[:, None].expand(n, 3),
                feat=torch.randn(n, 4), offset=torch.tensor([n]), image=torch.randn(1, v, 3, 16, 16),
                image_coord=uv, image_mask=mask, lab_primary=torch.zeros(n, dtype=torch.long),
                lab_keep=torch.ones(1, v, dtype=torch.bool), lab_pose=pose,
                lab_depth_ok=torch.tensor([True]), lab_depth=torch.rand(n, v) * 12 + 3,
                segment=torch.arange(n) % 3)


class ProposalInputs(unittest.TestCase):
    def test_matched_sparse_support_and_missing_strata_in_original_order(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            n = 90
            points = np.tile(np.array([30., 0, 40, 1, 0], dtype="<f4"), (n, 1))
            points.tofile(raw / "points.bin")
            ls = np.resize(np.array([17, 24, 20], dtype=np.uint8), n)
            ls.tofile(raw / "labels.bin")
            Image.new("RGB", (200, 100)).save(raw / "im.jpg")
            cams = {name: dict(data_path="im.jpg", sensor2lidar_rotation=np.eye(3),
                               sensor2lidar_translation=[0., 0., 0.],
                               camera_intrinsics=[[50, 0, 100], [0, 50, 50], [0, 0, 1]]) for name in RING}
            info = [{"lidar_token": "f", "lidar_path": "points.bin", "gt_segment_path": "labels.bin",
                     "cams": cams}]
            (root / "infos.pkl").write_bytes(pickle.dumps(info))
            paths = dict(data_root=str(root), info=str(root / "infos.pkl"))
            pred = np.resize(np.array([3, 10, 0]), n)
            np.save(root / "f_pred.npy", pred)
            clean = evaluate_predictions(paths, ["f"], root, "clean")
            none = evaluate_predictions(paths, ["f"], root, "none")
            self.assertEqual(none["groups"]["fallback"]["labelled_points"], 60)
            keep = retained_indices(points[:, :3], "f", "sparse")
            np.save(root / "f_pred.npy", pred[keep])
            sparse = evaluate_predictions(paths, ["f"], root, "sparse")
            self.assertEqual(clean["groups"]["sparse_support"]["per_class"], sparse["per_class"])

    def test_external_alignment_matches_camera_heading_without_changing_height(self):
        src = {"sensor2lidar_rotation": np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])}
        stf = {"sensor2lidar_rotation": np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])}
        np.testing.assert_allclose(heading_alignment(stf, src) @ [1, 0, 3], [0, 1, 3], atol=1e-6)

    def test_reprojection_and_missing_names_are_independent_of_array_order(self):
        names = tuple(reversed(RING))
        cameras = {n: dict(sensor2lidar_rotation=np.eye(3), sensor2lidar_translation=np.zeros(3),
                           camera_intrinsics=[[100, 0, 50], [0, 100, 40], [0, 0, 1]]) for n in names}
        xyz = np.array([[0., 0., 10], [1., 0., -10]])
        for condition, expected in [("clean", 50), ("tx_020", 52),
                                    ("yaw_m3", 50 - 100 * np.tan(np.deg2rad(3)))]:
            g = camera_geometry(xyz, cameras, (80, 100), condition)
            self.assertAlmostEqual(g["image_coord"][0, 0, 0], expected, places=4)
            self.assertFalse(g["image_mask"][1].any())
        subset = camera_geometry(xyz, cameras, (80, 100), "triple_a")
        self.assertEqual({n for n, keep in zip(names, subset["lab_keep"][0]) if keep}, set(RING[::2]))
        masks = route_groups(subset)
        self.assertTrue(masks["neighbor_only"][0])
        self.assertTrue(masks["fallback"][1])

    def test_sparse_joint_has_identical_targets_and_transform_keys_survive(self):
        xyz = np.tile([35., 1., .5], (200, 1))
        np.testing.assert_array_equal(retained_indices(xyz, "frame", "sparse"),
                                      retained_indices(xyz, "frame", "dark_sparse"))
        self.assertGreater(len(retained_indices(xyz, "frame", "sparse")), 0)
        self.assertLess(len(retained_indices(xyz, "frame", "sparse")), len(xyz))
        split = {"test_mode": True, "transform": [{"type": "GridSample", "keys": ("coord",)}],
                 "test_cfg": {"voxelize": {"type": "GridSample", "keys": ("coord",)},
                              "post_transform": [{"type": "Collect", "keys": ("image",)}]}}
        configure_fields(split)
        for grid in [split["transform"][0], split["test_cfg"]["voxelize"]]:
            self.assertIn("lab_depth", grid["keys"])
            self.assertNotIn("lab_pose", grid["keys"])
        self.assertIn("lab_depth_ok", split["test_cfg"]["post_transform"][0]["keys"])

    def test_nine_class_transfer_counts_out_of_scope_predictions_as_errors(self):
        label = np.array([3, 3, -1, 1])
        pred = np.array([3, 0, 0, 1])
        row = describe(confusion(pred, label), sorted(COMMON.values()))
        self.assertEqual(row["labelled_points"], 3)
        self.assertEqual(row["per_class"]["car"]["iou_pct"], 50)
        self.assertAlmostEqual(row["oa_pct"], 200 / 3)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "val/labels").mkdir(parents=True)
            (root / "val/val.txt").write_text("frame,rain\n", encoding="utf-8")
            np.array([1, 1, 20, 2], dtype="<u4").tofile(root / "val/labels/frame.label")
            np.save(root / "frame_pred.npy", pred)
            score = evaluate_predictions({"semanticstf_root": str(root)}, ["frame"], root, "stf_val")
            self.assertEqual(score["groups"]["rain"]["frames"], 1)
            self.assertEqual(score["per_class"], row["per_class"])


class FusionChecks(unittest.TestCase):
    def test_only_available_views_reach_student_encoder_and_control_assignment(self):
        data = tensor_batch()
        data["lab_keep"].zero_()
        data["lab_keep"][0, 1] = True
        encoder = nn.Conv2d(3, 6, 4, stride=4)
        encoder.output_channels, encoder.patch_size = 6, 4
        seen = []
        def extract(enc, batch):
            im = batch["image"]
            seen.append(im.shape[1])
            return enc(im.flatten(0, 1)).reshape(*im.shape[:2], 6, 4, 4), None, 4
        maps, _ = encode_views(encoder, extract, data, full_teacher=False)
        self.assertEqual(seen, [1])
        self.assertEqual(maps[:, 0].count_nonzero(), 0)
        encode_views(encoder, extract, data, full_teacher=True)
        data["lab_keep"].zero_()
        zero, _ = encode_views(encoder, extract, data, full_teacher=False)
        self.assertEqual(seen, [1, 6])
        self.assertEqual(zero.count_nonzero(), 0)
        model = nn.Module()
        model.backbone = nn.Module()
        model.backbone.img_enc = encoder
        module = SimpleNamespace(assign_image_feat=lambda d, *args: d["image_mask"])
        install_treatment(model, module, "control", dict(proposal_inputs=True, seed=2, camera_dropout=.2))
        self.assertFalse(module.assign_image_feat(data, maps, 4).any())
        self.assertTrue(data["image_mask"].any())

    def test_candidate_gradients_depth_supervision_and_tied_gates(self):
        data = tensor_batch()
        maps = torch.randn(1, 6, 6, 4, 4)
        for method in PROPOSAL_METHODS[2:]:
            with self.subTest(method=method):
                head = ProposalFusion(method, 6, 8, 5)
                p = torch.randn(12, 8, requires_grad=True)
                out, mask, depth = head(p, data, maps, 4)
                loss = out.square().mean()
                (loss if depth is None else loss + depth).backward()
                self.assertTrue(torch.isfinite(p.grad).all())
                self.assertTrue(all(torch.isfinite(t.grad).all() for t in head.parameters() if t.grad is not None))
                self.assertTrue(mask[8:].all())
                self.assertIsNotNone(head.inject.weight.grad)
                if method == "p_asym":
                    self.assertGreater(head.to_image.weight.grad.abs().sum().item(), 0)
                    self.assertGreater(head.offset.weight.grad.abs().sum().item(), 0)
                    self.assertGreater(head.image_depth.weight.grad.abs().sum().item(), 0)
                    self.assertGreater(head.null.weight.grad.abs().sum().item(), 0)
                    shifted = dict(data, lab_depth_ok=torch.tensor([False]))
                    self.assertIsNone(head(p, shifted, maps, 4)[2])
                if method == "p_shared":
                    head.eval()
                    with torch.no_grad():
                        head(p, data, maps, 4)
                    log = head.diagnostics()
                    self.assertAlmostEqual(log["mean_image_to_point"], log["mean_point_to_image"])

    def test_missing_view_invariance_eligible_neighbors_and_zero_observation(self):
        data = tensor_batch()
        data["lab_keep"][0, 0] = False
        maps = torch.randn(1, 6, 6, 4, 4)
        p = torch.randn(12, 8)
        head = ProposalFusion("p_distill", 6, 8, 5).eval()
        with torch.no_grad():
            first, fallback, _ = head(p, data, maps, 4)
            self.assertEqual(fallback.tolist(), [False] * 4 + [True] * 8)
            maps[:, 0].fill_(99)
            data["image"][:, 0].fill_(-90)
            second, _, _ = head(p, data, maps, 4)
            torch.testing.assert_close(first, second)
            data["lab_keep"].zero_()
            third, fallback, _ = head(p, data, maps, 4)
            self.assertTrue(fallback.all())
            torch.testing.assert_close(third[4:], second[4:])
            with torch.autocast("cpu", dtype=torch.bfloat16):
                self.assertTrue(torch.isfinite(head(p, data, maps, 4)[0]).all())

    def test_mix3d_keeps_calibration_with_source_frame(self):
        data = tensor_batch()
        combined = {k: torch.cat([v, v]) for k, v in data.items() if k != "offset"}
        combined["offset"] = torch.tensor([24])
        combined["unmix3d_offset"] = torch.tensor([12, 24])
        combined["lab_keep"][1].zero_()
        maps = torch.randn(2, 6, 6, 4, 4)
        head = ProposalFusion("p_reject", 6, 8, 5).eval()
        _, fallback, _ = head(torch.randn(24, 8), combined, maps, 4)
        self.assertEqual(fallback.tolist(), [False] * 8 + [True] * 16)

    def test_distillation_through_training_entry_and_strict_reload(self):
        class Point(dict):
            def __getattr__(self, key): return self[key]
            def __setattr__(self, key, value): self[key] = value
            def serialization(self, **kwargs): pass
            def sparsify(self): pass

        class Unpool(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj, self.proj_skip = nn.Identity(), nn.Identity()
                self.proj_image_feat = nn.Linear(6, 8)
                self.traceable = False

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Linear(4, 8)

            def forward(self, p):
                p.feat = self.embed(p.feat)
                for _ in range(4):
                    p = Point(feat=p.feat, pooling_parent=p, pooling_inverse=torch.arange(len(p.feat)))
                return p

        def model():
            root = nn.Module()
            root.backbone = b = nn.Module()
            b.img_enc = nn.Conv2d(3, 6, 4, stride=4)
            b.img_enc.output_channels = 6
            b.img_enc.patch_size = 4
            b.order, b.shuffle_orders = ["z"], False
            b.embedding, b.enc = nn.Identity(), Encoder()
            b.dec = nn.Sequential(*(Unpool() for _ in range(4)))
            root.seg_head, root.criteria = nn.Linear(8, 3), nn.CrossEntropyLoss()
            module = SimpleNamespace(Point=Point, SerializedUnpooling=Unpool, assign_image_feat=lambda *a: None,
                                     get_image_feat=lambda enc, d: (enc(d["image"].flatten(0, 1)).reshape(
                                         -1, 6, 6, 4, 4), None, 4))
            install_treatment(root, module, "p_distill",
                              dict(proposal_inputs=True, seed=1, gate_hidden=5, camera_dropout=.2))
            return root

        root = model()
        data = tensor_batch()
        data["lab_keep"].zero_()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = root.train()(data)["loss"]
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(root.lab_fusion.prior[-1].weight.grad.abs().sum().item(), 0)
        self.assertIsNone(root.lab_fusion.inject.weight.grad)  # teacher is detached
        self.assertFalse(root.backbone.img_enc.training)
        clone = model()
        clone.load_state_dict(root.state_dict(), strict=True)
        self.assertTrue(torch.equal(root.eval()(data)["seg_logits"], clone.eval()(data)["seg_logits"]))


class ProposalQueueChecks(unittest.TestCase):
    def test_campaign_entry_snapshots_jobs_uses_two_cards_and_resumes_once(self):
        from pointcloud_lab.campaign import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "run"
            plan = read_json(ROOT / DEFAULT_PLAN)
            plan["protocol"]["methods"] = ["control", "p_center"]
            plan.update(seeds=[20261101], groups={}, contrasts=[], external_methods=[])
            write_json(root / "plan.json", plan)
            (root / "info.pkl").write_bytes(pickle.dumps([dict(lidar_token=str(i)) for i in range(6019)]))
            write_json(root / "paths.json", {"info": str(root / "info.pkl")})
            calls = []
            def launch(argv, cwd, **kwargs):
                self.assertEqual(Path(cwd), out / "code")
                self.assertTrue((Path(cwd) / "pointcloud_lab/proposal_model.py").exists())
                gpu = argv[argv.index("--gpu") + 1]
                calls.append((argv[3], gpu))
                dest = Path(argv[argv.index("--out") + 1])
                dest.mkdir()
                train = "--method" in argv
                if train:
                    self.assertEqual(read_json(argv[argv.index("--protocol") + 1])["seed"], 20261101)
                else:
                    self.assertEqual(argv[argv.index("--proposal-condition") + 1], "clean")
                    self.assertEqual(len(read_json(argv[argv.index("--tokens") + 1])), 6019)
                write_json(dest / "status.json", {"status": "completed"})
                write_json(dest / "summary.json", {"optimizer_updates": 1} if train else
                           {"miou_pct": 50., "macc_pct": 50., "oa_pct": 50.})
                return SimpleNamespace(pid=100 + len(calls), poll=lambda: 0)
            original = ["campaign", "--out", str(out), "--plan", str(root / "plan.json"),
                        "--paths", str(root / "paths.json")]
            with patch("pointcloud_lab.campaign.subprocess.Popen", side_effect=launch), patch(
                "pointcloud_lab.campaign.gpu_idle", return_value=True), patch(
                "pointcloud_lab.campaign.time.sleep"), patch(
                "pointcloud_lab.proposal_queue.git_state", return_value={"sha": "execution"}):
                with patch.object(sys, "argv", original):
                    main()
                with patch.object(sys, "argv", ["campaign", "--out", str(out), "--resume"]):
                    main()
            self.assertEqual(calls, [("pointcloud_lab.train", "0"), ("pointcloud_lab.train", "1"),
                                     ("pointcloud_lab.evaluate", "0"), ("pointcloud_lab.evaluate", "1")])
            self.assertEqual(read_json(out / "status.json")["status"], "completed")

    def test_registered_schedule_and_external_commands(self):
        plan = read_json(ROOT / DEFAULT_PLAN)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            protocols, jobs = schedule(plan, out, with_stf=True)
            self.assertEqual((len(protocols), sum(j["kind"] == "train" for j in jobs), len(jobs)), (3, 42, 423))
            self.assertEqual(len({j["id"] for j in jobs}), 423)
            indexed = {j["id"]: j for j in jobs}
            for job in jobs:
                job["gpu"] = "1"
                args = command(job, out, indexed, 6019)
                self.assertEqual(args[args.index("--gpu") + 1], "1")
                if job.get("condition") == "stf_val":
                    self.assertIn(str(out / "stf_tokens.json"), args)
                    self.assertIn("250", args)
                if job["kind"] == "eval":
                    self.assertEqual(indexed[job["train_job"]]["seed"], job["seed"])
            with self.assertRaises(ValueError):
                schedule(plan, out, [3])

    def test_incomplete_pairs_statistics_and_compact_export(self):
        plan = read_json(ROOT / DEFAULT_PLAN)
        seeds = plan["seeds"][:2]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "batch"
            out.mkdir()
            _, jobs = schedule(plan, out, seeds)
            write_json(out / "plan.json", plan)
            write_json(out / "selection.json", dict(seeds=seeds, with_stf=False))
            write_json(out / "provenance.json", {"source_git": {"sha": "execution"}})
            for job in jobs:
                job["status"] = "blocked"
                if (job["kind"] == "eval" and job["method"] in {"p_reject", "p_imgdepth"}
                        and job.get("condition") in {"yaw_m3", "yaw_p3"}):
                    if job["method"] == "p_reject" and job["condition"] == "yaw_m3" and job["seed"] == seeds[1]:
                        continue
                    job.update(status="completed", summary={"miou_pct": 72. if job["method"] == "p_reject" else 70.,
                                                              "oa_pct": 85., "macc_pct": 78.})
                    run = Path(job["output"])
                    run.mkdir(parents=True)
                    write_json(run / "status.json", {"status": "completed"})
                    write_json(run / "summary.json", job["summary"])
                    write_json(run / "provenance.json", {"upstream_git": {"sha": "upstream"}})
                    (run / "model.pth").write_bytes(b"never publish")
            report(out, jobs)
            with (out / "contrasts.csv").open() as f:
                found = next(row for row in csv.DictReader(f) if row["contrast"] == "rejection")
            self.assertEqual((found["paired_seeds"], found["mean_delta_pp"], found["sample_sd_pp"]), ("1", "2.0", ""))
            write_json(out / "queue.json", {"jobs": jobs, "updated_at": "observed"})
            receipt = Path(tmp) / "receipts/one.json"
            snapshot(out, receipt)
            self.assertEqual(read_json(receipt)["queue_updated_at"], "observed")
            with self.assertRaises(FileExistsError):
                snapshot(out, receipt)
            write_json(out / "status.json", {"status": "completed_with_failures"})
            target = Path(tmp) / "record"
            export(out, target)
            self.assertEqual(len(list((target / "jobs").iterdir())), len(jobs))
            self.assertFalse(list(target.rglob("*.pth")))
