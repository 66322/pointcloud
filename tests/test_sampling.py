import importlib.util
from types import SimpleNamespace
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "Torch required")
class SamplingTests(unittest.TestCase):
    def test_center_selection_weighting_missing_and_mix3d_offsets(self):
        import torch
        from pointcloud_lab.sampling import assign_sampled
        features = torch.tensor([[[[[1., 2., 3.]]], [[[10., 20., 30.]]]],
                                 [[[[4., 5., 6.]]], [[[40., 50., 60.]]]]])
        coords = torch.tensor([[[0., 6.5], [20.5, 6.5]], [[20.5, 6.5], [0., 6.5]],
                               [[0., 6.5], [0., 6.5]], [[20.5, 6.5], [20.5, 6.5]]])
        data = {"offset": torch.tensor([4]), "unmix3d_offset": torch.tensor([3, 4]),
                "image_coord": coords, "image_mask": torch.tensor(
                    [[True, True], [True, False], [False, False], [True, True]])}
        chosen = assign_sampled(data, features, 14, mode="center_select")
        torch.testing.assert_close(chosen[:, 0], torch.tensor([20., 2., 0., 5.]))
        weighted = assign_sampled(data, features, 14, mode="center_weighted")
        self.assertGreater(float(weighted[0, 0]), 10.5)
        self.assertEqual(float(weighted[3, 0]), 27.5)
        reversed_data = dict(data, image_coord=coords.flip(1), image_mask=data["image_mask"].flip(1))
        torch.testing.assert_close(weighted, assign_sampled(
            reversed_data, features.flip(1), 14, mode="center_weighted"))

    def test_bilinear_token_centers_midpoint_border_and_invalid(self):
        import torch
        from pointcloud_lab.sampling import assign_sampled
        pixels = torch.tensor([-1., 6.5, 13.5, 20.5, 27.])
        data = {"offset": torch.tensor([5]), "image_coord": torch.stack(
            (pixels, torch.full_like(pixels, 6.5)), -1)[:, None],
            "image_mask": torch.ones(5, 1, dtype=torch.bool)}
        output = assign_sampled(data, torch.tensor([[[[[2., 6.]]]]]), 14, mode="bilinear")
        torch.testing.assert_close(output[:, 0], torch.tensor([0., 2., 4., 6., 6.]))

    def test_local_identity_gradients_and_checkpointed_chunks(self):
        import torch
        from pointcloud_lab.sampling import LocalTokenAttention
        torch.manual_seed(45)
        image, points = torch.randn(5, 3, 4), torch.randn(4, 4)
        uv = torch.tensor([[0., 0.], [1., 1.], [3., 2.], [2., 1.]])
        model = LocalTokenAttention(5, hidden=4, chunk_size=1)
        output = model(points, image, uv)
        nearest = image[:, uv[:, 1].long(), uv[:, 0].long()].T
        torch.testing.assert_close(output, nearest, rtol=0, atol=0)
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(model.alpha.grad))
        model.zero_grad()
        with torch.no_grad():
            model.alpha.fill_(0.4)
        output = model(points, image, uv)
        output.square().sum().backward()
        for parameter in model.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        first = output.detach()
        model.chunk_size = 8192
        torch.testing.assert_close(first, model(points, image, uv))

    def test_paired_camera_conditions_are_deterministic_no_input_mutation_and_eval_dropout_off(self):
        import torch
        from pointcloud_lab.eval_conditions import EvaluationAssignment
        from pointcloud_lab.interventions import assign_variant
        from pointcloud_lab.training_models import AssignmentPolicy
        data = {"offset": torch.tensor([1]), "image_coord": torch.zeros(1, 6, 2),
                "image_mask": torch.ones(1, 6, dtype=torch.bool)}
        features = torch.arange(1., 7.).reshape(1, 6, 1, 1, 1)
        original = lambda d, f, p, m: assign_variant(d, f, p, m, mode="last")
        policy = AssignmentPolicy(torch.nn.Linear(1, 1).eval(), original, "whole_camera_dropout",
                                  {"camera_dropout": .2, "seed": 42})
        self.assertEqual(float(EvaluationAssignment(policy, 3)(data, features, 14)), 5.)
        self.assertEqual(float(EvaluationAssignment(policy, 1)(data, features, 14)), 1.)
        self.assertEqual(float(EvaluationAssignment(policy, 0)(data, features, 14)), 0.)
        self.assertTrue(data["image_mask"].all())
        shifted = EvaluationAssignment(policy, 6, -4)(data, features, 14)
        self.assertEqual(float(shifted), 0.)
        self.assertTrue((data["image_coord"] == 0).all())
        self.assertEqual(policy.calls, 0)
        policy.backbone.train()
        policy.probability = 1.
        self.assertEqual(float(policy(data, features, 14)), 0.)
        self.assertEqual(policy.dropped, 6)


if __name__ == "__main__":
    unittest.main()
