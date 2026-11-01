import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "torch unavailable on authoring machine; run in server environment")
class InterventionTests(unittest.TestCase):
    def test_mean_order_invariance_first_and_missing(self):
        import torch
        from pointcloud_lab.interventions import assign_variant
        data = {
            "offset": torch.tensor([3]),
            "image_coord": torch.zeros((3, 2, 2)),
            "image_mask": torch.tensor([[True, True], [True, False], [False, False]]),
        }
        features = torch.tensor([[[[[2.0]]], [[[6.0]]]]])
        mean = assign_variant(data, features, 14, mode="mean")
        torch.testing.assert_close(mean[:, 0], torch.tensor([4.0, 2.0, 0.0]))
        torch.testing.assert_close(mean, assign_variant(data, features, 14, mode="mean", order="reverse"))
        torch.testing.assert_close(
            assign_variant(data, features, 14, mode="first"),
            assign_variant(data, features, 14, mode="last", order="reverse"))
        torch.testing.assert_close(assign_variant(data, features, 14, mode="last", dx=-1),
                                   torch.zeros((3, 1)))

    def test_multiple_batches_and_gradients(self):
        import torch
        from pointcloud_lab.interventions import assign_variant
        data = {"offset": torch.tensor([1, 2]), "image_coord": torch.zeros((2, 1, 2)),
                "image_mask": torch.ones((2, 1), dtype=torch.bool)}
        features = torch.tensor([[[[[2.0]]]], [[[[6.0]]]]], requires_grad=True)
        output = assign_variant(data, features, 14, mode="mean")
        torch.testing.assert_close(output[:, 0], torch.tensor([2.0, 6.0]))
        output.sum().backward()
        self.assertIsNotNone(features.grad)


if __name__ == "__main__":
    unittest.main()
