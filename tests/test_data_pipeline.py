import importlib.util
import unittest
from types import SimpleNamespace

from pointcloud_lab.data_pipeline import (
    FrameImageCache, TimedFrameLoader, OrderedRawReader, OrderedFramePrefetch,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None


class PipelineTests(unittest.TestCase):
    def test_prefetch_preserves_random_sampling_and_tta_sequence(self):
        import random
        import numpy as np

        class Dataset:
            def __init__(self):
                self.get_data = lambda index: np.arange(8) + index

            def __len__(self):
                return 5

            def __iter__(self):
                for index in range(len(self)):
                    raw = self.get_data(index)
                    # The real nuScenes test transform uses stochastic GridSample.
                    yield raw[np.random.randint(0, 8)] * random.choice((1, 2))

        random.seed(123)
        np.random.seed(123)
        def collect(stream):
            values = []
            for index, value in enumerate(stream):
                values.append(value)
                if index == 0:
                    # Lazy model initialization must finish before prefetch starts.
                    np.random.randint(0, 10, 20)
                    random.random()
            return values
        serial = collect(Dataset())
        random.seed(123)
        np.random.seed(123)
        dataset = Dataset()
        raw = OrderedRawReader(dataset.get_data, len(dataset), 2)
        dataset.get_data = raw
        fast = collect(OrderedFramePrefetch(dataset, 1, raw.close))
        self.assertEqual(serial, fast)

    def test_prefetch_propagates_loader_errors_and_closes_reader(self):
        closed = []
        def fail():
            yield 1
            raise ValueError("bad source frame")
        loader = OrderedFramePrefetch(fail(), on_close=lambda: closed.append(True))
        with self.assertRaisesRegex(ValueError, "bad source frame"):
            list(loader)
        self.assertEqual(closed, [True])
        self.assertEqual(list(OrderedFramePrefetch([1, 2], depth=0)), [1, 2])

    def test_frame_boundary_precedes_model_work_and_order_is_unchanged(self):
        frames = [[{"name": "b"}], [{"name": "a"}]]
        boundaries, messages = [], []
        cache = SimpleNamespace(begin_frame=boundaries.append)
        loader = TimedFrameLoader(frames, cache, messages.append)
        seen = []
        for batch in loader:
            self.assertEqual(boundaries[-1], batch[0]["name"])
            seen.append(batch[0]["name"])
        self.assertEqual(seen, ["b", "a"])
        self.assertEqual(boundaries, ["b", "a"])
        self.assertEqual(loader.stats()["completed_frames"], 2)
        self.assertEqual(len(messages), 1)
        self.assertGreaterEqual(loader.wall_seconds, loader.wait_seconds)


@unittest.skipUnless(HAS_TORCH, "torch unavailable on authoring machine")
class ImageCacheTests(unittest.TestCase):
    def test_reuse_matches_uncached_and_changed_images_frames_encoders_invalidate(self):
        import torch
        encoder = torch.nn.Linear(3, 2).eval().requires_grad_(False)
        calls = []

        def original(model, data):
            calls.append(1)
            return model(data["image"])

        cache = FrameImageCache(original)
        image = torch.tensor([[1., 2., 3.]])
        cache.begin_frame("a")
        with torch.no_grad():
            reference = original(encoder, {"image": image})
            actual = cache(encoder, {"image": image})
            repeated = cache(encoder, {"image": image.clone()})
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            torch.testing.assert_close(repeated, reference, rtol=0, atol=0)
            self.assertEqual(cache.stats()["hits"], 1)
            # Same storage mutated in place must invalidate the snapshot.
            image.add_(1)
            changed = cache(encoder, {"image": image})
            torch.testing.assert_close(changed, original(encoder, {"image": image}), rtol=0, atol=0)
            cache.begin_frame("b")
            cache(encoder, {"image": image})
            other = torch.nn.Linear(3, 2).eval().requires_grad_(False)
            cache(other, {"image": image})
        self.assertEqual(cache.stats()["computations"], 4)
        self.assertEqual(cache.stats()["hits"], 1)

    def test_training_and_gradient_paths_never_reuse_or_retain_stale_features(self):
        import torch
        encoder = torch.nn.Linear(3, 2).eval()
        cache = FrameImageCache(lambda model, data: model(data["image"]))
        cache.begin_frame("a")
        image = torch.ones(1, 3)
        with torch.no_grad():
            cache(encoder, {"image": image})
        x = image.clone().requires_grad_(True)
        output = cache(encoder, {"image": x})
        output.sum().backward()
        self.assertIsNotNone(x.grad)
        encoder.train()
        with torch.no_grad():
            cache(encoder, {"image": image})
            encoder.eval()
            cache(encoder, {"image": image})
        self.assertEqual(cache.stats()["hits"], 0)

    def test_autocast_change_invalidates_cached_output(self):
        import torch
        encoder = torch.nn.Linear(3, 2).eval().requires_grad_(False)
        cache = FrameImageCache(lambda model, data: model(data["image"]))
        cache.begin_frame("a")
        data = {"image": torch.ones(1, 3)}
        with torch.no_grad():
            reference = cache(encoder, data)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                actual = cache(encoder, data)
                self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertEqual(reference.dtype, torch.float32)
        self.assertEqual(cache.stats()["hits"], 0)


if __name__ == "__main__":
    unittest.main()
