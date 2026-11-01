"""Ordered CPU prefetch and frame-scoped reuse for unchanged DITR inference."""
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Full
from threading import Event, Thread


class OrderedRawReader:
    """Read deterministic NuScenes get_data in parallel; never run transforms here."""
    def __init__(self, get_data, frames, workers=2):
        if workers < 1:
            raise ValueError("positive raw I/O workers required")
        self.original, self.frames, self.workers = get_data, frames, workers
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="raw-io")
        self.pending = {}
        self.submitted = self.expected = 0

    def __call__(self, index):
        if index != self.expected:
            raise ValueError("Raw prefetch requires sequential, unshuffled frame order")
        self.expected += 1
        while self.submitted < min(self.frames, index + self.workers):
            key = self.submitted
            self.pending[key] = self.executor.submit(self.original, key)
            self.submitted += 1
        # Consume in the original order regardless of I/O completion order.
        return self.pending.pop(index).result()

    def close(self):
        for future in self.pending.values():
            future.cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.pending.clear()


class OrderedFramePrefetch:
    """One CPU producer keeps the original Python/NumPy transform RNG sequence.

    The base DataLoader must have num_workers=0. Its iterator is created on the
    consumer thread, preserving the main Torch RNG draw for the loader base seed.
    Upstream NuScenes transforms use Python/NumPy, not the model's Torch RNG.
    """
    def __init__(self, loader, depth=1, on_close=None):
        if depth < 0:
            raise ValueError("prefetch depth must be nonnegative")
        self.loader, self.depth, self.on_close = loader, depth, on_close

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        # This statement executes on the main consumer, before starting the producer.
        source = iter(self.loader)
        if self.depth == 0:
            try:
                yield from source
            finally:
                if self.on_close:
                    self.on_close()
            return
        queue, stopped = Queue(maxsize=self.depth), Event()

        def send(item):
            while not stopped.is_set():
                try:
                    queue.put(item, timeout=0.1)
                    return True
                except Full:
                    continue
            return False

        def produce():
            try:
                for batch in source:
                    if not send(("data", batch)):
                        return
            except BaseException as error:
                send(("error", error))
                return
            send(("end", None))

        producer = Thread(target=produce, name="frame-prefetch", daemon=True)
        try:
            # Complete the first actual frame synchronously, including lazy model
            # compilation in the consumer, before overlapping subsequent transforms.
            try:
                first = next(source)
            except StopIteration:
                return
            yield first
            producer.start()
            while True:
                kind, item = queue.get()
                if kind == "error":
                    raise item
                if kind == "end":
                    break
                yield item
        finally:
            stopped.set()
            if producer.ident is not None:
                producer.join()
            if self.on_close:
                self.on_close()


class FrameImageCache:
    """One frame only; exact image equality, eval and no-grad are required."""
    def __init__(self, original_get_image_feat):
        self.original = original_get_image_feat
        self.active_frame = None
        self.image = None
        self.output = None
        self.encoder = self.autocast = None
        self.calls = self.hits = self.computations = 0

    def begin_frame(self, token):
        # Even equal-looking images from another frame do not share entries.
        self.active_frame = token
        self.image = self.output = None
        self.encoder = self.autocast = None

    def __call__(self, encoder, data):
        import torch
        self.calls += 1
        training = getattr(encoder, "training", getattr(getattr(encoder, "vit", None), "training", True))
        if self.active_frame is None or training or torch.is_grad_enabled():
            self.image = self.output = None
            self.computations += 1
            return self.original(encoder, data)
        image = data["image"]
        autocast = (torch.is_autocast_enabled(image.device.type),
                    torch.get_autocast_dtype(image.device.type))
        if (self.image is not None
                and encoder is self.encoder and autocast == self.autocast
                and image.shape == self.image.shape
                and image.dtype == self.image.dtype
                and image.device == self.image.device
                and image.stride() == self.image.stride()
                and torch.equal(image, self.image)):
            self.hits += 1
            return self.output
        self.computations += 1
        self.output = self.original(encoder, data)
        self.image = image.detach().clone()
        self.encoder, self.autocast = encoder, autocast
        return self.output

    def stats(self):
        return {"calls": self.calls, "hits": self.hits, "computations": self.computations}


class TimedFrameLoader:
    """Same frames, same order; measure actual blocking wait and wall throughput."""
    def __init__(self, loader, cache=None, log=None):
        self.loader, self.cache, self.log = loader, cache, log
        self.wait_seconds = 0.0
        self.completed = 0
        self.wall_seconds = 0.0

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        started = time.perf_counter()
        iterator = iter(self.loader)
        while True:
            waiting = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                self.wall_seconds = time.perf_counter() - started
                return
            self.wait_seconds += time.perf_counter() - waiting
            if self.cache is not None:
                self.cache.begin_frame(batch[0]["name"])
            yield batch
            self.completed += 1
            self.wall_seconds = time.perf_counter() - started
            if self.log and (self.completed % 20 == 0 or self.completed == len(self)):
                average = self.wall_seconds / self.completed
                self.log(
                    f"Pipeline: {self.completed}/{len(self)}; wall/frame={average:.3f}s; "
                    f"data_wait/frame={self.wait_seconds / self.completed:.3f}s; "
                    f"ETA={(len(self) - self.completed) * average / 60:.1f}min"
                )

    def stats(self):
        return {"completed_frames": self.completed, "blocking_data_wait_seconds": self.wait_seconds,
                "wall_seconds": self.wall_seconds,
                "wall_seconds_per_frame": self.wall_seconds / self.completed if self.completed else None}
