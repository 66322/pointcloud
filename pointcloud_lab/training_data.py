"""Sample/epoch-keyed augmentation with bounded DataLoader prefetch.

NumPy payloads cross the multiprocessing pipe without Torch shared-memory tensor
reducers. The main pin-memory thread restores the exact tensor values/dtypes.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import random

import numpy as np
import torch

from .training_protocol import sample_seed


@contextmanager
def isolated_seed(seed):
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    random.seed(seed)
    np.random.seed(seed)
    # Seed CPU only; never reseed the training CUDA generator from data loading.
    torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def seed_image_transforms(transform, seed):
    setter = getattr(transform, "set_random_seed", None)
    if setter is not None:
        setter(seed)
        return
    for name in ("transforms", "transform"):
        child = getattr(transform, name, None)
        if child is None:
            continue
        children = child if isinstance(child, (tuple, list)) else [child]
        for index, item in enumerate(children):
            seed_image_transforms(item, sample_seed(seed, index))


class PairedDataset:
    def __init__(self, dataset, epoch, seed):
        self.dataset, self.epoch, self.seed = dataset, epoch, seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        epoch = self.epoch.value
        seed = sample_seed(self.seed, epoch, index)
        with isolated_seed(seed):
            seed_image_transforms(self.dataset.transform, seed)
            data = dict(self.dataset[index])
        data["_lab_sample"] = (epoch, index)
        return data


class EpochSampler:
    def __init__(self, dataset, epoch, seed):
        self.dataset, self.epoch, self.seed = dataset, epoch, seed

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        generator = torch.Generator().manual_seed(sample_seed(self.seed, self.epoch.value, 991))
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())


@dataclass
class TensorPayload:
    array: np.ndarray


def pack(value):
    if isinstance(value, torch.Tensor):
        return TensorPayload(value.detach().cpu().numpy())
    if isinstance(value, dict):
        return {key: pack(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(pack(item) for item in value)
    if isinstance(value, list):
        return [pack(item) for item in value]
    return value


def unpack(value, pin=False):
    if isinstance(value, TensorPayload):
        tensor = torch.from_numpy(value.array)
        return tensor.pin_memory() if pin else tensor
    if isinstance(value, dict):
        return {key: unpack(item, pin) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(unpack(item, pin) for item in value)
    if isinstance(value, list):
        return [unpack(item, pin) for item in value]
    return value


class PipeBatch:
    def __init__(self, data):
        self.payload, self.data = pack(data), None

    def pin_memory(self):
        if self.data is None:
            self.data = unpack(self.payload, pin=True)
            self.payload = None
        return self

    def tensors(self):
        if self.data is None:
            self.data = unpack(self.payload)
            self.payload = None
        return self.data


class PairedCollator:
    def __init__(self, original, mix_probability, seed):
        self.original, self.mix_probability, self.seed = original, mix_probability, seed

    def __call__(self, batch):
        identities = [item.pop("_lab_sample") for item in batch]
        epoch = identities[0][0]
        if any(item[0] != epoch for item in identities):
            raise ValueError("Mixed epochs in training batch")
        with isolated_seed(sample_seed(self.seed, epoch, *(item[1] for item in identities), 313)):
            result = self.original(batch, mix_prob=self.mix_probability)
        return PipeBatch(result)


def initialize_worker(worker_id):
    torch.set_num_threads(1)
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass
