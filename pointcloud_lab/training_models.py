"""Registered-DINO freezing, identity-initialized fusion gates and camera dropout."""
from types import MethodType

import torch
from torch import nn

from .interventions import assign_variant


class FusionGate(nn.Module):
    def __init__(self, channels, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(channels * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, point_feature, image_feature):
        return 1 + torch.tanh(self.net(torch.cat((point_feature, image_feature), dim=-1)))


def gated_unpool_forward(self, point):
    parent = point.pop("pooling_parent")
    inverse = point.pop("pooling_inverse")
    point = self.proj(point)
    parent = self.proj_skip(parent)
    image_feature = self.proj_image_feat(parent.image_feat)
    base = parent.feat + point.feat[inverse]
    parent.feat = base + self.lab_gate(base, image_feature) * image_feature
    if self.traceable:
        parent["unpooling_parent"] = point
    return parent


def frozen_train(self, mode=True):
    # Keep the original registered module and checkpoint names.
    return nn.Module.train(self, False)


def freeze_dino(backbone):
    encoder = backbone.img_enc
    encoder.requires_grad_(False)
    encoder.eval()
    encoder.train = MethodType(frozen_train, encoder)


class AssignmentPolicy:
    def __init__(self, backbone, original, method, protocol):
        self.backbone, self.original, self.method = backbone, original, method
        self.probability = protocol["camera_dropout"]
        self.generator = torch.Generator(device="cpu").manual_seed(protocol["seed"] + 73)
        self.calls = self.dropped = 0

    def __call__(self, data, features, patch, missing=None):
        if self.method == "mean":
            return assign_variant(data, features, patch, missing, mode="mean")
        if self.method != "camera_dropout" or not self.backbone.training:
            return self.original(data, features, patch, missing)
        keep = torch.rand(features.shape[:2], generator=self.generator) >= self.probability
        self.calls += 1
        self.dropped += int((~keep).sum())
        offsets = data.get("unmix3d_offset", data["offset"]).long()
        lengths = torch.diff(torch.cat((offsets.new_zeros(1), offsets)))
        if len(lengths) != features.shape[0]:
            raise ValueError("Camera dropout/image batch offsets disagree")
        point_keep = torch.repeat_interleave(
            keep.to(offsets.device), lengths, dim=0, output_size=data["image_mask"].shape[0])
        modified = dict(data)
        modified["image_mask"] = data["image_mask"] & point_keep
        return self.original(modified, features, patch, missing)


def install_treatment(model, image_module, method, protocol):
    if method not in ("control", "mean", "gate", "camera_dropout"):
        raise ValueError("Unsupported training treatment")
    backbone = model.backbone
    freeze_dino(backbone)
    if method == "gate":
        # Initialize new CPU modules without changing the pretrained model's RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(protocol["seed"] + 41).get_state())
            count = 0
            for module in list(backbone.modules()):
                if isinstance(module, image_module.SerializedUnpooling):
                    linear = next(x for x in module.proj.modules() if isinstance(x, nn.Linear))
                    gate = FusionGate(linear.out_features, protocol["gate_hidden"])
                    module.add_module("lab_gate", gate.to(device=linear.weight.device, dtype=linear.weight.dtype))
                    module.forward = MethodType(gated_unpool_forward, module)
                    count += 1
        if count != 4:
            raise ValueError(f"Expected four DITR decoder injection stages, found {count}")
    policy = AssignmentPolicy(backbone, image_module.assign_image_feat, method, protocol)
    image_module.assign_image_feat = policy
    return policy


def load_full_state(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = {key.removeprefix("module."): value for key, value in checkpoint["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    return checkpoint
