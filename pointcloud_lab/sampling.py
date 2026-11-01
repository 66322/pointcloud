"""Image-plane quality proxies and bounded-memory local token experiments."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def point_validity(data, features, patch):
    uv = data["image_coord"] / patch
    height, width = features.shape[-2:]
    return (data["image_mask"].bool() & torch.isfinite(uv).all(-1)
            & (uv[..., 0] >= 0) & (uv[..., 0] < width)
            & (uv[..., 1] >= 0) & (uv[..., 1] < height))


class LocalTokenAttention(nn.Module):
    """Raw xyz/strength query; 3x3 visual keys, residual initialized to zero.

    Project keys once per feature map. Checkpoint each point chunk so training
    does not retain N*9*1024 gathered visual values for backward.
    """
    def __init__(self, channels, hidden=32, chunk_size=2048):
        super().__init__()
        self.query = nn.Linear(4, hidden)
        self.key = nn.Linear(channels, hidden)
        self.alpha = nn.Parameter(torch.zeros(()))
        self.chunk_size = chunk_size

    def _chunk(self, point_features, image, keys, xy):
        if point_features.shape[-1] != 4:
            raise ValueError("Local query expects nuScenes xyz+strength (four features)")
        offsets = xy.new_tensor([(x, y) for y in (-1, 0, 1) for x in (-1, 0, 1)])
        candidate = xy[:, None, :] + offsets[None, :, :]
        height, width = image.shape[-2:]
        valid = ((candidate[..., 0] >= 0) & (candidate[..., 0] < width)
                 & (candidate[..., 1] >= 0) & (candidate[..., 1] < height))
        x = candidate[..., 0].clamp(0, width - 1)
        y = candidate[..., 1].clamp(0, height - 1)
        query = F.normalize(self.query(point_features).float(), dim=-1)
        scores = (query[:, None, :] * keys[y, x].float()).sum(-1) / 0.2
        weights = scores.masked_fill(~valid, -torch.inf).softmax(-1)
        values = image.permute(1, 2, 0)[y, x]
        pooled = (values * weights[..., None].to(values.dtype)).sum(1)
        center = image[:, xy[:, 1], xy[:, 0]].T
        return center + self.alpha.tanh() * (pooled - center)

    def forward(self, point_features, image, uv):
        keys = F.normalize(self.key(image.permute(1, 2, 0)).float(), dim=-1)
        chunks = []
        for start in range(0, len(uv), self.chunk_size):
            inputs = (point_features[start:start + self.chunk_size], image, keys,
                      uv[start:start + self.chunk_size].long())
            if self.training and torch.is_grad_enabled():
                # This chunk contains no random operations.
                chunks.append(checkpoint(self._chunk, *inputs, use_reentrant=False,
                                         preserve_rng_state=False))
            else:
                chunks.append(self._chunk(*inputs))
        return torch.cat(chunks)


def assign_sampled(data, features, patch, missing=None, *, mode, local=None):
    """Keep original last-camera choice except for the named center strategies."""
    if mode not in {"center_select", "center_weighted", "bilinear", "local_attention"}:
        raise ValueError(f"Unknown sampled assignment: {mode}")
    valid_all = point_validity(data, features, patch)
    offsets = data.get("unmix3d_offset", data["offset"]).tolist()
    if len(offsets) != len(features) or offsets[-1] != len(data["image_coord"]):
        raise ValueError("Point offsets and image batches disagree")
    outputs, start = [], 0
    height, width = features.shape[-2:]
    for batch, end in enumerate(offsets):
        n, channels = end - start, features.shape[2]
        result = (missing.expand(n, -1).clone() if missing is not None
                  else features.new_zeros((n, channels)))
        sums = torch.zeros_like(result) if mode == "center_weighted" else None
        weights = result.new_zeros((n, 1)) if mode == "center_weighted" else None
        best = result.new_full((n,), -torch.inf) if mode == "center_select" else None
        for camera in range(features.shape[1]):
            idx = valid_all[start:end, camera].nonzero(as_tuple=True)[0]
            if not len(idx):
                continue
            image = features[batch, camera]
            pixels = data["image_coord"][start:end, camera][idx]
            uv = pixels / patch
            if mode == "bilinear":
                # Image pixels are centered on integers; token centers are
                # ((patch-1)/2 + k*patch). align_corners=False, border padding.
                grid = (pixels.float() + 0.5) / pixels.new_tensor(
                    (width * patch, height * patch)).float() * 2 - 1
                values = F.grid_sample(image[None].float(), grid[None, None],
                                       mode="bilinear", padding_mode="border",
                                       align_corners=False)[0, :, 0].T.to(result.dtype)
            elif mode == "local_attention":
                values = local(data["feat"][start:end][idx], image, uv)
            else:
                values = image[:, uv[:, 1].long(), uv[:, 0].long()].T
            if mode.startswith("center_"):
                normalized = (pixels + 0.5) / pixels.new_tensor((width * patch, height * patch))
                radius2 = ((normalized - 0.5) * 2).square().sum(-1)
                quality = torch.exp(-2 * radius2).to(result.dtype)
                if mode == "center_select":
                    better = quality > best[idx]
                    chosen = idx[better]
                    result[chosen] = values[better]
                    best[chosen] = quality[better]
                else:
                    sums[idx] = sums[idx] + values * quality[:, None]
                    weights[idx] += quality[:, None]
            else:
                result[idx] = values
        if mode == "center_weighted":
            observed = weights[:, 0] > 0
            result[observed] = sums[observed] / weights[observed]
        outputs.append(result)
        start = end
    return torch.cat(outputs)


def local_average(features):
    shape = features.shape
    return F.avg_pool2d(features.flatten(0, 1), 3, stride=1, padding=1,
                        count_include_pad=False).reshape(shape)
