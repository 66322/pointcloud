"""Parameter-free point-image assignment interventions; torch imported lazily."""


def assign_variant(data, features, patch_size, missing=None, *, mode="mean",
                   order="normal", dx=0.0, dy=0.0):
    import torch
    if mode not in {"last", "first", "mean", "zero_features"}:
        raise ValueError(f"Unknown assignment mode: {mode}")
    if order not in {"normal", "reverse"}:
        raise ValueError("Order must be normal/reverse")
    offsets = data.get("unmix3d_offset", data["offset"]).tolist()
    outputs = []
    start = 0
    for batch, end in enumerate(offsets):
        coords = data["image_coord"][start:end].clone()
        coords[..., 0] += dx
        coords[..., 1] += dy
        coords = coords / patch_size
        n = end - start
        channels = features.shape[2]
        result = (missing.expand(n, -1).clone() if missing is not None
                  else features.new_zeros((n, channels)))
        if mode == "zero_features":
            outputs.append(torch.zeros_like(result))
            start = end
            continue
        sums, counts = torch.zeros_like(result), result.new_zeros((n, 1))
        seen = torch.zeros(n, dtype=torch.bool, device=result.device)
        camera_ids = list(range(features.shape[1]))
        if order == "reverse":
            camera_ids.reverse()
        for camera in camera_ids:
            uv = coords[:, camera]
            image = features[batch, camera]
            h, w = image.shape[-2:]
            valid = (
                data["image_mask"][start:end, camera].bool()
                & torch.isfinite(uv).all(-1)
                & (uv[:, 0] >= 0) & (uv[:, 0] < w)
                & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            )
            if mode == "first":
                valid = valid & ~seen
            idx = torch.nonzero(valid, as_tuple=True)[0]
            values = image[:, uv[idx, 1].long(), uv[idx, 0].long()].T
            if mode == "mean":
                sums[idx] += values
                counts[idx] += 1
            else:
                result[idx] = values
            seen |= valid
        if mode == "mean":
            valid = counts[:, 0] > 0
            result[valid] = sums[valid] / counts[valid]
        outputs.append(result)
        start = end
    if start != len(data["image_coord"]) or len(offsets) != features.shape[0]:
        raise ValueError("Point offsets and image batches disagree")
    return torch.cat(outputs)
