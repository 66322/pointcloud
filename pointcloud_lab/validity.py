"""Propagate observation support without changing upstream 3D pooling or RNG."""
import torch


def validity_pool_forward(self, point):
    valid = point["lab_image_valid"]
    saved = point.image_feat
    if self.lab_mask_pool:
        if self.reduce != "max":
            raise ValueError("Validity pooling adapter expects upstream max pooling")
        point.image_feat = saved.masked_fill(~valid[:, None], -torch.inf)
    try:
        result = self.lab_original_forward(point)
    finally:
        # The traceable skip parent references this same object.
        point.image_feat = saved
    inverse = result["pooling_inverse"].long()
    observed = torch.zeros(len(result.feat), device=valid.device, dtype=torch.int32)
    observed.scatter_reduce_(0, inverse, valid.int(), reduce="amax", include_self=True)
    result["lab_image_valid"] = observed.bool()
    if self.lab_mask_pool:
        result.image_feat = result.image_feat.masked_fill(~observed.bool()[:, None], 0)
    return result


def validity_unpool_forward(self, point):
    parent = point.pop("pooling_parent")
    inverse = point.pop("pooling_inverse")
    point, parent = self.proj(point), self.proj_skip(parent)
    image = self.proj_image_feat(parent.image_feat)
    image = image.masked_fill(~parent["lab_image_valid"][:, None], 0)
    parent.feat = parent.feat + point.feat[inverse] + image
    if self.traceable:
        parent["unpooling_parent"] = point
    return parent
