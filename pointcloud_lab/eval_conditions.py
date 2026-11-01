"""Deterministic, paired evidence removal and image-coordinate stress tests."""


def condition_description(camera_count=6, shift_x=0.0):
    subsets = {6: [0, 1, 2, 3, 4, 5], 3: [0, 2, 4], 1: [0], 0: []}
    if camera_count not in subsets:
        raise ValueError("Camera count must be 0, 1, 3 or 6")
    return {"camera_count": camera_count, "camera_indices": subsets[camera_count],
            "pixel_shift_x": float(shift_x),
            "interpretation": "fixed camera-index subset; image-coordinate shift, not physical SE(3)"}


class EvaluationAssignment:
    def __init__(self, original, camera_count=6, shift_x=0.0):
        self.original = original
        self.condition = condition_description(camera_count, shift_x)

    def __call__(self, data, features, patch, missing=None):
        import torch
        if features.shape[1] != 6:
            raise ValueError("Paired camera conditions expect the six nuScenes cameras")
        modified = dict(data)
        keep = torch.zeros(6, device=data["image_mask"].device, dtype=torch.bool)
        keep[self.condition["camera_indices"]] = True
        modified["image_mask"] = data["image_mask"] & keep[None]
        if self.condition["pixel_shift_x"]:
            coords = data["image_coord"].clone()
            coords[..., 0] += self.condition["pixel_shift_x"]
            modified["image_coord"] = coords
            uv = coords / patch
            height, width = features.shape[-2:]
            modified["image_mask"] = (modified["image_mask"] & torch.isfinite(uv).all(-1)
                                      & (uv[..., 0] >= 0) & (uv[..., 0] < width)
                                      & (uv[..., 1] >= 0) & (uv[..., 1] < height))
        result = self.original(modified, features, patch, missing)
        # The upstream backbone constructs Point(data) after assignment returns.
        if "lab_image_valid" in modified:
            data["lab_image_valid"] = modified["lab_image_valid"]
        return result
