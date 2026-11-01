"""Small explicit paired fine-tuning protocol; contains no Torch import."""

METHODS = ("control", "gate", "mean", "camera_dropout")


def validate_protocol(protocol):
    fields = {"id", "epochs", "batch_size", "workers", "prefetch_factor", "seed",
              "max_lr", "block_lr_ratio", "gate_hidden", "camera_dropout", "methods"}
    if set(protocol) != fields:
        raise ValueError(f"Training protocol keys must be {sorted(fields)}")
    for key in ("epochs", "batch_size", "prefetch_factor", "gate_hidden"):
        if not isinstance(protocol[key], int) or protocol[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(protocol["workers"], int) or protocol["workers"] < 0:
        raise ValueError("workers must be a nonnegative integer")
    if not 0 < protocol["max_lr"] or not 0 < protocol["block_lr_ratio"] <= 1:
        raise ValueError("Positive learning rates required")
    if not 0 < protocol["camera_dropout"] < 1:
        raise ValueError("camera_dropout must be between zero and one")
    methods = protocol["methods"]
    if (not methods or methods[0] != "control" or len(set(methods)) != len(methods)
            or set(methods) - set(METHODS)):
        raise ValueError("Unique known methods required, beginning with control")
    return protocol


def sample_seed(*parts):
    """Seed integers only; never hash data, images, checkpoints or arrays."""
    value = 0xA511E9B3
    for part in parts:
        value = (value * 1664525 + int(part) + 1013904223) & 0xFFFFFFFF
    return value
