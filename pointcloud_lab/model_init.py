"""Build DINO architecture locally before a full, strict DITR checkpoint load."""
from contextlib import contextmanager


@contextmanager
def checkpoint_only_dino_init(timm_module):
    """Skip DINO's extra pretrained fetch; caller must strictly load all weights."""
    original = timm_module.create_model

    def create_model(model_name, *args, **kwargs):
        if isinstance(model_name, str) and "dinov2" in model_name.lower():
            if args:
                args = (False,) + args[1:]
                kwargs.pop("pretrained", None)
            else:
                kwargs["pretrained"] = False
        return original(model_name, *args, **kwargs)

    timm_module.create_model = create_model
    try:
        yield
    finally:
        timm_module.create_model = original
