"""Apply fine-tuning overrides using the upstream configuration container types."""


def configure_optimizer(cfg, protocol):
    # Config.__setattr__ wraps a dict, but does not recursively wrap dicts inside
    # an assigned list. The upstream optimizer accesses each group's .lr/.keyword.
    from pointcept.utils.config import ConfigDict
    cfg.optimizer.lr = protocol["max_lr"]
    cfg.param_dicts = [ConfigDict(
        keyword=r"img_enc|block",
        lr=protocol["max_lr"] * protocol["block_lr_ratio"],
    )]
    if cfg.scheduler.type != "OneCycleLR":
        raise ValueError("This paired protocol expects the existing OneCycleLR scheduler")
    cfg.scheduler.max_lr = [protocol["max_lr"], protocol["max_lr"] * protocol["block_lr_ratio"]]
