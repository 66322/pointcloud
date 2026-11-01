"""Integration against actual upstream Python sources; no CUDA/model/data load.

Use configured ditr_root, POINTCLOUD_UPSTREAM_ROOT, or the ignored local source
cache. The test never downloads code or imports Pointcept's CUDA extensions.
"""
from contextlib import contextmanager, nullcontext
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def upstream_root():
    candidates = []
    if os.environ.get("POINTCLOUD_UPSTREAM_ROOT"):
        candidates.append(Path(os.environ["POINTCLOUD_UPSTREAM_ROOT"]))
    local = ROOT / "configs/paths.local.json"
    if local.is_file():
        candidates.append(Path(json.loads(local.read_text(encoding="utf-8"))["ditr_root"]))
    candidates.append(ROOT / "external/ditr_optimizer_contract")
    return next((root for root in candidates
                 if all((root / "pointcept/utils" / name).is_file()
                        for name in ("config.py", "optimizer.py", "registry.py", "scheduler.py"))), None)


@contextmanager
def original_utils(root):
    # Only unrelated logger/path/import helpers are isolated. The configuration,
    # registry, optimizer and scheduler modules execute their real source unchanged.
    package = ModuleType("pointcept")
    package.__path__ = []
    utils = ModuleType("pointcept.utils")
    utils.__path__ = [str(root / "pointcept/utils")]
    misc = ModuleType("pointcept.utils.misc")
    misc.is_seq_of = lambda value, kind: isinstance(value, (list, tuple)) and all(
        isinstance(item, kind) for item in value)
    misc.import_modules_from_strings = lambda imports, allow_failed_imports=False: None
    path = ModuleType("pointcept.utils.path")
    def check_file_exist(filename, msg_tmpl="file {} does not exist"):
        if not Path(filename).is_file():
            raise FileNotFoundError(msg_tmpl.format(filename))
    path.check_file_exist = check_file_exist
    logger = ModuleType("pointcept.utils.logger")
    logger.get_root_logger = lambda: logging.getLogger("upstream-optimizer-contract")
    modules = {"pointcept": package, "pointcept.utils": utils, "pointcept.utils.misc": misc,
               "pointcept.utils.path": path, "pointcept.utils.logger": logger}
    loaded = {}
    touched = set(modules) | {"pointcept.utils." + name for name in ("config", "registry", "optimizer", "scheduler")}
    absent = object()
    previous = {name: sys.modules.get(name, absent) for name in touched}
    sys.modules.update(modules)
    try:
        for name in ("config", "registry", "optimizer", "scheduler"):
            qualified = "pointcept.utils." + name
            spec = importlib.util.spec_from_file_location(qualified, root / "pointcept/utils" / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[qualified] = module
            # Exercise the deployed Linux parser branch on the Windows authoring host.
            with (patch("platform.system", return_value="Linux") if name == "config" else nullcontext()):
                spec.loader.exec_module(module)
            loaded[name] = module
        yield loaded
    finally:
        for name, old in previous.items():
            if old is absent:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


DEPENDENCIES = all(importlib.util.find_spec(name) is not None for name in ("torch", "addict", "yapf"))
UPSTREAM = upstream_root()


@unittest.skipUnless(DEPENDENCIES and UPSTREAM, "Configured upstream Python sources/Torch required")
class UpstreamOptimizerTests(unittest.TestCase):
    def test_real_config_optimizer_scheduler_and_update(self):
        import torch
        from pointcloud_lab.training_setup import configure_optimizer
        protocol = json.loads((ROOT / "configs/training/stage1.json").read_text())
        with original_utils(UPSTREAM) as upstream:
            Config, ConfigDict = upstream["config"].Config, upstream["config"].ConfigDict
            def config():
                return Config({
                    "optimizer": {"type": "AdamW", "lr": 0.002, "weight_decay": 0.005},
                    "scheduler": {"type": "OneCycleLR", "max_lr": [0.002, 0.0002],
                                  "total_steps": 100, "pct_start": 0.04, "div_factor": 10.0,
                                  "final_div_factor": 100.0, "anneal_strategy": "cos"},
                }, cfg_text="# optimizer integration fixture", filename="initial.py")
            model = torch.nn.Module()
            model.block = torch.nn.Linear(3, 3)
            model.head = torch.nn.Linear(3, 2)
            model.img_enc = torch.nn.Linear(3, 3).requires_grad_(False)
            bad = config()
            bad.param_dicts = [dict(keyword=r"img_enc|block", lr=0.00002)]
            with self.assertRaisesRegex(AttributeError, "lr"):
                upstream["optimizer"].build_optimizer(bad.optimizer, model, bad.param_dicts)

            cfg = config()
            configure_optimizer(cfg, protocol)
            self.assertIsInstance(cfg.param_dicts[0], ConfigDict)
            self.assertEqual(cfg.param_dicts[0].keyword, r"img_enc|block")
            with tempfile.TemporaryDirectory() as tmp:
                cfg.dump(str(Path(tmp) / "config.py"))
            optimizer = upstream["optimizer"].build_optimizer(cfg.optimizer, model, cfg.param_dicts)
            self.assertEqual(len(optimizer.param_groups), 2)
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], protocol["max_lr"])
            self.assertAlmostEqual(optimizer.param_groups[1]["lr"], protocol["max_lr"] * protocol["block_lr_ratio"])
            block_ids = {id(parameter) for parameter in optimizer.param_groups[1]["params"]}
            self.assertIn(id(model.block.weight), block_ids)
            self.assertNotIn(id(model.head.weight), block_ids)
            scheduler = upstream["scheduler"].build_scheduler(cfg.scheduler, optimizer)
            before = model.head.weight.detach().clone()
            frozen = model.img_enc.weight.detach().clone()
            optimizer.zero_grad()
            loss = model.head(model.block(torch.ones(2, 3))).square().mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            self.assertFalse(torch.equal(before, model.head.weight))
            self.assertTrue(torch.equal(frozen, model.img_enc.weight))
            self.assertEqual(scheduler.last_epoch, 1)


if __name__ == "__main__":
    unittest.main()
