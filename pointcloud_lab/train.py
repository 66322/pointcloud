"""Fine-tune the existing DITR checkpoint; the first real update checks gradients."""
import argparse
import importlib
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
import time

from .common import fresh_dir, git_state, read_json, timestamp, write_json
from .training_protocol import METHODS, validate_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--gpu", choices=("0", "1"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", action="store_true", help="Resume this run's own last checkpoint")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    paths, protocol = read_json(args.paths), validate_protocol(read_json(args.protocol))
    output = Path(args.out).resolve() if args.resume else fresh_dir(args.out)
    if args.resume:
        saved = read_json(output / "experiment.json")
        if saved != {"method": args.method, "protocol": protocol}:
            raise ValueError("Resume requires the identical saved method and training protocol")
    else:
        write_json(output / "experiment.json", {"method": args.method, "protocol": protocol})
    write_json(output / "status.json", {"status": "running", "started_at": timestamp()})
    try:
        os.environ.setdefault("HF_HOME", paths["hf_cache"])
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(name, "1")
        sys.path.insert(0, paths["ditr_root"])
        os.chdir(paths["ditr_root"])
        import numpy as np
        import torch
        import timm
        from pointcept.engines.defaults import default_config_parser, default_setup
        from pointcept.utils.env import set_seed
        from pointcept.utils.logger import get_root_logger
        from pointcept.models import build_model
        from pointcept.datasets import build_dataset, point_collate_fn
        from pointcept.utils.optimizer import build_optimizer
        from pointcept.utils.scheduler import build_scheduler
        from .model_init import checkpoint_only_dino_init
        from .training_models import install_treatment, load_full_state
        from .training_data import PairedDataset, EpochSampler, PairedCollator, initialize_worker

        cfg = default_config_parser(paths["config"], {
            "save_path": str(output), "weight": paths["weight"], "resume": False,
            "seed": protocol["seed"], "epoch": protocol["epochs"], "eval_epoch": protocol["epochs"],
            "batch_size": protocol["batch_size"], "num_worker": protocol["workers"],
            "model.backbone.dinov2_frozen": False, "wandb_project": None, "evaluate": False,
        })
        for split in ("train", "val", "test"):
            cfg.data[split].data_root = paths["data_root"]
        cfg.hooks = []  # This adapter owns checkpoints/progress; no extra per-epoch validation.
        cfg.optimizer.lr = protocol["max_lr"]
        cfg.param_dicts = [dict(keyword=r"img_enc|block",
                               lr=protocol["max_lr"] * protocol["block_lr_ratio"])]
        if cfg.scheduler.type != "OneCycleLR":
            raise ValueError("This paired protocol expects the existing OneCycleLR scheduler")
        cfg.scheduler.max_lr = [protocol["max_lr"], protocol["max_lr"] * protocol["block_lr_ratio"]]
        cfg = default_setup(cfg)
        # Do not multiply the model seed by a performance-related worker count.
        set_seed(protocol["seed"])
        cfg.dump(str(output / "config.py"))
        logger = get_root_logger(log_file=str(output / "train.log"), file_mode="a" if args.resume else "w")

        # This is the actual training load, not a separate scan or preflight.
        dataset = build_dataset(cfg.data.train)
        if not len(dataset):
            raise ValueError("Training split is empty")
        context = mp.get_context("spawn")
        epoch_value = context.Value("q", 0)
        paired = PairedDataset(dataset, epoch_value, protocol["seed"])
        loader_kwargs = dict(
            dataset=paired, batch_size=protocol["batch_size"],
            sampler=EpochSampler(paired, epoch_value, protocol["seed"]),
            num_workers=protocol["workers"], drop_last=True, pin_memory=True,
            collate_fn=PairedCollator(point_collate_fn, cfg.mix_prob, protocol["seed"]),
            generator=torch.Generator().manual_seed(protocol["seed"] + 503),
        )
        if protocol["workers"]:
            loader_kwargs.update(
                multiprocessing_context=context, persistent_workers=True,
                prefetch_factor=protocol["prefetch_factor"], worker_init_fn=initialize_worker,
            )
        loader = torch.utils.data.DataLoader(**loader_kwargs)
        if not len(loader):
            raise ValueError("Training split is smaller than the configured batch")

        image_module = importlib.import_module(
            "pointcept.models.point_transformer_v3.point_transformer_v3m1_image")
        with checkpoint_only_dino_init(timm):
            model = build_model(cfg.model)
        if cfg.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if not args.resume:
            load_full_state(model, paths["weight"])
        policy = install_treatment(model, image_module, args.method, protocol)
        resume_checkpoint = load_full_state(model, output / "model/model_last.pth") if args.resume else None
        model = model.cuda()
        optimizer = build_optimizer(cfg.optimizer, model, cfg.param_dicts)
        cfg.scheduler.total_steps = len(loader) * protocol["epochs"]
        scheduler = build_scheduler(cfg.scheduler, optimizer)
        scaler = torch.amp.GradScaler("cuda") if cfg.enable_amp else None
        start_epoch = next_batch = attempts = updates = 0
        if resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
            scheduler.load_state_dict(resume_checkpoint["scheduler"])
            if scaler:
                scaler.load_state_dict(resume_checkpoint["scaler"])
            start_epoch, next_batch = resume_checkpoint["epoch"], resume_checkpoint["next_batch"]
            attempts, updates = resume_checkpoint["attempts"], resume_checkpoint["updates"]
            rng = resume_checkpoint["rng"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            torch.cuda.set_rng_state(rng["cuda"])
            policy.generator.set_state(resume_checkpoint["camera_rng"])
            policy.calls = resume_checkpoint.get("camera_calls", 0)
            policy.dropped = resume_checkpoint.get("cameras_dropped", 0)
            del resume_checkpoint
        total_attempts = len(loader) * protocol["epochs"]
        write_json(output / "provenance.json", {
            "method": args.method, "protocol": protocol, "created_at": timestamp(),
            "lab_git": git_state(Path(__file__).resolve().parents[1]),
            "source_snapshot": (
                read_json(Path(__file__).resolve().parents[1] / "snapshot.json")
                if (Path(__file__).resolve().parents[1] / "snapshot.json").is_file() else None),
            "upstream_git": git_state(paths["ditr_root"]),
            "initial_checkpoint": paths["weight"], "physical_gpu": args.gpu,
            "train_frames_per_epoch": len(dataset), "steps_per_epoch": len(loader),
            "samples_used_per_epoch": len(loader) * protocol["batch_size"],
            "total_attempted_steps": total_attempts, "amp": bool(cfg.enable_amp),
            "trainable_parameter_elements": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "gate_parameter_elements": sum(p.numel() for name, p in model.named_parameters() if "lab_gate" in name),
            "dino_trainable_elements": sum(p.numel() for p in model.backbone.img_enc.parameters() if p.requires_grad),
            "checkpoint_loading": "strict; pretrained state first, then identity gate installation",
            "loader_transport": "numpy pipe; main-process pinning; no shared tensor storage",
            "augmentation_seed": "sample_seed(seed, epoch, row); model CPU/CUDA RNG isolated",
            "evaluation": "final checkpoint only, original full validation TTA, separate process",
        })

        def save_checkpoint(epoch, batch):
            payload = {
                "epoch": epoch, "next_batch": batch, "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
                "attempts": attempts, "updates": updates,
                "lab": {"method": args.method, "protocol": protocol},
                "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()},
                "camera_rng": policy.generator.get_state(),
                "camera_calls": policy.calls, "cameras_dropped": policy.dropped,
            }
            target = output / "model/model_last.pth"
            torch.save(payload, str(target) + ".tmp")
            os.replace(str(target) + ".tmp", target)

        checked = (output / "first_step.json").is_file() if args.resume else False
        started = time.perf_counter()
        resumed_attempts, wait_seconds = attempts, 0.0
        torch.cuda.reset_peak_memory_stats()
        logger.info("TRAIN %s: %d frames/epoch; %d epochs; batch=%d; total=%d steps; GPU=%s",
                    args.method, len(dataset), protocol["epochs"], protocol["batch_size"], total_attempts, args.gpu)
        with (output / "trajectory.jsonl").open("a", encoding="utf-8", buffering=1) as trajectory:
            import json
            for epoch in range(start_epoch, protocol["epochs"]):
                epoch_value.value = epoch
                model.train()
                iterator = iter(loader)
                for batch_index in range(len(loader)):
                    waiting = time.perf_counter()
                    batch = next(iterator).tensors()
                    wait_seconds += time.perf_counter() - waiting
                    if epoch == start_epoch and batch_index < next_batch:
                        continue
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.cuda(non_blocking=True)
                    optimizer.zero_grad()
                    with torch.autocast("cuda", enabled=cfg.enable_amp):
                        loss = model(batch)["loss"]
                    if scaler:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                    else:
                        loss.backward()
                    if not checked:
                        if not torch.isfinite(loss.detach()):
                            raise RuntimeError("Non-finite first training loss")
                        dino = model.backbone.img_enc
                        if dino.training or any(p.requires_grad or p.grad is not None for p in dino.parameters()):
                            raise RuntimeError("Frozen DINO state/gradient check failed")
                        trainable_grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
                        finite = bool(trainable_grads) and all(bool(torch.isfinite(g).all()) for g in trainable_grads)
                        if args.method == "gate" and not any(
                            p.grad is not None for name, p in model.named_parameters() if "lab_gate" in name
                        ):
                            raise RuntimeError("Gate is not connected to loss")
                        if args.method == "camera_dropout" and not policy.calls:
                            raise RuntimeError("Camera dropout policy was not called")
                    else:
                        finite = True
                    if cfg.clip_grad is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                    successful = True
                    if scaler:
                        before_scale = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        successful = before_scale <= scaler.get_scale()
                    else:
                        if not finite:
                            raise RuntimeError("Non-finite first training gradients")
                        optimizer.step()
                    if successful:
                        scheduler.step()
                        updates += 1
                    attempts += 1
                    if not checked and successful:
                        if not finite:
                            raise RuntimeError("First successful step has non-finite gradients")
                        write_json(output / "first_step.json", {
                            "status": "passed", "loss": float(loss.detach()),
                            "dino_frozen_and_eval": True, "backward": True,
                            "kind": "first actual optimizer update; no extra smoke training pass",
                        })
                        checked = True
                        logger.info("First actual update passed: forward/backward connected, DINO frozen/eval; saving checkpoint")
                        if batch_index + 1 == len(loader):
                            save_checkpoint(epoch + 1, 0)
                        else:
                            save_checkpoint(epoch, batch_index + 1)
                    if cfg.empty_cache:
                        torch.cuda.empty_cache()
                    if attempts == 1 or attempts % 20 == 0 or attempts == total_attempts:
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - started
                        rate = elapsed / max(1, attempts - resumed_attempts)
                        row = {"epoch": epoch + 1, "step": attempts, "updates": updates,
                               "loss": float(loss.detach()), "wall_seconds": elapsed,
                               "data_wait_seconds": wait_seconds,
                               "seconds_per_step": rate, "eta_minutes": (total_attempts - attempts) * rate / 60}
                        trajectory.write(json.dumps(row) + "\n")
                        logger.info("Train %s: [%d/%d] loss=%.5f wall/step=%.3fs wait/step=%.3fs ETA=%.1fmin",
                                    args.method, attempts, total_attempts, row["loss"], rate,
                                    wait_seconds / max(1, attempts - resumed_attempts), row["eta_minutes"])
                next_batch = 0
                save_checkpoint(epoch + 1, 0)
                if cfg.empty_cache_per_epoch:
                    torch.cuda.empty_cache()
        if not checked:
            raise RuntimeError("No successful optimizer update")
        write_json(output / "summary.json", {
            "kind": "training_completed_pending_full_validation", "method": args.method,
            "epochs": protocol["epochs"], "attempts": attempts, "optimizer_updates": updates,
            "amp_skipped_steps": attempts - updates,
            "wall_seconds_this_invocation": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "checkpoint": str(output / "model/model_last.pth"), "first_step_passed": checked,
            "camera_dropout_forward_calls": policy.calls, "cameras_dropped": policy.dropped,
        })
        write_json(output / "status.json", {"status": "completed", "ended_at": timestamp()})
    except BaseException as error:
        write_json(output / "status.json", {"status": "failed", "ended_at": timestamp(), "error": str(error)})
        raise


if __name__ == "__main__":
    main()
