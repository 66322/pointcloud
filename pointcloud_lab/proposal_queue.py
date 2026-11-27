"""Run paired proposal comparisons, screening with one registered seed by default."""
import csv
from collections import Counter
from pathlib import Path
import os
import random
import shutil
from statistics import mean, stdev

from . import campaign
from .common import fresh_dir, git_state, load_infos, read_json, replace_record, timestamp, write_json
from .proposal_data import CONDITIONS
from .run_training import training_command, validation_command
from .training_protocol import PROPOSAL_METHODS, validate_protocol

DEFAULT_PLAN = "configs/training/proposal_alignment_pilot.json"


def schedule(plan, output, seeds=None, with_stf=False):
    seeds = plan["seeds"][:1] if seeds is None else seeds
    if (not seeds or len(set(seeds)) != len(seeds) or set(seeds) - set(plan["seeds"])
            or len(set(plan["seeds"])) != len(plan["seeds"])):
        raise ValueError("Choose distinct registered training seeds")
    methods = plan["protocol"]["methods"]
    if set(methods) - set(PROPOSAL_METHODS) or "control" not in methods:
        raise ValueError("Invalid proposal methods or missing matched DITR control")
    conditions = {method: ["clean"] for method in methods}
    for group in plan["groups"].values():
        if "control" not in group["methods"] or set(group["conditions"]) - set(CONDITIONS):
            raise ValueError("Evaluation group needs its matched control and known conditions")
        for method in group["methods"]:
            conditions[method].extend(group["conditions"])
    if with_stf:
        if not plan["external_methods"]:
            raise ValueError("This screening plan registers no external evaluation")
        for method in plan["external_methods"]:
            conditions[method].append("stf_val")
    if any(len(v) != len(set(v)) for v in conditions.values()):
        raise ValueError("Duplicate evaluation of a checkpoint under the same condition")
    for contrast in plan["contrasts"]:
        for name in contrast["conditions"]:
            if name == "stf_val" and not with_stf:
                continue
            if any(name not in conditions[m] for m in (contrast["reference"], contrast["method"])):
                raise ValueError(f"Contrast {contrast['id']} lacks a registered evaluation")
    protocols, jobs = {}, []
    for seed in seeds:
        phase = f"seed_{seed}"
        protocols[phase] = validate_protocol(dict(plan["protocol"], seed=seed))
        order = list(methods)
        random.Random(seed).shuffle(order)
        def add(job_id, kind, method, dependencies, **extra):
            jobs.append(dict(id=job_id, kind=kind, phase=phase, seed=seed, method=method,
                             dependencies=dependencies, status="pending",
                             output=str(output / "jobs" / job_id), **extra))
        for method in order:
            train = f"{phase}_{method}_train"
            add(train, "train", method, [])
            # A clean result from the first finished model is available while the other GPU trains.
            for cond in conditions[method]:
                add(f"{phase}_{method}_{cond}", "eval", method, [train], train_job=train, condition=cond)
    return protocols, jobs


def command(job, output, by_id, frames):
    if job["kind"] == "train":
        return training_command(output / "paths.json", output / (job["phase"] + "_protocol.json"),
                                job["method"], job["gpu"], job["output"])
    cond = job["condition"]
    tokens, count = ("stf_tokens.json", 250) if cond == "stf_val" else ("tokens.json", frames)
    base = validation_command(output / "paths.json", by_id[job["train_job"]]["output"], output / tokens,
                              job["gpu"], job["output"], count,
                              output / "code/configs/experiments/default_assignment.json")
    return base + ["--proposal-condition", cond]


def csv_file(path, rows):
    if not rows:
        return
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    replace_record(tmp, path)


def report(output, jobs):
    plan = read_json(output / "plan.json")
    selection = read_json(output / "selection.json")
    scores = {(j["seed"], j["method"], j["condition"]): j["summary"] for j in jobs
              if j["kind"] == "eval" and j["status"] in campaign.DONE}
    rows = []
    for j in jobs:
        if j["kind"] != "eval":
            continue
        seed, method, cond = j["seed"], j["method"], j["condition"]
        metric = scores.get((seed, method, cond), {})
        value = metric.get("miou_pct")
        control = scores.get((seed, "control", cond), {}).get("miou_pct")
        clean_metric = scores.get((seed, method, "clean"), {})
        clean = clean_metric.get("miou_pct")
        if cond in {"sparse", "dark_sparse"}:
            clean = clean_metric.get("sparse_support_miou_pct")
        diff = value - control if value is not None and control is not None else None
        drop = clean - value if clean is not None and value is not None and cond != "stf_val" else None
        rows.append(dict(seed=seed, method=method, condition=cond, status=j["status"],
                         miou_pct=value, delta_vs_control_pp=diff, clean_loss_same_points_pp=drop,
                         macc_pct=metric.get("macc_pct"), oa_pct=metric.get("oa_pct")))
    csv_file(output / "comparison.csv", rows)
    contrasts, paired = [], []
    for item in plan["contrasts"]:
        conditions = item["conditions"]
        if "stf_val" in conditions and not selection["with_stf"]:
            continue
        values = []
        for seed in selection["seeds"]:
            differences = []
            for cond in conditions:
                ref = scores.get((seed, item["reference"], cond), {}).get("miou_pct")
                val = scores.get((seed, item["method"], cond), {}).get("miou_pct")
                if ref is not None and val is not None:
                    differences.append(val - ref)
            delta = mean(differences) if len(differences) == len(conditions) else None
            if delta is not None:
                values.append(delta)
            paired.append(dict(contrast=item["id"], seed=seed, method=item["method"],
                               reference=item["reference"], conditions=",".join(conditions),
                               completed_conditions=len(differences), delta_pp=delta))
        contrasts.append(dict(contrast=item["id"], paired_seeds=len(values),
                              expected_seeds=len(selection["seeds"]),
                              mean_delta_pp=mean(values) if values else None,
                              sample_sd_pp=stdev(values) if len(values) > 1 else None))
    csv_file(output / "seed_deltas.csv", paired)
    csv_file(output / "contrasts.csv", contrasts)
    campaign.atomic_json(output / "summary.json", {"kind": "proposal_v1", "counts": dict(Counter(
        j["status"] for j in jobs)), "contrasts": contrasts})


def run(args):
    output = Path(args.out or ".").resolve()
    if args.resume:
        if args.seed or args.with_stf or args.preview:
            raise ValueError("Resume uses the saved seeds, data and code snapshot")
        selection = read_json(output / "selection.json")
        jobs = read_json(output / "queue.json")["jobs"]
        paths = read_json(output / "paths.json")
        prior = selection["after"]
    else:
        plan = read_json(args.plan)
        protocols, jobs = schedule(plan, output, args.seed, args.with_stf)
        seeds = [protocol["seed"] for protocol in protocols.values()]
        if args.preview:
            print(f"Seeds {seeds}; GPU 0/1; " + str(dict(Counter(j["kind"] for j in jobs))))
            for group, entries in plan["groups"].items():
                print(group, ", ".join(entries["conditions"]))
            return
        paths = read_json(args.paths)
        prior = str(Path(args.after).resolve()) if args.after else None
        if prior and not Path(prior).is_dir():
            raise ValueError(f"Prior campaign not found: {prior}")
        infos = load_infos(paths["info"])
        tokens = [str(i["lidar_token"]) for i in infos]
        if len(tokens) != 6019:
            raise ValueError(f"Expected full nuScenes validation (6019), got {len(tokens)}")
        external = None
        if args.with_stf:
            from .semanticstf import manifest
            external = list(manifest(paths["semanticstf_root"]))
            if len(external) != 250:
                raise ValueError("Expected 250 SemanticSTF validation frames")
        fresh_dir(output)
        (output / "jobs").mkdir()
        (output / "logs").mkdir()
        shutil.copytree(campaign.ROOT / "pointcloud_lab", output / "code/pointcloud_lab",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        experiment = Path("configs/experiments/default_assignment.json")
        (output / "code" / experiment.parent).mkdir(parents=True)
        shutil.copy2(campaign.ROOT / experiment, output / "code" / experiment)
        origin = {"source_git": git_state(campaign.ROOT), "created_at": timestamp()}
        write_json(output / "code/snapshot.json", origin)
        write_json(output / "provenance.json", origin)
        write_json(output / "plan.json", plan)
        write_json(output / "paths.json", paths)
        write_json(output / "selection.json", {"seeds": seeds, "with_stf": args.with_stf, "after": prior})
        write_json(output / "tokens.json", tokens)
        if external is not None:
            write_json(output / "stf_tokens.json", external)
        for phase, protocol in protocols.items():
            write_json(output / (phase + "_protocol.json"), protocol)
        campaign.atomic_json(output / "queue.json", {"jobs": jobs, "updated_at": timestamp()})
    lock = campaign.controller_lock(output)
    try:
        campaign.atomic_json(output / "controller.json", {"pid": os.getpid(), "started_at": timestamp()})
        if prior:
            campaign.wait_for_prior(Path(prior), output)
        campaign.atomic_json(output / "status.json", {"status": "running", "started_at": timestamp()})
        final = campaign.execute_queue(output, jobs, len(read_json(output / "tokens.json")),
                                       command_builder=command, reporter=report)
        campaign.atomic_json(output / "status.json", {"status": final, "ended_at": timestamp()})
    except BaseException as error:
        campaign.atomic_json(output / "status.json", {"status": "failed", "error": str(error),
                                                     "ended_at": timestamp()})
        raise
    finally:
        lock.close()
