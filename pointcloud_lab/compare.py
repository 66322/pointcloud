"""Compare paired diagnostics using tokens, mapping version and label counts."""
import argparse
import csv
from pathlib import Path
from .common import fresh_dir, read_json, timestamp, write_json


def compare(reference, candidate, out):
    reference, candidate = Path(reference), Path(candidate)
    for folder in (reference, candidate):
        if read_json(folder / "status.json")["status"] != "completed":
            raise ValueError("Comparison needs two completed diagnostics")
    ref_meta, can_meta = [read_json(x / "provenance.json") for x in (reference, candidate)]
    for field in ("label_mapping_version", "selected_tokens"):
        if ref_meta.get(field, "legacy_mapping_raw20motorcycle") != can_meta.get(
                field, "legacy_mapping_raw20motorcycle"):
            raise ValueError(f"Paired comparison mismatch: {field}")
    ref, can = [read_json(x / "summary.json") for x in (reference, candidate)]
    if ref["groups"].keys() != can["groups"].keys():
        raise ValueError("Group definitions differ")
    rows = []
    for name, old in ref["groups"].items():
        new = can["groups"][name]
        if old["target_support"] != new["target_support"] or old["all_points"] != new["all_points"]:
            raise ValueError(f"Label/group support changed: {name}")
        rows.append({
            "group": name, "labelled_points": old["labelled_points"],
            "reference_miou_fixed16_pct": old["miou_fixed16_pct"],
            "candidate_miou_fixed16_pct": new["miou_fixed16_pct"],
            "delta_miou_pp": new["miou_fixed16_pct"] - old["miou_fixed16_pct"],
        })
    output = fresh_dir(out)
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "summary.json", {
        "kind": "paired_diagnostic_comparison", "frames": ref["frames"], "groups": rows,
        "note": "No trained-method/novelty claim; inspect per-class support and full validation.",
    })
    lines = ["# Paired diagnostic comparison", "",
             f"Frames: {ref['frames']}. Same tokens, mapping version and label support.", "",
             "| Group | Reference mIoU % | Candidate mIoU % | Delta pp |",
             "|---|---:|---:|---:|"]
    lines += [f"| {r['group']} | {r['reference_miou_fixed16_pct']:.4f} | "
              f"{r['candidate_miou_fixed16_pct']:.4f} | {r['delta_miou_pp']:+.4f} |" for r in rows]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "status.json", {"status": "completed", "ended_at": timestamp()})
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    compare(args.reference, args.candidate, args.out)


if __name__ == "__main__":
    main()
