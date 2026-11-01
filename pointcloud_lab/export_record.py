"""Publish a small reviewed evidence bundle; no git command is executed."""
import argparse
import shutil
from pathlib import Path

from .common import read_json, sha256, write_json

ALLOWED = (
    "status.json", "preflight.json", "summary.json", "confusions.json", "per_class.csv",
    "report.md", "provenance.json", "model_audit.json", "runtime.json", "tokens.json",
    "comparison.csv",
)


def export(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    if not source.is_dir():
        raise ValueError("Missing run directory")
    status = read_json(source / "status.json")
    if status["status"] == "running":
        raise ValueError("Cannot freeze evidence while a run is active")
    files = [source / name for name in ALLOWED if (source / name).is_file()]
    if any(x.stat().st_size > 5_000_000 for x in files) or sum(x.stat().st_size for x in files) > 10_000_000:
        raise ValueError("Evidence bundle too large; review before publication")
    target.mkdir(parents=True, exist_ok=False)
    checksums = {}
    for path in files:
        shutil.copy2(path, target / path.name)
        checksums[path.name] = sha256(path)
    # Keep small exact final metric lines, not the full runtime log.
    if (source / "test.log").is_file():
        lines = (source / "test.log").read_text(encoding="utf-8", errors="replace").splitlines()
        metrics = [x for x in lines if "Val result:" in x or
                   ("Class_" in x and "Result:" in x)]
        (target / "metric_lines.txt").write_text("\n".join(metrics) + "\n", encoding="utf-8")
        checksums["metric_lines.txt"] = sha256(target / "metric_lines.txt")
    write_json(target / "artifact_hashes.json", checksums)
    (target / "README.md").write_text(
        "# Experiment evidence\n\n"
        f"Run status: {status['status']}.\n\n"
        "Source outputs remain on the server. This folder contains lightweight evidence only.\n"
        "Before committing: add purpose, comparison, observed result, limitation and next decision here.\n"
        "A completed diagnostic/intervention is not automatically a trained-method improvement.\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--record", required=True)
    args = parser.parse_args()
    export(args.run, args.record)
    print(f"Review {args.record}, then git add that exact directory")


if __name__ == "__main__":
    main()
