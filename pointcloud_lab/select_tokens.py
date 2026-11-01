"""Generate nested deterministic pilot tokens from the same trusted val index."""
import argparse
from pathlib import Path
from .common import load_infos, read_json, select_infos, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths", required=True)
    p.add_argument("--limit", required=True, type=int)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", default=20261101, type=int)
    args = p.parse_args()
    target = Path(args.out)
    if target.exists():
        raise FileExistsError(target)
    paths = read_json(args.paths)
    infos = select_infos(load_infos(paths["info"]), args.limit, args.seed)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, [x["lidar_token"] for x in infos])


if __name__ == "__main__":
    main()
