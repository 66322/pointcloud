import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(root):
    def git(*args):
        p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    return {"sha": git("rev-parse", "HEAD"), "dirty": git("status", "--porcelain")}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def fresh_dir(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def load_infos(path):
    # Only trusted local preprocessing indexes; pickle can execute code.
    import pickle
    with Path(path).open("rb") as stream:
        infos = pickle.load(stream)
    if not isinstance(infos, list):
        raise ValueError("Expected the DITR list-of-frame-dicts index")
    tokens = [str(item["lidar_token"]) for item in infos]
    if len(set(tokens)) != len(tokens):
        raise ValueError("Duplicate lidar tokens in index")
    return infos


def select_infos(infos, limit=None, seed=20261101, tokens=None):
    """Stable hash ordering, independent of pickle list order; nested pilots."""
    by_token = {str(x["lidar_token"]): x for x in infos}
    if tokens is not None:
        if len(set(tokens)) != len(tokens):
            raise ValueError("Duplicate requested tokens")
        missing = set(tokens) - by_token.keys()
        if missing:
            raise ValueError(f"Requested tokens absent from index: {len(missing)}")
        selected = [by_token[x] for x in tokens]
    else:
        selected = sorted(infos, key=lambda x: hashlib.sha256(
            f"{seed}:{x['lidar_token']}".encode()).hexdigest())
    if limit is not None:
        if limit < 1 or limit > len(selected):
            raise ValueError("Limit must be between 1 and available frame count")
        selected = selected[:limit]
    return selected


def resolve_raw(data_root, relative):
    """Resolve an index path without allowing escape from the raw-data root."""
    root = (Path(data_root) / "raw").resolve()
    relative = Path(relative)
    if relative.is_absolute():
        raise ValueError("Expected relative raw-data path in index")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Raw-data path escaped data_root/raw")
    return path
