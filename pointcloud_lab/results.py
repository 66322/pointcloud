"""Read the existing upstream tester's final metrics; no extra data/model pass."""
import re
from pathlib import Path


def read_test_metrics(log):
    text = Path(log).read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"Val result: mIoU/mAcc/allAcc\s+([0-9.eE+-]+)/([0-9.eE+-]+)/([0-9.eE+-]+)", text
    )
    if not matches:
        raise ValueError(f"No final metrics found in {log}")
    miou, macc, oa = [100 * float(x) for x in matches[-1]]
    classes = re.findall(
        r"Class_(\d+)\s+-\s+(.+?)\s+Result:\s+iou/accuracy\s+([0-9.eE+-]+)/([0-9.eE+-]+)",
        text,
    )
    return {
        "miou_pct": miou, "macc_pct": macc, "oa_pct": oa,
        "per_class": {name: {"id": int(i), "iou_pct": 100 * float(iou),
                             "accuracy_pct": 100 * float(acc)}
                      for i, name, iou, acc in classes},
        "source": "upstream SemSegTester final test.log metrics",
    }
