"""Aggregate accuracy across ablation configs.

Scans data/ablation/reasoning_abs/<tag>/*.json for each known tag, computes per-config
overall accuracy + per-video breakdown, and prints a comparison table.
Also compares against the baseline data/reasoning/*.json if present.
"""
import glob
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TAGS = ["30_10", "50_30", "100_60"]


def accuracy_of_dir(d):
    files = sorted(glob.glob(str(PROJECT_ROOT / d / "*.json")))
    per_video = {}
    grand_c = grand_t = 0
    for f in files:
        video = Path(f).stem
        data = json.load(open(f))
        c = t = 0
        for qid, item in data.items():
            rec = item.get("reasoning", {})
            if isinstance(rec, dict) and "evaluate_correct" in rec:
                t += 1
                if rec["evaluate_correct"]:
                    c += 1
        if t:
            per_video[video] = (c, t)
            grand_c += c
            grand_t += t
    return per_video, grand_c, grand_t


def main():
    rows = {}
    for tag in TAGS:
        per_video, gc, gt = accuracy_of_dir(f"data/ablation/reasoning_abs/{tag}")
        rows[tag] = (per_video, gc, gt)
    # baseline (default memorization, no ablation)
    base_pv, base_gc, base_gt = accuracy_of_dir("data/reasoning")

    print("=" * 64)
    print(f"{'config':10s} {'correct/total':>16s} {'accuracy':>10s}")
    print("-" * 64)
    if base_gt:
        print(f"{'baseline':10s} {f'{base_gc}/{base_gt}':>16s} {base_gc/base_gt*100:>9.2f}%")
    for tag in TAGS:
        _, gc, gt = rows[tag]
        if gt:
            print(f"{tag:10s} {f'{gc}/{gt}':>16s} {gc/gt*100:>9.2f}%")
        else:
            print(f"{tag:10s} {'(no results)':>16s}")
    print("=" * 64)

    # Per-video comparison
    all_videos = sorted(set().union(*(rows[t][0].keys() for t in TAGS)) | set(base_pv.keys()))
    if all_videos:
        print("\nPer-video accuracy:")
        header = f"{'video':20s} " + " ".join(f"{t:>8s}" for t in (["baseline"] + TAGS))
        print(header)
        for v in all_videos:
            cells = []
            if v in base_pv:
                c, t = base_pv[v]
                cells.append(f"{c}/{t}")
            else:
                cells.append("-")
            for tag in TAGS:
                pv = rows[tag][0]
                if v in pv:
                    c, t = pv[v]
                    cells.append(f"{c}/{t}")
                else:
                    cells.append("-")
            print(f"{v:20s} " + " ".join(f"{c:>8s}" for c in cells))


if __name__ == "__main__":
    main()
