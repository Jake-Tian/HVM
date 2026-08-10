"""Extract an ablation's results from a reason_summary_*.json and write them as
per-video JSON files in the same format as data/reasoning/*.json.

This lets us reuse scripts/judge_compare.py and scripts/analyze_judge_agreement.py
on ablation runs (e.g. no_video_rewatch) by pointing them at the output directory.

Usage:
    python scripts/extract_ablation.py \
        --source data/reason_summary_gpt.json \
        --ablation no_video_rewatch \
        --output data/ablation/reasoning_nvr_gpt5
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="Source reason_summary_*.json file")
    ap.add_argument("--ablation", required=True,
                    help="Ablation key to extract (e.g. no_video_rewatch, k30, no_allocation)")
    ap.add_argument("--output", required=True,
                    help="Output directory (per-video JSON files will be written here)")
    args = ap.parse_args()

    src_path = PROJECT_ROOT / args.source
    out_dir = PROJECT_ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n_videos = 0
    n_questions = 0
    n_with_eval = 0
    n_correct = 0
    n_skipped = 0

    for video, vq in data.items():
        if not isinstance(vq, dict) or not vq:
            continue
        out = {}
        for qid, q in vq.items():
            n_questions += 1
            abls = q.get("ablations", {})
            if not isinstance(abls, dict):
                n_skipped += 1
                continue
            entry = abls.get(args.ablation)
            if not isinstance(entry, dict):
                n_skipped += 1
                continue
            # Reconstruct the per-question record in the data/reasoning/*.json schema.
            out[qid] = {
                "question": q.get("question", ""),
                "ground_truth_answer": q.get("ground_truth_answer", ""),
                "reasoning": entry,
                "timestamp": q.get("timestamp"),
                "type": q.get("type"),
            }
            if "evaluate_correct" in entry:
                n_with_eval += 1
                if entry["evaluate_correct"]:
                    n_correct += 1

        if out:
            with open(out_dir / f"{video}.json", "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            n_videos += 1

    print(f"Source: {src_path}")
    print(f"Ablation: {args.ablation}")
    print(f"Output dir: {out_dir}")
    print(f"Videos written: {n_videos}")
    print(f"Questions written: {n_with_eval + (n_questions - n_with_eval - n_skipped)}")
    print(f"  with evaluate_correct: {n_with_eval}")
    print(f"  correct: {n_correct} "
          f"= {n_correct / n_with_eval * 100:.2f}%" if n_with_eval else "")
    print(f"  skipped (no ablation entry): {n_skipped}")


if __name__ == "__main__":
    main()
