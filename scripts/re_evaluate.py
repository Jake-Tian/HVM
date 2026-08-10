"""Re-run only the LLM judge (evaluate_answer) on an existing reasoning JSON,
without re-running the agent. Saves tokens by reusing the stored final_answer.

Usage:
    python scripts/re_evaluate.py bedroom_01
    python scripts/re_evaluate.py bedroom_01 --in data/reasoning/bedroom_01.json --out data/reasoning/bedroom_01.json

Each entry's reasoning.final_answer is fed back to evaluate_answer(question,
ground_truth_answer, final_answer) and reasoning.evaluate_correct is updated.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from reason import evaluate_answer


def main():
    parser = argparse.ArgumentParser(description="Re-run only the judge on saved reasoning results.")
    parser.add_argument("video", help="Video name (or pass --in explicitly)")
    parser.add_argument("--in", dest="in_path", default=None,
                        help="Input reasoning JSON (default: data/reasoning/<video>.json)")
    parser.add_argument("--out", dest="out_path", default=None,
                        help="Output JSON (default: same as input, overwritten in place)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print changes without writing the file")
    args = parser.parse_args()

    in_path = Path(args.in_path) if args.in_path else PROJECT_ROOT / f"data/reasoning/{args.video}.json"
    if not in_path.is_absolute():
        in_path = PROJECT_ROOT / in_path
    out_path = Path(args.out_path) if args.out_path else in_path
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    if not in_path.exists():
        print(f"✗ Input not found: {in_path}")
        sys.exit(1)

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    changed = []
    n_true_before = 0
    n_true_after = 0
    total = 0

    for qid, entry in data.items():
        if not isinstance(entry, dict):
            continue
        question = entry.get("question", "")
        gt = entry.get("ground_truth_answer", "")
        reasoning = entry.get("reasoning")
        if not isinstance(reasoning, dict):
            continue
        final_ans = reasoning.get("final_answer", "")
        if not isinstance(final_ans, str) or not final_ans.strip():
            continue
        before = reasoning.get("evaluate_correct")
        total += 1
        if before:
            n_true_before += 1

        try:
            correct = evaluate_answer(question, gt, final_ans)
        except Exception as e:
            print(f"✗ Judge error for {qid}: {e}")
            correct = False

        reasoning["evaluate_correct"] = correct
        if correct:
            n_true_after += 1
        if before != correct:
            changed.append((qid, before, correct, final_ans[:80]))

    if not args.dry_run:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✓ Wrote {out_path}")
    else:
        print("(dry run — no file written)")

    print(f"\nTotal questions judged: {total}")
    print(f"evaluate_correct: {n_true_before} → {n_true_after}")
    print(f"Changed: {len(changed)}")
    for qid, b, a, ans in changed:
        arrow = "False→True" if (not b and a) else "True→False"
        print(f"  {qid}: {arrow}  | {ans}")
        if not b and a:
            print(f"      ↑ was wrongly False, now corrected to True")


if __name__ == "__main__":
    main()
