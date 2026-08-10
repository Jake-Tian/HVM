"""Analyze agreement between LLM judges produced by scripts/judge_compare.py.

Reports per-judge accuracy, pairwise agreement rate, Cohen's kappa, and a
breakdown by question type. Used to address reviewer R1 W5/Q1.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def cohen_kappa(a, b):
    """Compute Cohen's kappa over two parallel lists of booleans (None = missing)."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for x, y in pairs if x == y) / n
    pa = sum(1 for x, _ in pairs if x) / n
    pb = sum(1 for _, y in pairs if y) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/judge_comparison/judge_comparison.json")
    ap.add_argument("--models", nargs="+", default=["gpt-4o", "gpt-5.2"],
                    help="Judge model names whose <model>_judge fields are compared. "
                         "Use 'gpt5' to reference the existing gpt5_judge field "
                         "(the judge verdict already stored in the reasoning JSON).")
    ap.add_argument("--existing-judge-key", default="gpt5_judge",
                    help="Field name in the input JSON that holds the existing judge "
                         "verdict (default: gpt5_judge). When 'gpt5' is in --models, "
                         "this field is used.")
    args = ap.parse_args()

    in_path = PROJECT_ROOT / args.input
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = list(data.values())
    print(f"Loaded {len(entries)} entries from {in_path}\n")

    # Map model name -> field name in the JSON. 'gpt5' is a virtual name that
    # refers to the existing judge verdict (default: gpt5_judge).
    def field_for(model):
        if model == "gpt5":
            return args.existing_judge_key
        return f"{model}_judge"

    # Per-judge accuracy.
    print("== Per-judge accuracy ==")
    for m in args.models:
        verdicts = [e.get(field_for(m)) for e in entries]
        valid = [v for v in verdicts if v is not None]
        if not valid:
            print(f"  {m}: no verdicts")
            continue
        correct = sum(1 for v in valid if v)
        print(f"  {m}: {correct}/{len(valid)} = {correct / len(valid) * 100:.2f}% "
              f"({len(verdicts) - len(valid)} missing)")
    print()

    # Pairwise agreement.
    print("== Pairwise agreement ==")
    for i, m1 in enumerate(args.models):
        for m2 in args.models[i + 1:]:
            v1 = [e.get(field_for(m1)) for e in entries]
            v2 = [e.get(field_for(m2)) for e in entries]
            pairs = [(a, b) for a, b in zip(v1, v2) if a is not None and b is not None]
            if not pairs:
                continue
            agree = sum(1 for a, b in pairs if a == b)
            both_correct = sum(1 for a, b in pairs if a and b)
            both_wrong = sum(1 for a, b in pairs if (a is False) and (b is False))
            m1_only = sum(1 for a, b in pairs if a and not b)
            m2_only = sum(1 for a, b in pairs if b and not a)
            kappa = cohen_kappa(v1, v2)
            print(f"  {m1} vs {m2}:")
            print(f"    agreement: {agree}/{len(pairs)} = {agree / len(pairs) * 100:.2f}%")
            print(f"    Cohen's kappa: {kappa:.4f}")
            print(f"    both correct: {both_correct}, both wrong: {both_wrong}, "
                  f"{m1}-only: {m1_only}, {m2}-only: {m2_only}")
    print()

    # Breakdown by question type.
    print("== Agreement by question type ==")
    for i, m1 in enumerate(args.models):
        for m2 in args.models[i + 1:]:
            print(f"\n  -- {m1} vs {m2} --")
            by_type = defaultdict(lambda: {"agree": 0, "total": 0,
                                           "m1_correct": 0, "m2_correct": 0,
                                           "valid": 0})
            for e in entries:
                types = e.get("types")
                if isinstance(types, str):
                    types = [types]
                if not types:
                    types = ["(none)"]
                v1 = e.get(field_for(m1))
                v2 = e.get(field_for(m2))
                if v1 is None or v2 is None:
                    continue
                for t in types:
                    by_type[t]["total"] += 1
                    by_type[t]["valid"] += 1
                    if v1 == v2:
                        by_type[t]["agree"] += 1
                    if v1:
                        by_type[t]["m1_correct"] += 1
                    if v2:
                        by_type[t]["m2_correct"] += 1
            print(f"    {'type':<35} {'agree/total':<15} {'agree%':<8} "
                  f"{m1 + '_acc':<10} {m2 + '_acc':<10}")
            for t, s in sorted(by_type.items()):
                agree_pct = s["agree"] / s["valid"] * 100 if s["valid"] else 0
                m1_acc = s["m1_correct"] / s["valid"] * 100 if s["valid"] else 0
                m2_acc = s["m2_correct"] / s["valid"] * 100 if s["valid"] else 0
                print(f"    {t:<35} {s['agree']}/{s['valid']:<10} "
                      f"{agree_pct:<8.2f} {m1_acc:<10.2f} {m2_acc:<10.2f}")


if __name__ == "__main__":
    main()
