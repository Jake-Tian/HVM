"""Summarize HVM LLM/MLLM token usage and optionally estimate API cost.

Embedding calls are intentionally excluded from this report.
"""

import argparse
import json
from collections import defaultdict


AGGREGATE_KEYS = {
    "input", "output", "cached_input", "reasoning", "calls",
    "total", "by_model", "details",
}


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Warning: could not load {path}: {exc}")
        return {}


def total_from_summary(summary):
    """Read both the new rich format and the old stage-total format."""
    if not isinstance(summary, dict):
        return 0
    if "total" in summary:
        return int(summary.get("total", 0) or 0)
    return sum(
        int(value or 0)
        for key, value in summary.items()
        if key not in AGGREGATE_KEYS and isinstance(value, (int, float))
    )


def merge_breakdown(target, summary):
    if not isinstance(summary, dict):
        return
    for field in ("input", "output", "cached_input", "calls"):
        target[field] += int(summary.get(field, 0) or 0)
    target["reasoning_tokens"] += int(summary.get("reasoning", 0) or 0)
    for model, usage in (summary.get("by_model", {}) or {}).items():
        for field in ("input", "output", "cached_input", "reasoning", "calls", "total"):
            target["by_model"][model][field] += int(usage.get(field, 0) or 0)


def parse_prices(specs):
    """Parse MODEL=INPUT_PRICE,OUTPUT_PRICE (USD per one million tokens)."""
    prices = {}
    for spec in specs:
        try:
            model, rates = spec.split("=", 1)
            input_price, output_price = rates.split(",", 1)
            prices[model] = (float(input_price), float(output_price))
        except ValueError as exc:
            raise SystemExit(
                f"Invalid --price {spec!r}; expected MODEL=INPUT_PRICE,OUTPUT_PRICE"
            ) from exc
    return prices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--memorization",
        default="data/analysis/memorization_tokens.json",
    )
    parser.add_argument(
        "--reasoning",
        default="data/reason_summaries/reason_summary.json",
    )
    parser.add_argument(
        "--episodic-memory",
        default="data/episodic_memory/episodic_memory.json",
    )
    parser.add_argument(
        "--price",
        action="append",
        default=[],
        metavar="MODEL=INPUT,OUTPUT",
        help="USD per 1M tokens; repeat for each provider:model key",
    )
    args = parser.parse_args()

    mem_tokens = load_json(args.memorization)
    reason_data = load_json(args.reasoning)
    episodic_data = load_json(args.episodic_memory)

    totals = {
        "memorization": 0,
        "reasoning": 0,
        "input": 0,
        "output": 0,
        "cached_input": 0,
        "reasoning_tokens": 0,
        "calls": 0,
        "by_model": defaultdict(lambda: defaultdict(int)),
    }
    question_tokens = []
    videos = set(mem_tokens) | set(reason_data)

    for video_id in videos:
        mem_summary = mem_tokens.get(video_id, {})
        totals["memorization"] += total_from_summary(mem_summary)
        merge_breakdown(totals, mem_summary)

        for question_data in reason_data.get(video_id, {}).values():
            reasoning = (
                question_data.get("reasoning", {})
                if isinstance(question_data, dict)
                else {}
            )
            summary = (
                reasoning.get("token_summaries", {})
                if isinstance(reasoning, dict)
                else {}
            )
            q_total = total_from_summary(summary)
            totals["reasoning"] += q_total
            question_tokens.append(q_total)
            merge_breakdown(totals, summary)

    total_clips = sum(len(episodic_data.get(video_id, {})) for video_id in videos)
    duration_minutes = total_clips * 0.5
    total = totals["memorization"] + totals["reasoning"]
    classified = totals["input"] + totals["output"]
    unclassified = max(0, total - classified)

    print("=== Overall LLM/MLLM Summary (embeddings excluded) ===")
    print(f"Videos: {len(videos):,}")
    print(f"Questions: {len(question_tokens):,}")
    print(f"Estimated video duration: {duration_minutes:,.1f} minutes")
    print(f"Total tokens: {total:,}")
    print(f"  Memorization: {totals['memorization']:,}")
    print(f"  Reasoning: {totals['reasoning']:,}")
    print(f"  Input: {totals['input']:,}")
    print(f"  Output: {totals['output']:,}")
    print(f"  Cached input (subset of input): {totals['cached_input']:,}")
    print(f"  Reasoning tokens (subset of output): {totals['reasoning_tokens']:,}")
    print(f"  API calls: {totals['calls']:,}")
    if unclassified:
        print(f"  Unclassified legacy tokens: {unclassified:,}")

    if duration_minutes:
        print("\n=== Per Minute ===")
        print(f"Total: {total / duration_minutes:,.1f}")
        print(f"Input: {totals['input'] / duration_minutes:,.1f}")
        print(f"Output: {totals['output'] / duration_minutes:,.1f}")
    if question_tokens:
        print("\n=== Per Question ===")
        print(f"Reasoning: {totals['reasoning'] / len(question_tokens):,.1f}")
        print(f"Amortized total: {total / len(question_tokens):,.1f}")

    if totals["by_model"]:
        print("\n=== By Model ===")
        for model, usage in sorted(totals["by_model"].items()):
            print(
                f"{model}: input={usage['input']:,}, output={usage['output']:,}, "
                f"total={usage['total']:,}, calls={usage['calls']:,}"
            )

    prices = parse_prices(args.price)
    if prices:
        print("\n=== Estimated Cost (USD) ===")
        grand_cost = 0.0
        for model, (input_price, output_price) in prices.items():
            usage = totals["by_model"].get(model, {})
            cost = (
                usage.get("input", 0) * input_price
                + usage.get("output", 0) * output_price
            ) / 1_000_000
            grand_cost += cost
            print(f"{model}: ${cost:,.4f}")
        print(f"Total priced cost: ${grand_cost:,.4f}")


if __name__ == "__main__":
    main()
