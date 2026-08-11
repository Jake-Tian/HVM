#!/usr/bin/env python3
"""Benchmark triple extraction across models and output modes.

The benchmark replays behavior lists already stored in data/memorization. It
does not rebuild video memory or graphs. Structured and unstructured runs use
the same prompt; only the OpenAI response-format mechanism differs.
"""

import argparse
import json
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from classes.output_structure import TripleExtraction  # noqa: E402
from utils.prompts import prompt_extract_triples  # noqa: E402


DEFAULT_MODELS = ("gpt-5-mini", "gpt-5.6-luna")
DEFAULT_MODES = ("structured", "unstructured")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare triple-extraction latency and reliability for structured "
            "and unstructured outputs."
        )
    )
    parser.add_argument(
        "--input-glob",
        default="data/memorization/*.json",
        help="Glob for memorization JSON files, relative to the repo root.",
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=DEFAULT_MODES,
        default=list(DEFAULT_MODES),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=24,
        help="Number of clips sampled across the input-length distribution.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Retries after the first failed attempt.",
    )
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to data/triple_benchmarks/<timestamp>.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected cases and call count without using the API.",
    )
    return parser.parse_args()


def resolve_glob(pattern):
    path = Path(pattern)
    if path.is_absolute():
        return sorted(path.parent.glob(path.name))
    return sorted(REPO_ROOT.glob(pattern))


def load_cases(pattern):
    cases = []
    for path in resolve_glob(pattern):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        episodic_memory = payload.get("episodic_memory", payload)
        if not isinstance(episodic_memory, dict):
            continue
        for clip_id, clip in episodic_memory.items():
            if not isinstance(clip, dict):
                continue
            behaviors = clip.get("behaviors") or clip.get("characters_behavior") or []
            if not behaviors:
                continue
            serialized = json.dumps(behaviors, ensure_ascii=False)
            cases.append(
                {
                    "case_id": f"{path.stem}:{clip_id}",
                    "video": path.stem,
                    "clip_id": int(clip_id),
                    "behaviors": behaviors,
                    "behavior_count": len(behaviors),
                    "input_chars": len(serialized),
                }
            )
    if not cases:
        raise ValueError(f"No behavior lists found for input glob: {pattern}")
    return cases


def stratified_sample(cases, sample_size, rng):
    if sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    if sample_size >= len(cases):
        return sorted(cases, key=lambda item: item["input_chars"])

    ordered = sorted(cases, key=lambda item: item["input_chars"])
    selected = []
    for index in range(sample_size):
        start = index * len(ordered) // sample_size
        end = (index + 1) * len(ordered) // sample_size
        selected.append(rng.choice(ordered[start:end]))
    return selected


def build_prompt(behaviors):
    return prompt_extract_triples + "\n" + json.dumps(behaviors, ensure_ascii=False)


def parse_json_object(text):
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def usage_dict(response):
    usage = getattr(response, "usage", None)
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def extract_once(client, model, mode, prompt):
    request_start = time.perf_counter()
    if mode == "structured":
        response = client.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format=TripleExtraction,
        )
        request_latency = time.perf_counter() - request_start
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("Structured response did not contain parsed output")
        parse_latency = 0.0
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        request_latency = time.perf_counter() - request_start
        parse_start = time.perf_counter()
        raw_payload = parse_json_object(response.choices[0].message.content)
        parsed = TripleExtraction.model_validate(raw_payload)
        parse_latency = time.perf_counter() - parse_start

    triples = [
        [triple.source, triple.content, triple.target]
        for triple in parsed.triples
    ]
    return {
        "request_latency_s": request_latency,
        "local_parse_latency_s": parse_latency,
        "triples": triples,
        "usage": usage_dict(response),
    }


def run_case(client, case, model, mode, repeat, retries):
    prompt = build_prompt(case["behaviors"])
    started = time.perf_counter()
    attempts = []
    result = None

    for attempt_index in range(retries + 1):
        try:
            extracted = extract_once(client, model, mode, prompt)
            attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "success": True,
                    "request_latency_s": extracted["request_latency_s"],
                    "local_parse_latency_s": extracted["local_parse_latency_s"],
                }
            )
            result = extracted
            break
        except Exception as error:
            attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "success": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

    triples = result["triples"] if result else []
    null_targets = sum(1 for triple in triples if triple[2] is None)
    return {
        "case_id": case["case_id"],
        "video": case["video"],
        "clip_id": case["clip_id"],
        "behavior_count": case["behavior_count"],
        "input_chars": case["input_chars"],
        "model": model,
        "mode": mode,
        "repeat": repeat,
        "success": result is not None,
        "attempt_count": len(attempts),
        "retried": len(attempts) > 1,
        "wall_latency_s": time.perf_counter() - started,
        "request_latency_s": (
            sum(item.get("request_latency_s", 0.0) for item in attempts)
        ),
        "local_parse_latency_s": (
            sum(item.get("local_parse_latency_s", 0.0) for item in attempts)
        ),
        "triple_count": len(triples),
        "null_target_count": null_targets,
        "triples": triples,
        "usage": result["usage"] if result else usage_dict(None),
        "attempts": attempts,
    }


def percentile(values, percentile_value):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["model"], record["mode"])].append(record)

    groups = []
    for (model, mode), rows in sorted(grouped.items()):
        successes = [row for row in rows if row["success"]]
        latencies = [row["wall_latency_s"] for row in successes]
        request_latencies = [row["request_latency_s"] for row in successes]
        groups.append(
            {
                "model": model,
                "mode": mode,
                "runs": len(rows),
                "successes": len(successes),
                "success_rate": len(successes) / len(rows),
                "retry_rate": sum(row["retried"] for row in rows) / len(rows),
                "wall_latency_mean_s": statistics.mean(latencies) if latencies else None,
                "wall_latency_p50_s": percentile(latencies, 50),
                "wall_latency_p90_s": percentile(latencies, 90),
                "wall_latency_p95_s": percentile(latencies, 95),
                "request_latency_p50_s": percentile(request_latencies, 50),
                "request_latency_p90_s": percentile(request_latencies, 90),
                "total_tokens": sum(row["usage"]["total_tokens"] for row in successes),
                "mean_completion_tokens": (
                    statistics.mean(
                        row["usage"]["completion_tokens"] for row in successes
                    )
                    if successes
                    else None
                ),
                "mean_triple_count": (
                    statistics.mean(row["triple_count"] for row in successes)
                    if successes
                    else None
                ),
            }
        )
    return {"groups": groups}


def write_json(path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    cases = load_cases(args.input_glob)
    selected = stratified_sample(cases, args.sample_size, rng)

    schedule = [
        (case, model, mode, repeat)
        for repeat in range(1, args.repeats + 1)
        for case in selected
        for model in args.models
        for mode in args.modes
    ]
    rng.shuffle(schedule)

    print(
        f"Loaded {len(cases)} cases; selected {len(selected)}; "
        f"scheduled {len(schedule)} API runs."
    )
    print(
        "Selected input chars: "
        f"min={min(case['input_chars'] for case in selected)}, "
        f"median={statistics.median(case['input_chars'] for case in selected):.0f}, "
        f"max={max(case['input_chars'] for case in selected)}"
    )

    if args.dry_run:
        for case in sorted(selected, key=lambda item: item["input_chars"]):
            print(
                f"{case['case_id']:24s} behaviors={case['behavior_count']:2d} "
                f"chars={case['input_chars']:4d}"
            )
        return

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set before running the benchmark")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (
        REPO_ROOT / "data" / "triple_benchmarks" / timestamp
    )
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["selected_cases"] = [case["case_id"] for case in selected]
    config["scheduled_runs"] = len(schedule)
    write_json(output_dir / "config.json", config)

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
        timeout=args.timeout,
    )
    records = []
    results_path = output_dir / "results.jsonl"
    with results_path.open("a", encoding="utf-8") as results_file:
        for index, (case, model, mode, repeat) in enumerate(schedule, start=1):
            print(
                f"[{index}/{len(schedule)}] {case['case_id']} "
                f"model={model} mode={mode} repeat={repeat}",
                flush=True,
            )
            record = run_case(
                client=client,
                case=case,
                model=model,
                mode=mode,
                repeat=repeat,
                retries=args.retries,
            )
            records.append(record)
            results_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            results_file.flush()
            status = "ok" if record["success"] else "FAILED"
            print(
                f"  {status} wall={record['wall_latency_s']:.2f}s "
                f"attempts={record['attempt_count']} triples={record['triple_count']}",
                flush=True,
            )

    summary = summarize(records)
    summary["config"] = config
    write_json(output_dir / "summary.json", summary)

    print(f"Results: {results_path}")
    print(f"Summary: {output_dir / 'summary.json'}")
    for group in summary["groups"]:
        print(
            f"{group['model']:16s} {group['mode']:12s} "
            f"success={group['success_rate']:.1%} "
            f"p50={group['wall_latency_p50_s']} "
            f"p90={group['wall_latency_p90_s']}"
        )


if __name__ == "__main__":
    main()
