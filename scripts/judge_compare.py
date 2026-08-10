"""Re-judge existing reasoning outputs with multiple LLM judges using the same prompt.

This script addresses reviewer concern R1 W5/Q1 (cross-judge validation for the
LLM-as-judge evaluation on M3-Bench-Robot). It loads the (question, ground_truth,
predicted_answer) triples already saved in data/reasoning/*.json and re-evaluates
each triple with gpt-4o and gpt-5.2 using the same prompt_agent_verify_answer_referencing
prompt that the original pipeline uses.

Usage:
    python scripts/judge_compare.py --models gpt-4o gpt-5.2 \
        --concurrency 8 --output data/judge_comparison/judge_comparison.json

Resume support: already-judged entries in the output file are skipped, so the
script can be re-run safely after interruptions.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Make the project root importable when running from scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.llm_gpt import generate_text_response  # noqa: E402
from utils.prompts import prompt_agent_verify_answer_referencing  # noqa: E402


REASONING_DIR = PROJECT_ROOT / "data" / "reasoning"


def parse_judge_response(text):
    """Parse a Yes/No judge response. Returns True / False / None (ambiguous)."""
    if not text:
        return None
    s = text.strip().upper()
    if s.startswith("YES"):
        return True
    if s.startswith("NO"):
        return False
    return None


def load_triples(reasoning_dir):
    """Return a list of dicts: {key, video, question, ground_truth, predicted, gpt5_judge}."""
    triples = []
    for fp in sorted(reasoning_dir.glob("*.json")):
        video = fp.stem
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        for qid, q in data.items():
            reasoning = q.get("reasoning", {})
            if not isinstance(reasoning, dict):
                continue
            predicted = reasoning.get("final_answer")
            if predicted is None:
                continue
            triples.append({
                "key": f"{video}:{qid}",
                "video": video,
                "question_id": qid,
                "question": q.get("question", ""),
                "ground_truth": q.get("ground_truth_answer", ""),
                "predicted": predicted,
                "gpt5_judge": reasoning.get("evaluate_correct"),
                "types": q.get("type"),
            })
    return triples


def judge_one(triple, model, base_url, api_key, max_retries=3):
    prompt = prompt_agent_verify_answer_referencing.format(
        question=triple["question"],
        ground_truth_answer=triple["ground_truth"],
        agent_answer=triple["predicted"],
    )
    last_err = None
    for attempt in range(max_retries):
        try:
            text, _ = generate_text_response(
                prompt, model=model, base_url=base_url, api_key=api_key
            )
            verdict = parse_judge_response(text)
            if verdict is not None:
                return verdict, text.strip()
            last_err = f"ambiguous response: {text!r}"
        except Exception as e:
            last_err = str(e)
    return None, f"FAILED: {last_err}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gpt-4o", "gpt-5.2"])
    ap.add_argument("--output", default="data/judge_comparison/judge_comparison.json")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of triples (for debugging).")
    ap.add_argument("--reasoning-dir", default="data/reasoning",
                    help="Directory of per-video reasoning JSON files to re-judge "
                         "(default: data/reasoning).")
    ap.add_argument("--base-url", default=None,
                    help="Override OPENAI_BASE_URL for the judge calls.")
    ap.add_argument("--api-key", default=None,
                    help="Override OPENAI_API_KEY for the judge calls.")
    args = ap.parse_args()

    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY env var not set and --api-key not provided.")
        sys.exit(1)

    reasoning_dir = PROJECT_ROOT / args.reasoning_dir
    triples = load_triples(reasoning_dir)
    if args.limit:
        triples = triples[: args.limit]
    print(f"Loaded {len(triples)} triples from {reasoning_dir}")

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: load existing results.
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"Resuming with {len(results)} already-judged entries.")
    else:
        results = {}

    pending = [t for t in triples if t["key"] not in results]
    print(f"Pending: {len(pending)} triples × {len(args.models)} judges "
          f"= {len(pending) * len(args.models)} calls")

    # Initialize placeholders so partial writes are still valid.
    for t in pending:
        key = t["key"]
        if key not in results:
            results[key] = {
                "video": t["video"],
                "question_id": t["question_id"],
                "question": t["question"],
                "ground_truth": t["ground_truth"],
                "predicted": t["predicted"],
                "gpt5_judge": t["gpt5_judge"],
                "types": t["types"],
            }
            for m in args.models:
                results[key][f"{m}_judge"] = None
                results[key][f"{m}_raw"] = None

    # Build a flat list of (key, model) tasks for pending triples.
    tasks = []
    for t in pending:
        for m in args.models:
            tasks.append((t, m))

    done = 0
    failed = 0
    import time
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(judge_one, t, m, base_url, api_key): (t["key"], m)
                   for (t, m) in tasks}
        for fut in as_completed(futures):
            key, model = futures[fut]
            try:
                verdict, raw = fut.result()
            except Exception as e:
                verdict, raw = None, f"EXC: {e}"
            results[key][f"{model}_judge"] = verdict
            results[key][f"{model}_raw"] = raw
            done += 1
            if verdict is None:
                failed += 1
            if done % 25 == 0 or done == len(tasks):
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (len(tasks) - done) / rate if rate > 0 else 0
                pct = done / len(tasks) * 100
                print(f"[{pct:5.1f}%] {done}/{len(tasks)} calls "
                      f"({failed} failed/ambiguous) "
                      f"elapsed {elapsed:.0f}s, ETA {remaining:.0f}s "
                      f"({rate:.1f} calls/s)")
                # Checkpoint periodically.
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

    # Final write.
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone. {done} calls, {failed} failed/ambiguous.")
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
