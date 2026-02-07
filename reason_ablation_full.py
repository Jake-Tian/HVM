#!/usr/bin/env python3
"""
Run ablation studies: original, no_rewatch, no_highlevel.

Runs all three variants sequentially for each question.
Saves results to:
- data/results/results_original.json
- data/results/results_no_rewatch.json
- data/results/results_no_highlevel.json

Prints token usage summary per variant and comparison.
"""

import argparse
import json
import pickle
from pathlib import Path

from reason_ablation import reason_original, reason_no_rewatch, reason_no_highlevel
from reason_full import evaluate_answer, load_questions
from utils.token_monitor import TokenMonitor

def get_available_videos(semantic_memory_dir: str = "data/semantic_memory") -> list:
    """Return list of video names that have .pkl files in semantic memory."""
    path = Path(semantic_memory_dir)
    if not path.exists():
        return []
    return sorted(p.stem for p in path.glob("*.pkl"))


def run_ablation(
    output_dir: str = "data/results",
    questions_path: str = "data/questions/robot.json",
    semantic_memory_dir: str = "data/semantic_memory",
    available_videos: list = None,
):
    """
    Run all three ablation variants sequentially.
    Returns dict with results and token summaries per variant.
    """
    if available_videos is None:
        available_videos = DEFAULT_VIDEOS

    semantic_path = Path(semantic_memory_dir)
    questions_data = load_questions(questions_path)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Build question list
    all_questions = []
    for video_name, video_data in questions_data.items():
        if video_name in available_videos:
            for qa in video_data.get("qa_list", []):
                qa = dict(qa)
                qa["video_name"] = video_name
                all_questions.append(qa)

    print(f"Total questions: {len(all_questions)} across {len(available_videos)} videos")
    print()

    variants = [
        ("original", reason_original, "results_original.json"),
        ("no_rewatch", reason_no_rewatch, "results_no_rewatch.json"),
        ("no_highlevel", reason_no_highlevel, "results_no_highlevel.json"),
    ]

    all_results = {}
    token_summaries = {}

    for variant_name, reason_fn, filename in variants:
        print("=" * 60)
        print(f"Running ablation: {variant_name}")
        print("=" * 60)

        token_monitor = TokenMonitor()
        results = {}
        existing = {}
        result_path = out_path / filename
        if result_path.exists():
            try:
                with open(result_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except json.JSONDecodeError:
                pass

        for i, qa in enumerate(all_questions, 1):
            question_id = qa["question_id"]
            question = qa["question"]
            video_name = qa["video_name"]
            ground_truth = qa["answer"]
            reasoning = qa.get("reasoning", "")
            timestamp = qa.get("timestamp", "")
            qa_type = qa.get("type", [])
            before_clip = qa.get("before_clip", None)

            print(f"\n[{variant_name}] {i}/{len(all_questions)}: {question_id}")

            graph_path = semantic_path / f"{video_name}.pkl"
            if not graph_path.exists():
                print(f"  Skipping: graph not found for {video_name}")
                continue

            try:
                with open(graph_path, "rb") as f:
                    graph = pickle.load(f)

                reason_result = reason_fn(question, graph, video_name, token_monitor=token_monitor)
                predicted = reason_result.get("final_answer", reason_result.get("answer", ""))

                is_correct = evaluate_answer(question, ground_truth, predicted, token_monitor=token_monitor)

                reason_result["evaluator_correct"] = is_correct
                reason_result["ground_truth_answer"] = ground_truth
                reason_result["reasoning"] = reasoning
                reason_result["timestamp"] = timestamp
                reason_result["type"] = qa_type
                reason_result["before_clip"] = before_clip

                results[question_id] = reason_result
                print(f"  Predicted: {predicted[:60]}...")
                print(f"  Correct: {is_correct}")

            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
                results[question_id] = {
                    "error": str(e),
                    "video_name": video_name,
                    "question": question,
                    "ground_truth_answer": ground_truth,
                    "reasoning": reasoning,
                    "timestamp": timestamp,
                    "type": qa_type,
                    "before_clip": before_clip,
                    "evaluator_correct": False,
                }

        existing.update(results)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {result_path}")

        token_summaries[variant_name] = token_monitor.to_dict()
        print(f"\n{token_monitor.summary()}")
        all_results[variant_name] = existing

    # Print comparison
    print("\n" + "=" * 60)
    print("TOKEN USAGE COMPARISON")
    print("=" * 60)

    for vname, tdict in token_summaries.items():
        tot = tdict["total_tokens"]
        text = tdict["text_llm"]["total_tokens"]
        vision = tdict["vision_llm"]["total_tokens"]
        watch_calls = tdict["video_watch_calls"]
        print(f"\n{vname}:")
        print(f"  Total tokens: {tot} (text: {text}, vision: {vision})")
        print(f"  Video watch API calls: {watch_calls}")

    # Difference: no_rewatch vs original (video-watch saves tokens)
    if "original" in token_summaries and "no_rewatch" in token_summaries:
        orig = token_summaries["original"]["total_tokens"]
        no_rw = token_summaries["no_rewatch"]["total_tokens"]
        diff = orig - no_rw
        print(f"\nToken difference (original - no_rewatch): {diff}")
        print(f"  Video-watch adds ~{diff} tokens when clips are requested")

    # Accuracy summary
    print("\n" + "=" * 60)
    print("ACCURACY SUMMARY")
    print("=" * 60)
    for vname, res in all_results.items():
        total = len([r for r in res.values() if isinstance(r, dict) and "error" not in r])
        correct = sum(1 for r in res.values() if isinstance(r, dict) and r.get("evaluator_correct", False))
        acc = (correct / total * 100) if total else 0
        print(f"  {vname}: {correct}/{total} ({acc:.1f}%)")

    # Save token summary
    token_path = out_path / "token_summary_ablation.json"
    with open(token_path, "w", encoding="utf-8") as f:
        json.dump(token_summaries, f, indent=2)
    print(f"\nToken summary saved to {token_path}")

    return all_results, token_summaries


if __name__ == "__main__":

    # videos = get_available_videos()
    videos = ["living_room_03"]

    run_ablation(
        output_dir="data/results",
        semantic_memory_dir="data/semantic_memory",
        available_videos=videos,
    )
