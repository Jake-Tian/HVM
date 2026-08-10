
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

THRESHOLD_REASONING_DIRS = {
    "30_10": "data/ablation/reasoning_abs/30_10",
    "50_30": "data/ablation/reasoning_abs/50_30",
    "100_60": "data/ablation/reasoning_abs/100_60",
}


def load_json(path):
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_memorization(memorization_dir):
    memorization_dir = Path(memorization_dir)
    episodic_memory = {}
    memorization_tokens = {}

    for file_path in sorted(memorization_dir.glob("*.json")):
        video_name = file_path.stem
        try:
            payload = load_json(file_path)
        except Exception as exc:
            print(f"Warning: failed to read memorization file {file_path}: {exc}")
            continue

        if isinstance(payload, dict):
            episodic_memory[video_name] = payload.get("episodic_memory", {})
            memorization_tokens[video_name] = payload.get("memory_token_summaries", {})
        else:
            episodic_memory[video_name] = {}
            memorization_tokens[video_name] = {}

    return episodic_memory, memorization_tokens


def collect_reasoning(reasoning_dir):
    reasoning_dir = Path(reasoning_dir)
    reason_summary = {}

    for file_path in sorted(reasoning_dir.glob("*.json")):
        video_name = file_path.stem
        try:
            payload = load_json(file_path)
        except Exception as exc:
            print(f"Warning: failed to read reasoning file {file_path}: {exc}")
            continue

        # Keep question-id keyed dictionary as-is.
        if isinstance(payload, dict):
            reason_summary[video_name] = payload
        else:
            reason_summary[video_name] = {}

    return reason_summary


def merge_threshold_reasoning(reason_summary, threshold_summaries):
    for tag, summary in threshold_summaries.items():
        for video_name, questions in summary.items():
            if not isinstance(questions, dict):
                continue
            video_results = reason_summary.setdefault(video_name, {})

            for question_id, variant_item in questions.items():
                if not isinstance(variant_item, dict):
                    continue

                base_item = video_results.get(question_id)
                if not isinstance(base_item, dict):
                    base_item = {
                        key: value
                        for key, value in variant_item.items()
                        if key != "reasoning"
                    }
                    video_results[question_id] = base_item

                threshold_results = base_item.setdefault(
                    "threshold_abstraction", {}
                )
                if not isinstance(threshold_results, dict):
                    threshold_results = {}
                    base_item["threshold_abstraction"] = threshold_results
                threshold_results[tag] = variant_item.get("reasoning", {})

    return reason_summary


def reasoning_coverage(summary):
    videos = 0
    questions = 0
    for video_results in summary.values():
        if not isinstance(video_results, dict):
            continue
        videos += 1
        questions += sum(
            1 for item in video_results.values()
            if isinstance(item, dict)
            and isinstance(item.get("reasoning"), dict)
        )
    return videos, questions


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():

    episodic_memory, memorization_tokens = collect_memorization("data/memorization")
    baseline_summary = collect_reasoning("data/reasoning")
    baseline_coverage = reasoning_coverage(baseline_summary)
    threshold_summaries = {
        tag: collect_reasoning(directory)
        for tag, directory in THRESHOLD_REASONING_DIRS.items()
    }
    reason_summary = merge_threshold_reasoning(
        baseline_summary,
        threshold_summaries,
    )

    episodic_path = "data/episodic_memory/episodic_memory.json"
    reason_path = "data/reason_summaries/reason_summary.json"
    tokens_path = "data/analysis/memorization_tokens.json"

    write_json(episodic_path, episodic_memory)
    write_json(reason_path, reason_summary)
    write_json(tokens_path, memorization_tokens)

    print(f"Saved: {episodic_path}")
    print(f"Saved: {reason_path}")
    print(f"Saved: {tokens_path}")
    baseline_videos, baseline_questions = baseline_coverage
    print(
        f"Coverage: baseline videos={baseline_videos}, "
        f"questions={baseline_questions}"
    )
    for tag, summary in threshold_summaries.items():
        videos, questions = reasoning_coverage(summary)
        print(
            f"Coverage: {tag} videos={videos}, questions={questions}"
        )
    print(
        f"Summary: memorization videos={len(episodic_memory)}, "
        f"combined reasoning videos={len(reason_summary)}"
    )


if __name__ == "__main__":
    main()
