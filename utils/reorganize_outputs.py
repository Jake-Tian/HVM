
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():

    episodic_memory, memorization_tokens = collect_memorization("data/memorization")
    reason_summary = collect_reasoning("data/reasoning")

    episodic_path = "data/episodic_memory.json"
    reason_path = "data/reason_summary.json"
    tokens_path = "data/memorization_tokens.json"

    write_json(episodic_path, episodic_memory)
    write_json(reason_path, reason_summary)
    write_json(tokens_path, memorization_tokens)

    print(f"Saved: {episodic_path}")
    print(f"Saved: {reason_path}")
    print(f"Saved: {tokens_path}")
    print(
        f"Summary: memorization videos={len(episodic_memory)}, "
        f"reasoning videos={len(reason_summary)}"
    )


if __name__ == "__main__":
    main()
