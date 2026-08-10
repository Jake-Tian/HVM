#!/usr/bin/env python3
"""Run a two-clip memorization and reasoning smoke test in a temporary workspace."""

import argparse
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_NAME = "bedroom_01"
QUESTION_ID = "bedroom_01_Q09"


def load_question():
    qa_path = PROJECT_ROOT / "data" / "robot.json"
    payload = json.loads(qa_path.read_text(encoding="utf-8"))
    for item in payload[VIDEO_NAME]["qa_list"]:
        if item["question_id"] == QUESTION_ID:
            return item
    raise ValueError(f"Question not found: {QUESTION_ID}")


def first_clip_directories(count=2):
    frames_dir = PROJECT_ROOT / "data" / "frames" / VIDEO_NAME
    clips = sorted(
        (path for path in frames_dir.iterdir() if path.is_dir()),
        key=lambda path: int(path.name),
    )
    if len(clips) < count:
        raise FileNotFoundError(
            f"Expected at least {count} clips in {frames_dir}, found {len(clips)}"
        )
    return clips[:count]


def prepare_workspace(workspace):
    target = workspace / "data" / "frames" / VIDEO_NAME
    target.mkdir(parents=True)
    clips = first_clip_directories()
    for clip in clips:
        (target / clip.name).symlink_to(clip.resolve(), target_is_directory=True)
    return clips


def validate_memorization(workspace, episodic_memory):
    clip_ids = {int(clip_id) for clip_id in episodic_memory}
    if clip_ids != {1, 2}:
        raise AssertionError(f"Expected memorized clips {{1, 2}}, got {clip_ids}")

    graph_path = workspace / "data" / "graphs" / f"{VIDEO_NAME}.pkl"
    checkpoint_path = (
        workspace / "data" / "graphs" / f"{VIDEO_NAME}_preabstraction.pkl"
    )
    memory_path = workspace / "data" / "memorization" / f"{VIDEO_NAME}.json"
    for path in (graph_path, checkpoint_path, memory_path):
        if not path.is_file():
            raise AssertionError(f"Expected output was not created: {path}")

    saved_memory = json.loads(memory_path.read_text(encoding="utf-8"))
    if set(saved_memory.get("episodic_memory", {})) != {"1", "2"}:
        raise AssertionError("Saved episodic memory does not contain exactly two clips")

    with graph_path.open("rb") as file:
        graph = pickle.load(file)
    if not graph.edges:
        raise AssertionError("Memorization produced an empty graph")
    return graph


def run_smoke_test(preflight_only=False):
    question = load_question()
    with tempfile.TemporaryDirectory(prefix="hvm_two_clip_smoke_") as temp_dir:
        workspace = Path(temp_dir)
        clips = prepare_workspace(workspace)
        print(f"Workspace: {workspace}")
        print(f"Video: {VIDEO_NAME}, clips: {[clip.name for clip in clips]}")
        print(f"Question: {question['question']}")
        print(f"Reference answer: {question['answer']}")

        if preflight_only:
            print("Preflight passed. No API calls were made.")
            return

        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is required for the smoke test")

        sys.path.insert(0, str(PROJECT_ROOT))
        from process_full_video import process_full_video
        from reason import reason

        previous_cwd = Path.cwd()
        try:
            os.chdir(workspace)
            _, episodic_memory, token_summary = process_full_video(VIDEO_NAME)
            graph = validate_memorization(workspace, episodic_memory)
            result = reason(graph, VIDEO_NAME, question["question"])
        finally:
            os.chdir(previous_cwd)

        answer = str(result.get("final_answer", "")).strip()
        if not answer or answer == "Could not determine the answer.":
            raise AssertionError(f"Reasoning returned no usable answer: {answer!r}")

        print("\nSmoke test passed.")
        print(f"Memorized clips: {sorted(int(key) for key in episodic_memory)}")
        print(f"Graph edges: {len(graph.edges)}")
        print(f"Memorization tokens: {token_summary.get('total', 0)}")
        print(f"Reasoning tokens: {result.get('token_summaries', {}).get('total', 0)}")
        print(f"Answer: {answer}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate inputs and temporary isolation without calling any model.",
    )
    args = parser.parse_args()
    run_smoke_test(preflight_only=args.preflight_only)


if __name__ == "__main__":
    main()
