#!/usr/bin/env bash
set -euo pipefail

# Run full pipeline sequentially per video to conserve storage:
# 1) Download MP4
# 2) Add subtitles + extract frames
# 3) Build graph memory
# 4) Answer questions with ablation (original, no_rewatch, no_highlevel) → results_*.json
# 5) Cleanup MP4 and frames

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

cleanup_video() {
  local video_name="$1"
  rm -f "data/videos/${video_name}.mp4"
  rm -rf "data/frames/${video_name}"
}

if [[ "$#" -gt 0 ]]; then
  VIDEOS=("$@")
else
  if [[ ! -f "video_list.txt" ]]; then
    echo "video_list.txt not found. Pass video names as arguments."
    exit 1
  fi
  mapfile -t VIDEOS < "video_list.txt"
fi

for video in "${VIDEOS[@]}"; do
  if [[ -z "$video" ]]; then
    continue
  fi

  echo ""
  echo "============================================================"
  echo "Processing video: ${video}"
  echo "============================================================"

  # Step 1: Download video
  if ! python3 preprocessing/download_hf_videos.py "$video"; then
    echo "✗ Download failed for ${video}"
    cleanup_video "$video"
    continue
  fi

  # Step 2: Add subtitles + extract frames
  if [[ ! -f "data/subtitles/robot/${video}.srt" ]]; then
    echo "✗ Subtitle file missing for ${video}: data/subtitles/robot/${video}.srt"
    cleanup_video "$video"
    continue
  fi

  if ! python3 preprocessing/add_subtitles_and_extract_frames.py "$video"; then
    echo "✗ Frame extraction failed for ${video}"
    cleanup_video "$video"
    continue
  fi

  # Step 3: Build graph memory
  if python3 - <<PY
from pathlib import Path
from process_full_video import process_full_video

video_name = "${video}"
frames_dir = Path(f"data/frames/{video_name}")
if not frames_dir.exists():
    raise SystemExit(f"Frames directory not found: {frames_dir}")

process_full_video(frames_dir)
print(f"✓ Graph memory built for {video_name}")
PY
  then
    : # success
  else
    echo "✗ Graph processing failed for ${video}"
    cleanup_video "$video"
    continue
  fi

  # Step 4: Answer questions with ablation (original, no_rewatch, no_highlevel)
  if python3 - <<PY
import json
import pickle
from pathlib import Path

from reason_ablation import reason_original, reason_no_rewatch, reason_no_highlevel
from reason_full import evaluate_answer
from utils.token_monitor import TokenMonitor

video_name = "${video}"

questions_path = Path("data/questions/robot.json")
results_dir = Path("data/results")
results_dir.mkdir(parents=True, exist_ok=True)

result_files = {
    "original": results_dir / "results_original.json",
    "no_rewatch": results_dir / "results_no_rewatch.json",
    "no_highlevel": results_dir / "results_no_highlevel.json",
}
graph_path = Path("data/semantic_memory") / f"{video_name}.pkl"

if not graph_path.exists():
    raise SystemExit(f"Graph file not found: {graph_path}")

with open(graph_path, "rb") as f:
    graph = pickle.load(f)

with open(questions_path, "r", encoding="utf-8") as f:
    questions_data = json.load(f)

video_questions = questions_data.get(video_name, {}).get("qa_list", [])

variants = [
    ("original", reason_original),
    ("no_rewatch", reason_no_rewatch),
    ("no_highlevel", reason_no_highlevel),
]

token_summary_path = results_dir / "token_summary_ablation.json"
token_totals = {}
if token_summary_path.exists():
    try:
        with open(token_summary_path, "r", encoding="utf-8") as f:
            token_totals = json.load(f)
    except json.JSONDecodeError:
        token_totals = {}

def add_usage(totals, new):
    if not totals:
        return dict(new)
    for key in ("text_llm", "vision_llm"):
        if key in new and key in totals:
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                totals[key][k] = totals[key].get(k, 0) + new[key].get(k, 0)
    totals["video_watch_calls"] = totals.get("video_watch_calls", 0) + new.get("video_watch_calls", 0)
    totals["total_tokens"] = totals.get("total_tokens", 0) + new.get("total_tokens", 0)
    return totals

for variant_name, reason_fn in variants:
    print(f"\n  Running ablation: {variant_name}")
    token_monitor = TokenMonitor()
    existing = {}
    result_path = result_files[variant_name]
    if result_path.exists():
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            existing = {}

    for i, qa in enumerate(video_questions, 1):
        question_id = qa["question_id"]
        question = qa["question"]
        ground_truth = qa["answer"]
        reasoning = qa.get("reasoning", "")
        timestamp = qa.get("timestamp", "")
        qa_type = qa.get("type", [])
        before_clip = qa.get("before_clip", None)

        try:
            reason_result = reason_fn(question, graph, video_name, token_monitor=token_monitor)
            predicted = reason_result.get("final_answer", "")
            is_correct = evaluate_answer(question, ground_truth, predicted, token_monitor=token_monitor)

            reason_result["evaluator_correct"] = is_correct
            reason_result["ground_truth_answer"] = ground_truth
            reason_result["reasoning"] = reasoning
            reason_result["timestamp"] = timestamp
            reason_result["type"] = qa_type
            reason_result["before_clip"] = before_clip
            existing[question_id] = reason_result
        except Exception as e:
            import traceback
            traceback.print_exc()
            existing[question_id] = {
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

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    tdict = token_monitor.to_dict()
    if variant_name not in token_totals:
        token_totals[variant_name] = {"text_llm": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                                      "vision_llm": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                                      "video_watch_calls": 0, "total_tokens": 0}
    token_totals[variant_name] = add_usage(token_totals[variant_name], tdict)
    print(f"    Tokens: {tdict['total_tokens']} total, video calls: {tdict['video_watch_calls']}")

with open(token_summary_path, "w", encoding="utf-8") as f:
    json.dump(token_totals, f, indent=2)

print(f"✓ Updated results_original.json, results_no_rewatch.json, results_no_highlevel.json for {video_name} ({len(video_questions)} questions)")
PY
  then
    : # success
  else
    echo "✗ Reasoning failed for ${video}"
    cleanup_video "$video"
    continue
  fi

  # Step 5: Cleanup to free storage
  cleanup_video "$video"
  echo "✓ Cleaned up video and frames for ${video}"
done

echo ""
echo "All videos processed."
