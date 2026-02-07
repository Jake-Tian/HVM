#!/usr/bin/env bash
set -euo pipefail

# Run full pipeline per video. Videos can be processed in parallel.
# 1) Download MP4
# 2) Add subtitles + extract frames
# 3) Build graph memory
# 4) Answer questions with ablation (original, no_rewatch, no_highlevel) → per-video cache
# 5) Cleanup MP4 and frames
# 6) Merge all per-video caches into final results_*.json

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Number of videos to process in parallel (set to 1 for sequential)
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-4}"
CACHE_DIR="data/results/ablation_cache"

cleanup_video() {
  local video_name="$1"
  rm -f "data/videos/${video_name}.mp4"
  rm -rf "data/frames/${video_name}"
}

process_one_video() {
  local video="$1"
  [[ -z "$video" ]] && return 0

  echo ""
  echo "[$(date +%H:%M:%S)] Processing video: ${video}"
  echo "============================================================"

  # Step 1: Download video
  if ! python3 preprocessing/download_hf_videos.py "$video"; then
    echo "✗ [${video}] Download failed"
    cleanup_video "$video"
    return 1
  fi

  # Step 2: Add subtitles + extract frames
  if [[ ! -f "data/subtitles/robot/${video}.srt" ]]; then
    echo "✗ [${video}] Subtitle file missing: data/subtitles/robot/${video}.srt"
    cleanup_video "$video"
    return 1
  fi

  if ! python3 preprocessing/add_subtitles_and_extract_frames.py "$video"; then
    echo "✗ [${video}] Frame extraction failed"
    cleanup_video "$video"
    return 1
  fi

  # Step 3: Build graph memory
  if ! python3 - <<PY
from pathlib import Path
from process_full_video import process_full_video

video_name = "${video}"
frames_dir = Path(f"data/frames/{video_name}")
if not frames_dir.exists():
    raise SystemExit(f"Frames directory not found: {frames_dir}")

process_full_video(frames_dir)
print(f"✓ [{video_name}] Graph memory built")
PY
  then
    echo "✗ [${video}] Graph processing failed"
    cleanup_video "$video"
    return 1
  fi

  # Step 4: Answer questions with ablation - write to per-video cache (no race condition)
  mkdir -p "$CACHE_DIR"
  if ! python3 - <<PY
import json
import pickle
from pathlib import Path

from reason_ablation import reason_original, reason_no_rewatch, reason_no_highlevel
from reason_full import evaluate_answer
from utils.token_monitor import TokenMonitor

video_name = "${video}"
cache_dir = Path("${CACHE_DIR}")

questions_path = Path("data/questions/robot.json")
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

per_video_results = {}
per_video_tokens = {}

for variant_name, reason_fn in variants:
    token_monitor = TokenMonitor()
    results = {}

    for qa in video_questions:
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
            results[question_id] = reason_result
        except Exception as e:
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

    per_video_results[variant_name] = results
    per_video_tokens[variant_name] = token_monitor.to_dict()

# Write to per-video cache files (no conflicts between parallel jobs)
for variant_name in ("original", "no_rewatch", "no_highlevel"):
    cache_path = cache_dir / f"{video_name}_{variant_name}.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(per_video_results[variant_name], f, indent=2, ensure_ascii=False)

with open(cache_dir / f"{video_name}_tokens.json", "w", encoding="utf-8") as f:
    json.dump(per_video_tokens, f, indent=2)

print(f"✓ [{video_name}] Ablation done ({len(video_questions)} questions)")
PY
  then
    echo "✗ [${video}] Reasoning failed"
    cleanup_video "$video"
    return 1
  fi

  # Step 5: Cleanup to free storage
  cleanup_video "$video"
  echo "✓ [${video}] Done (cleaned up)"
  return 0
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

echo "Processing ${#VIDEOS[@]} videos with max ${MAX_PARALLEL_JOBS} parallel jobs"
echo ""

# Run videos in parallel (or sequential if MAX_PARALLEL_JOBS=1)
# Process in batches to avoid race conditions and limit concurrency
i=0
while (( i < ${#VIDEOS[@]} )); do
  batch=()
  for ((j=0; j < MAX_PARALLEL_JOBS && i < ${#VIDEOS[@]}; j++)); do
    video="${VIDEOS[i]}"
    if [[ -n "$video" ]]; then
      batch+=("$video")
    fi
    (( i++ )) || true
  done
  if [[ ${#batch[@]} -gt 0 ]]; then
    for video in "${batch[@]}"; do
      ( process_one_video "$video" ) &
    done
    wait || true  # Continue to merge even if some jobs failed
  fi
done
echo ""
echo "All video processing complete."
echo ""

# Step 6: Merge per-video cache files into final results
echo "Merging results..."
python3 - <<'MERGE'
import json
from pathlib import Path

cache_dir = Path("data/results/ablation_cache")
results_dir = Path("data/results")
results_dir.mkdir(parents=True, exist_ok=True)

if not cache_dir.exists():
    print("No cache directory found. Nothing to merge.")
    exit(0)

# Find all unique videos from cache files
videos = set()
for f in cache_dir.glob("*_original.json"):
    videos.add(f.stem.replace("_original", ""))

if not videos:
    print("No per-video cache files found.")
    exit(0)

# Merge each variant
for variant in ("original", "no_rewatch", "no_highlevel"):
    merged = {}
    for video in sorted(videos):
        cache_path = cache_dir / f"{video}_{variant}.json"
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                merged.update(data)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {cache_path}")
    out_path = results_dir / f"results_{variant}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"  Merged {len(merged)} results -> {out_path}")

# Merge token summaries
def add_usage(totals, new):
    if not totals:
        import copy
        return copy.deepcopy(new)
    for key in ("text_llm", "vision_llm"):
        if key in new and key in totals:
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                totals[key][k] = totals[key].get(k, 0) + new[key].get(k, 0)
    totals["video_watch_calls"] = totals.get("video_watch_calls", 0) + new.get("video_watch_calls", 0)
    totals["total_tokens"] = totals.get("total_tokens", 0) + new.get("total_tokens", 0)
    return totals

token_totals = {}
for video in sorted(videos):
    token_path = cache_dir / f"{video}_tokens.json"
    if token_path.exists():
        try:
            with open(token_path, "r", encoding="utf-8") as f:
                per_video = json.load(f)
            for variant, tdict in per_video.items():
                if variant not in token_totals:
                    token_totals[variant] = {"text_llm": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                                           "vision_llm": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                                           "video_watch_calls": 0, "total_tokens": 0}
                token_totals[variant] = add_usage(token_totals[variant], tdict)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse {token_path}")

if token_totals:
    token_path = results_dir / "token_summary_ablation.json"
    with open(token_path, "w", encoding="utf-8") as f:
        json.dump(token_totals, f, indent=2)
    print(f"  Merged token summary -> {token_path}")

print("✓ Merge complete.")
MERGE

echo ""
echo "Pipeline complete."
