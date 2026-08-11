#!/usr/bin/env bash
# Build memory once, derive three threshold graphs, then run four reasoning jobs.
#
# Per video outputs:
#   Graphs (5):
#     data/graphs/<video>_preabstraction.pkl
#     data/graphs/<video>.pkl                         (default one-shot)
#     data/ablation/graphs_abs/{30_10,50_30,100_60}/<video>.pkl
#   Reasoning JSONs (5):
#     data/ablation/reasoning_abs/{no_abstraction,default,30_10,50_30,100_60}/<video>.json
#
# Usage:
#   bash scripts/ablation/run_threshold_ablation.sh VIDEO_ID ...
#   bash scripts/ablation/run_threshold_ablation.sh  # uses first_100_videos.txt
#   DRY_RUN=1 bash scripts/ablation/run_threshold_ablation.sh VIDEO_ID
#   FORCE=1 bash scripts/ablation/run_threshold_ablation.sh VIDEO_ID
#   JOBS=4 bash scripts/ablation/run_threshold_ablation.sh VIDEO_ID ...
#   HVM_VERBOSE=1 ...  # mirror detailed memorization prints to the terminal

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PY="${PYTHON:-python3}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
JOBS="${JOBS:-4}"

if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "JOBS must be a positive integer (got: ${JOBS})." >&2
  exit 1
fi

TAGS=("30_10" "50_30" "100_60")
declare -A CONFIGS=(
  ["30_10"]="configs/abs_30_10.json"
  ["50_30"]="configs/abs_50_30.json"
  ["100_60"]="configs/abs_100_60.json"
)

if (( $# > 0 )); then
  VIDEOS=("$@")
elif [[ -f first_100_videos.txt ]]; then
  mapfile -t VIDEOS < <(sed '/^[[:space:]]*$/d; /^[[:space:]]*#/d' first_100_videos.txt)
else
  echo "No videos provided and first_100_videos.txt was not found." >&2
  exit 1
fi

mkdir -p data/logs data/ablation/graphs_abs data/ablation/reasoning_abs

run_cmd() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

valid_reasoning() {
  local result_path="$1"
  local video="$2"
  [[ -s "$result_path" ]] || return 1
  "$PY" - "$result_path" "$video" <<'PY'
import json
import sys
from pathlib import Path

result_path, video = Path(sys.argv[1]), sys.argv[2]
try:
    results = json.loads(result_path.read_text(encoding="utf-8"))
    questions = json.loads(Path("data/web_100.json").read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)

expected = len(questions.get(video, {}).get("qa_list", []))
if not isinstance(results, dict) or len(results) != expected or expected == 0:
    raise SystemExit(1)

for record in results.values():
    reasoning = record.get("reasoning", {}) if isinstance(record, dict) else {}
    if not isinstance(reasoning, dict) or reasoning.get("error"):
        raise SystemExit(1)
PY
}

run_reasoning() {
  local video="$1"
  local tag="$2"
  local graph_dir="$3"
  local graph_suffix="${4:-}"
  local out_dir="data/ablation/reasoning_abs/${tag}"
  local output="${out_dir}/${video}.json"

  mkdir -p "$out_dir"
  if [[ "$FORCE" != "1" ]] && valid_reasoning "$output" "$video"; then
    echo "[${video}/${tag}] valid reasoning exists, skipping"
    return 0
  fi

  run_cmd "$PY" reason.py "$video" \
    --graph-dir "$graph_dir" \
    --graph-suffix "$graph_suffix" \
    --out-dir "$out_dir" \
    --log-tag "threshold_${tag}"

  if [[ "$DRY_RUN" != "1" ]] && ! valid_reasoning "$output" "$video"; then
    echo "✗ Invalid or incomplete reasoning output: ${output}" >&2
    return 1
  fi
}

memory_ready() {
  local video="$1"
  [[ -s "data/memorization/${video}.json" \
    && -s "data/graphs/${video}.pkl" \
    && -s "data/graphs/${video}_preabstraction.pkl" ]]
}

video_complete() {
  local video="$1"
  local tag
  [[ "$FORCE" != "1" ]] || return 1
  memory_ready "$video" || return 1
  valid_reasoning "data/ablation/reasoning_abs/no_abstraction/${video}.json" "$video" || return 1
  valid_reasoning "data/ablation/reasoning_abs/default/${video}.json" "$video" || return 1
  for tag in "${TAGS[@]}"; do
    [[ -s "data/ablation/graphs_abs/${tag}/${video}.pkl" ]] || return 1
    valid_reasoning "data/ablation/reasoning_abs/${tag}/${video}.json" "$video" || return 1
  done
}

run_video_ablation() {
  local video="$1"
  local tag graph_dir variant_graph
  local job_failed=0

  if ! run_reasoning "$video" "no_abstraction" "data/graphs" "_preabstraction"; then
    job_failed=1
  fi

  if ! run_reasoning "$video" "default" "data/graphs"; then
    job_failed=1
  fi

  for tag in "${TAGS[@]}"; do
    graph_dir="data/ablation/graphs_abs/${tag}"
    variant_graph="${graph_dir}/${video}.pkl"
    mkdir -p "$graph_dir"

    if [[ "$FORCE" == "1" || ! -s "$variant_graph" ]]; then
      echo "[${video}/${tag}] generating threshold graph"
      if ! run_cmd "$PY" scripts/ablation/abstraction_ablation.py "$video" \
        --config "${CONFIGS[$tag]}" \
        --out "$variant_graph"; then
        echo "✗ Graph generation failed: ${video}/${tag}" >&2
        job_failed=1
        continue
      fi
    else
      echo "[${video}/${tag}] graph exists, skipping"
    fi

    if [[ "$DRY_RUN" != "1" && ! -s "$variant_graph" ]]; then
      echo "✗ Missing graph output: ${variant_graph}" >&2
      job_failed=1
      continue
    fi

    if ! run_reasoning "$video" "$tag" "$graph_dir"; then
      job_failed=1
    fi
  done

  echo "[${video}] expected outputs: 5 graphs, 5 reasoning JSONs"
  return "$job_failed"
}

# Phase 1: memorization uses one independent log per video and can run in
# parallel. Existing complete memory outputs are reused.
run_memory() {
  local video="$1"
  echo "[${video}] memorization"
  if ! run_cmd "$PY" process_full_video.py "$video"; then
    echo "✗ Memorization command failed: ${video}" >&2
    return 1
  fi
  if [[ "$DRY_RUN" != "1" ]] && ! memory_ready "$video"; then
    echo "✗ Missing memory output for ${video}" >&2
    return 1
  fi
  return 0
}

MEMORY_VIDEOS=()
for video in "${VIDEOS[@]}"; do
  [[ -n "$video" ]] || continue
  if [[ "$FORCE" == "1" ]] || ! memory_ready "$video"; then
    MEMORY_VIDEOS+=("$video")
  else
    echo "[${video}] memory outputs exist, skipping memorization"
  fi
done

echo "============================================================"
echo "Memory phase: videos=${#MEMORY_VIDEOS[@]} JOBS=${JOBS}"
running=0
memory_failed=0
for video in "${MEMORY_VIDEOS[@]}"; do
  if (( JOBS == 1 )); then
    if ! run_memory "$video"; then
      memory_failed=1
    fi
    continue
  fi

  runner_log="data/logs/memory_runner_${video}.log"
  run_memory "$video" >"$runner_log" 2>&1 &
  running=$((running + 1))
  echo "[memory dispatch] ${video} (running=${running}/${JOBS}, log=${runner_log})"
  if (( running >= JOBS )); then
    if ! wait -n; then
      memory_failed=1
    fi
    running=$((running - 1))
  fi
done
while (( running > 0 )); do
  if ! wait -n; then
    memory_failed=1
  fi
  running=$((running - 1))
done

READY_VIDEOS=()
for video in "${VIDEOS[@]}"; do
  [[ -n "$video" ]] || continue
  if [[ "$DRY_RUN" == "1" ]] || memory_ready "$video"; then
    READY_VIDEOS+=("$video")
  else
    echo "✗ ${video} has no complete memory output; skipping ablation" >&2
    memory_failed=1
  fi
done

# Phase 2: run each video's four reasoning variants concurrently across videos.
echo "============================================================"
echo "Ablation phase: videos=${#READY_VIDEOS[@]} JOBS=${JOBS}"

running=0
dispatched=0
skipped=0
failed="$memory_failed"
for video in "${READY_VIDEOS[@]}"; do
  if video_complete "$video"; then
    echo "[${video}] all 5 graphs and 5 reasoning JSONs are valid, skipping video"
    skipped=$((skipped + 1))
    continue
  fi

  if (( JOBS == 1 )); then
    if ! run_video_ablation "$video"; then
      failed=1
    fi
    dispatched=$((dispatched + 1))
    continue
  fi

  runner_log="data/logs/threshold_ablation_${video}.log"
  run_video_ablation "$video" >"$runner_log" 2>&1 &
  running=$((running + 1))
  dispatched=$((dispatched + 1))
  echo "[dispatch ${dispatched}] ${video} (running=${running}/${JOBS}, log=${runner_log})"

  if (( running >= JOBS )); then
    if ! wait -n; then
      failed=1
    fi
    running=$((running - 1))
  fi
done

while (( running > 0 )); do
  if ! wait -n; then
    failed=1
  fi
  running=$((running - 1))
done

if (( failed != 0 )); then
  echo "One or more jobs failed. Check data/logs/ and the messages above." >&2
  exit 1
fi

echo "All threshold-ablation jobs completed. dispatched=${dispatched} skipped=${skipped}"
