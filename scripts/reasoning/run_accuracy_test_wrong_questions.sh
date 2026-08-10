#!/usr/bin/env bash
set -uo pipefail

# Re-run reasoning only for the baseline errors in the 13-video evaluation set.
# Existing graph memory, episodic memory, and frames are reused. Memorization is
# never invoked by this script.
#
# Usage:
#   bash scripts/reasoning/run_accuracy_test_wrong_questions.sh
#   DRY_RUN=1 bash scripts/reasoning/run_accuracy_test_wrong_questions.sh
#   JOBS=2 OUT_DIR=data/ablation/reasoning_accuracy_test_v2 \
#     bash scripts/reasoning/run_accuracy_test_wrong_questions.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PY="${PYTHON:-python3}"
JOBS="${JOBS:-4}"
DRY_RUN="${DRY_RUN:-0}"
BASELINE_DIR="${BASELINE_DIR:-data/reasoning}"
GRAPH_DIR="${GRAPH_DIR:-data/graphs}"
OUT_DIR="${OUT_DIR:-data/ablation/reasoning_accuracy_test}"
LOG_TAG="${LOG_TAG:-accuracy_wrong_only}"
RUNNER_LOG_DIR="${RUNNER_LOG_DIR:-data/logs/accuracy_wrong_only}"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

VIDEOS=(
  bedroom_01
  bedroom_06
  kitchen_09
  kitchen_17
  living_room_02
  living_room_15
  living_room_22
  office_01
  study_03
  study_05
  study_06
  study_18
  study_23
)

if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "JOBS must be a positive integer, got: $JOBS" >&2
  exit 1
fi

missing=0
for video in "${VIDEOS[@]}"; do
  for path in \
    "${BASELINE_DIR}/${video}.json" \
    "${GRAPH_DIR}/${video}.pkl" \
    "data/memorization/${video}.json" \
    "data/frames/${video}"; do
    if [[ ! -e "$path" ]]; then
      echo "Missing required input: $path" >&2
      missing=1
    fi
  done
done
if (( missing )); then
  exit 1
fi

selection_summary="$(
  "$PY" - "$BASELINE_DIR" "${VIDEOS[@]}" <<'PY'
import json
import sys
from pathlib import Path

baseline_dir = Path(sys.argv[1])
total = 0
for video in sys.argv[2:]:
    records = json.loads((baseline_dir / f"{video}.json").read_text())
    wrong = sum(
        isinstance(record, dict)
        and isinstance(record.get("reasoning"), dict)
        and record["reasoning"].get("evaluate_correct") is False
        for record in records.values()
    )
    total += wrong
    print(f"{video}: {wrong} incorrect questions")
print(f"TOTAL: {total} incorrect questions")
PY
)"
echo "$selection_summary"

selected_total="$(awk '/^TOTAL:/ {print $2}' <<< "$selection_summary")"
if [[ "$selected_total" != "67" ]]; then
  echo "Expected 67 baseline errors across the 13 videos, found ${selected_total}." >&2
  exit 1
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1, inputs are valid. No API calls were made."
  exit 0
fi

if find "$OUT_DIR" -maxdepth 1 -type f -name "*.json" -print -quit 2>/dev/null |
    grep -q .; then
  echo "Output directory already contains JSON files: $OUT_DIR" >&2
  echo "Set OUT_DIR to a new directory to avoid overwriting a previous run." >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$RUNNER_LOG_DIR"

run_one_video() {
  local video="$1"
  local runner_log="${RUNNER_LOG_DIR}/${video}.log"
  echo "[$(date +%H:%M:%S)] start ${video}"
  if "$PY" reason.py "$video" \
      --graph-dir "$GRAPH_DIR" \
      --out-dir "$OUT_DIR" \
      --incorrect-only-from "$BASELINE_DIR" \
      --log-tag "$LOG_TAG" \
      >"$runner_log" 2>&1; then
    echo "[$(date +%H:%M:%S)] done  ${video}"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] FAIL  ${video} (see ${runner_log})" >&2
  return 1
}

running=0
failed=0
for video in "${VIDEOS[@]}"; do
  run_one_video "$video" &
  running=$((running + 1))
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

"$PY" - "$OUT_DIR" "$BASELINE_DIR" "${VIDEOS[@]}" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
baseline_dir = Path(sys.argv[2])
videos = sys.argv[3:]
total = 0
correct = 0
print("\nAccuracy on the previous baseline errors:")
for path in sorted(out_dir.glob("*.json")):
    records = json.loads(path.read_text())
    video_total = len(records)
    video_correct = sum(
        isinstance(record, dict)
        and isinstance(record.get("reasoning"), dict)
        and record["reasoning"].get("evaluate_correct") is True
        for record in records.values()
    )
    total += video_total
    correct += video_correct
    print(f"  {path.stem}: {video_correct}/{video_total}")
accuracy = 100 * correct / total if total else 0
print(f"TOTAL: {correct}/{total} = {accuracy:.2f}%")

baseline_total = 0
baseline_correct = 0
for video in videos:
    records = json.loads((baseline_dir / f"{video}.json").read_text())
    baseline_total += len(records)
    baseline_correct += sum(
        isinstance(record, dict)
        and isinstance(record.get("reasoning"), dict)
        and record["reasoning"].get("evaluate_correct") is True
        for record in records.values()
    )
projected_correct = baseline_correct + correct
projected_accuracy = 100 * projected_correct / baseline_total
print(
    f"Projected 13-video accuracy with previous correct answers unchanged: "
    f"{projected_correct}/{baseline_total} = {projected_accuracy:.2f}%"
)
PY

if (( failed )); then
  echo "One or more video jobs failed. Inspect ${RUNNER_LOG_DIR}." >&2
  exit 1
fi

echo "Results: $OUT_DIR"
echo "Runner logs: $RUNNER_LOG_DIR"
