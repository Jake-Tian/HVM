#!/usr/bin/env bash
set -uo pipefail

# Run one reasoning variant on the same 24 prebuilt graph memories.
# This script never runs memorization, triple extraction, or graph building.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

VARIANT="${1:-}"
DEFAULT_HVM_PY="/research/d7/gds/yztian25/miniconda3/envs/hivim2/bin/python3.11"
if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -x "$DEFAULT_HVM_PY" ]]; then
  PY="$DEFAULT_HVM_PY"
else
  PY="python3"
fi
JOBS="${JOBS:-4}"
DRY_RUN="${DRY_RUN:-0}"
GRAPH_DIR="${GRAPH_DIR:-data/graphs}"

case "$VARIANT" in
  three_route)
    ENTRYPOINT_FILE="reasoning_variants/three_route_backup.py"
    ENTRYPOINT_LABEL="reasoning_variants.three_route_backup"
    ENTRYPOINT_ARGS=(-m reasoning_variants.three_route_backup)
    OUT_DIR="${OUT_DIR:-data/ablation/reasoning_routes/three_route_luna_medium}"
    LOG_TAG="${LOG_TAG:-three_route_luna_medium}"
    RUNNER_LOG_DIR="${RUNNER_LOG_DIR:-data/logs/reasoning_24_three_route_luna_medium}"
    ;;
  five_tools)
    ENTRYPOINT_FILE="reason.py"
    ENTRYPOINT_LABEL="reason.py"
    ENTRYPOINT_ARGS=(reason.py)
    OUT_DIR="${OUT_DIR:-data/reasoning_24_five_tools_luna_medium}"
    LOG_TAG="${LOG_TAG:-five_tools_luna_medium}"
    RUNNER_LOG_DIR="${RUNNER_LOG_DIR:-data/logs/reasoning_24_five_tools_luna_medium}"
    ;;
  *)
    echo "Usage: bash scripts/reasoning/run_reasoning_24_variant.sh {three_route|five_tools}" >&2
    exit 2
    ;;
esac

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

VIDEOS=(
  bedroom_01
  bedroom_06
  bedroom_07
  bedroom_12
  gym_02
  kitchen_02
  kitchen_09
  kitchen_17
  living_room_02
  living_room_09
  living_room_15
  living_room_22
  meeting_room_02
  meeting_room_03
  office_01
  office_03
  office_06
  study_03
  study_05
  study_06
  study_08
  study_12
  study_18
  study_23
)

if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "JOBS must be a positive integer, got: $JOBS" >&2
  exit 1
fi

missing=0
for path in "$ENTRYPOINT_FILE" data/robot.json; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    missing=1
  fi
done
for video in "${VIDEOS[@]}"; do
  if [[ ! -f "${GRAPH_DIR}/${video}.pkl" ]]; then
    echo "Missing graph: ${GRAPH_DIR}/${video}.pkl" >&2
    missing=1
  fi
  if [[ ! -d "data/frames/${video}" ]]; then
    echo "Missing frames: data/frames/${video}" >&2
    missing=1
  fi
done
if (( missing )); then
  exit 1
fi

"$PY" - "$VARIANT" "$ENTRYPOINT_LABEL" data/robot.json "${VIDEOS[@]}" <<'PY'
import ast
import json
import sys
from pathlib import Path

variant, entrypoint, robot_path, *videos = sys.argv[1:]
llm_tree = ast.parse(Path("utils/llm_gpt.py").read_text())
constants = {}
for node in llm_tree.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "MODEL":
            constants[target.id] = ast.literal_eval(node.value)
expected = {"MODEL": "gpt-5.6-luna"}
if constants != expected:
    raise SystemExit(f"Expected {expected}, found {constants} in utils/llm_gpt.py")

questions = json.loads(Path(robot_path).read_text())
total = 0
for video in videos:
    count = len(questions.get(video, {}).get("qa_list", []))
    total += count
    print(f"{video}: {count} questions")
print(f"TOTAL: {total} questions across {len(videos)} videos")
print(f"VARIANT: {variant} via {entrypoint}")
print("MODEL: gpt-5.6-luna")
PY

if [[ "$DRY_RUN" == "1" ]]; then
  if [[ "$VARIANT" == "five_tools" ]]; then
    "$PY" -m py_compile \
      reason.py \
      utils/reasoning_trace.py \
      reasoning/agent.py \
      reasoning/runtime.py \
      reasoning/tools.py \
      reasoning/prompts.py
  else
    "$PY" -m py_compile \
      reasoning_variants/three_route_backup.py \
      reasoning_variants/three_route/agent.py \
      reasoning_variants/three_route/frequency_memory.py \
      reasoning_variants/three_route/object_event_search.py \
      reasoning_variants/three_route/tools.py \
      utils/reasoning_trace.py
  fi
  echo "DRY_RUN=1, inputs and Python syntax are valid. No API calls were made."
  exit 0
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set." >&2
  exit 1
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
  echo "[$(date +%H:%M:%S)] start ${video} (${VARIANT})"
  if "$PY" "${ENTRYPOINT_ARGS[@]}" "$video" \
      --graph-dir "$GRAPH_DIR" \
      --out-dir "$OUT_DIR" \
      --log-tag "$LOG_TAG" \
      >"$runner_log" 2>&1; then
    echo "[$(date +%H:%M:%S)] done  ${video} (${VARIANT})"
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

"$PY" - "$OUT_DIR" data/robot.json "${VIDEOS[@]}" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
questions = json.loads(Path(sys.argv[2]).read_text())
videos = sys.argv[3:]
expected = sum(len(questions.get(v, {}).get("qa_list", [])) for v in videos)
total = correct = tokens = 0
print("\n24-video accuracy:")
for video in videos:
    path = out_dir / f"{video}.json"
    if not path.exists():
        print(f"  {video}: MISSING")
        continue
    records = json.loads(path.read_text())
    video_correct = 0
    for record in records.values():
        reasoning = record.get("reasoning") if isinstance(record, dict) else None
        if isinstance(reasoning, dict):
            video_correct += reasoning.get("evaluate_correct") is True
            tokens += int((reasoning.get("token_summaries") or {}).get("total", 0) or 0)
    total += len(records)
    correct += video_correct
    print(f"  {video}: {video_correct}/{len(records)}")
accuracy = 100 * correct / total if total else 0
print(f"TOTAL: {correct}/{total} = {accuracy:.2f}%")
print(f"TOKENS: {tokens}")
if total != expected:
    raise SystemExit(f"Expected {expected} completed results, found {total}.")
PY

if (( failed )); then
  echo "One or more video jobs failed. Inspect ${RUNNER_LOG_DIR}." >&2
  exit 1
fi

echo "Results: $OUT_DIR"
echo "Runner logs: $RUNNER_LOG_DIR"
