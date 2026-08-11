#!/usr/bin/env bash
# Inject 2%, 5%, and 10% cross-video noise, rebuild 100/60 abstraction,
# then run reasoning. Existing valid outputs are reused.
#
# Usage:
#   JOBS=3 bash scripts/ablation/run_noise_ablation.sh VIDEO_ID ...
#   JOBS=3 bash scripts/ablation/run_noise_ablation.sh  # all complete graphs
#   DRY_RUN=1 bash scripts/ablation/run_noise_ablation.sh VIDEO_ID ...

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PY="${PYTHON:-python3}"
JOBS="${JOBS:-3}"
DRY_RUN="${DRY_RUN:-0}"
CONFIG="configs/abs_100_60.json"
RATES=("0.02" "0.05" "0.10")

if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "JOBS must be a positive integer (got: ${JOBS})." >&2
  exit 1
fi

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

if (( $# > 0 )); then
  VIDEOS=("$@")
else
  mapfile -t VIDEOS < <(
    find data/graphs -maxdepth 1 -type f -name '*.pkl' \
      ! -name '*_preabstraction.pkl' -printf '%f\n' |
      sed 's/\.pkl$//' |
      sort
  )
fi

rate_tag() {
  case "$1" in
    0.02) echo "p2" ;;
    0.05) echo "p5" ;;
    0.10) echo "p10" ;;
    *) echo "Unsupported noise rate: $1" >&2; return 1 ;;
  esac
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

valid_noise_graph() {
  local graph_path="$1"
  local video="$2"
  local rate="$3"
  local config_path="$4"
  local manifest="${graph_path%.pkl}.noise_manifest.json"
  [[ -s "$graph_path" && -s "$manifest" ]] || return 1
  "$PY" - "$manifest" "$video" "$rate" "$config_path" <<'PY'
import json
import sys
from pathlib import Path

manifest, video, rate, config = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3]), sys.argv[4]
try:
    data = json.loads(manifest.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
if data.get("video") != video or abs(float(data.get("noise_rate", -1)) - rate) > 1e-9:
    raise SystemExit(1)
if data.get("config") != config:
    raise SystemExit(1)
if int(data.get("n_injected", 0)) <= 0:
    raise SystemExit(1)
ocr_before = int(data.get("n_ocr_before", -1))
ocr_injected = int(data.get("n_ocr_injected", -1))
if ocr_before < 0 or ocr_injected != int(rate * ocr_before):
    raise SystemExit(1)
PY
}

run_cmd() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  [[ "$DRY_RUN" == "1" ]] || "$@"
}

preflight_failed=0
for video in "${VIDEOS[@]}"; do
  if [[ ! -s "data/graphs/${video}_preabstraction.pkl" ]]; then
    echo "✗ Missing pre-abstraction checkpoint: ${video}" >&2
    preflight_failed=1
  fi
  if ! valid_reasoning "data/ablation/reasoning_abs/100_60/${video}.json" "$video"; then
    echo "✗ Missing/invalid clean 100/60 reasoning baseline: ${video}" >&2
    preflight_failed=1
  fi
done
if (( preflight_failed != 0 )); then
  echo "Preflight failed; no noise jobs were started." >&2
  exit 1
fi

run_job() {
  local rate="$1"
  local video="$2"
  local tag graph_dir out_dir graph_path output log
  tag="$(rate_tag "$rate")" || return 1
  graph_dir="data/ablation/graphs_noise/${tag}"
  out_dir="data/ablation/reasoning_noise/${tag}"
  graph_path="${graph_dir}/${video}.pkl"
  output="${out_dir}/${video}.json"
  log="data/logs/noise_${tag}_${video}.log"
  mkdir -p "$graph_dir" "$out_dir" data/logs

  {
    if valid_noise_graph "$graph_path" "$video" "$rate" "$CONFIG"; then
      echo "[${tag}/${video}] valid noise graph exists, skipping generation"
    else
      run_cmd "$PY" scripts/ablation/noise_injection.py "$video" \
        --noise-rate "$rate" \
        --config "$CONFIG" \
        --out "$graph_path"
    fi

    if [[ "$DRY_RUN" != "1" ]] && ! valid_noise_graph "$graph_path" "$video" "$rate" "$CONFIG"; then
      echo "✗ Invalid noise graph: ${graph_path}" >&2
      return 1
    fi

    if valid_reasoning "$output" "$video"; then
      echo "[${tag}/${video}] valid reasoning exists, skipping"
    else
      run_cmd "$PY" reason.py "$video" \
        --graph-dir "$graph_dir" \
        --out-dir "$out_dir" \
        --log-tag "noise_${tag}"
    fi

    if [[ "$DRY_RUN" != "1" ]] && ! valid_reasoning "$output" "$video"; then
      echo "✗ Invalid reasoning output: ${output}" >&2
      return 1
    fi
    echo "[${tag}/${video}] complete"
  } >"$log" 2>&1
}

running=0
failed=0
dispatched=0
for rate in "${RATES[@]}"; do
  for video in "${VIDEOS[@]}"; do
    tag="$(rate_tag "$rate")"
    graph_path="data/ablation/graphs_noise/${tag}/${video}.pkl"
    output="data/ablation/reasoning_noise/${tag}/${video}.json"
    if valid_noise_graph "$graph_path" "$video" "$rate" "$CONFIG" && valid_reasoning "$output" "$video"; then
      echo "[${tag}/${video}] complete, skipping"
      continue
    fi

    run_job "$rate" "$video" &
    running=$((running + 1))
    dispatched=$((dispatched + 1))
    echo "[dispatch ${dispatched}] ${tag}/${video} (running=${running}/${JOBS})"
    if (( running >= JOBS )); then
      if ! wait -n; then failed=1; fi
      running=$((running - 1))
    fi
  done
done
while (( running > 0 )); do
  if ! wait -n; then failed=1; fi
  running=$((running - 1))
done

if (( failed != 0 )); then
  echo "One or more noise jobs failed. Check data/logs/noise_*.log." >&2
  exit 1
fi

echo "All noise jobs completed. dispatched=${dispatched}"
if [[ "$DRY_RUN" != "1" ]]; then
  "$PY" scripts/ablation/aggregate_noise.py "${VIDEOS[@]}"
fi
