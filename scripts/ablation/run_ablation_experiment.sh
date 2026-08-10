#!/usr/bin/env bash
# Threshold-based abstraction ablation experiment (parallel version).
#
# Compares incremental-abstraction frequency on the same memorization checkpoint:
#   30/10 (frequent), 50/30 (medium), 100/60 (rare)
# All three use incremental_enabled=true, final_lower_bound 10/3 (fixed).
#
# For each config x video:
#   1. Re-run abstraction from the _preabstraction.pkl checkpoint (no MLLM,
#      only the abstraction text-LLM calls) -> data/ablation/graphs_abs/<tag>/<video>.pkl
#   2. Run reasoning over that variant graph -> data/ablation/reasoning_abs/<tag>/<video>.json
# Resumable: skips a step if its output already exists.
#
# Parallelism: dispatches (tag,video) jobs concurrently with a pool of size JOBS.
#   JOBS=4 bash scripts/ablation/run_ablation_experiment.sh
#   JOBS=8 bash scripts/ablation/run_ablation_experiment.sh v1 v2
#
# Note: reasoning is the expensive part (gpt-5-mini). Each job's stdout/stderr
# is captured to data/logs/ablation_<tag>_<video>.log. The inner reason.py debug
# log (data/logs/<video>_reason.log) may interleave across tags for the same
# video — that's cosmetic only; the JSON results are tag-specific and safe.

set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
PY="${PYTHON:-python3}"
JOBS="${JOBS:-4}"

# Cap BLAS/OpenMP threads per process to avoid exhausting RLIMIT_NPROC (1024)
# when running JOBS>=4 concurrent Python processes (each would otherwise spawn
# ~40 OpenBLAS threads and crash at `import numpy`). Reasoning is LLM-bound.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# tag : config file
declare -a TAGS=( "30_10" "50_30" "100_60" )
declare -A CFG=( ["30_10"]="configs/abs_30_10.json" ["50_30"]="configs/abs_50_30.json" ["100_60"]="configs/abs_100_60.json" )

# Video list: explicit args, else all videos that have a _preabstraction.pkl checkpoint.
if [[ "$#" -gt 0 ]]; then
  VIDEOS=("$@")
else
  mapfile -t VIDEOS < <(ls data/graphs/*_preabstraction.pkl 2>/dev/null | sed 's#.*/##; s/_preabstraction\.pkl$//' | sort)
fi

if [[ ${#VIDEOS[@]} -eq 0 ]]; then
  echo "No videos with a _preabstraction.pkl checkpoint found. Run memorization first."
  exit 1
fi

mkdir -p data/logs

echo "Ablation: ${#VIDEOS[@]} videos x ${#TAGS[@]} configs (${TAGS[*]}) | JOBS=${JOBS}"
echo ""

# Run one (tag, video) job: abstraction then reasoning. Returns 0 on success.
run_job() {
  local tag="$1" video="$2"
  local cfg="${CFG[$tag]}"
  local gdir="data/ablation/graphs_abs/${tag}"
  local odir="data/ablation/reasoning_abs/${tag}"
  local gpkl="${gdir}/${video}.pkl"
  local opkl="${odir}/${video}.json"
  local log="data/logs/ablation_${tag}_${video}.log"
  mkdir -p "$gdir" "$odir"

  {
    if [[ -f "$gpkl" ]]; then
      echo "[${tag}/${video}] abstraction pkl exists, skipping"
    else
      echo "[${tag}/${video}] abstraction..."
      if ! $PY abstraction_ablation.py "$video" --config "$cfg" --out "$gpkl"; then
        echo "  ✗ abstraction failed for ${video} (${tag})"
        return 1
      fi
    fi

    if [[ -f "$opkl" ]]; then
      echo "[${tag}/${video}] reasoning json exists, skipping"
    else
      echo "[${tag}/${video}] reasoning..."
      if ! $PY reason.py "$video" --graph-dir "$gdir" --out-dir "$odir"; then
        echo "  ✗ reasoning failed for ${video} (${tag})"
        return 1
      fi
    fi
    echo "[${tag}/${video}] done ✓"
  } > "$log" 2>&1
}

# Dispatch jobs with a bounded pool. wait -n blocks until one job finishes.
running=0
dispatched=0
skipped=0
for tag in "${TAGS[@]}"; do
  for video in "${VIDEOS[@]}"; do
    [[ -z "$video" ]] && continue
    # Fast pre-check: skip dispatch if both outputs already exist.
    if [[ -f "data/ablation/graphs_abs/${tag}/${video}.pkl" && -f "data/ablation/reasoning_abs/${tag}/${video}.json" ]]; then
      skipped=$((skipped+1))
      continue
    fi
    run_job "$tag" "$video" &
    running=$((running+1))
    dispatched=$((dispatched+1))
    echo "[dispatch ${dispatched}] ${tag}/${video}  (running=${running}/${JOBS})"
    if (( running >= JOBS )); then
      # Wait for any one job to finish; -e disabled so a failure doesn't abort.
      if ! wait -n; then
        echo "[warn] a job exited non-zero (see data/logs/ablation_*_*.log)"
      fi
      running=$((running-1))
    fi
  done
done

# Drain remaining jobs.
echo "[drain] waiting for ${running} remaining job(s)..."
fail=0
wait || fail=$?
echo ""

# Summary: count completed reasoning outputs per tag.
echo "===================== summary ====================="
for tag in "${TAGS[@]}"; do
  odir="data/ablation/reasoning_abs/${tag}"
  njson=$(ls "$odir"/*.json 2>/dev/null | wc -l)
  npkl=$(ls "data/ablation/graphs_abs/${tag}"/*.pkl 2>/dev/null | wc -l)
  echo "  ${tag}: graphs=${npkl} reasoning=${njson}"
done
echo "  dispatched=${dispatched} skipped(already done)=${skipped}"
echo ""

echo "Aggregating..."
$PY scripts/aggregate_ablation.py
