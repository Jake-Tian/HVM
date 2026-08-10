#!/usr/bin/env bash
# Noise-injection experiment for R1W3.
#
# For each (noise_rate, video): inject cross-video noise edges into the
# pre-abstraction checkpoint, re-run abstraction at 50/30, then run reasoning.
# Records injected-edge manifest + reasoning-time noise-retrieval log for later
# aggregation (accuracy, abstraction pollution rate, noise retrieval count).
#
# Usage:
#   JOBS=1 bash scripts/ablation/run_noise_experiment.sh
#   NOISE_RATES="0.02 0.05 0.10" JOBS=2 bash scripts/ablation/run_noise_experiment.sh
#
# Requires OPENAI_API_KEY (and OPENAI_BASE_URL if using a proxy) in the environment.

set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
PY="${PYTHON:-python3}"
JOBS="${JOBS:-1}"
NOISE_RATES="${NOISE_RATES:-0.02 0.05 0.10}"

# Cap BLAS/OpenMP threads per process. Each Python process would otherwise spawn
# ~40 OpenBLAS threads; with JOBS>=4 that exhausts RLIMIT_NPROC (1024) and crashes
# later jobs at `import numpy` (pthread_create: Resource temporarily unavailable).
# Reasoning is LLM-bound, so capping BLAS threads to 1 has no measurable cost.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 13 benchmark videos (same set as the threshold ablation).
VIDEOS=( bedroom_01 bedroom_06 kitchen_09 kitchen_17 living_room_02 living_room_15 \
         living_room_22 office_01 study_03 study_05 study_06 study_18 study_23 )
CONFIG="configs/abs_50_30.json"

mkdir -p data/logs

rate_tag() {
  # 0.02 -> p2, 0.05 -> p5, 0.10 -> p10
  python3 -c "print('p'+str(int(float('$1')*100)))"
}

run_job() {
  local rate="$1" video="$2"
  local tag; tag="$(rate_tag "$rate")"
  local gdir="data/ablation/graphs_noise/${tag}"
  local odir="data/ablation/reasoning_noise/${tag}"
  local gpkl="${gdir}/${video}.pkl"
  local opkl="${odir}/${video}.json"
  local nlog="${odir}/${video}.noise.jsonl"
  local log="data/logs/noise_${tag}_${video}.log"
  mkdir -p "$gdir" "$odir"

  {
    if [[ -f "$gpkl" ]]; then
      echo "[${tag}/${video}] injected graph exists, skipping injection"
    else
      echo "[${tag}/${video}] injecting noise (rate=${rate})..."
      if ! $PY noise_injection.py "$video" --noise-rate "$rate" \
          --config "$CONFIG" --out "$gpkl" --noise-log "$nlog"; then
        echo "  ✗ noise injection failed for ${video} (${tag})"
        return 1
      fi
    fi

    if [[ -f "$opkl" ]]; then
      echo "[${tag}/${video}] reasoning json exists, skipping"
    else
      # fresh noise-retrieval log for this run
      rm -f "$nlog"
      echo "[${tag}/${video}] reasoning..."
      if ! $PY reason.py "$video" --graph-dir "$gdir" --out-dir "$odir"; then
        echo "  ✗ reasoning failed for ${video} (${tag})"
        return 1
      fi
    fi
    echo "[${tag}/${video}] done ✓"
  } > "$log" 2>&1
}

running=0
dispatched=0
skipped=0
for rate in $NOISE_RATES; do
  for video in "${VIDEOS[@]}"; do
    [[ -z "$video" ]] && continue
    tag="$(rate_tag "$rate")"
    if [[ -f "data/ablation/graphs_noise/${tag}/${video}.pkl" && -f "data/ablation/reasoning_noise/${tag}/${video}.json" ]]; then
      skipped=$((skipped+1))
      continue
    fi
    run_job "$rate" "$video" &
    running=$((running+1))
    dispatched=$((dispatched+1))
    echo "[dispatch ${dispatched}] ${tag}/${video}  (running=${running}/${JOBS})"
    if (( running >= JOBS )); then
      if ! wait -n; then
        echo "[warn] a job exited non-zero (see data/logs/noise_*_*.log)"
      fi
      running=$((running-1))
    fi
  done
done
echo "[drain] waiting for ${running} remaining job(s)..."
wait || true
echo ""
echo "===================== summary ====================="
for rate in $NOISE_RATES; do
  tag="$(rate_tag "$rate")"
  npkl=$(ls "data/ablation/graphs_noise/${tag}"/*.pkl 2>/dev/null | wc -l)
  njson=$(ls "data/ablation/reasoning_noise/${tag}"/*.json 2>/dev/null | wc -l)
  echo "  ${tag} (rate=${rate}): graphs=${npkl}/13  reasoning=${njson}/13"
done
echo "  dispatched=${dispatched} skipped(already done)=${skipped}"
echo ""
echo "Aggregating..."
$PY scripts/aggregate_noise.py
