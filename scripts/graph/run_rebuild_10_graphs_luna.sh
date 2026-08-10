#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

VIDEOS=(
  gym_02
  meeting_room_02
  study_12
  bedroom_12
  office_06
  living_room_09
  kitchen_02
  office_03
  bedroom_07
  meeting_room_03
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  python3 -m scripts.graph.rebuild_graph_from_memorization --dry-run "${VIDEOS[@]}"
  exit 0
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set" >&2
  exit 1
fi

backup_root="data/backups/rebuild_luna_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_root/memorization" "$backup_root/graphs" data/logs

for video in "${VIDEOS[@]}"; do
  for source in \
    "data/memorization/${video}.json" \
    "data/graphs/${video}.pkl" \
    "data/graphs/${video}_preabstraction.pkl"; do
    if [[ ! -f "$source" ]]; then
      echo "Missing required input: $source" >&2
      exit 1
    fi
  done
  cp -al "data/memorization/${video}.json" "$backup_root/memorization/"
  cp -al "data/graphs/${video}.pkl" "$backup_root/graphs/"
  cp -al "data/graphs/${video}_preabstraction.pkl" "$backup_root/graphs/"
done

echo "Backup: $backup_root"
echo "Rebuilding ${#VIDEOS[@]} videos with GPT-5.6 Luna, medium effort"

max_parallel_jobs="${MAX_PARALLEL_JOBS:-2}"
running=0
failed=0

for video in "${VIDEOS[@]}"; do
  (
    python3 -m scripts.graph.rebuild_graph_from_memorization "$video" \
      >"data/logs/${video}_rebuild_luna.log" 2>&1
  ) &
  ((running += 1))

  if (( running >= max_parallel_jobs )); then
    if ! wait -n; then
      failed=1
    fi
    ((running -= 1))
  fi
done

while (( running > 0 )); do
  if ! wait -n; then
    failed=1
  fi
  ((running -= 1))
done

if (( failed != 0 )); then
  echo "One or more rebuilds failed. Existing files were backed up at: $backup_root" >&2
  exit 1
fi

echo "All 10 triple and graph rebuilds completed."
echo "Logs: data/logs/*_rebuild_luna.log"
echo "Backup: $backup_root"
