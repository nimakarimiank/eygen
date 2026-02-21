#!/usr/bin/env bash
set -euo pipefail

EPOCHS=10
BATCH_SIZE=300
LAYER_NODES=(50)
START_PERCENTILE=1
END_PERCENTILE=95

for PRUNE in $(seq "$START_PERCENTILE" "$END_PERCENTILE"); do
  echo "Running prune_percentile=${PRUNE}"
  python3 main.py \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --layer_nodes "${LAYER_NODES[@]}" \
    --prune_percentile "$PRUNE" || exit 1
  echo "Completed prune_percentile=${PRUNE}"
  echo
done
