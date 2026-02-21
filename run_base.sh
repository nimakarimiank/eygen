#!/usr/bin/env bash
set -euo pipefail

EPOCHS=10
BATCH_SIZE=300
LAYER_NODES=(500)
FIRST_PERCENTILE=1
STEP_PERCENTILES=$(seq 5 5 95)

for PRUNE in $FIRST_PERCENTILE $STEP_PERCENTILES; do
  echo "Running prune_percentile=${PRUNE}"
  python3 base.py \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --layer_nodes "${LAYER_NODES[@]}" \
    --prune_percentile "$PRUNE" || exit 1
  echo "Completed prune_percentile=${PRUNE}"
  echo
done
