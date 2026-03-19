#!/usr/bin/env bash
set -euo pipefail

export PHASE3_CAPACITY_MODE="${PHASE3_CAPACITY_MODE:-finalist}"
export PHASE3_CAPACITY_TESTS="${PHASE3_CAPACITY_TESTS:-recur_3x6_d1152_share}"
export PHASE3_CAPACITY_SEEDS="${PHASE3_CAPACITY_SEEDS:-1337,2027}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
export NUM_HEADS="${NUM_HEADS:-8}"

kv_values="${KV_SWEEP_VALUES:-4,2}"
IFS=',' read -r -a kv_list <<<"$kv_values"

for kv_raw in "${kv_list[@]}"; do
  kv="$(echo "$kv_raw" | awk '{$1=$1; print}')"
  [[ -z "$kv" ]] && continue
  export NUM_KV_HEADS="$kv"
  echo
  echo "=== Finalist KV ablation: NUM_HEADS=${NUM_HEADS} NUM_KV_HEADS=${NUM_KV_HEADS} ==="
  bash ./phase3_capacity_matrix.sh
done
