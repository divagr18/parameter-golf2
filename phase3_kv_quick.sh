#!/usr/bin/env bash
set -euo pipefail

export PHASE3_CAPACITY_MODE="${PHASE3_CAPACITY_MODE:-quick}"
export PHASE3_CAPACITY_TESTS="${PHASE3_CAPACITY_TESTS:-recur_3x6_d1152_share}"
export PHASE3_CAPACITY_SEEDS="${PHASE3_CAPACITY_SEEDS:-1337}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
export NUM_HEADS="${NUM_HEADS:-8}"
export ITERATIONS="${ITERATIONS:-3000}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-32768}"
export VAL_MAX_TOKENS="${VAL_MAX_TOKENS:-524288}"
export FINAL_ROUNDTRIP_EVAL="${FINAL_ROUNDTRIP_EVAL:-0}"

kv_values="${KV_SWEEP_VALUES:-8,4,2,1}"
IFS=',' read -r -a kv_list <<<"$kv_values"

echo "Quick KV sweep config: iterations=${ITERATIONS} train_batch_tokens=${TRAIN_BATCH_TOKENS} grad_accum_steps=${GRAD_ACCUM_STEPS} val_max_tokens=${VAL_MAX_TOKENS} final_roundtrip_eval=${FINAL_ROUNDTRIP_EVAL}"

for kv_raw in "${kv_list[@]}"; do
  kv="$(echo "$kv_raw" | awk '{$1=$1; print}')"
  [[ -z "$kv" ]] && continue
  export NUM_KV_HEADS="$kv"
  echo
  echo "=== Quick KV ablation: NUM_HEADS=${NUM_HEADS} NUM_KV_HEADS=${NUM_KV_HEADS} ==="
  bash ./phase3_capacity_matrix.sh
done
