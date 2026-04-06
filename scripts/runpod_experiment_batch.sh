#!/usr/bin/env bash
set -euo pipefail

# Runs a compact 1-GPU experiment batch inside a RunPod/Linux node.
# Expected cwd: /workspace/parameter-golf

cd "${WORKDIR:-/workspace/parameter-golf}"
mkdir -p logs

run_one() {
  local run_id="$1"
  shift
  echo
  echo "=== RUN: ${run_id} ==="
  RUN_ID="${run_id}" "$@"
}

# 1) Control, int4 mixed, smaller keep-float budget
run_one ctrl_int4_kf131k_1gpu \
  env NPROC_PER_NODE=1 TARGET_GLOBAL_TOKENS=65536 \
      USE_SSM=0 USE_SWIGLU=1 MTP_ENABLED=0 \
      QUANT_SCHEME=mixed MIXED_LOW_PRECISION_SCHEME=int4 \
      MIXED_KEEP_FLOAT_MAX_NUMEL=131072 \
      QAT_SCHEME=int4 QAT_START_STEP=4200 \
      bash ./phase4_fixed_submission.sh

# 2) Control, int4 mixed, medium keep-float budget
run_one ctrl_int4_kf196k_1gpu \
  env NPROC_PER_NODE=1 TARGET_GLOBAL_TOKENS=65536 \
      USE_SSM=0 USE_SWIGLU=1 MTP_ENABLED=0 \
      QUANT_SCHEME=mixed MIXED_LOW_PRECISION_SCHEME=int4 \
      MIXED_KEEP_FLOAT_MAX_NUMEL=196608 \
      QAT_SCHEME=int4 QAT_START_STEP=4200 \
      bash ./phase4_fixed_submission.sh

# 3) Control, int4 mixed, aggressive keep-float budget
run_one ctrl_int4_kf262k_1gpu \
  env NPROC_PER_NODE=1 TARGET_GLOBAL_TOKENS=65536 \
      USE_SSM=0 USE_SWIGLU=1 MTP_ENABLED=0 \
      QUANT_SCHEME=mixed MIXED_LOW_PRECISION_SCHEME=int4 \
      MIXED_KEEP_FLOAT_MAX_NUMEL=262144 \
      QAT_SCHEME=int4 QAT_START_STEP=4200 \
      bash ./phase4_fixed_submission.sh

# 4) Optional MTP comparison, same export strategy
run_one mtp_int4_kf196k_1gpu \
  env NPROC_PER_NODE=1 TARGET_GLOBAL_TOKENS=65536 \
      USE_SSM=0 USE_SWIGLU=1 \
      MTP_ENABLED=1 MTP_STEPS=2 MTP_WEIGHT=0.10 MTP_DECAY=0.8 MTP_TIE_EMBEDDINGS=1 MTP_LR=0.01 \
      QUANT_SCHEME=mixed MIXED_LOW_PRECISION_SCHEME=int4 \
      MIXED_KEEP_FLOAT_MAX_NUMEL=196608 \
      QAT_SCHEME=int4 QAT_START_STEP=4200 \
      bash ./phase4_fixed_submission.sh


echo
echo "=== SUMMARY ==="
for f in \
  logs/ctrl_int4_kf131k_1gpu.txt \
  logs/ctrl_int4_kf196k_1gpu.txt \
  logs/ctrl_int4_kf262k_1gpu.txt \
  logs/mtp_int4_kf196k_1gpu.txt
  do
    echo "=== ${f} ==="
    grep -E '^submission_budget .*total:|^final_.*roundtrip_exact|^step:[0-9]+/[0-9]+ val_loss' "${f}" | tail -n 8 || true
  done
