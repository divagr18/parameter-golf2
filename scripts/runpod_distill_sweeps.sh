#!/usr/bin/env bash
set -euo pipefail

# Runs three phases on 1 GPU:
# 1) distill-weight sweep
# 2) temperature sweep (at chosen best distill weight)
# 3) logit-reg sweep (at chosen best distill weight/temp)
#
# Intended to be run on the pod. A local orchestrator can start/stop the pod around this.

cd "$(dirname "$0")/.."
mkdir -p logs

: "${NPROC_PER_NODE:=1}"
: "${TARGET_GLOBAL_TOKENS:=65536}"
: "${USE_SSM:=0}"
: "${USE_SWIGLU:=1}"
: "${MTP_ENABLED:=0}"
: "${QUANT_SCHEME:=mixed}"
: "${MIXED_LOW_PRECISION_SCHEME:=int4}"
: "${MIXED_KEEP_FLOAT_MAX_NUMEL:=229376}"
: "${QAT_SCHEME:=int4}"
: "${QAT_START_STEP:=4200}"

# Sweep knobs (can be overridden)
: "${DISTILL_START_FRAC:=0.7}"
: "${DISTILL_EMA_DECAY:=0.999}"
: "${DISTILL_WEIGHT_SWEEP:=0.04 0.08 0.12}"
: "${DISTILL_BEST_WEIGHT:=0.08}"
: "${DISTILL_TEMP_SWEEP:=1.2 1.5 1.8}"
: "${DISTILL_BEST_TEMP:=1.5}"
: "${LOGIT_REG_SWEEP:=0 1e-6 3e-6}"

run_case() {
  local run_id="$1"
  shift
  echo "=== Running ${run_id} ==="
  env \
    NPROC_PER_NODE="${NPROC_PER_NODE}" \
    TARGET_GLOBAL_TOKENS="${TARGET_GLOBAL_TOKENS}" \
    USE_SSM="${USE_SSM}" \
    USE_SWIGLU="${USE_SWIGLU}" \
    MTP_ENABLED="${MTP_ENABLED}" \
    QUANT_SCHEME="${QUANT_SCHEME}" \
    MIXED_LOW_PRECISION_SCHEME="${MIXED_LOW_PRECISION_SCHEME}" \
    MIXED_KEEP_FLOAT_MAX_NUMEL="${MIXED_KEEP_FLOAT_MAX_NUMEL}" \
    QAT_SCHEME="${QAT_SCHEME}" \
    QAT_START_STEP="${QAT_START_STEP}" \
    RUN_ID="${run_id}" \
    "$@" \
    bash ./phase4_fixed_submission.sh
}

summarize_log() {
  local f="$1"
  echo "=== ${f} ==="
  grep -E "^submission_budget .*total:|^final_.*roundtrip_exact|^step:[0-9]+/[0-9]+ val_loss" "${f}" | tail -n 8 || true
}

echo "### Phase 1: Distill weight sweep"
for w in ${DISTILL_WEIGHT_SWEEP}; do
  run_case "distill_w${w}_1gpu" \
    DISTILL_ENABLED=1 \
    DISTILL_START_FRAC="${DISTILL_START_FRAC}" \
    DISTILL_WEIGHT="${w}" \
    DISTILL_TEMP="${DISTILL_BEST_TEMP}" \
    DISTILL_EMA_DECAY="${DISTILL_EMA_DECAY}" \
    LOGIT_REG_WEIGHT=0.0
  summarize_log "logs/distill_w${w}_1gpu.txt"
done

echo "### Phase 2: Distill temperature sweep"
for t in ${DISTILL_TEMP_SWEEP}; do
  run_case "distill_w${DISTILL_BEST_WEIGHT}_t${t}_1gpu" \
    DISTILL_ENABLED=1 \
    DISTILL_START_FRAC="${DISTILL_START_FRAC}" \
    DISTILL_WEIGHT="${DISTILL_BEST_WEIGHT}" \
    DISTILL_TEMP="${t}" \
    DISTILL_EMA_DECAY="${DISTILL_EMA_DECAY}" \
    LOGIT_REG_WEIGHT=0.0
  summarize_log "logs/distill_w${DISTILL_BEST_WEIGHT}_t${t}_1gpu.txt"
done

echo "### Phase 3: Logit-reg sweep"
for lw in ${LOGIT_REG_SWEEP}; do
  run_case "distill_w${DISTILL_BEST_WEIGHT}_t${DISTILL_BEST_TEMP}_lr${lw}_1gpu" \
    DISTILL_ENABLED=1 \
    DISTILL_START_FRAC="${DISTILL_START_FRAC}" \
    DISTILL_WEIGHT="${DISTILL_BEST_WEIGHT}" \
    DISTILL_TEMP="${DISTILL_BEST_TEMP}" \
    DISTILL_EMA_DECAY="${DISTILL_EMA_DECAY}" \
    LOGIT_REG_WEIGHT="${lw}"
  summarize_log "logs/distill_w${DISTILL_BEST_WEIGHT}_t${DISTILL_BEST_TEMP}_lr${lw}_1gpu.txt"
done

echo "### Done: distill/temperature/logit sweeps complete"
