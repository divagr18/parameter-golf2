#!/usr/bin/env bash
set -euo pipefail

# Self-contained 2xH100 finalist/full-time run.
# Usage:
#   bash ./phase3_2xh100_full_regime.sh

cd "$(dirname "$0")"
mkdir -p logs

if ! command -v torchrun >/dev/null 2>&1; then
  echo "torchrun not found. Activate your venv with PyTorch installed." >&2
  exit 1
fi

if ! "${PYTHON_BIN:-python}" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('zstandard') else 1)" >/dev/null 2>&1; then
  echo "zstandard package is missing. Install with: ${PYTHON_BIN:-python} -m pip install zstandard" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"

# Core run id / device
export RUN_ID="phase3_2xh100_full_${timestamp}"
export DEVICE="cuda"
export USE_TORCH_COMPILE="0"
export SDP_BACKEND_MODE="auto"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Repro
export SEED="1337"

# Full-time regime (use wallclock cap)
export ITERATIONS="20000"
export MAX_WALLCLOCK_SECONDS="600"
export WARMUP_STEPS="40"
export TRAIN_LOG_EVERY="200"

# Throughput tuning for 2xH100
export TRAIN_BATCH_TOKENS="131072"
export GRAD_ACCUM_STEPS="2"

# Validation / export
export VAL_LOSS_EVERY="0"
export VAL_BATCH_SIZE="262144"
export VAL_MAX_TOKENS="1048576"
export FINAL_ROUNDTRIP_EVAL="1"
export SUBMISSION_SIZE_BUDGET_BYTES="16777216"
export QUANT_SCHEME="int8"
export COMPRESSOR="auto"
export WEIGHT_ORDER="none"
export MIXED_LOW_PRECISION_SCHEME="int8"

# Best-known architecture lane
export MODEL_DIM="1152"
export NUM_LAYERS="9"
export RECURRENT_CORE_LAYERS="3"
export RECURRENT_STEPS="6"
export SHARE_FFN_ACROSS_BLOCKS="1"
export NUM_HEADS="8"
export NUM_KV_HEADS="4"

echo "Launching 2xH100 run: RUN_ID=${RUN_ID}"
echo "Config: TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS} GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS} MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS}"
echo "Config: MODEL_DIM=${MODEL_DIM} NUM_HEADS=${NUM_HEADS} NUM_KV_HEADS=${NUM_KV_HEADS} RECURRENT_CORE_LAYERS=${RECURRENT_CORE_LAYERS} RECURRENT_STEPS=${RECURRENT_STEPS}"

torchrun --standalone --nnodes=1 --nproc_per_node=2 train_gpt.py
