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

gpu_count="$("${PYTHON_BIN:-python}" -c "import torch; print(torch.cuda.device_count())")"
if [[ -z "${gpu_count}" || "${gpu_count}" -lt 1 ]]; then
  echo "No CUDA GPU detected. This script requires at least 1 CUDA GPU." >&2
  exit 1
fi

# Auto-select 1 or 2 workers based on available GPUs; allow manual override.
if [[ -n "${NPROC_PER_NODE:-}" ]]; then
  nproc_per_node="${NPROC_PER_NODE}"
else
  if [[ "${gpu_count}" -ge 2 ]]; then
    nproc_per_node="2"
  else
    nproc_per_node="1"
  fi
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

# Throughput tuning (auto-sized for 1x/2x GPU)
if [[ "${nproc_per_node}" -ge 2 ]]; then
  export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-131072}"
  export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-262144}"
else
  export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-65536}"
  export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-131072}"
fi
export GRAD_ACCUM_STEPS="2"

# Validation / export
export VAL_LOSS_EVERY="0"
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

echo "Launching H100 run: RUN_ID=${RUN_ID}"
echo "CUDA GPUs detected=${gpu_count} using nproc_per_node=${nproc_per_node}"
echo "Config: TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS} GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS} MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS}"
echo "Config: MODEL_DIM=${MODEL_DIM} NUM_HEADS=${NUM_HEADS} NUM_KV_HEADS=${NUM_KV_HEADS} RECURRENT_CORE_LAYERS=${RECURRENT_CORE_LAYERS} RECURRENT_STEPS=${RECURRENT_STEPS}"

torchrun --standalone --nnodes=1 --nproc_per_node="${nproc_per_node}" train_gpt.py
