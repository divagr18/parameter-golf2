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
  nproc_per_node="${gpu_count}"
fi

if [[ "${nproc_per_node}" -lt 1 ]]; then
  echo "NPROC_PER_NODE must be >=1, got ${nproc_per_node}" >&2
  exit 1
fi
if [[ "${nproc_per_node}" -gt "${gpu_count}" ]]; then
  echo "NPROC_PER_NODE=${nproc_per_node} exceeds visible GPU count=${gpu_count}" >&2
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

export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
local_train_tokens_per_rank="${LOCAL_TRAIN_TOKENS_PER_RANK:-32768}"
local_val_tokens_per_rank="${LOCAL_VAL_TOKENS_PER_RANK:-65536}"
default_train_batch_tokens=$(( nproc_per_node * GRAD_ACCUM_STEPS * local_train_tokens_per_rank ))
default_val_batch_size=$(( nproc_per_node * GRAD_ACCUM_STEPS * local_val_tokens_per_rank ))

# Throughput tuning scales with world size by default; can still be overridden externally.
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-${default_train_batch_tokens}}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${default_val_batch_size}}"

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
echo "Config: TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS} VAL_BATCH_SIZE=${VAL_BATCH_SIZE} GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS} MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS}"
echo "Config: MODEL_DIM=${MODEL_DIM} NUM_HEADS=${NUM_HEADS} NUM_KV_HEADS=${NUM_KV_HEADS} RECURRENT_CORE_LAYERS=${RECURRENT_CORE_LAYERS} RECURRENT_STEPS=${RECURRENT_STEPS}"

torchrun --standalone --nnodes=1 --nproc_per_node="${nproc_per_node}" train_gpt.py
