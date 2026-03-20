#!/usr/bin/env bash
set -euo pipefail

# 6-minute submission-style run targeting 8xH100 (auto-falls back to available GPU count).
# Uses the current best Muon profile from local sweeps (mom_098).
#
# Usage:
#   bash ./phase3_8xh100_submission_6min.sh
#
# Optional overrides:
#   NPROC_PER_NODE=8
#   MAX_WALLCLOCK_SECONDS=360
#   TRAIN_BATCH_TOKENS=131072
#   VAL_BATCH_SIZE=262144
#   USE_TORCH_COMPILE=0
#   FINAL_ROUNDTRIP_EVAL=1
#   RUN_ID=custom_name

cd "$(dirname "$0")"
mkdir -p logs

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

if ! command -v torchrun >/dev/null 2>&1; then
  echo "torchrun not found. Run setup first: bash ./setup_h100_env_and_data.sh" >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('zstandard') else 1)" >/dev/null 2>&1; then
  echo "zstandard is missing. Install deps first: bash ./setup_h100_env_and_data.sh" >&2
  exit 1
fi

if [[ ! -f "./data/tokenizers/fineweb_1024_bpe.model" ]]; then
  echo "Tokenizer not found at ./data/tokenizers/fineweb_1024_bpe.model" >&2
  echo "Run: bash ./setup_h100_env_and_data.sh" >&2
  exit 1
fi
if [[ ! -d "./data/datasets/fineweb10B_sp1024" ]]; then
  echo "Dataset path missing at ./data/datasets/fineweb10B_sp1024" >&2
  echo "Run: bash ./setup_h100_env_and_data.sh" >&2
  exit 1
fi

gpu_count="$("${PYTHON_BIN}" -c "import torch; print(torch.cuda.device_count())")"
if [[ -z "${gpu_count}" || "${gpu_count}" -lt 1 ]]; then
  echo "No CUDA GPUs detected." >&2
  exit 1
fi

target_gpus="${TARGET_GPUS:-8}"
if [[ -n "${NPROC_PER_NODE:-}" ]]; then
  nproc_per_node="${NPROC_PER_NODE}"
else
  if [[ "${gpu_count}" -ge "${target_gpus}" ]]; then
    nproc_per_node="${target_gpus}"
  else
    nproc_per_node="${gpu_count}"
  fi
fi

if [[ "${nproc_per_node}" -lt 1 ]]; then
  echo "NPROC_PER_NODE must be >=1, got ${nproc_per_node}" >&2
  exit 1
fi
if [[ "${nproc_per_node}" -gt "${gpu_count}" ]]; then
  echo "NPROC_PER_NODE=${nproc_per_node} exceeds visible GPUs=${gpu_count}" >&2
  exit 1
fi
if [[ "${nproc_per_node}" -lt 8 ]]; then
  echo "Warning: running with ${nproc_per_node} GPU(s), not full 8xH100."
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
export RUN_ID="${RUN_ID:-phase3_8xh100_6min_${timestamp}}"

# Core runtime
export DEVICE="${DEVICE:-cuda}"
export USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-0}"
export SDP_BACKEND_MODE="${SDP_BACKEND_MODE:-auto}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Fixed data/tokenizer lane
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

# 6-minute wallclock regime
export ITERATIONS="${ITERATIONS:-20000}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-360}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}"

# Keep global train tokens close to known-good 2xH100 regime by default.
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
local_train_tokens_per_rank="${LOCAL_TRAIN_TOKENS_PER_RANK:-8192}"
local_val_tokens_per_rank="${LOCAL_VAL_TOKENS_PER_RANK:-16384}"
default_train_batch_tokens=$(( nproc_per_node * GRAD_ACCUM_STEPS * local_train_tokens_per_rank ))
default_val_batch_size=$(( nproc_per_node * GRAD_ACCUM_STEPS * local_val_tokens_per_rank ))
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-${default_train_batch_tokens}}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${default_val_batch_size}}"

# Validation/export
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export VAL_MAX_TOKENS="${VAL_MAX_TOKENS:-1048576}"
export FINAL_ROUNDTRIP_EVAL="${FINAL_ROUNDTRIP_EVAL:-1}"
export SUBMISSION_SIZE_BUDGET_BYTES="${SUBMISSION_SIZE_BUDGET_BYTES:-16000000}"
export QUANT_SCHEME="${QUANT_SCHEME:-int8}"
export MIXED_LOW_PRECISION_SCHEME="${MIXED_LOW_PRECISION_SCHEME:-int8}"
export COMPRESSOR="${COMPRESSOR:-zstd}"
export WEIGHT_ORDER="${WEIGHT_ORDER:-none}"

# Current best architecture lane
export MODEL_DIM="${MODEL_DIM:-1152}"
export NUM_LAYERS="${NUM_LAYERS:-9}"
export RECURRENT_CORE_LAYERS="${RECURRENT_CORE_LAYERS:-3}"
export RECURRENT_STEPS="${RECURRENT_STEPS:-6}"
export SHARE_FFN_ACROSS_BLOCKS="${SHARE_FFN_ACROSS_BLOCKS:-1}"
export NUM_HEADS="${NUM_HEADS:-8}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-4}"

# Best Muon profile from latest sweep: mom_098
export MUON_MOMENTUM="${MUON_MOMENTUM:-0.98}"
export MUON_BACKEND_STEPS="${MUON_BACKEND_STEPS:-5}"
export MUON_MOMENTUM_WARMUP_START="${MUON_MOMENTUM_WARMUP_START:-0.85}"
export MUON_MOMENTUM_WARMUP_STEPS="${MUON_MOMENTUM_WARMUP_STEPS:-500}"

# Repro
export SEED="${SEED:-1337}"

log_path="logs/${RUN_ID}.txt"
echo "Launching run_id=${RUN_ID} nproc_per_node=${nproc_per_node} (visible_gpus=${gpu_count})"
echo "wallclock=${MAX_WALLCLOCK_SECONDS}s train_batch_tokens=${TRAIN_BATCH_TOKENS} val_batch_size=${VAL_BATCH_SIZE} grad_accum_steps=${GRAD_ACCUM_STEPS}"
echo "model_dim=${MODEL_DIM} heads=${NUM_HEADS} kv_heads=${NUM_KV_HEADS} muon_momentum=${MUON_MOMENTUM}"

set +e
torchrun --standalone --nnodes=1 --nproc_per_node="${nproc_per_node}" train_gpt.py
exit_code=$?
set -e

if [[ -f "${log_path}" ]]; then
  echo
  echo "=== Final score + budget (${log_path}) ==="
  grep -E '^final_.*_roundtrip_exact .*val_bpb:|^submission_budget .*total:.*budget:' "${log_path}" | tail -n 4 || true
else
  echo "Run log not found at ${log_path}" >&2
fi

exit "${exit_code}"
