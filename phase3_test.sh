#!/usr/bin/env bash
set -euo pipefail

set_default_env() {
  local name="$1"
  local value="$2"
  if [[ -z "${!name:-}" ]]; then
    export "${name}=${value}"
  fi
}

set_default_env RUN_ID "phase3_recurrent_default"
set_default_env DEVICE "cuda"
set_default_env USE_TORCH_COMPILE "0"
set_default_env SEED "1337"

set_default_env ITERATIONS "400"
set_default_env WARMUP_STEPS "20"
set_default_env TRAIN_BATCH_TOKENS "8192"
set_default_env VAL_LOSS_EVERY "0"
set_default_env VAL_BATCH_SIZE "131072"
set_default_env VAL_MAX_TOKENS "1048576"
set_default_env FINAL_ROUNDTRIP_EVAL "1"
set_default_env SUBMISSION_SIZE_BUDGET_BYTES "16777216"

set_default_env QUANT_SCHEME "int8"
set_default_env COMPRESSOR "auto"
set_default_env WEIGHT_ORDER "none"
set_default_env MIXED_LOW_PRECISION_SCHEME "int8"

set_default_env MODEL_DIM "640"
set_default_env NUM_LAYERS "9"
set_default_env RECURRENT_CORE_LAYERS "3"
set_default_env RECURRENT_STEPS "6"
set_default_env SHARE_FFN_ACROSS_BLOCKS "1"

"${PYTHON_BIN:-python}" train_gpt.py
