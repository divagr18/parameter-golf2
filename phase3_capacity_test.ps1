# Phase 3 capacity sanity run (single config).
# Uses the recurrent 3x6 shared family and scales width by default.
# This script sets defaults but respects pre-set env vars from the caller.
function Set-DefaultEnv([string]$Name, [string]$Value) {
    $current = [Environment]::GetEnvironmentVariable($Name, "Process")
    if ([string]::IsNullOrWhiteSpace($current)) {
        [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
    }
}

Set-DefaultEnv "RUN_ID" "phase3_capacity_single"
Set-DefaultEnv "DEVICE" "cuda"
Set-DefaultEnv "USE_TORCH_COMPILE" "0"
Set-DefaultEnv "SEED" "1337"

Set-DefaultEnv "ITERATIONS" "500"
Set-DefaultEnv "WARMUP_STEPS" "20"
Set-DefaultEnv "TRAIN_BATCH_TOKENS" "8192"
Set-DefaultEnv "VAL_LOSS_EVERY" "0"
Set-DefaultEnv "VAL_BATCH_SIZE" "131072"
Set-DefaultEnv "VAL_MAX_TOKENS" "1048576"
Set-DefaultEnv "FINAL_ROUNDTRIP_EVAL" "1"
Set-DefaultEnv "SUBMISSION_SIZE_BUDGET_BYTES" "16777216"

Set-DefaultEnv "QUANT_SCHEME" "int8"
Set-DefaultEnv "COMPRESSOR" "auto"
Set-DefaultEnv "WEIGHT_ORDER" "none"
Set-DefaultEnv "MIXED_LOW_PRECISION_SCHEME" "int8"

# Capacity target config (change MODEL_DIM externally to sweep).
Set-DefaultEnv "MODEL_DIM" "768"
Set-DefaultEnv "NUM_LAYERS" "9"
Set-DefaultEnv "RECURRENT_CORE_LAYERS" "3"
Set-DefaultEnv "RECURRENT_STEPS" "6"
Set-DefaultEnv "SHARE_FFN_ACROSS_BLOCKS" "1"

python train_gpt.py
