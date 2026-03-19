# Phase 2 moderate sanity run (faster than baseline, but does real training)
# This script sets defaults but respects pre-set env vars from the caller.
function Set-DefaultEnv([string]$Name, [string]$Value) {
    $current = [Environment]::GetEnvironmentVariable($Name, "Process")
    if ([string]::IsNullOrWhiteSpace($current)) {
        [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
    }
}

Set-DefaultEnv "RUN_ID" "phase2_moderate_sanity"
Set-DefaultEnv "DEVICE" "cuda"
Set-DefaultEnv "USE_TORCH_COMPILE" "0"

Set-DefaultEnv "ITERATIONS" "150"
Set-DefaultEnv "WARMUP_STEPS" "10"
Set-DefaultEnv "TRAIN_BATCH_TOKENS" "8192"

Set-DefaultEnv "VAL_LOSS_EVERY" "0"
Set-DefaultEnv "VAL_BATCH_SIZE" "131072"
Set-DefaultEnv "VAL_MAX_TOKENS" "262144"
Set-DefaultEnv "FINAL_ROUNDTRIP_EVAL" "1"

Set-DefaultEnv "QUANT_SCHEME" "int8"          # try int8/int4/mixed
Set-DefaultEnv "COMPRESSOR" "zlib"            # try zlib/auto/zstd
Set-DefaultEnv "WEIGHT_ORDER" "none"          # try none/name/size_desc/dtype_name
Set-DefaultEnv "MIXED_LOW_PRECISION_SCHEME" "int8"

python train_gpt.py
