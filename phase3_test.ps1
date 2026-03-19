# Phase 3 recurrent sanity run (defaults; can be overridden by pre-set env vars)
function Set-DefaultEnv([string]$Name, [string]$Value) {
    $current = [Environment]::GetEnvironmentVariable($Name, "Process")
    if ([string]::IsNullOrWhiteSpace($current)) {
        [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
    }
}

Set-DefaultEnv "RUN_ID" "phase3_recurrent_sanity"
Set-DefaultEnv "DEVICE" "cuda"
Set-DefaultEnv "USE_TORCH_COMPILE" "0"

Set-DefaultEnv "ITERATIONS" "150"
Set-DefaultEnv "WARMUP_STEPS" "10"
Set-DefaultEnv "TRAIN_BATCH_TOKENS" "8192"

Set-DefaultEnv "VAL_LOSS_EVERY" "0"
Set-DefaultEnv "VAL_BATCH_SIZE" "131072"
Set-DefaultEnv "VAL_MAX_TOKENS" "262144"
Set-DefaultEnv "FINAL_ROUNDTRIP_EVAL" "1"

# Phase 2 best export defaults while testing architecture changes
Set-DefaultEnv "QUANT_SCHEME" "int8"
Set-DefaultEnv "COMPRESSOR" "auto"
Set-DefaultEnv "WEIGHT_ORDER" "none"
Set-DefaultEnv "MIXED_LOW_PRECISION_SCHEME" "int8"

# Phase 3 defaults
Set-DefaultEnv "RECURRENT_CORE_LAYERS" "2"
Set-DefaultEnv "RECURRENT_STEPS" "6"
Set-DefaultEnv "SHARE_FFN_ACROSS_BLOCKS" "1"

python train_gpt.py
