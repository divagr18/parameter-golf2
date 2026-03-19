# Phase 3 stronger single-run sanity test.
# Defaults can be overridden by pre-set env vars.
function Set-DefaultEnv([string]$Name, [string]$Value) {
    $current = [Environment]::GetEnvironmentVariable($Name, "Process")
    if ([string]::IsNullOrWhiteSpace($current)) {
        [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
    }
}

Set-DefaultEnv "RUN_ID" "phase3_recurrent_strong"
Set-DefaultEnv "DEVICE" "cuda"
Set-DefaultEnv "USE_TORCH_COMPILE" "0"
Set-DefaultEnv "SEED" "1337"

# Stronger validation settings
Set-DefaultEnv "ITERATIONS" "400"
Set-DefaultEnv "WARMUP_STEPS" "20"
Set-DefaultEnv "TRAIN_BATCH_TOKENS" "8192"
Set-DefaultEnv "VAL_LOSS_EVERY" "0"
Set-DefaultEnv "VAL_BATCH_SIZE" "131072"
Set-DefaultEnv "VAL_MAX_TOKENS" "1048576"
Set-DefaultEnv "FINAL_ROUNDTRIP_EVAL" "1"

# Keep best Phase 2 export defaults while testing architecture
Set-DefaultEnv "QUANT_SCHEME" "int8"
Set-DefaultEnv "COMPRESSOR" "auto"
Set-DefaultEnv "WEIGHT_ORDER" "none"
Set-DefaultEnv "MIXED_LOW_PRECISION_SCHEME" "int8"

# Capacity-matched recurrent defaults (not tiny)
Set-DefaultEnv "MODEL_DIM" "704"
Set-DefaultEnv "NUM_LAYERS" "9"
Set-DefaultEnv "RECURRENT_CORE_LAYERS" "2"
Set-DefaultEnv "RECURRENT_STEPS" "6"
Set-DefaultEnv "SHARE_FFN_ACROSS_BLOCKS" "1"

python train_gpt.py
