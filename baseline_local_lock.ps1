# Locked local baseline run (RTX 3050-friendly)

# Clear smoke overrides
Remove-Item Env:VAL_MAX_TOKENS -ErrorAction SilentlyContinue
Remove-Item Env:SDP_BACKEND_MODE -ErrorAction SilentlyContinue
Remove-Item Env:FINAL_INT8_ROUNDTRIP_EVAL -ErrorAction SilentlyContinue

# Baseline config
$env:RUN_ID = "baseline_local_lock"
$env:DEVICE = "cuda"
$env:USE_TORCH_COMPILE = "0"
$env:ITERATIONS = "400"
$env:WARMUP_STEPS = "20"
$env:TRAIN_BATCH_TOKENS = "8192"
$env:VAL_LOSS_EVERY = "0"
$env:VAL_BATCH_SIZE = "262144"
$env:FINAL_INT8_ROUNDTRIP_EVAL = "1"

python train_gpt.py
