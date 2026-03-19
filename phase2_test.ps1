# Phase 2 moderate sanity run (faster than baseline, but does real training)
$env:RUN_ID = "phase2_moderate_sanity"
$env:DEVICE = "cuda"
$env:USE_TORCH_COMPILE = "0"

$env:ITERATIONS = "150"
$env:WARMUP_STEPS = "10"
$env:TRAIN_BATCH_TOKENS = "8192"

$env:VAL_LOSS_EVERY = "0"
$env:VAL_BATCH_SIZE = "131072"
$env:VAL_MAX_TOKENS = "262144"
$env:FINAL_ROUNDTRIP_EVAL = "1"

$env:QUANT_SCHEME = "int8"          # try int8/int4/mixed
$env:COMPRESSOR = "zlib"            # try zlib/auto/zstd
$env:WEIGHT_ORDER = "none"          # try none/name/size_desc/dtype_name
$env:MIXED_LOW_PRECISION_SCHEME = "int8"

python train_gpt.py
