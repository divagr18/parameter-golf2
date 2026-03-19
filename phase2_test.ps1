# Phase 2 fast export smoke test (minimal runtime)
$env:RUN_ID = "phase2_smoke_export"
$env:DEVICE = "cuda"
$env:USE_TORCH_COMPILE = "0"

$env:ITERATIONS = "0"
$env:WARMUP_STEPS = "0"
$env:TRAIN_BATCH_TOKENS = "8192"

$env:VAL_LOSS_EVERY = "0"
$env:VAL_BATCH_SIZE = "8192"
$env:VAL_MAX_TOKENS = "8192"
$env:FINAL_ROUNDTRIP_EVAL = "0"

$env:QUANT_SCHEME = "int4"          # try int8/int4/mixed
$env:COMPRESSOR = "zlib"            # try zlib/auto/zstd
$env:WEIGHT_ORDER = "name"          # try none/name/size_desc/dtype_name
$env:MIXED_LOW_PRECISION_SCHEME = "int8"

python train_gpt.py
