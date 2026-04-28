#!/usr/bin/env bash
set -euo pipefail

cd /workspace/parameter-golf

# ---------- Config ----------
BPE_DATA="/workspace/parameter-golf/data/dual_bpe/datasets/fineweb10B_sp8192"
BPE_TOK="/workspace/parameter-golf/data/tokenizers/fineweb_8192_bpe.model"
MAX_SHARDS="${MAX_TRAIN_SHARDS:-80}"

RUN_GROUP="${RUN_GROUP:-sp8192_parallel_8gpu_$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-sp8192_bpe_submission_8gpu_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "logs/runs/${RUN_GROUP}" "artifacts/${RUN_GROUP}"

# ---------- Guard: dataset ready ----------
train_count="$(find "$BPE_DATA" -maxdepth 1 -name 'fineweb_train_*.bin' | wc -l)"
if [ "$train_count" -lt "$MAX_SHARDS" ]; then
  echo "BPE not ready yet: train shards=$train_count expected>=$MAX_SHARDS"
  exit 1
fi

echo "BPE ready (train shards=$train_count). Starting 8-GPU training RUN_ID=$RUN_ID"

# ---------- 8-GPU submission run ----------
RUN_ID="$RUN_ID" \
VOCAB_SIZE=8192 \
DATA_PATH="$BPE_DATA" \
TOKENIZER_PATH="$BPE_TOK" \
TARGET_GPUS=8 NPROC_PER_NODE=8 \
DEVICE=cuda USE_TORCH_COMPILE=1 SDP_BACKEND_MODE=flash \
ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
SEED=1337 GRAD_ACCUM_STEPS=1 \
TRAIN_BATCH_TOKENS=524288 VAL_BATCH_SIZE=524288 VAL_LOSS_EVERY=0 VAL_MAX_TOKENS=0 FINAL_ROUNDTRIP_EVAL=1 \
MODEL_DIM=448 NUM_LAYERS=9 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2 TIE_EMBEDDINGS=1 \
RECURRENT_CORE_LAYERS=0 RECURRENT_STEPS=0 INTRA_LOOP_START=-1 INTRA_LOOP_END=-1 INTRA_LOOP_STEPS=1 \
USE_SWIGLU=1 PARALLEL_RESIDUAL=0 USE_SSM=0 MTP_ENABLED=0 MOE_NUM_EXPERTS=0 COPY_CACHE_ENABLED=0 BYTE_WEIGHTED_LOSS_ENABLED=0 \
DISTILL_ENABLED=0 DISTILL_START_STEP=-1 DISTILL_START_FRAC=-1 DISTILL_START_WALLCLOCK_FRAC=-1 \
SWA_ENABLED=1 SWA_COLLECT_EVERY=10 EVAL_STRIDE_FRAC=0.5 \
QUANT_SCHEME=int8 COMPRESSOR=zstd QAT_SCHEME=none QAT_LSQ=0 \
GPTQ=1 GPTQ_NSAMPLES=128 GPTQ_BLOCKSIZE=128 GPTQ_PERCDAMP=0.01 \
BIGRAM_RANK=0 RESIDUAL_NGRAM_ENABLED=0 TTT_ENABLED=0 \
bash ./phase4_fixed_submission.sh | tee "logs/runs/${RUN_GROUP}/${RUN_ID}.console.txt"

# ---------- Archive outputs ----------
cp "logs/${RUN_ID}.txt" "logs/runs/${RUN_GROUP}/${RUN_ID}.train.txt" || true
env | grep -E '^(RUN_ID|VOCAB_SIZE|DATA_PATH|TOKENIZER_PATH|TARGET_GPUS|NPROC_PER_NODE|TRAIN_BATCH_TOKENS|VAL_BATCH_SIZE|MODEL_DIM|NUM_LAYERS|DISTILL_ENABLED|QUANT_SCHEME|COMPRESSOR|GPTQ|MAX_WALLCLOCK_SECONDS)=' \
  > "logs/runs/${RUN_GROUP}/${RUN_ID}.env" || true
grep -E '^stopping_early:|^step:[0-9]+/[0-9]+ val_loss:|^final_.*_roundtrip_exact .*val_bpb:|^submission_budget .*total:.*budget:' \
  "logs/${RUN_ID}.txt" > "logs/runs/${RUN_GROUP}/${RUN_ID}.summary.txt" || true

for f in final_model.pt final_model.int8.zstd.ptc final_model.int8.ptz final_model.int5.zstd.ptc final_model.int4.zstd.ptc; do
  [ -f "$f" ] && mv "$f" "artifacts/${RUN_GROUP}/${RUN_ID}.${f}"
done

echo "Done. Logs: logs/runs/${RUN_GROUP}  Artifacts: artifacts/${RUN_GROUP}"
