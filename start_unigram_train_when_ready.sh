#!/usr/bin/env bash
set -euo pipefail

cd /workspace/parameter-golf

UNIGRAM_DATA="/workspace/parameter-golf/data/dual_unigram/datasets/fineweb10B_sp8192"
UNIGRAM_TOK="/workspace/parameter-golf/data/tokenizers/fineweb_8192_unigram_20260422_225958.model"
WAIT_SHARDS="${WAIT_SHARDS:-80}"
SEED="${1:-${SEED:-1337}}"
DISTILL_START_STEP="${DISTILL_START_STEP:-6500}"

RUN_GROUP="${RUN_GROUP:-sp8192_unigram_parallel_s${SEED}_$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-sp8192_unigram_submission_8gpu_jepa_s${SEED}_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "logs/runs/${RUN_GROUP}" "artifacts/${RUN_GROUP}"

echo "[wait] waiting for unigram shards in ${UNIGRAM_DATA} ..."
while true; do
  c="$(find "${UNIGRAM_DATA}" -maxdepth 1 -name 'fineweb_train_*.bin' | wc -l)"
  echo "[wait] train shards: ${c}/${WAIT_SHARDS}"
  if [ "${c}" -ge "${WAIT_SHARDS}" ]; then
    break
  fi
  sleep 5
done

echo "[start] unigram ready, launching training RUN_ID=${RUN_ID} SEED=${SEED}"

RUN_ID="${RUN_ID}" \
VOCAB_SIZE=8192 \
DATA_PATH="${UNIGRAM_DATA}" \
TOKENIZER_PATH="${UNIGRAM_TOK}" \
TARGET_GPUS=8 NPROC_PER_NODE=8 \
DEVICE=cuda USE_TORCH_COMPILE=1 SDP_BACKEND_MODE=flash \
ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
SEED="${SEED}" GRAD_ACCUM_STEPS=1 \
TRAIN_BATCH_TOKENS=524288 VAL_BATCH_SIZE=524288 VAL_LOSS_EVERY=0 VAL_MAX_TOKENS=0 FINAL_ROUNDTRIP_EVAL=1 \
MODEL_DIM=448 NUM_LAYERS=9 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2 TIE_EMBEDDINGS=1 \
RECURRENT_CORE_LAYERS=0 RECURRENT_STEPS=0 INTRA_LOOP_START=3 INTRA_LOOP_END=5 INTRA_LOOP_STEPS=2 \
USE_SWIGLU=1 PARALLEL_RESIDUAL=0 USE_SSM=0 MTP_ENABLED=0 MOE_NUM_EXPERTS=0 COPY_CACHE_ENABLED=0 BYTE_WEIGHTED_LOSS_ENABLED=0 \
DISTILL_ENABLED=1 DISTILL_START_STEP="${DISTILL_START_STEP}" DISTILL_START_FRAC=-1 DISTILL_START_WALLCLOCK_FRAC=-1 DISTILL_WEIGHT=0.08 DISTILL_TEMP=2.0 DISTILL_EMA_DECAY=0.999 \
JPCR_ENABLED=1 JPCR_HIDDEN=64 JPCR_PROJ_DIM=32 JPCR_WEIGHT=0.08 JPCR_BLEND_INIT=-2.0 JPCR_LR=0.02 JPCR_WARMUP_STEPS=100 \
SWA_ENABLED=1 SWA_COLLECT_EVERY=10 EVAL_STRIDE_FRAC=0.5 \
QUANT_SCHEME=int8 COMPRESSOR=zstd QAT_SCHEME=none QAT_LSQ=0 \
GPTQ=1 GPTQ_NSAMPLES=128 GPTQ_BLOCKSIZE=128 GPTQ_PERCDAMP=0.01 \
BIGRAM_RANK=0 RESIDUAL_NGRAM_ENABLED=0 TTT_ENABLED=0 \
bash ./phase4_fixed_submission.sh | tee "logs/runs/${RUN_GROUP}/${RUN_ID}.console.txt"

cp "logs/${RUN_ID}.txt" "logs/runs/${RUN_GROUP}/${RUN_ID}.train.txt" || true
grep -E '^stopping_early:|^step:[0-9]+/[0-9]+ val_loss:|^final_.*_roundtrip_exact .*val_bpb:|^submission_budget .*total:.*budget:' \
  "logs/${RUN_ID}.txt" > "logs/runs/${RUN_GROUP}/${RUN_ID}.summary.txt" || true

for f in final_model.pt final_model.int8.zstd.ptc final_model.int8.ptz final_model.int5.zstd.ptc final_model.int4.zstd.ptc; do
  [ -f "$f" ] && mv "$f" "artifacts/${RUN_GROUP}/${RUN_ID}.${f}"
done

echo "[done] logs: logs/runs/${RUN_GROUP}  artifacts: artifacts/${RUN_GROUP}"
