#!/usr/bin/env bash
set -euo pipefail

cd /workspace/parameter-golf

BPE_DATA="${BPE_DATA:-/workspace/parameter-golf/data/dual_bpe/datasets/fineweb10B_sp8192}"
BPE_TOK="${BPE_TOK:-/workspace/parameter-golf/data/tokenizers/fineweb_8192_bpe.model}"
WAIT_SHARDS="${WAIT_SHARDS:-80}"
SEED="${1:-${SEED:-1337}}"

# 25 minutes
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-1500}"
ITERATIONS="${ITERATIONS:-50000}"

# JPCR activation timing for this run.
DISTILL_START_WALLCLOCK_FRAC="${DISTILL_START_WALLCLOCK_FRAC:-0.65}"
DISTILL_START_STEP="${DISTILL_START_STEP:--1}"

# Apply distill/JPCR every Nth step (correct; no stale-target cache reuse).
JPCR_APPLY_EVERY="${JPCR_APPLY_EVERY:-2}"

RUN_GROUP="${RUN_GROUP:-sp8192_bpe_25m_jepa_s${SEED}_$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-sp8192_bpe_submission_8gpu_25m_jepa_s${SEED}_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "logs/runs/${RUN_GROUP}" "artifacts/${RUN_GROUP}" ".cache/torchinductor" ".cache/triton"

export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/workspace/parameter-golf/.cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/parameter-golf/.cache/triton}"
# Safer compile mode for fixed-shape training + phase changes (avoid over-dynamic traces).
export TORCH_COMPILE_DYNAMIC="${TORCH_COMPILE_DYNAMIC:-none}"
# We zero-touch conditional params in loss, so this can be off for speed.
export DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-false}"

echo "[wait] waiting for BPE shards in ${BPE_DATA} ..."
while true; do
  c="$(find "${BPE_DATA}" -maxdepth 1 -name 'fineweb_train_*.bin' | wc -l)"
  echo "[wait] train shards: ${c}/${WAIT_SHARDS}"
  if [ "${c}" -ge "${WAIT_SHARDS}" ]; then
    break
  fi
  sleep 5
done

echo "[start] launching 25m BPE JEPA run RUN_ID=${RUN_ID} SEED=${SEED}"
echo "[start] JPCR trigger frac=${DISTILL_START_WALLCLOCK_FRAC} apply_every=${JPCR_APPLY_EVERY} compile_dynamic=${TORCH_COMPILE_DYNAMIC} ddp_find_unused=${DDP_FIND_UNUSED_PARAMETERS}"

RUN_ID="${RUN_ID}" \
VOCAB_SIZE=8192 \
DATA_PATH="${BPE_DATA}" \
TOKENIZER_PATH="${BPE_TOK}" \
TARGET_GPUS=8 NPROC_PER_NODE=8 \
DEVICE=cuda USE_TORCH_COMPILE=1 SDP_BACKEND_MODE=flash \
ITERATIONS="${ITERATIONS}" MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS}" \
SEED="${SEED}" GRAD_ACCUM_STEPS=1 \
TRAIN_BATCH_TOKENS=524288 VAL_BATCH_SIZE=524288 VAL_LOSS_EVERY=0 VAL_MAX_TOKENS=0 FINAL_ROUNDTRIP_EVAL=1 \
MODEL_DIM=448 NUM_LAYERS=9 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2 TIE_EMBEDDINGS=1 \
RECURRENT_CORE_LAYERS=0 RECURRENT_STEPS=0 INTRA_LOOP_START=3 INTRA_LOOP_END=5 INTRA_LOOP_STEPS=2 \
USE_SWIGLU=1 PARALLEL_RESIDUAL=0 USE_SSM=0 MTP_ENABLED=0 MOE_NUM_EXPERTS=0 COPY_CACHE_ENABLED=0 BYTE_WEIGHTED_LOSS_ENABLED=0 \
DISTILL_ENABLED=1 DISTILL_START_STEP="${DISTILL_START_STEP}" DISTILL_START_FRAC=-1 DISTILL_START_WALLCLOCK_FRAC="${DISTILL_START_WALLCLOCK_FRAC}" DISTILL_WEIGHT=0.08 DISTILL_TEMP=2.0 DISTILL_EMA_DECAY=0.999 \
JPCR_ENABLED=1 JPCR_HIDDEN=64 JPCR_PROJ_DIM=32 JPCR_WEIGHT=0.08 JPCR_BLEND_INIT=-2.0 JPCR_LR=0.02 JPCR_WARMUP_STEPS=100 JPCR_APPLY_EVERY="${JPCR_APPLY_EVERY}" \
SWA_ENABLED=1 SWA_COLLECT_EVERY=10 EVAL_STRIDE_FRAC=0.5 \
QUANT_SCHEME=int8 COMPRESSOR=zstd QAT_SCHEME=none QAT_LSQ=0 \
GPTQ=1 GPTQ_NSAMPLES=128 GPTQ_BLOCKSIZE=128 GPTQ_PERCDAMP=0.01 \
BIGRAM_RANK=0 RESIDUAL_NGRAM_ENABLED=0 TTT_ENABLED=0 \
bash ./phase4_fixed_submission.sh | tee "logs/runs/${RUN_GROUP}/${RUN_ID}.console.txt"

cp "logs/${RUN_ID}.txt" "logs/runs/${RUN_GROUP}/${RUN_ID}.train.txt" || true
grep -E '^distill_start:|^jpcr_apply_every:|^stopping_early:|^step:[0-9]+/[0-9]+ val_loss:|^final_.*_roundtrip_exact .*val_bpb:|^submission_budget .*total:.*budget:' \
  "logs/${RUN_ID}.txt" > "logs/runs/${RUN_GROUP}/${RUN_ID}.summary.txt" || true

for f in final_model.pt final_model.int8.zstd.ptc final_model.int8.ptz final_model.int5.zstd.ptc final_model.int4.zstd.ptc; do
  [ -f "$f" ] && mv "$f" "artifacts/${RUN_GROUP}/${RUN_ID}.${f}"
done

echo "[done] logs: logs/runs/${RUN_GROUP}  artifacts: artifacts/${RUN_GROUP}"
