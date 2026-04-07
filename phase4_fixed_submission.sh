#!/usr/bin/env bash
set -euo pipefail

# Phase 4: Fixed submission script — corrects all 4 bugs from phase3:
#   Bug 1: MAX_WALLCLOCK_SECONDS was 360 (6 min), not 600 (10 min)
#   Bug 2: USE_TORCH_COMPILE was 0; H100 gets ~30-50% speedup from compile
#   Bug 3: TRAIN_BATCH_TOKENS was 131K (LOCAL=8192 × 8 GPUs × ACCUM=2); baseline used 524K
#   Bug 4: RECURRENT_STEPS=6 at MODEL_DIM=1152 → ~499ms/step → only ~1200 steps vs baseline's 13780
#
# Default config: pure GPT + Muon (highest confidence to beat 1.2244 baseline)
# Override arch via env:  MODEL_DIM=512 NUM_LAYERS=9  (default)
#                         MODEL_DIM=576 NUM_LAYERS=9  (try if fits budget)
#                         MODEL_DIM=448 NUM_LAYERS=12 (deeper-narrow)
# Override recurrence:    RECURRENT_CORE_LAYERS=3 RECURRENT_STEPS=2 MODEL_DIM=768 (light recur)
#
# Usage:
#   bash ./phase4_fixed_submission.sh
#   MODEL_DIM=576 bash ./phase4_fixed_submission.sh
#   USE_SWIGLU=1 bash ./phase4_fixed_submission.sh

cd "$(dirname "$0")"
mkdir -p logs

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

if ! command -v torchrun >/dev/null 2>&1; then
  echo "torchrun not found. Run: bash ./setup_h100_env_and_data.sh" >&2
  exit 1
fi

if [[ ! -f "./data/tokenizers/fineweb_1024_bpe.model" ]]; then
  echo "Tokenizer missing at ./data/tokenizers/fineweb_1024_bpe.model" >&2
  exit 1
fi
if [[ ! -d "./data/datasets/fineweb10B_sp1024" ]]; then
  echo "Dataset missing at ./data/datasets/fineweb10B_sp1024" >&2
  exit 1
fi

gpu_count="$("${PYTHON_BIN}" -c "import torch; print(torch.cuda.device_count())")"
if [[ -z "${gpu_count}" || "${gpu_count}" -lt 1 ]]; then
  echo "No CUDA GPUs detected." >&2
  exit 1
fi

target_gpus="${TARGET_GPUS:-8}"
if [[ -n "${NPROC_PER_NODE:-}" ]]; then
  nproc_per_node="${NPROC_PER_NODE}"
elif [[ "${gpu_count}" -ge "${target_gpus}" ]]; then
  nproc_per_node="${target_gpus}"
else
  nproc_per_node="${gpu_count}"
fi
if [[ "${nproc_per_node}" -lt "${target_gpus}" ]]; then
  echo "Warning: running with ${nproc_per_node}/${target_gpus} GPUs."
fi

# -----------------------------------------------------------------
# BUG FIX 3+4: Compute global batch properly for current GPU count.
# TARGET_GLOBAL_TOKENS (default 524288) is divided across GPUs × accum steps.
# Override with TARGET_GLOBAL_TOKENS=65536 for 1-GPU runs to avoid OOM.
# If TRAIN_BATCH_TOKENS is set directly it takes priority over the computed value.
# -----------------------------------------------------------------
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
seq_len=1024
if [[ -z "${TRAIN_BATCH_TOKENS:-}" ]]; then
  target_global_tokens="${TARGET_GLOBAL_TOKENS:-524288}"
  local_tokens=$(( target_global_tokens / (nproc_per_node * GRAD_ACCUM_STEPS) ))
  local_tokens=$(( (local_tokens / seq_len) * seq_len ))
  if [[ "${local_tokens}" -lt "${seq_len}" ]]; then
    local_tokens="${seq_len}"
  fi
  export TRAIN_BATCH_TOKENS=$(( nproc_per_node * GRAD_ACCUM_STEPS * local_tokens ))
else
  local_tokens=$(( TRAIN_BATCH_TOKENS / (nproc_per_node * GRAD_ACCUM_STEPS) ))
fi
export VAL_BATCH_SIZE=$(( nproc_per_node * GRAD_ACCUM_STEPS * local_tokens * 2 ))

timestamp="$(date +%Y%m%d_%H%M%S)"
export RUN_ID="${RUN_ID:-phase4_fixed_${timestamp}_s${SEED:-1337}}"

# -----------------------------------------------------------------
# Core runtime — BUG FIX 2: compile=1
# -----------------------------------------------------------------
export DEVICE="${DEVICE:-cuda}"
export USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-1}"          # FIX: was 0
export SDP_BACKEND_MODE="${SDP_BACKEND_MODE:-flash}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Data
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

# -----------------------------------------------------------------
# BUG FIX 1: wallclock = 600 (10 minutes), not 360 (6 minutes)
# -----------------------------------------------------------------
export ITERATIONS="${ITERATIONS:-20000}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"  # FIX: was 360
export WARMUP_STEPS="${WARMUP_STEPS:-200}"  # longer warmup avoids Muon overshoot spike at step 2
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}"

# Validation
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export VAL_MAX_TOKENS="${VAL_MAX_TOKENS:-0}"                 # full val set
export FINAL_ROUNDTRIP_EVAL="${FINAL_ROUNDTRIP_EVAL:-1}"
export SUBMISSION_SIZE_BUDGET_BYTES="${SUBMISSION_SIZE_BUDGET_BYTES:-16000000}"

# Export / compression
export QUANT_SCHEME="${QUANT_SCHEME:-int8}"
export COMPRESSOR="${COMPRESSOR:-zstd}"
export WEIGHT_ORDER="${WEIGHT_ORDER:-none}"
export MIXED_LOW_PRECISION_SCHEME="${MIXED_LOW_PRECISION_SCHEME:-int8}"

# -----------------------------------------------------------------
# BUG FIX 4: Architecture — pure GPT, no recurrence.
# Recurrence at STEPS=6 gave ~499ms/step (only ~1200 steps in 600s).
# Baseline vanilla transformer gets ~43ms/step (~13800 steps).
# Muon alone on the same-size model should beat 1.2244 baseline bpb.
# -----------------------------------------------------------------
export MODEL_DIM="${MODEL_DIM:-512}"
export NUM_LAYERS="${NUM_LAYERS:-9}"
export NUM_HEADS="${NUM_HEADS:-8}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
export MLP_MULT="${MLP_MULT:-2}"
export TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-1}"
export RECURRENT_CORE_LAYERS="${RECURRENT_CORE_LAYERS:-0}"   # FIX: was 3
export RECURRENT_STEPS="${RECURRENT_STEPS:-0}"               # FIX: was 6
export SHARE_FFN_ACROSS_BLOCKS="${SHARE_FFN_ACROSS_BLOCKS:-0}"

# SwiGLU: better activation than relu² at same parameter budget
export USE_SWIGLU="${USE_SWIGLU:-0}"

# Mixture of Experts (MoE): replace dense MLPs with sparse expert routing.
# MOE_NUM_EXPERTS=0 → disabled (dense).  2+ → Expert Choice routing.
# MOE_EVERY_N=1 → all layers MoE; =2 → alternating even layers; =3 → every 3rd.
# MOE_CAPACITY_FACTOR: tokens each expert sees = int(cf * B*T / E); 1.0 = balanced.
# MOE_AUX_LOSS_COEFF: router Z-loss weight (prevents routing collapse).
export MOE_NUM_EXPERTS="${MOE_NUM_EXPERTS:-0}"
export MOE_EVERY_N="${MOE_EVERY_N:-2}"
export MOE_CAPACITY_FACTOR="${MOE_CAPACITY_FACTOR:-1.0}"
export MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-1e-3}"

# Quantization-Aware Training (QAT): fake-quantise weights late in training.
# QAT_SCHEME: "none" | "int8" | "int4"  — should match QUANT_SCHEME at export.
# QAT_START_STEP: delay QAT until ~65-75% of expected total steps.
#   int4 uses a 3-stage progressive schedule (256→64→16 levels) to avoid spikes.
export QAT_SCHEME="${QAT_SCHEME:-none}"
export QAT_START_STEP="${QAT_START_STEP:-9000}"
# QAT_LSQ=1 enables Learned Step-Size Quantization: per-row learnable
# log-scale trained through STE during QAT, then exported as the int4/int8
# packing scale (no extra eval-time cost). Targets ~0.025 quant penalty
# vs ~0.054 for baseline progressive QAT.
export QAT_LSQ="${QAT_LSQ:-0}"

# Optional hybrid SSM blocks: replace every Nth attention block with an SSM-style mixer.
export USE_SSM="${USE_SSM:-0}"
export SSM_EVERY_N="${SSM_EVERY_N:-2}"
export SSM_EXPAND="${SSM_EXPAND:-2.0}"
export SSM_KERNEL="${SSM_KERNEL:-4}"

# Multi-token prediction (training-only auxiliary loss).
export MTP_ENABLED="${MTP_ENABLED:-0}"
export MTP_STEPS="${MTP_STEPS:-2}"
export MTP_WEIGHT="${MTP_WEIGHT:-0.3}"
export MTP_DECAY="${MTP_DECAY:-1.0}"
export MTP_TIE_EMBEDDINGS="${MTP_TIE_EMBEDDINGS:-1}"
export MTP_LR="${MTP_LR:-0.02}"

# On-the-fly distillation + logit range regularization
export DISTILL_ENABLED="${DISTILL_ENABLED:-0}"
export DISTILL_START_FRAC="${DISTILL_START_FRAC:-0.7}"
export DISTILL_WEIGHT="${DISTILL_WEIGHT:-0.1}"
export DISTILL_TEMP="${DISTILL_TEMP:-1.5}"
export DISTILL_EMA_DECAY="${DISTILL_EMA_DECAY:-0.999}"
export LOGIT_REG_WEIGHT="${LOGIT_REG_WEIGHT:-0.0}"
export BYTE_WEIGHTED_LOSS_ENABLED="${BYTE_WEIGHTED_LOSS_ENABLED:-0}"
export BYTE_WEIGHTED_LOSS_ALPHA="${BYTE_WEIGHTED_LOSS_ALPHA:-1.0}"

# Gradient clipping: helps stability, especially with Muon at high momentum
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"

# -----------------------------------------------------------------------
# MAJOR IMPROVEMENTS
# -----------------------------------------------------------------------
# 1. Sliding window eval: only score tokens with ≥ prefix context.
#    EVAL_STRIDE_FRAC=0.5 → stride=512, prefix=512 context guaranteed per token.
#    EVAL_STRIDE_FRAC=1.0 (default) = original non-overlapping behavior.
export EVAL_STRIDE_FRAC="${EVAL_STRIDE_FRAC:-0.5}"

# 2. Long-context eval: evaluate at longer sequence than training.
#    0 = same as TRAIN_SEQ_LEN.  E.g. EVAL_SEQ_LEN=2048 with EVAL_ROPE_SCALE=4.
export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-0}"
export EVAL_ROPE_SCALE="${EVAL_ROPE_SCALE:-1.0}"

# 3. Low-rank bigram logit bias: learnable factored n-gram prior on top of neural model.
#    BIGRAM_RANK=32 adds ~64K int8 params (≈32KB), well within the 164KB budget headroom.
#    Set to 0 to disable.
export BIGRAM_RANK="${BIGRAM_RANK:-32}"
export BIGRAM_LR="${BIGRAM_LR:-0.04}"

# 3b. Residual n-gram modeling: mixture of neural LM and cheap n-gram baseline.
#     Keep off by default; enable for focused sweeps.
export RESIDUAL_NGRAM_ENABLED="${RESIDUAL_NGRAM_ENABLED:-0}"
export RESIDUAL_BIGRAM_RANK="${RESIDUAL_BIGRAM_RANK:-0}"
export RESIDUAL_TRIGRAM_RANK="${RESIDUAL_TRIGRAM_RANK:-0}"
export RESIDUAL_NGRAM_LR="${RESIDUAL_NGRAM_LR:-0.04}"
export RESIDUAL_NGRAM_MIX_INIT="${RESIDUAL_NGRAM_MIX_INIT:--2.5}"

# Pointer-style local copy/cache head.
export COPY_CACHE_ENABLED="${COPY_CACHE_ENABLED:-0}"
export COPY_CACHE_WINDOW="${COPY_CACHE_WINDOW:-256}"
export COPY_CACHE_DIM="${COPY_CACHE_DIM:-64}"
export COPY_CACHE_LR="${COPY_CACHE_LR:-0.02}"
export COPY_CACHE_GATE_INIT="${COPY_CACHE_GATE_INIT:--4.0}"

# 4. SWA: average weights during warmdown (confirmed 0.5-1.5% gain, also improves quantization)
export SWA_ENABLED="${SWA_ENABLED:-1}"
export SWA_COLLECT_EVERY="${SWA_COLLECT_EVERY:-10}"

# 5. Sequence length curriculum (disabled by default; set to 1 to test 2-4% gain)
export CURRICULUM_ENABLED="${CURRICULUM_ENABLED:-0}"
export CURRICULUM_MIN_SEQ_LEN="${CURRICULUM_MIN_SEQ_LEN:-256}"
export CURRICULUM_STEPS="${CURRICULUM_STEPS:-5000}"

# Best Muon profile from phase3 sweep
export MUON_MOMENTUM="${MUON_MOMENTUM:-0.98}"
export MUON_BACKEND_STEPS="${MUON_BACKEND_STEPS:-5}"
export MUON_MOMENTUM_WARMUP_START="${MUON_MOMENTUM_WARMUP_START:-0.85}"
export MUON_MOMENTUM_WARMUP_STEPS="${MUON_MOMENTUM_WARMUP_STEPS:-500}"

# LR knobs (these are tuned for the Muon-dominant regime)
export MATRIX_LR="${MATRIX_LR:-0.04}"
export SCALAR_LR="${SCALAR_LR:-0.04}"
export EMBED_LR="${EMBED_LR:-0.6}"
export TIED_EMBED_LR="${TIED_EMBED_LR:-0.05}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-3000}"  # 3000 × 44ms ≈ 132s = 22% of 600s budget

export SEED="${SEED:-1337}"

log_path="logs/${RUN_ID}.txt"
echo "=== Phase 4 Fixed Submission ==="
echo "run_id:         ${RUN_ID}"
echo "nproc:          ${nproc_per_node}/${gpu_count} GPUs"
echo "wallclock:      ${MAX_WALLCLOCK_SECONDS}s  [FIX: was 360]"
echo "compile:        ${USE_TORCH_COMPILE}        [FIX: was 0]"
echo "train_batch:    ${TRAIN_BATCH_TOKENS} tokens [FIX: was ~131K]"
echo "grad_accum:     ${GRAD_ACCUM_STEPS}"
echo "model:          dim=${MODEL_DIM} layers=${NUM_LAYERS} heads=${NUM_HEADS} kv=${NUM_KV_HEADS} mlp_mult=${MLP_MULT}"
echo "recurrence:     core=${RECURRENT_CORE_LAYERS} steps=${RECURRENT_STEPS}  [FIX: was 3×6]"
echo "use_swiglu:     ${USE_SWIGLU}"
echo "moe:            experts=${MOE_NUM_EXPERTS} every_n=${MOE_EVERY_N} cap=${MOE_CAPACITY_FACTOR} aux=${MOE_AUX_LOSS_COEFF}"
echo "qat:            scheme=${QAT_SCHEME} start_step=${QAT_START_STEP} lsq=${QAT_LSQ}"
echo "use_ssm:        ${USE_SSM} (every_n=${SSM_EVERY_N} expand=${SSM_EXPAND} kernel=${SSM_KERNEL})"
echo "use_mtp:        ${MTP_ENABLED} (steps=${MTP_STEPS} weight=${MTP_WEIGHT} decay=${MTP_DECAY} tie=${MTP_TIE_EMBEDDINGS} lr=${MTP_LR})"
echo "distill:        ${DISTILL_ENABLED} (start_frac=${DISTILL_START_FRAC} weight=${DISTILL_WEIGHT} temp=${DISTILL_TEMP} ema=${DISTILL_EMA_DECAY})"
echo "logit_reg_w:    ${LOGIT_REG_WEIGHT}"
echo "byte_loss:      ${BYTE_WEIGHTED_LOSS_ENABLED} (alpha=${BYTE_WEIGHTED_LOSS_ALPHA})"
echo "eval_stride:    ${EVAL_STRIDE_FRAC}  (sliding window eval)"
echo "eval_seq_len:   ${EVAL_SEQ_LEN}  (0=train_seq_len)"
echo "eval_rope_scale:${EVAL_ROPE_SCALE}"
echo "bigram_rank:    ${BIGRAM_RANK}  (0=disabled)"
echo "residual_ngram: ${RESIDUAL_NGRAM_ENABLED} (bigram_rank=${RESIDUAL_BIGRAM_RANK} trigram_rank=${RESIDUAL_TRIGRAM_RANK} lr=${RESIDUAL_NGRAM_LR} mix_init=${RESIDUAL_NGRAM_MIX_INIT})"
echo "copy_cache:     ${COPY_CACHE_ENABLED} (window=${COPY_CACHE_WINDOW} dim=${COPY_CACHE_DIM} lr=${COPY_CACHE_LR} gate_init=${COPY_CACHE_GATE_INIT})"
echo "muon_momentum:  ${MUON_MOMENTUM}"
echo "grad_clip_norm: ${GRAD_CLIP_NORM}"
echo "quant:          ${QUANT_SCHEME}+${COMPRESSOR}"
echo "log:            ${log_path}"

set +e
torchrun --standalone --nnodes=1 --nproc_per_node="${nproc_per_node}" train_gpt.py
exit_code=$?
set -e

if [[ -f "${log_path}" ]]; then
  echo
  echo "=== Final Score + Budget ==="
  grep -E '^final_.*_roundtrip_exact .*val_bpb:|^submission_budget .*total:.*budget:' "${log_path}" | tail -n 4 || true
  echo
  echo "=== Step Stats ==="
  grep -E '^stopping_early:|^step:[0-9]+/[0-9]+ val_loss:' "${log_path}" | tail -n 5 || true
fi

exit "${exit_code}"
