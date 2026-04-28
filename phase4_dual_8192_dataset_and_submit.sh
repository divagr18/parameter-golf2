#!/bin/bash
# Comprehensive script to:
# 1. Build fineweb10B_sp8192 (BPE) dataset
# 2. Build fineweb10B_sp8192_unigram dataset
# 3. Run best-known submission with both tokenizers

set -e

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${WORKSPACE_ROOT}/data"
TOKENIZERS_DIR="${DATA_ROOT}/tokenizers"

# Best-known locked parameters from BEST_KNOWN_PARAMS.md
MODEL_DIM=448
NUM_LAYERS=9
RESIDUAL_NGRAM_ENABLED=0
BIGRAM_RANK=0
DISTILL_ENABLED=0
SWA_ENABLED=1
QK_GAIN_INIT=5.0
GPTQ=1
QUANT_SCHEME=int8
COMPRESSOR=zstd

BPE_OUTPUT_ROOT="${DATA_ROOT}/dual_bpe"
UNIGRAM_OUTPUT_ROOT="${DATA_ROOT}/dual_unigram"

echo "=========================================="
echo "Phase 4: Dual 8192 Dataset Build & Submit"
echo "=========================================="
echo "Locked Parameters:"
echo "  MODEL_DIM=${MODEL_DIM}"
echo "  DISTILL_ENABLED=${DISTILL_ENABLED}"
echo "  BIGRAM_RANK=${BIGRAM_RANK}"
echo "  RESIDUAL_NGRAM_ENABLED=${RESIDUAL_NGRAM_ENABLED}"
echo "  SWA_ENABLED=${SWA_ENABLED}"
echo "  GPTQ=${GPTQ}"
echo "  QUANT_SCHEME=${QUANT_SCHEME}"
echo "  COMPRESSOR=${COMPRESSOR}"
echo ""

# ============================================================================
# PHASE 1: BUILD DATASETS
# ============================================================================

echo "=========================================="
echo "PHASE 1: Building Datasets"
echo "=========================================="

# Build 8192 BPE dataset
echo ""
echo ">>> Building fineweb10B_sp8192 (BPE)..."
if [ ! -d "${BPE_OUTPUT_ROOT}/datasets/fineweb10B_sp8192" ]; then
  export VOCAB_SIZE=8192
  export EXISTING_TOKENIZER_MODEL="${TOKENIZERS_DIR}/fineweb_8192_bpe.model"
  export OUTPUT_ROOT="${BPE_OUTPUT_ROOT}"
  bash "${WORKSPACE_ROOT}/build_sp_dataset.sh"
  echo "✓ fineweb10B_sp8192 (BPE) dataset built successfully"
else
  echo "✓ fineweb10B_sp8192 (BPE) dataset already exists, skipping build"
fi

# Build 8192 Unigram dataset
echo ""
echo ">>> Building fineweb10B_sp8192_unigram..."
if [ ! -d "${UNIGRAM_OUTPUT_ROOT}/datasets/fineweb10B_sp8192" ]; then
  export VOCAB_SIZE=8192
  export EXISTING_TOKENIZER_MODEL="${TOKENIZERS_DIR}/fineweb_8192_unigram_20260422_225958.model"
  export OUTPUT_ROOT="${UNIGRAM_OUTPUT_ROOT}"
  bash "${WORKSPACE_ROOT}/build_sp_dataset.sh"
  echo "✓ fineweb10B_sp8192 (Unigram) dataset built successfully"
else
  echo "✓ fineweb10B_sp8192 (Unigram) dataset already exists, skipping build"
fi

echo ""
echo "=========================================="
echo "Datasets ready. Proceeding to submissions..."
echo "=========================================="
echo ""

# ============================================================================
# PHASE 2: RUN BEST-KNOWN SUBMISSIONS WITH BOTH TOKENIZERS
# ============================================================================

echo "=========================================="
echo "PHASE 2: Running Best-Known Submissions"
echo "=========================================="

# Set common environment for both runs
export MODEL_DIM
export NUM_LAYERS
export RESIDUAL_NGRAM_ENABLED
export BIGRAM_RANK
export DISTILL_ENABLED
export SWA_ENABLED
export QK_GAIN_INIT
export GPTQ
export QUANT_SCHEME
export COMPRESSOR

# ---- Run 1: BPE 8192 ----
echo ""
echo ">>> RUN 1: Best-Known with 8192 BPE"
echo "=========================================="

export VOCAB_SIZE=8192
export DATA_PATH="${BPE_OUTPUT_ROOT}/datasets/fineweb10B_sp8192"
export TOKENIZER_PATH="${TOKENIZERS_DIR}/fineweb_8192_bpe.model"
export RUN_ID="best_sp8192_bpe_$(date +%Y%m%d_%H%M%S)"

echo "RUN_ID: ${RUN_ID}"
echo "VOCAB_SIZE: ${VOCAB_SIZE}"
echo "TOKENIZER: BPE"
echo "DATA_PATH: ${DATA_PATH}"
echo ""

bash "${WORKSPACE_ROOT}/phase4_fixed_submission.sh"

echo ""
echo "✓ Run 1 (BPE) completed"

# ---- Run 2: Unigram 8192 ----
echo ""
echo ">>> RUN 2: Best-Known with 8192 Unigram"
echo "=========================================="

export VOCAB_SIZE=8192
export DATA_PATH="${UNIGRAM_OUTPUT_ROOT}/datasets/fineweb10B_sp8192"
export TOKENIZER_PATH="${TOKENIZERS_DIR}/fineweb_8192_unigram_20260422_225958.model"
export RUN_ID="best_sp8192_unigram_$(date +%Y%m%d_%H%M%S)"

echo "RUN_ID: ${RUN_ID}"
echo "VOCAB_SIZE: ${VOCAB_SIZE}"
echo "TOKENIZER: Unigram"
echo "DATA_PATH: ${DATA_PATH}"
echo ""

bash "${WORKSPACE_ROOT}/phase4_fixed_submission.sh"

echo ""
echo "✓ Run 2 (Unigram) completed"

echo ""
echo "=========================================="
echo "✓ All submissions completed successfully"
echo "=========================================="
