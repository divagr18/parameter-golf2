#!/usr/bin/env bash
set -euo pipefail

# Build a custom SentencePiece tokenized FineWeb dataset.
#
# Downloads docs_selected.jsonl (if not already cached), trains a BPE tokenizer,
# and exports the full tokenized FineWeb corpus as .bin shards.
#
# Usage:
#   bash ./build_sp_dataset.sh               # builds sp8192 (default)
#   bash ./build_sp_dataset.sh 4096          # builds sp4096
#   VOCAB_SIZE=16384 bash ./build_sp_dataset.sh
#
# Optional env overrides:
#   VOCAB_SIZE=8192          vocabulary size for the new tokenizer
#   VENV_DIR=.venv           virtualenv to use (must be set up by setup_h100_env_and_data.sh first)
#   OUTPUT_ROOT=./data       root dir for tokenizers/ and datasets/ output
#   TOKENIZER_TRAIN_DOCS=    limit docs used to train the SP model (default: all ~15M)
#   EXISTING_TOKENIZER_MODEL= path to an existing .model file to reuse for VOCAB_SIZE
#   HF_TOKEN=                HuggingFace token for faster/authenticated downloads

cd "$(dirname "$0")"

VOCAB_SIZE="${1:-${VOCAB_SIZE:-8192}}"
VENV_DIR="${VENV_DIR:-.venv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./data}"
MAX_TRAIN_SHARDS="${MAX_TRAIN_SHARDS:-80}"
TOKENIZER_TRAIN_DOCS="${TOKENIZER_TRAIN_DOCS:-}"
EXISTING_TOKENIZER_MODEL="${EXISTING_TOKENIZER_MODEL:-}"

# Helpers
log()    { echo "[$(date '+%H:%M:%S')] $*"; }
step()   { echo; echo "=== $* ==="; }
elapsed(){ echo "[$(date '+%H:%M:%S')] done (${SECONDS}s elapsed total)"; }

T_START="${SECONDS}"
log "build_sp_dataset.sh starting"
log "vocab_size=${VOCAB_SIZE}  max_train_shards=${MAX_TRAIN_SHARDS}  venv=${VENV_DIR}  output_root=${OUTPUT_ROOT}"

# -------------------------------------------------------------------
step "1/3  Activate virtualenv"
# -------------------------------------------------------------------
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "ERROR: venv not found at ${VENV_DIR}." >&2
  echo "       Run setup_h100_env_and_data.sh first, then retry." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
log "activated: $(python --version 2>&1)"

# Honour HF_TOKEN for higher rate limits
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
  log "HF_TOKEN set — authenticated requests enabled"
else
  log "HF_TOKEN not set — unauthenticated (may be rate-limited)"
fi

# -------------------------------------------------------------------
step "2/3  Build temporary tokenizer spec"
# -------------------------------------------------------------------
# We write a spec containing the requested vocab size with improved
# training settings:
#   - model_type=unigram  : probabilistic, lower-entropy segmentations
#   - split_by_whitespace=false : allows cross-word merges (" of the" etc.)
#   - num_sub_iterations=4 : more EM passes → better vocab selection
#   - max_sentencepiece_length=32 : capture longer common tokens
#   - nfkc normalization : less aggressive than nmt_nfkc, preserves more
#   - input_sentence_size=10M + shuffle : better frequency estimates
#   - byte_fallback=false + coverage=0.99999 : reclaim 256 slots for merges
#
# Any *other* existing SP models are passed via --reuse-sp-model so they
# are not needlessly retrained. By default, the target VOCAB_SIZE is
# retrained; set EXISTING_TOKENIZER_MODEL to explicitly reuse a specific
# target model when exporting shards.

SPEC_FILE="$(mktemp /tmp/sp_spec_XXXXXX.json)"
cat > "${SPEC_FILE}" <<JSON
{
  "tokenizers": [
    {
      "name": "sp_unigram_${VOCAB_SIZE}",
      "dataset_suffix": "sp${VOCAB_SIZE}",
      "vocab_size": ${VOCAB_SIZE},
      "trainer_overrides": {
        "model_type": "unigram",
        "character_coverage": 0.99999,
        "byte_fallback": false,
        "split_digits": true,
        "split_by_whitespace": false,
        "normalization_rule_name": "nfkc",
        "add_dummy_prefix": false,
        "num_sub_iterations": 4,
        "max_sentencepiece_length": 32,
        "shuffle_input_sentence": true,
        "input_sentence_size": 1000000
      }
    }
  ]
}
JSON
log "tokenizer spec written to ${SPEC_FILE}"

# Auto-detect any existing SentencePiece models to reuse.
# Skip the target VOCAB_SIZE — always retrain it so stale BPE models
# are never silently reused with the wrong model_type/settings.
REUSE_ARGS=()

# Optional explicit reuse for the target VOCAB_SIZE.
# This allows rebuilding/exporting shards with a specific existing tokenizer file
# without retraining the target vocab model.
if [[ -n "${EXISTING_TOKENIZER_MODEL}" ]]; then
  if [[ ! -f "${EXISTING_TOKENIZER_MODEL}" ]]; then
    echo "ERROR: EXISTING_TOKENIZER_MODEL not found: ${EXISTING_TOKENIZER_MODEL}" >&2
    exit 1
  fi
  log "explicit tokenizer reuse enabled for sp${VOCAB_SIZE}: ${EXISTING_TOKENIZER_MODEL}"
  REUSE_ARGS+=(--reuse-sp-model "${VOCAB_SIZE}=${EXISTING_TOKENIZER_MODEL}")
fi

for MODEL_FILE in "${OUTPUT_ROOT}"/tokenizers/fineweb_*_bpe.model; do
  [[ -f "${MODEL_FILE}" ]] || continue
  # Extract vocab size from filename: fineweb_1024_bpe.model -> 1024
  EXISTING_VS="$(python3 - "${MODEL_FILE}" <<'PY'
import sys, re
m = re.search(r'fineweb_(\d+)_bpe\.model$', sys.argv[1])
print(m.group(1) if m else "")
PY
)"
  if [[ -n "${EXISTING_VS}" && "${EXISTING_VS}" != "${VOCAB_SIZE}" ]]; then
    log "reusing existing sp${EXISTING_VS} tokenizer: ${MODEL_FILE}"
    REUSE_ARGS+=(--reuse-sp-model "${EXISTING_VS}=${MODEL_FILE}")
  elif [[ "${EXISTING_VS}" == "${VOCAB_SIZE}" ]]; then
    if [[ -n "${EXISTING_TOKENIZER_MODEL}" ]]; then
      log "found existing sp${EXISTING_VS} model and will reuse explicit model path"
    else
      log "found existing sp${EXISTING_VS} model but NOT reusing — will retrain with improved settings"
    fi
  fi
done

# -------------------------------------------------------------------
step "3/3  Download docs + train tokenizer + export shards"
# -------------------------------------------------------------------
# Estimated time on 8xH100 host (~96 CPUs):
#   - docs_selected.jsonl download  : 5-10 min  (one-time; HF cache is reused after)
#   - SP tokenizer training          : 3-8  min
#   - Shard export (195 x 100MB)     : 15-25 min
# Total: ~25-45 min

EXTRA_ARGS=()
if [[ -n "${TOKENIZER_TRAIN_DOCS}" ]]; then
  EXTRA_ARGS+=(--tokenizer-train-docs "${TOKENIZER_TRAIN_DOCS}")
  log "limiting tokenizer training to ${TOKENIZER_TRAIN_DOCS} docs"
fi

log "running download_hf_docs_and_tokenize.py ..."
python3 data/download_hf_docs_and_tokenize.py \
  --output-root   "${OUTPUT_ROOT}" \
  --tokenizer-config "${SPEC_FILE}" \
  --skip-byte \
  --max-train-shards "${MAX_TRAIN_SHARDS}" \
  "${REUSE_ARGS[@]+"${REUSE_ARGS[@]}"}" \
  "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

rm -f "${SPEC_FILE}"
elapsed

# -------------------------------------------------------------------
echo
T_TOTAL=$(( SECONDS - T_START ))
log "=================================================="
log "Build complete in ${T_TOTAL}s"
log "Dataset : ${OUTPUT_ROOT}/datasets/fineweb10B_sp${VOCAB_SIZE}/"
log "Tokenizer: ${OUTPUT_ROOT}/tokenizers/fineweb_${VOCAB_SIZE}_bpe.model"
log ""
log "To train with this dataset (phase4_fixed_submission.sh):"
log "  env \\"
log "    DATA_PATH=${OUTPUT_ROOT}/datasets/fineweb10B_sp${VOCAB_SIZE} \\"
log "    TOKENIZER_PATH=${OUTPUT_ROOT}/tokenizers/fineweb_${VOCAB_SIZE}_bpe.model \\"
log "    VOCAB_SIZE=${VOCAB_SIZE} \\"
log "    bash ./phase4_fixed_submission.sh"
log "=================================================="
