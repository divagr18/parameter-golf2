#!/usr/bin/env bash
set -euo pipefail

# One-shot setup for Parameter Golf on CUDA machines.
# Installs Python env, dependencies (including CUDA torch + zstd), and downloads FineWeb cache.
#
# Usage:
#   bash ./setup_h100_env_and_data.sh
#
# Optional overrides:
#   PYTHON_BIN=python3.11
#   VENV_DIR=.venv
#   FORCE_CUDA_TORCH=1
#   TORCH_VERSION=2.10.0
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
#   FINEWEB_VARIANT=sp1024
#   TRAIN_SHARDS=80
#   INSTALL_DATA=1

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
FORCE_CUDA_TORCH="${FORCE_CUDA_TORCH:-1}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
FINEWEB_VARIANT="${FINEWEB_VARIANT:-sp1024}"
TRAIN_SHARDS="${TRAIN_SHARDS:-80}"
INSTALL_DATA="${INSTALL_DATA:-1}"

# Helpers
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
step() { echo; echo "=== $* ==="; }
elapsed() { echo "[$(date '+%H:%M:%S')] done (${SECONDS}s elapsed total)"; }

T_START="${SECONDS}"
log "setup_h100_env_and_data.sh starting"
log "python_bin=${PYTHON_BIN}  venv=${VENV_DIR}  force_cuda_torch=${FORCE_CUDA_TORCH}"
log "torch_version=${TORCH_VERSION}  index_url=${TORCH_INDEX_URL}"
log "fineweb_variant=${FINEWEB_VARIANT}  train_shards=${TRAIN_SHARDS}  install_data=${INSTALL_DATA}"

# -------------------------------------------------------------------
step "1/5  Python check"
# -------------------------------------------------------------------
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi
log "found $(${PYTHON_BIN} --version 2>&1) at $(command -v ${PYTHON_BIN})"

# -------------------------------------------------------------------
step "2/5  Virtualenv"
# -------------------------------------------------------------------
if [[ ! -d "${VENV_DIR}" ]]; then
  log "creating virtualenv at ${VENV_DIR} ..."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  log "virtualenv created"
else
  log "virtualenv already exists at ${VENV_DIR}, reusing"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
log "activated: $(python --version 2>&1)"

# -------------------------------------------------------------------
step "3/5  pip / base tools"
# -------------------------------------------------------------------
log "upgrading pip, setuptools, wheel ..."
python -m pip install --upgrade pip setuptools wheel 2>&1 | \
  grep -E --line-buffered '(Collecting|Downloading|Installing|Successfully|already)' | \
  while IFS= read -r line; do log "  pip: ${line}"; done
elapsed

# -------------------------------------------------------------------
step "4/5  Python packages  (requirements.txt + zstandard)"
# -------------------------------------------------------------------
log "installing requirements.txt + zstandard ..."
python -m pip install --upgrade -r requirements.txt zstandard 2>&1 | \
  grep -E --line-buffered '(Collecting|Downloading|Installing|Successfully|already|WARNING|ERROR)' | \
  while IFS= read -r line; do log "  pip: ${line}"; done
elapsed

if [[ "${FORCE_CUDA_TORCH}" == "1" ]]; then
  log "installing CUDA torch==${TORCH_VERSION} from ${TORCH_INDEX_URL} ..."
  log "(this downloads ~2-3 GB — may take several minutes on first run)"
  python -m pip install --upgrade "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}" 2>&1 | \
    grep -E --line-buffered '(Collecting|Downloading|Installing|Successfully|already|WARNING|ERROR|MB|%|kB)' | \
    while IFS= read -r line; do log "  pip: ${line}"; done
  elapsed
fi

log "verifying installed packages ..."
python - <<'PY'
import importlib.util, sys, time

mods = ["torch", "zstandard", "sentencepiece", "datasets", "huggingface_hub", "tqdm", "tiktoken"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"ERROR: missing packages after install: {missing}")

import torch
print(f"  torch:             {torch.__version__}")
print(f"  cuda_available:    {torch.cuda.is_available()}")
print(f"  cuda_device_count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  gpu[{i}]: {p.name}  {p.total_memory // 1024**3} GB")
import zstandard
print(f"  zstandard:         {zstandard.__version__}")
import sentencepiece as sp
print(f"  sentencepiece:     {sp.__version__}")
print("  all required packages present")
PY
elapsed

# -------------------------------------------------------------------
step "5/5  FineWeb dataset download"
# -------------------------------------------------------------------
if [[ "${INSTALL_DATA}" == "1" ]]; then
  DATA_DIR="./data/datasets/fineweb10B_${FINEWEB_VARIANT}"
  TOK_DIR="./data/tokenizers"
  log "target dataset dir : ${DATA_DIR}"
  log "target tokenizer dir: ${TOK_DIR}"
  log "downloading variant=${FINEWEB_VARIANT}  train_shards=${TRAIN_SHARDS}"
  log "(each shard is ~100 MB; ${TRAIN_SHARDS} shards ≈ $((TRAIN_SHARDS * 100)) MB — may take several minutes)"
  python data/cached_challenge_fineweb.py \
    --variant "${FINEWEB_VARIANT}" \
    --train-shards "${TRAIN_SHARDS}"
  log "dataset files:"
  ls -lh "${DATA_DIR}"/ 2>/dev/null | head -20 || true
  log "tokenizer files:"
  ls -lh "${TOK_DIR}"/ 2>/dev/null || true
  elapsed
else
  log "INSTALL_DATA=0 — skipping dataset download"
fi

# -------------------------------------------------------------------
echo
T_TOTAL=$(( SECONDS - T_START ))
log "=================================================="
log "Setup complete in ${T_TOTAL}s"
log "Next: bash ./phase4_fixed_submission.sh"
log "=================================================="
