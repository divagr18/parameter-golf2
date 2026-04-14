#!/usr/bin/env bash
set -euo pipefail

# One-shot setup for Parameter Golf on CUDA machines (RunPod / Lambda etc.).
# Assumes the pod already has Python + torch pre-installed system-wide.
# Installs remaining dependencies and downloads FineWeb cache.
#
# Usage:
#   bash ./setup_h100_env_and_data.sh
#
# Optional overrides:
#   PYTHON_BIN=python3.11
#   FINEWEB_VARIANT=sp1024
#   TRAIN_SHARDS=80
#   INSTALL_DATA=1

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
FINEWEB_VARIANT="${FINEWEB_VARIANT:-sp1024}"
TRAIN_SHARDS="${TRAIN_SHARDS:-80}"
INSTALL_DATA="${INSTALL_DATA:-1}"

# Helpers
log()     { echo "[$(date '+%H:%M:%S')] $*"; }
step()    { echo; echo "=== $* ==="; }
elapsed() { echo "[$(date '+%H:%M:%S')] done (${SECONDS}s elapsed total)"; }

T_START="${SECONDS}"
log "setup_h100_env_and_data.sh starting"
log "python_bin=${PYTHON_BIN}"
log "fineweb_variant=${FINEWEB_VARIANT}  train_shards=${TRAIN_SHARDS}  install_data=${INSTALL_DATA}"

# -------------------------------------------------------------------
step "1/3  Python & UV check"
# -------------------------------------------------------------------
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi
log "found $(${PYTHON_BIN} --version 2>&1) at $(command -v ${PYTHON_BIN})"

# Ensure uv is available — install it into the system Python if not
if command -v uv >/dev/null 2>&1; then
  log "found $(uv --version 2>&1) at $(command -v uv)"
else
  log "uv not found — installing via pip ..."
  "${PYTHON_BIN}" -m pip install -q uv
  log "uv installed: $(uv --version 2>&1)"
fi
elapsed

# -------------------------------------------------------------------
step "2/3  Python packages  (requirements.txt + zstandard)"
# -------------------------------------------------------------------
log "installing requirements.txt into system Python with uv ..."
uv pip install --system --link-mode=copy -r requirements.txt
elapsed

# zstandard: try pre-built binary first (no C compilation = no hangs).
# Fall back to source build only if no wheel is available.
log "installing zstandard (binary wheel preferred) ..."
_ZST_ERR=$(mktemp)
if uv pip install --system --link-mode=copy "zstandard>=0.22" --no-build 2>"${_ZST_ERR}"; then
  log "zstandard installed from pre-built wheel"
elif (cat "${_ZST_ERR}" >&2; uv pip install --system --link-mode=copy "zstandard>=0.22" 2>"${_ZST_ERR}"); then
  log "zstandard installed (compiled from source)"
else
  cat "${_ZST_ERR}" >&2
  log "WARNING: uv failed for zstandard, falling back to pip ..."
  "${PYTHON_BIN}" -m pip install "zstandard>=0.22" --only-binary=:all: || \
    "${PYTHON_BIN}" -m pip install "zstandard>=0.22"
fi
rm -f "${_ZST_ERR}"
elapsed

log "verifying installed packages ..."
"${PYTHON_BIN}" - <<'PY'
import importlib.util, sys

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
step "3/3  FineWeb dataset download"
# -------------------------------------------------------------------
if [[ "${INSTALL_DATA}" == "1" ]]; then
  DATA_DIR="./data/datasets/fineweb10B_${FINEWEB_VARIANT}"
  TOK_DIR="./data/tokenizers"
  log "target dataset dir : ${DATA_DIR}"
  log "target tokenizer dir: ${TOK_DIR}"
  log "downloading variant=${FINEWEB_VARIANT}  train_shards=${TRAIN_SHARDS}"
  log "(each shard is ~100 MB; ${TRAIN_SHARDS} shards ≈ $((TRAIN_SHARDS * 100)) MB — may take several minutes)"
  "${PYTHON_BIN}" data/cached_challenge_fineweb.py \
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
