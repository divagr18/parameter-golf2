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
#   TOKENIZER_VOCAB_SIZE=8192   shorthand for FINEWEB_VARIANT=sp8192
#   FINEWEB_VARIANT=sp1024
#   TRAIN_SHARDS=80
#   INSTALL_DATA=1

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
FORCE_CUDA_TORCH="${FORCE_CUDA_TORCH:-1}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
# Derive matching torchvision/torchaudio versions from TORCH_VERSION.
# Pattern: torch 2.X.Y -> torchvision 0.(X+15).Y, torchaudio 2.X.Y
_TORCH_MINOR=$(echo "${TORCH_VERSION}" | cut -d. -f2)
_TORCH_PATCH=$(echo "${TORCH_VERSION}" | cut -d. -f3)
_VISION_MINOR=$(( _TORCH_MINOR + 15 ))
VISION_VERSION="${VISION_VERSION:-0.${_VISION_MINOR}.${_TORCH_PATCH}}"
AUDIO_VERSION="${AUDIO_VERSION:-${TORCH_VERSION}}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TOKENIZER_VOCAB_SIZE="${TOKENIZER_VOCAB_SIZE:-}"
FINEWEB_VARIANT="${FINEWEB_VARIANT:-sp1024}"
TRAIN_SHARDS="${TRAIN_SHARDS:-80}"
INSTALL_DATA="${INSTALL_DATA:-1}"

# Allow shorthand: TOKENIZER_VOCAB_SIZE=8192 or FINEWEB_VARIANT=8192
if [[ -n "${TOKENIZER_VOCAB_SIZE}" ]]; then
  FINEWEB_VARIANT="sp${TOKENIZER_VOCAB_SIZE}"
elif [[ "${FINEWEB_VARIANT}" =~ ^[0-9]+$ ]]; then
  FINEWEB_VARIANT="sp${FINEWEB_VARIANT}"
fi

# Place the uv cache on the persistent /workspace volume (same filesystem as the venv)
# so uv can hardlink wheels instead of byte-copying them. Avoids multi-minute stalls on
# the torch wheel (~7 GB of files). Override UV_CACHE_DIR to pin elsewhere.
UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.uv_cache}"
export UV_CACHE_DIR
mkdir -p "${UV_CACHE_DIR}"

# Helpers
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
step() { echo; echo "=== $* ==="; }
elapsed() { echo "[$(date '+%H:%M:%S')] done (${SECONDS}s elapsed total)"; }

T_START="${SECONDS}"
log "setup_h100_env_and_data.sh starting"
log "python_bin=${PYTHON_BIN}  venv=${VENV_DIR}  force_cuda_torch=${FORCE_CUDA_TORCH}"
log "torch_version=${TORCH_VERSION}  vision_version=${VISION_VERSION}  audio_version=${AUDIO_VERSION}  index_url=${TORCH_INDEX_URL}"
log "fineweb_variant=${FINEWEB_VARIANT}  train_shards=${TRAIN_SHARDS}  install_data=${INSTALL_DATA}"

# -------------------------------------------------------------------
step "1/4  Python & UV check"
# -------------------------------------------------------------------
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi
log "found $(${PYTHON_BIN} --version 2>&1) at $(command -v ${PYTHON_BIN})"

if command -v uv >/dev/null 2>&1; then
  log "found $(uv --version 2>&1) at $(command -v uv)"
else
  log "uv not found globally. Will install locally in virtualenv."
fi

# -------------------------------------------------------------------
step "2/4  Virtualenv & uv install"
# -------------------------------------------------------------------
if command -v uv >/dev/null 2>&1; then
  if [[ ! -d "${VENV_DIR}" ]]; then
    log "creating virtualenv at ${VENV_DIR} with uv ..."
    uv venv "${VENV_DIR}" --python "${PYTHON_BIN}"
    log "virtualenv created"
  else
    log "virtualenv already exists at ${VENV_DIR}, reusing"
  fi
else
  if [[ ! -d "${VENV_DIR}" ]]; then
    log "creating virtualenv at ${VENV_DIR} with standard venv ..."
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
    log "virtualenv created"
  else
    log "virtualenv already exists at ${VENV_DIR}, reusing"
  fi
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
log "activated: $(python --version 2>&1)"

VENV_PYTHON="${VENV_DIR}/bin/python"
if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "ERROR: virtualenv python not found at ${VENV_PYTHON}" >&2
  exit 1
fi

# Ensure pip exists inside the venv so installs never leak to global Python.
if ! "${VENV_PYTHON}" -m pip --version >/dev/null 2>&1; then
  log "pip missing in virtualenv; bootstrapping with ensurepip ..."
  "${VENV_PYTHON}" -m ensurepip --upgrade
fi
"${VENV_PYTHON}" -m pip install -q --upgrade pip setuptools wheel
log "venv pip: $("${VENV_PYTHON}" -m pip --version 2>&1)"

if ! command -v uv >/dev/null 2>&1; then
  log "installing uv in virtualenv ..."
  "${VENV_PYTHON}" -m pip install -q uv
fi
log "using $(uv --version 2>&1)"
elapsed

# -------------------------------------------------------------------
step "3/4  Python packages  (requirements.txt + zstandard)"
# -------------------------------------------------------------------
# --link-mode defaults to hardlink when cache and venv share a filesystem (see UV_CACHE_DIR
# setup at the top). If you need to override (e.g., a split-fs setup), pass UV_LINK_MODE=copy.
UV_LINK_MODE_FLAG=()
if [[ -n "${UV_LINK_MODE:-}" ]]; then
  UV_LINK_MODE_FLAG=(--link-mode="${UV_LINK_MODE}")
fi

log "installing requirements.txt with uv (cache=${UV_CACHE_DIR}) ..."
if ! uv pip install --python "${VENV_PYTHON}" "${UV_LINK_MODE_FLAG[@]}" -U -r requirements.txt; then
  log "WARNING: uv failed to inspect the venv interpreter (likely a 'python' pkg conflict)."
  log "Falling back to plain pip for requirements.txt ..."
  "${VENV_PYTHON}" -m pip install -U -r requirements.txt
fi
elapsed

# zstandard: try pre-built binary first (no C compilation = no hangs).
# Fall back to source build only if no wheel is available.
log "installing zstandard (binary wheel preferred) ..."
if uv pip install --python "${VENV_PYTHON}" "${UV_LINK_MODE_FLAG[@]}" "zstandard>=0.22" --no-build 2>/dev/null; then
  log "zstandard installed from pre-built wheel"
elif uv pip install --python "${VENV_PYTHON}" "${UV_LINK_MODE_FLAG[@]}" "zstandard>=0.22" 2>/dev/null; then
  log "zstandard installed (compiled from source)"
else
  log "WARNING: uv failed for zstandard, falling back to pip ..."
  "${VENV_PYTHON}" -m pip install "zstandard>=0.22" --only-binary=:all: || "${VENV_PYTHON}" -m pip install "zstandard>=0.22"
fi
elapsed

if [[ "${FORCE_CUDA_TORCH}" == "1" ]]; then
  log "installing CUDA torch==${TORCH_VERSION} + torchvision==${VISION_VERSION} + torchaudio==${AUDIO_VERSION} ..."
  log "(uv hangs on the torch wheel install in some RunPod setups; pip handles it reliably)"
  "${VENV_PYTHON}" -m pip install --no-cache-dir --progress-bar on -U \
    "torch==${TORCH_VERSION}" \
    "torchvision==${VISION_VERSION}" \
    "torchaudio==${AUDIO_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"
  elapsed
fi

log "verifying installed packages ..."
python - <<'PY'
import importlib, importlib.util, sys
# Invalidate stale caches so packages installed moments ago by pip are visible.
importlib.invalidate_caches()

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
step "4/4  FineWeb dataset download"
# -------------------------------------------------------------------
if [[ "${INSTALL_DATA}" == "1" ]]; then
  DATA_DIR="./data/datasets/fineweb10B_${FINEWEB_VARIANT}"
  TOK_DIR="./data/tokenizers"
  log "target dataset dir : ${DATA_DIR}"
  log "target tokenizer dir: ${TOK_DIR}"

  # sp1024/byte260: use cached challenge artifacts from HF.
  # Other SP variants (e.g. sp8192): build/export locally via build_sp_dataset.sh.
  if [[ "${FINEWEB_VARIANT}" == "sp1024" || "${FINEWEB_VARIANT}" == "byte260" ]]; then
    log "downloading cached variant=${FINEWEB_VARIANT}  train_shards=${TRAIN_SHARDS}"
    log "(each shard is ~100 MB; ${TRAIN_SHARDS} shards ≈ $((TRAIN_SHARDS * 100)) MB — may take several minutes)"
    python data/cached_challenge_fineweb.py \
      --variant "${FINEWEB_VARIANT}" \
      --train-shards "${TRAIN_SHARDS}"
  elif [[ "${FINEWEB_VARIANT}" =~ ^sp([0-9]+)$ ]]; then
    vocab_size="${BASH_REMATCH[1]}"
    log "building/exporting local variant=${FINEWEB_VARIANT} (VOCAB_SIZE=${vocab_size})"
    log "this path trains/reuses SentencePiece then exports shards"
    VOCAB_SIZE="${vocab_size}" \
    MAX_TRAIN_SHARDS="${TRAIN_SHARDS}" \
    VENV_DIR="${VENV_DIR}" \
    OUTPUT_ROOT="./data" \
    bash ./build_sp_dataset.sh
  else
    echo "ERROR: unsupported FINEWEB_VARIANT='${FINEWEB_VARIANT}'. Expected byte260 or sp<VOCAB_SIZE>." >&2
    exit 1
  fi

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
