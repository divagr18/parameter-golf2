#!/usr/bin/env bash
set -euo pipefail

# Simplified setup for Parameter Golf on H100: Install torch 2.10 + dependencies only.
# Skips dataset download (datasets/tokenizers must be pre-loaded or built separately).
#
# Usage:
#   bash ./setup_h100_env_torch_only.sh
#
# Optional overrides:
#   PYTHON_BIN=python3.11
#   VENV_DIR=.venv
#   TORCH_VERSION=2.10.0
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
# Derive matching torchvision/torchaudio versions from TORCH_VERSION.
# Pattern: torch 2.X.Y -> torchvision 0.(X+15).Y, torchaudio 2.X.Y
_TORCH_MINOR=$(echo "${TORCH_VERSION}" | cut -d. -f2)
_TORCH_PATCH=$(echo "${TORCH_VERSION}" | cut -d. -f3)
_VISION_MINOR=$(( _TORCH_MINOR + 15 ))
VISION_VERSION="${VISION_VERSION:-0.${_VISION_MINOR}.${_TORCH_PATCH}}"
AUDIO_VERSION="${AUDIO_VERSION:-${TORCH_VERSION}}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

# Place the uv cache on persistent storage to avoid multi-minute stalls on large wheels
UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.uv_cache}"
export UV_CACHE_DIR
mkdir -p "${UV_CACHE_DIR}"

# Helpers
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
step() { echo; echo "=== $* ==="; }
elapsed() { echo "[$(date '+%H:%M:%S')] done (${SECONDS}s elapsed total)"; }

T_START="${SECONDS}"
log "setup_h100_env_torch_only.sh starting"
log "python_bin=${PYTHON_BIN}  venv=${VENV_DIR}"
log "torch_version=${TORCH_VERSION}  vision_version=${VISION_VERSION}  audio_version=${AUDIO_VERSION}"
log "index_url=${TORCH_INDEX_URL}"

# -------------------------------------------------------------------
step "1/3  Python & UV check"
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
step "2/3  Virtualenv setup & torch install"
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

# Ensure pip exists inside the venv
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

# Install torch + torchvision + torchaudio
log "installing CUDA torch==${TORCH_VERSION} + torchvision==${VISION_VERSION} + torchaudio==${AUDIO_VERSION} ..."
log "(using pip for reliability; uv hangs on large wheels in some environments)"
"${VENV_PYTHON}" -m pip install --no-cache-dir --progress-bar on -U \
  "torch==${TORCH_VERSION}" \
  "torchvision==${VISION_VERSION}" \
  "torchaudio==${AUDIO_VERSION}" \
  --index-url "${TORCH_INDEX_URL}"
elapsed

# Install additional core dependencies
log "installing core dependencies (zstandard, sentencepiece, etc) ..."
UV_LINK_MODE_FLAG=()
if [[ -n "${UV_LINK_MODE:-}" ]]; then
  UV_LINK_MODE_FLAG=(--link-mode="${UV_LINK_MODE}")
fi

# zstandard: try pre-built binary first (no C compilation)
if uv pip install --python "${VENV_PYTHON}" "${UV_LINK_MODE_FLAG[@]}" "zstandard>=0.22" --no-build 2>/dev/null; then
  log "zstandard installed from pre-built wheel"
elif uv pip install --python "${VENV_PYTHON}" "${UV_LINK_MODE_FLAG[@]}" "zstandard>=0.22" 2>/dev/null; then
  log "zstandard installed (compiled from source)"
else
  log "WARNING: uv failed for zstandard, falling back to pip ..."
  "${VENV_PYTHON}" -m pip install "zstandard>=0.22" --only-binary=:all: || "${VENV_PYTHON}" -m pip install "zstandard>=0.22"
fi

# Install remaining requirements from requirements.txt (if it exists and has packages)
if [[ -f "requirements.txt" ]]; then
  log "installing requirements.txt with uv ..."
  if ! uv pip install --python "${VENV_PYTHON}" "${UV_LINK_MODE_FLAG[@]}" -U -r requirements.txt 2>/dev/null; then
    log "uv failed; falling back to pip for requirements.txt ..."
    "${VENV_PYTHON}" -m pip install -U -r requirements.txt
  fi
fi
elapsed

# -------------------------------------------------------------------
step "3/3  Verification"
# -------------------------------------------------------------------
log "verifying installed packages ..."
python - <<'PY'
import importlib, importlib.util, sys
importlib.invalidate_caches()

# Core packages required for training
mods = ["torch", "zstandard"]
# Optional but recommended
opt_mods = ["sentencepiece", "datasets", "huggingface_hub", "tqdm", "tiktoken"]

missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"ERROR: missing core packages: {missing}")

missing_opt = [m for m in opt_mods if importlib.util.find_spec(m) is None]
if missing_opt:
    print(f"  WARNING: optional packages not installed: {missing_opt}")

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
print("  core packages verified ✓")
PY
elapsed

# -------------------------------------------------------------------
echo
T_TOTAL=$(( SECONDS - T_START ))
log "=================================================="
log "Setup complete in ${T_TOTAL}s"
log "Torch 2.10 environment ready"
log "Note: datasets and tokenizers must be in ./data/"
log "Next: bash ./phase4_fixed_submission.sh"
log "=================================================="
