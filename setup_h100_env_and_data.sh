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

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtualenv at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# Install repo deps first (includes torch pin), then force CUDA torch if requested.
python -m pip install --upgrade -r requirements.txt zstandard
if [[ "${FORCE_CUDA_TORCH}" == "1" ]]; then
  python -m pip install --upgrade "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}"
fi

python - <<'PY'
import importlib.util
import sys

mods = ["torch", "zstandard", "sentencepiece", "datasets", "huggingface_hub", "tqdm", "tiktoken"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"Missing Python packages after install: {missing}")

import torch
print(f"torch: {torch.__version__}")
print(f"cuda_available: {torch.cuda.is_available()}")
print(f"cuda_device_count: {torch.cuda.device_count()}")
PY

if [[ "${INSTALL_DATA}" == "1" ]]; then
  echo "Downloading FineWeb cache variant=${FINEWEB_VARIANT} train_shards=${TRAIN_SHARDS}"
  python data/cached_challenge_fineweb.py --variant "${FINEWEB_VARIANT}" --train-shards "${TRAIN_SHARDS}"
fi

echo "Setup complete."
