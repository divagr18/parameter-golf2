#!/usr/bin/env bash
set -euo pipefail

# Performance-first 600s sweep: runs a few faster architecture candidates
# and prints a ranked table by final roundtrip-exact val_bpb.
#
# Usage:
#   bash ./phase3_perf_sweep_600s.sh
#
# Optional overrides:
#   NPROC_PER_NODE=8
#   LOCAL_TRAIN_TOKENS_PER_RANK=16384
#   LOCAL_VAL_TOKENS_PER_RANK=32768
#   MAX_WALLCLOCK_SECONDS=600
#   SEED=1337

cd "$(dirname "$0")"
mkdir -p logs

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

if ! command -v torchrun >/dev/null 2>&1; then
  echo "torchrun not found. Activate/install env first." >&2
  exit 1
fi

gpu_count="$("${PYTHON_BIN}" -c "import torch; print(torch.cuda.device_count())")"
if [[ -z "${gpu_count}" || "${gpu_count}" -lt 1 ]]; then
  echo "No CUDA GPUs detected." >&2
  exit 1
fi

if [[ -n "${NPROC_PER_NODE:-}" ]]; then
  nproc_per_node="${NPROC_PER_NODE}"
else
  nproc_per_node="${gpu_count}"
fi
if [[ "${nproc_per_node}" -gt "${gpu_count}" ]]; then
  echo "NPROC_PER_NODE=${nproc_per_node} exceeds visible_gpus=${gpu_count}" >&2
  exit 1
fi

local_train_tokens_per_rank="${LOCAL_TRAIN_TOKENS_PER_RANK:-16384}"
local_val_tokens_per_rank="${LOCAL_VAL_TOKENS_PER_RANK:-32768}"
if (( local_train_tokens_per_rank % 1024 != 0 )); then
  echo "LOCAL_TRAIN_TOKENS_PER_RANK must be a multiple of 1024." >&2
  exit 1
fi
if (( local_val_tokens_per_rank % 1024 != 0 )); then
  echo "LOCAL_VAL_TOKENS_PER_RANK must be a multiple of 1024." >&2
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
csv_path="logs/phase3_perf_sweep_${timestamp}.csv"
printf 'Candidate,ExitCode,DurationSec,ValBpbRoundtripExact,StepAtStop,StepAvgMs,SubmissionBytes,UnderBudget,LogPath\n' > "${csv_path}"

# Common run settings (performance-first).
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export DEVICE="${DEVICE:-cuda}"
export USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-1}"
export SDP_BACKEND_MODE="${SDP_BACKEND_MODE:-flash}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export ITERATIONS="${ITERATIONS:-20000}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export VAL_MAX_TOKENS="${VAL_MAX_TOKENS:-0}"
export VAL_BATCH_SIZE="$(( nproc_per_node * local_val_tokens_per_rank ))"
export TRAIN_BATCH_TOKENS="$(( nproc_per_node * local_train_tokens_per_rank ))"
export FINAL_ROUNDTRIP_EVAL="${FINAL_ROUNDTRIP_EVAL:-1}"
export SUBMISSION_SIZE_BUDGET_BYTES="${SUBMISSION_SIZE_BUDGET_BYTES:-16000000}"
export QUANT_SCHEME="${QUANT_SCHEME:-int8}"
export COMPRESSOR="${COMPRESSOR:-zstd}"
export WEIGHT_ORDER="${WEIGHT_ORDER:-none}"
export MIXED_LOW_PRECISION_SCHEME="${MIXED_LOW_PRECISION_SCHEME:-int8}"
export SEED="${SEED:-1337}"

# Keep optimizer lane fixed to your current best Muon profile.
export MUON_MOMENTUM="${MUON_MOMENTUM:-0.98}"
export MUON_BACKEND_STEPS="${MUON_BACKEND_STEPS:-5}"
export MUON_MOMENTUM_WARMUP_START="${MUON_MOMENTUM_WARMUP_START:-0.85}"
export MUON_MOMENTUM_WARMUP_STEPS="${MUON_MOMENTUM_WARMUP_STEPS:-500}"
export NUM_LAYERS="${NUM_LAYERS:-9}"
export NUM_HEADS="${NUM_HEADS:-8}"
export SHARE_FFN_ACROSS_BLOCKS="${SHARE_FFN_ACROSS_BLOCKS:-1}"
export TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-1}"

echo "Perf sweep config: nproc=${nproc_per_node} train_batch_tokens=${TRAIN_BATCH_TOKENS} val_batch_size=${VAL_BATCH_SIZE} max_wallclock=${MAX_WALLCLOCK_SECONDS}s"
echo "Using compile=${USE_TORCH_COMPILE} sdp_mode=${SDP_BACKEND_MODE} grad_accum=${GRAD_ACCUM_STEPS}"

# name|model_dim|core_layers|recurrent_steps|num_kv_heads
candidates=(
  "fast_d896_r3x4_kv2|896|3|4|2"
  "balanced_d960_r3x5_kv4|960|3|5|4"
  "deepnarrow_d896_r4x5_kv2|896|4|5|2"
)

for c in "${candidates[@]}"; do
  IFS='|' read -r name d core rep kv <<<"${c}"
  export MODEL_DIM="${d}"
  export RECURRENT_CORE_LAYERS="${core}"
  export RECURRENT_STEPS="${rep}"
  export NUM_KV_HEADS="${kv}"
  export RUN_ID="perf600_${timestamp}_${name}_s${SEED}"
  log_path="logs/${RUN_ID}.txt"

  echo
  echo "=== Running ${name} (d=${d}, core=${core}, rep=${rep}, kv=${kv}) ==="
  start_epoch="$(date +%s)"
  set +e
  torchrun --standalone --nnodes=1 --nproc_per_node="${nproc_per_node}" train_gpt.py
  exit_code=$?
  set -e
  end_epoch="$(date +%s)"
  duration_sec="$((end_epoch - start_epoch))"

  val_bpb="$(grep -E '^final_.*_roundtrip_exact .*val_bpb:' "${log_path}" | tail -n 1 | sed -E 's/.*val_bpb:([0-9.]+).*/\1/' || true)"
  if [[ -z "${val_bpb}" ]]; then
    val_bpb="$(grep -E '^step:[0-9]+/[0-9]+ val_loss:.* val_bpb:' "${log_path}" | tail -n 1 | sed -E 's/.*val_bpb:([0-9.]+).*/\1/' || true)"
  fi
  stop_line="$(grep -E '^stopping_early: wallclock_cap' "${log_path}" | tail -n 1 || true)"
  if [[ -n "${stop_line}" ]]; then
    step_at_stop="$(sed -E 's/.* step:([0-9]+)\/[0-9]+.*/\1/' <<<"${stop_line}")"
    step_avg_ms="$(sed -E 's/.*step_avg:([0-9.]+)ms.*/\1/' <<<"${stop_line}")"
  else
    last_val_line="$(grep -E '^step:[0-9]+/[0-9]+ val_loss:.* val_bpb:' "${log_path}" | tail -n 1 || true)"
    step_at_stop="$(sed -E 's/step:([0-9]+)\/[0-9]+.*/\1/' <<<"${last_val_line}")"
    step_avg_ms="$(sed -E 's/.*step_avg:([0-9.]+)ms.*/\1/' <<<"${last_val_line}")"
  fi
  budget_line="$(grep -E '^submission_budget .* total:[0-9]+ budget:[0-9]+ (headroom_bytes|over_bytes):[0-9]+' "${log_path}" | tail -n 1 || true)"
  submission_bytes="$(sed -E 's/.* total:([0-9]+) budget:.*/\1/' <<<"${budget_line}")"
  if [[ "${budget_line}" == *"headroom_bytes"* ]]; then
    under_budget="True"
  elif [[ "${budget_line}" == *"over_bytes"* ]]; then
    under_budget="False"
  else
    under_budget=""
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${name}" "${exit_code}" "${duration_sec}" "${val_bpb}" "${step_at_stop}" "${step_avg_ms}" "${submission_bytes}" "${under_budget}" "${log_path}" \
    >> "${csv_path}"
done

echo
echo "=== Performance Sweep Results (sorted by ValBpbRoundtripExact) ==="
if command -v column >/dev/null 2>&1; then
  {
    head -n 1 "${csv_path}"
    tail -n +2 "${csv_path}" | sort -t, -k4,4g
  } | column -s, -t
else
  head -n 1 "${csv_path}"
  tail -n +2 "${csv_path}" | sort -t, -k4,4g
fi

echo "Saved CSV: ${csv_path}"
