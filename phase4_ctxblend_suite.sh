#!/usr/bin/env bash
set -euo pipefail

# Runs the phase-4 long-context / blend / train-mask experiment suite and prints:
# 1. A ranked CSV summary by official roundtrip-exact val_bpb
# 2. A composite grep for every run showing the final eval lines and budget
#
# Usage:
#   bash ./phase4_ctxblend_suite.sh
#
# Optional overrides:
#   RUN_TAG=mytag
#   SEED=1337
#   TARGET_GPUS=1
#   TARGET_GLOBAL_TOKENS=65536
#   MAX_WALLCLOCK_SECONDS=600

cd "$(dirname "$0")"
mkdir -p logs

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

if ! command -v torchrun >/dev/null 2>&1; then
  echo "torchrun not found. Activate/install the environment first." >&2
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
run_tag="${RUN_TAG:-phase4_ctxblend_${timestamp}}"
summary_csv="logs/${run_tag}_summary.csv"

printf 'RunId,ExitCode,DurationSec,OfficialValBpb,FinalEvalMode,SubmissionBytes,BudgetStatus,LogPath\n' > "${summary_csv}"

RUN_IDS=()

base_env=(
  "TARGET_GPUS=${TARGET_GPUS:-1}"
  "TARGET_GLOBAL_TOKENS=${TARGET_GLOBAL_TOKENS:-65536}"
  "MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-600}"
  "DISTILL_ENABLED=${DISTILL_ENABLED:-1}"
  "DISTILL_WEIGHT=${DISTILL_WEIGHT:-0.08}"
  "DISTILL_TEMP=${DISTILL_TEMP:-2.0}"
  "DISTILL_START_FRAC=${DISTILL_START_FRAC:-0.70}"
  "DISTILL_START_STEP=${DISTILL_START_STEP:--1}"
  "DISTILL_START_WALLCLOCK_FRAC=${DISTILL_START_WALLCLOCK_FRAC:--1.0}"
  "DISTILL_EMA_DECAY=${DISTILL_EMA_DECAY:-0.999}"
  "BYTE_WEIGHTED_LOSS_ENABLED=${BYTE_WEIGHTED_LOSS_ENABLED:-0}"
  "SWA_ENABLED=${SWA_ENABLED:-1}"
  "EVAL_STRIDE_FRAC=${EVAL_STRIDE_FRAC:-0.5}"
  "TRAIN_LOSS_MASK_ENABLED=${TRAIN_LOSS_MASK_ENABLED:-0}"
  "TRAIN_LOSS_MASK_STRIDE_FRAC=${TRAIN_LOSS_MASK_STRIDE_FRAC:-0.0}"
  "EVAL_SEQ_LEN=${EVAL_SEQ_LEN:-0}"
  "EVAL_ROPE_SCALE=${EVAL_ROPE_SCALE:-1.0}"
  "EVAL_SWEEP_SEQ_LENS=${EVAL_SWEEP_SEQ_LENS:-}"
  "EVAL_SWEEP_ROPE_SCALES=${EVAL_SWEEP_ROPE_SCALES:-}"
  "EVAL_BLEND_SEQ_LENS=${EVAL_BLEND_SEQ_LENS:-}"
  "EVAL_BLEND_ROPE_SCALES=${EVAL_BLEND_ROPE_SCALES:-}"
  "EVAL_BLEND_WEIGHTS=${EVAL_BLEND_WEIGHTS:-}"
  "EVAL_BLEND_STRIDE_FRAC=${EVAL_BLEND_STRIDE_FRAC:-0.0}"
  "EVAL_BLEND_POSITION_BIAS=${EVAL_BLEND_POSITION_BIAS:-0.0}"
  "EVAL_BLEND_POSITION_POWER=${EVAL_BLEND_POSITION_POWER:-1.0}"
  "EVAL_CONT_CACHE_ENABLED=${EVAL_CONT_CACHE_ENABLED:-0}"
  "EVAL_CONT_CACHE_WINDOW=${EVAL_CONT_CACHE_WINDOW:-8192}"
  "EVAL_CONT_CACHE_TOPK=${EVAL_CONT_CACHE_TOPK:-64}"
  "EVAL_CONT_CACHE_WEIGHT=${EVAL_CONT_CACHE_WEIGHT:-0.12}"
  "EVAL_CONT_CACHE_LOGIT_SCALE=${EVAL_CONT_CACHE_LOGIT_SCALE:-12.0}"
  "EVAL_CONT_CACHE_CONF_POWER=${EVAL_CONT_CACHE_CONF_POWER:-1.0}"
  "EVAL_CONT_CACHE_BATCH_SEQS=${EVAL_CONT_CACHE_BATCH_SEQS:-8}"
  "FINAL_EVAL_MODE=${FINAL_EVAL_MODE:-primary}"
  "SEED=${SEED:-1337}"
)

parse_run_log() {
  local log_path="$1"
  local official_line budget_line official_bpb final_mode submission_bytes budget_status

  if [[ ! -f "${log_path}" ]]; then
    printf '\t\t\t\tmissing_log\n'
    return
  fi

  official_line="$(grep -E '^final_.*_roundtrip_exact .*val_bpb:' "${log_path}" | tail -n 1 || true)"
  budget_line="$(grep -E '^submission_budget .*total:[0-9]+ budget:[0-9]+ (headroom_bytes|over_bytes):[0-9]+' "${log_path}" | tail -n 1 || true)"

  official_bpb=""
  final_mode=""
  submission_bytes=""
  budget_status="missing"

  if [[ -n "${official_line}" ]]; then
    official_bpb="$(sed -E 's/.* val_bpb:([0-9.]+).*/\1/' <<<"${official_line}")"
    if [[ "${official_line}" =~ mode: ]]; then
      final_mode="$(sed -E 's/.* mode:([^ ]+).*/\1/' <<<"${official_line}")"
    fi
  fi

  if [[ -n "${budget_line}" ]]; then
    submission_bytes="$(sed -E 's/.* total:([0-9]+) budget:.*/\1/' <<<"${budget_line}")"
    if [[ "${budget_line}" == *"headroom_bytes"* ]]; then
      budget_status="under"
    elif [[ "${budget_line}" == *"over_bytes"* ]]; then
      budget_status="over"
    fi
  fi

  printf '%s\t%s\t%s\t%s\n' "${official_bpb}" "${final_mode}" "${submission_bytes}" "${budget_status}"
}

run_phase4() {
  local suffix="$1"
  shift

  local run_id="${run_tag}_${suffix}"
  local log_path="logs/${run_id}.txt"
  local start_epoch end_epoch duration_sec exit_code
  local official_bpb final_mode submission_bytes budget_status

  RUN_IDS+=("${run_id}")

  echo
  echo "=== Running ${suffix} (RUN_ID=${run_id}) ==="
  start_epoch="$(date +%s)"
  set +e
  env "${base_env[@]}" RUN_ID="${run_id}" "$@" bash ./phase4_fixed_submission.sh
  exit_code=$?
  set -e
  end_epoch="$(date +%s)"
  duration_sec="$((end_epoch - start_epoch))"

  IFS=$'\t' read -r official_bpb final_mode submission_bytes budget_status <<<"$(parse_run_log "${log_path}")"

  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${run_id}" \
    "${exit_code}" \
    "${duration_sec}" \
    "${official_bpb}" \
    "${final_mode}" \
    "${submission_bytes}" \
    "${budget_status}" \
    "${log_path}" \
    >> "${summary_csv}"

  if [[ "${exit_code}" -ne 0 ]]; then
    echo "Warning: ${run_id} failed with exit code ${exit_code}" >&2
  fi
}

run_phase4 "01_ctx_sweep_base" \
  EVAL_SWEEP_SEQ_LENS=1536,2048,3072

run_phase4 "02_ctx_primary_2048" \
  EVAL_SEQ_LEN=2048 \
  EVAL_ROPE_SCALE=4.0

run_phase4 "03_blend_probe_50_50" \
  EVAL_BLEND_SEQ_LENS=1024,2048 \
  EVAL_BLEND_WEIGHTS=0.5,0.5

run_phase4 "04_blend_probe_35_65" \
  EVAL_BLEND_SEQ_LENS=1024,2048 \
  EVAL_BLEND_WEIGHTS=0.35,0.65

run_phase4 "05_blend_official_35_65" \
  EVAL_BLEND_SEQ_LENS=1024,2048 \
  EVAL_BLEND_WEIGHTS=0.35,0.65 \
  FINAL_EVAL_MODE=blend

run_phase4 "06_trainmask_ctx_sweep" \
  TRAIN_LOSS_MASK_ENABLED=1 \
  EVAL_SWEEP_SEQ_LENS=1536,2048

run_phase4 "07_trainmask_blend_official" \
  TRAIN_LOSS_MASK_ENABLED=1 \
  EVAL_SWEEP_SEQ_LENS=1536,2048 \
  EVAL_BLEND_SEQ_LENS=1024,2048 \
  EVAL_BLEND_WEIGHTS=0.35,0.65 \
  FINAL_EVAL_MODE=blend

echo
echo "=== Ranked Summary (official roundtrip_exact val_bpb) ==="
if command -v column >/dev/null 2>&1; then
  {
    head -n 1 "${summary_csv}"
    tail -n +2 "${summary_csv}" \
      | awk -F, '{ sort_key = ($4 == "" ? 999 : $4); print sort_key "," $0 }' \
      | sort -t, -k1,1g \
      | cut -d, -f2-
  } | column -s, -t
else
  head -n 1 "${summary_csv}"
  tail -n +2 "${summary_csv}" \
    | awk -F, '{ sort_key = ($4 == "" ? 999 : $4); print sort_key "," $0 }' \
    | sort -t, -k1,1g \
    | cut -d, -f2-
fi

echo
echo "=== Composite Final Results ==="
for run_id in "${RUN_IDS[@]}"; do
  log_path="logs/${run_id}.txt"
  echo "=== ${run_id} ==="
  grep -E \
    '^(eval_primary|eval_sweep|eval_blend|train_loss_mask):|^final_.*(_ctx_exact|_blend_exact|_roundtrip_exact) .*val_bpb:|^submission_budget .*total:.*budget:|^stopping_early:|^step:[0-9]+/[0-9]+ val_loss:' \
    "${log_path}" || true
  echo
done

echo "Saved summary CSV: ${summary_csv}"
