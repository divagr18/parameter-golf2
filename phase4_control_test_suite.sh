#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

RUN_TAG="${RUN_TAG:-phase4_control_test_$(date +%Y%m%d_%H%M%S)}"
SUMMARY_CSV="logs/${RUN_TAG}_summary.csv"
mkdir -p logs

printf 'RunId,Label,ExitCode,DurationSec,OfficialValBpb,FinalEvalMode,SubmissionBytes,BudgetStatus,LogPath\n' > "${SUMMARY_CSV}"

RUN_IDS=()
RUN_LABELS=()

base_env=(
  "TARGET_GPUS=${TARGET_GPUS:-1}"
  "TARGET_GLOBAL_TOKENS=${TARGET_GLOBAL_TOKENS:-65536}"
  "MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-600}"
  "DISTILL_ENABLED=${DISTILL_ENABLED:-1}"
  "DISTILL_WEIGHT=${DISTILL_WEIGHT:-0.08}"
  "DISTILL_TEMP=${DISTILL_TEMP:-2.0}"
  "DISTILL_START_FRAC=${DISTILL_START_FRAC:-0.70}"
  "DISTILL_START_STEP=${DISTILL_START_STEP:--1}"
  "DISTILL_START_WALLCLOCK_FRAC=${DISTILL_START_WALLCLOCK_FRAC:-0.70}"
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
  "EVAL_CONT_CACHE_WEIGHT=${EVAL_CONT_CACHE_WEIGHT:-0.10}"
  "EVAL_CONT_CACHE_LOGIT_SCALE=${EVAL_CONT_CACHE_LOGIT_SCALE:-12.0}"
  "EVAL_CONT_CACHE_CONF_POWER=${EVAL_CONT_CACHE_CONF_POWER:-1.0}"
  "EVAL_CONT_CACHE_BATCH_SEQS=${EVAL_CONT_CACHE_BATCH_SEQS:-4}"
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
    submission_bytes="$(sed -E 's/.* total:([0-9]+) .*/\1/' <<<"${budget_line}")"
    if [[ "${budget_line}" =~ headroom_bytes: ]]; then
      budget_status="ok"
    elif [[ "${budget_line}" =~ over_bytes: ]]; then
      budget_status="over"
    fi
  fi

  printf '%s\t%s\t%s\t%s\n' "${official_bpb}" "${final_mode}" "${submission_bytes}" "${budget_status}"
}

run_phase4() {
  local suffix="$1"
  local label="$2"
  shift 2

  local run_id="${RUN_TAG}_${suffix}"
  local log_path="logs/${run_id}.txt"
  local start_ts end_ts duration exit_code parsed official_bpb final_mode submission_bytes budget_status

  RUN_IDS+=("${run_id}")
  RUN_LABELS+=("${label}")

  echo
  echo "=== Running ${label} (RUN_ID=${run_id}) ==="
  start_ts="$(date +%s)"
  set +e
  env "${base_env[@]}" RUN_ID="${run_id}" "$@" bash ./phase4_fixed_submission.sh
  exit_code=$?
  set -e
  end_ts="$(date +%s)"
  duration=$((end_ts - start_ts))

  parsed="$(parse_run_log "${log_path}")"
  official_bpb="$(cut -f1 <<<"${parsed}")"
  final_mode="$(cut -f2 <<<"${parsed}")"
  submission_bytes="$(cut -f3 <<<"${parsed}")"
  budget_status="$(cut -f4 <<<"${parsed}")"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${run_id}" "${label}" "${exit_code}" "${duration}" "${official_bpb}" "${final_mode}" \
    "${submission_bytes}" "${budget_status}" "${log_path}" >> "${SUMMARY_CSV}"

  echo "status=${exit_code} duration=${duration}s official_val_bpb=${official_bpb:-NA} mode=${final_mode:-NA} budget=${budget_status}"
}

lookup_bpb() {
  local run_id="$1"
  awk -F',' -v rid="${run_id}" 'NR > 1 && $1 == rid { print $5; exit }' "${SUMMARY_CSV}"
}

print_pair_delta() {
  local title="$1"
  local control_id="$2"
  local test_id="$3"
  local control_bpb test_bpb delta

  control_bpb="$(lookup_bpb "${control_id}")"
  test_bpb="$(lookup_bpb "${test_id}")"
  if [[ -z "${control_bpb}" || -z "${test_bpb}" ]]; then
    echo "${title}: missing results"
    return
  fi

  delta="$(python -c "print(float('${test_bpb}') - float('${control_bpb}'))")"
  echo "${title}: control=${control_bpb} test=${test_bpb} delta=${delta}"
}

run_phase4 "01_control_baseline" "control_baseline" \
  DISTILL_START_WALLCLOCK_FRAC=-1.0 \
  EVAL_SEQ_LEN=0 \
  EVAL_ROPE_SCALE=1.0 \
  EVAL_BLEND_SEQ_LENS= \
  EVAL_BLEND_WEIGHTS= \
  EVAL_BLEND_POSITION_BIAS=0.0 \
  EVAL_CONT_CACHE_ENABLED=0 \
  FINAL_EVAL_MODE=primary

run_phase4 "02_test_distill_wallclock" "test_distill_wallclock" \
  DISTILL_START_WALLCLOCK_FRAC=0.70 \
  EVAL_SEQ_LEN=0 \
  EVAL_ROPE_SCALE=1.0 \
  EVAL_BLEND_SEQ_LENS= \
  EVAL_BLEND_WEIGHTS= \
  EVAL_BLEND_POSITION_BIAS=0.0 \
  EVAL_CONT_CACHE_ENABLED=0 \
  FINAL_EVAL_MODE=primary

run_phase4 "03_control_ctx1536" "control_ctx1536" \
  DISTILL_START_WALLCLOCK_FRAC=0.70 \
  EVAL_SEQ_LEN=1536 \
  EVAL_ROPE_SCALE=2.25 \
  EVAL_BLEND_SEQ_LENS= \
  EVAL_BLEND_WEIGHTS= \
  EVAL_BLEND_POSITION_BIAS=0.0 \
  EVAL_CONT_CACHE_ENABLED=0 \
  FINAL_EVAL_MODE=primary

run_phase4 "04_test_ctx1536_cache" "test_ctx1536_cache" \
  DISTILL_START_WALLCLOCK_FRAC=0.70 \
  EVAL_SEQ_LEN=1536 \
  EVAL_ROPE_SCALE=2.25 \
  EVAL_BLEND_SEQ_LENS= \
  EVAL_BLEND_WEIGHTS= \
  EVAL_BLEND_POSITION_BIAS=0.0 \
  EVAL_CONT_CACHE_ENABLED=1 \
  EVAL_CONT_CACHE_WINDOW=8192 \
  EVAL_CONT_CACHE_TOPK=64 \
  EVAL_CONT_CACHE_WEIGHT=0.10 \
  EVAL_CONT_CACHE_LOGIT_SCALE=12.0 \
  EVAL_CONT_CACHE_CONF_POWER=1.0 \
  EVAL_CONT_CACHE_BATCH_SEQS=4 \
  FINAL_EVAL_MODE=primary

run_phase4 "05_control_blend_flat" "control_blend_flat" \
  DISTILL_START_WALLCLOCK_FRAC=0.70 \
  EVAL_SEQ_LEN=0 \
  EVAL_ROPE_SCALE=1.0 \
  EVAL_BLEND_SEQ_LENS=1024,1536 \
  EVAL_BLEND_WEIGHTS=0.50,0.50 \
  EVAL_BLEND_POSITION_BIAS=0.0 \
  EVAL_CONT_CACHE_ENABLED=0 \
  FINAL_EVAL_MODE=blend

run_phase4 "06_test_blend_ramp" "test_blend_ramp" \
  DISTILL_START_WALLCLOCK_FRAC=0.70 \
  EVAL_SEQ_LEN=0 \
  EVAL_ROPE_SCALE=1.0 \
  EVAL_BLEND_SEQ_LENS=1024,1536 \
  EVAL_BLEND_WEIGHTS=0.50,0.50 \
  EVAL_BLEND_POSITION_BIAS=1.25 \
  EVAL_BLEND_POSITION_POWER=1.0 \
  EVAL_CONT_CACHE_ENABLED=0 \
  FINAL_EVAL_MODE=blend

run_phase4 "07_test_blend_ramp_cache" "test_blend_ramp_cache" \
  DISTILL_START_WALLCLOCK_FRAC=0.70 \
  EVAL_SEQ_LEN=0 \
  EVAL_ROPE_SCALE=1.0 \
  EVAL_BLEND_SEQ_LENS=1024,1536 \
  EVAL_BLEND_WEIGHTS=0.50,0.50 \
  EVAL_BLEND_POSITION_BIAS=1.25 \
  EVAL_BLEND_POSITION_POWER=1.0 \
  EVAL_CONT_CACHE_ENABLED=1 \
  EVAL_CONT_CACHE_WINDOW=8192 \
  EVAL_CONT_CACHE_TOPK=64 \
  EVAL_CONT_CACHE_WEIGHT=0.10 \
  EVAL_CONT_CACHE_LOGIT_SCALE=12.0 \
  EVAL_CONT_CACHE_CONF_POWER=1.0 \
  EVAL_CONT_CACHE_BATCH_SEQS=4 \
  FINAL_EVAL_MODE=blend

echo
echo "=== Ranked Summary (official roundtrip_exact val_bpb) ==="
ranked_summary="$({
  head -n 1 "${SUMMARY_CSV}"
  tail -n +2 "${SUMMARY_CSV}" | sort -t',' -k5,5g
})"
if command -v column >/dev/null 2>&1; then
  printf '%s\n' "${ranked_summary}" | column -s, -t
else
  printf '%s\n' "${ranked_summary}"
fi

echo
echo "=== Paired Deltas (test - control; lower is better) ==="
print_pair_delta "distill_wallclock" "${RUN_TAG}_01_control_baseline" "${RUN_TAG}_02_test_distill_wallclock"
print_pair_delta "ctx1536_cache" "${RUN_TAG}_03_control_ctx1536" "${RUN_TAG}_04_test_ctx1536_cache"
print_pair_delta "blend_ramp" "${RUN_TAG}_05_control_blend_flat" "${RUN_TAG}_06_test_blend_ramp"
print_pair_delta "blend_ramp_plus_cache_vs_flat" "${RUN_TAG}_05_control_blend_flat" "${RUN_TAG}_07_test_blend_ramp_cache"

echo
echo "=== Composite Final Results ==="
for run_id in "${RUN_IDS[@]}"; do
  log_path="logs/${run_id}.txt"
  echo "--- ${run_id} ---"
  grep -E \
    '^(eval_primary|eval_sweep|eval_blend|eval_cont_cache|train_loss_mask|distill_start):|^final_.*(_ctx_exact|_blend_exact|_roundtrip_exact) .*val_bpb:|^submission_budget .*total:.*budget:|^stopping_early:|^step:[0-9]+/[0-9]+ val_loss:' \
    "${log_path}" || true
  echo
done

echo "Summary CSV: ${SUMMARY_CSV}"
