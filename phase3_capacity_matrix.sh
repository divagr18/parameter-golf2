#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

set_env_var() {
  export "$1=$2"
}

set_default_env() {
  local name="$1"
  local value="$2"
  if [[ -z "${!name:-}" ]]; then
    export "${name}=${value}"
  fi
}

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

bytes_to_mib() {
  local b="$1"
  if [[ -z "$b" ]]; then
    printf ''
  else
    awk -v n="$b" 'BEGIN { printf "%.3f", n / 1048576.0 }'
  fi
}

parse_run_log() {
  local log_path="$1"
  if [[ ! -f "$log_path" ]]; then
    printf '\t\t\t\t\t\t\t\t\t\tmissing_log\n'
    return
  fi

  local params_line export_line artifact_line total_line budget_line roundtrip_line final_val_line
  params_line="$(grep -E '^model_params:[0-9]+' "$log_path" | tail -n 1 || true)"
  export_line="$(grep -E '^export_config ' "$log_path" | tail -n 1 || true)"
  artifact_line="$(grep -E '^Serialized model .+\+[a-z0-9]+: [0-9]+ bytes' "$log_path" | tail -n 1 || true)"
  total_line="$(grep -E '^Total submission size .+\+[a-z0-9]+: [0-9]+ bytes' "$log_path" | tail -n 1 || true)"
  budget_line="$(grep -E '^submission_budget .+ total:[0-9]+ budget:[0-9]+ (headroom_bytes|over_bytes):[0-9]+' "$log_path" | tail -n 1 || true)"
  roundtrip_line="$(grep -E '^final_.*_roundtrip_exact .*val_bpb:' "$log_path" | tail -n 1 || true)"
  final_val_line="$(grep -E '^step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:[0-9.]+' "$log_path" | tail -n 1 || true)"

  local model_params quant comp bpb metric_source artifact total budget headroom under_budget parse_status
  model_params=""
  quant=""
  comp=""
  bpb=""
  metric_source=""
  artifact=""
  total=""
  budget=""
  headroom=""
  under_budget=""
  parse_status="ok"

  if [[ -n "$params_line" ]]; then
    model_params="$(sed -E 's/^model_params:([0-9]+).*/\1/' <<<"$params_line")"
  fi

  if [[ -n "$export_line" ]]; then
    quant="$(sed -E 's/.* quant_scheme:([^ ]+).*/\1/' <<<"$export_line")"
    comp="$(sed -E 's/.* compressor:([^ ]+).*/\1/' <<<"$export_line")"
  fi

  if [[ -n "$roundtrip_line" ]]; then
    bpb="$(sed -E 's/.* val_bpb:([0-9.]+).*/\1/' <<<"$roundtrip_line")"
    metric_source="roundtrip_exact"
  elif [[ -n "$final_val_line" ]]; then
    bpb="$(sed -E 's/.* val_bpb:([0-9.]+).*/\1/' <<<"$final_val_line")"
    metric_source="final_val"
    parse_status="ok_no_roundtrip"
  else
    parse_status="missing_bpb"
  fi

  if [[ -n "$artifact_line" ]]; then
    artifact="$(sed -E 's/.*: ([0-9]+) bytes.*/\1/' <<<"$artifact_line")"
  fi

  if [[ -n "$total_line" ]]; then
    total="$(sed -E 's/.*: ([0-9]+) bytes.*/\1/' <<<"$total_line")"
  fi

  if [[ -n "$budget_line" ]]; then
    budget="$(sed -E 's/.* budget:([0-9]+) .*/\1/' <<<"$budget_line")"
    if [[ "$budget_line" =~ headroom_bytes:([0-9]+) ]]; then
      headroom="${BASH_REMATCH[1]}"
      under_budget="True"
    elif [[ "$budget_line" =~ over_bytes:([0-9]+) ]]; then
      headroom="-${BASH_REMATCH[1]}"
      under_budget="False"
    fi
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$model_params" "$quant" "$comp" "$bpb" "$metric_source" "$artifact" "$total" "$budget" "$headroom" "$under_budget" "$parse_status" "$log_path"
}

mode="${PHASE3_CAPACITY_MODE:-full}"
tests=()
profile_iterations=""
profile_warmup=""
profile_val_max_tokens=""
profile_final_roundtrip_eval=""
profile_seed_spec=""

case "$mode" in
  quick)
    profile_iterations="300"
    profile_warmup="10"
    profile_val_max_tokens="262144"
    profile_final_roundtrip_eval="0"
    profile_seed_spec="1337"
    tests=(
      "recur_3x6_d704_share|704|3|6|1"
      "recur_3x6_d768_share|768|3|6|1"
      "recur_3x6_d832_share|832|3|6|1"
      "recur_3x6_d896_share|896|3|6|1"
      "recur_3x6_d960_share|960|3|6|1"
      "recur_3x6_d1024_share|1024|3|6|1"
      "recur_3x6_d1152_share|1152|3|6|1"
    )
    ;;
  finalist)
    profile_iterations="600"
    profile_warmup="20"
    profile_val_max_tokens="1048576"
    profile_final_roundtrip_eval="1"
    profile_seed_spec="1337,2027"
    tests=(
      "recur_3x6_d832_share|832|3|6|1"
      "recur_3x6_d896_share|896|3|6|1"
      "recur_3x6_d960_share|960|3|6|1"
      "recur_3x6_d1024_share|1024|3|6|1"
      "recur_3x6_d1152_share|1152|3|6|1"
    )
    ;;
  full|*)
    profile_iterations="500"
    profile_warmup="20"
    profile_val_max_tokens="1048576"
    profile_final_roundtrip_eval="1"
    profile_seed_spec="1337"
    tests=(
      "recur_3x6_d640_share|640|3|6|1"
      "recur_3x6_d704_share|704|3|6|1"
      "recur_3x6_d768_share|768|3|6|1"
      "recur_3x6_d832_share|832|3|6|1"
      "recur_3x6_d896_share|896|3|6|1"
      "recur_3x8_d768_share|768|3|8|1"
      "recur_4x6_d768_share|768|4|6|1"
    )
    ;;
esac

set_default_env DEVICE "cuda"
set_default_env USE_TORCH_COMPILE "0"
set_default_env ITERATIONS "$profile_iterations"
set_default_env WARMUP_STEPS "$profile_warmup"
set_default_env TRAIN_BATCH_TOKENS "8192"
set_default_env VAL_LOSS_EVERY "0"
set_default_env VAL_BATCH_SIZE "131072"
set_default_env VAL_MAX_TOKENS "$profile_val_max_tokens"
set_default_env FINAL_ROUNDTRIP_EVAL "$profile_final_roundtrip_eval"
set_default_env SUBMISSION_SIZE_BUDGET_BYTES "16777216"
set_default_env QUANT_SCHEME "int8"
set_default_env COMPRESSOR "auto"
set_default_env WEIGHT_ORDER "none"
set_default_env MIXED_LOW_PRECISION_SCHEME "int8"
set_default_env NUM_LAYERS "9"

if [[ "${COMPRESSOR}" == "auto" || "${COMPRESSOR}" == "zstd" ]]; then
  if ! "${PYTHON_BIN:-python}" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('zstandard') else 1)" >/dev/null 2>&1; then
    echo "Warning: zstandard package not found; COMPRESSOR=${COMPRESSOR} will use zlib fallback and may miss 16MB budget." >&2
    echo "Install with: ${PYTHON_BIN:-python} -m pip install zstandard" >&2
  fi
fi

if [[ -n "${PHASE3_CAPACITY_TESTS:-}" ]]; then
  declare -A allowed=()
  IFS=',' read -r -a requested <<<"$PHASE3_CAPACITY_TESTS"
  for name in "${requested[@]}"; do
    key="$(trim "$name")"
    [[ -n "$key" ]] && allowed["$key"]=1
  done

  filtered=()
  for t in "${tests[@]}"; do
    IFS='|' read -r name _ <<<"$t"
    if [[ -n "${allowed[$name]:-}" ]]; then
      filtered+=("$t")
    fi
  done
  tests=("${filtered[@]}")
  if [[ ${#tests[@]} -eq 0 ]]; then
    echo "PHASE3_CAPACITY_TESTS filter removed all tests. Check names." >&2
    exit 1
  fi
fi

seed_spec="${PHASE3_CAPACITY_SEEDS:-$profile_seed_spec}"
IFS=',' read -r -a seeds <<<"$seed_spec"
if [[ ${#seeds[@]} -eq 0 ]]; then
  echo "No seeds provided. Set PHASE3_CAPACITY_SEEDS, e.g. 1337,2027" >&2
  exit 1
fi

echo
echo "=== Phase 3 Capacity Config ==="
echo "mode=${mode} iterations=${ITERATIONS} warmup_steps=${WARMUP_STEPS} val_max_tokens=${VAL_MAX_TOKENS} final_roundtrip_eval=${FINAL_ROUNDTRIP_EVAL}"
echo "seeds=${seed_spec} tests=${#tests[@]}"

timestamp="$(date +%Y%m%d_%H%M%S)"
csv_path="logs/phase3_capacity_matrix_${timestamp}.csv"
agg_path="logs/phase3_capacity_matrix_${timestamp}_aggregate.csv"
printf 'Test,Seed,ModelDim,NumLayers,CoreLayers,RecurrentSteps,ShareFFN,ModelParams,ExitCode,DurationSec,ValBpb,MetricSource,TotalSubmissionMiB,TotalSubmissionBytes,BudgetMiB,HeadroomMiB,UnderBudget,QuantScheme,CompressorResolved,ParseStatus,LogPath\n' >"$csv_path"

for test in "${tests[@]}"; do
  IFS='|' read -r test_name model_dim core_layers recurrent_steps share_ffn <<<"$test"
  for seed_raw in "${seeds[@]}"; do
    seed="$(trim "$seed_raw")"
    [[ -z "$seed" ]] && continue
    run_id="phase3cap_${timestamp}_${test_name}_s${seed}"
    echo
    echo "=== Running ${test_name} seed=${seed} (RUN_ID=${run_id}) ==="

    set_env_var RUN_ID "$run_id"
    set_env_var SEED "$seed"
    set_env_var MODEL_DIM "$model_dim"
    set_env_var RECURRENT_CORE_LAYERS "$core_layers"
    set_env_var RECURRENT_STEPS "$recurrent_steps"
    set_env_var SHARE_FFN_ACROSS_BLOCKS "$share_ffn"

    start_epoch="$(date +%s)"
    "${PYTHON_BIN:-python}" train_gpt.py
    exit_code=$?
    end_epoch="$(date +%s)"
    duration_sec="$((end_epoch - start_epoch))"

    log_path="logs/${run_id}.txt"
    IFS=$'\t' read -r model_params quant_scheme compressor_resolved val_bpb metric_source artifact_bytes total_bytes budget_bytes headroom_bytes under_budget parse_status parsed_log_path <<<"$(parse_run_log "$log_path")"

    total_mib="$(bytes_to_mib "$total_bytes")"
    budget_mib="$(bytes_to_mib "$budget_bytes")"
    headroom_mib="$(bytes_to_mib "$headroom_bytes")"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$test_name" "$seed" "$model_dim" "$NUM_LAYERS" "$core_layers" "$recurrent_steps" "$share_ffn" "$model_params" \
      "$exit_code" "$duration_sec" "$val_bpb" "$metric_source" "$total_mib" "$total_bytes" "$budget_mib" "$headroom_mib" \
      "$under_budget" "$quant_scheme" "$compressor_resolved" "$parse_status" "$parsed_log_path" \
      >>"$csv_path"

    if [[ "$exit_code" -ne 0 ]]; then
      echo "Warning: run ${test_name} seed=${seed} failed with exit code ${exit_code}" >&2
    fi
  done
done

echo
echo "=== Phase 3 Capacity Per-Run Summary ==="
if command -v column >/dev/null 2>&1; then
  column -s, -t <"$csv_path"
else
  cat "$csv_path"
fi

awk -F, '
NR==1 {next}
{
  test=$1
  runs[test]++
  if ($20 ~ /^ok/ && $11!="") {
    ok[test]++
    sum_bpb[test]+=$11
    sum_mib[test]+=$13
  }
}
END {
  print "Test,Runs,OkRuns,AvgValBpb,AvgSubmissionMiB"
  for (t in runs) {
    avg_bpb = (ok[t] > 0) ? sprintf("%.8f", sum_bpb[t] / ok[t]) : ""
    avg_mib = (ok[t] > 0) ? sprintf("%.3f", sum_mib[t] / ok[t]) : ""
    print t "," runs[t] "," (ok[t] + 0) "," avg_bpb "," avg_mib
  }
}
' "$csv_path" >"$agg_path"

echo
echo "=== Phase 3 Capacity Aggregate Summary ==="
if command -v column >/dev/null 2>&1; then
  column -s, -t <"$agg_path"
else
  cat "$agg_path"
fi

echo "Saved per-run CSV: $csv_path"
echo "Saved aggregate CSV: $agg_path"
