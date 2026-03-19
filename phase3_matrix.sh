#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

set_env_var() {
  export "$1=$2"
}

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

parse_run_log() {
  local log_path="$1"
  if [[ ! -f "$log_path" ]]; then
    printf '\t\t\t\tmissing_log\n'
    return
  fi

  local export_line artifact_line total_line roundtrip_line
  export_line="$(grep -E '^export_config ' "$log_path" | tail -n 1 || true)"
  artifact_line="$(grep -E '^Serialized model .+\+[a-z0-9]+: [0-9]+ bytes' "$log_path" | tail -n 1 || true)"
  total_line="$(grep -E '^Total submission size .+\+[a-z0-9]+: [0-9]+ bytes' "$log_path" | tail -n 1 || true)"
  roundtrip_line="$(grep -E '^final_.*_roundtrip_exact .*val_bpb:' "$log_path" | tail -n 1 || true)"

  local bpb artifact total comp parse_status
  bpb=""
  artifact=""
  total=""
  comp=""
  parse_status="ok"

  if [[ -n "$export_line" ]]; then
    comp="$(sed -E 's/.* compressor:([^ ]+).*/\1/' <<<"$export_line")"
  fi

  if [[ -n "$roundtrip_line" ]]; then
    bpb="$(sed -E 's/.* val_bpb:([0-9.]+).*/\1/' <<<"$roundtrip_line")"
  else
    parse_status="missing_roundtrip"
  fi

  if [[ -n "$artifact_line" ]]; then
    artifact="$(sed -E 's/.*: ([0-9]+) bytes.*/\1/' <<<"$artifact_line")"
  fi

  if [[ -n "$total_line" ]]; then
    total="$(sed -E 's/.*: ([0-9]+) bytes.*/\1/' <<<"$total_line")"
  fi

  printf '%s\t%s\t%s\t%s\t%s\n' "$bpb" "$artifact" "$total" "$comp" "$parse_status"
}

timestamp="$(date +%Y%m%d_%H%M%S)"

tests=(
  "stacked_512_l9|512|9|0|0|0"
  "recur_2x6_d704_share|704|9|2|6|1"
  "recur_2x6_d704_noshare|704|9|2|6|0"
  "recur_3x6_d640_share|640|9|3|6|1"
)

seed_spec="${PHASE3_SEEDS:-1337,2027}"
IFS=',' read -r -a seeds <<<"$seed_spec"
if [[ ${#seeds[@]} -eq 0 ]]; then
  echo "No seeds provided. Set PHASE3_SEEDS, e.g. 1337,2027" >&2
  exit 1
fi

csv_path="logs/phase3_matrix_${timestamp}.csv"
agg_path="logs/phase3_matrix_${timestamp}_aggregate.csv"
printf 'Test,Seed,ModelDim,NumLayers,CoreLayers,RecurrentSteps,ShareFFN,ExitCode,DurationSec,ValBpbRoundtrip,ArtifactBytes,TotalSubmissionBytes,CompressorResolved,ParseStatus,LogPath\n' >"$csv_path"

for test in "${tests[@]}"; do
  IFS='|' read -r test_name model_dim num_layers core_layers recurrent_steps share_ffn <<<"$test"
  for seed_raw in "${seeds[@]}"; do
    seed="$(trim "$seed_raw")"
    [[ -z "$seed" ]] && continue
    run_id="phase3_${timestamp}_${test_name}_s${seed}"
    echo
    echo "=== Running ${test_name} seed=${seed} (RUN_ID=${run_id}) ==="

    set_env_var DEVICE "cuda"
    set_env_var USE_TORCH_COMPILE "0"
    set_env_var ITERATIONS "400"
    set_env_var WARMUP_STEPS "20"
    set_env_var TRAIN_BATCH_TOKENS "8192"
    set_env_var VAL_LOSS_EVERY "0"
    set_env_var VAL_BATCH_SIZE "131072"
    set_env_var VAL_MAX_TOKENS "1048576"
    set_env_var FINAL_ROUNDTRIP_EVAL "1"
    set_env_var QUANT_SCHEME "int8"
    set_env_var COMPRESSOR "auto"
    set_env_var WEIGHT_ORDER "none"
    set_env_var MIXED_LOW_PRECISION_SCHEME "int8"

    set_env_var RUN_ID "$run_id"
    set_env_var SEED "$seed"
    set_env_var MODEL_DIM "$model_dim"
    set_env_var NUM_LAYERS "$num_layers"
    set_env_var RECURRENT_CORE_LAYERS "$core_layers"
    set_env_var RECURRENT_STEPS "$recurrent_steps"
    set_env_var SHARE_FFN_ACROSS_BLOCKS "$share_ffn"

    start_epoch="$(date +%s)"
    "${PYTHON_BIN:-python}" train_gpt.py
    exit_code=$?
    end_epoch="$(date +%s)"
    duration_sec="$((end_epoch - start_epoch))"

    log_path="logs/${run_id}.txt"
    IFS=$'\t' read -r val_bpb artifact_bytes total_bytes compressor_resolved parse_status <<<"$(parse_run_log "$log_path")"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$test_name" "$seed" "$model_dim" "$num_layers" "$core_layers" "$recurrent_steps" "$share_ffn" \
      "$exit_code" "$duration_sec" "$val_bpb" "$artifact_bytes" "$total_bytes" "$compressor_resolved" "$parse_status" "$log_path" \
      >>"$csv_path"

    if [[ "$exit_code" -ne 0 ]]; then
      echo "Warning: run ${test_name} seed=${seed} failed with exit code ${exit_code}" >&2
    fi
  done
done

echo
echo "=== Phase 3 Per-Run Summary ==="
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
  if ($14=="ok" && $10!="") {
    ok[test]++
    sum_bpb[test]+=$10
    sum_total[test]+=$12
  }
}
END {
  print "Test,Runs,OkRuns,AvgValBpb,AvgTotalSubmissionBytes"
  for (t in runs) {
    avg_bpb = (ok[t] > 0) ? sprintf("%.8f", sum_bpb[t] / ok[t]) : ""
    avg_total = (ok[t] > 0) ? sprintf("%.0f", sum_total[t] / ok[t]) : ""
    print t "," runs[t] "," (ok[t] + 0) "," avg_bpb "," avg_total
  }
}
' "$csv_path" >"$agg_path"

echo
echo "=== Phase 3 Aggregate Summary ==="
if command -v column >/dev/null 2>&1; then
  column -s, -t <"$agg_path"
else
  cat "$agg_path"
fi

echo "Saved per-run CSV: $csv_path"
echo "Saved aggregate CSV: $agg_path"
