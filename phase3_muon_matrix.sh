#!/usr/bin/env bash
set -euo pipefail

# Muon hyperparameter sweep for the current best architecture lane.
# Runs multiple Muon configs, parses logs, and prints ranked summaries.
#
# Usage:
#   bash ./phase3_muon_matrix.sh
#
# Optional overrides:
#   MUON_TUNE_SEEDS=1337,2027
#   MAX_WALLCLOCK_SECONDS=300
#   VAL_MAX_TOKENS=262144
#   NPROC_PER_NODE=4

cd "$(dirname "$0")"
mkdir -p logs

if [[ ! -f "./phase3_2xh100_full_regime.sh" ]]; then
  echo "Missing phase3_2xh100_full_regime.sh in repo root." >&2
  exit 1
fi

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

parse_run_log() {
  local log_path="$1"
  if [[ ! -f "$log_path" ]]; then
    printf '\t\t\t\t\tmissing_log\n'
    return
  fi

  local roundtrip_line final_val_line total_line budget_line
  roundtrip_line="$(grep -E '^final_.*_roundtrip_exact .*val_bpb:' "$log_path" | tail -n 1 || true)"
  final_val_line="$(grep -E '^step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:[0-9.]+' "$log_path" | tail -n 1 || true)"
  total_line="$(grep -E '^Total submission size .+\+[a-z0-9]+: [0-9]+ bytes' "$log_path" | tail -n 1 || true)"
  budget_line="$(grep -E '^submission_budget .+ total:[0-9]+ budget:[0-9]+ (headroom_bytes|over_bytes):[0-9]+' "$log_path" | tail -n 1 || true)"

  local bpb metric_source total_bytes headroom under_budget parse_status
  bpb=""
  metric_source=""
  total_bytes=""
  headroom=""
  under_budget=""
  parse_status="ok"

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

  if [[ -n "$total_line" ]]; then
    total_bytes="$(sed -E 's/.*: ([0-9]+) bytes.*/\1/' <<<"$total_line")"
  fi

  if [[ -n "$budget_line" ]]; then
    if [[ "$budget_line" =~ headroom_bytes:([0-9]+) ]]; then
      headroom="${BASH_REMATCH[1]}"
      under_budget="True"
    elif [[ "$budget_line" =~ over_bytes:([0-9]+) ]]; then
      headroom="-${BASH_REMATCH[1]}"
      under_budget="False"
    fi
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$bpb" "$metric_source" "$total_bytes" "$headroom" "$under_budget" "$parse_status"
}

timestamp="$(date +%Y%m%d_%H%M%S)"
seed_spec="${MUON_TUNE_SEEDS:-1337}"
IFS=',' read -r -a seeds <<<"$seed_spec"

if [[ ${#seeds[@]} -eq 0 ]]; then
  echo "No seeds provided. Set MUON_TUNE_SEEDS, e.g. 1337,2027" >&2
  exit 1
fi

# Muon sweep profiles: name|momentum|backend_steps|warmup_start|warmup_steps
profiles=(
  "muon_base|0.95|5|0.85|500"
  "mom_090|0.90|5|0.85|500"
  "mom_098|0.98|5|0.85|500"
  "steps_3|0.95|3|0.85|500"
  "steps_7|0.95|7|0.85|500"
  "warmup_soft|0.95|5|0.80|1500"
)

# Reasonable defaults for tuning speed; override externally if needed.
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-300}"
export ITERATIONS="${ITERATIONS:-20000}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"
export VAL_MAX_TOKENS="${VAL_MAX_TOKENS:-262144}"
export FINAL_ROUNDTRIP_EVAL="${FINAL_ROUNDTRIP_EVAL:-1}"

# Keep best known architecture fixed while tuning optimizer.
export MODEL_DIM="${MODEL_DIM:-1152}"
export NUM_LAYERS="${NUM_LAYERS:-9}"
export RECURRENT_CORE_LAYERS="${RECURRENT_CORE_LAYERS:-3}"
export RECURRENT_STEPS="${RECURRENT_STEPS:-6}"
export SHARE_FFN_ACROSS_BLOCKS="${SHARE_FFN_ACROSS_BLOCKS:-1}"
export NUM_HEADS="${NUM_HEADS:-8}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-4}"

# Keep export fixed.
export QUANT_SCHEME="${QUANT_SCHEME:-int8}"
export COMPRESSOR="${COMPRESSOR:-auto}"
export WEIGHT_ORDER="${WEIGHT_ORDER:-none}"
export MIXED_LOW_PRECISION_SCHEME="${MIXED_LOW_PRECISION_SCHEME:-int8}"

echo
echo "=== Muon Sweep Config ==="
echo "seeds=${seed_spec} profiles=${#profiles[@]} max_wallclock_seconds=${MAX_WALLCLOCK_SECONDS} val_max_tokens=${VAL_MAX_TOKENS} final_roundtrip_eval=${FINAL_ROUNDTRIP_EVAL}"
echo "arch=model_dim:${MODEL_DIM} core_layers:${RECURRENT_CORE_LAYERS} recurrent_steps:${RECURRENT_STEPS} num_heads:${NUM_HEADS} num_kv_heads:${NUM_KV_HEADS}"

csv_path="logs/phase3_muon_matrix_${timestamp}.csv"
agg_path="logs/phase3_muon_matrix_${timestamp}_aggregate.csv"
printf 'Profile,Seed,MuonMomentum,MuonBackendSteps,MuonWarmupStart,MuonWarmupSteps,ExitCode,DurationSec,ValBpb,MetricSource,TotalSubmissionBytes,HeadroomBytes,UnderBudget,ParseStatus,LogPath\n' >"$csv_path"

for profile in "${profiles[@]}"; do
  IFS='|' read -r profile_name muon_momentum muon_backend_steps muon_warmup_start muon_warmup_steps <<<"$profile"

  for seed_raw in "${seeds[@]}"; do
    seed="$(trim "$seed_raw")"
    [[ -z "$seed" ]] && continue
    run_id="phase3muon_${timestamp}_${profile_name}_s${seed}"
    log_path="logs/${run_id}.txt"

    echo
    echo "=== Running ${profile_name} seed=${seed} (RUN_ID=${run_id}) ==="
    echo "Muon: momentum=${muon_momentum} backend_steps=${muon_backend_steps} warmup_start=${muon_warmup_start} warmup_steps=${muon_warmup_steps}"

    export RUN_ID="$run_id"
    export SEED="$seed"
    export MUON_MOMENTUM="$muon_momentum"
    export MUON_BACKEND_STEPS="$muon_backend_steps"
    export MUON_MOMENTUM_WARMUP_START="$muon_warmup_start"
    export MUON_MOMENTUM_WARMUP_STEPS="$muon_warmup_steps"

    start_epoch="$(date +%s)"
    set +e
    bash ./phase3_2xh100_full_regime.sh
    exit_code=$?
    set -e
    end_epoch="$(date +%s)"
    duration_sec="$((end_epoch - start_epoch))"

    IFS=$'\t' read -r val_bpb metric_source total_bytes headroom under_budget parse_status <<<"$(parse_run_log "$log_path")"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$profile_name" "$seed" "$muon_momentum" "$muon_backend_steps" "$muon_warmup_start" "$muon_warmup_steps" \
      "$exit_code" "$duration_sec" "$val_bpb" "$metric_source" "$total_bytes" "$headroom" "$under_budget" "$parse_status" "$log_path" \
      >>"$csv_path"

    if [[ "$exit_code" -ne 0 ]]; then
      echo "Warning: run ${profile_name} seed=${seed} failed with exit code ${exit_code}" >&2
    fi
  done
done

echo
echo "=== Muon Per-Run Summary ==="
if command -v column >/dev/null 2>&1; then
  column -s, -t <"$csv_path"
else
  cat "$csv_path"
fi

awk -F, '
NR==1 {next}
{
  p=$1
  runs[p]++
  if ($14 ~ /^ok/ && $9!="") {
    ok[p]++
    sum_bpb[p]+=$9
    if ($13=="True") valid[p]++
  }
}
END {
  print "Profile,Runs,OkRuns,ValidRuns,AvgValBpb"
  for (p in runs) {
    avg_bpb = (ok[p] > 0) ? sprintf("%.8f", sum_bpb[p] / ok[p]) : ""
    print p "," runs[p] "," (ok[p] + 0) "," (valid[p] + 0) "," avg_bpb
  }
}
' "$csv_path" >"$agg_path"

echo
echo "=== Muon Aggregate Summary ==="
if command -v column >/dev/null 2>&1; then
  column -s, -t <"$agg_path"
else
  cat "$agg_path"
fi

echo "Saved per-run CSV: $csv_path"
echo "Saved aggregate CSV: $agg_path"
