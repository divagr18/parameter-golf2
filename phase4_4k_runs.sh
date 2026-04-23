#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mode="${1:-both}"

run_mixed_576_4k() {
  RUN_ID="${RUN_ID:-mixed576_4k_int5}" \
  MODEL_DIM=576 \
  NUM_LAYERS=9 \
  ITERATIONS=4000 \
  MAX_WALLCLOCK_SECONDS=0 \
  QUANT_SCHEME=mixed \
  MIXED_LOW_PRECISION_SCHEME="${MIXED_LOW_PRECISION_SCHEME:-int5}" \
  JPCR_ENABLED="${JPCR_ENABLED:-0}" \
  DISTILL_ENABLED="${DISTILL_ENABLED:-0}" \
  BIGRAM_RANK="${BIGRAM_RANK:-0}" \
  RESIDUAL_NGRAM_ENABLED="${RESIDUAL_NGRAM_ENABLED:-0}" \
  bash ./phase4_fixed_submission.sh
}

run_best_current_4k() {
  RUN_ID="${RUN_ID:-bestcurrent_4k}" \
  MODEL_DIM="${MODEL_DIM:-448}" \
  NUM_LAYERS="${NUM_LAYERS:-9}" \
  ITERATIONS=4000 \
  MAX_WALLCLOCK_SECONDS=0 \
  QUANT_SCHEME="${QUANT_SCHEME:-int8}" \
  MIXED_LOW_PRECISION_SCHEME="${MIXED_LOW_PRECISION_SCHEME:-int8}" \
  JPCR_ENABLED="${JPCR_ENABLED:-0}" \
  DISTILL_ENABLED="${DISTILL_ENABLED:-0}" \
  BIGRAM_RANK="${BIGRAM_RANK:-0}" \
  RESIDUAL_NGRAM_ENABLED="${RESIDUAL_NGRAM_ENABLED:-0}" \
  bash ./phase4_fixed_submission.sh
}

case "$mode" in
  mixed576|mixed)
    run_mixed_576_4k
    ;;
  best|bestcurrent|best4k)
    run_best_current_4k
    ;;
  both)
    run_mixed_576_4k
    run_best_current_4k
    ;;
  *)
    echo "Usage: bash ./phase4_4k_runs.sh [mixed576|best4k|both]" >&2
    exit 1
    ;;
esac