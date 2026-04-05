#!/usr/bin/env bash
set -euo pipefail

# Linux/bash version of tokenizer_1gpu_matrix.ps1
# Usage:
#   TOKENIZER_1GPU_MODE=quick TOKENIZER_1GPU_SEEDS=1337,2027 bash ./tokenizer_1gpu_matrix.sh

cd "$(dirname "$0")"
mkdir -p logs

timestamp="$(date +%Y%m%d_%H%M%S)"
: "${TOKENIZER_1GPU_MODE:=quick}"
: "${TOKENIZER_1GPU_SEEDS:=1337,2027}"
: "${TOKENIZER_1GPU_SWEEP_ID:=tok1gpu_${timestamp}}"
: "${TOKENIZER_1GPU_CONFIG:=./data/tokenizer_specs_1gpu_matrix.json}"
: "${TOKENIZER_1GPU_OUTPUT_ROOT:=./data/tokenizer_sweeps/${TOKENIZER_1GPU_SWEEP_ID}}"
: "${TOKENIZER_1GPU_SKIP_EXPORT:=0}"
: "${TOKENIZER_1GPU_TOKENIZER_NAMES:=}"
: "${TOKENIZER_1GPU_TRAINER_DOCS:=}"

: "${MATCHED_FINEWEB_REPO_ID:=willdepueoai/parameter-golf}"
: "${MATCHED_FINEWEB_REMOTE_ROOT_PREFIX:=datasets}"

if [[ "${TOKENIZER_1GPU_MODE}" == "full" ]]; then
  default_wallclock=600
  default_val_max_tokens=0
  default_warmup=200
else
  default_wallclock=240
  default_val_max_tokens=1048576
  default_warmup=50
fi

# Training defaults (override with env before calling script)
: "${DEVICE:=cuda}"
: "${USE_TORCH_COMPILE:=1}"
: "${ITERATIONS:=20000}"
: "${MAX_WALLCLOCK_SECONDS:=${default_wallclock}}"
: "${WARMUP_STEPS:=${default_warmup}}"
: "${TRAIN_LOG_EVERY:=200}"
: "${TRAIN_BATCH_TOKENS:=65536}"
: "${VAL_BATCH_SIZE:=131072}"
: "${VAL_LOSS_EVERY:=0}"
: "${VAL_MAX_TOKENS:=${default_val_max_tokens}}"
: "${FINAL_ROUNDTRIP_EVAL:=1}"
: "${SUBMISSION_SIZE_BUDGET_BYTES:=16000000}"

: "${MODEL_DIM:=512}"
: "${NUM_LAYERS:=9}"
: "${NUM_HEADS:=8}"
: "${NUM_KV_HEADS:=4}"
: "${MLP_MULT:=2}"
: "${TIE_EMBEDDINGS:=1}"
: "${RECURRENT_CORE_LAYERS:=0}"
: "${RECURRENT_STEPS:=0}"
: "${SHARE_FFN_ACROSS_BLOCKS:=0}"
: "${USE_SWIGLU:=1}"
: "${GRAD_CLIP_NORM:=1.0}"

: "${EVAL_STRIDE_FRAC:=0.5}"
: "${EVAL_SEQ_LEN:=0}"
: "${EVAL_ROPE_SCALE:=1.0}"

: "${BIGRAM_RANK:=32}"
: "${BIGRAM_LR:=0.04}"
: "${SWA_ENABLED:=1}"
: "${SWA_COLLECT_EVERY:=10}"
: "${CURRICULUM_ENABLED:=0}"
: "${CURRICULUM_MIN_SEQ_LEN:=256}"
: "${CURRICULUM_STEPS:=5000}"

: "${MUON_MOMENTUM:=0.98}"
: "${MUON_BACKEND_STEPS:=5}"
: "${MUON_MOMENTUM_WARMUP_START:=0.85}"
: "${MUON_MOMENTUM_WARMUP_STEPS:=500}"
: "${MATRIX_LR:=0.04}"
: "${SCALAR_LR:=0.04}"
: "${EMBED_LR:=0.6}"
: "${TIED_EMBED_LR:=0.05}"
: "${WARMDOWN_ITERS:=3000}"

: "${QUANT_SCHEME:=mixed}"
: "${MIXED_LOW_PRECISION_SCHEME:=int4}"
: "${QAT_SCHEME:=int4}"
: "${QAT_START_STEP:=4200}"
: "${COMPRESSOR:=zstd}"
: "${WEIGHT_ORDER:=none}"

if [[ "${TOKENIZER_1GPU_SKIP_EXPORT}" != "1" ]]; then
  mkdir -p "${TOKENIZER_1GPU_OUTPUT_ROOT}"
  echo "=== Exporting tokenizer datasets ==="
  export_args=(
    data/download_hf_docs_and_tokenize.py
    --repo-id "${MATCHED_FINEWEB_REPO_ID}"
    --remote-root "${MATCHED_FINEWEB_REMOTE_ROOT_PREFIX}"
    --output-root "${TOKENIZER_1GPU_OUTPUT_ROOT}"
    --tokenizer-config "${TOKENIZER_1GPU_CONFIG}"
  )
  if [[ -n "${TOKENIZER_1GPU_TRAINER_DOCS}" ]]; then
    export_args+=(--tokenizer-train-docs "${TOKENIZER_1GPU_TRAINER_DOCS}")
  fi
  python "${export_args[@]}"
fi

manifest_path="${TOKENIZER_1GPU_OUTPUT_ROOT}/manifest.json"
if [[ ! -f "${manifest_path}" ]]; then
  echo "Manifest not found: ${manifest_path}" >&2
  exit 1
fi

runs_tsv="$(mktemp)"
python - "$manifest_path" "$TOKENIZER_1GPU_OUTPUT_ROOT" "$TOKENIZER_1GPU_TOKENIZER_NAMES" > "$runs_tsv" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
root = Path(sys.argv[2])
name_filter_raw = sys.argv[3].strip()
name_filter = {x.strip() for x in name_filter_raw.split(',') if x.strip()} if name_filter_raw else None
m = json.loads(manifest_path.read_text(encoding='utf-8'))

tok_map = {str(t.get('name')): t for t in m.get('tokenizers', [])}
for ds in m.get('datasets', []):
    tok_name = str(ds.get('tokenizer_name', ''))
    if not tok_name:
        continue
    if name_filter is not None and tok_name not in name_filter:
        continue
    tok = tok_map.get(tok_name)
    if not tok:
        continue
    model_path_rel = tok.get('model_path')
    if not model_path_rel:
        continue
    dataset_path_rel = ds.get('path')
    if not dataset_path_rel:
        continue
    data_path = (root / dataset_path_rel).as_posix()
    tok_path = (root / model_path_rel).as_posix()
    vocab = int(ds.get('vocab_size', tok.get('vocab_size', 1024)))
    print('\t'.join([tok_name, str(ds.get('name', '')), str(vocab), data_path, tok_path]))
PY

if [[ ! -s "$runs_tsv" ]]; then
  echo "No tokenizer datasets selected from ${manifest_path}" >&2
  rm -f "$runs_tsv"
  exit 1
fi

echo "=== Tokenizer 1-GPU Matrix Config ==="
echo "mode=${TOKENIZER_1GPU_MODE} seeds=${TOKENIZER_1GPU_SEEDS}"
echo "output_root=${TOKENIZER_1GPU_OUTPUT_ROOT}"

csv_runs="logs/tokenizer_1gpu_matrix_${timestamp}.csv"
csv_agg="logs/tokenizer_1gpu_matrix_${timestamp}_aggregate.csv"
echo "TokenizerName,DatasetName,Seed,RunId,ExitCode,DurationSec,VocabSize,ValBpbPreQuant,ValBpbRoundtrip,QuantizationGap,ValLossRoundtrip,TotalSubmissionBytes,BudgetBytes,BudgetHeadroomBytes,UnderBudget,ParseStatus,LogPath" > "$csv_runs"

parse_log() {
  local log_path="$1"
  python - "$log_path" <<'PY'
import re
import sys
from pathlib import Path

p = Path(sys.argv[1])
if not p.exists():
    print("\t\t\t\t\t\t\tmissing_log")
    sys.exit(0)

lines = p.read_text(encoding='utf-8', errors='ignore').splitlines()
pre = ""
rt_bpb = ""
rt_loss = ""
total = ""
budget = ""
headroom = ""
under = ""
status = "ok"

for line in lines:
    if re.match(r'^step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:[0-9.]+', line):
        m = re.search(r'val_bpb:([0-9.]+)', line)
        if m:
            pre = m.group(1)

for line in lines:
    if re.match(r'^final_.*_roundtrip_exact .*val_loss:[0-9.]+ val_bpb:[0-9.]+', line):
        m = re.search(r'val_bpb:([0-9.]+)', line)
        if m:
            rt_bpb = m.group(1)
        m = re.search(r'val_loss:([0-9.]+)', line)
        if m:
            rt_loss = m.group(1)

if not rt_bpb:
    status = "missing_roundtrip"

for line in lines:
    if re.match(r'^submission_budget .+ total:[0-9]+ budget:[0-9]+ (headroom_bytes|over_bytes):[0-9]+', line):
        m = re.search(r'total:([0-9]+)', line)
        if m:
            total = m.group(1)
        m = re.search(r'budget:([0-9]+)', line)
        if m:
            budget = m.group(1)
        m = re.search(r'headroom_bytes:([0-9]+)', line)
        if m:
            headroom = m.group(1)
            under = "True"
        m = re.search(r'over_bytes:([0-9]+)', line)
        if m:
            headroom = "-" + m.group(1)
            under = "False"

print('\t'.join([pre, rt_bpb, rt_loss, total, budget, headroom, under, status]))
PY
}

IFS=',' read -r -a seeds <<< "${TOKENIZER_1GPU_SEEDS}"
while IFS=$'\t' read -r tok_name ds_name vocab data_path tok_path; do
  for seed_raw in "${seeds[@]}"; do
    seed="$(echo "$seed_raw" | xargs)"
    [[ -z "$seed" ]] && continue
    run_id="tok1gpu_${timestamp}_${tok_name}_s${seed}"
    log_path="logs/${run_id}.txt"

    echo "=== Running tokenizer=${tok_name} seed=${seed} (RUN_ID=${run_id}) ==="

    start_ts="$(date +%s)"
    (
      export RUN_ID="$run_id"
      export SEED="$seed"
      export DATA_PATH="$data_path"
      export TOKENIZER_PATH="$tok_path"
      export VOCAB_SIZE="$vocab"

      export DEVICE USE_TORCH_COMPILE ITERATIONS MAX_WALLCLOCK_SECONDS WARMUP_STEPS TRAIN_LOG_EVERY
      export TRAIN_BATCH_TOKENS VAL_BATCH_SIZE VAL_LOSS_EVERY VAL_MAX_TOKENS FINAL_ROUNDTRIP_EVAL SUBMISSION_SIZE_BUDGET_BYTES
      export MODEL_DIM NUM_LAYERS NUM_HEADS NUM_KV_HEADS MLP_MULT TIE_EMBEDDINGS RECURRENT_CORE_LAYERS RECURRENT_STEPS SHARE_FFN_ACROSS_BLOCKS USE_SWIGLU GRAD_CLIP_NORM
      export EVAL_STRIDE_FRAC EVAL_SEQ_LEN EVAL_ROPE_SCALE
      export BIGRAM_RANK BIGRAM_LR SWA_ENABLED SWA_COLLECT_EVERY CURRICULUM_ENABLED CURRICULUM_MIN_SEQ_LEN CURRICULUM_STEPS
      export MUON_MOMENTUM MUON_BACKEND_STEPS MUON_MOMENTUM_WARMUP_START MUON_MOMENTUM_WARMUP_STEPS MATRIX_LR SCALAR_LR EMBED_LR TIED_EMBED_LR WARMDOWN_ITERS
      export QUANT_SCHEME MIXED_LOW_PRECISION_SCHEME QAT_SCHEME QAT_START_STEP COMPRESSOR WEIGHT_ORDER

      torchrun --standalone --nnodes=1 --nproc_per_node=1 train_gpt.py
    )
    exit_code=$?
    end_ts="$(date +%s)"
    duration=$((end_ts - start_ts))

    IFS=$'\t' read -r pre_bpb rt_bpb rt_loss total budget headroom under status < <(parse_log "$log_path")

    gap=""
    if [[ -n "$pre_bpb" && -n "$rt_bpb" ]]; then
      gap="$(python - <<PY
pre=float('${pre_bpb}')
rt=float('${rt_bpb}')
print(f"{rt-pre:.6f}")
PY
)"
    fi

    echo "${tok_name},${ds_name},${seed},${run_id},${exit_code},${duration},${vocab},${pre_bpb},${rt_bpb},${gap},${rt_loss},${total},${budget},${headroom},${under},${status},${log_path}" >> "$csv_runs"

    if [[ "$exit_code" -ne 0 ]]; then
      echo "Run failed: tokenizer=${tok_name} seed=${seed} exit=${exit_code}" >&2
    fi
  done
done < "$runs_tsv"

python - "$csv_runs" "$csv_agg" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict

runs_path, agg_path = sys.argv[1], sys.argv[2]
rows = []
with open(runs_path, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

groups = defaultdict(list)
for r in rows:
    groups[r['TokenizerName']].append(r)

with open(agg_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow([
        'TokenizerName', 'Runs', 'OkRuns',
        'MedianValBpbRoundtrip', 'MeanValBpbRoundtrip',
        'MedianQuantizationGap', 'MeanQuantizationGap'
    ])
    for tok, rs in sorted(groups.items()):
        ok = [r for r in rs if r.get('ParseStatus') == 'ok' and r.get('ValBpbRoundtrip')]
        bpb = [float(r['ValBpbRoundtrip']) for r in ok]
        gap = [float(r['QuantizationGap']) for r in ok if r.get('QuantizationGap')]
        med_bpb = f"{statistics.median(bpb):.8f}" if bpb else ''
        mean_bpb = f"{(sum(bpb)/len(bpb)):.8f}" if bpb else ''
        med_gap = f"{statistics.median(gap):.8f}" if gap else ''
        mean_gap = f"{(sum(gap)/len(gap)):.8f}" if gap else ''
        w.writerow([tok, len(rs), len(ok), med_bpb, mean_bpb, med_gap, mean_gap])
PY

rm -f "$runs_tsv"

echo "Saved per-run CSV: $csv_runs"
echo "Saved aggregate CSV: $csv_agg"
