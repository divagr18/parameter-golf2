# Best Known Parameters (Live)

Last updated: 2026-04-07

This file tracks the best-performing known settings for the 1-GPU 10-minute cap16 workflow, plus what has been tested and rejected.

## Objective
- Primary metric: `final_*_roundtrip_exact val_bpb` (lower is better)
- Hard constraint: `submission_budget ... total <= 16000000`

## Best 8-GPU Non-Residual Run (current)
- `RUN_ID=best8g_nonres_dist_s070_w008_t20`
- Train-stop step/time: `13464` steps in `600s`
- Pre-quant val: `val_bpb=1.1839`
- Final roundtrip: `final_int8_zstd_roundtrip_exact val_bpb=1.19242312`
- Budget: `int8+zstd total=15951668` (headroom `48332`)

This is currently the strongest known non-residual 8-GPU result in this workspace.

---

## Current Locked Best Distill Knobs (1-GPU, 10 min, cap16)
Use these as the default unless a new sweep clearly beats them:

- `DISTILL_ENABLED=1`
- `DISTILL_WEIGHT=0.08`
- `DISTILL_TEMP=2.0`
- `DISTILL_START_FRAC=0.70`
- `DISTILL_EMA_DECAY=0.999`
- `BYTE_WEIGHTED_LOSS_ENABLED=0`

### Best run (latest schedule sweep)
- `RUN_ID=dist_sched_A_s070_w008_t20`
- Final: `val_bpb=1.30455613`
- Budget: `int8+zstd total=15907369` (headroom `92631`)

Reference log: `logs/dist_sched_A_s070_w008_t20.txt`

---

## Near-Best Distill Variants (latest sweep)
- `dist_sched_A_s080_w008_t20` -> `1.30558440`
- `dist_sched_B_s080_w010_t20` -> `1.30564603`
- `dist_sched_B_s075_w008_t20_baseline` -> `1.30625276`

Interpretation:
- `DISTILL_START_FRAC=0.70` beat both `0.75` baseline and `0.80` in this wallclock sweep.
- `weight=0.08` remains preferred over `0.10` at `start_frac=0.80`.

---

## Launcher/Runtime Defaults That Are Working Well
From `phase4_fixed_submission.sh` mainline:

- `TARGET_GPUS=1` (for quick sweeps)
- `MAX_WALLCLOCK_SECONDS=600`
- `ITERATIONS=20000` (wallclock normally stops earlier)
- `TARGET_GLOBAL_TOKENS=65536`
- `GRAD_ACCUM_STEPS=1`
- `MODEL_DIM=512`
- `NUM_LAYERS=9`
- `NUM_HEADS=8`
- `NUM_KV_HEADS=4`
- `MLP_MULT=2`
- `RECURRENT_CORE_LAYERS=0`
- `RECURRENT_STEPS=0`
- `USE_TORCH_COMPILE=1`
- `QUANT_SCHEME=int8`
- `COMPRESSOR=zstd`
- `MIXED_LOW_PRECISION_SCHEME=int8`
- `SWA_ENABLED=1`
- `EVAL_STRIDE_FRAC=0.5`
- `BIGRAM_RANK=32`
- `BIGRAM_LR=0.04`

---

## Tested/Rejected (for now)

### Byte-weighted loss
- Status: **deprioritized** until clean step-matched rerun confirms value.
- Observed severe regression in one timed run (`~1.366`), but that run also had fewer steps; fairness rerun still pending.
- Current production choice: `BYTE_WEIGHTED_LOSS_ENABLED=0`.

### SSM hybrid blocks
- Status: rejected for current objective.
- Reason: slower and/or worse bpb under tight 10-minute + 16MB budget.

### MTP auxiliary loss
- Status: rejected for current objective.
- Reason: added complexity and empirical regressions in this regime.

### Logit regularization
- Status: not a winner at tested strengths.

---

## Rules for Fair Comparisons
Use these for all A/B conclusions:

1. Keep same `SEED` unless intentionally testing seed robustness.
2. Change only one logical factor at a time.
3. Prefer fixed-step comparison (`MAX_WALLCLOCK_SECONDS=99999`, fixed `ITERATIONS`) when measuring objective deltas.
4. Always compare both:
   - `final_*_roundtrip_exact val_bpb`
   - `submission_budget ... total`
5. Use unique `RUN_ID` every run.

---

## Suggested Next Sequence
1. Run fixed-step confirmation for `start_frac=0.70` vs `0.75` (same seed, same iterations).
2. If `0.70` holds, keep it as production default.
3. Then move to trigram/engram experiment.

---

## Quick Grep Template
```bash
for f in logs/*.txt; do
  echo "=== $f ==="
  grep -E "^step:[0-9]+/[0-9]+ val_loss|^submission_budget .*total:|^final_.*roundtrip_exact|^stopping_early:" "$f" | tail -n 12
done
```
