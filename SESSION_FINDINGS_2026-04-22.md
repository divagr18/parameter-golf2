# Session Findings (2026-04-22)

## Scope
- Goal: improve `final_*_roundtrip_exact val_bpb` under 16MB budget.
- Focus: `sp8192` tokenizer path, quantization strategy, and late-stage eval improvements (TTT / residual engrams).

## Important Code/Config Updates Made
- Fixed `int5` validation mismatch in `train_gpt.py`:
  - `MIXED_LOW_PRECISION_SCHEME` now consistently accepts `int5` where expected.
- Added `INT5_KEEP_FLOAT_FP32_NAME_PATTERNS` handling in `train_gpt.py` so `int5` has dedicated keep-float controls.
- Updated `phase4_fixed_submission.sh` comments/docs to include `int5` in QAT docs.
- Confirmed latest commit (`fix vocab`) behavior:
  - `VOCAB_SIZE` now drives default `DATA_PATH` and `TOKENIZER_PATH` in `phase4_fixed_submission.sh`.
  - Use `unset DATA_PATH TOKENIZER_PATH` before runs to avoid stale path overrides.

## Key Experimental Results

### Tokenizer Fairness Baseline
- `sp1024_r1_int8_gptq`: `val_bpb=1.45229886`, total `11047122` bytes.
- `sp8192` clearly beats `sp1024` in this run family.

### Int8 vs Int5 (sp8192)
- Best `int8` run family clearly beats all tested `int5` families on roundtrip bpb.
- `int5` repeatedly showed strong pre-quant scores but large roundtrip degradation (quantization gap).
- Conclusion in this session: `int5` is currently not competitive for best-bpb objective on this setup.

### Best Int8 Progression
- `sp8192_r1_int8_gptq`: `1.40895069`.
- `sp8192_int8_sweep_r1_ctrl`: improved to `1.40703282` (better than other small sweeps).
- Capacity bump test:
  - `sp8192_scale4k_r1_d448_l8`: `1.40881464`, total `13986821`.
  - `sp8192_scale4k_r2_d448_l9`: `1.40168936`, total `15284512` (winner at 4k).
- Proper 600s run on winner:
  - `sp8192_winner_600s_d448_l9`
  - pre-quant val around `1.2920`, roundtrip valid under budget:
  - total `15313924` bytes, headroom `686076`.

### Engram / TTT Runs (600s, d448_l9)
- Engram-only:
  - `val_bpb=1.31742338`
  - **over budget** by `87494` bytes.
- TTT-only:
  - `val_bpb=1.31850231`
  - valid, headroom `823971`.
- Distill activation issue found:
  - Distill configured but often not active due to late start trigger.
  - For wallclock-capped runs, use explicit `DISTILL_START_STEP` (e.g. `5000`) and disable frac triggers.

## Operational Lessons
- Reused `RUN_ID` can mix old/new lines in logs and confuse interpretation; always use unique run IDs.
- Distill start by `iter_frac` can miss activation in 600s runs if steps are low.
- To force distill in 600s regime:
  - `DISTILL_START_STEP=5000`
  - `DISTILL_START_FRAC=-1`
  - `DISTILL_START_WALLCLOCK_FRAC=-1`

## Current Best Direction
- Mainline candidate: `sp8192`, `int8+zstd+GPTQ`, `MODEL_DIM=448`, `NUM_LAYERS=9`, 600s wallclock.
- For added eval-time gains:
  - Keep TTT conservative (`TTT_STEPS=1`, `TTT_LR=5e-4`).
  - If using residual engrams, lower ranks to stay under budget (e.g. 24/12 or 16/8).

## Suggested Immediate Next Run
- `engram + ttt` with smaller engram ranks and explicit distill start step:
  - `RESIDUAL_BIGRAM_RANK=24`
  - `RESIDUAL_TRIGRAM_RANK=12`
  - `TTT_ENABLED=1`, `TTT_LR=5e-4`, `TTT_STEPS=1`
  - `DISTILL_START_STEP=5000`, fracs disabled.

