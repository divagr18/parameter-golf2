# Tokenizer 1-GPU Test Plan

## Goal
Find tokenizer variants that improve final roundtrip `val_bpb` on 1 GPU while keeping training setup fixed, then promote only strong candidates to 8-GPU tests.

## Constraints to keep fixed
- Same model/training hyperparameters across tokenizer runs.
- Same wallclock cap and evaluation path.
- Same quantization/export settings.
- Same docs source (verify `docs_sha256` in manifest).

Relevant implementation points:
- Tokenizer retraining/export pipeline: [data/download_hf_docs_and_tokenize.py](data/download_hf_docs_and_tokenize.py)
- Tokenizer config schema: [data/tokenizer_specs.json](data/tokenizer_specs.json)
- Data workflow notes: [data/README.md](data/README.md)
- Training requires SentencePiece `.model`: [train_gpt.py](train_gpt.py#L1255-L1257)

## 1-GPU protocol (phased)

### Phase 0 — Build and sanity check
For each tokenizer config candidate:
1. Export tokenizer + shards from fixed docs cache.
2. Confirm output has:
   - `manifest.json`
   - tokenizer `.model` and `.vocab`
   - `fineweb_train_*.bin`, `fineweb_val_*.bin`
3. Run one short smoke train to ensure no runtime or vocab mismatch errors.

### Phase 1 — Fast screening
- Hardware: 1 GPU.
- Keep all model and optimizer knobs fixed.
- Use short wallclock (e.g. 180s–240s) for throughput + early quality signal.
- Run each candidate with 2 seeds.

Track:
- `val_bpb` (final roundtrip exact)
- pre/post quant gap
- steps reached in wallclock
- avg step time
- artifact bytes and budget headroom

Promotion rule from Phase 1:
- Promote if median `val_bpb` beats control by a meaningful margin and no severe quantization gap regression.

### Phase 2 — Full 1-GPU confirmation
- Re-run promoted candidates at full 600s wallclock.
- 3 seeds each.
- Rank by median and variance of final roundtrip `val_bpb`.

Promotion rule to 8-GPU:
- Candidate beats control at 1 GPU with stable variance and acceptable quantization gap.

## Candidate tokenizer matrix (start small)

Use 6 initial variants:
1. `sp1024_control` — baseline equivalent settings.
2. `sp1024_nodigit` — `split_digits=false`.
3. `sp1024_dummy_prefix` — `add_dummy_prefix=true`.
4. `sp1024_nfkc` — alternate normalization rule.
5. `sp1024_cov_hi` — higher `character_coverage`.
6. `sp1024_cov_lo` — lower `character_coverage`.

Notes:
- Keep vocab size fixed at first (1024) to isolate tokenizer behavior.
- Add vocab sweep (e.g. 896/1152) only after one setting family clearly wins.

## Run bookkeeping template
For each run, record:
- `run_id`
- tokenizer config name
- tokenizer config hash
- docs `docs_sha256`
- seed
- wallclock
- final roundtrip `val_bpb` exact
- pre-quant `val_bpb`
- quantization gap
- submission bytes and headroom
- steps reached
- avg step ms

## Decision policy
- Use median across seeds, not single best run.
- Reject candidates that improve pre-quant but worsen roundtrip heavily.
- Keep at least one control run in every batch window.

## Exit criteria for 1-GPU stage
Move to 8-GPU only when a tokenizer candidate shows:
- repeatable improvement over control on 600s 1-GPU runs,
- stable variance,
- no budget or export regressions.
