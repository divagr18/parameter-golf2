# Parameter Golf Plan Refinement

Plan is solid. I’d tighten it into an execution loop so we get signal fast and avoid misleading gains.

1. Lock a true baseline first.
- Use full validation (`VAL_MAX_TOKENS=0`, `FINAL_INT8_ROUNDTRIP_EVAL=1`).
- Record `val_bpb`, total artifact bytes, train wallclock.

2. Do Phase 2 before retraining.
- Add export options: `QUANT_SCHEME=int8|int4|mixed`, `COMPRESSOR=zlib|zstd`.
- Run these on the same trained checkpoint first to isolate pure compression effects.

3. Then do recurrence/tying in one controlled branch.
- Keep tokenizer fixed while changing architecture.
- Compare only against same train budget and same eval settings.

4. Only then touch tokenizer.
- Tokenizer changes can dominate bpb and make architecture comparisons noisy.

5. Track every run in one table.
- `run_id, code_sha, train_tokens, val_bpb, model_bytes, code_bytes, total_bytes, wallclock`.

If you want, I’ll start now with Phase 2 implementation in `train_gpt.py`: `int4 + zstd + configurable export pipeline`, then run a baseline-vs-new comparison.
