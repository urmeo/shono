# Predictions

One JSON Lines file per (system, slice): the hypotheses a system produced for a
slice's segments, keyed by segment id. Produced by the baseline/eval notebooks
on Kaggle (where the models and audio live), then committed here so any reported
number is reproducible from `python -m shono.eval`.

Format — header line, then one line per segment:

```jsonl
{"predictions": {"system": "whisper-large-v3 (zero-shot)", "manifest": "cv-bn-test", "run_context": {…}}}
{"id": "cv-bn-000001", "hypothesis": "…"}
```

The `run_context` mirrors `shono.provenance.RunContext` (seed, config, packages,
git SHA) so the exact inference run behind each hypothesis set is recorded.
Naming: `<system-slug>__<slice-name>.jsonl`.
