# Reports

Generated per-slice evaluation reports. Each report header carries the normalizer
version, the exact command, the git SHA, and the seed that produced it.

## Layout

- **`specs/<name>.json`** — a report *definition*: which systems, which slices
  (each a manifest), and where each system's predictions live. Committed.
- **`predictions/<system>__<slice>.jsonl`** — a system's hypotheses for one
  slice (header line + `{"id", "hypothesis"}` per segment). Produced by the
  baseline/eval notebooks on Kaggle, where the models and audio live.
- **`<name>.md` / `<name>.json`** — the rendered report. Committed as evidence
  *once it carries real numbers*; regenerated whenever the scoring script or
  predictions change.

## Regenerate

```bash
python -m shono.eval --report baselines
```

A slice whose manifest or predictions do not exist yet renders `—` (pending) —
never a fabricated number. The command produces an honest skeleton today and
fills in as the gated Kaggle runs land their predictions.
