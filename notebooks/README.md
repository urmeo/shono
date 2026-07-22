# Notebooks

Kaggle training and benchmark notebooks. Each is a **thin driver** over the tested
`shono` package — the logic lives in `src/shono/`, so the notebooks stay short and
the behavior is unit-tested. Every run records its seed + full config and
checkpoints for session-limit resume.

- **`prepare_data.ipynb`** — raw datasets → manifests. Builds the audio-free
  train + eval manifests M2/M3 consume (Common Voice + FLEURS via the package
  builders; OpenSLR/MUCS/Bengali-Loop via `build_manifest`), validates each
  against the license floor, and runs the train/test leakage audit. **Run first.**
- **`baselines.ipynb`** — the numbers every fine-tune must beat. Runs each
  zero-shot Whisper baseline over every eval slice via `run_over_manifest`, writes
  predictions, and regenerates the baseline report.
- **`train_whisper_medium.ipynb`** — full fine-tune of Whisper-medium (Bengali)
  with 8-bit AdamW, resumable across the ~12 h Kaggle session cap. Drives
  `shono.train`: builds the license-checked training mix, runs a one-step CPU
  smoke, trains, and writes the experiment record.
- **`predict.ipynb`** — runs the fine-tuned model through the rest of the stack:
  CT2 conversion + long-form pipeline + RTF (M4), pyannote diarization + DER (M5),
  the code-switch slice (M6), and the commercial APIs within a $0 budget guard
  (M7) — filling the `longform`, `codeswitch`, and `full` reports plus a DER report.

Run order: `prepare_data` → `baselines` (M2) → `train_whisper_medium` (M3) →
`predict` (M4–M7).

## Training environment

torch ships with the Kaggle image; the notebook installs the rest:

```bash
pip install 'transformers>=4.44' 'accelerate>=0.33' 'bitsandbytes>=0.43' \
            'datasets>=2.20' 'librosa>=0.10' 'evaluate>=0.4'
```

These heavy, GPU-oriented dependencies are intentionally **not** in the project
lockfile (the scoring harness stays lightweight). To run the training code or its
CPU smoke locally without touching the lock:

```bash
uv run --with torch --with transformers pytest tests/test_train.py -k smoke
```
