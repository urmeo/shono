# Notebooks

Kaggle training and benchmark notebooks. Each is a **thin driver** over the tested
`shono` package — the logic lives in `src/shono/`, so the notebooks stay short and
the behavior is unit-tested. Every run records its seed + full config and
checkpoints for session-limit resume.

- **`train_whisper_medium.ipynb`** — full fine-tune of Whisper-medium (Bengali)
  with 8-bit AdamW, resumable across the ~12 h Kaggle session cap. Drives
  `shono.train`: builds the license-checked training mix, runs a one-step CPU
  smoke, trains, and writes the experiment record.

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
