# Shono (শোনো) — Bengali ASR that survives the real world

Most Bengali speech recognition is measured on short, clean, read-aloud clips. Real audio isn't like that. **Shono** targets the hard parts:

- **Long-form audio** — lectures and podcasts, chunked, transcribed, merged, timestamped
- **Speaker diarization** — who said what, not just what was said
- **Bangla-English code-switching** — the way people actually talk
- **Honest evaluation** — WER *and* CER per segment type, with confidence intervals, against the base model and commercial APIs

> **Status: foundation.** The evaluation harness is built and tested (41 tests, one `./verify` command). No model is fine-tuned yet — **no accuracy numbers are claimed yet.** When they appear, they ship with the scoring script, config, and confidence intervals that produced them.

## The pipeline

```mermaid
flowchart LR
    A["🎙️ Long-form audio<br/>lecture · podcast · interview"] --> B["Silero VAD<br/>speech detection"]
    B --> C["Chunking at<br/>natural pauses (20–28 s)"]
    C --> D["Fine-tuned Whisper<br/>(Bengali)"]
    D --> E["Merge + word<br/>timestamps"]
    B --> F["pyannote diarization<br/>Bengali-tuned segmentation"]
    F --> G["VAD-intersection<br/>(kills boundary hallucinations)"]
    E --> H["🗣️ Speaker-attributed<br/>Bengali transcript"]
    G --> H
```

Every number the pipeline produces passes one gate:

```mermaid
flowchart LR
    R["Reference"] --> N["Frozen normalizer v1.1.0<br/>NFC → bnunicodenormalizer →<br/>punct/symbols→space → digits"]
    Y["Hypothesis"] --> N
    N --> S["WER + CER<br/>(raw and normalized)"]
    S --> CI["Blockwise bootstrap<br/>95% CI per slice"]
    CI --> P["📊 Per-slice report<br/>read · long-form · code-switch"]
```

## Why the scorer came first

Bengali WER is easy to fake by accident — the ecosystem's most common normalizer (Whisper's default) strips vowel signs and *inflates* apparent accuracy:

```mermaid
flowchart LR
    K["কি&nbsp;&nbsp;(input)"] --> W["generic normalizer<br/>(Whisper default)"]
    W --> BAD["ক&nbsp;&nbsp;❌ word corrupted,<br/>errors hidden"]
    K --> SH["shono normalizer<br/>(frozen, version-stamped)"]
    SH --> OK["কি&nbsp;&nbsp;✅ intact —<br/>there's a test proving it"]
```

Three guarantees, each pinned by tests that fail if the behavior mutates:

- **`normalize`** — matras survive; Latin code-switch tokens keep case; symbol spacing can't flip a score (`৳১০০` ≡ `৳ ১০০`)
- **`score`** — explicit raw contract, no hidden case folding; both sides always normalized together
- **`ci`** — recordings resampled, not segments (they're correlated); every number ships as `x% [lo, hi]`

## The bar to beat

Published results on the [Bengali-Loop](https://arxiv.org/abs/2602.14291) long-form benchmark — Shono's bar is added only when measured:

```mermaid
xychart-beta
    title "Long-form Bengali WER, % (lower is better)"
    x-axis ["tugstugi medium (open baseline)", "WhisperAlign 2026", "Bangla-WhisperDiar 2026"]
    y-axis "WER %" 0 --> 40
    bar [34.07, 25.2, 24.41]
```

```mermaid
xychart-beta
    title "Bengali diarization DER, % (lower is better)"
    x-axis ["pyannote out-of-box", "+ Bengali-tuned segmentation (2026)"]
    y-axis "DER %" 0 --> 45
    bar [40.08, 19.4]
```

Sources: [Bengali-Loop](https://arxiv.org/abs/2602.14291), [WhisperAlign](https://arxiv.org/abs/2603.04809), [Bangla-WhisperDiar](https://arxiv.org/abs/2605.08214). For scale: zero-shot Whisper large-v3 sits at ~33.9% WER on Bengali FLEURS ([FLEURS-SLU](https://arxiv.org/abs/2501.06117)).

**What nobody publishes yet — where Shono aims:** a Bangla-English **code-switch slice**, an **independent per-slice comparison vs commercial APIs**, and a **usable public demo**.

## Data

```mermaid
pie showData title Training & eval hours by source
    "Common Voice bn — CC0" : 54
    "OpenSLR SLR53 — CC BY-SA 4.0" : 215
    "Bengali-Loop — CC BY 4.0" : 158.6
    "MUCS Bn-En code-switch — CC BY-SA 4.0" : 46
```

Audio never enters this repo — [`data/`](data/) holds manifests, checksums, and per-source licenses only.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/); Python 3.11 resolves automatically.

```bash
git clone https://github.com/urme-b/shono && cd shono
./verify        # install + lint + full test suite
```

```python
from shono.eval import score, blockwise_bootstrap_ci

report = score(references, hypotheses)      # raw + normalized WER/CER
ci = blockwise_bootstrap_ci(records)        # records = (recording_id, ref, hyp)
print(f"WER {ci.point:.1%} [{ci.lower:.1%}, {ci.upper:.1%}]")
```

```
src/shono/eval/   frozen normalization · WER/CER scoring · bootstrap CIs
tests/            fixture tests with hand-computed values
data/             dataset manifests + licenses — never audio
notebooks/        Kaggle training/benchmark notebooks
reports/          generated evaluation reports
```

## Roadmap

```mermaid
flowchart LR
    M1["✅ Eval harness"] --> M2["Baselines<br/>measured"] --> M3["Fine-tune<br/>Whisper (Kaggle)"] --> M4["Long-form<br/>pipeline"]
    M4 --> M5["Diarization"] --> M6["Code-switch<br/>slice"] --> M7["Benchmark vs<br/>commercial APIs"] --> M8["🚀 Public demo<br/>(HF Space)"]
```

## License

Code: [MIT](LICENSE). Datasets and model weights carry their own licenses, recorded per source in `data/`.
