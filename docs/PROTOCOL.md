# Data and runtime protocol

## Inputs

Use one explicit `AUDIO_ROOT`. Manifest audio paths are relative POSIX paths beneath it, including the actual dataset mount. Validate every file before loading models or clients.

| Field | Contract |
| :--- | :--- |
| `id`, `text` | Unique segment ID; nonempty reference |
| `audio` | Relative file path; no escape or executable Kaldi recipe |
| `duration_s` | Finite, positive duration verified against decoded audio |
| `start_s` | `None`: whole file; explicit value: aligned crop |
| `recording_id` | Bootstrap block; may represent a speaker across clips |
| `language`, `audio_sha256` | Preserve supplied values; report missing hash coverage |

A recording collapse requires one actual file and known span ordering. Its duration covers the timeline, including gaps; separate clips cannot be combined as one recording.

| Preparation | Required input |
| :--- | :--- |
| Common Voice | Selected release; `train.tsv`, `dev.tsv`, `test.tsv`; actual clips mount |
| SLR53 | `utt_spk_text.tsv` and matching WAV files |
| MUCS | Kaldi `text`, `wav.scp`, optional `segments`/`utt2spk`; file paths only |
| Bengali-Loop | Published recording JSONL; speaker CSV uses `start_time`, `end_time`, `speaker_id` |

The four-row MUCS file is an [example fixture](../tests/fixtures/mucs-example.manifest.jsonl), not production test data. Recording transcripts are evaluation references, not aligned 28-second training targets.

## Source terms

[Catalog](../data/LICENSES.md) is generated from the registry. It records source metadata and local use tiers; it grants no additional rights. Hours refer to source records, not project training totals.

- Bengali-Loop: annotation/media terms unconfirmed; excluded until recorded.
- Private media proposal and OOD-Speech: excluded while permissions remain unresolved.
- Confirm the selected release, attribution and redistribution obligations before preparation.

## Model identity

| Label | Weights |
| :--- | :--- |
| Tugstugi medium, base | `bengaliAI/tugstugi_bengaliai-asr_whisper-medium` |
| Large-v3, zero-shot | `openai/whisper-large-v3` |
| Large-v3-turbo, zero-shot | `openai/whisper-large-v3-turbo` |
| Project fine-tune | Explicit trained checkpoint; currently unavailable |

The [Loop paper](https://arxiv.org/html/2602.14291v1) cites a separate **regional** Tugstugi checkpoint. Its published score is not a result from this project.

## Optional runtime

Core verification installs no models or provider SDKs. Install only the runtime used by your job; notebooks state bounded versions.

| Path | Dependencies |
| :--- | :--- |
| Training / HF inference | Transformers `>=4.46.3,<5`; Torch `>=2.6,<3`; accelerate; soundfile; librosa; NumPy |
| CT2 transcription | faster-whisper `>=1.2,<2`; CTranslate2 `>=4,<5`; Silero; soundfile; Torch; librosa; NumPy |
| Speaker labels | pyannote.audio `>=4,<5`; accepted model access conditions |
| Cloud | deepgram-sdk `>=7,<8` or google-cloud-speech `>=2,<3` |
| Demo | Gradio `>=5,<6` plus the selected transcription runtime |

```sh
uv sync --locked --extra audio
uv run --locked python -m shono.eval --report baselines --base-dir . --out-dir outputs
uv run --locked python -m shono.transcribe --help
uv run --locked python scripts/check_evidence.py --check
```

Run notebooks in order: `prepare_data` → `baselines` → `train_whisper_medium` → `predict`. GPU/model execution is separate from notebook syntax verification.

## Execution boundaries

- Training requires licensed train/dev inputs, declared held-out data and reviewed leakage. ID/audio overlap blocks execution; text collisions need exact-key decisions.
- Short-form input must fit its model window. Long-form adapters require whole recordings or an explicitly implemented crop protocol.
- Cloud runs default off; allowances default to 0. Reserve cumulative declared duration before requests. Account credit, retries and provider billing remain external.

Speaker attribution is dominant-speaker labeling of transcript chunks after intersecting speaker turns with detected speech. It does not establish word-level boundaries or identity.

## Scores and artifacts

| Domain | Normalization |
| :--- | :--- |
| Raw | Whitespace canonicalization only |
| Bengali `1.1.0` | NFC; encoding repair; punctuation/symbol spaces; digit canonicalization; case retained |
| Code-switch `1.0.0` | Bengali normalization then Unicode casefold; no transliteration matches |

WER/CER pool edits over reference units. Confidence intervals resample declared recording/speaker blocks, not individual segments. Fewer than 2 blocks cannot support these intervals. Source benchmark policies may differ.

Each report cell records its policy/version, input hashes and inference context. Missing manifests, predictions, permissions or unsupported protocols remain pending. Upstream model training data is unknown unless supplied; local leakage checks cannot establish its independence.

Predictions and reports share `outputs/`; checkpoints remain outside Git. Checkpoint resume selects the latest saved step with its required markers; interruption-safe completeness and training convergence require a real run.

## Primary references

[Whisper paper](https://arxiv.org/abs/2212.04356) · [Loop format and protocol](https://arxiv.org/html/2602.14291v1) · [SLR53](https://www.openslr.org/53/) · [MUCS](https://www.openslr.org/104/)

[Pyannote output contract](https://huggingface.co/pyannote/speaker-diarization-community-1) · [Deepgram 7 migration](https://raw.githubusercontent.com/deepgram/deepgram-python-sdk/main/docs/Migrating-v6-to-v7.md) · [Google input limits](https://docs.cloud.google.com/speech-to-text/docs/quotas) · [Transformers 4.46.3](https://huggingface.co/docs/transformers/v4.46.3/en/main_classes/trainer)
