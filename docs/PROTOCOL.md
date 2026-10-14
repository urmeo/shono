# Data and runtime protocol

## Inputs

Set `AUDIO_ROOT` to the directory containing the local datasets. Each manifest's
`audio` path is relative to that root and includes the dataset directory. For
example, with `AUDIO_ROOT=/kaggle/input`, a clip might be
`common-voice-bn/clips/example.mp3`.

Validate paths and time windows before loading a model or creating a provider
client.

| Field | Contract |
| :--- | :--- |
| `id` | Unique, nonempty segment ID within the manifest |
| `text` | Nonempty reference transcript |
| `audio` | Relative POSIX file path beneath `AUDIO_ROOT`; no path traversal or executable Kaldi commands |
| `duration_s` | Finite, positive duration checked against the file's audio metadata |
| `start_s` | `None` means the whole file; an explicit value, including zero, selects an aligned crop |
| `recording_id` | Block used for bootstrap resampling; may group one speaker across clips |
| `language` | Preserve the supplied language tag |
| `audio_sha256` | SHA256 of complete file bytes; computed during local leakage audits if absent |

To collapse segments into one recording reference, they must refer to the same
file and have a known time order. The resulting duration includes gaps between
segments. Separate clip files cannot be treated as one recording.

| Preparation | Required input |
| :--- | :--- |
| Common Voice | Selected release; `train.tsv`, `dev.tsv`, `test.tsv`; actual clips mount |
| SLR53 | `utt_spk_text.tsv` and matching WAV files |
| MUCS | Kaldi `text`, `wav.scp`, optional `segments`/`utt2spk`; file paths only |
| Bengali-Loop | Published recording JSONL; speaker CSV uses `start_time`, `end_time`, `speaker_id` |

The [four-row MUCS fixture](../tests/fixtures/mucs-example.manifest.jsonl) is
example data. Do not use it as a production test set. A transcript of a whole
recording can serve as an evaluation reference; training on shorter windows
requires text aligned to those windows.

## Source terms

The [source catalog](../data/LICENSES.md) is generated from
[`data/licenses.json`](../data/licenses.json). It records dataset metadata and
whether the project permits training, evaluation, or neither. Source hours are
not project training totals. The registry grants no additional rights.

- Bengali-Loop is excluded until its annotation and media terms are confirmed
  and recorded.
- The proposed private media set and OOD-Speech are excluded while permissions
  remain unresolved.
- Confirm the selected release, attribution requirements, and redistribution
  terms before preparing a dataset.

## Model identity

| Label | Weights |
| :--- | :--- |
| Tugstugi medium, base | `bengaliAI/tugstugi_bengaliai-asr_whisper-medium` |
| Large-v3, zero-shot | `openai/whisper-large-v3` |
| Large-v3-turbo, zero-shot | `openai/whisper-large-v3-turbo` |
| Project fine-tune | Explicit trained checkpoint; currently unavailable |

The [Loop paper](https://arxiv.org/html/2602.14291v1) cites a separate regional
Tugstugi checkpoint. Its published score does not measure this project's base
checkpoint or fine-tune.

Hub training requires `TrainConfig.model_revision` to be a full 40-character
commit SHA. Set `SHONO_MODEL_REVISION` to that SHA before running the training
notebook. Branches, tags, and shortened hashes are rejected. Processor,
configuration, and weights load from the same model ID and revision.

`training-inputs.json` records the base checkpoint identity. Local checkpoint
directories load offline and use SHA256 for every file. Changed Hub revisions
or local file contents block resume. Local files are checked again after loading.

## Optional runtime

`./verify` uses the core scoring and development dependencies. It does not
install model weights or provider SDKs. Install the dependencies for the workflow
you plan to run; the notebooks specify version ranges.

| Path | Dependencies |
| :--- | :--- |
| Training / Transformers inference | Transformers `>=4.52.4,<5`; PyTorch `>=2.6,<3`; accelerate `>=1.4,<2`; bitsandbytes for the default training optimizer; soundfile; librosa; NumPy |
| CT2 transcription | faster-whisper `>=1.2,<2`; CTranslate2 `>=4,<5`; Silero; soundfile; PyTorch; librosa; NumPy |
| Speaker labels | pyannote.audio `>=4,<5`; access to the selected gated model |
| Cloud | deepgram-sdk `>=7,<8` or google-cloud-speech `>=2,<3` |
| Demo | Gradio `>=5,<6`; CT2 transcription and speaker-label dependencies; `HF_TOKEN` with model access |

```sh
uv sync --locked --extra audio
uv run --locked --extra audio python -m shono.eval --report baselines
uv run --locked --extra audio python -m shono.transcribe --help
```

Run the notebooks in this order: `prepare_data` → `baselines` →
`train_whisper_medium` → `predict`. Set their repository, dataset, checkpoint,
and output paths for your machine. Review the `INSTALL_RUNTIME`, `RUN_MODELS`,
and other execution flags before enabling a run. Syntax checks do not execute
models, training, or provider calls.

For CPU transcription on Linux or Apple silicon, install the runtime in the
project environment:

```sh
uv sync --locked --extra audio
uv pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cpu
uv pip install 'faster-whisper>=1.2,<2' 'librosa>=0.10,<0.12'
```

Use the environment's Python for model commands. `uv sync` and `./verify` restore
the core environment, so repeat the runtime install after running them.

With an existing local CTranslate2 checkpoint, transcribe a recording with:

```sh
.venv/bin/python -m shono.transcribe /path/to/recording.wav \
  --model /path/to/ct2-checkpoint --json outputs/transcript.json
```

The CLI defaults to CPU and int8. For a configured CUDA environment, pass
`--device cuda --compute-type float16`. The checkpoint must contain model,
tokenizer, and preprocessing files. Conversion from a local Transformers
checkpoint also requires Transformers `>=4.52.4,<5`; inspect its options with
`.venv/bin/python -m shono.transcribe.convert --help`.

On a cold cache, Silero code and weights download on first use. The adapter pins
[Silero commit `1e261b0`](https://github.com/snakers4/silero-vad/commit/1e261b036686cd0017d500ee96acd1c4ba572a9d)
and decodes audio with librosa. Its repository, full revision, and settings are
recorded in CLI JSON, prediction metadata, and diarization reports. The CLI also
prints the revision to stderr. Runtime performance remains unmeasured.

The long-form pipeline checks each decoder segment's average log probability,
compression ratio, and no-speech probability before merging accepted text.
A segment is dropped if any configured threshold fails. These guards are
heuristics; they do not establish transcription accuracy.

`ChunkTranscriber.transcribe_chunk` returns a tuple of decoder segments, each
with recording-local bounds and segment-local word times. Empty output returns
an empty tuple. CLI `n_chunks` counts input audio windows; `dropped` counts
rejected decoder segments. Segments without word times retain coarse timing.

## Execution boundaries

- Training requires permitted train and development inputs, declared held-out
  data, and a leakage review. ID or audio overlap blocks execution. Matching
  normalized text requires a recorded decision for the exact text key.
- Short-form input must fit its model window. Long-form adapters require whole
  recordings unless a crop protocol is explicitly implemented.
- Cloud runs default to disabled and provider allowances default to zero. The
  guard reserves cumulative declared audio duration before requests. Account
  credits, retries, and provider billing must be checked separately.

With `audio_root`, leakage audits hash the local file bytes and verify supplied
digests. Without it, every segment must declare `audio_sha256`. Missing hashes
or partial check selections leave the audit `INCOMPLETE` and unapproved.
Training and scored fine-tuned reports require complete audits. Hash checks
detect byte-identical copies; they do not detect re-encoded or edited audio.
Set `HASH_AUDIO=True` when preparing manifests for reports without `audio_root`.

Speaker attribution first clips speaker turns to detected speech, then labels
each transcript chunk with its dominant speaker. These labels do not establish
word boundaries or identify the person speaking.

## Scores and artifacts

| Domain | Normalization |
| :--- | :--- |
| Raw | Whitespace canonicalization only |
| Bengali `1.1.0` | Unicode NFC; word-level encoding repair; punctuation and symbols replaced with spaces; zero-width marks removed; whitespace collapsed; Bengali digits mapped to ASCII; Latin case retained |
| Code-switch `1.0.0` | Bengali normalization followed by Unicode casefold; no cross-script transliteration matches |

WER and CER divide total edits by the total reference words or characters.
The same policy is applied to references and predictions. Confidence intervals
resample whole recording or speaker blocks and require at least two blocks.
Other benchmarks may use different normalization and scoring rules.

Each report cell records its scoring policy and version, input hashes, and
inference context. Cells remain pending when manifests, predictions, or source
permissions are missing, or when the inference protocol is unsupported. Local
leakage checks cover the supplied manifests. They cannot establish whether an
upstream model was trained on the evaluation data.

Report definitions are in [`configs/reports/`](../configs/reports/). Predictions
and reports go to `outputs/`; checkpoints stay outside Git. Resume selects the
latest checkpoint with valid saved-step metadata. Weight and optimizer
completeness are checked by the training runtime. The CPU integration check
covers checkpoint resume with local random weights. Recovery during GPU
training and convergence still require validation.

## Primary references

[Whisper paper](https://arxiv.org/abs/2212.04356) · [Loop format and protocol](https://arxiv.org/html/2602.14291v1) · [SLR53](https://www.openslr.org/53/) · [MUCS](https://www.openslr.org/104/)

[Pyannote output contract](https://huggingface.co/pyannote/speaker-diarization-community-1) · [Deepgram 7 migration](https://raw.githubusercontent.com/deepgram/deepgram-python-sdk/main/docs/Migrating-v6-to-v7.md) · [Google input limits](https://docs.cloud.google.com/speech-to-text/docs/quotas) · [Transformers 4.52.4 Trainer source](https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/trainer.py)
