# Shono (শোনো)

Shono is a Python toolkit for transcribing local Bengali audio and comparing transcripts with reference text. It brings dataset preparation, Whisper inference, optional fine-tuning, and error reporting into one codebase.

**Current status:** working CPU prototype. **Readiness score: 7/10.** Four reproduced correctness issues remain open, so the project is not ready for final submission.

## What the project does

| Part | What it does |
| :--- | :--- |
| Prepare data | Reads locally supplied Common Voice, FLEURS, OpenSLR SLR53, MUCS/Kaldi, and Bengali Loop annotations into a common manifest format. |
| Check inputs | Validates audio paths, declared durations, source permissions, and data splits. Checks train/evaluation overlap through IDs, normalized text, and exact file hashes. |
| Transcribe audio | Uses Silero to find speech, plans bounded audio windows, and decodes them with a local Whisper checkpoint through faster-whisper. |
| Check segments | Applies configured confidence and repetition rules before merging decoder segments into a transcript. |
| Measure errors | Compares predictions with reference text and reports word error rate (WER) and character error rate (CER), before and after normalization. |
| Fine-tune a model | Provides a Python API for training Whisper with PyTorch and Transformers, development evaluation, checkpoint saving, and resume controls. |
| Record a run | Writes reports with settings, package versions, available Git state, model metadata, and relevant input hashes. |

Model weights, audio datasets, and source-registry records are supplied separately.

## How it works

**Transcription**

Local audio → speech detection → bounded Whisper windows → segment checks → merged transcript.

The default chunk limit is 28 seconds. Windows include configurable padding and must stay within Whisper's 30-second limit.

**Evaluation**

Transcript + reference → declared text normalizer → WER/CER → Markdown and JSON reports.

Scoring pools errors across segments. Bootstrap intervals resample recording or speaker blocks when at least two blocks are available.

Bengali normalization repairs supported encoding problems, replaces punctuation and symbols with spaces, collapses whitespace, and maps Bengali digits to ASCII. It preserves Latin case. Code-switch scoring also folds case; it does not equate Bengali and Latin spellings through transliteration.

The baseline report scores supplied predictions for the Bengali base checkpoint alone. Long-form, code-switch, and full reports compare that base with supplied fine-tuned predictions. Fine-tuned scored cells require the declared training-data audit. Missing inputs remain pending rather than receiving invented scores.

**Optional training**

Local train/development manifests → source and overlap checks → examples within the configured duration bounds → Whisper fine-tuning → saved checkpoint and evaluation.

The training setup records its base checkpoint and input identity. Remote training loaders use the same immutable model revision. The inference CLI expects a local CTranslate2 checkpoint; first use also needs the pinned Silero code and weights.

## Core tools

| Role | Tools |
| :--- | :--- |
| Training and model loading | Python 3.11 · PyTorch · Transformers |
| Speech inference | Whisper · faster-whisper · CTranslate2 · Silero VAD |
| Audio processing | NumPy · librosa · soundfile |
| Text evaluation | jiwer · bnunicodenormalizer |

## What has been verified

| Check | Result |
| :--- | :--- |
| Updated runtime test suite | 732 passed, including real CPU training, evaluation, saving, resume, and model reload. |
| Lightweight test suite | 731 passed; one training integration test skipped because PyTorch is absent in that environment. |
| Real CPU training integration | Passed separately in the full runtime: train to step 2, evaluate and save, resume checkpoint 1 to step 2, then reload the model and processor. |
| Model conversion | Whisper-tiny converted from Transformers format to int8 CTranslate2. All five exported file hashes matched the checkpoint used for the successful inference check. |
| Speech and silence | The English sample produced “This is a local speech transcription test.” with one chunk and no dropped segments. One second of silence produced no text and zero chunks. |
| Evaluation CLI | An exact-match synthetic case scored five segments in two recording blocks and wrote Markdown and JSON reports. |
| Error handling | Missing audio returned exit code 2 with a clear message and no traceback. |
| Code checks | Ruff lint passed. All 43 Python files passed the format check. |
| Scorer and chart data | All six WER and five CER expectations matched. The three result charts agreed with their recorded inputs and computed values. |

The bootstrap check reproduced 243, 513, and 244 draws at WER 0, 1/6, and 1/3 with 1,000 resamples and seed 0. Its point estimate and 95% interval describe the synthetic fixture.

Both inference outputs recorded Silero revision `1e261b036686cd0017d500ee96acd1c4ba572a9d`.

**What these checks establish:** conversion, inference, scoring, and a small training/resume workflow execute successfully on the tested CPU setup.

**What they leave unmeasured:** Bengali recognition accuracy, useful training improvement, GPU behavior, long-recording alignment quality, and controlled throughput. The English inference sample and the small randomly initialized training model do not establish those results. Most other runtime tests use controlled substitutes for model components.

Result data: [WER/CER checks](outputs/scorer-check.csv) · [Figure inputs and values](outputs/visual-checks.csv).

## Security check

No real API keys, tokens, passwords, private keys, or sensitive environment files were found in the checked repository files and Git blobs. Credential-shaped strings in the tests are deliberate invalid examples.

Metadata export now redacts alternate credential-key spellings, supported authorization headers, URL credentials, and nested commands. Prediction exports also sanitize run context again after mutation.

The updated PyTorch and Transformers pins address known [checkpoint-loading](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p) and [processor-saving](https://github.com/advisories/GHSA-xrqw-3rrv-vx5w) vulnerabilities. An OSV query on 2026-10-24 returned no advisories for the 70 installed package versions.

That result has a limit: Accelerate 1.15.0 still contains the checkpoint-index loader behavior described in [CVE-2026-69112](https://github.com/advisories/GHSA-4j2p-28q2-5m79). Shono's configured loading, training, resume, and conversion paths do not invoke those loaders. This is a source-level reachability finding, not an upstream fix.

## Submission readiness

**Verdict: not ready for final submission.**

The 7/10 score is an engineering-readiness judgment. The package structure is clear and the normal workflows run. The remaining gaps affect text integrity, checkpoint selection, and audio completeness.

All four P2 correctness issues below remain open in the source snapshot recorded at the end of this review.

### 1. Unicode text can break file loading

**Observed:** U+0085, U+2028, and U+2029 inside valid transcript strings are treated as record boundaries by `splitlines()`. A prediction containing an actual U+2028 character writes successfully but fails when read back.

The data probes reproduced 15 failures across manifest file, text, snapshot, Loop, and Kaldi loading. Ordinary spaces passed all five controls. Prediction-file round trips also failed for the three characters.

**Why it matters:** valid text can be rejected during import or after serialization.

**Fix:** read physical file lines, or split JSONL on LF. Preserve CRLF handling and physical error-line numbers. Kaldi parsing must still accept an ordinary final newline without creating an empty record.

Sources: [manifest reader](src/shono/data/manifest.py#L179) · [Loop importer](src/shono/data/build.py#L323) · [Kaldi reader](src/shono/data/build.py#L363) · [prediction reader](src/shono/eval/report.py#L88).

### 2. Resume can select a checkpoint from an earlier run

**Observed:** retain an earlier `checkpoint-5`, start fresh with `resume=False`, create the new run's `checkpoint-1`, then enable resume. The next invocation selects the earlier `checkpoint-5`.

The probe exercised the training orchestration with controlled trainer components.

**Why it matters:** a restart can silently restore the wrong run's progress.

**Fix:** isolate checkpoint generations, or require a separate output directory when old checkpoints remain. Keep the operational resume flag outside the scientific training identity.

Sources: [training orchestration](src/shono/train/trainer.py#L209) · [checkpoint selection](src/shono/train/checkpoint.py#L41).

### 3. Inference can accept incomplete audio

**Observed:** a WAV containing one second of samples but declaring four seconds in its header reports a four-second duration. Actual decoding returns one second. The short-form path accepted it with a controlled model; the training loader correctly rejected the mismatch.

The Silero and CTranslate2 input paths have the same missing decoded-length check.

**Why it matters:** inference can omit missing samples while duration and real-time-factor metadata use the longer header duration.

**Fix:** verify that decoded samples cover the requested window before VAD or model inference. Reuse the training loader's tolerance and test whole recordings and cropped windows.

Sources: [short-form decoder](src/shono/transcribe/whisper.py#L104) · [Silero input](src/shono/transcribe/pipeline.py#L196) · [CTranslate2 input](src/shono/transcribe/pipeline.py#L284) · [training check](src/shono/train/audio.py#L30).

### 4. Tokenization can change training references

**Observed:** with the real cached Whisper tokenizer, `hello <|endoftext|> world`, `hello <|en|> world`, and `hello <|0.00|> world` all passed label validation. Their decoded references became `hello world` after whitespace normalization.

Against the original targets under the declared normalizer, the word error rates are 1/3, 1/3, and 1/2. Scoring against the changed decoded reference would instead report zero for `hello world`.

**Why it matters:** embedded control tokens can alter both the training target and the reference used for evaluation.

**Fix:** reject embedded reserved-token syntax or require equivalent normalized original and decoded targets before loading model weights. Preserve the supported timestamp-label path.

Sources: [label validation](src/shono/train/audio.py#L120) · [metric references](src/shono/train/trainer.py#L109).

## Five next steps

1. Fix physical record boundaries and add Unicode round-trip regressions.
2. Isolate fresh runs from previous checkpoint generations and test interruption followed by resume.
3. Apply decoded-audio completeness checks consistently across inference paths.
4. Validate that tokenization preserves the meaning of training targets.
5. Escape model identifiers and inference descriptions in published Markdown.

Rerun the suite, the new regressions, and the real CPU integration after these changes.

## Evidence needed for a Bengali ASR submission

The three checked-in images are programmatic charts of synthetic test outputs. They verify scorer and bootstrap behavior.

A submission presenting Bengali recognition results needs a permitted held-out evaluation with:

- Recording count and audio duration.
- Data splits and leakage-audit results.
- Exact checkpoint identity and normalization policy.
- Hardware and evaluation settings.
- Measured WER/CER with recording-block confidence intervals.

The reviewed repository includes no project-trained Bengali checkpoint or held-out Bengali benchmark.

<details>
<summary>Additional technical findings</summary>

- **P3 — Markdown metadata:** model identifiers containing backticks or newlines can break report lines. Escape these fields and use a safe code-span delimiter. [Source](src/shono/eval/report.py#L493).
- **Numeric API consistency:** `Fraction(1, 2)` passes duration/hour validation but later fails JSON serialization or license-table formatting. The annotated API expects floats and normal file adapters produce native numbers. Store the validated float or restrict accepted types.

</details>

<details>
<summary>Review record and test conditions</summary>

Date: 2026-10-24 · Source snapshot SHA-256: `9df0dc36f4636988d3b7d05afa7e86e6e5e0e2c02d17634a28f9ba9cdbde0612`.

The updated runtime used Python 3.11.15 on macOS 27.0.1, arm64, with PyTorch 2.13.0, Transformers 5.10.1, and Accelerate 1.15.0. All ten pinned requirements matched a fresh isolated installation. Compatibility checks passed for all 70 installed packages.

Runtime checks used writable workspace temporary directories and cached model files. Packages were installed into a fresh environment; model and VAD downloads were not repeated. Conversion, speech, silence, and CPU training/resume were rerun with the updated packages.

The small CPU training integration disables timestamp supervision and gradient checkpointing. It does not validate those modes, full-dataset convergence, or GPU fine-tuning. No controlled throughput benchmark was performed.

The snapshot digest hashes each sorted tracked path, a NUL byte, its file bytes, and a NUL byte. This review file is excluded to avoid a circular identity.

Every file in the recorded source snapshot was read. External datasets, downloaded model internals, and dependency source trees are outside this source review.

| Review area | Files | Lines read |
| :--- | ---: | ---: |
| Data, checkpoints, shared validation, data tests, and LICENSE | 11 | 2,820 |
| Evaluation, protocol, provenance, output guards, evidence script, support inputs, evaluation tests, and result files | 20 | 4,302 |
| Training, transcription, and runtime tests | 18 | 4,767 |
| README and requirements | 2 | 110 |
| **Total** | **51** | **11,999** |

</details>
