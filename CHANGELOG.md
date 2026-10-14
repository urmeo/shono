# Changelog

## Unreleased

- Separate resume controls from training identity and reject empty normalized targets.
- Reject multichannel RTTM recordings, duplicate CSV headers, and inconsistent row widths.
- Use declared model kinds for report and model-card training audit requirements.
- Escape model-card table text containing pipes or line breaks.
- Filter decoder segments before merging text; retain segment scores and timings.
- Hash local audio during leakage audits and block incomplete hash coverage.
- Bind training loaders and resume to one Hub commit or hashed local checkpoint.
- Record the pinned Silero revision in inference and diarization outputs.
- Protect shared training manifests from report output collisions.
- Pass audio attention masks to Whisper generation and default the transcription
  CLI to CPU with int8 computation.
- Pin Silero code and load its audio through librosa.
- Correct training dependency ranges and check CPU training, evaluation,
  checkpoint resume, and model export with local random weights.

## 0.1.1

- Validate audio paths, time windows, training splits, and model inputs before
  loading models or calling providers.
- Correct provider adapters, cumulative allowance checks, speaker attribution,
  and report metadata.
- Retain Bengali normalization version 1.1.0 and code-switch normalization
  version 1.0.0. Add offline notebook checks and hand-computed scoring fixtures.
