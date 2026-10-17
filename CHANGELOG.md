# Changelog

## Unreleased

- Hash training audio. Reject copied windows, development/test audio overlap,
  incomplete leakage audits, and targets emptied by normalization or Whisper decoding.
- Load training and inference components from one Hub commit or hashed local
  checkpoint. Keep resume controls separate from training identity.
- Pin Silero, decode its audio with librosa, and record its revision in outputs.
  Pass attention masks to Whisper; default the transcription CLI to CPU and int8.
- Check decoder segments before merging. Preserve scores, timing, loaded model
  identity, and distinct overlapping text.
- Reject multichannel RTTM, duplicate CSV headers, inconsistent rows, and malformed
  quoting. Skip speaker-model loading for empty transcripts.
- Validate prediction hashes, report grids, bootstrap settings, and model audit
  requirements. Identify evaluated checkpoints and escape model-card table cells.
- Protect input files from output collisions and remove staged files after failed
  report writes. Reuse speaker-scoring intervals and remove duplicate notebook runs.
- Correct training dependency ranges. Check CPU training, evaluation, resume, and
  model export with local random weights.

## 0.1.1

- Validate audio paths, time windows, training splits, and model inputs before
  loading models or calling providers.
- Correct provider adapters, cumulative allowance checks, speaker attribution,
  and report metadata.
- Retain Bengali normalization version 1.1.0 and code-switch normalization
  version 1.0.0. Add offline notebook checks and hand-computed scoring fixtures.
