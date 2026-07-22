# Changelog

All notable changes to Shono are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning: [SemVer](https://semver.org/).

## [0.1.0] — 2026-07-16

### Added
- Evaluation harness — the foundation everything else must pass through:
  - Frozen Bengali normalization pipeline (`shono.eval.normalize`, normalizer v1.1.0): Unicode NFC → bnunicodenormalizer word-by-word (`allow_english`) → punctuation/symbols replaced with spaces → whitespace collapse → Bengali-to-ASCII digit canonicalization. Matras provably survive; Latin casing preserved.
  - Corpus WER + CER scoring (`shono.eval.score`) with an explicit raw contract (whitespace-canonicalization only) and normalized scoring applied to both sides.
  - Blockwise bootstrap 95% confidence intervals (`shono.eval.ci`), nearest-rank percentiles, deterministic seeding.
  - Mutation-pinning test suite (41 tests): removing the normalizer repair step, adding case folding, or switching to segment-wise resampling each fails a named test.
  - `./verify` — one command: dependency sync, lint, full test suite.
