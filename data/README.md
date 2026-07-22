# Data

Dataset **manifests** and per-source **license records** only. Audio and
checkpoints never enter git — see [`.gitignore`](../.gitignore). A manifest
describes a slice of audio (ids, durations, reference text, recording grouping)
without shipping the audio itself.

## Files

- **[`licenses.json`](licenses.json)** — the license registry, the single
  authority for the dataset license floor. Every source a manifest references
  must be registered here first. Loaded by `shono.data.LicenseRegistry`.
- **[`LICENSES.md`](LICENSES.md)** — the human-readable license table,
  *generated* from `licenses.json` (regeneration command inside the file).
- `*.manifest.jsonl` — dataset slices in header-line JSON Lines (see below).
  Real manifests are generated at data-prep time on Kaggle, where the audio
  lives; they are not committed when large.

## Manifest format

Header-line JSON Lines: the first line carries split-level metadata, every
following line is one segment.

```jsonl
{"manifest": {"name": "cv-bn-test", "source": "common_voice_bn", "split": "test", "domain": "read", "version": "cv-corpus-26.0", "language": "bn"}}
{"id": "...", "audio": "clip.wav", "text": "…", "duration_s": 4.2, "recording_id": "…", "start_s": 0.0, "speaker": "S1"}
```

`recording_id` is the bootstrap block a segment belongs to — the recording (or
speaker) it was cut from — and is what the blockwise-bootstrap CI resamples.
Load and validate with:

```python
from shono.data import Manifest, LicenseRegistry

reg = LicenseRegistry.load("data/licenses.json")
m = Manifest.from_jsonl("data/cv-bn-test.manifest.jsonl")
m.validate_against(reg)   # source registered? not excluded? eval-only not used as train?
```

`status` in the registry encodes the floor: **active** (train + eval),
**eval-only** (eval only — e.g. the private lecture/podcast set, never
training), **excluded** (neither, until its blocker clears — e.g. OOD-Speech's
unconfirmed license).
