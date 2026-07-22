# Dataset licenses

**Generated from [`licenses.json`](licenses.json) — do not edit by hand.**
Regenerate:

```bash
uv run python -c "from shono.data import LicenseRegistry as R; \
print(R.load('data/licenses.json').render_table())"
```

Shono never commits or redistributes raw audio; `data/` holds manifests, checksums, and these license records only. `Redistribute audio?` reflects each source's own license, not Shono's practice.

| Source | Hours | Domain | License | Redistribute audio? | Status |
|---|---|---|---|---|---|
| [Bengali-Loop (long-form + diarization benchmark)](https://arxiv.org/abs/2602.14291) | 158.6 | long-form | CC-BY-4.0 | no | active |
| [Common Voice Bengali (cv-corpus-26.0, validated)](https://commonvoice.mozilla.org/) | 54.32 | read | CC0-1.0 | yes | active |
| [MUCS 2021 / OpenSLR SLR104 (Bengali-English code-switch)](https://www.openslr.org/104/) | 46.11 | code-switch | CC-BY-SA-4.0 | yes | active |
| [OpenSLR SLR53 (Bengali ASR, Google)](https://openslr.org/53/) | 215 | read | CC-BY-SA-4.0 | yes | active |
| [FLEURS Bengali (Google)](https://huggingface.co/datasets/google/fleurs) | 10 | read | CC-BY-4.0 | yes | eval-only |
| Shono private lecture/podcast eval set (YouTube-sourced) | — | long-form | mixed-source; fair-dealing-for-research; not redistributed | no | eval-only |
| [OOD-Speech / Bengali.AI (17-domain robustness test)](https://arxiv.org/abs/2305.09688) | 1178 | out-of-domain | unconfirmed | no | excluded |

_Registry as of 2026-07-23._
