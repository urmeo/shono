# Dataset licenses

This table is generated from [`licenses.json`](licenses.json). Hours describe
the source records, not the amount used to train a project model. Source terms
and access requirements remain separate from the project's use status:

- `active`: permitted by the registry for training and evaluation.
- `eval-only`: permitted for evaluation, excluded from training.
- `excluded`: blocked until its source terms or permissions are resolved.

The repository stores metadata rather than audio. The redistribution column
records the source's audio terms; it does not grant additional rights.

To print the current table:

```sh
uv run --locked python -c "from shono.data import LicenseRegistry; print(LicenseRegistry.load('data/licenses.json').render_table())"
```

| Source | Hours | Domain | License | Redistribute audio? | Status |
|---|---|---|---|---|---|
| [Common Voice Bengali (cv-corpus-26.0, validated)](https://commonvoice.mozilla.org/) | 54.32 | read | CC0-1.0 | yes | active |
| [MUCS 2021 / OpenSLR SLR104 (Bengali-English code-switch)](https://www.openslr.org/104/) | 46.11 | code-switch | CC-BY-SA-4.0 | yes | active |
| [OpenSLR SLR53 (Bengali ASR, Google)](https://openslr.org/53/) | 215 | read | CC-BY-SA-4.0 | yes | active |
| [FLEURS Bengali (Google)](https://huggingface.co/datasets/google/fleurs) | 10 | read | CC-BY-4.0 | yes | eval-only |
| [Bengali-Loop (long-form + diarization benchmark)](https://arxiv.org/abs/2602.14291) | 158.6 | long-form | unconfirmed | no | excluded |
| [OOD-Speech / Bengali.AI (17-domain robustness test)](https://arxiv.org/abs/2305.09688) | 1178 | out-of-domain | unconfirmed | no | excluded |
| Shono private lecture/podcast eval set (YouTube-sourced) | unknown | long-form | source-specific; permission unconfirmed | no | excluded |

_Registry as of 2026-10-07._
