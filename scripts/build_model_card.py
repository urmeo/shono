#!/usr/bin/env python
"""Generate MODEL_CARD.md from an evaluation report — numbers come from the report.

    python scripts/build_model_card.py --system "shono whisper-medium (fine-tuned)" \
        --model-id urmeo/shono-whisper-medium-bn --base-model <base> --repo-url <url>

Anything not yet measured renders as `—`; the card cannot state a number the frozen
harness did not produce.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shono.data import LicenseRegistry
from shono.demo.model_card import render_model_card

_LIMITATIONS = [
    "Trained and evaluated on Bangladeshi-register Bengali; Indian-register (bn-IN) "
    "coverage is only what the data incidentally provides.",
    "Long-form and diarization are tuned for lectures and podcasts; far-field capture "
    "and heavy background noise are out of scope.",
    "Code-switch scoring credits within-script normalization only, not cross-script "
    "transliteration equivalence.",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--report", default="reports/full.json")
    p.add_argument("--system", required=True, help="the system name in the report to feature")
    p.add_argument("--model-id", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--repo-url", required=True)
    p.add_argument("--license", default="MIT")
    p.add_argument("--licenses", default="data/licenses.json")
    p.add_argument("--out", default="MODEL_CARD.md")
    args = p.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    registry = LicenseRegistry.load(args.licenses)
    card = render_model_card(
        report,
        model_id=args.model_id,
        system=args.system,
        base_model=args.base_model,
        license_spdx=args.license,
        repo_url=args.repo_url,
        license_table=registry.render_table(),
        limitations=_LIMITATIONS,
    )
    Path(args.out).write_text(card, encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
