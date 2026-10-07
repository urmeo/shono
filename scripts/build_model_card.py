#!/usr/bin/env python
"""Generate a card with explicit training and weight-license declarations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from shono.data import LicenseRegistry
from shono.demo.model_card import render_model_card
from shono.output_paths import source_files, validate_output_destinations, write_text_atomic

_LIMITATIONS = [
    "GPU/model and cloud quality remain unmeasured until actual predictions are supplied.",
    "Speaker attribution follows transcription segments rather than word-level identities.",
    "Code-switch scoring does not credit cross-script transliteration equivalence.",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--report", default="outputs/full.json")
    parser.add_argument("--system", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-license", required=True)
    parser.add_argument(
        "--training-status", required=True, choices=("pending", "trained", "unknown")
    )
    parser.add_argument("--repo-url", required=True)
    parser.add_argument("--license", required=True, help="model-weight license, or unknown")
    parser.add_argument("--licenses", default="data/licenses.json")
    parser.add_argument("--out", default="outputs/MODEL_CARD.md")
    args = parser.parse_args(argv)
    try:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        registry = LicenseRegistry.load(args.licenses)
        card = render_model_card(
            report,
            model_id=args.model_id,
            system=args.system,
            base_model=args.base_model,
            base_model_license=args.base_license,
            training_status=args.training_status,
            license_spdx=args.license,
            repo_url=args.repo_url,
            license_table=registry.render_table(),
            limitations=_LIMITATIONS,
        )
        output = validate_output_destinations(
            (args.out,), protected_paths=(args.report, args.licenses, *source_files(Path.cwd()))
        )[0]
        output.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(output, card)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        print(f"error generating model card: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
