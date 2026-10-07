"""Verify hand-computed scoring fixtures; these are not model benchmarks."""

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from html import escape
from pathlib import Path

from shono.eval import cer, wer
from shono.output_paths import source_files, validate_output_destinations, write_text_atomic

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/wer_pairs.json"
OUTPUT = ROOT / "outputs"


def artifacts() -> dict[str, str]:
    raw = FIXTURE.read_bytes()
    cases = json.loads(raw)["cases"]
    rows = []
    for case in cases:
        for metric, compute in (("wer", wer), ("cer", cer)):
            expected = case[f"expected_{metric}"]
            if expected is None:
                continue
            actual = compute(case["references"], case["hypotheses"])
            if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"{case['name']} {metric}: expected {expected}, got {actual}")
            rows.append((case["name"], metric, expected, actual))
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("case", "metric", "expected", "actual"))
    writer.writerows(rows)
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="670" '
        'viewBox="0 0 1200 670" role="img" aria-labelledby="title description">',
        '<title id="title">Scorer verification</title>',
        '<desc id="description">Hand-computed text fixtures. All six word-error '
        "and five character-error checks match. No model accuracy is measured.</desc>",
        '<rect width="1200" height="670" fill="#f7fafc"/>',
        '<g font-family="Arial, sans-serif" fill="#172b42">',
        '<text x="46" y="59" font-size="32" font-weight="700">Scorer verification</text>',
        '<text x="46" y="94" font-size="21">Hand-computed fixtures; no model predictions</text>',
        f'<text x="46" y="135" font-size="22" fill="#146650">'
        f"{sum(r[1] == 'wer' for r in rows)}/6 WER · "
        f"{sum(r[1] == 'cer' for r in rows)}/5 CER checks passed</text>",
    ]
    wer_rows = [row for row in rows if row[1] == "wer"]
    labels = [
        "Identical",
        "1 word substituted",
        "1 word deleted",
        "Identical mixed script",
        "Pooled: equal lengths",
        "Pooled: unequal lengths",
    ]
    for index, ((_, _, expected, actual), label) in enumerate(zip(wer_rows, labels, strict=True)):
        y = 190 + 62 * index
        x = 365
        width = actual / 0.4 * 650
        marker = x + expected / 0.4 * 650
        svg += [
            f'<text x="46" y="{y + 24}" font-size="21">{escape(label)}</text>',
            f'<rect x="{x}" y="{y}" width="650" height="32" rx="4" fill="#e5edf4"/>',
            f'<rect x="{x}" y="{y}" width="{width:.3f}" height="32" rx="4" fill="#2873ab"/>',
            f'<path d="M{marker:.3f},{y - 4}v40" stroke="#172b42" stroke-width="3"/>',
            f'<text x="1044" y="{y + 24}" font-size="22">{actual:.1%}</text>',
        ]
    svg += [
        '<text x="365" y="590" font-size="18">0%</text>',
        '<text x="980" y="590" font-size="18">40%</text>',
        '<text x="46" y="637" font-size="19">'
        "Blue: measured WER · marker: hand-computed expectation</text>",
        "</g></svg>\n",
    ]
    metadata = (
        "[verification]\n"
        'kind = "hand-computed text fixtures; not model evaluation"\n'
        'raw_policy = "whitespace canonicalization only"\n'
        "absolute_tolerance = 1e-12\n"
        f"checks = {len(rows)}\n"
        f'fixture_sha256 = "{hashlib.sha256(raw).hexdigest()}"\n'
    )
    return {
        "scorer-check.csv": stream.getvalue(),
        "scorer-check.svg": "\n".join(svg),
        "scorer-check.toml": metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true", help="read-only verification, default")
    parser.add_argument(
        "--write", action="store_true", help="write only the three scoring artifacts"
    )
    args = parser.parse_args()
    if args.check and args.write:
        parser.error("--check and --write are mutually exclusive")
    expected = artifacts()
    if args.write:
        validate_output_destinations(
            (OUTPUT / name for name in expected),
            protected_paths=source_files(ROOT),
            output_root=OUTPUT,
        )
        OUTPUT.mkdir(exist_ok=True)
        for name, text in expected.items():
            write_text_atomic(OUTPUT / name, text)
    else:
        for name, text in expected.items():
            if (OUTPUT / name).read_text(encoding="utf-8") != text:
                raise ValueError(f"stale or edited artifact: {name}")
    print("6 WER and 5 CER fixtures verified; model accuracy remains unmeasured")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
