"""Verify hand-computed scoring fixtures; these are not model benchmarks."""

import argparse
import csv
import io
import math
import runpy
import sys
from html import escape
from pathlib import Path

from shono.eval.score import cer, wer
from shono.output_paths import source_files, validate_output_destinations, write_text_outputs

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/support.py"
OUTPUT = ROOT / "outputs"


def artifacts() -> dict[str, str]:
    cases = runpy.run_path(str(FIXTURE))["SCORE_CASES"]
    rows = []
    wer_rows = []
    for case in cases:
        for metric, compute in (("wer", wer), ("cer", cer)):
            expected = case[f"expected_{metric}"]
            if expected is None:
                continue
            actual = compute(case["references"], case["hypotheses"])
            if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"{case['name']} {metric}: expected {expected}, got {actual}")
            rows.append((case["name"], metric, expected, actual))
            if metric == "wer":
                wer_rows.append((case["label"], expected, actual))
    wer_count = len(wer_rows)
    cer_count = len(rows) - wer_count
    if not wer_count or not cer_count:
        raise ValueError("scoring fixtures must include WER and CER expectations")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("case", "metric", "expected", "actual"))
    writer.writerows(rows)
    height = 298 + 62 * wer_count
    axis_max = max(0.1, math.ceil(max(row[2] for row in wer_rows) * 10) / 10)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" '
        f'viewBox="0 0 1200 {height}" role="img" aria-labelledby="title description">',
        '<title id="title">Word error rate checks</title>',
        '<desc id="description">Hand-computed text fixtures. '
        f"All {wer_count} word-error and {cer_count} character-error checks match. "
        "No model accuracy is measured.</desc>",
        f'<rect width="1200" height="{height}" fill="#f7fafc"/>',
        '<g font-family="Arial, sans-serif" fill="#172b42">',
        '<text x="46" y="59" font-size="32" font-weight="700">Word error rate checks</text>',
        '<text x="46" y="94" font-size="21">Text fixtures only; no model predictions</text>',
        f'<text x="46" y="135" font-size="22" fill="#146650">'
        f"{wer_count}/{wer_count} WER · {cer_count}/{cer_count} CER checks passed</text>",
    ]
    for index, (label, expected, actual) in enumerate(wer_rows):
        y = 190 + 62 * index
        x = 365
        width = actual / axis_max * 650
        marker = x + expected / axis_max * 650
        svg += [
            f'<text x="46" y="{y + 24}" font-size="21">{escape(label)}</text>',
            f'<rect x="{x}" y="{y}" width="650" height="32" rx="4" fill="#e5edf4"/>',
            f'<rect x="{x}" y="{y}" width="{width:.3f}" height="32" rx="4" fill="#2873ab"/>',
            f'<path d="M{marker:.3f},{y - 4}v40" stroke="#172b42" stroke-width="3"/>',
            f'<text x="1044" y="{y + 24}" font-size="22">{actual:.1%}</text>',
        ]
    svg += [
        f'<text x="365" y="{height - 80}" font-size="18">0%</text>',
        f'<text x="1015" y="{height - 80}" font-size="18" text-anchor="end">{axis_max:.0%}</text>',
        f'<text x="46" y="{height - 33}" font-size="19">Bars: computed · ticks: expected</text>',
        "</g></svg>\n",
    ]
    return {
        "scorer-check.csv": stream.getvalue(),
        "scorer-check.svg": "\n".join(svg),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true", help="read-only verification, default")
    parser.add_argument("--write", action="store_true", help="write the scoring CSV and chart")
    args = parser.parse_args()
    if args.check and args.write:
        parser.error("--check and --write are mutually exclusive")
    expected = artifacts()
    if args.write:
        destinations = validate_output_destinations(
            (OUTPUT / name for name in expected),
            protected_paths=source_files(ROOT),
            output_root=OUTPUT,
        )
        OUTPUT.mkdir(exist_ok=True)
        write_text_outputs(dict(zip(destinations, expected.values(), strict=True)))
    else:
        for name, text in expected.items():
            if (OUTPUT / name).read_text(encoding="utf-8") != text:
                raise ValueError(f"stale or edited artifact: {name}")
    print("Scorer fixtures verified; model accuracy remains unmeasured")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
