"""``python -m shono.eval --report <name>`` — regenerate an evaluation report.

The report is defined by a spec at ``reports/specs/<name>.json``; the CLI loads
the referenced manifests and any available predictions, scores each cell, and
writes ``reports/<name>.md`` and ``reports/<name>.json``. Cells whose predictions
have not been produced yet render ``—`` — running the command before the Kaggle
baseline runs exist yields an honest skeleton, never an invented number.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from shono.data.license import LicenseRegistry
from shono.eval.report import ReportSpec, build_report, write_report
from shono.provenance import RunContext


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m shono.eval",
        description="Regenerate a per-slice evaluation report from manifests + predictions.",
    )
    p.add_argument("--report", required=True, help="report name (spec at <specs-dir>/<name>.json)")
    p.add_argument("--spec", type=Path, help="explicit spec path (overrides --report lookup)")
    p.add_argument(
        "--base-dir",
        type=Path,
        default=Path.cwd(),
        help="root the spec's relative paths resolve against (default: cwd)",
    )
    p.add_argument("--specs-dir", type=Path, default=Path("reports/specs"))
    p.add_argument("--out-dir", type=Path, default=Path("reports"))
    p.add_argument(
        "--licenses",
        type=Path,
        default=Path("data/licenses.json"),
        help="license registry path (relative to --base-dir if not absolute)",
    )
    return p


def _resolve(base: Path, path: Path) -> Path:
    return path if path.is_absolute() else base / path


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    base = args.base_dir

    spec_path = args.spec or _resolve(base, args.specs_dir / f"{args.report}.json")
    if not spec_path.exists():
        print(
            f"error: no spec for report {args.report!r} at {spec_path}. "
            f"Create it under {args.specs_dir}/ (see reports/specs/ for the format).",
            file=sys.stderr,
        )
        return 2

    registry = LicenseRegistry.load(_resolve(base, args.licenses))
    spec = ReportSpec.from_json(spec_path)

    command = "python -m shono.eval " + " ".join(argv if argv is not None else sys.argv[1:])
    ctx = RunContext.capture(
        spec.seed,
        {"report": spec.name, "n_resamples": spec.n_resamples},
        command=command,
        repo=base,
    )

    try:
        report = build_report(spec, base, registry, command=command, run_context=ctx)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        print(f"error building report {spec.name!r}: {exc}", file=sys.stderr)
        return 1

    md_path, json_path = write_report(report, _resolve(base, args.out_dir))

    scored = sum(1 for c in report.cells if not c.is_pending)
    pending = sum(1 for c in report.cells if c.is_pending)
    print(f"wrote {md_path} and {json_path}")
    print(f"cells: {scored} scored, {pending} pending (—)")
    if pending:
        print("pending cells await predictions — run the baseline/eval notebooks to fill them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
