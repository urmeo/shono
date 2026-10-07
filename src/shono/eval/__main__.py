"""Generate scored or pending reports from explicit input specs."""

from __future__ import annotations

import argparse
import re
import shlex
import sys
from dataclasses import replace
from pathlib import Path

from shono.data.license import LicenseRegistry
from shono.eval.report import ReportSpec, build_report, write_report
from shono.provenance import RunContext


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m shono.eval",
        allow_abbrev=False,
        description="Generate per-slice reports from manifests and predictions.",
    )
    parser.add_argument("--report", required=True, help="report name")
    parser.add_argument("--spec", type=Path, help="explicit spec path, relative to --base-dir")
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--specs-dir", type=Path, default=Path("reports/specs"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--licenses", type=Path, default=Path("data/licenses.json"))
    return parser


def _resolve(base: Path, path: Path) -> Path:
    return path if path.is_absolute() else base / path


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.report):
            raise ValueError("report name must contain letters, digits, underscores or hyphens")
        base = args.base_dir
        spec_path = _resolve(base, args.spec or args.specs_dir / f"{args.report}.json")
        if not spec_path.is_file():
            print(f"error: no spec for report {args.report!r} at {spec_path}", file=sys.stderr)
            return 2
        license_path = _resolve(base, args.licenses)
        registry = LicenseRegistry.load(license_path)
        spec = ReportSpec.from_json(spec_path)
        command = "python -m shono.eval " + shlex.join(argv if argv is not None else sys.argv[1:])
        context = RunContext.capture(
            spec.seed,
            {"report": spec.name, "n_resamples": spec.n_resamples},
            command=command,
            repo=base,
        )
        report = build_report(spec, base, registry, run_context=context)
        report = replace(report, input_paths=(*report.input_paths, str(license_path)))
        md_path, json_path = write_report(report, _resolve(base, args.out_dir))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"error generating report {args.report!r}: {exc}", file=sys.stderr)
        return 1
    scored = sum(not cell.is_pending for cell in report.cells)
    pending = len(report.cells) - scored
    print(f"wrote {md_path} and {json_path}")
    print(f"cells: {scored} scored, {pending} pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
