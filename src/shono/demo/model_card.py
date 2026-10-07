"""Render reported scores with explicit training and weight-license metadata."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string; use 'unknown' when unresolved")
    return value


def _wer_str(cell: Mapping | None) -> str:
    if not cell or cell.get("status") == "pending":
        return "pending"
    if cell.get("status") != "scored" or not isinstance(cell.get("wer_norm"), Mapping):
        raise ValueError("invalid scored model-card cell")
    metric = cell["wer_norm"]
    for key in ("point", "lo", "hi"):
        value = metric.get(key)
        if key != "point" and value is None:
            continue
        try:
            finite = math.isfinite(value) if isinstance(value, (int, float)) else False
        except OverflowError:
            finite = False
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not finite
            or value < 0
        ):
            raise ValueError("WER values must be finite nonnegative numbers")
    if (metric.get("lo") is None) != (metric.get("hi") is None):
        raise ValueError("WER interval needs both bounds or neither")
    if metric.get("lo") is None:
        return f"{metric['point']:.1%}"
    if metric["lo"] > metric["hi"]:
        raise ValueError("WER interval bounds are reversed")
    return f"{metric['point']:.1%} [{metric['lo']:.1%}, {metric['hi']:.1%}]"


def render_model_card(
    report: Mapping,
    *,
    model_id: str,
    system: str,
    base_model: str,
    license_spdx: str,
    base_model_license: str,
    training_status: str,
    repo_url: str,
    license_table: str = "",
    limitations: Sequence[str] = (),
) -> str:
    """Quote per-cell WER and policy; report scores do not establish training status."""
    for name, value in (
        ("model_id", model_id),
        ("system", system),
        ("base_model", base_model),
        ("model-weight license", license_spdx),
        ("base-weight license", base_model_license),
        ("repo_url", repo_url),
    ):
        _text(value, name)
    if not isinstance(training_status, str) or training_status not in {
        "pending",
        "trained",
        "unknown",
    }:
        raise ValueError("training_status must be pending, trained or unknown")
    if (
        not isinstance(report, Mapping)
        or not isinstance(report.get("slices"), list)
        or not isinstance(report.get("cells"), list)
    ):
        raise ValueError("report must contain slices and cells arrays")
    slices = report["slices"]
    if any(not isinstance(name, str) or not name for name in slices) or len(set(slices)) != len(
        slices
    ):
        raise ValueError("report slice names must be unique strings")
    cells = {}
    for cell in report["cells"]:
        if (
            not isinstance(cell, Mapping)
            or not isinstance(cell.get("system"), str)
            or cell.get("slice") not in slices
        ):
            raise ValueError("invalid model-card report cell")
        key = cell["system"], cell["slice"]
        if key in cells:
            raise ValueError("duplicate model-card report cell")
        cells[key] = cell
    if not any(name == system for name, _ in cells):
        raise ValueError(f"system {system!r} is absent from report")
    if not isinstance(limitations, (tuple, list)) or any(
        not isinstance(item, str) for item in limitations
    ):
        raise ValueError("limitations must be a sequence of strings")
    lines = [
        "---",
        f"license: {json.dumps(license_spdx)}",
        "language: bn",
        "pipeline_tag: automatic-speech-recognition",
        f"base_model: {json.dumps(base_model)}",
        "---",
        "",
        f"# {model_id}",
        "",
        f"System: `{system}`. Training status: **{training_status}**.",
        f"Base weights: `{base_model}`. Base-weight license: **{base_model_license}**.",
        f"Model-weight license: **{license_spdx}**. Code license: MIT.",
        "",
        "## Results",
        "",
        "WER is quoted from the supplied report. Each row states its scoring policy; "
        "pending has no measured score.",
        "",
        "| Slice | WER (normalized, 95% CI when available) | Policy |",
        "|---|---|---|",
    ]
    for name in slices:
        cell = cells.get((system, name))
        if cell is None:
            raise ValueError(f"missing model-card cell for {system!r}/{name!r}")
        if (
            cell.get("status") == "scored"
            and "fine-tuned" in system
            and cell.get("leakage_audit", {}).get("status") != "reviewed"
        ):
            raise ValueError("scored fine-tuned cards require a reviewed training audit")
        policy = cell.get("normalization_policy", "unknown")
        version = cell.get("normalizer_version", "unknown")
        lines.append(f"| {name} | {_wer_str(cell)} | {policy} v{version} |")
    lines += ["", f"Protocol and inputs: [Shono]({repo_url})."]
    if limitations:
        lines += ["", "## Limits", "", *[f"- {item}" for item in limitations]]
    if license_table:
        lines += ["", "## Declared data sources", "", license_table]
    lines += [
        "",
        "## License",
        "",
        "Repository code is MIT. Model weights, base weights and data "
        "have separate stated licenses.",
        "",
    ]
    return "\n".join(lines)
