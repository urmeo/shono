"""Render reported scores with explicit training and weight-license metadata."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from shono.eval.model import validate_scored_model
from shono.markdown import escape_cell
from shono.transcribe._validation import finite_real


def _validate_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string; use 'unknown' when unresolved")


def _format_wer(cell: Mapping) -> str:
    if cell.get("status") == "pending":
        return "pending"
    if cell.get("status") != "scored" or not isinstance(cell.get("wer_norm"), Mapping):
        raise ValueError("invalid scored model-card cell")
    metric = dict(cell["wer_norm"])
    for key in ("point", "lo", "hi"):
        value = metric.get(key)
        if key != "point" and value is None:
            continue
        metric[key] = finite_real(value, f"WER {key}", minimum=0)
        finite_real(metric[key] * 100, f"WER {key} percentage", minimum=0)
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
    """Render reported WER, scoring policies, and supplied training and license metadata."""
    for name, value in (
        ("model_id", model_id),
        ("system", system),
        ("base_model", base_model),
        ("model-weight license", license_spdx),
        ("base-weight license", base_model_license),
        ("repo_url", repo_url),
    ):
        _validate_text(value, name)
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
    checkpoints = set()
    for (cell_system, _slice), cell in cells.items():
        if cell_system == system and cell.get("status") == "scored":
            checkpoint = cell.get("model_id", "unknown")
            _validate_text(checkpoint, "evaluated checkpoint")
            if checkpoint != "unknown":
                checkpoints.add(checkpoint)
    if len(checkpoints) > 1:
        raise ValueError("model-card results refer to conflicting evaluated checkpoints")
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
        "Word error rate (WER; lower is better) comes from the supplied report. "
        "Each row states its scoring policy. Pending rows have no measured score.",
        "",
        "| Slice | WER (normalized, 95% CI when available) | Policy | Evaluated checkpoint |",
        "|---|---|---|---|",
    ]
    for name in slices:
        cell = cells.get((system, name))
        if cell is None:
            raise ValueError(f"missing model-card cell for {system!r}/{name!r}")
        if cell.get("status") == "scored":
            validate_scored_model(cell.get("model_kind", "unknown"), cell.get("leakage_audit"))
        policy = cell.get("normalization_policy", "unknown")
        version = cell.get("normalizer_version", "unknown")
        checkpoint = (
            cell.get("model_id", "unknown") if cell.get("status") == "scored" else "pending"
        )
        row = (name, _format_wer(cell), f"{policy} v{version}", checkpoint)
        lines.append("| " + " | ".join(escape_cell(value) for value in row) + " |")
    lines += ["", f"Protocol and inputs: [Shono]({repo_url})."]
    if limitations:
        lines += ["", "## Limits", "", *[f"- {item}" for item in limitations]]
    if license_table:
        lines += [
            "",
            "## Data sources",
            "",
            "The source registry records licenses and usage status. "
            "It does not identify which data trained these weights.",
            "",
            license_table,
        ]
    return "\n".join(lines) + "\n"
