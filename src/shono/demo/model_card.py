"""Generate a model card from an evaluation report — numbers come from data, not hands.

The card quotes results *only* from a report produced by the frozen harness
(``reports/*.json``): every metric is looked up from a scored cell, and anything
not yet measured renders ``—``. It is impossible to state a number the harness did
not produce, which is the whole point — a model card is where accuracy claims meet
the world.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def _our_cell(report: Mapping, system: str, slice_name: str) -> dict | None:
    for cell in report.get("cells", []):
        if cell["system"] == system and cell["slice"] == slice_name:
            return cell
    return None


def _wer_str(cell: dict | None) -> str:
    if not cell or cell.get("status") != "scored":
        return "—"
    wn = cell["wer_norm"]
    if wn.get("lo") is None:
        return f"{wn['point']:.1%}"
    return f"{wn['point']:.1%} [{wn['lo']:.1%}, {wn['hi']:.1%}]"


def render_model_card(
    report: Mapping,
    *,
    model_id: str,
    system: str,
    base_model: str,
    license_spdx: str,
    repo_url: str,
    license_table: str = "",
    limitations: Sequence[str] = (),
) -> str:
    """Render a Hugging Face model card for ``system`` from an evaluation ``report``."""
    norm_v = report.get("normalizer_version", "—")
    lines = [
        "---",
        f"license: {license_spdx}",
        "language: bn",
        "pipeline_tag: automatic-speech-recognition",
        f"base_model: {base_model}",
        "---",
        "",
        f"# {model_id}",
        "",
        f"Bengali speech recognition fine-tuned from `{base_model}`, built for real-world "
        "audio: long-form lectures and podcasts, speaker diarization, and Bangla-English "
        "code-switching. Part of [Shono]({repo_url}).".format(repo_url=repo_url),
        "",
        "## Results",
        "",
        f"Word error rate (lower is better), scored under the frozen Shono normalizer "
        f"(v{norm_v}) with a 95% blockwise-bootstrap CI. `—` = not yet measured.",
        "",
        "| Slice | WER (normalized, 95% CI) |",
        "|---|---|",
    ]
    for s in report.get("slices", []):
        cell = _our_cell(report, system, s)
        lines.append(f"| {s} | {_wer_str(cell)} |")
    lines.append("")
    lines.append(
        f"Full per-slice comparison against the base model and commercial APIs, with the "
        f"exact scoring script, lives in the [Shono repository]({repo_url})."
    )
    if limitations:
        lines += ["", "## Limitations", ""]
        lines += [f"- {item}" for item in limitations]
    if license_table:
        lines += ["", "## Training data", "", license_table]
    lines += [
        "",
        "## License",
        "",
        f"Model code: {license_spdx}. Training data and any base weights carry their own "
        "licenses, recorded per source in the repository.",
    ]
    return "\n".join(lines) + "\n"
