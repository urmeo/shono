"""The experiment record — a run's hypothesis, config, environment, and metrics.

A training run that does not record its seed, full config, environment, and
metrics did not happen. This module renders that record as a self-contained
Markdown block: it is generated on the machine that ran the training (where the
provenance is real), saved beside the checkpoints, and carries everything needed
to reproduce or contest a number. ``metrics=None`` renders a pre-run record (the
plan); filling metrics in after the run completes it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from shono.provenance import RunContext
from shono.train.config import TrainConfig


def render_experiment(
    name: str,
    hypothesis: str,
    config: TrainConfig,
    run_context: RunContext,
    *,
    data_sources: Sequence[str] = (),
    n_examples: int = 0,
    train_hours: float = 0.0,
    eval_command: str = "",
    metrics: Mapping[str, float] | None = None,
    baseline: Mapping[str, float] | None = None,
) -> str:
    """Render a complete experiment record as Markdown.

    ``metrics`` is the run's measured numbers (e.g. ``{"eval_wer": 0.27}``);
    leave it ``None`` before the run to render the plan, then re-render with
    metrics to complete it. ``baseline`` is the comparison point the run must beat.
    """
    rc = run_context
    lines = [
        f"### experiment: {name}",
        f"- **Hypothesis:** {hypothesis}",
        f"- **Base model:** {config.model_id} · full fine-tune, {config.optim}",
        f"- **Seed:** {config.seed} · **effective batch:** {config.effective_batch_size} "
        f"· **lr:** {config.learning_rate:g} {config.lr_scheduler_type} "
        f"(warmup {config.warmup_steps}) · **epochs:** {config.num_train_epochs:g}",
        f"- **Data:** {n_examples} examples, {train_hours:.1f} h from "
        f"{list(data_sources) if data_sources else '—'}; "
        f"timestamped fraction {config.timestamp_sample_fraction:g}",
        f"- **Environment:** python {rc.python} · {rc.platform} · "
        f"git {rc.git.sha or '—'}{' (dirty)' if rc.git.dirty else ''} · "
        f"normalizer v{rc.normalizer_version}",
        f"- **Packages:** {_fmt_packages(rc.packages)}",
        f"- **Eval command:** `{eval_command or '—'}`",
        f"- **Metrics:** {_fmt_metrics(metrics, baseline)}",
        "",
        "<details><summary>Full config</summary>",
        "",
        "```json",
        json.dumps(config.to_dict(), indent=2, ensure_ascii=False),
        "```",
        "</details>",
    ]
    return "\n".join(lines)


def _fmt_packages(packages: Mapping[str, str]) -> str:
    keys = ("torch", "transformers", "ctranslate2", "bitsandbytes")
    parts = [f"{k} {packages[k]}" for k in keys if k in packages]
    return ", ".join(parts) if parts else "—"


def _fmt_metrics(
    metrics: Mapping[str, float] | None, baseline: Mapping[str, float] | None
) -> str:
    if not metrics:
        return "— (not run yet)"
    out = []
    for key, value in metrics.items():
        base = baseline.get(key) if baseline else None
        if base is not None:
            delta = value - base
            out.append(f"{key} {value:.4f} (baseline {base:.4f}, Δ {delta:+.4f})")
        else:
            out.append(f"{key} {value:.4f}")
    return " · ".join(out)
