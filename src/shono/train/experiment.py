"""Render declared training settings, provenance and measured or pending metrics."""

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
    """Render measured metrics, or a pending record when metrics is None."""
    rc = run_context
    lines = [
        f"### experiment: {name}",
        f"- **Hypothesis:** {hypothesis}",
        f"- **Base model:** {config.model_id} · "
        f"{'encoder frozen' if config.freeze_encoder else 'full fine-tune'}, {config.optim}",
        f"- **Seed:** {config.seed} · **effective batch:** {config.effective_batch_size} "
        f"· **lr:** {config.learning_rate:g} {config.lr_scheduler_type} "
        f"(warmup {config.warmup_steps}) · **epochs:** {config.num_train_epochs:g}",
        f"- **Data:** {n_examples} examples, {train_hours:.1f} h from "
        f"{list(data_sources) if data_sources else 'unknown'}; "
        f"timestamped fraction {config.timestamp_sample_fraction:g}",
        f"- **Environment:** python {rc.python} · {rc.platform} · "
        f"git {rc.git.sha or 'unknown'}{' (dirty)' if rc.git.dirty else ''} · "
        f"normalizer v{rc.normalizer_version}",
        f"- **Packages:** {_fmt_packages(rc.packages)}",
        f"- **Eval command:** `{eval_command or 'unknown'}`",
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
    return ", ".join(parts) if parts else "unknown"


def _fmt_metrics(metrics: Mapping[str, float] | None, baseline: Mapping[str, float] | None) -> str:
    if not metrics:
        return "unknown (not run yet)"
    out = []
    for key, value in metrics.items():
        base = baseline.get(key) if baseline else None
        if base is not None:
            delta = value - base
            out.append(f"{key} {value:.4f} (baseline {base:.4f}, Δ {delta:+.4f})")
        else:
            out.append(f"{key} {value:.4f}")
    return " · ".join(out)
