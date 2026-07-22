"""Per-slice, per-system evaluation reports — the honest scoreboard.

A report is a matrix of **systems** (our model, the base model, commercial APIs)
against **slices** (read, long-form, code-switch — each a manifest). Every cell
is a system's score on a slice: raw and normalized WER/CER, with a
blockwise-bootstrap CI on the normalized numbers (the frozen protocol's official
comparison). A report also carries the run context, the normalizer version, the
license table, and any leakage findings — everything a reader needs to trust or
reproduce a number.

Two rules keep it honest, and they are the reason this module exists:

    * A slice with **no predictions file** is *pending* and renders ``—`` — a
      number that has not been measured is never invented.
    * A predictions file that is **present but incomplete** is a hard error, not
      a partial score — a half-run must never masquerade as a result.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from shono.data.license import LicenseRegistry
from shono.data.manifest import Manifest
from shono.eval.ci import BootstrapCI, blockwise_bootstrap_ci
from shono.eval.codeswitch import CS_NORMALIZER_VERSION, code_switch_normalize
from shono.eval.normalize import NORMALIZER_VERSION, normalize
from shono.eval.score import ScoreReport, score
from shono.provenance import RunContext

_DEFAULT_RESAMPLES = 1000

# The code-switch slice is scored under the documented script-normalized policy.
_CODE_SWITCH_DOMAIN = "code-switch"


def _normalizer_for(domain: str) -> tuple[Callable[[str], str], str]:
    if domain == _CODE_SWITCH_DOMAIN:
        return code_switch_normalize, CS_NORMALIZER_VERSION
    return normalize, NORMALIZER_VERSION


# ---- predictions ---------------------------------------------------------


@dataclass(frozen=True)
class Predictions:
    """One system's hypotheses for one manifest, keyed by segment id."""

    system: str
    manifest: str
    hypotheses: dict[str, str]
    run_context: dict | None = None

    @classmethod
    def from_jsonl(cls, path: str | Path) -> Predictions:
        raw = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not raw:
            raise ValueError(f"predictions file {path} is empty")
        head = json.loads(raw[0])
        if "predictions" not in head:
            raise ValueError(
                f"predictions file {path} must start with a "
                '{"predictions": {"system": ..., "manifest": ...}} header'
            )
        meta = head["predictions"]
        hyps: dict[str, str] = {}
        for lineno, ln in enumerate(raw[1:], start=2):
            rec = json.loads(ln)
            sid = rec["id"]
            if sid in hyps:
                raise ValueError(
                    f"predictions file {path} has a duplicate id {sid!r} on line {lineno}; "
                    "a re-run must not silently overwrite an earlier hypothesis"
                )
            hyps[sid] = rec["hypothesis"]
        return cls(
            system=meta["system"],
            manifest=meta.get("manifest", ""),
            hypotheses=hyps,
            run_context=meta.get("run_context"),
        )


# ---- one scored cell -----------------------------------------------------


@dataclass(frozen=True)
class SliceScore:
    """One (system, slice) cell. ``pending`` means no predictions yet — renders ``—``."""

    system: str
    slice_name: str
    domain: str
    status: str  # "scored" | "pending"
    n_segments: int = 0
    n_recordings: int = 0
    score: ScoreReport | None = None
    wer_ci: BootstrapCI | None = None
    cer_ci: BootstrapCI | None = None

    @property
    def is_pending(self) -> bool:
        return self.status == "pending"


def score_slice(
    manifest: Manifest,
    predictions: Predictions,
    *,
    seed: int = 0,
    n_resamples: int = _DEFAULT_RESAMPLES,
    slice_name: str | None = None,
) -> SliceScore:
    """Score one system on one slice — point estimates plus normalized CIs.

    Raises if the predictions and manifest do not correspond exactly: a missing
    hypothesis (partial run), an extra hypothesis (stale/mispaired file), or a
    manifest-name mismatch. An incomplete or mismatched run is an error, never a
    partial score. A slice with a single recording block is scored without a CI
    (a bootstrap over one block has no between-block variance) rather than
    aborting — the point estimate still stands.
    """
    if predictions.manifest and predictions.manifest != manifest.name:
        raise ValueError(
            f"predictions declare manifest {predictions.manifest!r} but are being "
            f"scored against {manifest.name!r}; mispaired predictions file"
        )
    extra = set(predictions.hypotheses) - set(manifest.ids())
    if extra:
        preview = ", ".join(sorted(extra)[:5])
        raise ValueError(
            f"predictions for {manifest.name!r} contain {len(extra)} id(s) not in the "
            f"manifest: {preview}. Predictions must match the slice exactly."
        )

    raw_records = manifest.records_with(predictions.hypotheses)
    refs = [r for _, r, _ in raw_records]
    hyps = [h for _, _, h in raw_records]
    normalizer, normalizer_version = _normalizer_for(manifest.domain)
    report = score(refs, hyps, normalizer=normalizer, normalizer_version=normalizer_version)

    norm_records = [(rid, normalizer(r), normalizer(h)) for rid, r, h in raw_records]
    single_block = len(manifest.recording_ids()) < 2
    wer_ci = None if single_block else blockwise_bootstrap_ci(
        norm_records, "wer", n_resamples=n_resamples, seed=seed
    )
    cer_ci = None if single_block else blockwise_bootstrap_ci(
        norm_records, "cer", n_resamples=n_resamples, seed=seed
    )

    return SliceScore(
        system=predictions.system,
        slice_name=slice_name or manifest.name,
        domain=manifest.domain,
        status="scored",
        n_segments=len(manifest.segments),
        n_recordings=len(manifest.recording_ids()),
        score=report,
        wer_ci=wer_ci,
        cer_ci=cer_ci,
    )


# ---- report specification ------------------------------------------------


@dataclass(frozen=True)
class SliceSpec:
    name: str
    manifest: str  # path to the manifest, relative to the spec's base dir
    domain: str = ""  # intended domain, shown in the skeleton before the manifest exists


@dataclass(frozen=True)
class SystemSpec:
    name: str
    predictions: dict[str, str]  # slice name -> predictions path (relative), when it exists


@dataclass(frozen=True)
class ReportSpec:
    """Declarative definition of a report: which systems, which slices, which files."""

    name: str
    title: str
    slices: tuple[SliceSpec, ...]
    systems: tuple[SystemSpec, ...]
    seed: int = 0
    n_resamples: int = _DEFAULT_RESAMPLES
    train_manifests: tuple[str, ...] = ()  # optional, for the leakage section

    @classmethod
    def from_json(cls, path: str | Path) -> ReportSpec:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        slices = tuple(
            SliceSpec(name=s["name"], manifest=s["manifest"], domain=s.get("domain", ""))
            for s in doc["slices"]
        )
        systems = tuple(
            SystemSpec(name=s["name"], predictions=dict(s.get("predictions", {})))
            for s in doc["systems"]
        )
        return cls(
            name=doc["name"],
            title=doc["title"],
            slices=slices,
            systems=systems,
            seed=doc.get("seed", 0),
            n_resamples=doc.get("n_resamples", _DEFAULT_RESAMPLES),
            train_manifests=tuple(doc.get("train_manifests", [])),
        )


# ---- the assembled report ------------------------------------------------


@dataclass(frozen=True)
class Report:
    """A rendered-ready report: scored cells plus the context to trust them."""

    name: str
    title: str
    run_context: RunContext
    cells: tuple[SliceScore, ...]
    slice_order: tuple[str, ...]
    system_order: tuple[str, ...]
    license_table: str
    leakage_notes: tuple[str, ...] = ()

    def cell(self, system: str, slice_name: str) -> SliceScore:
        for c in self.cells:
            if c.system == system and c.slice_name == slice_name:
                return c
        raise KeyError(f"no cell for system={system!r}, slice={slice_name!r}")

    # -- rendering --

    @staticmethod
    def _pct_ci(ci: BootstrapCI) -> str:
        return f"{ci.point:.1%} [{ci.lower:.1%}, {ci.upper:.1%}]"

    def _row(self, system: str, slice_name: str) -> str:
        c = self.cell(system, slice_name)
        if c.is_pending:
            return f"| {system} | — | — | — | — |"
        s = c.score
        assert s is not None
        no_ci = "{:.1%} (1 block, no CI)"
        wer = self._pct_ci(c.wer_ci) if c.wer_ci else no_ci.format(s.wer_normalized)
        cer = self._pct_ci(c.cer_ci) if c.cer_ci else no_ci.format(s.cer_normalized)
        return f"| {system} | {wer} | {cer} | {s.wer_raw:.1%} | {s.cer_raw:.1%} |"

    def render_markdown(self) -> str:
        rc = self.run_context
        lines = [
            f"# {self.title}",
            "",
            f"- **Normalizer:** v{rc.normalizer_version} (frozen; `src/shono/eval/normalize.py`)",
            f"- **Generated:** {rc.timestamp}",
            f"- **Command:** `{rc.command}`" if rc.command else "- **Command:** —",
            f"- **Git:** {rc.git.sha or '—'}{' (dirty)' if rc.git.dirty else ''}",
            f"- **Seed:** {rc.seed} · **CI:** blockwise bootstrap, "
            f"{rc.config.get('n_resamples', '—')} resamples, 95%",
            "",
            "Normalized WER/CER carry a 95% blockwise-bootstrap CI (the frozen "
            "protocol's official numbers); raw columns are point estimates for "
            "transparency. `—` means not yet measured — never an assumed value.",
            "",
        ]
        for slice_name in self.slice_order:
            domain = next(
                (c.domain for c in self.cells if c.slice_name == slice_name), ""
            )
            scored = next(
                (c for c in self.cells if c.slice_name == slice_name and not c.is_pending), None
            )
            meta = ""
            if scored is not None:
                meta = f" — {scored.n_segments} segments, {scored.n_recordings} recordings"
            lines += [
                f"## {slice_name} ({domain}){meta}",
                "",
                "| System | WER (norm, 95% CI) | CER (norm, 95% CI) | WER (raw) | CER (raw) |",
                "|---|---|---|---|---|",
            ]
            lines += [self._row(sys, slice_name) for sys in self.system_order]
            lines.append("")
        if any(c.domain == _CODE_SWITCH_DOMAIN for c in self.cells):
            lines += [
                "## Code-switch scoring",
                "",
                f"Code-switch slices are scored under the script-normalized policy "
                f"(v{CS_NORMALIZER_VERSION}): the frozen normalizer plus case-insensitive "
                "Latin. Cross-script transliteration equivalence (a Latin word vs its "
                "Bengali spelling) is deliberately **not** credited — no reliable Bn-En "
                "transliteration lexicon exists, and asserting equivalence would fabricate "
                "matches.",
                "",
            ]
        if self.leakage_notes:
            lines += ["## Leakage audit", ""]
            lines += [f"- {n}" for n in self.leakage_notes]
            lines.append("")
        lines += ["## Data licenses", "", self.license_table, ""]
        return "\n".join(lines)

    def to_json_dict(self) -> dict:
        def _cell(c: SliceScore) -> dict:
            base = {
                "system": c.system,
                "slice": c.slice_name,
                "domain": c.domain,
                "status": c.status,
            }
            if c.is_pending:
                return base
            assert c.score is not None

            def _norm(point: float, ci):
                return {
                    "point": point,
                    "lo": ci.lower if ci else None,
                    "hi": ci.upper if ci else None,
                }

            base.update(
                n_segments=c.n_segments,
                n_recordings=c.n_recordings,
                wer_raw=c.score.wer_raw,
                cer_raw=c.score.cer_raw,
                wer_norm=_norm(c.score.wer_normalized, c.wer_ci),
                cer_norm=_norm(c.score.cer_normalized, c.cer_ci),
            )
            return base

        return {
            "name": self.name,
            "title": self.title,
            "normalizer_version": self.run_context.normalizer_version,
            "run_context": self.run_context.to_dict(),
            "slices": list(self.slice_order),
            "systems": list(self.system_order),
            "cells": [_cell(c) for c in self.cells],
            "leakage_notes": list(self.leakage_notes),
        }


# ---- building ------------------------------------------------------------


def build_report(
    spec: ReportSpec,
    base_dir: str | Path,
    registry: LicenseRegistry,
    *,
    command: str | None = None,
    run_context: RunContext | None = None,
) -> Report:
    """Assemble a :class:`Report` from a spec, loading manifests and predictions from ``base_dir``.

    A slice whose predictions file is absent becomes a *pending* cell; a present
    file is scored (and errors if it is incomplete).
    """
    base = Path(base_dir)
    ctx = run_context or RunContext.capture(
        spec.seed,
        {"report": spec.name, "n_resamples": spec.n_resamples},
        command=command,
    )

    # A manifest that does not exist yet (it is generated from gated data on
    # Kaggle) makes its whole slice pending — the skeleton still renders, with
    # the intended domain from the spec.
    manifests: dict[str, Manifest] = {}
    domains: dict[str, str] = {}
    for s in spec.slices:
        mpath = base / s.manifest
        if mpath.exists():
            m = Manifest.from_jsonl(mpath)
            m.validate_against(registry)
            manifests[s.name] = m
            domains[s.name] = m.domain
        else:
            domains[s.name] = s.domain

    cells: list[SliceScore] = []
    for sysspec in spec.systems:
        for s in spec.slices:
            manifest = manifests.get(s.name)
            pred_path = sysspec.predictions.get(s.name)
            if manifest is None or not pred_path or not (base / pred_path).exists():
                cells.append(
                    SliceScore(
                        system=sysspec.name,
                        slice_name=s.name,
                        domain=domains[s.name],
                        status="pending",
                    )
                )
                continue
            preds = Predictions.from_jsonl(base / pred_path)
            cells.append(
                score_slice(
                    manifest,
                    preds,
                    seed=spec.seed,
                    n_resamples=spec.n_resamples,
                    slice_name=s.name,
                )
            )

    leakage_notes: list[str] = []
    if spec.train_manifests:
        from shono.data.leakage import audit_leakage  # local: avoids a data<->eval import cycle

        train_paths = [base / p for p in spec.train_manifests]
        if all(p.exists() for p in train_paths):
            trains = [Manifest.from_jsonl(p) for p in train_paths]
            for s in spec.slices:
                if s.name in manifests:
                    leakage_notes.append(audit_leakage(trains, manifests[s.name]).summary())

    return Report(
        name=spec.name,
        title=spec.title,
        run_context=ctx,
        cells=tuple(cells),
        slice_order=tuple(s.name for s in spec.slices),
        system_order=tuple(s.name for s in spec.systems),
        license_table=registry.render_table(),
        leakage_notes=tuple(leakage_notes),
    )


def write_report(report: Report, out_dir: str | Path) -> tuple[Path, Path]:
    """Write ``<name>.md`` and ``<name>.json`` into ``out_dir``; return both paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / f"{report.name}.md"
    json_path = out / f"{report.name}.json"
    md_path.write_text(report.render_markdown(), encoding="utf-8")
    json_path.write_text(
        json.dumps(report.to_json_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return md_path, json_path
