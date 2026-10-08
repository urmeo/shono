"""Score complete prediction cells and retain their input and inference metadata."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath

from shono.data.license import LicenseRegistry
from shono.data.manifest import Manifest
from shono.data.validation import json_object
from shono.eval.ci import BootstrapCI, blockwise_bootstrap_ci, validate_bootstrap_settings
from shono.eval.codeswitch import CS_NORMALIZER_VERSION, code_switch_normalize
from shono.eval.normalize import NORMALIZER_VERSION, normalize
from shono.eval.score import ScoreReport, score
from shono.output_paths import source_files, validate_output_destinations, write_text_atomic
from shono.provenance import RunContext, safe_command, safe_metadata

_DEFAULT_RESAMPLES = 1000
_CODE_SWITCH_DOMAIN = "code-switch"


def _normalizer_for(domain: str) -> tuple[Callable[[str], str], str]:
    if domain == _CODE_SWITCH_DOMAIN:
        return code_switch_normalize, CS_NORMALIZER_VERSION
    return normalize, NORMALIZER_VERSION


def _policy_for(domain: str) -> tuple[str, str]:
    return (
        ("code-switch-script", CS_NORMALIZER_VERSION)
        if domain == _CODE_SWITCH_DOMAIN
        else ("bengali", NORMALIZER_VERSION)
    )


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _relative_path(value: object, name: str) -> str:
    value = _text(value, name)
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"{name} must be a relative POSIX path without traversal")
    return value


def _load_manifest(path: Path) -> tuple[Manifest, str]:
    raw = path.read_bytes()
    manifest = Manifest.from_jsonl(path)
    if path.read_bytes() != raw:
        raise ValueError(f"manifest changed while loading: {path}")
    return manifest, hashlib.sha256(raw).hexdigest()


def _md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").replace("\r", " ")


@dataclass(frozen=True)
class Predictions:
    """Hypotheses keyed by segment ID, with optional inference metadata."""

    system: str
    manifest: str
    hypotheses: dict[str, str]
    run_context: dict | None = None
    input_paths: tuple[str, ...] = ()
    source_sha256: str = "unknown"

    def __post_init__(self) -> None:
        _text(self.system, "prediction system")
        _text(self.manifest, "prediction manifest")
        if not isinstance(self.hypotheses, dict):
            raise ValueError("hypotheses must be an object")
        for sid, hypothesis in self.hypotheses.items():
            _text(sid, "prediction id")
            if not isinstance(hypothesis, str):
                raise ValueError("hypotheses must be strings")
        if self.run_context is not None:
            if not isinstance(self.run_context, dict):
                raise ValueError("run_context must be an object or null")
            context = safe_metadata(self.run_context)
            if "command" in context:
                context["command"] = safe_command(context["command"])
            object.__setattr__(self, "run_context", context)

    def to_jsonl(self, path: str | Path, *, protected_paths=()) -> None:
        destination = validate_output_destinations(
            (path,), protected_paths=(*self.input_paths, *protected_paths)
        )[0]
        header = {"system": self.system, "manifest": self.manifest}
        if self.run_context is not None:
            header["run_context"] = self.run_context
        lines = [json.dumps({"predictions": header}, ensure_ascii=False, allow_nan=False)]
        lines += [
            json.dumps({"id": sid, "hypothesis": hyp}, ensure_ascii=False)
            for sid, hyp in self.hypotheses.items()
        ]
        write_text_atomic(destination, "\n".join(lines) + "\n")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> Predictions:
        raw = Path(path).read_bytes()
        lines = [line for line in raw.decode("utf-8").splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"predictions file {path} is empty")
        header = json_object(lines[0], str(path))
        if not isinstance(header, dict) or not isinstance(header.get("predictions"), dict):
            raise ValueError(f"predictions file {path} must start with a predictions header")
        meta = header["predictions"]
        hypotheses = {}
        for lineno, line in enumerate(lines[1:], start=2):
            record = json_object(line, f"{path}:{lineno}")
            if not isinstance(record, dict) or "id" not in record or "hypothesis" not in record:
                raise ValueError(f"invalid prediction record on line {lineno} of {path}")
            sid = _text(record["id"], "prediction id")
            if sid in hypotheses:
                raise ValueError(
                    f"predictions file {path} has a duplicate id {sid!r} on line {lineno}"
                )
            hypotheses[sid] = record["hypothesis"]
        return cls(
            meta.get("system"),
            meta.get("manifest"),
            hypotheses,
            meta.get("run_context"),
            source_sha256=hashlib.sha256(raw).hexdigest(),
        )


@dataclass(frozen=True)
class SliceScore:
    """A scored or pending system/slice cell."""

    system: str
    slice_name: str
    domain: str
    status: str
    n_segments: int = 0
    n_recordings: int = 0
    score: ScoreReport | None = None
    wer_ci: BootstrapCI | None = None
    cer_ci: BootstrapCI | None = None
    normalization_policy: str = "unknown"
    normalizer_version: str = "unknown"
    inference_context: dict | str = "unknown"
    input_hashes: dict[str, str] = field(default_factory=dict)
    leakage_audit: dict = field(default_factory=lambda: {"status": "unknown"})
    pending_reason: str = ""
    model_id: str = "unknown"
    inference_mode: str = "unknown"

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
    input_hashes: dict[str, str] | None = None,
    leakage_audit: dict | None = None,
    model_id: str = "unknown",
    inference_mode: str = "unknown",
) -> SliceScore:
    """Score exact manifest/prediction identities; one block has no CI."""
    validate_bootstrap_settings(n_resamples, 0.95, seed)
    predictions.__post_init__()
    if predictions.manifest != manifest.name:
        raise ValueError(
            f"predictions declare manifest {predictions.manifest!r}, not {manifest.name!r}; "
            "mispaired predictions file"
        )
    extra = set(predictions.hypotheses) - set(manifest.ids())
    if extra:
        raise ValueError(
            f"predictions for {manifest.name!r} contain id(s) not in the manifest: "
            f"{sorted(extra)[:5]}"
        )
    records = manifest.records_with(predictions.hypotheses)
    normalizer, version = _normalizer_for(manifest.domain)
    result = score(
        [r for _, r, _ in records],
        [h for _, _, h in records],
        normalizer=normalizer,
        normalizer_version=version,
    )
    normalized = [(rid, normalizer(r), normalizer(h)) for rid, r, h in records]
    one_block = len(manifest.recording_ids()) < 2
    return SliceScore(
        system=predictions.system,
        slice_name=slice_name or manifest.name,
        domain=manifest.domain,
        status="scored",
        n_segments=len(manifest.segments),
        n_recordings=len(manifest.recording_ids()),
        score=result,
        wer_ci=None
        if one_block
        else blockwise_bootstrap_ci(normalized, "wer", n_resamples=n_resamples, seed=seed),
        cer_ci=None
        if one_block
        else blockwise_bootstrap_ci(normalized, "cer", n_resamples=n_resamples, seed=seed),
        normalization_policy=_policy_for(manifest.domain)[0],
        normalizer_version=version,
        inference_context=predictions.run_context
        if predictions.run_context is not None
        else "unknown",
        input_hashes=input_hashes or {"manifest": "unknown", "predictions": "unknown"},
        leakage_audit=leakage_audit or {"status": "unknown"},
        model_id=model_id,
        inference_mode=inference_mode,
    )


@dataclass(frozen=True)
class SliceSpec:
    name: str
    manifest: str
    domain: str = ""

    def __post_init__(self) -> None:
        _text(self.name, "slice name")
        _relative_path(self.manifest, "manifest path")
        if not isinstance(self.domain, str):
            raise ValueError("slice domain must be a string")


@dataclass(frozen=True)
class SystemSpec:
    name: str
    predictions: dict[str, str]
    model_id: str = "unknown"
    inference_modes: dict[str, str] = field(default_factory=dict)
    training_audit: str = "unknown"
    train_manifests: tuple[str, ...] = ()
    text_decisions: dict[str, dict[str, str]] = field(default_factory=dict)
    pending_slices: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _text(self.name, "system name")
        _text(self.model_id, "model_id")
        if not isinstance(self.predictions, dict) or not isinstance(self.inference_modes, dict):
            raise ValueError("predictions and inference_modes must be objects")
        for name, path in self.predictions.items():
            _text(name, "prediction slice")
            _relative_path(path, "prediction path")
        for name, mode in self.inference_modes.items():
            _text(name, "inference slice")
            _text(mode, "inference mode")
        if not isinstance(self.training_audit, str) or self.training_audit not in {
            "required",
            "unknown",
            "not_applicable",
        }:
            raise ValueError("training_audit must be required, unknown or not_applicable")
        if "fine-tuned" in self.name and self.training_audit != "required":
            raise ValueError("fine-tuned systems require a declared training leakage audit")
        if not isinstance(self.train_manifests, (tuple, list)):
            raise ValueError("train_manifests must be unique paths")
        for path in self.train_manifests:
            _relative_path(path, "training manifest path")
        if len(set(self.train_manifests)) != len(self.train_manifests):
            raise ValueError("train_manifests must be unique paths")
        if not isinstance(self.text_decisions, dict) or not isinstance(self.pending_slices, dict):
            raise ValueError("text_decisions and pending_slices must be objects")
        for name, decisions in self.text_decisions.items():
            _text(name, "review slice")
            if not isinstance(decisions, dict):
                raise ValueError("each text review must be an object of exact keys and reasons")
            for key, reason in decisions.items():
                _text(key, "text review key")
                _text(reason, "text review reason")
        for name, reason in self.pending_slices.items():
            _text(name, "pending slice")
            _text(reason, "pending reason")


@dataclass(frozen=True)
class ReportSpec:
    name: str
    title: str
    slices: tuple[SliceSpec, ...]
    systems: tuple[SystemSpec, ...]
    seed: int = 0
    n_resamples: int = _DEFAULT_RESAMPLES
    train_manifests: tuple[str, ...] = ()
    audio_root: str | None = None
    source_path: str | None = None
    source_sha256: str = "unknown"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]*", self.name
        ):
            raise ValueError(
                "report name must contain only letters, digits, underscores and hyphens"
            )
        _text(self.title, "report title")
        validate_bootstrap_settings(self.n_resamples, 0.95, self.seed)
        for values, kind, cls in [
            (self.slices, "slices", SliceSpec),
            (self.systems, "systems", SystemSpec),
        ]:
            if (
                not isinstance(values, (list, tuple))
                or not values
                or not all(isinstance(v, cls) for v in values)
            ):
                raise ValueError(f"{kind} must be a nonempty sequence of {cls.__name__}")
            if len({v.name for v in values}) != len(values):
                raise ValueError(f"duplicate {kind} names")
        names = {s.name for s in self.slices}
        for system in self.systems:
            for mapping in (
                system.predictions,
                system.inference_modes,
                system.text_decisions,
                system.pending_slices,
            ):
                if set(mapping) - names:
                    raise ValueError(f"system {system.name!r} references unknown slices")
        if not isinstance(self.train_manifests, (tuple, list)):
            raise ValueError("train_manifests must be a sequence")
        for path in self.train_manifests:
            _relative_path(path, "training manifest path")
        if self.audio_root is not None:
            _text(self.audio_root, "audio_root")

    @classmethod
    def from_json(cls, path: str | Path) -> ReportSpec:
        raw = Path(path).read_bytes()
        doc = json_object(raw.decode("utf-8"), str(path))
        if set(doc) - {
            "name",
            "title",
            "slices",
            "systems",
            "seed",
            "n_resamples",
            "train_manifests",
            "audio_root",
        }:
            raise ValueError("unknown report spec fields")
        if not isinstance(doc.get("slices"), list) or not isinstance(doc.get("systems"), list):
            raise ValueError("report slices and systems must be arrays")
        if any(not isinstance(item, dict) for item in doc["slices"] + doc["systems"]):
            raise ValueError("each report slice and system must be an object")
        if any(set(item) - {"name", "manifest", "domain"} for item in doc["slices"]):
            raise ValueError("unknown slice spec fields")
        if any(
            set(item)
            - {
                "name",
                "predictions",
                "model_id",
                "inference_modes",
                "training_audit",
                "train_manifests",
                "text_decisions",
                "pending_slices",
            }
            for item in doc["systems"]
        ):
            raise ValueError("unknown system spec fields")
        return cls(
            name=doc.get("name"),
            title=doc.get("title"),
            slices=tuple(
                SliceSpec(s.get("name"), s.get("manifest"), s.get("domain", ""))
                for s in doc["slices"]
            ),
            systems=tuple(
                SystemSpec(
                    s.get("name"),
                    s.get("predictions", {}),
                    s.get("model_id", "unknown"),
                    s.get("inference_modes", {}),
                    s.get("training_audit", "unknown"),
                    s.get("train_manifests", ()),
                    s.get("text_decisions", {}),
                    s.get("pending_slices", {}),
                )
                for s in doc["systems"]
            ),
            seed=doc.get("seed", 0),
            n_resamples=doc.get("n_resamples", _DEFAULT_RESAMPLES),
            train_manifests=doc.get("train_manifests", ()),
            audio_root=doc.get("audio_root"),
            source_path=str(Path(path).resolve()),
            source_sha256=hashlib.sha256(raw).hexdigest(),
        )


@dataclass(frozen=True)
class Report:
    name: str
    title: str
    run_context: RunContext
    cells: tuple[SliceScore, ...]
    slice_order: tuple[str, ...]
    system_order: tuple[str, ...]
    license_table: str
    leakage_notes: tuple[str, ...] = ()
    input_paths: tuple[str, ...] = ()

    def _validate_release(self) -> None:
        for cell in self.cells:
            if (
                cell.status == "scored"
                and "fine-tuned" in cell.system
                and cell.leakage_audit.get("status") != "reviewed"
            ):
                raise ValueError(
                    "scored fine-tuned cells require a reviewed declared training audit"
                )

    def cell(self, system: str, slice_name: str) -> SliceScore:
        for cell in self.cells:
            if cell.system == system and cell.slice_name == slice_name:
                return cell
        raise KeyError(f"no cell for system={system!r}, slice={slice_name!r}")

    @staticmethod
    def _pct_ci(ci: BootstrapCI) -> str:
        return f"{ci.point:.1%} [{ci.lower:.1%}, {ci.upper:.1%}]"

    def _row(self, system: str, slice_name: str) -> str:
        cell = self.cell(system, slice_name)
        policy = f"{cell.normalization_policy} v{cell.normalizer_version}"
        if cell.is_pending:
            return f"| {_md(system)} | pending | pending | pending | pending | {policy} |"
        assert cell.score is not None
        no_ci = "{:.1%} (1 block, no CI)"
        wer = self._pct_ci(cell.wer_ci) if cell.wer_ci else no_ci.format(cell.score.wer_normalized)
        cer = self._pct_ci(cell.cer_ci) if cell.cer_ci else no_ci.format(cell.score.cer_normalized)
        return (
            f"| {_md(system)} | {wer} | {cer} | {cell.score.wer_raw:.1%} | "
            f"{cell.score.cer_raw:.1%} | {policy} |"
        )

    def render_markdown(self) -> str:
        self._validate_release()
        rc = self.run_context
        lines = [
            f"# {_md(self.title)}",
            "",
            f"- Generated: {rc.timestamp}",
            f"- Scoring command: `{rc.command or 'unknown'}`",
            f"- Git: {rc.git.sha or 'unknown'}{' (dirty)' if rc.git.dirty else ''}",
            f"- Seed: {rc.seed}; CI: blockwise bootstrap, "
            f"{rc.config.get('n_resamples', 'unknown')} resamples, 95%.",
            "",
            "Each cell states its normalization policy. Pending cells have no measured score.",
            "",
        ]
        for name in self.slice_order:
            first = self.cell(self.system_order[0], name)
            lines += [
                f"## {_md(name)} ({_md(first.domain)})",
                "",
                "| System | WER (norm, 95% CI) | CER (norm, 95% CI) | "
                "WER (raw) | CER (raw) | Policy |",
                "|---|---|---|---|---|---|",
            ]
            lines += [self._row(system, name) for system in self.system_order]
            lines += [""]
            for system in self.system_order:
                cell = self.cell(system, name)
                context_status = (
                    "unknown" if cell.inference_context == "unknown" else "recorded in JSON"
                )
                lines.append(
                    f"- {_md(system)}: model `{cell.model_id}`; mode {cell.inference_mode}; "
                    f"inference context {context_status}; leakage {cell.leakage_audit['status']}."
                )
                if cell.pending_reason:
                    lines.append(f"  Pending reason: {_md(cell.pending_reason)}.")
            lines += [""]
        if any(cell.domain == _CODE_SWITCH_DOMAIN for cell in self.cells):
            lines += [
                "## Code-switch scoring",
                "",
                "Script normalization casefolds Latin. "
                "Cross-script transliteration equivalence is not credited.",
                "",
            ]
        if self.leakage_notes:
            lines += [
                "## Leakage audit",
                "",
                *[f"- {_md(note)}" for note in self.leakage_notes],
                "",
            ]
        lines += ["## Data licenses", "", self.license_table, ""]
        return "\n".join(lines)

    def to_json_dict(self) -> dict:
        self._validate_release()
        cells = []
        for cell in self.cells:
            item = {
                "system": cell.system,
                "slice": cell.slice_name,
                "domain": cell.domain,
                "status": cell.status,
                "normalization_policy": cell.normalization_policy,
                "normalizer_version": cell.normalizer_version,
                "inference_context": cell.inference_context,
                "input_hashes": cell.input_hashes,
                "leakage_audit": cell.leakage_audit,
                "model_id": cell.model_id,
                "inference_mode": cell.inference_mode,
            }
            if cell.is_pending:
                item["pending_reason"] = cell.pending_reason
            else:
                assert cell.score is not None
                item.update(
                    n_segments=cell.n_segments,
                    n_recordings=cell.n_recordings,
                    wer_raw=cell.score.wer_raw,
                    cer_raw=cell.score.cer_raw,
                )
                for metric, point, ci in [
                    ("wer", cell.score.wer_normalized, cell.wer_ci),
                    ("cer", cell.score.cer_normalized, cell.cer_ci),
                ]:
                    item[f"{metric}_norm"] = {
                        "point": point,
                        "lo": ci.lower if ci else None,
                        "hi": ci.upper if ci else None,
                    }
            cells.append(item)
        return {
            "name": self.name,
            "title": self.title,
            "normalizer_version": self.run_context.normalizer_version,
            "run_context": self.run_context.to_dict(),
            "slices": list(self.slice_order),
            "systems": list(self.system_order),
            "cells": cells,
            "leakage_notes": list(self.leakage_notes),
        }


def _audit(
    system: SystemSpec,
    spec: ReportSpec,
    manifest: Manifest,
    name: str,
    base: Path,
    *,
    scored: bool,
    registry: LicenseRegistry,
) -> dict:
    if system.training_audit != "required":
        return {"status": system.training_audit, "scope": "upstream training", "checked_kinds": []}
    paths = system.train_manifests or spec.train_manifests
    missing = [path for path in paths if not (base / path).is_file()]
    if not paths or missing:
        if scored:
            raise ValueError(
                f"scored fine-tuned system {system.name!r} requires every declared training "
                f"manifest; missing {missing or 'declaration'}"
            )
        return {
            "status": "pending",
            "scope": "declared training mix",
            "upstream_base_training": "unknown",
            "missing_train_manifests": missing or ["unknown"],
        }
    if not scored:
        return {
            "status": "pending",
            "scope": "declared training mix",
            "upstream_base_training": "unknown",
        }
    from shono.data.leakage import require_no_leakage

    loaded = [_load_manifest(base / path) for path in paths]
    trains = [manifest for manifest, _ in loaded]
    for training in trains:
        training.validate_against(registry)
    audit = require_no_leakage(
        trains,
        manifest,
        text_decisions=system.text_decisions.get(name),
        audio_root=None if spec.audio_root is None else base / spec.audio_root,
    )
    return {
        "status": "reviewed",
        "scope": "declared training mix",
        "upstream_base_training": "unknown",
        **asdict(audit),
        "training_input_hashes": {
            path: digest for path, (_, digest) in zip(paths, loaded, strict=True)
        },
    }


def build_report(
    spec: ReportSpec,
    base_dir: str | Path,
    registry: LicenseRegistry,
    *,
    command: str | None = None,
    run_context: RunContext | None = None,
) -> Report:
    """Missing inputs remain pending; existing incomplete or mismatched inputs fail."""
    spec.__post_init__()
    for item in spec.slices:
        item.__post_init__()
    for system in spec.systems:
        system.__post_init__()
    base = Path(base_dir)
    context = run_context or RunContext.capture(
        spec.seed, {"report": spec.name, "n_resamples": spec.n_resamples}, command=command
    )
    manifests = {}
    hashes = {}
    domains = {}
    license_blocks = {}
    paths = [str(path) for path in source_files(base)]
    if spec.source_path:
        paths.append(spec.source_path)
    for item in spec.slices:
        path = base / item.manifest
        paths.append(str(path))
        if path.exists():
            manifest, digest = _load_manifest(path)
            license = registry.require(manifest.source)
            if license.status == "excluded":
                license_blocks[item.name] = "source permission unconfirmed or excluded"
            else:
                manifest.validate_against(registry)
            if item.domain and manifest.domain != item.domain:
                raise ValueError(
                    f"manifest domain {manifest.domain!r} differs from slice {item.name!r} "
                    f"domain {item.domain!r}"
                )
            manifests[item.name] = manifest
            hashes[item.name] = digest
            domains[item.name] = manifest.domain
        else:
            domains[item.name] = item.domain
    cells = []
    for system in spec.systems:
        paths.extend(str(base / path) for path in system.train_manifests)
        for item in spec.slices:
            manifest = manifests.get(item.name)
            pred_path = system.predictions.get(item.name)
            path = base / pred_path if pred_path else None
            if path:
                paths.append(str(path))
            blocked = system.pending_slices.get(item.name) or license_blocks.get(item.name)
            if blocked and path is not None and path.exists():
                raise ValueError(
                    f"unsupported cell {system.name!r}/{item.name!r} has predictions: {blocked}"
                )
            available = manifest is not None and path is not None and path.exists() and not blocked
            policy, version = _policy_for(domains[item.name])
            audit = _audit(
                system, spec, manifest, item.name, base, scored=available, registry=registry
            )
            input_hashes = {
                "manifest": hashes.get(item.name, "unknown"),
                "predictions": "unknown",
                "spec": spec.source_sha256,
            }
            if not available:
                cells.append(
                    SliceScore(
                        system.name,
                        item.name,
                        domains[item.name],
                        "pending",
                        normalization_policy=policy,
                        normalizer_version=version,
                        input_hashes=input_hashes,
                        leakage_audit=audit,
                        pending_reason=blocked
                        or (
                            "manifest unavailable"
                            if manifest is None
                            else "predictions unavailable"
                        ),
                        model_id=system.model_id,
                        inference_mode=system.inference_modes.get(item.name, "unknown"),
                    )
                )
                continue
            predictions = Predictions.from_jsonl(path)
            if predictions.system != system.name:
                raise ValueError(
                    f"prediction system {predictions.system!r} differs from spec system "
                    f"{system.name!r}"
                )
            if predictions.run_context:
                config = predictions.run_context.get("config", {})
                if not isinstance(config, dict):
                    raise ValueError("prediction context config must be an object")
                recorded = config.get("model_id", config.get("model"))
                if recorded and system.model_id != "unknown" and recorded != system.model_id:
                    raise ValueError(
                        f"prediction model {recorded!r} differs from spec model {system.model_id!r}"
                    )
            input_hashes["predictions"] = predictions.source_sha256
            cells.append(
                score_slice(
                    manifest,
                    predictions,
                    seed=spec.seed,
                    n_resamples=spec.n_resamples,
                    slice_name=item.name,
                    input_hashes=input_hashes,
                    leakage_audit=audit,
                    model_id=system.model_id,
                    inference_mode=system.inference_modes.get(item.name, "unknown"),
                )
            )
    return Report(
        spec.name,
        spec.title,
        context,
        tuple(cells),
        tuple(s.name for s in spec.slices),
        tuple(s.name for s in spec.systems),
        registry.render_table(),
        input_paths=tuple(paths),
    )


def write_report(report: Report, out_dir: str | Path) -> tuple[Path, Path]:
    """Render both files before staging and replacing their destinations."""
    markdown = report.render_markdown()
    payload = (
        json.dumps(report.to_json_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", report.name):
        raise ValueError("unsafe report name")
    out = Path(out_dir)
    destinations = validate_output_destinations(
        (out / f"{report.name}.md", out / f"{report.name}.json"),
        protected_paths=report.input_paths,
        output_root=out,
    )
    out.mkdir(parents=True, exist_ok=True)
    staged = []
    try:
        for text in (markdown, payload):
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=out, delete=False
            ) as handle:
                handle.write(text)
                staged.append(Path(handle.name))
        for temporary, destination in zip(staged, destinations, strict=True):
            os.replace(temporary, destination)
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)
    return destinations
