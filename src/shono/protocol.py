"""Shared system identities, report cells and explicitly enabled run orchestration."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from shono.data.license import LicenseRegistry
from shono.data.manifest import Manifest
from shono.output_paths import source_files, validate_output_destinations
from shono.provenance import RunContext, safe_metadata, validate_seed

BASE_MODEL_ID = "bengaliAI/tugstugi_bengaliai-asr_whisper-medium"
BASE_SYSTEM = "tugstugi whisper-medium (base)"
OUR_SYSTEM = "shono whisper-medium (fine-tuned)"
TRAIN_MANIFESTS = (
    "data/cv-bn-train.manifest.jsonl",
    "data/slr53-train.manifest.jsonl",
    "data/mucs-cs-train.manifest.jsonl",
)

_SYSTEMS = {
    "large-v3": ("whisper-large-v3 (zero-shot)", "openai/whisper-large-v3", "local"),
    "large-v3-turbo": (
        "whisper-large-v3-turbo (zero-shot)",
        "openai/whisper-large-v3-turbo",
        "local",
    ),
    "tugstugi-medium": (BASE_SYSTEM, BASE_MODEL_ID, "local"),
    "shono-medium": (OUR_SYSTEM, "unknown", "local"),
    "google-chirp": ("Google Chirp (bn-BD)", "chirp_2", "google"),
    "deepgram-nova3": ("Deepgram Nova-3 (bn)", "nova-3", "deepgram"),
}
_SLICES = {
    "cv-bn-test": "read",
    "fleurs-bn-test": "read",
    "bengali-loop-sample": "long-form",
    "bengali-loop-test": "long-form",
    "lecture-podcast-eval": "long-form",
    "mucs-cs-test": "code-switch",
    "cs-mini-eval": "code-switch",
}
_REPORTS = {
    "baselines": (
        ("large-v3", "large-v3-turbo", "tugstugi-medium"),
        ("cv-bn-test", "fleurs-bn-test", "bengali-loop-sample"),
    ),
    "longform": (
        ("shono-medium", "tugstugi-medium", "large-v3"),
        ("bengali-loop-test", "lecture-podcast-eval"),
    ),
    "codeswitch": (
        ("shono-medium", "tugstugi-medium", "large-v3"),
        ("mucs-cs-test", "cs-mini-eval"),
    ),
    "full": (
        ("shono-medium", "tugstugi-medium", "google-chirp", "deepgram-nova3"),
        ("cv-bn-test", "bengali-loop-test", "mucs-cs-test"),
    ),
}


@dataclass(frozen=True)
class RunEntry:
    slug: str
    system: str
    slice_name: str
    manifest: str
    predictions: str
    domain: str
    model_id: str
    inference_mode: str
    provider: str
    optional: bool
    supported: bool = True
    pending_reason: str = ""

    def __post_init__(self) -> None:
        for name in ("slug", "system", "slice_name", "domain", "model_id", "inference_mode"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"run entry {name} must be a nonempty string")
        for name in ("manifest", "predictions"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"run entry {name} must be a path")
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts or "\\" in value:
                raise ValueError(f"run entry {name} must be a relative POSIX path")
        if (
            len(PurePosixPath(self.predictions).parts) != 2
            or PurePosixPath(self.predictions).parts[0] != "outputs"
        ):
            raise ValueError("predictions must be a file directly under outputs")
        if self.provider not in {"local", "google", "deepgram"}:
            raise ValueError("unknown run provider")
        if not isinstance(self.optional, bool) or not isinstance(self.supported, bool):
            raise ValueError("run entry flags must be boolean")
        if not isinstance(self.pending_reason, str):
            raise ValueError("pending_reason must be a string")


def run_matrix(*, fine_tuned_model: str = "unknown") -> tuple[RunEntry, ...]:
    """Return the union of all declared report cells, without duplicate runs."""
    if not isinstance(fine_tuned_model, str) or not fine_tuned_model.strip():
        raise ValueError("fine_tuned_model must be a nonempty string or 'unknown'")
    pairs = dict.fromkeys(
        (slug, name) for systems, slices in _REPORTS.values() for slug in systems for name in slices
    )
    entries = []
    for slug, name in pairs:
        system, model_id, provider = _SYSTEMS[slug]
        domain = _SLICES[name]
        mode = "long-form-vad" if domain == "long-form" and provider == "local" else "bounded-short"
        if provider != "local":
            mode = "provider-whole-recording" if domain == "long-form" else "provider-window"
        supported = not (provider == "google" and domain == "long-form")
        entries.append(
            RunEntry(
                slug,
                system,
                name,
                f"data/{name}.manifest.jsonl",
                f"outputs/{slug}__{name}.jsonl",
                domain,
                fine_tuned_model if slug == "shono-medium" else model_id,
                mode,
                provider,
                name in {"lecture-podcast-eval", "cs-mini-eval"},
                supported,
                "Google Recognize long-form protocol is unsupported" if not supported else "",
            )
        )
    return tuple(entries)


def report_spec(name: str) -> dict:
    """Generate the input spec from the same identities and cells as the drivers."""
    if name not in _REPORTS:
        raise ValueError(f"unknown report {name!r}")
    systems, slices = _REPORTS[name]
    matrix = {(entry.slug, entry.slice_name): entry for entry in run_matrix()}
    output = []
    for slug in systems:
        label, model, provider = _SYSTEMS[slug]
        entries = [matrix[slug, slice_name] for slice_name in slices]
        output.append(
            {
                "name": label,
                "model_id": model,
                "predictions": {entry.slice_name: entry.predictions for entry in entries},
                "inference_modes": {entry.slice_name: entry.inference_mode for entry in entries},
                "training_audit": "required" if slug == "shono-medium" else "unknown",
                "train_manifests": list(TRAIN_MANIFESTS) if slug == "shono-medium" else [],
                "pending_slices": {
                    entry.slice_name: entry.pending_reason
                    for entry in entries
                    if not entry.supported
                },
            }
        )
    return {
        "name": name,
        "title": f"Shono {name} results",
        "seed": 0,
        "n_resamples": 1000,
        "slices": [
            {
                "name": slice_name,
                "manifest": f"data/{slice_name}.manifest.jsonl",
                "domain": _SLICES[slice_name],
            }
            for slice_name in slices
        ],
        "systems": output,
    }


@dataclass(frozen=True)
class RunOutcome:
    entry: RunEntry
    status: str
    reason: str = ""


def execute_matrix(
    entries: Sequence[RunEntry],
    *,
    base_dir: str | Path,
    audio_root: str | Path,
    registry: LicenseRegistry,
    transcriber_for: Callable,
    duration_of: Callable | None = None,
    run_models: bool = False,
    run_apis: bool = False,
    allowances: Mapping[str, float] | None = None,
    allow_paid: bool = False,
    seed: int = 0,
    inference_settings: Mapping[str, Mapping] | None = None,
) -> tuple[RunOutcome, ...]:
    """Preflight all enabled inputs and provider totals before creating adapters."""
    from shono.api import BudgetGuard, preflight_manifest, run_over_manifest
    from shono.data.audio_paths import validate_audio_window

    validate_seed(seed)
    if any(not isinstance(flag, bool) for flag in (run_models, run_apis, allow_paid)):
        raise ValueError("run flags must be boolean")
    if not isinstance(entries, (list, tuple)) or not all(
        isinstance(entry, RunEntry) for entry in entries
    ):
        raise ValueError("entries must be a sequence of RunEntry")
    if len({(entry.slug, entry.slice_name) for entry in entries}) != len(entries):
        raise ValueError("duplicate run entries")
    if not callable(transcriber_for):
        raise ValueError("transcriber_for must be callable")
    if allowances is not None and not isinstance(allowances, Mapping):
        raise ValueError("allowances must be a provider mapping")
    declared = dict(allowances or {})
    if set(declared) - {"google", "deepgram"}:
        raise ValueError("unknown allowance provider")
    budgets = {
        provider: BudgetGuard(declared.get(provider, 0.0), allow_paid=allow_paid)
        for provider in ("google", "deepgram")
    }
    if inference_settings is not None and (
        not isinstance(inference_settings, Mapping)
        or any(not isinstance(value, Mapping) for value in inference_settings.values())
    ):
        raise ValueError("inference_settings must map system slugs to configuration objects")
    settings = safe_metadata(inference_settings or {})
    base = Path(base_dir)
    pending = {}
    planned = []
    totals = {"google": 0.0, "deepgram": 0.0}
    protected = list(source_files(base))
    for entry in entries:
        if not entry.supported:
            pending[entry] = entry.pending_reason
            continue
        if (entry.provider == "local" and not run_models) or (
            entry.provider != "local" and not run_apis
        ):
            pending[entry] = (
                "model runs disabled" if entry.provider == "local" else "API runs disabled"
            )
            continue
        if entry.model_id == "unknown":
            pending[entry] = "model checkpoint unknown"
            continue
        path = base / entry.manifest
        if not path.is_file():
            pending[entry] = "manifest unavailable"
            continue
        manifest = Manifest.from_jsonl(path)
        license = registry.require(manifest.source)
        if license.status == "excluded":
            pending[entry] = "source permission unconfirmed or excluded"
            continue
        manifest.validate_against(registry)
        if manifest.domain != entry.domain:
            raise ValueError(f"manifest domain differs from run entry {entry.slice_name!r}")
        audio_paths, hours = preflight_manifest(
            manifest, audio_root=audio_root, duration_of=duration_of
        )
        for audio_path, segment in zip(audio_paths, manifest.segments, strict=True):
            seconds = validate_audio_window(
                audio_path,
                segment.start_s,
                segment.duration_s,
                duration_of=duration_of,
                whole_file=True if entry.inference_mode == "long-form-vad" else None,
            )
            if entry.inference_mode == "bounded-short" and seconds > 30:
                raise ValueError("bounded-short entries require audio windows <= 30 s")
            if entry.provider == "google" and seconds >= 60:
                raise ValueError("Google Recognize windows must be < 60 s")
        protected.extend(audio_paths)
        protected.append(path)
        if entry.provider != "local":
            totals[entry.provider] += hours
        planned.append((entry, manifest))
    validate_output_destinations(
        (base / entry.predictions for entry, _ in planned),
        protected_paths=protected,
        output_root=base / "outputs",
    )
    for provider, hours in totals.items():
        budgets[provider].check(hours)
    completed = set()
    for entry, manifest in planned:
        configured = settings.get(
            f"{entry.slug}:{entry.inference_mode}", settings.get(entry.slug, {})
        )
        context = RunContext.capture(
            seed,
            {
                **settings.get(
                    f"{entry.slug}:{entry.inference_mode}", settings.get(entry.slug, {})
                ),
                "model_id": entry.model_id,
                "model_revision": "unknown",
                "provider": entry.provider,
                "inference_mode": entry.inference_mode,
                "language": configured.get("language", "bn"),
                "decoding": configured.get("decoding", "unknown"),
            },
            command="execute_matrix",
            repo=base,
            extra={
                "manifest_sha256": hashlib.sha256((base / entry.manifest).read_bytes()).hexdigest()
            },
        )
        predictions = run_over_manifest(
            transcriber_for(entry),
            manifest,
            entry.system,
            audio_root=audio_root,
            budget=budgets.get(entry.provider),
            duration_of=duration_of,
            run_context=context,
        )
        destination = base / entry.predictions
        destination.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_jsonl(destination, protected_paths=protected)
        completed.add(entry)
    return tuple(
        RunOutcome(entry, "completed" if entry in completed else "pending", pending.get(entry, ""))
        for entry in entries
    )
