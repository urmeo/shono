"""Define model identities and evaluation report cells."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

BASE_MODEL_ID = "bengaliAI/tugstugi_bengaliai-asr_whisper-medium"
BASE_SYSTEM = "tugstugi whisper-medium (base)"
OUR_SYSTEM = "shono whisper-medium (fine-tuned)"
TRAIN_MANIFESTS = (
    "data/cv-bn-train.manifest.jsonl",
    "data/slr53-train.manifest.jsonl",
    "data/mucs-cs-train.manifest.jsonl",
)

_SYSTEMS = {
    "tugstugi-medium": (BASE_SYSTEM, BASE_MODEL_ID),
    "shono-medium": (OUR_SYSTEM, "unknown"),
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
        ("tugstugi-medium",),
        ("cv-bn-test", "fleurs-bn-test", "bengali-loop-sample"),
    ),
    "longform": (
        ("shono-medium", "tugstugi-medium"),
        ("bengali-loop-test", "lecture-podcast-eval"),
    ),
    "codeswitch": (
        ("shono-medium", "tugstugi-medium"),
        ("mucs-cs-test", "cs-mini-eval"),
    ),
    "full": (
        ("shono-medium", "tugstugi-medium"),
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
        if self.inference_mode not in {"bounded-short", "long-form-vad"}:
            raise ValueError(f"unknown inference mode {self.inference_mode!r}")


def run_matrix(*, fine_tuned_model: str = "unknown") -> tuple[RunEntry, ...]:
    """Return the union of all declared report cells, without duplicate runs."""
    if not isinstance(fine_tuned_model, str) or not fine_tuned_model.strip():
        raise ValueError("fine_tuned_model must be a nonempty string or 'unknown'")
    pairs = dict.fromkeys(
        (slug, name) for systems, slices in _REPORTS.values() for slug in systems for name in slices
    )
    entries = []
    for slug, name in pairs:
        system, model_id = _SYSTEMS[slug]
        domain = _SLICES[name]
        entries.append(
            RunEntry(
                slug=slug,
                system=system,
                slice_name=name,
                manifest=f"data/{name}.manifest.jsonl",
                predictions=f"outputs/{slug}__{name}.jsonl",
                domain=domain,
                model_id=fine_tuned_model if slug == "shono-medium" else model_id,
                inference_mode="long-form-vad" if domain == "long-form" else "bounded-short",
            )
        )
    return tuple(entries)


def report_spec(name: str) -> dict:
    """Generate a report spec from the declared models and slices."""
    if name not in _REPORTS:
        raise ValueError(f"unknown report {name!r}")
    systems, slices = _REPORTS[name]
    matrix = {(entry.slug, entry.slice_name): entry for entry in run_matrix()}
    output = []
    for slug in systems:
        label, model = _SYSTEMS[slug]
        entries = [matrix[slug, slice_name] for slice_name in slices]
        output.append(
            {
                "name": label,
                "model_id": model,
                "model_kind": "fine_tuned" if slug == "shono-medium" else "pretrained",
                "predictions": {entry.slice_name: entry.predictions for entry in entries},
                "inference_modes": {entry.slice_name: entry.inference_mode for entry in entries},
                "training_audit": "required" if slug == "shono-medium" else "unknown",
                "train_manifests": list(TRAIN_MANIFESTS) if slug == "shono-medium" else [],
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
