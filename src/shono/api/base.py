"""Run validated audio manifests with a cumulative declared-duration allowance."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Protocol, runtime_checkable

from shono.data.audio_paths import validate_audio_window, validate_manifest_audio
from shono.data.manifest import Manifest
from shono.eval.report import Predictions
from shono.provenance import RunContext, safe_metadata
from shono.transcribe._validation import finite_real, finite_sum


@runtime_checkable
class ApiTranscriber(Protocol):
    def transcribe(
        self, audio_path: str, start_s: float | None = None, duration_s: float | None = None
    ) -> str: ...


class BudgetError(RuntimeError):
    """The declared provider allowance would be exceeded."""


@dataclass
class BudgetGuard:
    """Reserve declared audio hours; account credit and actual charges remain unknown."""

    free_hours: float = 0.0
    allow_paid: bool = False
    _reservations: list[float] = field(default_factory=list, init=False, repr=False)
    _lock: object = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        finite_real(self.free_hours, "free_hours", minimum=0)
        if not isinstance(self.allow_paid, bool):
            raise ValueError("allow_paid must be boolean")

    @property
    def used_hours(self) -> float:
        with self._lock:
            return finite_sum(self._reservations, "reserved hours")

    def _checked(self, hours: float) -> float:
        requested = finite_real(hours, "requested hours", minimum=0)
        allowance = finite_real(self.free_hours, "free_hours", minimum=0)
        if not isinstance(self.allow_paid, bool):
            raise ValueError("allow_paid must be boolean")
        total = finite_sum([*self._reservations, requested], "cumulative requested hours")
        if total > allowance and not self.allow_paid:
            raise BudgetError(
                f"cumulative run needs {total:.3f} h but declared allowance is {allowance:.3f} h; "
                "paid spend requires an explicit decision (allow_paid=True)"
            )
        return requested

    def check(self, hours: float) -> None:
        """Preflight a workload against the remaining allowance without reserving it."""
        with self._lock:
            self._checked(hours)

    def reserve(self, hours: float) -> None:
        """Consume a workload before calls; failed requests do not refund it."""
        with self._lock:
            self._reservations.append(self._checked(hours))


def preflight_manifest(
    manifest: Manifest,
    *,
    audio_root: str | Path = ".",
    duration_of: Callable[[Path], float] | None = None,
) -> tuple[tuple[Path, ...], float]:
    """Validate all selected files/windows and return paths plus actual audio hours."""
    paths = validate_manifest_audio(manifest, audio_root, duration_of=duration_of)
    seconds = [
        validate_audio_window(path, seg.start_s, seg.duration_s, duration_of=duration_of)
        for path, seg in zip(paths, manifest.segments, strict=True)
    ]
    total = finite_sum(seconds, "manifest audio workload")
    return paths, total / 3600


def run_over_manifest(
    transcriber: ApiTranscriber,
    manifest: Manifest,
    system: str,
    *,
    audio_root: str | Path = ".",
    budget: BudgetGuard | None = None,
    run_context: RunContext | None = None,
    duration_of: Callable[[Path], float] | None = None,
) -> Predictions:
    """Preflight and reserve the entire manifest before invoking its transcriber."""
    if not isinstance(system, str) or not system.strip():
        raise ValueError("system label must be non-empty text")
    paths, hours = preflight_manifest(manifest, audio_root=audio_root, duration_of=duration_of)
    if budget is None and getattr(transcriber, "requires_budget", False):
        budget = BudgetGuard()
    if budget is not None:
        budget.reserve(hours)
    hypotheses: dict[str, str] = {}
    for path, seg in zip(paths, manifest.segments, strict=True):
        text = transcriber.transcribe(str(path), seg.start_s, seg.duration_s)
        if not isinstance(text, str):
            raise ValueError(f"transcriber returned non-text for segment {seg.id!r}")
        hypotheses[seg.id] = text
    context = run_context.to_dict() if run_context else None
    observer = getattr(transcriber, "observed_runtime_context", None)
    if callable(observer):
        context = context or {}
        context["observed_runtime"] = safe_metadata(observer())
    return Predictions(
        system,
        manifest.name,
        hypotheses,
        run_context=context,
        input_paths=tuple(str(path) for path in paths),
    )
