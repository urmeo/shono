"""Commercial ASR APIs behind one interface, with a hard $0-spend guard.

Every API (Google Chirp, Deepgram, …) implements the same :class:`ApiTranscriber`
protocol, so the benchmark treats them as interchangeable systems and — via
:func:`run_over_manifest` — turns each into a :class:`Predictions` set that the
existing report scores against ours and the base model. No parallel comparison
path; the APIs are just more columns.

:class:`BudgetGuard` enforces the project's rule: benchmark within free credits
only. A run whose audio exceeds the provider's free allowance is refused unless
paid spend is *explicitly* opted into — the point where a human L-gate belongs.
Tests drive all of this with a fake transcriber; no adapter here makes a real
call until its lazy client is constructed with real credentials.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from shono.data.manifest import Manifest
from shono.eval.report import Predictions
from shono.provenance import RunContext


@runtime_checkable
class ApiTranscriber(Protocol):
    """Transcribe one audio span. ``start_s``/``duration_s`` select a window for long-form."""

    def transcribe(
        self, audio_path: str, start_s: float | None = None, duration_s: float | None = None
    ) -> str: ...


class BudgetError(RuntimeError):
    """Raised when a benchmark run would spend money without an explicit opt-in."""


@dataclass(frozen=True)
class BudgetGuard:
    """Refuse runs beyond a provider's free allowance unless paid spend is opted into."""

    free_hours: float
    allow_paid: bool = False

    def check(self, hours: float) -> None:
        if hours > self.free_hours and not self.allow_paid:
            raise BudgetError(
                f"run needs {hours:.2f} h of transcription but only {self.free_hours:.2f} h "
                "are within the free allowance; paid spend requires an explicit decision "
                "(allow_paid=True) — do not exceed free credits without one"
            )


def run_over_manifest(
    transcriber: ApiTranscriber,
    manifest: Manifest,
    system: str,
    *,
    audio_root: str | Path = ".",
    budget: BudgetGuard | None = None,
    run_context: RunContext | None = None,
) -> Predictions:
    """Transcribe every segment of ``manifest`` with ``transcriber`` into a Predictions set.

    If ``budget`` is given, the run is checked against it first — the whole
    manifest is refused before a single paid call if it would exceed free credits.
    """
    if budget is not None:
        budget.check(manifest.total_hours())
    root = Path(audio_root)
    hypotheses = {
        seg.id: transcriber.transcribe(str(root / seg.audio), seg.start_s, seg.duration_s)
        for seg in manifest.segments
    }
    return Predictions(
        system=system,
        manifest=manifest.name,
        hypotheses=hypotheses,
        run_context=run_context.to_dict() if run_context else None,
    )
