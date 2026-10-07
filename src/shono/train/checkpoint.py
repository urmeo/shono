"""Select the latest valid saved-step marker; weight and optimizer completeness is unverified."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")
_COMPLETION_MARKER = "trainer_state.json"


def _step_of(path: Path) -> int | None:
    m = _CKPT_RE.match(path.name)
    return int(m.group(1)) if m else None


def is_complete_checkpoint(path: Path) -> bool:
    """A candidate has valid saved-step metadata, not verified optimizer/weight bytes."""
    if not path.is_dir() or _step_of(path) is None or not (path / _COMPLETION_MARKER).is_file():
        return False
    try:
        state = json.loads((path / _COMPLETION_MARKER).read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return False
    return (
        isinstance(state, dict)
        and type(state.get("global_step")) is int
        and state["global_step"] == _step_of(path)
    )


def list_checkpoints(output_dir: str | Path) -> list[Path]:
    """Saved-step candidates in ascending step order."""
    out = Path(output_dir)
    if not out.is_dir():
        return []
    ckpts = [p for p in out.iterdir() if is_complete_checkpoint(p)]
    return sorted(ckpts, key=lambda p: _step_of(p) or 0)


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    """Return the highest valid saved-step candidate, or None."""
    ckpts = list_checkpoints(output_dir)
    return ckpts[-1] if ckpts else None


@dataclass(frozen=True)
class ResumeDecision:
    """What a restart should do: resume from a checkpoint, or start fresh."""

    resume: bool
    checkpoint: Path | None
    global_step: int

    @property
    def summary(self) -> str:
        if self.resume:
            return f"resuming from {self.checkpoint} at global step {self.global_step}"
        return "starting a fresh run (no saved-step checkpoint found)"


def decide_resume(output_dir: str | Path, *, enabled: bool = True) -> ResumeDecision:
    """Select the latest saved-step candidate unless explicitly disabled."""
    if not enabled:
        return ResumeDecision(resume=False, checkpoint=None, global_step=0)
    latest = find_latest_checkpoint(output_dir)
    if latest is None:
        return ResumeDecision(resume=False, checkpoint=None, global_step=0)
    return ResumeDecision(resume=True, checkpoint=latest, global_step=_step_of(latest) or 0)
