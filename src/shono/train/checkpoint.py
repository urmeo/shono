"""Find restart candidates from saved training-step metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from shono.validation import json_object

_CKPT_RE = re.compile(r"checkpoint-(\d+)")
_STATE_FILE = "trainer_state.json"


def _step_of(path: Path) -> int | None:
    match = _CKPT_RE.fullmatch(path.name)
    return int(match.group(1)) if match else None


def _has_saved_step(path: Path) -> bool:
    step = _step_of(path)
    marker = path / _STATE_FILE
    if not path.is_dir() or step is None or not marker.is_file():
        return False
    try:
        state = json_object(marker.read_text(encoding="utf-8"), str(marker))
    except (ValueError, OSError):
        return False
    return type(state.get("global_step")) is int and state["global_step"] == step


def list_checkpoints(output_dir: str | Path) -> list[Path]:
    """Saved-step candidates in ascending step order."""
    directory = Path(output_dir)
    if not directory.is_dir():
        return []
    checkpoints = [path for path in directory.iterdir() if _has_saved_step(path)]
    return sorted(checkpoints, key=lambda path: _step_of(path) or 0)


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    """Return the latest saved-step candidate; model files are validated on load."""
    checkpoints = list_checkpoints(output_dir)
    return checkpoints[-1] if checkpoints else None


@dataclass(frozen=True)
class ResumeDecision:
    """Whether to resume from a checkpoint or start fresh."""

    resume: bool
    checkpoint: Path | None
    global_step: int

    @property
    def summary(self) -> str:
        if self.resume:
            return f"resuming from {self.checkpoint} at global step {self.global_step}"
        return "starting a fresh run"


def decide_resume(output_dir: str | Path, *, enabled: bool = True) -> ResumeDecision:
    """Select the latest saved-step candidate unless explicitly disabled."""
    if not enabled:
        return ResumeDecision(resume=False, checkpoint=None, global_step=0)
    latest = find_latest_checkpoint(output_dir)
    if latest is None:
        return ResumeDecision(resume=False, checkpoint=None, global_step=0)
    return ResumeDecision(resume=True, checkpoint=latest, global_step=_step_of(latest) or 0)
