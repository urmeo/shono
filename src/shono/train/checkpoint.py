"""Checkpoint-resume — the logic that lets a run survive a killed session.

Kaggle sessions are capped (~12 h) and can be pre-empted, so every training run
must be resumable: on restart it finds the latest *complete* checkpoint and
continues from it. This is decision D-0001's non-negotiable, and it is pure
filesystem logic — no torch — so it is fully unit-tested here rather than
discovered to be broken three hours into a real run.

Hugging Face ``Trainer`` writes ``checkpoint-<global_step>/`` directories and
finalizes each with ``trainer_state.json``; a directory missing that file is a
half-written checkpoint (the session died mid-save) and must be skipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")
_COMPLETION_MARKER = "trainer_state.json"


def _step_of(path: Path) -> int | None:
    m = _CKPT_RE.match(path.name)
    return int(m.group(1)) if m else None


def is_complete_checkpoint(path: Path) -> bool:
    """A checkpoint is resumable only if its completion marker was written."""
    return path.is_dir() and _step_of(path) is not None and (path / _COMPLETION_MARKER).is_file()


def list_checkpoints(output_dir: str | Path) -> list[Path]:
    """All complete ``checkpoint-N`` directories under ``output_dir``, oldest step first."""
    out = Path(output_dir)
    if not out.is_dir():
        return []
    ckpts = [p for p in out.iterdir() if is_complete_checkpoint(p)]
    return sorted(ckpts, key=lambda p: _step_of(p) or 0)


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    """The highest-step complete checkpoint, or ``None`` if there is none."""
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
        return "starting a fresh run (no complete checkpoint found)"


def decide_resume(output_dir: str | Path, *, enabled: bool = True) -> ResumeDecision:
    """Decide whether to resume, and from where.

    ``enabled=False`` (e.g. an intentional fresh restart) forces a fresh run even
    when checkpoints exist. Otherwise, resume from the latest complete checkpoint
    if one exists.
    """
    if not enabled:
        return ResumeDecision(resume=False, checkpoint=None, global_step=0)
    latest = find_latest_checkpoint(output_dir)
    if latest is None:
        return ResumeDecision(resume=False, checkpoint=None, global_step=0)
    return ResumeDecision(resume=True, checkpoint=latest, global_step=_step_of(latest) or 0)
