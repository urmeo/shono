"""Run provenance — seed, config, and environment captured with every result.

The rule is absolute: a run without a recorded seed, config, and environment did
not happen. :class:`RunContext` is the vehicle. Every generated report and every
training experiment embeds one, so any number can be traced back to the exact
code, data, and machine that produced it.

The capture is made deterministic for its own tests by injecting the clock and
the git accessor — the component that certifies reproducibility is itself
reproducible.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from shono.eval.normalize import NORMALIZER_VERSION

# Packages whose exact versions change results and belong in every record.
# Absent ones are simply omitted (the local Mac has no torch/CUDA stack).
_TRACKED_PACKAGES = (
    "shono",
    "jiwer",
    "bnunicodenormalizer",
    "torch",
    "transformers",
    "faster-whisper",
    "ctranslate2",
    "pyannote.audio",
    "librosa",
)


@dataclass(frozen=True)
class GitState:
    """The commit a run was launched from, and whether the tree was dirty."""

    sha: str | None
    dirty: bool


def _read_git_state(repo: Path) -> GitState:
    def _run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=repo,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    sha = _run("rev-parse", "HEAD")
    status = _run("status", "--porcelain")
    return GitState(sha=sha, dirty=bool(status))


def _package_versions(names: tuple[str, ...]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


@dataclass(frozen=True)
class RunContext:
    """Everything needed to reproduce and trust a run's numbers."""

    seed: int
    config: Mapping[str, object]
    timestamp: str
    python: str
    platform: str
    packages: Mapping[str, str]
    normalizer_version: str
    git: GitState
    command: str | None = None
    extra: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def capture(
        cls,
        seed: int,
        config: Mapping[str, object],
        *,
        command: str | None = None,
        extra: Mapping[str, object] | None = None,
        repo: str | Path | None = None,
        now: Callable[[], datetime] | None = None,
        git_state: GitState | None = None,
        packages: tuple[str, ...] = _TRACKED_PACKAGES,
    ) -> RunContext:
        """Snapshot the current run's seed, config, and environment.

        ``now`` and ``git_state`` are injectable so this capture is itself
        deterministically testable; in production both default to the live
        clock and the git tree rooted at ``repo`` (or the caller's file tree).
        """
        clock = now or (lambda: datetime.now(UTC))
        root = Path(repo) if repo is not None else Path(__file__).resolve().parent
        git = git_state if git_state is not None else _read_git_state(root)
        return cls(
            seed=seed,
            config=dict(config),
            timestamp=clock().astimezone(UTC).isoformat(),
            python=sys.version.split()[0],
            platform=platform.platform(),
            packages=_package_versions(packages),
            normalizer_version=NORMALIZER_VERSION,
            git=git,
            command=command,
            extra=dict(extra or {}),
        )

    def to_dict(self) -> dict[str, object]:
        """A JSON-serializable view, suitable for a report or experiment header."""
        return {
            "seed": self.seed,
            "config": dict(self.config),
            "timestamp": self.timestamp,
            "python": self.python,
            "platform": self.platform,
            "packages": dict(self.packages),
            "normalizer_version": self.normalizer_version,
            "git": {"sha": self.git.sha, "dirty": self.git.dirty},
            "command": self.command,
            "extra": dict(self.extra),
        }


def seed_everything(seed: int) -> int:
    """Seed every RNG that affects a run, and return the seed for recording.

    Seeds Python's ``random`` always; ``numpy`` and ``torch`` when importable
    (they are absent on the local Mac but present in the Kaggle training image).
    Deterministic cuDNN is left to the training notebook, which owns the GPU.
    """
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed
