"""Capture configuration and environment metadata without credentials."""

from __future__ import annotations

import math
import platform
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from shono.eval.normalize import NORMALIZER_VERSION

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
    "numpy",
    "soundfile",
    "accelerate",
    "bitsandbytes",
    "datasets",
    "deepgram-sdk",
    "google-cloud-speech",
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
            versions[name] = "unknown"
    return versions


def _credential_key(key: str) -> bool:
    name = key.lower().lstrip("-").replace("-", "_")
    return name in {
        "token",
        "api_key",
        "apikey",
        "deepgram_key",
        "google_application_credentials",
        "password",
        "secret",
        "secret_key",
        "credentials",
        "authorization",
    } or name.endswith(("_token", "_api_key", "_password", "_secret"))


def safe_command(command: str | None) -> str | None:
    """Redact named credential flags and environment assignments, not ordinary text."""
    if command is None:
        return None
    if not isinstance(command, str):
        raise ValueError("command must be a string or None")
    try:
        words = shlex.split(command)
    except ValueError as exc:
        raise ValueError("command must use valid shell quoting") from exc
    redact_next = False
    result = []
    for word in words:
        if redact_next:
            result.append("[redacted]")
            redact_next = False
        elif "=" in word and _credential_key(word.split("=", 1)[0]):
            result.append(word.split("=", 1)[0] + "=[redacted]")
        elif word.startswith("-") and _credential_key(word):
            result.append(word)
            redact_next = True
        else:
            result.append(word)
    return shlex.join(result)


def safe_metadata(value: object) -> object:
    """Copy finite JSON metadata and redact credential fields recursively."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata numbers must be finite")
        return value
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("metadata keys must be strings")
            if key == "command" and (isinstance(item, str) or item is None):
                result[key] = safe_command(item)
            else:
                result[key] = "[redacted]" if _credential_key(key) and item else safe_metadata(item)
        return result
    if isinstance(value, (list, tuple)):
        return [safe_metadata(item) for item in value]
    raise ValueError(f"metadata must contain JSON values, got {type(value).__name__}")


def validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")


@dataclass(frozen=True)
class RunContext:
    """Recorded configuration, environment and code state for a run."""

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
        """Capture metadata; clock and Git state can be supplied for tests."""
        validate_seed(seed)
        if not isinstance(config, Mapping) or (
            extra is not None and not isinstance(extra, Mapping)
        ):
            raise ValueError("config and extra must be mappings")
        if command is not None and not isinstance(command, str):
            raise ValueError("command must be a string or None")
        clock = now or (lambda: datetime.now(UTC))
        root = Path(repo) if repo is not None else Path(__file__).resolve().parent
        git = git_state if git_state is not None else _read_git_state(root)
        moment = clock()
        if not isinstance(moment, datetime) or moment.tzinfo is None or moment.utcoffset() is None:
            raise ValueError("capture clock must return an aware datetime")
        return cls(
            seed=seed,
            config=safe_metadata(config),
            timestamp=moment.astimezone(UTC).isoformat(),
            python=sys.version.split()[0],
            platform=platform.platform(),
            packages=_package_versions(packages),
            normalizer_version=NORMALIZER_VERSION,
            git=git,
            command=safe_command(command),
            extra=safe_metadata(extra or {}),
        )

    def to_dict(self) -> dict[str, object]:
        """A JSON-serializable view, suitable for a report or experiment header."""
        return {
            "seed": self.seed,
            "config": safe_metadata(self.config),
            "timestamp": self.timestamp,
            "python": self.python,
            "platform": self.platform,
            "packages": dict(self.packages),
            "normalizer_version": self.normalizer_version,
            "git": {"sha": self.git.sha or "unknown", "dirty": self.git.dirty},
            "command": safe_command(self.command),
            "extra": safe_metadata(self.extra),
        }


def seed_everything(seed: int) -> int:
    """Seed available RNGs; GPU operations may still be nondeterministic."""
    import random

    validate_seed(seed)
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
