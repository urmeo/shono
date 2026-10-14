"""Identify the base checkpoint before loading training inputs."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


def local_checkpoint(model_id: str, revision: str | None) -> Path | None:
    path = Path(model_id).expanduser()
    if path.exists():
        if not path.is_dir():
            raise ValueError("local model_id must be a checkpoint directory")
        if revision is not None:
            raise ValueError("local checkpoints use content hashes, not model_revision")
        return path.resolve()
    if path.is_absolute() or model_id.startswith(("./", "../", "~")):
        raise ValueError("local model_id checkpoint directory does not exist")
    if "://" in model_id:
        raise ValueError("model_id must be a Hub repository ID or local checkpoint directory")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("model_revision must be a full lowercase 40-character Hub commit SHA")
    return None


def checkpoint_identity(model_id: str, revision: str | None) -> dict:
    path = local_checkpoint(model_id, revision)
    if path is None:
        return {"source": "hub", "model_id": model_id, "revision": revision}
    files = {}
    try:
        for file in sorted(path.rglob("*")):
            if file.is_dir():
                if file.is_symlink():
                    raise ValueError("local checkpoint cannot contain symlinked directories")
                continue
            if not file.is_file():
                raise ValueError(f"local checkpoint contains an unreadable file: {file.name}")
            with file.open("rb") as stream:
                files[file.relative_to(path).as_posix()] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    except OSError as exc:
        raise ValueError("cannot hash local checkpoint files") from exc
    if not files:
        raise ValueError("local checkpoint directory is empty")
    return {"source": "local", "path": str(path), "files_sha256": files}
