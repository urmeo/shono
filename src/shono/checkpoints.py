"""Hash the files used by a local model checkpoint."""

from __future__ import annotations

import hashlib
from pathlib import Path


def hash_checkpoint(path: Path) -> dict:
    path = path.resolve()
    files = {}
    try:
        directories = [path]
        checkpoint_files = []
        while directories:
            for entry in directories.pop().iterdir():
                if entry.is_dir():
                    if entry.is_symlink():
                        raise ValueError("local checkpoint cannot contain symlinked directories")
                    directories.append(entry)
                elif entry.is_file():
                    checkpoint_files.append(entry)
                else:
                    raise ValueError(f"local checkpoint contains an unreadable file: {entry.name}")
        for file in sorted(checkpoint_files):
            with file.open("rb") as stream:
                files[file.relative_to(path).as_posix()] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    except OSError as exc:
        raise ValueError("cannot hash local checkpoint files") from exc
    if not files:
        raise ValueError("local checkpoint directory is empty")
    return {"source": "local", "path": str(path), "files_sha256": files}
