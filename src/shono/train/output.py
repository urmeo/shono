"""Protect inputs and repository sources from training artifact writes."""

from pathlib import Path

from shono.data.audio_paths import resolved_path


def _within(path: Path, parent: Path) -> bool:
    child = tuple(p.casefold() for p in path.parts)
    ancestor = tuple(p.casefold() for p in parent.parts)
    return child[: len(ancestor)] == ancestor


def validate_training_output(output_dir: str | Path, audio_root: str | Path) -> Path:
    out, audio = resolved_path(output_dir), resolved_path(audio_root)
    if _within(out, audio) or _within(audio, out):
        raise ValueError("output_dir and input audio_root must be disjoint")
    repo = Path(__file__).resolve().parents[3]
    if (repo / "src" / "shono").is_dir():
        if _within(repo, out) or any(
            _within(out, repo / name)
            for name in (
                "src",
                "docs",
                "data",
                "tests",
                "notebooks",
                ".github",
                "scripts",
                "reports",
            )
        ):
            raise ValueError("output_dir cannot overwrite repository source subtrees")
    if out.exists():
        if not out.is_dir():
            raise ValueError("output_dir must be a directory")
        for path in out.rglob("*"):
            if path.is_symlink() or (path.is_file() and path.stat().st_nlink > 1):
                raise ValueError(f"output artifact is a symlink or hardlink: {path.name}")
    identity = out / "training-inputs.json"
    if not _within(resolved_path(identity), out):
        raise ValueError("training identity path escapes output_dir")
    return out
