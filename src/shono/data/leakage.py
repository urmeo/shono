"""Train/evaluation overlap checks with explicit text-collision review."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from shono.data.audio_paths import file_sha256, validate_audio_paths
from shono.data.manifest import Manifest
from shono.eval.normalize import normalize
from shono.validation import text_value

_KINDS = ("id", "text", "audio")


@dataclass(frozen=True)
class Overlap:
    kind: str
    key: str
    train_segments: tuple[str, ...]
    eval_segments: tuple[str, ...]


@dataclass(frozen=True)
class TextDecision:
    key: str
    reason: str
    train_segments: tuple[str, ...]
    eval_segments: tuple[str, ...]


@dataclass(frozen=True)
class LeakageReport:
    train_names: tuple[str, ...]
    eval_name: str
    overlaps: tuple[Overlap, ...]
    checked_kinds: tuple[str, ...]
    audio_coverage: float
    train_hashed: int = 0
    train_total: int = 0
    eval_hashed: int = 0
    eval_total: int = 0
    reviewed_text: tuple[TextDecision, ...] = ()

    def by_kind(self, kind: str) -> list[Overlap]:
        return [o for o in self.overlaps if o.kind == kind]

    @property
    def is_complete(self) -> bool:
        return (
            set(_KINDS).issubset(self.checked_kinds)
            and self.train_total > 0
            and self.eval_total > 0
            and self.train_hashed == self.train_total
            and self.eval_hashed == self.eval_total
        )

    @property
    def is_clean(self) -> bool:
        return self.is_complete and not self.overlaps

    @property
    def is_approved(self) -> bool:
        return (
            self.is_complete
            and not any(o.kind != "text" for o in self.overlaps)
            and {o.key for o in self.by_kind("text")} == {r.key for r in self.reviewed_text}
        )

    def summary(self) -> str:
        if not self.is_complete:
            state = "INCOMPLETE"
        elif self.is_clean:
            state = "CLEAN"
        elif self.is_approved:
            state = "REVIEWED"
        else:
            state = "OVERLAP"
        kinds = ", ".join(f"{len(self.by_kind(k))} {k}" for k in self.checked_kinds)
        coverage = (
            f"audio hashes: train {self.train_hashed}/{self.train_total}, "
            f"eval {self.eval_hashed}/{self.eval_total}"
        )
        note = (
            "; review each text collision" if self.by_kind("text") and not self.is_approved else ""
        )
        return (
            f"{state}: {self.eval_name!r} vs {list(self.train_names)}; {kinds}; {coverage}{note}."
        )


def _index(manifests, kinds, audio_root, hashes):
    indexes = {kind: defaultdict(list) for kind in kinds}
    names = defaultdict(set)
    hashed = total = 0
    for manifest in manifests:
        paths = (
            validate_audio_paths(manifest, audio_root)
            if audio_root is not None
            else [None] * len(manifest.segments)
        )
        for seg, path in zip(manifest.segments, paths, strict=True):
            digest, inode = seg.audio_sha256, None
            if path is not None:
                stat = path.stat()
                inode = (stat.st_dev, stat.st_ino)
                if inode not in hashes:
                    hashes[inode] = file_sha256(path)
                if digest is not None and digest != hashes[inode]:
                    raise ValueError(f"segment {seg.id!r}: audio_sha256 does not match file bytes")
                digest = hashes[inode]
                names[inode].add(seg.audio)
            keys = {
                "id": f"{manifest.source}:{seg.id}",
                "text": (normalize(seg.text) or None) if "text" in indexes else None,
                "audio": digest,
                "file": inode,
            }
            for kind, index in indexes.items():
                if keys[kind] is not None:
                    index[keys[kind]].append(f"{manifest.name}:{seg.id}")
            hashed += digest is not None
            total += 1
    return indexes, names, hashed, total


def audit_leakage(
    train: Manifest | Sequence[Manifest],
    evaluation: Manifest,
    *,
    check: Sequence[str] = _KINDS,
    audio_root: str | Path | None = None,
) -> LeakageReport:
    """Inspect IDs, normalized text and file hashes; read bytes when audio_root is set."""
    if (
        not check
        or any(not isinstance(k, str) or k not in _KINDS for k in check)
        or len(set(check)) != len(check)
    ):
        raise ValueError(f"check needs distinct kinds from {_KINDS}")
    trains = [train] if isinstance(train, Manifest) else list(train)
    if any(not isinstance(m, Manifest) for m in trains) or not isinstance(evaluation, Manifest):
        raise ValueError("leakage preflight requires Manifest records")
    if not trains:
        raise ValueError("need at least one training manifest to audit against")
    kinds = tuple(check) + (("file",) if audio_root is not None else ())
    hashes = {}
    left, left_names, train_hashed, train_total = _index(trains, kinds, audio_root, hashes)
    right, right_names, eval_hashed, eval_total = _index([evaluation], kinds, audio_root, hashes)
    overlaps = []
    for kind in (("file",) if audio_root is not None else ()) + tuple(check):
        for key in sorted(left[kind].keys() & right[kind].keys()):
            label = (
                "same-file:" + "|".join(sorted(left_names[key] | right_names[key]))
                if kind == "file"
                else key
            )
            overlaps.append(Overlap(kind, label, tuple(left[kind][key]), tuple(right[kind][key])))
    return LeakageReport(
        tuple(m.name for m in trains),
        evaluation.name,
        tuple(overlaps),
        kinds,
        (train_hashed + eval_hashed) / (train_total + eval_total),
        train_hashed,
        train_total,
        eval_hashed,
        eval_total,
    )


def require_no_leakage(
    train: Manifest | Sequence[Manifest],
    evaluation: Manifest,
    *,
    text_decisions: Mapping[str, str] | None = None,
    audio_root: str | Path | None = None,
) -> LeakageReport:
    """Block hard overlaps and require exact-key benign text-review reasons."""
    trains = [train] if isinstance(train, Manifest) else list(train)
    if any(not isinstance(m, Manifest) for m in trains) or not isinstance(evaluation, Manifest):
        raise ValueError("leakage preflight requires Manifest records")
    if any(m.split != "train" for m in trains):
        raise ValueError("training audit requires train manifests")
    if evaluation.split not in {"dev", "test"}:
        raise ValueError("evaluation audit requires a dev or test manifest")
    if len({m.name for m in trains}) != len(trains):
        raise ValueError("duplicate training manifest names")
    report = audit_leakage(trains, evaluation, audio_root=audio_root)
    if any(overlap.kind != "text" for overlap in report.overlaps):
        raise ValueError(f"hard train/evaluation overlap: {report.summary()}")
    if not report.is_complete:
        raise ValueError(
            f"incomplete train/evaluation leakage audit: {report.summary()} "
            "Provide audio_root to compute missing hashes, or declare every audio_sha256."
        )
    decisions = {} if text_decisions is None else text_decisions
    if not isinstance(decisions, Mapping):
        raise ValueError("text_decisions must map exact normalized keys to review reasons")
    keys = {o.key for o in report.by_kind("text")}
    if set(decisions) != keys:
        raise ValueError("text collisions require exact-key review decisions; missing/stale keys")
    reviewed = []
    for overlap in report.by_kind("text"):
        reason = text_value(decisions[overlap.key], "text review reason")
        reviewed.append(
            TextDecision(overlap.key, reason, overlap.train_segments, overlap.eval_segments)
        )
    return replace(report, reviewed_text=tuple(reviewed))
