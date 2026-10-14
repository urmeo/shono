"""Train/evaluation overlap checks with explicit text-collision review."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from shono.data.audio_paths import file_sha256, validate_audio_paths
from shono.data.manifest import Manifest
from shono.data.validation import text_value
from shono.eval.normalize import normalize

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


def _index(manifests: Sequence[Manifest], kind: str) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for manifest in manifests:
        for seg in manifest.segments:
            if kind == "id":
                key = f"{manifest.source}:{seg.id}"
            elif kind == "text":
                key = normalize(seg.text) or None
            else:
                key = seg.audio_sha256
            if key is not None:
                index[key].append(f"{manifest.name}:{seg.id}")
    return index


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
    overlaps = []
    kinds = tuple(check)
    if audio_root is not None:
        file_hashes: dict[tuple[int, int], str] = {}

        def file_index(manifests):
            hashed = []
            index = defaultdict(list)
            names = defaultdict(set)
            for manifest in manifests:
                paths = validate_audio_paths(manifest, audio_root)
                segments = []
                for seg, path in zip(manifest.segments, paths, strict=True):
                    stat = path.stat()
                    key = (stat.st_dev, stat.st_ino)
                    if key not in file_hashes:
                        file_hashes[key] = file_sha256(path)
                    digest = file_hashes[key]
                    if seg.audio_sha256 is not None and seg.audio_sha256 != digest:
                        raise ValueError(
                            f"segment {seg.id!r}: audio_sha256 does not match file bytes"
                        )
                    segments.append(replace(seg, audio_sha256=digest))
                    index[key].append(f"{manifest.name}:{seg.id}")
                    names[key].add(seg.audio)
                hashed.append(replace(manifest, segments=tuple(segments)))
            return hashed, index, names

        trains, left, left_names = file_index(trains)
        evaluations, right, right_names = file_index([evaluation])
        evaluation = evaluations[0]
        for key in sorted(left.keys() & right.keys()):
            name = "same-file:" + "|".join(sorted(left_names[key] | right_names[key]))
            overlaps.append(Overlap("file", name, tuple(left[key]), tuple(right[key])))
        kinds += ("file",)
    for kind in check:
        left, right = _index(trains, kind), _index([evaluation], kind)
        overlaps.extend(
            Overlap(kind, key, tuple(left[key]), tuple(right[key]))
            for key in sorted(left.keys() & right.keys())
        )
    train_segs = [seg for manifest in trains for seg in manifest.segments]
    eval_segs = evaluation.segments
    train_hashed = sum(seg.audio_sha256 is not None for seg in train_segs)
    eval_hashed = sum(seg.audio_sha256 is not None for seg in eval_segs)
    total = len(train_segs) + len(eval_segs)
    return LeakageReport(
        tuple(m.name for m in trains),
        evaluation.name,
        tuple(overlaps),
        kinds,
        (train_hashed + eval_hashed) / total,
        train_hashed,
        len(train_segs),
        eval_hashed,
        len(eval_segs),
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
