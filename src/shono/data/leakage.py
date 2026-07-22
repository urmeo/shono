"""Train/test leakage audit — the check that runs before any source enters the mix.

Overlap between training and evaluation data inflates every reported number and
is the single most common way an honest-looking benchmark lies. This audit runs
before any source enters the mix, because sources like Common Voice, OpenSLR
SLR53, and Bengali-Loop plausibly share clips. Three notions of "the same":

    * **id** — the identical segment id appears in both splits.
    * **text** — transcripts that collapse to the same string under the *frozen*
      normalizer (the same canonicalization used for scoring). This surfaces
      re-used sentences even when ids and audio differ.
    * **audio** — the same ``audio_sha256`` in both, when checksums are present.

The audit *surfaces* overlaps for review; it never deletes. Text collisions on
short, common utterances ("ধন্যবাদ") may be benign — the summary says so — but
they must be seen, not assumed away.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from shono.data.manifest import Manifest, Segment
from shono.eval.normalize import normalize

_KINDS = ("id", "text", "audio")


def _key_for(kind: str, seg: Segment) -> str | None:
    if kind == "id":
        return seg.id
    if kind == "text":
        canon = normalize(seg.text)
        return canon or None  # a segment that normalizes to nothing can't collide meaningfully
    if kind == "audio":
        return seg.audio_sha256
    raise ValueError(f"unknown leakage kind {kind!r}; expected one of {_KINDS}")


@dataclass(frozen=True)
class Overlap:
    """One shared key linking segments in the train split to segments in the eval split."""

    kind: str
    key: str
    train_segments: tuple[str, ...]
    eval_segments: tuple[str, ...]


@dataclass(frozen=True)
class LeakageReport:
    """The overlaps found between one eval manifest and the training mix it is scored against."""

    train_names: tuple[str, ...]
    eval_name: str
    overlaps: tuple[Overlap, ...]
    checked_kinds: tuple[str, ...]
    audio_coverage: float  # fraction of segment pairs where audio checksums were available

    def by_kind(self, kind: str) -> list[Overlap]:
        return [o for o in self.overlaps if o.kind == kind]

    @property
    def is_clean(self) -> bool:
        return not self.overlaps

    def summary(self) -> str:
        if self.is_clean:
            tail = "" if self.audio_coverage >= 1.0 else (
                f" (audio checked on {self.audio_coverage:.0%} of segments)"
            )
            return (
                f"CLEAN: no id/text/audio overlap between {self.eval_name!r} and "
                f"{list(self.train_names)}{tail}."
            )
        counts = {k: len(self.by_kind(k)) for k in self.checked_kinds}
        parts = ", ".join(f"{n} {k}" for k, n in counts.items() if n)
        note = ""
        if counts.get("text"):
            note = " — text collisions on short common phrases may be benign; review each."
        return f"LEAKAGE in {self.eval_name!r} vs {list(self.train_names)}: {parts}{note}"


def _index(segments: Iterable[Segment], kind: str) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for seg in segments:
        key = _key_for(kind, seg)
        if key is not None:
            index[key].append(seg.id)
    return index


def audit_leakage(
    train: Manifest | Sequence[Manifest],
    evaluation: Manifest,
    *,
    check: Sequence[str] = _KINDS,
) -> LeakageReport:
    """Audit ``evaluation`` against one or more ``train`` manifests for overlap.

    Returns a :class:`LeakageReport`; callers decide what to do with it (the
    release checks and the baseline report treat any non-clean result as a
    blocker to investigate).
    """
    for kind in check:
        if kind not in _KINDS:
            raise ValueError(f"unknown leakage kind {kind!r}; expected from {_KINDS}")
    trains = [train] if isinstance(train, Manifest) else list(train)
    if not trains:
        raise ValueError("need at least one training manifest to audit against")

    train_segs = [seg for m in trains for seg in m.segments]
    eval_segs = list(evaluation.segments)

    overlaps: list[Overlap] = []
    for kind in check:
        train_index = _index(train_segs, kind)
        eval_index = _index(eval_segs, kind)
        for key in sorted(train_index.keys() & eval_index.keys()):
            overlaps.append(
                Overlap(
                    kind=kind,
                    key=key,
                    train_segments=tuple(train_index[key]),
                    eval_segments=tuple(eval_index[key]),
                )
            )

    with_audio = sum(1 for seg in (*train_segs, *eval_segs) if seg.audio_sha256 is not None)
    total = len(train_segs) + len(eval_segs)
    coverage = 1.0 if "audio" not in check else (with_audio / total if total else 1.0)

    return LeakageReport(
        train_names=tuple(m.name for m in trains),
        eval_name=evaluation.name,
        overlaps=tuple(overlaps),
        checked_kinds=tuple(check),
        audio_coverage=coverage,
    )
