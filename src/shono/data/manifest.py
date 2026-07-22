"""Dataset manifests — a slice of audio described without shipping the audio.

A manifest is one named dataset split (``cv-bn-test``, ``bengali-loop-asr``,
``mucs-slr104-cs-test`` …). It is metadata only: ids, audio *references*,
durations, reference transcripts, and — crucially — the ``recording_id`` that
groups segments into the blocks the bootstrap CI resamples. Audio itself never
enters git (see ``.gitignore``); a manifest is what the ship repo can hold in
place of terabytes of speech.

On-disk format is JSON Lines: the **first line** is a header object
``{"manifest": {...}}`` carrying the split-level metadata, and every following
line is one :class:`Segment`. One self-describing file, streamable line by line,
and readable with ``jq`` — the header is skippable by its distinct shape.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from shono.data.license import LicenseRegistry

_SPLITS = frozenset({"train", "dev", "test"})
_REQUIRED_HEADER = ("name", "source", "split", "domain", "version")
_REQUIRED_SEGMENT = ("id", "audio", "text", "duration_s", "recording_id")


@dataclass(frozen=True)
class Segment:
    """One scored unit of audio, described without the audio.

    ``recording_id`` is the bootstrap block a segment belongs to — the recording
    (or speaker) it was cut from. For read-speech corpora where every clip is its
    own recording, set it equal to ``id``. ``start_s``/``duration_s`` locate the
    segment inside ``audio`` for long-form files; ``audio_sha256`` (optional)
    powers audio-level leakage detection and integrity checks.
    """

    id: str
    audio: str
    text: str
    duration_s: float
    recording_id: str
    start_s: float | None = None
    speaker: str | None = None
    language: str = "bn"
    audio_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("segment id must be non-empty")
        if not self.recording_id:
            raise ValueError(f"segment {self.id!r} has an empty recording_id")
        if not self.text.strip():
            raise ValueError(
                f"segment {self.id!r} has empty reference text; "
                "an empty reference is unscorable — fix or drop the segment"
            )
        if self.duration_s <= 0:
            raise ValueError(
                f"segment {self.id!r} has non-positive duration_s={self.duration_s}"
            )
        if self.start_s is not None and self.start_s < 0:
            raise ValueError(f"segment {self.id!r} has negative start_s={self.start_s}")


@dataclass(frozen=True)
class Manifest:
    """A named dataset split: split-level metadata plus its segments."""

    name: str
    source: str
    split: str
    domain: str
    version: str
    segments: tuple[Segment, ...]
    language: str = "bn"
    checksum: str | None = None
    _HEADER_FIELDS = ("name", "source", "split", "domain", "version", "language", "checksum")

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("manifest name must be non-empty")
        if not self.source:
            raise ValueError(f"manifest {self.name!r} has an empty source")
        if self.split not in _SPLITS:
            raise ValueError(
                f"manifest {self.name!r} has split {self.split!r}; "
                f"expected one of {sorted(_SPLITS)}"
            )
        if not self.segments:
            raise ValueError(f"manifest {self.name!r} has no segments")
        seen: set[str] = set()
        for seg in self.segments:
            if seg.id in seen:
                raise ValueError(f"manifest {self.name!r} has duplicate segment id {seg.id!r}")
            seen.add(seg.id)

    # ---- derived views ---------------------------------------------------

    def total_hours(self) -> float:
        """Total audio duration across all segments, in hours."""
        return sum(seg.duration_s for seg in self.segments) / 3600.0

    def recording_ids(self) -> list[str]:
        """Distinct recording (block) ids, sorted."""
        return sorted({seg.recording_id for seg in self.segments})

    def references(self) -> list[str]:
        """Reference transcripts, in segment order."""
        return [seg.text for seg in self.segments]

    def ids(self) -> list[str]:
        """Segment ids, in segment order."""
        return [seg.id for seg in self.segments]

    def records_with(self, hypotheses: Mapping[str, str]) -> list[tuple[str, str, str]]:
        """Join references with ``hypotheses`` into ``(recording_id, ref, hyp)`` triples.

        Every segment must have a hypothesis; a missing prediction raises rather
        than being silently dropped — a partial run must never masquerade as a
        complete score. The triples feed :func:`shono.eval.ci.blockwise_bootstrap_ci`.
        """
        missing = [seg.id for seg in self.segments if seg.id not in hypotheses]
        if missing:
            preview = ", ".join(missing[:5])
            more = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
            raise KeyError(
                f"{len(missing)} of {len(self.segments)} segments in manifest "
                f"{self.name!r} have no hypothesis: {preview}{more}. "
                "Score only a complete prediction set."
            )
        return [(seg.recording_id, seg.text, hypotheses[seg.id]) for seg in self.segments]

    # ---- license linkage -------------------------------------------------

    def validate_against(self, registry: LicenseRegistry) -> None:
        """Assert the source is registered and permitted for this split's use.

        Raises if the source is unregistered, or if an ``excluded`` source is
        referenced at all, or if an ``eval-only`` source is used as ``train``.
        This is the check that keeps unlicensed data out of the mix.
        """
        lic = registry.require(self.source)
        if lic.status == "excluded":
            raise ValueError(
                f"manifest {self.name!r} references excluded source {lic.id!r}: {lic.notes}"
            )
        if lic.status == "eval-only" and self.split == "train":
            raise ValueError(
                f"manifest {self.name!r} uses eval-only source {lic.id!r} as training data; "
                "eval-only sources may never enter the train split"
            )

    # ---- serialization ---------------------------------------------------

    def to_jsonl(self, path: str | Path) -> None:
        """Write the manifest as header-line JSONL."""
        header = {"manifest": {k: getattr(self, k) for k in self._HEADER_FIELDS}}
        lines = [json.dumps(header, ensure_ascii=False)]
        lines += [
            json.dumps({k: v for k, v in asdict(seg).items() if v is not None}, ensure_ascii=False)
            for seg in self.segments
        ]
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> Manifest:
        """Load a manifest from header-line JSONL."""
        raw = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not raw:
            raise ValueError(f"manifest file {path} is empty")
        head = json.loads(raw[0])
        if "manifest" not in head:
            raise ValueError(
                f"manifest file {path} must start with a header line "
                '{"manifest": {...}}; got a segment line first'
            )
        meta = head["manifest"]
        missing_head = [k for k in _REQUIRED_HEADER if k not in meta]
        if missing_head:
            raise ValueError(
                f"manifest header in {path} is missing required field(s): "
                f"{missing_head}. A manifest header needs {list(_REQUIRED_HEADER)}."
            )
        seg_fields = {f.name for f in fields(Segment)}
        segments = []
        for lineno, ln in enumerate(raw[1:], start=2):
            rec = json.loads(ln)
            missing_seg = [k for k in _REQUIRED_SEGMENT if k not in rec]
            if missing_seg:
                raise ValueError(
                    f"segment on line {lineno} of {path} is missing required field(s): "
                    f"{missing_seg}. A segment needs {list(_REQUIRED_SEGMENT)}."
                )
            segments.append(Segment(**{k: v for k, v in rec.items() if k in seg_fields}))
        head_fields = {f.name for f in fields(cls)}
        return cls(segments=tuple(segments), **{k: v for k, v in meta.items() if k in head_fields})
