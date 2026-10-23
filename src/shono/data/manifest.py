"""Dataset metadata: one header and one JSON object per scored segment."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from itertools import chain
from pathlib import Path

from shono.data.license import LicenseRegistry
from shono.validation import audio_reference, json_object, real_number, text_value

_SPLITS = frozenset({"train", "dev", "test"})
_REQUIRED_HEADER = ("name", "source", "split", "domain", "version")
_REQUIRED_SEGMENT = ("id", "audio", "text", "duration_s", "recording_id")


def _record(record, required, allowed, location):
    if not isinstance(record, dict):
        raise ValueError(f"{location}: manifest header must be an object")
    missing, unknown = set(required) - record.keys(), record.keys() - set(allowed)
    if missing or unknown:
        raise ValueError(
            f"{location}: missing required field(s) {sorted(missing)}; "
            f"unknown fields {sorted(unknown)}"
        )
    return record


@dataclass(frozen=True)
class Segment:
    """A reference span: None start means a whole file; numeric start means an aligned crop.

    recording_id is the bootstrap block. audio_sha256 hashes complete file bytes.
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
        text_value(self.id, "segment id")
        text_value(self.recording_id, "recording_id")
        audio_reference(self.audio)
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError(
                f"segment {self.id!r} has empty reference text; fix or omit empty references"
            )
        duration = real_number(self.duration_s, "duration_s")
        if duration <= 0:
            raise ValueError(f"segment {self.id!r} has non-positive duration_s={self.duration_s}")
        if self.start_s is not None:
            start = real_number(self.start_s, "start_s")
            if start < 0:
                raise ValueError(f"segment {self.id!r} has negative start_s={self.start_s}")
            if not math.isfinite(start + duration):
                raise ValueError("start_s + duration_s must be finite")
        text_value(self.language, "language")
        if self.speaker is not None:
            text_value(self.speaker, "speaker")
        if self.audio_sha256 is not None:
            if not isinstance(self.audio_sha256, str) or not re.fullmatch(
                r"[0-9a-fA-F]{64}", self.audio_sha256
            ):
                raise ValueError("audio_sha256 must be a 64-character hexadecimal file digest")
            object.__setattr__(self, "audio_sha256", self.audio_sha256.lower())


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
        for key in ("name", "source", "domain", "version", "language"):
            text_value(getattr(self, key), f"manifest {key}")
        text_value(self.split, "manifest split")
        if self.split not in _SPLITS:
            raise ValueError(
                f"manifest {self.name!r} has split {self.split!r}; "
                f"expected one of {sorted(_SPLITS)}"
            )
        if not self.segments:
            raise ValueError(f"manifest {self.name!r} has no segments")
        if not isinstance(self.segments, (tuple, list)):
            raise ValueError("manifest segments must be a sequence of Segment records")
        object.__setattr__(self, "segments", tuple(self.segments))
        seen: set[str] = set()
        for seg in self.segments:
            if not isinstance(seg, Segment):
                raise ValueError("manifest segments must be Segment records")
            if seg.id in seen:
                raise ValueError(f"manifest {self.name!r} has duplicate segment id {seg.id!r}")
            seen.add(seg.id)
        if not math.isfinite(sum(seg.duration_s for seg in self.segments)):
            raise ValueError("total manifest duration must be finite")
        if self.checksum is not None:
            text_value(self.checksum, "checksum")

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
        """Join a complete hypothesis set into (block, reference, hypothesis) triples."""
        missing = [seg.id for seg in self.segments if seg.id not in hypotheses]
        if missing:
            raise KeyError(
                f"manifest {self.name!r}: {len(missing)} segments have no hypothesis: {missing[:5]}"
            )
        return [(seg.recording_id, seg.text, hypotheses[seg.id]) for seg in self.segments]

    def validate_against(self, registry: LicenseRegistry, *, training: bool = False) -> None:
        """Check source permissions and reject excluded or invalid training uses."""
        if type(training) is not bool:
            raise ValueError("training must be a boolean")
        lic = registry.require(self.source)
        if lic.status == "excluded":
            raise ValueError(
                f"manifest {self.name!r} references excluded source {lic.id!r}: {lic.notes}"
            )
        if training and self.split != "train":
            raise ValueError(f"manifest {self.name!r}: training requires a train split")
        if lic.status == "eval-only" and (self.split == "train" or training):
            raise ValueError(
                f"manifest {self.name!r} uses eval-only source {lic.id!r} as training data; "
                "eval-only sources may never enter the train split"
            )

    def to_jsonl(self, path: str | Path) -> None:
        """Write the manifest as header-line JSONL."""
        header = {"manifest": {k: getattr(self, k) for k in self._HEADER_FIELDS}}
        records = chain(
            (header,),
            ({k: v for k, v in asdict(seg).items() if v is not None} for seg in self.segments),
        )
        lines = [json.dumps(record, ensure_ascii=False, allow_nan=False) for record in records]
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> Manifest:
        """Load a manifest from header-line JSONL."""
        return cls.from_text(Path(path).read_text(encoding="utf-8"), location=str(path))

    @classmethod
    def from_text(cls, text: str, *, location: str = "<manifest>") -> Manifest:
        """Parse header-line JSONL, retaining physical line numbers in errors."""
        raw = [(i, ln) for i, ln in enumerate(text.splitlines(), 1) if ln.strip()]
        if not raw:
            raise ValueError(f"manifest file {location} is empty")
        head = json_object(raw[0][1], f"{location}: line {raw[0][0]}")
        if set(head) != {"manifest"}:
            raise ValueError(
                f"manifest file {location} must start with a header line "
                '{"manifest": {...}}; got a segment line first'
            )
        meta = _record(head["manifest"], _REQUIRED_HEADER, cls._HEADER_FIELDS, location)
        seg_fields = {f.name for f in fields(Segment)}
        segments = []
        for lineno, ln in raw[1:]:
            where = f"{location}: line {lineno}"
            rec = _record(json_object(ln, where), _REQUIRED_SEGMENT, seg_fields, where)
            try:
                segments.append(Segment(**rec))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{location}: line {lineno}: {exc}") from exc
        return cls(segments=tuple(segments), **meta)


def load_manifest(path: str | Path) -> tuple[Manifest, str]:
    """Parse and hash the same file bytes."""
    raw = Path(path).read_bytes()
    manifest = Manifest.from_text(raw.decode("utf-8"), location=str(path))
    return manifest, hashlib.sha256(raw).hexdigest()
