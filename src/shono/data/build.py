"""Build manifests from raw dataset layouts — the bridge from downloaded data to a slice.

The package has the manifest *schema* (:mod:`shono.data.manifest`); this module
turns a real dataset's index (a Common Voice ``.tsv``, a FLEURS ``.tsv``, or any
row iterable) into one. Durations rarely live in the index, so a ``duration_of``
callback supplies them (default: a lazy librosa read, which handles mp3); inject a
fake and the whole builder is testable without touching audio.

The dataset-specific parsers here cover the two well-specified formats (Common
Voice, FLEURS). For sources whose on-disk layout varies (OpenSLR, MUCS,
Bengali-Loop), map their rows in the data-prep notebook and hand them to
:func:`build_manifest` — where the real format is in front of you, not guessed here.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from shono.data.manifest import Manifest, Segment

FLEURS_SAMPLE_RATE = 16_000
DurationFn = Callable[[Path], float]


def duration_from_audio(path: Path) -> float:
    """Seconds of audio at ``path`` (lazy librosa — reads mp3/wav/flac)."""
    import librosa

    return float(librosa.get_duration(path=str(path)))


def build_manifest(
    *,
    name: str,
    source: str,
    split: str,
    domain: str,
    version: str,
    rows: Iterable[Mapping[str, object]],
    language: str = "bn",
    checksum: str | None = None,
    audio_root: str | Path = ".",
    duration_of: DurationFn | None = None,
) -> Manifest:
    """Build a :class:`Manifest` from ``rows`` of ``{id, audio, text, ...}`` mappings.

    Each row needs ``id``, ``audio`` (path relative to ``audio_root``), and ``text``;
    ``recording_id`` (defaults to ``id``), ``start_s``, ``speaker``, and ``duration_s``
    are optional. A row missing ``duration_s`` has it computed with ``duration_of``
    (default: read the file). Rows with empty text are skipped — an empty reference is
    unscorable — and the count of skipped rows is available on the returned manifest's
    build (callers compare ``len(rows)`` to ``len(manifest.segments)``).
    """
    dur_fn = duration_of or duration_from_audio
    root = Path(audio_root)
    segments: list[Segment] = []
    for row in rows:
        text = str(row.get("text", "")).strip()
        if not text:
            continue  # unscorable; a data-prep artefact, not a segment
        audio = str(row["audio"])
        duration = row.get("duration_s")
        duration_s = float(duration) if duration is not None else dur_fn(root / audio)
        start = row.get("start_s")
        segments.append(
            Segment(
                id=str(row["id"]),
                audio=audio,
                text=text,
                duration_s=duration_s,
                recording_id=str(row.get("recording_id") or row["id"]),
                start_s=float(start) if start is not None else None,
                speaker=(str(row["speaker"]) if row.get("speaker") is not None else None),
            )
        )
    return Manifest(
        name=name, source=source, split=split, domain=domain, version=version,
        language=language, checksum=checksum, segments=tuple(segments),
    )


def collapse_to_recordings(manifest: Manifest) -> Manifest:
    """Collapse a segment-level manifest to one entry per recording, for long-form eval.

    A recording's segments are joined in start-time order into a single reference; the
    entry's audio is the recording file and its duration the sum. Scoring the whole
    recording (block = recording) against the long-form pipeline's merged transcript is
    the honest long-form protocol — not per-segment, which the pipeline never produces.
    """
    groups: dict[str, list[Segment]] = defaultdict(list)
    for seg in manifest.segments:
        groups[seg.recording_id].append(seg)
    segments = []
    for rid in sorted(groups):
        segs = sorted(groups[rid], key=lambda s: (s.start_s if s.start_s is not None else 0.0))
        segments.append(
            Segment(
                id=rid,
                audio=segs[0].audio,
                text=" ".join(s.text for s in segs),
                duration_s=sum(s.duration_s for s in segs),
                recording_id=rid,
            )
        )
    return Manifest(
        name=manifest.name, source=manifest.source, split=manifest.split,
        domain=manifest.domain, version=manifest.version, language=manifest.language,
        checksum=manifest.checksum, segments=tuple(segments),
    )


def from_common_voice_tsv(
    tsv_path: str | Path,
    *,
    split: str,
    name: str,
    version: str,
    source: str = "common_voice_bn",
    audio_subdir: str = "clips",
    durations_tsv: str | Path | None = None,
    audio_root: str | Path = ".",
    duration_of: DurationFn | None = None,
) -> Manifest:
    """Build a manifest from a Common Voice split ``.tsv`` (validated/test/train/dev).

    Parsed by column name (``path``, ``sentence``, ``client_id``) so it survives the
    corpus's column drift across releases. ``client_id`` becomes the recording block
    (speaker) for the bootstrap CI. If a ``clip_durations.tsv`` (``clip``/``path`` →
    ``duration[ms]``) is given, durations come from it; otherwise from ``duration_of``.
    """
    tsv_path = Path(tsv_path)
    durations = _load_cv_durations(durations_tsv) if durations_tsv else {}
    rows: list[dict[str, object]] = []
    with tsv_path.open(encoding="utf-8", newline="") as f:
        for rec in csv.DictReader(f, delimiter="\t"):
            clip = rec.get("path") or ""
            sentence = (rec.get("sentence") or "").strip()
            if not clip or not sentence:
                continue
            row: dict[str, object] = {
                "id": Path(clip).stem,
                "audio": f"{audio_subdir}/{clip}",
                "text": sentence,
                "recording_id": rec.get("client_id") or Path(clip).stem,
            }
            if clip in durations:
                row["duration_s"] = durations[clip]
            rows.append(row)
    return build_manifest(
        name=name, source=source, split=split, domain="read", version=version,
        rows=rows, audio_root=audio_root, duration_of=duration_of,
    )


def _load_cv_durations(durations_tsv: str | Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with Path(durations_tsv).open(encoding="utf-8", newline="") as f:
        for rec in csv.DictReader(f, delimiter="\t"):
            clip = rec.get("clip") or rec.get("path")
            ms = rec.get("duration[ms]") or rec.get("duration")
            if clip and ms:
                out[clip] = float(ms) / 1000.0
    return out


def from_fleurs_tsv(
    tsv_path: str | Path,
    *,
    split: str,
    name: str,
    audio_subdir: str = "audio",
    source: str = "fleurs_bn",
    version: str = "fleurs",
    audio_root: str | Path = ".",
) -> Manifest:
    """Build a manifest from a FLEURS split ``.tsv``.

    FLEURS rows are headerless: ``id  file_name  raw_transcription  transcription
    num_samples  gender``. Duration comes from ``num_samples`` (16 kHz), so no audio
    read is needed. The normalized ``transcription`` column is the reference.
    """
    rows: list[dict[str, object]] = []
    with Path(tsv_path).open(encoding="utf-8", newline="") as f:
        for parts in csv.reader(f, delimiter="\t"):
            if len(parts) < 5:
                continue
            _id, file_name, _raw, transcription, num_samples = parts[:5]
            text = transcription.strip()
            if not text:
                continue
            rows.append({
                "id": Path(file_name).stem,
                "audio": f"{audio_subdir}/{file_name}",
                "text": text,
                "recording_id": Path(file_name).stem,
                "duration_s": float(num_samples) / FLEURS_SAMPLE_RATE,
            })
    return build_manifest(
        name=name, source=source, split=split, domain="read", version=version,
        rows=rows, audio_root=audio_root,
    )
