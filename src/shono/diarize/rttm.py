"""Read validated turns from one RTTM recording or Bengali-Loop CSV."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from shono.diarize.types import SpeakerSegment
from shono.transcribe._validation import finite_real


def load_rttm(path: str | Path, *, file_id: str | None = None) -> list[SpeakerSegment]:
    """Require an explicit file ID when an RTTM contains multiple recordings."""
    records: list[tuple[str, SpeakerSegment]] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts or parts[0] != "SPEAKER":
            continue
        try:
            if len(parts) < 8:
                raise ValueError("SPEAKER row needs at least eight columns")
            start = finite_real(float(parts[3]), "start", minimum=0)
            duration = finite_real(float(parts[4]), "duration", minimum=0)
            if duration <= 0:
                raise ValueError("duration must be > 0")
            records.append((parts[1], SpeakerSegment(start, start + duration, parts[7])))
        except ValueError as exc:
            raise ValueError(f"{path}: line {lineno}: {exc}") from exc
    ids = {recording for recording, _turn in records}
    if file_id is None and len(ids) > 1:
        raise ValueError("RTTM contains multiple recording IDs; specify file_id")
    return sorted(
        (turn for recording, turn in records if file_id is None or recording == file_id),
        key=lambda s: s.start_s,
    )


def _clock(text: str) -> float:
    if not re.fullmatch(r"\d+:\d{2}:\d{2}(?:\.\d+)?", text):
        raise ValueError("time must use HH:MM:SS[.fraction]")
    hours, minutes, seconds = text.split(":")
    if int(minutes) >= 60 or float(seconds) >= 60:
        raise ValueError("minutes and seconds must be < 60")
    return finite_real(int(hours) * 3600 + int(minutes) * 60 + float(seconds), "clock", minimum=0)


def load_loop_csv(path: str | Path) -> list[SpeakerSegment]:
    """Read published start_time/end_time/speaker_id columns without guessing alignments."""
    turns: list[SpeakerSegment] = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"start_time", "end_time", "speaker_id"}.issubset(reader.fieldnames or []):
            raise ValueError("Loop CSV requires start_time, end_time and speaker_id columns")
        for lineno, row in enumerate(reader, 2):
            try:
                label = row["speaker_id"].strip()
                if not label.isdecimal() or int(label) < 1:
                    raise ValueError("speaker_id must be a positive integer")
                turns.append(
                    SpeakerSegment(
                        _clock(row["start_time"].strip()), _clock(row["end_time"].strip()), label
                    )
                )
            except (ValueError, AttributeError, TypeError, OverflowError) as exc:
                raise ValueError(f"{path}: line {lineno}: {exc}") from exc
    return sorted(turns, key=lambda s: s.start_s)
