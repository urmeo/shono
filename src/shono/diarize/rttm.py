"""Read validated turns from one RTTM recording or Bengali-Loop CSV."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from shono.diarize.types import SpeakerSegment
from shono.transcribe._validation import finite_real


def load_rttm(path: str | Path, *, file_id: str | None = None) -> list[SpeakerSegment]:
    """Read one recording and reject multiple audio channels within it."""
    records: list[tuple[str, str, SpeakerSegment]] = []
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
            records.append((parts[1], parts[2], SpeakerSegment(start, start + duration, parts[7])))
        except ValueError as exc:
            raise ValueError(f"{path}: line {lineno}: {exc}") from exc
    ids = {recording for recording, _channel, _turn in records}
    if file_id is None and len(ids) > 1:
        raise ValueError("RTTM contains multiple recording IDs; specify file_id")
    if file_id is not None and ids and file_id not in ids:
        raise ValueError(f"{path}: RTTM recording {file_id!r} was not found")
    selected = [
        (channel, turn)
        for recording, channel, turn in records
        if file_id is None or recording == file_id
    ]
    if len({channel for channel, _turn in selected}) > 1:
        raise ValueError(f"{path}: RTTM recording contains multiple channels; unsupported input")
    return sorted((turn for _channel, turn in selected), key=lambda s: s.start_s)


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
        reader = csv.reader(stream, strict=True)
        try:
            header = next(reader, [])
            if len(header) != len(set(header)):
                raise ValueError(f"{path}: duplicate CSV headers")
            if not {"start_time", "end_time", "speaker_id"}.issubset(header):
                raise ValueError("Loop CSV requires start_time, end_time and speaker_id columns")
            for values in reader:
                lineno = reader.line_num
                if len(values) != len(header):
                    raise ValueError(
                        f"{path}: line {lineno}: expected {len(header)} columns, got {len(values)}"
                    )
                row = dict(zip(header, values, strict=True))
                try:
                    label = row["speaker_id"].strip()
                    if not label.isdecimal() or int(label) < 1:
                        raise ValueError("speaker_id must be a positive integer")
                    turns.append(
                        SpeakerSegment(
                            _clock(row["start_time"].strip()),
                            _clock(row["end_time"].strip()),
                            label,
                        )
                    )
                except (ValueError, OverflowError) as exc:
                    raise ValueError(f"{path}: line {lineno}: {exc}") from exc
        except csv.Error as exc:
            raise ValueError(f"{path}: invalid CSV on line {reader.line_num}: {exc}") from exc
    return sorted(turns, key=lambda s: s.start_s)
