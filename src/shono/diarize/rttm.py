"""Load reference diarization from RTTM — the standard speaker-turn annotation format.

An RTTM line is space-separated: ``SPEAKER <file> <chan> <start> <dur> <NA> <NA>
<speaker> <NA> <NA>``. This reads the ``SPEAKER`` lines of one recording into the
:class:`SpeakerSegment` list :func:`shono.diarize.der` scores against.
"""

from __future__ import annotations

from pathlib import Path

from shono.diarize.types import SpeakerSegment


def load_rttm(path: str | Path, *, file_id: str | None = None) -> list[SpeakerSegment]:
    """Read ``SPEAKER`` turns from an RTTM file into speaker segments.

    ``file_id`` optionally filters to one recording (RTTM column 2); zero/negative
    duration turns are skipped.
    """
    segments: list[SpeakerSegment] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        if file_id is not None and parts[1] != file_id:
            continue
        start, duration, speaker = float(parts[3]), float(parts[4]), parts[7]
        if duration > 0:
            segments.append(SpeakerSegment(start, start + duration, speaker))
    return segments
