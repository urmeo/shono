"""Encode a validated finite audio window as mono PCM WAV bytes."""

from __future__ import annotations

import io

from shono.data.audio_paths import validate_audio_window
from shono.transcribe._validation import finite_real


def window_wav_bytes(
    audio_path: str,
    start_s: float | None = None,
    duration_s: float | None = None,
    *,
    max_duration_s: float | None = None,
) -> tuple[bytes, int]:
    """Read a complete selected span; an optional maximum is exclusive."""
    import numpy as np
    import soundfile as sf

    with sf.SoundFile(audio_path) as stream:
        sr, frames = stream.samplerate, stream.frames
        if isinstance(sr, bool) or not isinstance(sr, int) or sr <= 0 or frames <= 0:
            raise ValueError("audio must have samples and a positive sample rate")
        actual = validate_audio_window(
            audio_path,
            start_s,
            duration_s,
            duration_of=lambda _path: frames / sr,
        )
        if max_duration_s is not None:
            limit = finite_real(max_duration_s, "max_duration_s", minimum=0)
            if actual >= limit:
                raise ValueError(f"this API supports selected windows shorter than {limit:g} s")
        first = round((start_s if start_s is not None else 0) * sr)
        selected = min(round(actual * sr), frames - first)
        if selected <= 0:
            raise ValueError("selected audio window contains no samples")
        stream.seek(first)
        data = stream.read(frames=selected, dtype="float64", always_2d=True)
        if len(data) != selected or not np.isfinite(data).all():
            raise ValueError("selected audio must contain complete finite samples")
        mono = data.mean(axis=1)
        if not np.isfinite(mono).all() or np.any(np.abs(mono) > 1):
            raise ValueError("selected audio cannot be encoded as PCM without clipping")
    buffer = io.BytesIO()
    sf.write(buffer, mono, sr, format="WAV", subtype="PCM_16")
    return buffer.getvalue(), sr
