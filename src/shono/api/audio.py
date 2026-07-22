"""Read an audio window as WAV bytes for the API adapters (lazy soundfile)."""

from __future__ import annotations


def window_wav_bytes(
    audio_path: str, start_s: float | None = None, duration_s: float | None = None
) -> tuple[bytes, int]:
    """Return ``(wav_bytes, sample_rate)`` for the ``[start_s, +duration_s]`` window.

    The whole file when both bounds are ``None``. Used to send just the segment
    under test to a commercial API rather than the full recording.
    """
    import io

    import soundfile as sf

    with sf.SoundFile(audio_path) as f:
        sr = f.samplerate
        if start_s:
            f.seek(int(start_s * sr))
        frames = int(duration_s * sr) if duration_s else -1
        data = f.read(frames=frames, dtype="int16")
    buffer = io.BytesIO()
    sf.write(buffer, data, sr, format="WAV", subtype="PCM_16")
    return buffer.getvalue(), sr
