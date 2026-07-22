"""``python -m shono.transcribe <audio> --model <ct2-dir>`` — transcribe one recording.

The thin CLI over :class:`LongFormTranscriber`: Silero VAD → chunk → faster-whisper
→ de-hallucinate → merge, printing the transcript and the measured real-time
factor. It needs a converted CTranslate2 model and audio, so it runs on the GPU
box; the pipeline logic it drives is unit-tested without either.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _audio_duration_s(path: str) -> float:
    try:
        import soundfile as sf

        info = sf.info(path)
        return info.frames / info.samplerate
    except Exception:  # noqa: BLE001 - fall back to librosa if soundfile can't read it
        import librosa

        return float(librosa.get_duration(path=path))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m shono.transcribe")
    p.add_argument("audio", help="path to the recording to transcribe")
    p.add_argument("--model", required=True, help="path to the CTranslate2 model directory")
    p.add_argument("--language", default="bn")
    p.add_argument("--device", default="cuda")
    p.add_argument("--compute-type", default="float16")
    p.add_argument("--max-chunk-s", type=float, default=28.0)
    p.add_argument("--json", type=Path, help="also write the transcript + timings as JSON here")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not Path(args.audio).exists():
        print(f"error: audio file not found: {args.audio}", file=sys.stderr)
        return 2

    from shono.transcribe import (
        FasterWhisperTranscriber,
        LongFormTranscriber,
        SileroVAD,
    )

    pipeline = LongFormTranscriber(
        SileroVAD(),
        FasterWhisperTranscriber(
            args.model, device=args.device, compute_type=args.compute_type, language=args.language
        ),
        max_chunk_s=args.max_chunk_s,
    )
    result = pipeline.transcribe(args.audio, _audio_duration_s(args.audio))

    print(result.transcript.text)
    print(
        f"\n[{result.n_chunks} chunks · RTF {result.rtf:.2f} · {result.filtering.summary()}]",
        file=sys.stderr,
    )
    if args.json:
        payload = {
            "text": result.transcript.text,
            "rtf": result.rtf,
            "audio_duration_s": result.audio_duration_s,
            "processing_s": result.processing_s,
            "n_chunks": result.n_chunks,
            "dropped": result.filtering.n_dropped,
            "segments": [
                {"start_s": s.start_s, "end_s": s.end_s, "text": s.text}
                for s in result.transcript.segments
            ],
        }
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
