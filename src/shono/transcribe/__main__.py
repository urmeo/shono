"""Transcribe one recording with optional atomic JSON export."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

from shono.data.audio_paths import validate_audio_window
from shono.output_paths import source_files, validate_output_destinations
from shono.transcribe.chunking import plan_chunks
from shono.transcribe.convert import validate_ct2_checkpoint


def _audio_duration_s(path: str) -> float:
    return validate_audio_window(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m shono.transcribe", allow_abbrev=False)
    parser.add_argument("audio", help="path to the recording")
    parser.add_argument("--model", required=True, help="existing CTranslate2 model directory")
    parser.add_argument("--language", default="bn")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--max-chunk-s", type=float, default=28.0)
    parser.add_argument("--pad-s", type=float, default=0.2)
    parser.add_argument("--max-chunks", type=int, default=100_000)
    parser.add_argument(
        "--json", type=Path, help="JSON output, separate from input/model/source files"
    )
    return parser


def _output_path(path: Path, audio: Path, model: Path) -> Path:
    if path.suffix.lower() != ".json":
        raise ValueError("--json destination must use a .json extension")
    try:
        target = path.resolve()
        package = Path(__file__).resolve().parents[1]
        model = model.resolve()
        protected = [audio.resolve(), *package.rglob("*.py"), *model.rglob("*")]
        project = package.parent.parent
        if (project / "pyproject.toml").is_file():
            protected.extend(source_files(project))
        validate_output_destinations((path,), protected_paths=protected)
        if target.is_relative_to(model) or target.is_relative_to(package):
            raise ValueError("--json destination cannot overwrite model or package sources")
        for source in protected:
            if target == source.resolve() or (
                target.exists() and source.is_file() and target.samefile(source)
            ):
                raise ValueError("--json destination aliases an input or source file")
        if target.exists() and not target.is_file():
            raise ValueError("--json destination must be a file")
        parent = target.parent
        while not parent.exists():
            parent = parent.parent
        if not parent.is_dir() or not os.access(parent, os.W_OK):
            raise ValueError("--json destination parent must be writable")
        return target
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"cannot resolve JSON destination: {exc}") from exc


def _write_json(path: Path, payload: dict) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        audio = Path(args.audio).resolve()
        if not audio.is_file():
            print(f"error: audio file not found: {args.audio}", file=sys.stderr)
            return 2
        model = Path(args.model).resolve()
        if not model.is_dir():
            print("error: --model must be an existing CTranslate2 directory", file=sys.stderr)
            return 2
        model = validate_ct2_checkpoint(model)
        plan_chunks([], max_chunk_s=args.max_chunk_s, pad_s=args.pad_s, max_chunks=args.max_chunks)
        if args.max_chunk_s > 30:
            raise ValueError("Whisper windows must be <= 30 s including padding")
        output = _output_path(args.json, audio, model) if args.json else None
        duration = _audio_duration_s(str(audio))
        from shono.transcribe import FasterWhisperTranscriber, LongFormTranscriber, SileroVAD

        pipeline = LongFormTranscriber(
            SileroVAD(),
            FasterWhisperTranscriber(
                str(model),
                device=args.device,
                compute_type=args.compute_type,
                language=args.language,
            ),
            max_chunk_s=args.max_chunk_s,
            pad_s=args.pad_s,
            max_chunks=args.max_chunks,
        )
        result = pipeline.transcribe(str(audio), duration)
        if output:
            _write_json(
                output,
                {
                    "schema_version": 1,
                    "text": result.transcript.text,
                    "rtf": result.rtf,
                    "audio_duration_s": result.audio_duration_s,
                    "processing_s": result.processing_s,
                    "n_chunks": result.n_chunks,
                    "n_forced_splits": result.n_forced_splits,
                    "dropped": result.filtering.n_dropped,
                    "warnings": result.transcript.warnings,
                    "segments": [asdict(s) for s in result.transcript.segments],
                },
            )
        print(result.transcript.text)
        print(
            f"[{result.n_chunks} chunks | RTF {result.rtf:.2f} | {result.filtering.summary()}]",
            file=sys.stderr,
        )
        for warning in result.transcript.warnings:
            print(f"warning: {warning}", file=sys.stderr)
        return 0
    except (ValueError, OSError, RuntimeError, ImportError, OverflowError) as exc:
        print(f"error transcribing: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
