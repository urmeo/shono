"""Convert an explicit local Whisper checkpoint into a separate CTranslate2 directory."""

from __future__ import annotations

import json
from pathlib import Path

from shono.output_paths import source_files

_QUANTIZATIONS = {
    "float32",
    "float16",
    "bfloat16",
    "int8",
    "int8_float32",
    "int8_float16",
    "int8_bfloat16",
    "int16",
}


def _conversion_paths(
    model_dir: str | Path, output_dir: str | Path, force: bool
) -> tuple[Path, Path]:
    try:
        if Path(output_dir).is_symlink():
            raise ValueError("conversion output must not be a symlink")
        source, output = Path(model_dir).resolve(), Path(output_dir).resolve()
        if not source.is_dir():
            raise ValueError("model_dir must be an existing local Whisper checkpoint directory")
        config = source / "config.json"
        if (
            not config.is_file()
            or json.loads(config.read_text(encoding="utf-8")).get("model_type") != "whisper"
        ):
            raise ValueError("local checkpoint requires a Whisper config.json")
        weights = [
            source / name
            for name in ("model.safetensors", "pytorch_model.bin")
            if (source / name).is_file()
        ]
        for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            index = source / name
            if index.is_file():
                mapping = json.loads(index.read_text(encoding="utf-8")).get("weight_map")
                if not isinstance(mapping, dict) or not mapping:
                    raise ValueError("checkpoint index requires a nonempty weight_map")
                for filename in set(mapping.values()):
                    shard = (source / filename).resolve()
                    if not shard.is_relative_to(source) or not shard.is_file():
                        raise ValueError("checkpoint weight shard is missing or escapes model_dir")
                    weights.append(shard)
        if not weights or any(path.stat().st_size == 0 for path in weights):
            raise ValueError("local checkpoint requires nonempty model weights")
        for name in ("preprocessor_config.json", "tokenizer_config.json"):
            path = source / name
            if not path.is_file() or not isinstance(
                json.loads(path.read_text(encoding="utf-8")), dict
            ):
                raise ValueError(f"local checkpoint requires {name}")
        tokenizer = source / "tokenizer.json"
        slow_files = (source / "vocab.json", source / "merges.txt")
        if not tokenizer.is_file() and not all(
            path.is_file() and path.stat().st_size for path in slow_files
        ):
            raise ValueError(
                "local checkpoint requires tokenizer.json or vocab.json plus merges.txt"
            )

        def inside(path: Path, directory: Path) -> bool:
            return Path(str(path).casefold()).is_relative_to(Path(str(directory).casefold()))

        if inside(output, source) or inside(source, output):
            raise ValueError("conversion output must be separate from the source checkpoint")
        inputs = [path.resolve() for path in source.rglob("*") if path.is_file()]
        if any(path.is_relative_to(output) for path in inputs):
            raise ValueError("conversion output contains resolved checkpoint files")
        package = Path(__file__).resolve().parents[1]
        protected = [package]
        protected_files = list(inputs)
        project = package.parent.parent
        if (project / "pyproject.toml").is_file():
            protected_files.extend(path.resolve() for path in source_files(project))
            protected.extend(
                project / name
                for name in (
                    "src",
                    "data",
                    "docs",
                    "scripts",
                    "tests",
                    "notebooks",
                    "reports",
                    ".github",
                )
            )
        if any(inside(output, path) or inside(path, output) for path in protected):
            raise ValueError("conversion output cannot replace repository sources")
        if any(path.is_relative_to(output) for path in protected_files):
            raise ValueError("conversion output cannot replace repository sources")
        if output.exists():
            if not output.is_dir():
                raise ValueError("conversion output must be a directory")
            if not force:
                raise ValueError("conversion output exists; use force=True to replace it")
            if any(
                target.samefile(path)
                for target in output.rglob("*")
                if target.is_file()
                for path in protected_files
            ):
                raise ValueError("conversion output contains aliases of input or source files")
        return source, output
    except (OSError, RuntimeError, TypeError, AttributeError) as exc:
        raise ValueError(f"cannot validate conversion paths: {exc}") from exc


def to_ct2(
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    quantization: str = "float16",
    force: bool = False,
) -> str:
    """Validate local inputs before loading the optional converter; no remote IDs."""
    if not isinstance(force, bool):
        raise ValueError("force must be boolean")
    if not isinstance(quantization, str) or quantization not in _QUANTIZATIONS:
        raise ValueError(f"unsupported quantization; choose from {sorted(_QUANTIZATIONS)}")
    source, output = _conversion_paths(model_dir, output_dir, force)
    try:
        from ctranslate2.converters import TransformersConverter

        tokenizer = None
        copies = ["preprocessor_config.json"]
        if (source / "tokenizer.json").is_file():
            copies.append("tokenizer.json")
        else:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                str(source), use_fast=True, local_files_only=True
            )
            if not tokenizer.is_fast:
                raise ValueError("local checkpoint must support a fast tokenizer for CT2 export")
    except ImportError as exc:
        raise RuntimeError("conversion requires the CTranslate2 and Transformers runtimes") from exc
    converter = TransformersConverter(str(source), copy_files=copies)
    converter.convert(str(output), quantization=quantization, force=force)
    if tokenizer is not None:
        tokenizer.save_pretrained(str(output))
    return str(output)


def validate_ct2_checkpoint(model_dir: str | Path) -> Path:
    """Require local model and tokenizer files before optional inference loads."""
    try:
        source = Path(model_dir).resolve()
        required = ("model.bin", "config.json", "tokenizer.json", "preprocessor_config.json")
        if not source.is_dir() or any(
            not (source / name).is_file() or (source / name).stat().st_size == 0
            for name in required
        ):
            raise ValueError(
                "local CTranslate2 checkpoint requires model.bin, config.json, "
                "tokenizer.json and preprocessor_config.json; convert the checkpoint first"
            )
        for name in required[1:]:
            if not isinstance(json.loads((source / name).read_text(encoding="utf-8")), dict):
                raise ValueError(f"CTranslate2 {name} must contain a JSON object")
        return source
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"cannot validate CTranslate2 checkpoint: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="python -m shono.transcribe.convert", allow_abbrev=False)
    parser.add_argument("model", help="existing local Whisper checkpoint directory")
    parser.add_argument("output", help="separate CTranslate2 output directory")
    parser.add_argument("--quantization", choices=sorted(_QUANTIZATIONS), default="float16")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        _conversion_paths(args.model, args.output, args.force)
    except ValueError as exc:
        print(f"error converting: {exc}", file=sys.stderr)
        return 2
    try:
        print(to_ct2(args.model, args.output, quantization=args.quantization, force=args.force))
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"error converting: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
