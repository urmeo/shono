"""CLI and conversion guards run before any real optional runtime loads."""

import json
import os
import subprocess
import sys
import wave
from types import SimpleNamespace as NS

import pytest

from shono.transcribe.__main__ import _write_json, main
from shono.transcribe.convert import to_ct2, validate_ct2_checkpoint


def checkpoint(tmp_path, name="model", ct2=False, slow=False):
    model = tmp_path / name
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "whisper"}))
    (model / ("model.bin" if ct2 else "model.safetensors")).write_bytes(b"fake weights")
    for filename in ("preprocessor_config.json", "tokenizer_config.json"):
        (model / filename).write_text("{}")
    if slow:
        (model / "vocab.json").write_text('{"a":1}')
        (model / "merges.txt").write_text("a b\n")
    else:
        (model / "tokenizer.json").write_text("{}")
    return model


def audio_file(tmp_path):
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 10, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * 10)
    return path


def fake_converter(monkeypatch):
    calls = []

    class Converter:
        def __init__(self, source, *, copy_files):
            calls.append(("load", source, copy_files))

        def convert(self, output, *, quantization, force):
            calls.append(("convert", output, quantization, force))

    monkeypatch.setitem(sys.modules, "ctranslate2.converters", NS(TransformersConverter=Converter))
    return calls


def test_conversion_is_local_and_copies_processor_files(tmp_path, monkeypatch):
    model = checkpoint(tmp_path)
    calls = fake_converter(monkeypatch)
    output = tmp_path / "out"
    assert to_ct2(model, output, quantization="int8") == str(output)
    assert calls == [
        ("load", str(model), ["preprocessor_config.json", "tokenizer.json"]),
        ("convert", str(output), "int8", False),
    ]


def test_slow_tokenizer_is_resolved_locally_and_saved(tmp_path, monkeypatch):
    model = checkpoint(tmp_path, slow=True)
    fake_converter(monkeypatch)
    calls = []

    def load(source, *, use_fast, local_files_only):
        calls.append((source, use_fast, local_files_only))
        return NS(is_fast=True, save_pretrained=lambda output: calls.append(output))

    monkeypatch.setitem(sys.modules, "transformers", NS(AutoTokenizer=NS(from_pretrained=load)))
    output = tmp_path / "out"
    to_ct2(model, output)
    assert calls == [(str(model), True, True), str(output)]


@pytest.mark.parametrize("options", [{"force": 1}, {"quantization": "wrong"}])
def test_conversion_invalid_flags_precede_converter(tmp_path, monkeypatch, options):
    calls = fake_converter(monkeypatch)
    with pytest.raises(ValueError):
        to_ct2(checkpoint(tmp_path), tmp_path / "out", **options)
    assert calls == []


def test_conversion_rejects_source_ancestors_descendants_and_aliases(tmp_path, monkeypatch):
    model = checkpoint(tmp_path)
    calls = fake_converter(monkeypatch)
    for output in (model, model / "nested", tmp_path):
        with pytest.raises(ValueError, match="separate"):
            to_ct2(model, output, force=True)
    alias = tmp_path / "linked"
    alias.symlink_to(model, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        to_ct2(model, alias, force=True)
    target = tmp_path / "target"
    target.mkdir()
    os.link(model / "model.safetensors", target / "alias.bin")
    with pytest.raises(ValueError, match="aliases"):
        to_ct2(model, target, force=True)
    assert calls == []


def test_conversion_protects_resolved_source_file_inside_output(tmp_path, monkeypatch):
    model = checkpoint(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    weights = model / "model.safetensors"
    weights.rename(output / "weights")
    weights.symlink_to(output / "weights")
    calls = fake_converter(monkeypatch)
    with pytest.raises(ValueError, match="resolved checkpoint"):
        to_ct2(model, output, force=True)
    assert calls == []


def test_conversion_protects_repo_docs_scripts_and_case_aliases(tmp_path, monkeypatch):
    import shono.transcribe.convert as module

    project = tmp_path / "project"
    package = project / "src" / "shono" / "transcribe"
    package.mkdir(parents=True)
    (project / "pyproject.toml").write_text("fake project")
    (project / "docs").mkdir()
    (project / "docs" / "notes.txt").write_text("keep")
    (project / "scripts").mkdir()
    monkeypatch.setattr(module, "__file__", str(package / "convert.py"))
    model = checkpoint(tmp_path)
    calls = fake_converter(monkeypatch)
    for output in (project / "docs", project / "scripts", project / "Docs" / "new", project):
        with pytest.raises(ValueError, match="repository sources"):
            to_ct2(model, output, force=True)
    assert calls == []
    to_ct2(model, project / "outputs" / "converted")
    assert calls[-1][0] == "convert"


def test_incomplete_checkpoint_and_missing_runtime_have_clear_errors(tmp_path, monkeypatch):
    model = checkpoint(tmp_path)
    (model / "model.safetensors").unlink()
    calls = fake_converter(monkeypatch)
    with pytest.raises(ValueError, match="weights"):
        to_ct2(model, tmp_path / "out")
    assert calls == []
    (model / "model.safetensors").write_bytes(b"fake")
    monkeypatch.setitem(sys.modules, "ctranslate2.converters", None)
    with pytest.raises(RuntimeError, match="requires.*runtimes"):
        to_ct2(model, tmp_path / "out")


def test_json_export_aliases_reject_before_pipeline(tmp_path, monkeypatch, capsys):
    import shono.transcribe as module

    model = checkpoint(tmp_path, ct2=True)
    audio = audio_file(tmp_path)
    original = audio.read_bytes()
    monkeypatch.setattr(module, "SileroVAD", lambda: pytest.fail("VAD must remain unloaded"))
    alias = tmp_path / "alias.json"
    os.link(audio, alias)
    assert main([str(audio), "--model", str(model), "--json", str(alias)]) == 1
    alias.unlink()
    alias.symlink_to(audio)
    assert main([str(audio), "--model", str(model), "--json", str(alias)]) == 1
    assert main([str(audio), "--model", str(model), "--json", str(model / "config.json")]) == 1
    assert audio.read_bytes() == original
    assert "Traceback" not in capsys.readouterr().err


def test_incomplete_ct2_rejects_before_vad_and_module_help_is_clean(tmp_path, monkeypatch):
    import shono.transcribe as module

    model = tmp_path / "empty"
    model.mkdir()
    monkeypatch.setattr(module, "SileroVAD", lambda: pytest.fail("VAD must remain unloaded"))
    assert main([str(audio_file(tmp_path)), "--model", str(model)]) == 1
    for module_name in ("shono.transcribe", "shono.transcribe.convert", "shono.demo.app"):
        result = subprocess.run(
            [sys.executable, "-m", module_name, "--help"], capture_output=True, text=True
        )
        assert result.returncode == 0 and "usage:" in result.stdout
        assert "Traceback" not in result.stderr


def test_conversion_cli_input2_and_missing_runtime1(tmp_path, monkeypatch, capsys):
    from shono.transcribe.convert import main as convert_main

    assert convert_main([str(tmp_path / "missing"), str(tmp_path / "out")]) == 2
    model = checkpoint(tmp_path)
    monkeypatch.setitem(sys.modules, "ctranslate2.converters", None)
    assert convert_main([str(model), str(tmp_path / "out")]) == 1
    assert "Traceback" not in capsys.readouterr().err


def test_atomic_json_failure_preserves_existing_file(tmp_path, monkeypatch):
    target = tmp_path / "output.json"
    target.write_text('{"old":true}\n')

    def fail(*args):
        raise OSError("fake replace failure")

    monkeypatch.setattr("shono.transcribe.__main__.os.replace", fail)
    with pytest.raises(OSError):
        _write_json(target, {"new": "বাংলা"})
    assert target.read_text() == '{"old":true}\n'
    assert list(tmp_path.iterdir()) == [target]


def test_successful_cli_json_has_coarse_timing_and_warning(tmp_path, monkeypatch):
    import shono.transcribe as module
    from shono.transcribe.types import Transcript, TranscriptSegment

    model = checkpoint(tmp_path, ct2=True)
    audio = audio_file(tmp_path)
    original = audio.read_bytes()
    result = NS(
        transcript=Transcript(
            (TranscriptSegment(0, 1, "বাংলা", timing_precision="chunk"),), ("timing approximate",)
        ),
        rtf=0.5,
        audio_duration_s=1,
        processing_s=0.5,
        n_chunks=1,
        n_forced_splits=0,
        filtering=NS(n_dropped=0, summary=lambda: "kept all 1 chunks"),
    )
    monkeypatch.setattr(module, "SileroVAD", lambda: "fake VAD")
    monkeypatch.setattr(module, "FasterWhisperTranscriber", lambda *a, **kw: "fake decoder")
    monkeypatch.setattr(
        module, "LongFormTranscriber", lambda *a, **kw: NS(transcribe=lambda *a: result)
    )
    target = tmp_path / "output" / "transcript.json"
    assert main([str(audio), "--model", str(model), "--json", str(target)]) == 0
    data = json.loads(target.read_text())
    assert data["text"] == "বাংলা" and data["segments"][0]["timing_precision"] == "chunk"
    assert data["warnings"] == ["timing approximate"]
    assert audio.read_bytes() == original


def test_ct2_checkpoint_requires_processor_json_objects(tmp_path):
    model = checkpoint(tmp_path, ct2=True)
    assert validate_ct2_checkpoint(model) == model
    (model / "preprocessor_config.json").write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        validate_ct2_checkpoint(model)
