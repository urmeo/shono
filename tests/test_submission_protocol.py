"""Prediction metadata describes the inputs actually used by a run."""

import hashlib
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
from data_fixtures import synthetic_registry

from shono.data import Manifest, Segment
from shono.eval.report import Predictions
from shono.markdown import escape_cell
from shono.protocol import execute_matrix, run_matrix
from shono.provenance import RunContext


@pytest.fixture
def run_input(tmp_path):
    entry = next(e for e in run_matrix() if e.slug == "tugstugi-medium" and e.domain == "read")
    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "clip.wav").write_bytes(b"synthetic input; duration supplied by test")
    manifest = Manifest(
        entry.slice_name,
        "synthetic",
        "test",
        entry.domain,
        "fixture",
        (Segment("clip", "clip.wav", "কথা", 1, "recording"),),
    )
    path = tmp_path / entry.manifest
    path.parent.mkdir()
    manifest.to_jsonl(path)
    return NS(base=tmp_path, audio=audio, entry=entry, manifest=manifest, path=path)


def execute(inputs, factory, **kwargs):
    return execute_matrix(
        [inputs.entry],
        base_dir=inputs.base,
        audio_root=inputs.audio,
        registry=synthetic_registry("synthetic"),
        transcriber_for=factory,
        duration_of=lambda path: 1,
        run_models=True,
        **kwargs,
    )


def change_manifest(inputs):
    replace(
        inputs.manifest,
        segments=(replace(inputs.manifest.segments[0], text="changed reference"),),
    ).to_jsonl(inputs.path)


def test_manifest_change_during_preflight_fails_before_model_creation(run_input, monkeypatch):
    from shono.api import preflight_manifest

    def preflight(manifest, **kwargs):
        result = preflight_manifest(manifest, **kwargs)
        change_manifest(run_input)
        return result

    monkeypatch.setattr("shono.api.preflight_manifest", preflight)
    calls = []

    def factory(entry):
        calls.append(entry)
        return NS(transcribe=lambda *args: "কথা")

    with pytest.raises(ValueError, match="manifest changed"):
        execute(run_input, factory)
    assert calls == []
    assert not (run_input.base / run_input.entry.predictions).exists()


def test_manifest_change_during_inference_does_not_publish_predictions(run_input):
    def transcribe(*args):
        change_manifest(run_input)
        return "কথা"

    with pytest.raises(ValueError, match="manifest changed"):
        execute(run_input, lambda entry: NS(transcribe=transcribe))
    assert not (run_input.base / run_input.entry.predictions).exists()


def test_declared_revision_and_input_digest_are_preserved(run_input):
    revision = "a" * 40
    execute(
        run_input,
        lambda entry: NS(transcribe=lambda *args: "কথা"),
        inference_settings={run_input.entry.slug: {"model_revision": revision}},
    )
    context = Predictions.from_jsonl(run_input.base / run_input.entry.predictions).run_context
    assert context["config"]["model_revision"] == revision
    expected_digest = hashlib.sha256(run_input.path.read_bytes()).hexdigest()
    assert context["extra"]["manifest_sha256"] == expected_digest


def test_unknown_git_status_does_not_claim_a_clean_tree(monkeypatch):
    def git(command, **kwargs):
        if command[1] == "rev-parse":
            return NS(returncode=0, stdout="a" * 40)
        return NS(returncode=128, stdout="")

    monkeypatch.setattr("shono.provenance.subprocess.run", git)
    context = RunContext.capture(0, {}, packages=())
    assert context.git.sha == "a" * 40
    assert context.to_dict()["git"]["dirty"] is None


def test_backslashes_cannot_cancel_table_pipe_escaping():
    value = escape_cell("slice\\|extra\nrow")
    prefix, suffix = value.split("|")
    assert len(prefix) - len(prefix.rstrip("\\")) == 3
    assert suffix == "extra row"
