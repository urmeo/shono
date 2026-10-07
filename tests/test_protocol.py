"""Offline matrix identity, preflight, output preservation and notebook drivers."""

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from shono.api import BudgetError
from shono.data import Manifest, Segment
from shono.data.license import DatasetLicense, LicenseRegistry
from shono.eval.report import Predictions
from shono.protocol import BASE_MODEL_ID, BASE_SYSTEM, execute_matrix, report_spec, run_matrix

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def registry():
    return LicenseRegistry(
        [
            DatasetLicense(
                "fixture",
                "Fixture",
                "",
                "fixture",
                False,
                False,
                False,
                "local",
                "active",
                None,
                "read",
                "v",
                "Synthetic only",
                "fixture",
            )
        ]
    )


class FakeTranscriber:
    def transcribe(self, path, start_s, duration_s):
        return "কথা"


def prepare(base, entries, *, duration=1.0, start=None):
    (base / "data").mkdir(exist_ok=True)
    (base / "audio").mkdir(exist_ok=True)
    for entry in entries:
        audio = entry.slice_name + ".wav"
        (base / "audio" / audio).write_bytes(b"fixture audio; not decoded")
        manifest = Manifest(
            entry.slice_name,
            "fixture",
            "test",
            entry.domain,
            "v",
            (
                Segment(
                    entry.slice_name + "-0", audio, "কথা", duration, entry.slice_name, start_s=start
                ),
            ),
        )
        manifest.to_jsonl(base / entry.manifest)
    return base / "audio"


def test_canonical_base_is_not_generic_or_regional_checkpoint():
    assert BASE_MODEL_ID == "bengaliAI/tugstugi_bengaliai-asr_whisper-medium"
    assert BASE_SYSTEM == "tugstugi whisper-medium (base)"
    assert "regional" not in BASE_MODEL_ID
    for name in ("baselines", "longform", "codeswitch", "full"):
        actual = json.loads((REPO / f"reports/specs/{name}.json").read_text())
        assert actual == report_spec(name)
        base = next(system for system in actual["systems"] if system["name"] == BASE_SYSTEM)
        assert base["model_id"] == BASE_MODEL_ID


def test_every_spec_cell_is_in_the_run_matrix_with_same_paths_and_modes():
    entries = {(entry.system, entry.slice_name): entry for entry in run_matrix()}
    for name in ("baselines", "longform", "codeswitch", "full"):
        spec = report_spec(name)
        for system in spec["systems"]:
            for item in spec["slices"]:
                entry = entries[system["name"], item["name"]]
                assert entry.predictions == system["predictions"][item["name"]]
                assert entry.manifest == item["manifest"]
                assert entry.inference_mode == system["inference_modes"][item["name"]]
    assert all(
        entry.inference_mode == "long-form-vad"
        for entry in entries.values()
        if entry.domain == "long-form" and entry.provider == "local"
    )
    assert all(
        entry.inference_mode == "bounded-short"
        for entry in entries.values()
        if entry.slug == "shono-medium" and entry.domain != "long-form"
    )
    assert all(
        not entry.supported
        for entry in entries.values()
        if entry.provider == "google" and entry.domain == "long-form"
    )


def test_defaults_construct_no_models_or_clients(tmp_path, registry):
    calls = []
    outcomes = execute_matrix(
        run_matrix(),
        base_dir=tmp_path,
        audio_root=tmp_path,
        registry=registry,
        transcriber_for=lambda entry: calls.append(entry),
    )
    assert not calls and all(outcome.status == "pending" for outcome in outcomes)
    assert not (tmp_path / "outputs").exists()


def test_complete_fake_matrix_matches_all_supported_headers(tmp_path, registry):
    entries = run_matrix(fine_tuned_model="explicit-fixture-checkpoint")
    root = prepare(tmp_path, entries)
    calls = []

    def factory(entry):
        calls.append(entry)
        return FakeTranscriber()

    outcomes = execute_matrix(
        entries,
        base_dir=tmp_path,
        audio_root=root,
        registry=registry,
        transcriber_for=factory,
        duration_of=lambda _: 1,
        run_models=True,
        run_apis=True,
        allowances={"google": 1, "deepgram": 1},
        inference_settings={"tugstugi-medium": {"beam_size": 5}},
    )
    supported = {entry for entry in entries if entry.supported}
    assert set(calls) == supported
    assert {outcome.entry for outcome in outcomes if outcome.status == "completed"} == supported
    for entry in supported:
        predictions = Predictions.from_jsonl(tmp_path / entry.predictions)
        assert predictions.system == entry.system and predictions.manifest == entry.slice_name
        assert predictions.hypotheses == {entry.slice_name + "-0": "কথা"}
        assert predictions.run_context["config"]["model_id"] == entry.model_id
        assert predictions.run_context["config"]["inference_mode"] == entry.inference_mode
        assert len(predictions.run_context["extra"]["manifest_sha256"]) == 64


def test_provider_total_is_checked_before_first_factory(tmp_path, registry):
    entries = tuple(entry for entry in run_matrix() if entry.provider == "deepgram")[:2]
    root = prepare(tmp_path, entries)
    calls = []
    with pytest.raises(BudgetError):
        execute_matrix(
            entries,
            base_dir=tmp_path,
            audio_root=root,
            registry=registry,
            transcriber_for=lambda entry: calls.append(entry),
            duration_of=lambda _: 1,
            run_apis=True,
            allowances={"deepgram": 1.5 / 3600},
        )
    assert not calls and not (tmp_path / "outputs").exists()


def test_invalid_later_input_prevents_earlier_requests(tmp_path, registry):
    entries = tuple(entry for entry in run_matrix() if entry.provider == "deepgram")[:2]
    root = prepare(tmp_path, entries)
    (root / (entries[1].slice_name + ".wav")).unlink()
    calls = []
    with pytest.raises(ValueError, match="missing audio"):
        execute_matrix(
            entries,
            base_dir=tmp_path,
            audio_root=root,
            registry=registry,
            transcriber_for=lambda entry: calls.append(entry),
            duration_of=lambda _: 1,
            run_apis=True,
            allowances={"deepgram": 1},
        )
    assert not calls


@pytest.mark.parametrize(
    "kind", ["symlink", "hardlink", "root-symlink", "cyclic", "planned-alias", "case-alias"]
)
def test_output_aliases_cannot_overwrite_audio_or_construct_adapters(tmp_path, registry, kind):
    entry = next(entry for entry in run_matrix() if entry.provider == "deepgram")
    root = prepare(tmp_path, [entry])
    source = root / (entry.slice_name + ".wav")
    before = source.read_bytes()
    out = tmp_path / "outputs"
    entries = [entry]
    if kind == "root-symlink":
        out.symlink_to(root, target_is_directory=True)
    elif kind == "cyclic":
        out.symlink_to("outputs", target_is_directory=True)
    else:
        out.mkdir()
        if kind == "symlink":
            (tmp_path / entry.predictions).symlink_to(source)
        elif kind == "hardlink":
            os.link(source, tmp_path / entry.predictions)
        elif kind == "planned-alias":
            entries.append(replace(entry, slug="other", slice_name="other"))
        else:
            entries.append(
                replace(
                    entry,
                    slug="other",
                    slice_name="other",
                    predictions=entry.predictions.upper().replace("OUTPUTS/", "outputs/"),
                )
            )
        if len(entries) > 1:
            other = entries[1]
            original = Manifest.from_jsonl(tmp_path / entry.manifest)
            other = replace(other, manifest="data/other.manifest.jsonl")
            entries[1] = other
            replace(original, name="other").to_jsonl(tmp_path / other.manifest)
    calls = []
    with pytest.raises(ValueError):
        execute_matrix(
            entries,
            base_dir=tmp_path,
            audio_root=root,
            registry=registry,
            transcriber_for=lambda entry: calls.append(entry),
            duration_of=lambda _: 1,
            run_apis=True,
            allowances={"deepgram": 1},
        )
    assert not calls and source.read_bytes() == before


def test_existing_generated_regular_file_can_be_replaced(tmp_path, registry):
    entry = next(entry for entry in run_matrix() if entry.provider == "deepgram")
    root = prepare(tmp_path, [entry])
    out = tmp_path / "outputs"
    out.mkdir()
    path = tmp_path / entry.predictions
    path.write_text("old generated output")
    result = execute_matrix(
        [entry],
        base_dir=tmp_path,
        audio_root=root,
        registry=registry,
        transcriber_for=lambda _: FakeTranscriber(),
        duration_of=lambda _: 1,
        run_apis=True,
        allowances={"deepgram": 1},
    )
    assert result[0].status == "completed" and Predictions.from_jsonl(path).system == entry.system


@pytest.mark.parametrize("duration,slug", [(31, "tugstugi-medium"), (60, "google-chirp")])
def test_actual_window_limits_reject_before_factory(tmp_path, registry, duration, slug):
    entry = next(entry for entry in run_matrix() if entry.slug == slug and entry.domain == "read")
    root = prepare(tmp_path, [entry], duration=duration)
    calls = []
    with pytest.raises(ValueError, match="windows"):
        execute_matrix(
            [entry],
            base_dir=tmp_path,
            audio_root=root,
            registry=registry,
            transcriber_for=lambda entry: calls.append(entry),
            duration_of=lambda _: duration,
            run_models=True,
            run_apis=True,
            allowances={"google": 1},
        )
    assert not calls


def test_longform_nonzero_crop_is_rejected_before_factory(tmp_path, registry):
    entry = next(entry for entry in run_matrix() if entry.inference_mode == "long-form-vad")
    root = prepare(tmp_path, [entry], start=1)
    calls = []
    with pytest.raises(ValueError, match="whole-file"):
        execute_matrix(
            [entry],
            base_dir=tmp_path,
            audio_root=root,
            registry=registry,
            transcriber_for=lambda entry: calls.append(entry),
            duration_of=lambda _: 2,
            run_models=True,
        )
    assert not calls


def test_excluded_inputs_stay_pending_without_factory(tmp_path, registry):
    entry = next(entry for entry in run_matrix() if entry.provider == "local")
    root = prepare(tmp_path, [entry])
    excluded = LicenseRegistry([replace(registry.require("fixture"), status="excluded")])
    calls = []
    outcomes = execute_matrix(
        [entry],
        base_dir=tmp_path,
        audio_root=root,
        registry=excluded,
        transcriber_for=lambda entry: calls.append(entry),
        run_models=True,
    )
    assert not calls and outcomes[0].status == "pending" and "permission" in outcomes[0].reason


@pytest.mark.parametrize(
    "kwargs",
    [
        {"run_apis": "false"},
        {"allow_paid": True, "allowances": {"deepgram": True}},
        {"allowances": {"deepgram": float("nan")}},
        {"seed": True},
    ],
)
def test_invalid_protocol_controls_do_not_construct_clients(tmp_path, registry, kwargs):
    calls = []
    with pytest.raises(ValueError):
        execute_matrix(
            run_matrix(),
            base_dir=tmp_path,
            audio_root=tmp_path,
            registry=registry,
            transcriber_for=lambda entry: calls.append(entry),
            **kwargs,
        )
    assert not calls


def test_prediction_writer_preserves_audio_inputs(tmp_path):
    source = tmp_path / "audio.wav"
    source.write_bytes(b"original")
    predictions = Predictions("s", "m", {"a": "text"}, input_paths=(str(source),))
    with pytest.raises(ValueError, match="overwrite input"):
        predictions.to_jsonl(source)
    assert source.read_bytes() == b"original"


@pytest.mark.parametrize("name", ["baselines", "predict"])
def test_driver_notebook_compile_and_default_off(name):
    notebook = json.loads((REPO / f"notebooks/{name}.ipynb").read_text())
    sources = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            compile(source, f"{name}:{index}", "exec")
            sources.append(source)
            assert cell["outputs"] == [] and cell["execution_count"] is None
    source = "\n".join(sources)
    assert "RUN_MODELS = False" in source and "INSTALL_RUNTIME = False" in source
    assert "execute_matrix(" in source and "run_matrix(" in source
    assert "openai/whisper-medium" not in source and "next((loop" not in source
    assert "to_jsonl(DATA" not in source and "'reports' / 'predictions'" not in source
    if name == "predict":
        assert "RUN_APIS = False" in source
        assert "{'google': 0.0, 'deepgram': 0.0}" in source
        assert "sample = paths[0]" in source and "load_loop_csv" in source


@pytest.mark.parametrize("name", ["baselines", "predict"])
def test_notebook_local_factory_rebuilds_once_from_explicit_checkpoint(tmp_path, name):
    import ast

    notebook = json.loads((REPO / f"notebooks/{name}.ipynb").read_text())
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "def local_transcriber" in "".join(cell["source"])
    )
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "local_transcriber"
    )
    module = ast.Module(body=[function], type_ignores=[])
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    ct2 = tmp_path / "ct2"
    ct2.mkdir()
    conversions = []
    pipelines = []

    def convert(model, output, **kwargs):
        conversions.append((model, output, kwargs))

    def pipeline(vad, transcriber, **kwargs):
        pipelines.append(kwargs)
        return "pipeline"

    namespace = {
        "Path": Path,
        "LOCAL_CHECKPOINTS": {"tugstugi-medium": checkpoint},
        "CT2_ROOT": ct2,
        "CONVERTED_MODELS": {},
        "to_ct2": convert,
        "SileroVAD": lambda: "vad",
        "FasterWhisperTranscriber": lambda *args, **kwargs: "decoder",
        "LongFormTranscriber": pipeline,
        "LongFormPipelineTranscriber": lambda item: item,
    }
    exec(compile(module, "factory", "exec"), namespace)
    entry = next(
        entry
        for entry in run_matrix()
        if entry.slug == "tugstugi-medium" and entry.domain == "long-form"
    )
    factory = namespace["local_transcriber"]
    assert factory(entry) == "pipeline" and factory(entry) == "pipeline"
    assert conversions == [
        (checkpoint, ct2 / "tugstugi-medium", {"quantization": "float16", "force": True})
    ]
    assert pipelines == [{"max_chunk_s": 30.0, "pad_s": 0.25}] * 2
    namespace["LOCAL_CHECKPOINTS"].clear()
    with pytest.raises(ValueError, match="local checkpoint"):
        factory(entry)
    assert len(conversions) == 1


@pytest.mark.parametrize("name", ["baselines", "predict"])
def test_notebook_short_factory_uses_bounded_adapter(tmp_path, name):
    import ast

    notebook = json.loads((REPO / f"notebooks/{name}.ipynb").read_text())
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "def local_transcriber" in "".join(cell["source"])
    )
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "local_transcriber"
    )
    calls = []

    def short(model, **kwargs):
        calls.append((model, kwargs))
        return "short"

    namespace = {"ShortFormWhisperTranscriber": short}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "factory", "exec"), namespace)
    entry = next(
        entry
        for entry in run_matrix()
        if entry.slug == "tugstugi-medium" and entry.domain == "read"
    )
    assert namespace["local_transcriber"](entry) == "short"
    assert calls == [(BASE_MODEL_ID, {"language": "bn", "device": "cuda"})]
