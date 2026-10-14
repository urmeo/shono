"""Regression checks for report identity, intervals and annotation parsing."""

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import pytest

from shono.data.license import DatasetLicense, LicenseRegistry
from shono.data.manifest import Manifest, Segment
from shono.demo.model_card import render_model_card
from shono.demo.pipeline import transcribe_recording
from shono.diarize import (
    DERConfig,
    SpeakerSegment,
    der,
    load_loop_csv,
    load_rttm,
    render_der_report,
)
from shono.eval.report import Predictions, ReportSpec, SliceSpec, SystemSpec, build_report
from shono.provenance import GitState, RunContext
from shono.transcribe.types import Transcript


@pytest.fixture
def report_inputs(tmp_path):
    manifest = Manifest(
        "evaluation",
        "synthetic",
        "test",
        "read",
        "fixture",
        (
            Segment("a", "a.wav", "কথা", 1.0, "recording-a"),
            Segment("b", "b.wav", "ভাষা", 1.0, "recording-b"),
        ),
    )
    path = tmp_path / "evaluation.jsonl"
    manifest.to_jsonl(path)
    context = {
        "config": {"model_id": "checkpoint-a", "inference_mode": "bounded-short"},
        "extra": {"manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
    }
    predictions = Predictions("system", manifest.name, {"a": "কথা", "b": "ভাষা"}, context)
    predictions.to_jsonl(tmp_path / "predictions.jsonl")
    spec = ReportSpec(
        "results",
        "Results",
        (SliceSpec("test", path.name),),
        (
            SystemSpec(
                "system",
                {"test": "predictions.jsonl"},
                model_id="checkpoint-a",
                inference_modes={"test": "bounded-short"},
                model_kind="pretrained",
            ),
        ),
        n_resamples=100,
    )
    registry = LicenseRegistry(
        [
            DatasetLicense(
                "synthetic",
                "Synthetic",
                "",
                "fixture",
                False,
                False,
                False,
                "local",
                "active",
                None,
                "read",
                "fixture",
                "",
                "fixture",
            )
        ]
    )
    return tmp_path, manifest, spec, registry


def test_report_rejects_predictions_for_changed_manifest_bytes(report_inputs):
    base, manifest, spec, registry = report_inputs
    altered = replace(manifest.segments[0], text="বদলানো কথা")
    replace(manifest, segments=(altered, manifest.segments[1])).to_jsonl(base / "evaluation.jsonl")

    with pytest.raises(ValueError, match="prediction manifest.*hash"):
        build_report(spec, base, registry)


def test_report_rejects_inference_mode_mismatch(report_inputs):
    base, _manifest, spec, registry = report_inputs
    wrong = replace(spec.systems[0], inference_modes={"test": "long-form-vad"})

    with pytest.raises(ValueError, match="inference mode"):
        build_report(replace(spec, systems=(wrong,)), base, registry)


def test_report_retains_recorded_model_and_mode_when_spec_is_unknown(report_inputs):
    base, _manifest, spec, registry = report_inputs
    unspecified = replace(spec.systems[0], model_id="unknown", inference_modes={})
    report = build_report(replace(spec, systems=(unspecified,)), base, registry)

    cell = report.to_json_dict()["cells"][0]
    assert cell["model_id"] == "checkpoint-a"
    assert cell["inference_mode"] == "bounded-short"


@pytest.mark.parametrize("seed,n_resamples", [(1, 100), (0, 200)])
def test_report_context_must_match_actual_bootstrap_settings(report_inputs, seed, n_resamples):
    base, _manifest, spec, registry = report_inputs
    context = RunContext.capture(seed, {"n_resamples": n_resamples}, packages=())

    with pytest.raises(ValueError, match="report context"):
        build_report(spec, base, registry, run_context=context)


@pytest.mark.parametrize("export", ["render_markdown", "to_json_dict"])
@pytest.mark.parametrize(
    "change",
    [
        {"point": 0.5},
        {"metric": "cer"},
        {"confidence": 0.8},
        {"n_blocks": 3},
        {"n_segments": 1},
        {"n_resamples": 200},
    ],
)
def test_report_rejects_mismatched_interval_metadata(report_inputs, export, change):
    base, _manifest, spec, registry = report_inputs
    report = build_report(spec, base, registry)
    cell = report.cells[0]
    broken = replace(cell, wer_ci=replace(cell.wer_ci, **change))

    with pytest.raises(ValueError, match="interval"):
        getattr(replace(report, cells=(broken,)), export)()


def test_report_escapes_normalization_policy(report_inputs):
    base, _manifest, spec, registry = report_inputs
    report = build_report(spec, base, registry)
    cell = replace(report.cells[0], normalization_policy="custom|policy\nnext")

    assert r"custom\|policy next v1.1.0 |" in replace(report, cells=(cell,)).render_markdown()


@pytest.mark.parametrize("export", ["render_markdown", "to_json_dict"])
@pytest.mark.parametrize("kind", ["duplicate", "extra", "missing", "duplicate-order"])
def test_report_exports_require_one_cell_per_declared_system_and_slice(report_inputs, export, kind):
    base, _manifest, spec, registry = report_inputs
    report = build_report(spec, base, registry)
    if kind == "duplicate":
        report = replace(report, cells=report.cells * 2)
    elif kind == "extra":
        report = replace(report, cells=(*report.cells, replace(report.cells[0], system="hidden")))
    elif kind == "missing":
        report = replace(report, cells=())
    else:
        report = replace(report, slice_order=report.slice_order * 2)

    with pytest.raises(ValueError, match="report.*cell"):
        getattr(report, export)()


@pytest.mark.parametrize("label", ["system|extra", "system\nextra\rrow"])
def test_der_report_preserves_table_shape(label):
    config = DERConfig(collar_s=0)
    result = der([SpeakerSegment(0, 1, "A")], [], config)
    report = render_der_report([(label, result, None)], config)

    expected = label.replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    assert f"| {expected} | 100.0% |" in report
    assert len([line for line in report.splitlines() if line.startswith("|")]) == 3


@pytest.mark.parametrize("suffix", ['"1', '"1"trailing'])
def test_loop_csv_rejects_malformed_quotes(tmp_path, suffix):
    path = tmp_path / "annotations.csv"
    path.write_text(
        "start_time,end_time,speaker_id\n" + f"00:00:00,00:00:02,{suffix}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="CSV.*line 2"):
        load_loop_csv(path)


def test_rttm_rejects_absent_recording_selector(tmp_path):
    path = tmp_path / "hypothesis.rttm"
    path.write_text("SPEAKER recording-a 1 0 2 <NA> <NA> A\n", encoding="utf-8")

    with pytest.raises(ValueError, match="recording-b.*not found"):
        load_rttm(path, file_id="recording-b")


def test_rttm_allows_empty_hypothesis_for_selected_recording(tmp_path):
    path = tmp_path / "hypothesis.rttm"
    path.write_text("", encoding="utf-8")

    assert load_rttm(path, file_id="recording-a") == []


def _card(report):
    return render_model_card(
        report,
        model_id="urmeo/publication-alias",
        system="system",
        base_model="base",
        license_spdx="unknown",
        base_model_license="unknown",
        training_status="unknown",
        repo_url="https://github.com/urmeo/shono",
    )


def test_model_card_names_evaluated_checkpoint_separately_from_publication_alias(report_inputs):
    base, _manifest, spec, registry = report_inputs
    report = build_report(spec, base, registry)
    card = _card(report.to_json_dict())

    assert "# urmeo/publication-alias" in card
    assert "| Evaluated checkpoint |" in card
    assert "| test | 0.0% [0.0%, 0.0%] | bengali v1.1.0 | checkpoint-a |" in card


def test_model_card_rejects_conflicting_evaluated_checkpoints(report_inputs):
    base, _manifest, spec, registry = report_inputs
    payload = build_report(spec, base, registry).to_json_dict()
    payload["slices"].append("second")
    payload["cells"].append({**payload["cells"][0], "slice": "second", "model_id": "checkpoint-b"})

    with pytest.raises(ValueError, match="conflicting evaluated checkpoints"):
        _card(payload)


def test_model_card_escapes_evaluated_checkpoint(report_inputs):
    base, _manifest, spec, registry = report_inputs
    payload = build_report(spec, base, registry).to_json_dict()
    payload["cells"][0]["model_id"] = "checkpoint|extra\nrow"

    assert "| checkpoint\\|extra row |" in _card(payload)


def test_empty_transcript_does_not_load_diarization_model():
    transcript = Transcript(())
    pipeline = SimpleNamespace(
        transcribe=lambda *_: SimpleNamespace(transcript=transcript, speech=())
    )

    def unexpected_load(_path):
        raise AssertionError("empty transcripts need no speaker labels")

    diarizer = SimpleNamespace(diarize=unexpected_load)
    assert transcribe_recording("silence.wav", 1, pipeline, diarizer) == transcript


def test_report_distinguishes_unknown_working_tree_from_clean(report_inputs):
    base, _manifest, spec, registry = report_inputs
    report = build_report(spec, base, registry)
    context = replace(report.run_context, git=GitState("a" * 40, None))

    assert "(working tree unknown)" in replace(report, run_context=context).render_markdown()
