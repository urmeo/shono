"""Complete-cell scores, pending inputs and report release guards."""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from shono.data.license import DatasetLicense, LicenseRegistry
from shono.data.manifest import Manifest, Segment
from shono.eval.__main__ import main
from shono.eval.report import (
    Predictions,
    ReportSpec,
    build_report,
    score_slice,
    write_report,
)

_FIXTURES = Path(__file__).parent / "fixtures"
_TINY = (
    (_FIXTURES / "tiny_manifest.jsonl")
    .read_text(encoding="utf-8")
    .replace('"source": "bengali_loop"', '"source": "synthetic_fixture"')
)
_LICENSES = Path(__file__).resolve().parents[1] / "data" / "licenses.json"

_IDS = ["loop01-000", "loop01-001", "loop01-002", "loop02-000", "loop02-001"]
_REFS = {
    "loop01-000": "আজকে আমরা কথা বলব বাংলা ভাষা নিয়ে",
    "loop01-001": "ভাষা হল যোগাযোগের প্রধান মাধ্যম",
    "loop01-002": "চলুন শুরু করা যাক",
    "loop02-000": "দ্বিতীয় রেকর্ডিং এ স্বাগতম",
    "loop02-001": "আমরা দুইশো ছাব্বিশ সালে আছি",
}


def _predictions_file(path: Path, hyps: dict[str, str], system: str) -> None:
    lines = [json.dumps({"predictions": {"system": system, "manifest": "tiny-longform-test"}})]
    lines += [json.dumps({"id": k, "hypothesis": v}, ensure_ascii=False) for k, v in hyps.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def registry() -> LicenseRegistry:
    return LicenseRegistry([_synthetic_license()])


def _synthetic_license():
    return DatasetLicense(
        "synthetic_fixture",
        "Synthetic text fixture",
        "",
        "fixture only",
        False,
        False,
        False,
        "local",
        "active",
        None,
        "long-form",
        "fixture-1",
        "No dataset/audio grant is represented",
        "fixture",
    )


@pytest.fixture
def base(tmp_path: Path) -> Path:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "tiny.manifest.jsonl").write_text(_TINY, encoding="utf-8")
    (tmp_path / "data" / "licenses.json").write_text(
        json.dumps({"licenses": [asdict(_synthetic_license())]}), encoding="utf-8"
    )
    (tmp_path / "reports" / "specs").mkdir(parents=True)
    (tmp_path / "reports" / "predictions").mkdir()
    return tmp_path


def test_perfect_predictions_score_zero(registry):
    manifest = Manifest.from_jsonl(_FIXTURES / "tiny_manifest.jsonl")
    preds = Predictions("perfect", "tiny-longform-test", dict(_REFS))
    cell = score_slice(manifest, preds, n_resamples=200)
    assert cell.status == "scored"
    assert cell.score.wer_raw == 0.0
    assert cell.wer_ci.point == 0.0 and cell.wer_ci.upper == 0.0
    assert cell.n_recordings == 2


def test_ci_point_matches_normalized_score():
    manifest = Manifest.from_jsonl(_FIXTURES / "tiny_manifest.jsonl")
    hyps = dict(_REFS)
    hyps["loop01-000"] = "আজকে আমরা কথা বলব ইংরেজি ভাষা নিয়ে"
    cell = score_slice(manifest, Predictions("s", "tiny-longform-test", hyps), n_resamples=300)
    assert cell.wer_ci.point == pytest.approx(cell.score.wer_normalized)
    assert cell.score.wer_raw > 0.0


def test_incomplete_predictions_are_an_error_not_a_partial_score():
    manifest = Manifest.from_jsonl(_FIXTURES / "tiny_manifest.jsonl")
    partial = {k: v for k, v in _REFS.items() if k != "loop02-001"}
    with pytest.raises(KeyError, match="no hypothesis"):
        score_slice(manifest, Predictions("s", "tiny-longform-test", partial), n_resamples=200)


def test_extra_predictions_are_rejected():
    manifest = Manifest.from_jsonl(_FIXTURES / "tiny_manifest.jsonl")
    hyps = dict(_REFS)
    hyps["ghost-id"] = "স্টেল অনুমান"
    with pytest.raises(ValueError, match="not in the manifest"):
        score_slice(manifest, Predictions("s", "tiny-longform-test", hyps), n_resamples=200)


def test_mispaired_predictions_manifest_name_rejected():
    manifest = Manifest.from_jsonl(_FIXTURES / "tiny_manifest.jsonl")
    with pytest.raises(ValueError, match="mispaired"):
        score_slice(manifest, Predictions("s", "some-other-manifest", dict(_REFS)), n_resamples=200)


def test_single_recording_slice_scored_without_ci():
    one = Manifest(
        name="single",
        source="openslr_slr53",
        split="test",
        domain="read",
        version="v",
        segments=(
            Segment(id="a", audio="r.wav", text="আমি ভাত খাই", duration_s=2.0, recording_id="r"),
            Segment(id="b", audio="r.wav", text="তুমি কেমন আছ", duration_s=2.0, recording_id="r"),
        ),
    )
    preds = Predictions("s", "single", {"a": "আমি ভাত খাই", "b": "তুমি কেমন আছ"})
    cell = score_slice(one, preds, n_resamples=200)
    assert cell.status == "scored"
    assert cell.wer_ci is None and cell.cer_ci is None
    assert cell.score.wer_normalized == 0.0


def test_predictions_require_header(tmp_path):
    bad = tmp_path / "p.jsonl"
    bad.write_text(json.dumps({"id": "x", "hypothesis": "y"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        Predictions.from_jsonl(bad)


def test_predictions_round_trip(tmp_path):
    preds = Predictions(
        "sys-a", "cv-bn-test", {"s0": "কথা এক", "s1": "কথা দুই"}, run_context={"seed": 0}
    )
    path = tmp_path / "p.jsonl"
    preds.to_jsonl(path)
    loaded = Predictions.from_jsonl(path)
    assert loaded.system == "sys-a"
    assert loaded.manifest == "cv-bn-test"
    assert loaded.hypotheses == preds.hypotheses


def test_duplicate_prediction_id_rejected(tmp_path):
    p = tmp_path / "p.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps({"predictions": {"system": "s", "manifest": "m"}}),
                json.dumps({"id": "a", "hypothesis": "first"}),
                json.dumps({"id": "a", "hypothesis": "second"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate id"):
        Predictions.from_jsonl(p)


def _spec(base: Path, systems) -> ReportSpec:
    spec = {
        "name": "t",
        "title": "Test report",
        "n_resamples": 200,
        "slices": [{"name": "tiny", "manifest": "data/tiny.manifest.jsonl", "domain": "long-form"}],
        "systems": systems,
    }
    path = base / "reports" / "specs" / "t.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return ReportSpec.from_json(path)


def test_pending_cell_renders_pending(base, registry):
    spec = _spec(base, [{"name": "no-preds-yet", "predictions": {}}])
    report = build_report(spec, base, registry)
    assert report.cells[0].is_pending
    md = report.render_markdown()
    assert "| no-preds-yet | pending | pending | pending | pending | bengali v1.1.0 |" in md


def test_scored_cell_renders_numbers(base, registry):
    _predictions_file(base / "reports" / "predictions" / "s__tiny.jsonl", dict(_REFS), "sys-a")
    spec = _spec(
        base,
        [{"name": "sys-a", "predictions": {"tiny": "reports/predictions/s__tiny.jsonl"}}],
    )
    report = build_report(spec, base, registry)
    cell = report.cell("sys-a", "tiny")
    assert not cell.is_pending
    md = report.render_markdown()
    assert "0.0% [0.0%, 0.0%]" in md
    assert "## Data licenses" in md


def test_missing_manifest_makes_slice_pending(base, registry):
    spec = {
        "name": "t",
        "title": "T",
        "n_resamples": 200,
        "slices": [{"name": "future", "manifest": "data/not-yet.manifest.jsonl", "domain": "read"}],
        "systems": [{"name": "sys", "predictions": {}}],
    }
    path = base / "reports" / "specs" / "t.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    report = build_report(ReportSpec.from_json(path), base, registry)
    assert report.cells[0].is_pending
    assert report.cells[0].domain == "read"


def test_cli_writes_md_and_json(base):
    _predictions_file(base / "reports" / "predictions" / "s__tiny.jsonl", dict(_REFS), "sys-a")
    _spec(base, [{"name": "sys-a", "predictions": {"tiny": "reports/predictions/s__tiny.jsonl"}}])
    rc = main(["--report", "t", "--base-dir", str(base)])
    assert rc == 0
    assert (base / "outputs" / "t.md").exists()
    payload = json.loads((base / "outputs" / "t.json").read_text(encoding="utf-8"))
    assert payload["normalizer_version"]
    assert payload["cells"][0]["status"] == "scored"


def test_cli_unknown_report_exits_2(base):
    rc = main(["--report", "does-not-exist", "--base-dir", str(base)])
    assert rc == 2


def test_cli_incomplete_predictions_exits_1_and_writes_nothing(base):
    partial = {k: v for k, v in _REFS.items() if k != "loop02-001"}
    _predictions_file(base / "reports" / "predictions" / "s__tiny.jsonl", partial, "sys-a")
    _spec(base, [{"name": "sys-a", "predictions": {"tiny": "reports/predictions/s__tiny.jsonl"}}])
    rc = main(["--report", "t", "--base-dir", str(base)])
    assert rc == 1
    assert not (base / "outputs" / "t.md").exists()


def test_cli_real_baselines_spec_is_valid_skeleton(base):
    repo = Path(__file__).resolve().parents[1]
    spec_src = (repo / "reports" / "specs" / "baselines.json").read_text(encoding="utf-8")
    (base / "reports" / "specs" / "baselines.json").write_text(spec_src, encoding="utf-8")
    rc = main(["--report", "baselines", "--base-dir", str(base)])
    assert rc == 0
    payload = json.loads((base / "outputs" / "baselines.json").read_text(encoding="utf-8"))
    assert len(payload["cells"]) == 9
    assert all(c["status"] == "pending" for c in payload["cells"])


def test_system_mismatch_fails_before_writing(base, registry):
    _predictions_file(base / "reports/predictions/wrong.jsonl", dict(_REFS), "wrong")
    spec = _spec(
        base, [{"name": "right", "predictions": {"tiny": "reports/predictions/wrong.jsonl"}}]
    )
    with pytest.raises(ValueError, match="prediction system"):
        build_report(spec, base, registry)
    assert main(["--report", "t", "--base-dir", str(base)]) == 1
    assert not (base / "outputs").exists()


def test_context_and_actual_input_hashes_survive_serialization(base, registry):
    path = base / "reports/predictions/p.jsonl"
    Predictions(
        "s",
        "tiny-longform-test",
        dict(_REFS),
        {
            "config": {
                "model_id": "checkpoint-a",
                "decoding": {"beam_size": 5},
                "api_key": "secret",
            },
            "command": "runner --token private --max-new-tokens 50",
        },
    ).to_jsonl(path)
    spec = _spec(
        base,
        [
            {
                "name": "s",
                "predictions": {"tiny": "reports/predictions/p.jsonl"},
                "model_id": "checkpoint-a",
            }
        ],
    )
    report = build_report(spec, base, registry)
    cell = report.to_json_dict()["cells"][0]
    assert cell["inference_context"]["config"]["decoding"]["beam_size"] == 5
    assert (
        cell["input_hashes"]["manifest"]
        == hashlib.sha256((base / "data/tiny.manifest.jsonl").read_bytes()).hexdigest()
    )
    assert cell["input_hashes"]["predictions"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert cell["normalization_policy"] == "bengali" and cell["normalizer_version"] == "1.1.0"
    text = json.dumps(cell)
    assert "secret" not in text and "private" not in text
    assert cell["leakage_audit"]["status"] == "unknown"


def test_missing_context_is_unknown(base, registry):
    _predictions_file(base / "reports/predictions/p.jsonl", dict(_REFS), "s")
    spec = _spec(base, [{"name": "s", "predictions": {"tiny": "reports/predictions/p.jsonl"}}])
    assert build_report(spec, base, registry).cells[0].inference_context == "unknown"


def test_known_model_id_mismatch_is_rejected(base, registry):
    Predictions("s", "tiny-longform-test", dict(_REFS), {"config": {"model_id": "wrong"}}).to_jsonl(
        base / "reports/predictions/p.jsonl"
    )
    spec = _spec(
        base,
        [
            {
                "name": "s",
                "model_id": "correct",
                "predictions": {"tiny": "reports/predictions/p.jsonl"},
            }
        ],
    )
    with pytest.raises(ValueError, match="prediction model"):
        build_report(spec, base, registry)


def test_mixed_policies_and_distinct_contexts(base, registry):
    read = Manifest.from_jsonl(base / "data/tiny.manifest.jsonl")
    cs = replace(read, name="cs", domain="code-switch")
    cs.to_jsonl(base / "data/cs.jsonl")
    Predictions("s", read.name, dict(_REFS), {"config": {"decoding": "read-settings"}}).to_jsonl(
        base / "reports/predictions/read.jsonl"
    )
    Predictions("s", cs.name, dict(_REFS), {"config": {"decoding": "cs-settings"}}).to_jsonl(
        base / "reports/predictions/cs.jsonl"
    )
    from shono.eval.report import SliceSpec, SystemSpec

    spec = ReportSpec(
        "mixed",
        "Mixed",
        (
            SliceSpec("read", "data/tiny.manifest.jsonl", "long-form"),
            SliceSpec("cs", "data/cs.jsonl", "code-switch"),
        ),
        (
            SystemSpec(
                "s",
                {"read": "reports/predictions/read.jsonl", "cs": "reports/predictions/cs.jsonl"},
            ),
        ),
        n_resamples=100,
    )
    report = build_report(spec, base, registry)
    cells = report.to_json_dict()["cells"]
    assert [(c["normalization_policy"], c["normalizer_version"]) for c in cells] == [
        ("bengali", "1.1.0"),
        ("code-switch-script", "1.0.0"),
    ]
    assert cells[0]["inference_context"] != cells[1]["inference_context"]
    assert cells[0]["input_hashes"] != cells[1]["input_hashes"]
    assert (
        "bengali v1.1.0" in report.render_markdown()
        and "code-switch-script v1.0.0" in report.render_markdown()
    )


def _fine_spec(base):
    from shono.protocol import OUR_SYSTEM

    return _spec(
        base,
        [
            {
                "name": OUR_SYSTEM,
                "predictions": {"tiny": "reports/predictions/fine.jsonl"},
                "training_audit": "required",
                "train_manifests": ["data/train.jsonl"],
            }
        ],
    )


def _train_manifest(base, *, text="ভিন্ন প্রশিক্ষণ বাক্য", sid="train-a", checksum=None):
    manifest = Manifest(
        "train",
        "synthetic_fixture",
        "train",
        "read",
        "fixture",
        (Segment(sid, "train.wav", text, 1.0, "train-r", audio_sha256=checksum),),
    )
    manifest.to_jsonl(base / "data/train.jsonl")
    return manifest


def test_fine_tuned_scoring_requires_declared_training_inputs(base, registry):
    from shono.protocol import OUR_SYSTEM

    _predictions_file(base / "reports/predictions/fine.jsonl", dict(_REFS), OUR_SYSTEM)
    spec = _fine_spec(base)
    with pytest.raises(ValueError, match="training manifest"):
        build_report(spec, base, registry)
    assert main(["--report", "t", "--base-dir", str(base)]) == 1
    assert not (base / "outputs").exists()


def test_pending_fine_tuned_cell_keeps_missing_training_audit(base, registry):
    report = build_report(_fine_spec(base), base, registry)
    assert report.cells[0].is_pending
    assert report.cells[0].leakage_audit == {
        "status": "pending",
        "scope": "declared training mix",
        "upstream_base_training": "unknown",
        "missing_train_manifests": ["data/train.jsonl"],
    }


def test_fine_tuned_text_collision_needs_exact_review(base, registry):
    from shono.protocol import OUR_SYSTEM

    _train_manifest(base, text=_REFS["loop01-000"])
    _predictions_file(base / "reports/predictions/fine.jsonl", dict(_REFS), OUR_SYSTEM)
    spec = _fine_spec(base)
    with pytest.raises(ValueError, match="text"):
        build_report(spec, base, registry)
    from shono.eval.normalize import normalize

    system = replace(
        spec.systems[0],
        text_decisions={
            "tiny": {
                normalize(_REFS["loop01-000"]): "Synthetic common-text fixture; distinct audio"
            }
        },
    )
    report = build_report(replace(spec, systems=(system,)), base, registry)
    audit = report.to_json_dict()["cells"][0]["leakage_audit"]
    assert audit["status"] == "reviewed" and audit["audio_coverage"] == 0
    assert audit["train_hashed"] == 0 and audit["eval_hashed"] == 0
    assert audit["reviewed_text"][0]["reason"].startswith("Synthetic")
    assert (
        audit["training_input_hashes"]["data/train.jsonl"]
        == hashlib.sha256((base / "data/train.jsonl").read_bytes()).hexdigest()
    )


def test_hard_overlap_cannot_be_reviewed_away(base, registry):
    from shono.protocol import OUR_SYSTEM

    _train_manifest(base, sid="loop01-000")
    _predictions_file(base / "reports/predictions/fine.jsonl", dict(_REFS), OUR_SYSTEM)
    with pytest.raises(ValueError, match="overlap"):
        build_report(_fine_spec(base), base, registry)


def test_excluded_source_pending_and_never_scored(base):
    raw = (
        (base / "data/tiny.manifest.jsonl").read_text().replace("synthetic_fixture", "bengali_loop")
    )
    (base / "data/tiny.manifest.jsonl").write_text(raw)
    registry = LicenseRegistry.load(_LICENSES)
    spec = _spec(base, [{"name": "s", "predictions": {}}])
    cell = build_report(spec, base, registry).cells[0]
    assert cell.is_pending and "permission" in cell.pending_reason
    _predictions_file(base / "reports/predictions/p.jsonl", dict(_REFS), "s")
    system = replace(spec.systems[0], predictions={"tiny": "reports/predictions/p.jsonl"})
    with pytest.raises(ValueError, match="permission"):
        build_report(replace(spec, systems=(system,)), base, registry)


@pytest.mark.parametrize(
    "bad", [[], None, "bad", {"name": "../x", "title": "x", "slices": [], "systems": []}]
)
def test_invalid_spec_shapes_cli_has_no_traceback(base, bad, capsys):
    (base / "reports/specs/t.json").write_text(json.dumps(bad))
    assert main(["--report", "t", "--base-dir", str(base)]) == 1
    assert "Traceback" not in capsys.readouterr().err
    assert not (base / "outputs").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", True),
        ("seed", -1),
        ("n_resamples", True),
        ("n_resamples", 100.1),
        ("n_resamples", float("nan")),
    ],
)
def test_spec_bootstrap_settings_are_strict(base, field, value):
    spec = _spec(base, [{"name": "s", "predictions": {}}])
    with pytest.raises(ValueError):
        replace(spec, **{field: value})


def test_duplicate_systems_and_unknown_slices_reject(base):
    spec = _spec(base, [{"name": "s", "predictions": {}}])
    with pytest.raises(ValueError, match="duplicate"):
        replace(spec, systems=(spec.systems[0], spec.systems[0]))
    with pytest.raises(ValueError, match="unknown slices"):
        replace(spec, systems=(replace(spec.systems[0], predictions={"ghost": "p.jsonl"}),))


@pytest.mark.parametrize(
    "header",
    [
        [],
        {"predictions": []},
        {"predictions": {"system": True, "manifest": "m"}},
        {"predictions": {"system": "s", "manifest": "m", "run_context": []}},
    ],
)
def test_prediction_header_shapes_reject(tmp_path, header):
    path = tmp_path / "p.jsonl"
    path.write_text(json.dumps(header) + "\n")
    with pytest.raises(ValueError):
        Predictions.from_jsonl(path)


def test_duplicate_header_fields_reject(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text('{"predictions":{"system":"a","system":"b","manifest":"m"}}\n')
    with pytest.raises(ValueError, match="duplicate JSON"):
        Predictions.from_jsonl(path)


def test_cli_explicit_relative_spec_resolves_against_base(base):
    _spec(base, [{"name": "s", "predictions": {}}])
    assert main(["--report", "t", "--spec", "reports/specs/t.json", "--base-dir", str(base)]) == 0
    assert (base / "outputs/t.json").is_file()


@pytest.mark.parametrize("which", ["registry", "spec", "output"])
def test_cli_io_failures_are_clear(base, which, capsys):
    _spec(base, [{"name": "s", "predictions": {}}])
    if which == "registry":
        (base / "data/licenses.json").write_text("[]")
    elif which == "spec":
        (base / "reports/specs/t.json").write_text("{")
    else:
        (base / "outputs").symlink_to("outputs")
    assert main(["--report", "t", "--base-dir", str(base)]) == 1
    assert "Traceback" not in capsys.readouterr().err


def test_cli_rejects_abbreviated_options(base):
    with pytest.raises(SystemExit) as error:
        main(["--report", "t", "--base-d", str(base)])
    assert error.value.code == 2


def test_render_failure_writes_no_files(base, registry, monkeypatch):
    report = build_report(_spec(base, [{"name": "s", "predictions": {}}]), base, registry)
    from shono.eval.report import Report

    def fail(self):
        raise ValueError("broken cell")

    monkeypatch.setattr(Report, "render_markdown", fail)
    out = base / "scratch"
    with pytest.raises(ValueError, match="broken"):
        write_report(report, out)
    assert not out.exists()


def test_report_cannot_overwrite_spec(base, registry):
    spec = _spec(base, [{"name": "s", "predictions": {}}])
    before = (base / "reports/specs/t.json").read_bytes()
    report = build_report(spec, base, registry)
    with pytest.raises(ValueError, match="overwrite input"):
        write_report(report, base / "reports/specs")
    assert (base / "reports/specs/t.json").read_bytes() == before


def test_declared_training_sources_must_pass_license_floor(base, registry):
    from shono.protocol import OUR_SYSTEM

    training = _train_manifest(base)
    replace(training, source="blocked_training").to_jsonl(base / "data/train.jsonl")
    registry = LicenseRegistry(
        [
            _synthetic_license(),
            replace(_synthetic_license(), id="blocked_training", status="excluded"),
        ]
    )
    _predictions_file(base / "reports/predictions/fine.jsonl", dict(_REFS), OUR_SYSTEM)
    with pytest.raises(ValueError, match="excluded source"):
        build_report(_fine_spec(base), base, registry)


def test_prediction_hash_is_of_raw_bytes_including_crlf(tmp_path):
    path = tmp_path / "p.jsonl"
    raw = b'{"predictions":{"system":"s","manifest":"m"}}\r\n{"id":"a","hypothesis":"text"}\r\n'
    path.write_bytes(raw)
    assert Predictions.from_jsonl(path).source_sha256 == hashlib.sha256(raw).hexdigest()


def test_direct_report_cannot_publish_unreviewed_fine_tuned_cell(base, registry):
    from shono.protocol import OUR_SYSTEM

    report = build_report(_spec(base, [{"name": "s", "predictions": {}}]), base, registry)
    manifest = Manifest.from_jsonl(base / "data/tiny.manifest.jsonl")
    cell = score_slice(
        manifest, Predictions(OUR_SYSTEM, manifest.name, dict(_REFS)), n_resamples=100
    )
    report = replace(report, cells=(cell,), system_order=(OUR_SYSTEM,))
    with pytest.raises(ValueError, match="reviewed"):
        write_report(report, base / "outputs")
    assert not (base / "outputs").exists()
