"""The report generator: scores real cells, renders `—` for pending, errors on partial runs."""

import json
from pathlib import Path

import pytest

from shono.data.license import LicenseRegistry
from shono.data.manifest import Manifest, Segment
from shono.eval.__main__ import main
from shono.eval.report import (
    Predictions,
    ReportSpec,
    build_report,
    score_slice,
)

_FIXTURES = Path(__file__).parent / "fixtures"
_TINY = (_FIXTURES / "tiny_manifest.jsonl").read_text(encoding="utf-8")
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
    return LicenseRegistry.load(_LICENSES)


@pytest.fixture
def base(tmp_path: Path) -> Path:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "tiny.manifest.jsonl").write_text(_TINY, encoding="utf-8")
    (tmp_path / "data" / "licenses.json").write_text(
        _LICENSES.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "reports" / "specs").mkdir(parents=True)
    (tmp_path / "reports" / "predictions").mkdir()
    return tmp_path


# ---- scoring a cell ------------------------------------------------------


def test_perfect_predictions_score_zero(registry):
    manifest = Manifest.from_jsonl(_FIXTURES / "tiny_manifest.jsonl")
    preds = Predictions("perfect", "tiny-longform-test", dict(_REFS))
    cell = score_slice(manifest, preds, n_resamples=200)
    assert cell.status == "scored"
    assert cell.score.wer_raw == 0.0
    assert cell.wer_ci.point == 0.0 and cell.wer_ci.upper == 0.0
    assert cell.n_recordings == 2  # the two rec-loop blocks


def test_ci_point_matches_normalized_score():
    # The CI point estimate must equal the corpus normalized WER — one truth, two paths.
    manifest = Manifest.from_jsonl(_FIXTURES / "tiny_manifest.jsonl")
    hyps = dict(_REFS)
    hyps["loop01-000"] = "আজকে আমরা কথা বলব ইংরেজি ভাষা নিয়ে"  # one substitution
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
    hyps["ghost-id"] = "স্টেল অনুমান"  # an id not in the manifest
    with pytest.raises(ValueError, match="not in the manifest"):
        score_slice(manifest, Predictions("s", "tiny-longform-test", hyps), n_resamples=200)


def test_mispaired_predictions_manifest_name_rejected():
    manifest = Manifest.from_jsonl(_FIXTURES / "tiny_manifest.jsonl")
    with pytest.raises(ValueError, match="mispaired"):
        score_slice(manifest, Predictions("s", "some-other-manifest", dict(_REFS)), n_resamples=200)


def test_single_recording_slice_scored_without_ci():
    # A slice from one recording has no between-block variance — it is scored as a
    # point estimate (no CI) rather than aborting the whole report.
    one = Manifest(
        name="single", source="openslr_slr53", split="test", domain="read", version="v",
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


# ---- predictions I/O -----------------------------------------------------


def test_predictions_require_header(tmp_path):
    bad = tmp_path / "p.jsonl"
    bad.write_text(json.dumps({"id": "x", "hypothesis": "y"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        Predictions.from_jsonl(bad)


def test_predictions_round_trip(tmp_path):
    preds = Predictions("sys-a", "cv-bn-test", {"s0": "কথা এক", "s1": "কথা দুই"},
                        run_context={"seed": 0})
    path = tmp_path / "p.jsonl"
    preds.to_jsonl(path)
    loaded = Predictions.from_jsonl(path)
    assert loaded.system == "sys-a"
    assert loaded.manifest == "cv-bn-test"
    assert loaded.hypotheses == preds.hypotheses


def test_duplicate_prediction_id_rejected(tmp_path):
    # A re-run that appends must not silently overwrite an earlier hypothesis —
    # symmetric with Manifest rejecting duplicate segment ids.
    p = tmp_path / "p.jsonl"
    p.write_text(
        "\n".join([
            json.dumps({"predictions": {"system": "s", "manifest": "m"}}),
            json.dumps({"id": "a", "hypothesis": "first"}),
            json.dumps({"id": "a", "hypothesis": "second"}),
        ]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate id"):
        Predictions.from_jsonl(p)


# ---- full report assembly ------------------------------------------------


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


def test_pending_cell_renders_dash(base, registry):
    spec = _spec(base, [{"name": "no-preds-yet", "predictions": {}}])
    report = build_report(spec, base, registry)
    assert report.cells[0].is_pending
    md = report.render_markdown()
    assert "| no-preds-yet | — | — | — | — |" in md


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
    assert "0.0% [0.0%, 0.0%]" in md  # perfect predictions
    assert "## Data licenses" in md  # license table embedded


def test_missing_manifest_makes_slice_pending(base, registry):
    # The manifest for a slice may not exist yet (gated data); the skeleton still renders.
    spec = {
        "name": "t", "title": "T", "n_resamples": 200,
        "slices": [{"name": "future", "manifest": "data/not-yet.manifest.jsonl", "domain": "read"}],
        "systems": [{"name": "sys", "predictions": {}}],
    }
    path = base / "reports" / "specs" / "t.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    report = build_report(ReportSpec.from_json(path), base, registry)
    assert report.cells[0].is_pending
    assert report.cells[0].domain == "read"  # intended domain from the spec


# ---- the CLI -------------------------------------------------------------


def test_cli_writes_md_and_json(base):
    _predictions_file(base / "reports" / "predictions" / "s__tiny.jsonl", dict(_REFS), "sys-a")
    _spec(base, [{"name": "sys-a", "predictions": {"tiny": "reports/predictions/s__tiny.jsonl"}}])
    rc = main(["--report", "t", "--base-dir", str(base)])
    assert rc == 0
    assert (base / "reports" / "t.md").exists()
    payload = json.loads((base / "reports" / "t.json").read_text(encoding="utf-8"))
    assert payload["normalizer_version"]
    assert payload["cells"][0]["status"] == "scored"


def test_cli_unknown_report_exits_2(base):
    rc = main(["--report", "does-not-exist", "--base-dir", str(base)])
    assert rc == 2


def test_cli_incomplete_predictions_exits_1_and_writes_nothing(base):
    # The highest-stakes honesty path, end to end: an incomplete predictions file
    # must fail the CLI, not emit a partial report.
    partial = {k: v for k, v in _REFS.items() if k != "loop02-001"}
    _predictions_file(base / "reports" / "predictions" / "s__tiny.jsonl", partial, "sys-a")
    _spec(base, [{"name": "sys-a", "predictions": {"tiny": "reports/predictions/s__tiny.jsonl"}}])
    rc = main(["--report", "t", "--base-dir", str(base)])
    assert rc == 1
    assert not (base / "reports" / "t.md").exists()  # nothing written on failure


def test_cli_real_baselines_spec_is_valid_skeleton(base):
    # The shipped baselines spec must load and render an all-pending skeleton
    # (its manifests/predictions are gated), never crash.
    repo = Path(__file__).resolve().parents[1]
    spec_src = (repo / "reports" / "specs" / "baselines.json").read_text(encoding="utf-8")
    (base / "reports" / "specs" / "baselines.json").write_text(spec_src, encoding="utf-8")
    rc = main(["--report", "baselines", "--base-dir", str(base)])
    assert rc == 0
    payload = json.loads((base / "reports" / "baselines.json").read_text(encoding="utf-8"))
    assert len(payload["cells"]) == 9  # 3 systems x 3 slices
    assert all(c["status"] == "pending" for c in payload["cells"])
