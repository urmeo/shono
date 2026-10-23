import hashlib
import json
import random
import re
import runpy
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import pytest
from support import CI_BLOCKS, NORMALIZATION_CASES, SCORE_CASES, TINY_MANIFEST, synthetic_registry

import shono.output_paths as output_paths
from shono.data.license import LicenseRegistry
from shono.data.manifest import Manifest, Segment
from shono.eval.__main__ import main as evaluation_main
from shono.eval.ci import blockwise_bootstrap_ci
from shono.eval.codeswitch import (
    CS_NORMALIZER_VERSION,
    bengali_fraction,
    code_switch_normalize,
    is_code_switched,
    script_counts,
)
from shono.eval.normalize import NORMALIZER_VERSION, normalize
from shono.eval.report import (
    Predictions,
    Report,
    ReportSpec,
    SliceSpec,
    SystemSpec,
    build_report,
    score_slice,
    write_report,
)
from shono.eval.score import cer, score, wer
from shono.protocol import OUR_SYSTEM
from shono.provenance import GitState, RunContext, seed_everything

_BOOTSTRAP_RECORDS = [tuple(record) for record in CI_BLOCKS["records"]]
_BLOCKWISE_ONLY_VALUES = {0.0, 1 / 6, 1 / 3}
_bootstrap = partial(blockwise_bootstrap_ci, records=_BOOTSTRAP_RECORDS, n_resamples=200, seed=0)
_TINY = TINY_MANIFEST.replace('"source": "bengali_loop"', '"source": "synthetic_fixture"')
_SYNTHETIC_LICENSE = next(iter(synthetic_registry("synthetic_fixture")))
_TRAIN_AUDIO_HASH = hashlib.sha256(b"synthetic training audio").hexdigest()
_REFS = {
    "loop01-000": "আজকে আমরা কথা বলব বাংলা ভাষা নিয়ে",
    "loop01-001": "ভাষা হল যোগাযোগের প্রধান মাধ্যম",
    "loop01-002": "চলুন শুরু করা যাক",
    "loop02-000": "দ্বিতীয় রেকর্ডিং এ স্বাগতম",
    "loop02-001": "আমরা দুইশো ছাব্বিশ সালে আছি",
}
_CAPTURE_TIME = datetime(2026, 7, 23, 9, 30, tzinfo=UTC)
ROOT = Path(__file__).resolve().parents[1]


def _make_code_switch_manifest(domain):
    return Manifest(
        "m",
        "mucs_slr104",
        "test",
        domain,
        "v",
        (
            Segment("s0", "a.wav", "আজকের Lecture এ Machine Learning", 4, "r0"),
            Segment("s1", "a.wav", "Database এ Query চালাও", 3, "r1"),
        ),
    )


def _write_predictions(path: Path, hypotheses: dict[str, str], system: str) -> None:
    Predictions(system, "tiny-longform-test", hypotheses).to_jsonl(path)


def _make_report_spec(base: Path, *, name="s", predictions=None, **metadata) -> ReportSpec:
    system = {"name": name, "predictions": {"tiny": predictions} if predictions else {}, **metadata}
    spec = {
        "name": "t",
        "title": "Test report",
        "n_resamples": 200,
        "slices": [{"name": "tiny", "manifest": "data/tiny.manifest.jsonl", "domain": "long-form"}],
        "systems": [{"model_kind": "pretrained", **system}],
    }
    path = base / "specs" / "t.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return ReportSpec.from_json(path)


def _prediction_report(base, registry, *, name="s", filename="p.jsonl", context=None, **metadata):
    predictions = f"reports/predictions/{filename}"
    Predictions(name, "tiny-longform-test", dict(_REFS), context).to_jsonl(base / predictions)
    return build_report(
        _make_report_spec(base, name=name, predictions=predictions, **metadata), base, registry
    )


def _make_fine_tuned_report_spec(base):
    return _make_report_spec(
        base,
        name=OUR_SYSTEM,
        predictions="reports/predictions/fine.jsonl",
        model_kind="fine_tuned",
        training_audit="required",
        train_manifests=["data/train.jsonl"],
    )


def _write_training_manifest(
    base, *, text="ভিন্ন প্রশিক্ষণ বাক্য", sid="train-a", checksum=_TRAIN_AUDIO_HASH
):
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


def _evaluation_cli(base, *arguments):
    defaults = ("--report", "t", "--specs-dir", "specs", "--licenses", "data/licenses.json")
    return evaluation_main(["--base-dir", str(base), *(arguments or defaults)])


def _capture_run_context(**overrides):
    defaults = dict(
        seed=13,
        config={"model": "whisper-medium", "lr": 5e-06},
        command="python -m shono.eval --report baselines",
        now=lambda: _CAPTURE_TIME,
        git_state=GitState(sha="abc123", dirty=False),
    )
    return RunContext.capture(**(defaults | overrides))


class _ReportFixtures:
    @pytest.fixture
    def pending_report(self, base, registry):
        return build_report(_make_report_spec(base), base, registry)

    @pytest.fixture
    def report(self, report_inputs):
        base, _, spec, registry = report_inputs
        return build_report(spec, base, registry)

    @pytest.fixture
    def registry(self) -> LicenseRegistry:
        return synthetic_registry("synthetic_fixture")

    @pytest.fixture
    def base(self, tmp_path: Path) -> Path:
        (tmp_path / "data").mkdir()
        manifest_path = tmp_path / "data" / "tiny.manifest.jsonl"
        manifest_path.write_text(_TINY, encoding="utf-8")
        manifest = Manifest.from_jsonl(manifest_path)
        replace(
            manifest,
            segments=tuple(
                replace(
                    seg,
                    audio_sha256=hashlib.sha256(
                        f"synthetic audio: {seg.audio}".encode()
                    ).hexdigest(),
                )
                for seg in manifest.segments
            ),
        ).to_jsonl(manifest_path)
        (tmp_path / "data" / "licenses.json").write_text(
            json.dumps({"licenses": [asdict(_SYNTHETIC_LICENSE)]}), encoding="utf-8"
        )
        (tmp_path / "specs").mkdir()
        (tmp_path / "reports" / "predictions").mkdir(parents=True)
        return tmp_path

    @pytest.fixture
    def report_inputs(self, base, registry):
        original = Manifest.from_jsonl(base / "data/tiny.manifest.jsonl")
        manifest = replace(
            original,
            name="evaluation",
            domain="read",
            segments=(original.segments[0], original.segments[3]),
        )
        path = base / "evaluation.jsonl"
        manifest.to_jsonl(path)
        context = {
            "config": {"model_id": "checkpoint-a", "inference_mode": "bounded-short"},
            "extra": {"manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        }
        hypotheses = dict(zip(manifest.ids(), manifest.references(), strict=True))
        Predictions("system", manifest.name, hypotheses, context).to_jsonl(
            base / "predictions.jsonl"
        )
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
        return base, manifest, spec, registry


class TestNormalization:
    def test_latin_is_casefolded(self):
        assert code_switch_normalize("Machine Learning") == "machine learning"

    def test_bengali_is_left_intact(self):
        assert code_switch_normalize("কি") == "কি"
        assert "ি" in code_switch_normalize("কি?")

    def test_code_switch_makes_english_case_match(self):
        assert code_switch_normalize("আমি Bank এ যাব") == code_switch_normalize("আমি bank এ যাব")
        assert normalize("আমি Bank এ যাব") != normalize("আমি bank এ যাব")

    def test_cross_script_is_not_credited(self):
        assert code_switch_normalize("school") != code_switch_normalize("স্কুল")

    def test_cs_version_is_semver(self):
        assert re.fullmatch("\\d+\\.\\d+\\.\\d+", CS_NORMALIZER_VERSION)

    def test_script_counts(self):
        counts = script_counts("আমি bank 2026")
        assert counts["bengali"] == 3
        assert counts["latin"] == 4
        assert counts["digit"] == 4

    def test_bengali_digits_count_as_digits_not_letters(self):
        counts = script_counts("২০২৬")
        assert counts["digit"] == 4
        assert counts["bengali"] == 0

    def test_bengali_fraction(self):
        assert bengali_fraction("শুধু বাংলা") == 1.0
        assert bengali_fraction("only english") == 0.0
        assert bengali_fraction("আমি bank") == pytest.approx(3 / 7)

    def test_is_code_switched(self):
        assert is_code_switched("আমি bank এ যাব")
        assert not is_code_switched("আমি ব্যাংকে যাব")
        assert not is_code_switched("only english")

    def test_bengali_currency_is_not_a_script_letter(self):
        counts = script_counts("৳ price")
        assert counts == {"bengali": 0, "latin": 5, "digit": 0, "other": 1}
        assert not is_code_switched("৳ price")
        assert bengali_fraction("৳ price") == 0.0

    def test_code_switch_slice_scores_case_insensitively(self):
        hyps = {"s0": "আজকের lecture এ machine learning", "s1": "database এ query চালাও"}
        cs = score_slice(
            _make_code_switch_manifest("code-switch"), Predictions("s", "m", hyps), n_resamples=200
        )
        read = score_slice(
            _make_code_switch_manifest("read"), Predictions("s", "m", hyps), n_resamples=200
        )
        assert cs.score.wer_normalized == 0.0
        assert cs.score.normalizer_version == CS_NORMALIZER_VERSION
        assert read.score.wer_normalized > 0.0
        assert read.score.normalizer_version == NORMALIZER_VERSION

    @pytest.mark.parametrize(
        "case", NORMALIZATION_CASES, ids=[c["name"] for c in NORMALIZATION_CASES]
    )
    def test_normalization_fixtures_are_exact_and_idempotent(self, case):
        assert normalize(case["input"]) == case["expected"]
        assert normalize(normalize(case["input"])) == case["expected"]

    def test_nfc_composes_decomposed_o_kar(self):
        decomposed = "কো"
        composed = "কো"
        assert decomposed != composed
        assert normalize(decomposed) == composed

    def test_zero_width_characters_removed(self):
        for zw in ("\u200b", "\u200c", "\u200d", "\ufeff"):
            assert zw not in normalize(f"আ{zw}মি")

    def test_version_is_semver(self):
        assert re.fullmatch("\\d+\\.\\d+\\.\\d+", NORMALIZER_VERSION)


class TestCorpusScoring:
    @pytest.mark.parametrize(
        "compute,references,hypotheses,message",
        [
            (wer, ["আমি"], ["আমি", "তুমি"], "references but"),
            (wer, [], [], "empty corpus"),
            (wer, [" "], ["আমি"], "reference 0 is empty"),
            (score, ["৳!"], ["টাকা"], "empty"),
        ],
    )
    def test_invalid_corpora(self, compute, references, hypotheses, message):
        with pytest.raises(ValueError, match=message):
            compute(references, hypotheses)

    @pytest.mark.parametrize("case", SCORE_CASES, ids=[c["name"] for c in SCORE_CASES])
    def test_hand_computed_wer_cer(self, case):
        assert wer(case["references"], case["hypotheses"]) == pytest.approx(case["expected_wer"])
        if case["expected_cer"] is not None:
            assert cer(case["references"], case["hypotheses"]) == pytest.approx(
                case["expected_cer"]
            )

    def test_raw_wer_treats_any_whitespace_as_separator(self):
        assert wer(["আমি ভাত খাই"], ["আমি ভাত\nখাই"]) == 0.0
        assert wer(["আমি ভাত খাই"], ["আমি  ভাত\tখাই"]) == 0.0

    def test_raw_scoring_never_folds_case(self):
        assert wer(["আমি Bank এ যাব"], ["আমি bank এ যাব"]) == pytest.approx(0.25)
        report = score(["আমি Bank এ যাব"], ["আমি bank এ যাব"])
        assert report.wer_normalized == pytest.approx(0.25)

    def test_punctuation_mismatch_penalized_raw_but_not_normalized(self):
        report = score(["আমি ভাত খাই।"], ["আমি ভাত খাই"])
        assert report.wer_raw == pytest.approx(1 / 3)
        assert report.wer_normalized == 0.0
        assert report.normalizer_version == NORMALIZER_VERSION
        assert report.n_segments == 1

    def test_normalization_applied_to_both_sides(self):
        report = score(["খাই।"], ["খাই।"])
        assert report.wer_raw == 0.0
        assert report.wer_normalized == 0.0

    def test_taka_spacing_cannot_flip_normalized_scores(self):
        report = score(["৳ ১০০ দাম"], ["৳১০০ দাম"])
        assert report.wer_normalized == 0.0

    @pytest.mark.parametrize("compute", [wer, cer, score])
    @pytest.mark.parametrize(
        "references,hypotheses",
        [
            ("আমি", "আমি"),
            (["আমি"], [None]),
            ([7], ["আমি"]),
            (["আমি"], [False]),
            ([b"word"], ["আমি"]),
        ],
    )
    def test_invalid_text_inputs_fail_clearly(self, compute, references, hypotheses):
        with pytest.raises(ValueError, match="strings"):
            compute(references, hypotheses)


class TestBlockBootstrap:
    @pytest.mark.parametrize(
        "setting,value",
        [
            *(
                ("confidence", value)
                for value in (True, False, float("nan"), float("inf"), 0, 1, "0.95", 1j, 1.5)
            ),
            *(
                ("n_resamples", value)
                for value in (True, False, 100.5, "100", float("inf"), None, 10)
            ),
            *(("seed", value) for value in (True, -1, 2**32, 0.5, "1", None)),
            ("keep_samples", "false"),
            ("metric", "der"),
        ],
    )
    def test_invalid_settings(self, setting, value):
        message = "unknown metric" if setting == "metric" else setting
        with pytest.raises(ValueError, match=message):
            _bootstrap(**{setting: value})

    def test_point_estimate_matches_hand_count(self):
        result = _bootstrap()
        assert result.point == pytest.approx(CI_BLOCKS["expected_point_wer"])
        assert result.n_blocks == 2
        assert result.n_segments == 4

    def test_resampling_keeps_recording_blocks_together(self):
        result = _bootstrap(n_resamples=500, seed=7, keep_samples=True)
        assert result.samples is not None and len(result.samples) == 500
        for value in result.samples:
            assert any(value == pytest.approx(v) for v in _BLOCKWISE_ONLY_VALUES), (
                f"sample {value} is impossible under blockwise resampling"
            )

    def test_interval_brackets_point_within_possible_range(self):
        result = _bootstrap(n_resamples=1000)
        assert 0.0 <= result.lower <= result.point <= result.upper
        assert result.upper <= CI_BLOCKS["max_possible_wer"] + 1e-12

    def test_deterministic_for_same_seed(self):
        a = _bootstrap(n_resamples=300, seed=42)
        b = _bootstrap(n_resamples=300, seed=42)
        assert (a.lower, a.upper) == (b.lower, b.upper)

    def test_samples_omitted_by_default(self):
        result = _bootstrap(n_resamples=100)
        assert result.samples is None

    def test_cer_metric_supported(self):
        result = _bootstrap(metric="cer")
        assert 0.0 <= result.lower <= result.point <= result.upper

    def test_single_block_raises(self):
        single = [r for r in _BOOTSTRAP_RECORDS if r[0] == "A"]
        with pytest.raises(ValueError, match=">= 2 blocks"):
            blockwise_bootstrap_ci(single, n_resamples=100)

    @pytest.mark.parametrize(
        "records",
        [
            None,
            "abc",
            [],
            [("r", "text")],
            [(1, "text", "hyp")],
            [("", "text", "hyp")],
            [("r", "text", None)],
        ],
    )
    def test_input_triples_are_validated(self, records):
        with pytest.raises(ValueError):
            blockwise_bootstrap_ci(records)


class TestPredictionFiles(_ReportFixtures):
    def test_predictions_require_header(self, tmp_path):
        bad = tmp_path / "p.jsonl"
        bad.write_text(json.dumps({"id": "x", "hypothesis": "y"}) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="header"):
            Predictions.from_jsonl(bad)

    def test_predictions_round_trip(self, tmp_path):
        preds = Predictions(
            "sys-a", "cv-bn-test", {"s0": "কথা এক", "s1": "কথা দুই"}, run_context={"seed": 0}
        )
        path = tmp_path / "p.jsonl"
        preds.to_jsonl(path)
        loaded = Predictions.from_jsonl(path)
        assert loaded.system == "sys-a"
        assert loaded.manifest == "cv-bn-test"
        assert loaded.hypotheses == preds.hypotheses

    def test_export_redacts_context_mutated_after_construction(self, tmp_path):
        predictions = Predictions("s", "m", {"a": "text"}, run_context={"config": {}})
        predictions.run_context["config"]["accessToken"] = "credential-sentinel"
        predictions.run_context["url"] = "https://example.invalid/?token=credential-sentinel"
        path = tmp_path / "predictions.jsonl"
        predictions.to_jsonl(path)
        assert "credential-sentinel" not in path.read_text()
        restored = Predictions.from_jsonl(path)
        assert restored.run_context["config"]["accessToken"] == "[redacted]"
        assert restored.run_context["url"] == "https://example.invalid/?token=[redacted]"

    def test_duplicate_prediction_id_rejected(self, tmp_path):
        path = tmp_path / "p.jsonl"
        Predictions("s", "m", {"a": "first"}).to_jsonl(path)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"id": "a", "hypothesis": "second"}) + "\n")
        with pytest.raises(ValueError, match="duplicate id"):
            Predictions.from_jsonl(path)

    @pytest.mark.parametrize(
        "header",
        [
            [],
            {"predictions": []},
            {"predictions": {"system": True, "manifest": "m"}},
            {"predictions": {"system": "s", "manifest": "m", "run_context": []}},
        ],
    )
    def test_prediction_header_shapes_reject(self, tmp_path, header):
        path = tmp_path / "p.jsonl"
        path.write_text(json.dumps(header) + "\n")
        with pytest.raises(ValueError):
            Predictions.from_jsonl(path)

    def test_duplicate_header_fields_reject(self, tmp_path):
        path = tmp_path / "p.jsonl"
        path.write_text('{"predictions":{"system":"a","system":"b","manifest":"m"}}\n')
        with pytest.raises(ValueError, match="duplicate JSON"):
            Predictions.from_jsonl(path)

    def test_prediction_hash_is_of_raw_bytes_including_crlf(self, tmp_path):
        path = tmp_path / "p.jsonl"
        raw = b'{"predictions":{"system":"s","manifest":"m"}}\r\n{"id":"a","hypothesis":"text"}\r\n'
        path.write_bytes(raw)
        assert Predictions.from_jsonl(path).source_sha256 == hashlib.sha256(raw).hexdigest()


class TestReportSpecifications(_ReportFixtures):
    def test_dict_spec_matches_json_values_without_claiming_a_source_file(self, base):
        spec = _make_report_spec(base)
        path = base / "specs/t.json"
        doc = json.loads(path.read_text())
        builtin = ReportSpec.from_dict(doc)
        assert builtin == replace(spec, source_path=None, source_sha256=builtin.source_sha256)
        assert len(builtin.source_sha256) == 64
        assert (
            builtin.source_sha256 == ReportSpec.from_dict(dict(reversed(doc.items()))).source_sha256
        )
        assert spec.source_path == str(path.resolve())
        assert spec.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()

    @pytest.mark.parametrize("doc", [None, [], "invalid", {"unexpected": True}])
    def test_dict_spec_requires_a_valid_object(self, doc):
        with pytest.raises(ValueError):
            ReportSpec.from_dict(doc)

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
    def test_spec_bootstrap_settings_are_strict(self, base, field, value):
        spec = _make_report_spec(base)
        with pytest.raises(ValueError):
            replace(spec, **{field: value})

    def test_duplicate_systems_and_unknown_slices_reject(self, base):
        spec = _make_report_spec(base)
        with pytest.raises(ValueError, match="duplicate"):
            replace(spec, systems=(spec.systems[0], spec.systems[0]))
        with pytest.raises(ValueError, match="unknown slices"):
            replace(spec, systems=(replace(spec.systems[0], predictions={"ghost": "p.jsonl"}),))

    @pytest.mark.parametrize("name", ["fine-tuned Whisper", "Fine-tuned Whisper", "custom model"])
    @pytest.mark.parametrize("audit", ["unknown", "not_applicable"])
    def test_fine_tuned_metadata_requires_audit_declaration(self, name, audit):
        with pytest.raises(ValueError, match="declared training leakage audit"):
            SystemSpec(name, {}, model_kind="fine_tuned", training_audit=audit)

    @pytest.mark.parametrize("kind", [None, True, [], "fine-tuned", "PRETRAINED"])
    def test_invalid_model_kind_metadata_is_rejected(self, kind):
        with pytest.raises(ValueError, match="model_kind"):
            SystemSpec("s", {}, model_kind=kind)


class TestReportScoring(_ReportFixtures):
    @pytest.fixture
    def manifest(self):
        return Manifest.from_text(TINY_MANIFEST)

    def test_perfect_predictions_score_zero(self, manifest):
        preds = Predictions("perfect", "tiny-longform-test", dict(_REFS))
        cell = score_slice(manifest, preds, n_resamples=200)
        assert cell.status == "scored"
        assert cell.score.wer_raw == 0.0
        assert cell.wer_ci.point == 0.0 and cell.wer_ci.upper == 0.0
        assert cell.n_recordings == 2

    def test_ci_point_matches_normalized_score(self, manifest):
        hyps = dict(_REFS)
        hyps["loop01-000"] = "আজকে আমরা কথা বলব ইংরেজি ভাষা নিয়ে"
        cell = score_slice(manifest, Predictions("s", "tiny-longform-test", hyps), n_resamples=300)
        assert cell.wer_ci.point == pytest.approx(cell.score.wer_normalized)
        assert cell.score.wer_raw > 0.0

    def test_incomplete_predictions_are_an_error_not_a_partial_score(self, manifest):
        partial = {k: v for (k, v) in _REFS.items() if k != "loop02-001"}
        with pytest.raises(KeyError, match="no hypothesis"):
            score_slice(manifest, Predictions("s", "tiny-longform-test", partial), n_resamples=200)

    def test_extra_predictions_are_rejected(self, manifest):
        hyps = dict(_REFS)
        hyps["ghost-id"] = "স্টেল অনুমান"
        with pytest.raises(ValueError, match="not in the manifest"):
            score_slice(manifest, Predictions("s", "tiny-longform-test", hyps), n_resamples=200)

    def test_mispaired_predictions_manifest_name_rejected(self, manifest):
        with pytest.raises(ValueError, match="mispaired"):
            score_slice(
                manifest, Predictions("s", "some-other-manifest", dict(_REFS)), n_resamples=200
            )

    def test_single_recording_slice_scored_without_ci(self, base):
        manifest = Manifest.from_jsonl(base / "data/tiny.manifest.jsonl")
        one = replace(manifest, name="single", segments=manifest.segments[:2])
        hypotheses = dict(zip(one.ids(), one.references(), strict=True))
        cell = score_slice(one, Predictions("s", one.name, hypotheses), n_resamples=200)
        assert cell.status == "scored"
        assert cell.wer_ci is None and cell.cer_ci is None
        assert cell.score.wer_normalized == 0.0

    def test_pending_cell_renders_pending(self, base, registry):
        spec = _make_report_spec(base, name="no-preds-yet")
        report = build_report(spec, base, registry)
        assert report.cells[0].is_pending
        md = report.render_markdown()
        assert "| no-preds-yet | pending | pending | pending | pending | bengali v1.1.0 |" in md

    def test_scored_cell_renders_numbers(self, base, registry):
        report = _prediction_report(base, registry, name="sys-a", filename="s__tiny.jsonl")
        cell = report.cell("sys-a", "tiny")
        assert not cell.is_pending
        md = report.render_markdown()
        assert "0.0% [0.0%, 0.0%]" in md
        assert "## Data licenses" in md

    def test_missing_manifest_makes_slice_pending(self, base, registry):
        spec = ReportSpec(
            "t",
            "T",
            (SliceSpec("future", "data/absent.jsonl", "read"),),
            (SystemSpec("sys", {}),),
            n_resamples=200,
        )
        cell = build_report(spec, base, registry).cells[0]
        assert cell.is_pending
        assert cell.domain == "read"

    def test_system_mismatch_fails_before_writing(self, base, registry):
        _write_predictions(base / "reports/predictions/wrong.jsonl", dict(_REFS), "wrong")
        spec = _make_report_spec(base, name="right", predictions="reports/predictions/wrong.jsonl")
        with pytest.raises(ValueError, match="prediction system"):
            build_report(spec, base, registry)
        assert _evaluation_cli(base) == 1
        assert not (base / "outputs").exists()

    def test_context_and_actual_input_hashes_survive_serialization(self, base, registry):
        path = base / "reports/predictions/p.jsonl"
        report = _prediction_report(
            base,
            registry,
            model_id="checkpoint-a",
            context={
                "config": {
                    "model_id": "checkpoint-a",
                    "decoding": {"beam_size": 5},
                    "api_key": "secret",
                },
                "command": "runner --token private --max-new-tokens 50",
            },
        )
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

    def test_missing_context_is_unknown(self, base, registry):
        assert _prediction_report(base, registry).cells[0].inference_context == "unknown"

    def test_known_model_id_mismatch_is_rejected(self, base, registry):
        with pytest.raises(ValueError, match="prediction model"):
            _prediction_report(
                base, registry, model_id="correct", context={"config": {"model_id": "wrong"}}
            )

    def test_mixed_policies_and_distinct_contexts(self, base, registry):
        read = Manifest.from_jsonl(base / "data/tiny.manifest.jsonl")
        cs = replace(read, name="cs", domain="code-switch")
        cs.to_jsonl(base / "data/cs.jsonl")
        Predictions(
            "s", read.name, dict(_REFS), {"config": {"decoding": "read-settings"}}
        ).to_jsonl(base / "reports/predictions/read.jsonl")
        Predictions("s", cs.name, dict(_REFS), {"config": {"decoding": "cs-settings"}}).to_jsonl(
            base / "reports/predictions/cs.jsonl"
        )
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
                    {
                        "read": "reports/predictions/read.jsonl",
                        "cs": "reports/predictions/cs.jsonl",
                    },
                    model_kind="pretrained",
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

    def test_fine_tuned_scoring_requires_declared_training_inputs(self, base, registry):
        _write_predictions(base / "reports/predictions/fine.jsonl", dict(_REFS), OUR_SYSTEM)
        spec = _make_fine_tuned_report_spec(base)
        with pytest.raises(ValueError, match="training manifest"):
            build_report(spec, base, registry)
        assert _evaluation_cli(base) == 1
        assert not (base / "outputs").exists()

    def test_pending_fine_tuned_cell_keeps_missing_training_audit(self, base, registry):
        report = build_report(_make_fine_tuned_report_spec(base), base, registry)
        assert report.cells[0].is_pending
        assert report.cells[0].leakage_audit == {
            "status": "pending",
            "scope": "declared training mix",
            "upstream_base_training": "unknown",
            "missing_train_manifests": ["data/train.jsonl"],
        }

    def test_fine_tuned_text_collision_needs_exact_review(self, base, registry):
        _write_training_manifest(base, text=_REFS["loop01-000"])
        _write_predictions(base / "reports/predictions/fine.jsonl", dict(_REFS), OUR_SYSTEM)
        spec = _make_fine_tuned_report_spec(base)
        with pytest.raises(ValueError, match="text"):
            build_report(spec, base, registry)
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
        assert audit["status"] == "reviewed" and audit["audio_coverage"] == 1
        assert audit["train_hashed"] == 1 and audit["eval_hashed"] == 5
        assert audit["reviewed_text"][0]["reason"].startswith("Synthetic")
        assert (
            audit["training_input_hashes"]["data/train.jsonl"]
            == hashlib.sha256((base / "data/train.jsonl").read_bytes()).hexdigest()
        )

    def test_hard_overlap_cannot_be_reviewed_away(self, base, registry):
        _write_training_manifest(base, sid="loop01-000")
        _write_predictions(base / "reports/predictions/fine.jsonl", dict(_REFS), OUR_SYSTEM)
        with pytest.raises(ValueError, match="overlap"):
            build_report(_make_fine_tuned_report_spec(base), base, registry)

    @pytest.mark.parametrize("missing_side", ["train", "evaluation"])
    def test_fine_tuned_scoring_rejects_incomplete_hash_audit(self, base, registry, missing_side):
        if missing_side == "train":
            _write_training_manifest(base, checksum=None)
        else:
            _write_training_manifest(base)
            (base / "data/tiny.manifest.jsonl").write_text(_TINY, encoding="utf-8")
        _write_predictions(base / "reports/predictions/fine.jsonl", dict(_REFS), OUR_SYSTEM)
        with pytest.raises(ValueError, match="incomplete train/evaluation leakage audit"):
            build_report(_make_fine_tuned_report_spec(base), base, registry)

    def test_excluded_source_pending_and_never_scored(self, base):
        registry = synthetic_registry("synthetic_fixture", status="excluded")
        spec = _make_report_spec(base)
        cell = build_report(spec, base, registry).cells[0]
        assert cell.is_pending and "permission" in cell.pending_reason
        _write_predictions(base / "reports/predictions/p.jsonl", dict(_REFS), "s")
        system = replace(spec.systems[0], predictions={"tiny": "reports/predictions/p.jsonl"})
        with pytest.raises(ValueError, match="permission"):
            build_report(replace(spec, systems=(system,)), base, registry)

    def test_declared_training_sources_must_pass_license_floor(self, base, registry):
        training = _write_training_manifest(base)
        replace(training, source="blocked_training").to_jsonl(base / "data/train.jsonl")
        registry = LicenseRegistry(
            [
                _SYNTHETIC_LICENSE,
                replace(_SYNTHETIC_LICENSE, id="blocked_training", status="excluded"),
            ]
        )
        _write_predictions(base / "reports/predictions/fine.jsonl", dict(_REFS), OUR_SYSTEM)
        with pytest.raises(ValueError, match="excluded source"):
            build_report(_make_fine_tuned_report_spec(base), base, registry)

    @pytest.mark.parametrize("name", ["Fine-tuned Whisper", "WHISPER", "custom model"])
    def test_scored_system_requires_model_kind_metadata(self, base, registry, name):
        _write_predictions(base / "reports/predictions/p.jsonl", dict(_REFS), name)
        _make_report_spec(base, name=name, predictions="reports/predictions/p.jsonl")
        path = base / "specs/t.json"
        raw = json.loads(path.read_text())
        raw["systems"][0].pop("model_kind", None)
        path.write_text(json.dumps(raw))
        spec = ReportSpec.from_json(path)
        with pytest.raises(ValueError, match="model_kind"):
            build_report(spec, base, registry)

    def test_fine_tuned_audit_and_model_kind_survive_system_rename(self, base, registry):
        name = "Custom Whisper"
        _write_training_manifest(base)
        _write_predictions(base / "reports/predictions/fine.jsonl", dict(_REFS), name)
        spec = _make_fine_tuned_report_spec(base)
        spec = replace(spec, systems=(replace(spec.systems[0], name=name),))
        report = build_report(spec, base, registry)
        cell = report.to_json_dict()["cells"][0]
        assert cell["model_kind"] == "fine_tuned"
        assert cell["leakage_audit"]["status"] == "reviewed"
        assert "| Custom Whisper | 0.0%" in report.render_markdown()

    @pytest.mark.parametrize("kind", ["pretrained", "provider"])
    def test_upstream_model_metadata_is_independent_of_display_name(self, base, registry, kind):
        name = "fine-tuned Whisper"
        report = _prediction_report(base, registry, name=name, model_kind=kind)
        cell = report.to_json_dict()["cells"][0]
        assert cell["model_kind"] == kind
        assert cell["leakage_audit"]["status"] == "unknown"

    def test_report_rejects_predictions_for_changed_manifest_bytes(self, report_inputs):
        (base, manifest, spec, registry) = report_inputs
        altered = replace(manifest.segments[0], text="বদলানো কথা")
        replace(manifest, segments=(altered, manifest.segments[1])).to_jsonl(
            base / "evaluation.jsonl"
        )
        with pytest.raises(ValueError, match="prediction manifest.*hash"):
            build_report(spec, base, registry)

    def test_report_rejects_inference_mode_mismatch(self, report_inputs):
        (base, _manifest, spec, registry) = report_inputs
        wrong = replace(spec.systems[0], inference_modes={"test": "long-form-vad"})
        with pytest.raises(ValueError, match="inference mode"):
            build_report(replace(spec, systems=(wrong,)), base, registry)

    def test_report_retains_recorded_model_and_mode_when_spec_is_unknown(self, report_inputs):
        (base, _manifest, spec, registry) = report_inputs
        unspecified = replace(spec.systems[0], model_id="unknown", inference_modes={})
        report = build_report(replace(spec, systems=(unspecified,)), base, registry)
        cell = report.to_json_dict()["cells"][0]
        assert cell["model_id"] == "checkpoint-a"
        assert cell["inference_mode"] == "bounded-short"

    @pytest.mark.parametrize("seed,n_resamples", [(1, 100), (0, 200)])
    def test_report_context_must_match_actual_bootstrap_settings(
        self, report_inputs, seed, n_resamples
    ):
        (base, _manifest, spec, registry) = report_inputs
        context = RunContext.capture(seed, {"n_resamples": n_resamples}, packages=())
        with pytest.raises(ValueError, match="report context"):
            build_report(spec, base, registry, run_context=context)


class TestReportExports(_ReportFixtures):
    def test_render_failure_writes_no_files(self, base, pending_report, monkeypatch):
        def fail(self):
            raise ValueError("broken cell")

        monkeypatch.setattr(Report, "render_markdown", fail)
        out = base / "scratch"
        with pytest.raises(ValueError, match="broken"):
            write_report(pending_report, out)
        assert not out.exists()

    def test_report_write_failure_cleans_staging_and_preserves_outputs(
        self, base, pending_report, monkeypatch
    ):
        report = pending_report
        out = base / "outputs"
        out.mkdir()
        originals = {out / "t.md": "existing markdown\n", out / "t.json": '{"existing":true}\n'}
        for path, text in originals.items():
            path.write_text(text, encoding="utf-8")
        named_temporary_file = output_paths.tempfile.NamedTemporaryFile
        staged_count = 0

        def fail_write(_text):
            raise OSError("staged write failed")

        @contextmanager
        def fail_second_write(**kwargs):
            nonlocal staged_count
            staged_count += 1
            with named_temporary_file(**kwargs) as handle:
                if staged_count == 2:
                    monkeypatch.setattr(handle, "write", fail_write)
                yield handle

        monkeypatch.setattr(output_paths.tempfile, "NamedTemporaryFile", fail_second_write)
        with pytest.raises(OSError, match="staged write failed"):
            write_report(report, out)
        assert set(out.iterdir()) == set(originals)
        assert {path: path.read_text(encoding="utf-8") for path in originals} == originals

    def test_report_failed_restore_retains_backup_and_original_error(
        self, base, pending_report, monkeypatch
    ):
        report = pending_report
        out = (base / "outputs").resolve()
        out.mkdir()
        markdown = out / "t.md"
        metadata = out / "t.json"
        markdown.write_bytes(b"original markdown")
        metadata.write_bytes(b"original JSON")
        replace_file = output_paths.os.replace
        markdown_published = False

        def fail_json_replacement(source, destination):
            nonlocal markdown_published
            if destination == metadata:
                raise OSError("JSON replacement failed")
            if markdown_published:
                raise OSError("restore failed")
            replace_file(source, destination)
            markdown_published = True

        monkeypatch.setattr(output_paths.os, "replace", fail_json_replacement)
        with pytest.raises(
            RuntimeError, match="JSON replacement failed; could not restore"
        ) as error:
            write_report(report, out)
        assert isinstance(error.value.__cause__, OSError)
        backups = set(out.iterdir()) - {markdown, metadata}
        assert len(backups) == 1
        backup = backups.pop()
        assert backup.read_bytes() == b"original markdown"
        assert str(backup) in str(error.value)
        assert "restore failed" in str(error.value)
        assert metadata.read_bytes() == b"original JSON"

    def test_report_cannot_overwrite_spec(self, base, registry):
        spec = _make_report_spec(base)
        before = (base / "specs/t.json").read_bytes()
        report = build_report(spec, base, registry)
        with pytest.raises(ValueError, match="overwrite input"):
            write_report(report, base / "specs")
        assert (base / "specs/t.json").read_bytes() == before

    def test_report_cannot_overwrite_shared_training_manifest(self, base, registry):
        training = _write_training_manifest(base)
        output = base / "outputs"
        output.mkdir()
        training_path = output / "t.json"
        training.to_jsonl(training_path)
        before = training_path.read_bytes()
        _write_predictions(base / "reports/predictions/fine.jsonl", dict(_REFS), OUR_SYSTEM)
        spec = _make_fine_tuned_report_spec(base)
        spec = replace(
            spec,
            train_manifests=("outputs/t.json",),
            systems=(replace(spec.systems[0], train_manifests=()),),
        )
        report = build_report(spec, base, registry)
        assert report.cells[0].leakage_audit["status"] == "reviewed"
        with pytest.raises(ValueError, match="overwrite input"):
            write_report(report, output)
        assert training_path.read_bytes() == before
        assert not (output / "t.md").exists()

    @pytest.mark.parametrize(
        "name,model_kind,status,message",
        [
            ("fine-tuned Whisper", "fine_tuned", "scored", "reviewed"),
            ("Fine-tuned Whisper", "fine_tuned", "scored", "reviewed"),
            ("custom model", "fine_tuned", "scored", "reviewed"),
            ("Fine-tuned Whisper", "unknown", "scored", "model_kind"),
            *[
                ("s", "unknown", status, "cell status")
                for status in ("complete", "SCORED", "", None, False)
            ],
        ],
    )
    def test_direct_report_rejects_invalid_cells(
        self, base, pending_report, name, model_kind, status, message
    ):
        manifest = Manifest.from_jsonl(base / "data/tiny.manifest.jsonl")
        cell = score_slice(
            manifest,
            Predictions(name, manifest.name, dict(_REFS)),
            n_resamples=100,
            slice_name="tiny",
            model_kind=model_kind,
        )
        report = replace(
            pending_report, cells=(replace(cell, status=status),), system_order=(name,)
        )
        with pytest.raises(ValueError, match=message):
            write_report(report, base / "outputs")
        assert not (base / "outputs").exists()

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
    def test_report_rejects_mismatched_interval_metadata(self, report, export, change):
        cell = report.cells[0]
        broken = replace(cell, wer_ci=replace(cell.wer_ci, **change))
        with pytest.raises(ValueError, match="interval"):
            getattr(replace(report, cells=(broken,)), export)()

    @pytest.mark.parametrize("export", ["render_markdown", "to_json_dict"])
    def test_report_rejects_interval_percentage_overflow(self, report, export):
        cell = report.cells[0]
        invalid = replace(cell, wer_ci=replace(cell.wer_ci, upper=1e308))
        with pytest.raises(ValueError, match="report interval percentage"):
            getattr(replace(report, cells=(invalid,)), export)()

    def test_report_escapes_normalization_policy(self, report):
        cell = replace(report.cells[0], normalization_policy="custom|policy\nnext")
        assert "custom\\|policy next v1.1.0 |" in replace(report, cells=(cell,)).render_markdown()

    @pytest.mark.parametrize("export", ["render_markdown", "to_json_dict"])
    @pytest.mark.parametrize("kind", ["duplicate", "extra", "missing", "duplicate-order"])
    def test_report_exports_require_one_cell_per_declared_system_and_slice(
        self, report, export, kind
    ):
        if kind == "duplicate":
            report = replace(report, cells=report.cells * 2)
        elif kind == "extra":
            report = replace(
                report,
                cells=(*report.cells, replace(report.cells[0], system="hidden")),
            )
        elif kind == "missing":
            report = replace(report, cells=())
        else:
            report = replace(report, slice_order=report.slice_order * 2)
        with pytest.raises(ValueError, match="report.*cell"):
            getattr(report, export)()

    def test_report_distinguishes_unknown_working_tree_from_clean(self, report):
        context = replace(report.run_context, git=GitState("a" * 40, None))
        assert "(working tree unknown)" in replace(report, run_context=context).render_markdown()


class TestEvaluationCli(_ReportFixtures):
    def test_cli_writes_md_and_json(self, base):
        _write_predictions(base / "reports" / "predictions" / "s__tiny.jsonl", dict(_REFS), "sys-a")
        _make_report_spec(base, name="sys-a", predictions="reports/predictions/s__tiny.jsonl")
        rc = _evaluation_cli(base)
        assert rc == 0
        assert (base / "outputs" / "t.md").exists()
        payload = json.loads((base / "outputs" / "t.json").read_text(encoding="utf-8"))
        assert payload["normalizer_version"]
        assert payload["cells"][0]["status"] == "scored"

    def test_cli_unknown_report_exits_2(self, base):
        rc = _evaluation_cli(base, "--report", "does-not-exist")
        assert rc == 2

    @pytest.mark.parametrize("option,path", [("--spec", "absent.json"), ("--specs-dir", "absent")])
    def test_cli_missing_custom_spec_does_not_fall_back_to_builtin(
        self, base, option, path, capsys
    ):
        assert _evaluation_cli(base, "--report", "baselines", option, path) == 2
        assert "no spec" in capsys.readouterr().err
        assert not (base / "outputs").exists()

    def test_cli_incomplete_predictions_exits_1_and_writes_nothing(self, base):
        partial = {k: v for (k, v) in _REFS.items() if k != "loop02-001"}
        _write_predictions(base / "reports" / "predictions" / "s__tiny.jsonl", partial, "sys-a")
        _make_report_spec(base, name="sys-a", predictions="reports/predictions/s__tiny.jsonl")
        rc = _evaluation_cli(base)
        assert rc == 1
        assert not (base / "outputs" / "t.md").exists()

    @pytest.mark.parametrize(
        "name,cells", [("baselines", 3), ("longform", 4), ("codeswitch", 4), ("full", 6)]
    )
    def test_cli_builtin_report_is_valid_without_spec_files(self, base, name, cells):
        assert _evaluation_cli(base, "--report", name) == 0
        payload = json.loads((base / "outputs" / f"{name}.json").read_text(encoding="utf-8"))
        assert len(payload["cells"]) == cells
        assert all(c["status"] == "pending" for c in payload["cells"])

    def test_cli_requires_source_metadata_for_existing_manifest(self, base, capsys):
        _write_predictions(base / "reports/predictions/p.jsonl", dict(_REFS), "sys-a")
        _make_report_spec(base, name="sys-a", predictions="reports/predictions/p.jsonl")
        assert _evaluation_cli(base, "--report", "t", "--specs-dir", "specs") == 1
        assert "not in the license registry" in capsys.readouterr().err
        assert not (base / "outputs").exists()

    def test_cli_explicit_missing_registry_fails_instead_of_using_empty_registry(
        self, base, capsys
    ):
        assert _evaluation_cli(base, "--report", "baselines", "--licenses", "absent.toml") == 1
        assert "absent.toml" in capsys.readouterr().err
        assert not (base / "outputs").exists()

    @pytest.mark.parametrize(
        "bad", [[], None, "bad", {"name": "../x", "title": "x", "slices": [], "systems": []}]
    )
    def test_invalid_spec_shapes_cli_has_no_traceback(self, base, bad, capsys):
        (base / "specs/t.json").write_text(json.dumps(bad))
        assert _evaluation_cli(base) == 1
        assert "Traceback" not in capsys.readouterr().err
        assert not (base / "outputs").exists()

    def test_cli_explicit_relative_spec_resolves_against_base(self, base):
        _make_report_spec(base)
        assert (
            _evaluation_cli(
                base,
                "--report",
                "t",
                "--spec",
                "specs/t.json",
                "--specs-dir",
                "missing",
                "--licenses",
                "data/licenses.json",
            )
            == 0
        )
        assert (base / "outputs/t.json").is_file()

    @pytest.mark.parametrize("which", ["registry", "spec", "output"])
    def test_cli_io_failures_are_clear(self, base, which, capsys):
        _make_report_spec(base)
        if which == "registry":
            (base / "data/licenses.json").write_text("[]")
        elif which == "spec":
            (base / "specs/t.json").write_text("{")
        else:
            (base / "outputs").symlink_to("outputs")
        assert _evaluation_cli(base) == 1
        assert "Traceback" not in capsys.readouterr().err

    def test_cli_rejects_abbreviated_options(self, base):
        with pytest.raises(SystemExit) as error:
            evaluation_main(["--report", "t", "--base-d", str(base)])
        assert error.value.code == 2


class TestRunContext:
    def test_capture_records_reproducible_context(self):
        context = _capture_run_context()
        repeated = _capture_run_context()
        assert context.seed == 13
        assert context.config["model"] == "whisper-medium"
        assert context.command.startswith("python -m shono.eval")
        assert context.normalizer_version == NORMALIZER_VERSION
        assert context.timestamp == repeated.timestamp == "2026-07-23T09:30:00+00:00"
        assert context.git.sha == "abc123" and context.git.dirty is False
        round_trip = json.loads(json.dumps(context.to_dict()))
        assert round_trip["seed"] == 13
        assert round_trip["git"] == {"sha": "abc123", "dirty": False}
        assert round_trip["normalizer_version"] == NORMALIZER_VERSION
        assert "jiwer" in context.packages
        assert "bnunicodenormalizer" in context.packages

    def test_dirty_tree_is_recorded(self):
        ctx = _capture_run_context(git_state=GitState(sha="deadbeef", dirty=True))
        assert ctx.to_dict()["git"]["dirty"] is True

    def test_seed_everything_makes_python_random_reproducible(self):
        seed_everything(7)
        first = [random.random() for _ in range(5)]
        seed_everything(7)
        second = [random.random() for _ in range(5)]
        assert first == second

    def test_unknown_packages_are_explicit(self):
        assert _capture_run_context(packages=("shono-nonexistent-test-package",)).packages == {
            "shono-nonexistent-test-package": "unknown"
        }

    def test_named_credentials_are_redacted_without_changing_ordinary_arguments(self):
        context = _capture_run_context(
            command="HF_TOKEN=abc python run.py --api-key secret --max-new-tokens 20 "
            '--text "ordinary answer"',
            config={"api_key": "secret", "max_new_tokens": 20, "nested": {"hf_token": "abc"}},
            extra={"password": "p"},
        )
        text = json.dumps(context.to_dict())
        assert "secret" not in text and "abc" not in context.command
        assert "ordinary answer" in context.command and "--max-new-tokens 20" in context.command
        assert context.config["max_new_tokens"] == 20
        assert context.extra["password"] == "[redacted]"

    def test_metadata_nonfinite_is_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            _capture_run_context(config={"lr": float("nan")})

    def test_naive_capture_clock_is_rejected(self):
        with pytest.raises(ValueError, match="aware"):
            _capture_run_context(now=lambda: datetime(2026, 1, 1))

    def test_invalid_seed_never_seeds_python(self):
        original_state = random.getstate()
        with pytest.raises(ValueError, match="seed"):
            seed_everything(True)
        assert random.getstate() == original_state

    @pytest.mark.parametrize("key", ["command", "Command", "COMMAND"])
    def test_extra_named_command_and_api_aliases_do_not_retain_credentials(self, key):
        context = _capture_run_context(
            extra={key: "run --hf-token abc", "apiKey": "private", "DEEPGRAM_KEY": "private"}
        )
        text = json.dumps(context.extra)
        assert "abc" not in text and "private" not in text

    @pytest.mark.parametrize(
        "key",
        [
            "private_key",
            "PRIVATE_KEY",
            "private-key",
            "privateKey",
            "google_private_key",
            "secret_access_key",
            "AWS_SECRET_ACCESS_KEY",
            "aws-secret-access-key",
            "awsSecretAccessKey",
            "accessToken",
            "access_token",
            "AccessToken",
            "clientSecret",
            "CLIENT_SECRET",
            "hfToken",
            "HFToken",
            "Proxy-Authorization",
            "proxyAuthorization",
            "Cookie",
            "Set-Cookie",
            "signature",
            "x-amz-signature",
        ],
    )
    def test_credentials_are_redacted_from_nested_metadata(self, key):
        private_value = (
            "-----BEGIN PRIVATE KEY-----\nprivate-value-sentinel\n-----END PRIVATE KEY-----"
        )
        ordinary_fields = {
            "key": "ordinary-key",
            "model_key": "whisper-medium",
            "checkpoint_path": "/models/checkpoint-1",
        }
        context = _capture_run_context(
            config={key: private_value, **ordinary_fields},
            extra={"providers": [{key: private_value}], **ordinary_fields},
        )
        payload = context.to_dict()
        assert payload["config"][key] == "[redacted]"
        assert payload["extra"]["providers"][0][key] == "[redacted]"
        assert "private-value-sentinel" not in json.dumps(payload)
        for field, value in ordinary_fields.items():
            assert payload["config"][field] == payload["extra"][field] == value

    @pytest.mark.parametrize(
        "environment_name,flag",
        [
            ("PRIVATE_KEY", "--private-key"),
            ("GOOGLE_PRIVATE_KEY", "--private_key"),
            ("AWS_SECRET_ACCESS_KEY", "--aws-secret-access-key"),
            ("ACCESS_TOKEN", "--accessToken"),
            ("CLIENT_SECRET", "--clientSecret"),
            ("HF_TOKEN", "--HFToken"),
        ],
    )
    def test_private_credentials_are_redacted_from_quoted_commands(self, environment_name, flag):
        private_value = (
            "-----BEGIN PRIVATE KEY-----\ncommand-value-sentinel\n-----END PRIVATE KEY-----"
        )
        ordinary_arguments = ["--key", "ordinary-key", "--checkpoint", "/models/checkpoint-1"]
        context = _capture_run_context(
            command=shlex.join(
                [f"{environment_name}={private_value}", "run.py", flag, private_value]
                + ordinary_arguments
            )
        )
        assert shlex.split(context.command) == [
            f"{environment_name}=[redacted]",
            "run.py",
            flag,
            "[redacted]",
            *ordinary_arguments,
        ]
        assert "command-value-sentinel" not in json.dumps(context.to_dict())

    @pytest.mark.parametrize(
        "value,expected",
        [
            (
                "https://user:credential-sentinel@example.invalid/path?view=clean",
                "https://[redacted]@example.invalid/path?view=clean",
            ),
            (
                "https://example.invalid/path?token=credential-sentinel&view=clean",
                "https://example.invalid/path?token=[redacted]&view=clean",
            ),
            (
                "https://example.invalid/path?view=clean&access_token=credential-sentinel",
                "https://example.invalid/path?view=clean&access_token=[redacted]",
            ),
            (
                "https://example.invalid/path?access_token=credential-sentinel&revision=main",
                "https://example.invalid/path?access_token=[redacted]&revision=main",
            ),
            (
                "https://example.invalid/path?access%5Ftoken=credential-sentinel",
                "https://example.invalid/path?access%5Ftoken=[redacted]",
            ),
            (
                "https://example.invalid/#access_token=credential-sentinel",
                "https://example.invalid/#access_token=[redacted]",
            ),
            ("Authorization: Bearer credential-sentinel", "Authorization: [redacted]"),
            ("Proxy-Authorization: Basic credential-sentinel", "Proxy-Authorization: [redacted]"),
            ("Cookie: session=credential-sentinel", "Cookie: [redacted]"),
            ("X-API-Key: credential-sentinel", "X-API-Key: [redacted]"),
            (
                "https://example.invalid/path?view=clean#readme",
                "https://example.invalid/path?view=clean#readme",
            ),
        ],
    )
    def test_structured_credentials_are_redacted_in_metadata_and_commands(self, value, expected):
        context = _capture_run_context(
            command=shlex.join(["run.py", "--value", value, "--max-new-tokens", "20"]),
            config={"endpoint": value, "model_key": "whisper-medium"},
            extra={"requests": [value]},
        )
        payload = context.to_dict()
        assert payload["config"]["endpoint"] == payload["extra"]["requests"][0] == expected
        assert shlex.split(payload["command"]) == [
            "run.py",
            "--value",
            expected,
            "--max-new-tokens",
            "20",
        ]
        assert payload["config"]["model_key"] == "whisper-medium"
        assert "credential-sentinel" not in json.dumps(payload)

    def test_provenance_imports_before_evaluation_in_fresh_python(self):
        result = subprocess.run(
            [
                sys.executable,
                "-W",
                "error",
                "-c",
                "from shono.provenance import RunContext\n"
                "from shono.eval.report import Report, build_report\n"
                "assert RunContext.capture(0, {}).normalizer_version == '1.1.0'\n"
                "assert callable(Report) and callable(build_report)\n",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stderr == ""


class TestEvidenceTools:
    @pytest.fixture
    def scoring_fixtures(self, tmp_path, monkeypatch):
        artifacts = runpy.run_path(str(ROOT / "scripts/check_evidence.py"))["artifacts"]
        fixture = tmp_path / "scoring_cases.py"
        monkeypatch.setitem(artifacts.__globals__, "FIXTURE", fixture)
        return artifacts, fixture

    def test_chart_labels_follow_reordered_fixtures(self, scoring_fixtures):
        artifacts, fixture = scoring_fixtures
        cases = [SCORE_CASES[1], SCORE_CASES[0], *SCORE_CASES[2:]]
        fixture.write_text(f"SCORE_CASES = {cases!r}\n", encoding="utf-8")
        svg = ET.fromstring(artifacts()["scorer-check.svg"])
        texts = svg.findall(".//{http://www.w3.org/2000/svg}text")
        for label, percentage in [("1 word substituted", "33.3%"), ("Identical", "0.0%")]:
            label_node = next(node for node in texts if node.text == label)
            value = next(
                node
                for node in texts
                if node.get("x") == "1044" and node.get("y") == label_node.get("y")
            )
            assert value.text == percentage

    def test_chart_counts_follow_metric_expectations(self, scoring_fixtures):
        artifacts, fixture = scoring_fixtures
        cases = [{**SCORE_CASES[0], "expected_cer": None}, *SCORE_CASES[1:]]
        fixture.write_text(f"SCORE_CASES = {cases!r}\n", encoding="utf-8")
        svg = artifacts()["scorer-check.svg"]
        assert "6/6 WER · 4/4 CER checks passed" in svg
        assert "All 6 word-error and 4 character-error checks match" in svg

    def test_chart_grows_and_scales_to_fit_additional_fixtures(self, scoring_fixtures):
        artifacts, fixture = scoring_fixtures
        cases = [
            *SCORE_CASES,
            {
                "name": "two_insertions",
                "label": "2 inserted <words> & more",
                "references": ["আমি"],
                "hypotheses": ["আমি ভাত খাই"],
                "expected_wer": 2.0,
                "expected_cer": None,
            },
        ]
        fixture.write_text(f"SCORE_CASES = {cases!r}\n", encoding="utf-8")
        svg = ET.fromstring(artifacts()["scorer-check.svg"])
        assert svg.get("viewBox") == "0 0 1200 732"
        texts = {node.text for node in svg.findall(".//{http://www.w3.org/2000/svg}text")}
        assert {"200%", "2 inserted <words> & more", "7/7 WER · 5/5 CER checks passed"} <= texts
        bars = [
            node
            for node in svg.findall(".//{http://www.w3.org/2000/svg}rect")
            if node.get("fill") == "#2873ab"
        ]
        assert len(bars) == 7
        assert all(0 <= float(node.get("width")) <= 650 for node in bars)

    @pytest.mark.parametrize(
        "cases", [[], [{**case, "expected_cer": None} for case in SCORE_CASES]]
    )
    def test_empty_metric_checks_fail(self, scoring_fixtures, cases):
        artifacts, fixture = scoring_fixtures
        fixture.write_text(f"SCORE_CASES = {cases!r}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must include WER and CER"):
            artifacts()

    @pytest.mark.parametrize("argument,status", [("--help", 0), ("--invalid", 2)])
    def test_evidence_cli_handles_options_without_work(self, argument, status):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/check_evidence.py"), argument],
            capture_output=True,
            text=True,
        )
        assert result.returncode == status
        assert "Traceback" not in result.stderr

    def test_hand_computed_artifact_is_current(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/check_evidence.py"), "--check"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "model accuracy remains unmeasured" in result.stdout

    def test_artifact_generation_writes_only_csv_and_svg(self, tmp_path, monkeypatch):
        expected = {
            name: (ROOT / "outputs" / name).read_bytes()
            for name in ("scorer-check.csv", "scorer-check.svg")
        }
        script = ROOT / "scripts/check_evidence.py"
        evidence_main = runpy.run_path(str(script))["main"]
        output = tmp_path / "artifacts"
        monkeypatch.setitem(evidence_main.__globals__, "OUTPUT", output)
        monkeypatch.setattr(sys, "argv", [str(script), "--write"])
        assert evidence_main() == 0
        assert {path.name: path.read_bytes() for path in output.iterdir()} == expected
        monkeypatch.setattr(sys, "argv", [str(script), "--check"])
        assert evidence_main() == 0
        assert {path.name for path in output.iterdir()} == set(expected)


class TestArtifactPublication(_ReportFixtures):
    @pytest.fixture(params=["report", "evidence"])
    def publication(self, request, base, pending_report, monkeypatch):
        output = base / "artifacts"
        if request.param == "report":
            names = ("t.md", "t.json")
            publish = partial(write_report, pending_report, output)
        else:
            script = ROOT / "scripts/check_evidence.py"
            publish = runpy.run_path(str(script))["main"]
            monkeypatch.setitem(publish.__globals__, "OUTPUT", output)
            monkeypatch.setattr(sys, "argv", [str(script), "--write"])
            names = ("scorer-check.csv", "scorer-check.svg")
        return output, names, publish

    @pytest.mark.parametrize("existing", [(), (0,), (1,), (0, 1)])
    def test_second_replacement_failure_restores_outputs(self, publication, monkeypatch, existing):
        output, names, publish = publication
        output.mkdir()
        originals = {
            output / names[index]: f"original {names[index]}\n".encode() for index in existing
        }
        for path, content in originals.items():
            path.write_bytes(content)
        replace_file = output_paths.os.replace
        message = f"{Path(names[1]).suffix[1:].upper()} replacement failed"

        def fail_second_replacement(source, destination):
            if destination == output / names[1]:
                raise OSError(message)
            replace_file(source, destination)

        monkeypatch.setattr(output_paths.os, "replace", fail_second_replacement)
        with pytest.raises(OSError, match=message):
            publish()
        assert set(output.iterdir()) == set(originals)
        assert {path: path.read_bytes() for path in originals} == originals
