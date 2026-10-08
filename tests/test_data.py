"""The data layer: manifests round-trip, validate hard, and honour the license floor."""

import json
from pathlib import Path

import pytest
from data_fixtures import synthetic_registry

from shono.data import DatasetLicense, LicenseRegistry, Manifest, Segment

_FIXTURES = Path(__file__).parent / "fixtures"
_TINY = _FIXTURES / "tiny_manifest.jsonl"
_LICENSES = Path(__file__).resolve().parents[1] / "data" / "licenses.json"


@pytest.fixture
def registry() -> LicenseRegistry:
    return LicenseRegistry.load(_LICENSES)


def test_tiny_manifest_loads():
    m = Manifest.from_jsonl(_TINY)
    assert m.name == "tiny-longform-test"
    assert m.source == "bengali_loop"
    assert m.split == "test"
    assert len(m.segments) == 5
    assert m.recording_ids() == ["rec-loop-01", "rec-loop-02"]


def test_total_hours_sums_durations():
    m = Manifest.from_jsonl(_TINY)
    assert m.total_hours() == pytest.approx(16.8 / 3600.0)


def test_jsonl_round_trip_is_lossless(tmp_path):
    m = Manifest.from_jsonl(_TINY)
    out = tmp_path / "rt.jsonl"
    m.to_jsonl(out)
    again = Manifest.from_jsonl(out)
    assert again == m


def test_header_line_is_required(tmp_path):
    bad = tmp_path / "noheader.jsonl"
    bad.write_text(
        json.dumps(
            {"id": "x", "audio": "a.wav", "text": "কথা", "duration_s": 1.0, "recording_id": "r"}
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="header line"):
        Manifest.from_jsonl(bad)


def test_malformed_manifest_header_gives_actionable_error(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps({"manifest": {"source": "openslr_slr53", "split": "test"}})
        + "\n"
        + json.dumps(
            {"id": "s", "audio": "a.wav", "text": "কথা", "duration_s": 1.0, "recording_id": "r"}
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required field"):
        Manifest.from_jsonl(bad)


def test_segment_missing_required_field_gives_line_number(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps(
            {
                "manifest": {
                    "name": "m",
                    "source": "openslr_slr53",
                    "split": "test",
                    "domain": "read",
                    "version": "v",
                }
            }
        )
        + "\n"
        + json.dumps({"id": "s", "audio": "a.wav", "text": "কথা"})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="line 2"):
        Manifest.from_jsonl(bad)


def test_empty_reference_text_rejected():
    with pytest.raises(ValueError, match="empty reference"):
        Segment(id="s", audio="a.wav", text="   ", duration_s=1.0, recording_id="r")


def test_nonpositive_duration_rejected():
    with pytest.raises(ValueError, match="non-positive duration"):
        Segment(id="s", audio="a.wav", text="কথা", duration_s=0.0, recording_id="r")


def test_empty_recording_id_rejected():
    with pytest.raises(ValueError, match="recording_id"):
        Segment(id="s", audio="a.wav", text="কথা", duration_s=1.0, recording_id="")


def test_negative_start_s_rejected():
    with pytest.raises(ValueError, match="negative start_s"):
        Segment(id="s", audio="a.wav", text="কথা", duration_s=1.0, recording_id="r", start_s=-0.5)


def test_duplicate_segment_ids_rejected():
    seg = Segment(id="dup", audio="a.wav", text="কথা", duration_s=1.0, recording_id="r")
    with pytest.raises(ValueError, match="duplicate segment id"):
        Manifest(
            name="m",
            source="openslr_slr53",
            split="test",
            domain="read",
            version="v",
            segments=(seg, seg),
        )


def test_unknown_split_rejected():
    seg = Segment(id="s", audio="a.wav", text="কথা", duration_s=1.0, recording_id="r")
    with pytest.raises(ValueError, match="split"):
        Manifest(
            name="m",
            source="openslr_slr53",
            split="validation",
            domain="read",
            version="v",
            segments=(seg,),
        )


def test_empty_manifest_rejected():
    with pytest.raises(ValueError, match="no segments"):
        Manifest(
            name="m",
            source="openslr_slr53",
            split="test",
            domain="read",
            version="v",
            segments=(),
        )


def test_records_with_joins_on_id():
    m = Manifest.from_jsonl(_TINY)
    hyps = {sid: "অনুমান" for sid in m.ids()}
    records = m.records_with(hyps)
    assert len(records) == 5
    assert records[0][0] == "rec-loop-01"
    assert records[0][2] == "অনুমান"


def test_records_with_missing_hypothesis_raises():
    m = Manifest.from_jsonl(_TINY)
    hyps = {sid: "অনুমান" for sid in m.ids()[:-1]}
    with pytest.raises(KeyError, match="no hypothesis"):
        m.records_with(hyps)


def test_registry_loads_all_real_sources(registry):
    for src in ("common_voice_bn", "openslr_slr53", "bengali_loop", "mucs_slr104"):
        assert src in registry
        assert isinstance(registry[src], DatasetLicense)


def test_active_sources_exclude_ood_speech(registry):
    assert "ood_speech" not in registry.sources("active")
    assert registry["ood_speech"].status == "excluded"


def test_require_unknown_source_is_actionable(registry):
    with pytest.raises(KeyError, match="not in the license registry"):
        registry.require("no_such_corpus")


def test_render_table_lists_sources_and_licenses(registry):
    table = registry.render_table()
    assert "CC0-1.0" in table
    assert "CC-BY-SA-4.0" in table
    assert "Common Voice" in table
    assert table.index("active") < table.index("excluded")


def test_excluded_source_cannot_be_referenced(registry):
    seg = Segment(id="s", audio="a.wav", text="কথা", duration_s=1.0, recording_id="r")
    m = Manifest(
        name="ood",
        source="ood_speech",
        split="test",
        domain="out-of-domain",
        version="v",
        segments=(seg,),
    )
    with pytest.raises(ValueError, match="excluded source"):
        m.validate_against(registry)


def test_eval_only_source_cannot_be_training_data(registry):
    registry = synthetic_registry("youtube_eval", status="eval-only")
    seg = Segment(id="s", audio="a.wav", text="কথা", duration_s=1.0, recording_id="r")
    m = Manifest(
        name="yt-train",
        source="youtube_eval",
        split="train",
        domain="long-form",
        version="v",
        segments=(seg,),
    )
    with pytest.raises(ValueError, match="eval-only"):
        m.validate_against(registry)


def test_active_source_validates_clean(registry):
    m = Manifest.from_jsonl(_TINY)
    m.validate_against(synthetic_registry("bengali_loop"))
