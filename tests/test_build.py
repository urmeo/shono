"""Manifest builders: rows/CV/FLEURS → a valid, license-checked Manifest."""

from pathlib import Path

import pytest

from shono.data import (
    LicenseRegistry,
    Manifest,
    Segment,
    build_manifest,
    collapse_to_recordings,
    from_common_voice_tsv,
    from_fleurs_tsv,
)

_LICENSES = Path(__file__).resolve().parents[1] / "data" / "licenses.json"


def _fake_duration(_path):
    return 3.0  # inject a fixed duration instead of reading audio


# ---- build_manifest ------------------------------------------------------


def test_build_manifest_from_rows_uses_injected_duration():
    rows = [
        {"id": "a", "audio": "a.wav", "text": "আমি ভালো"},
        {"id": "b", "audio": "b.wav", "text": "তুমি কেমন", "duration_s": 5.0},
    ]
    m = build_manifest(
        name="m", source="openslr_slr53", split="train", domain="read",
        version="v", rows=rows, duration_of=_fake_duration,
    )
    assert len(m.segments) == 2
    assert m.segments[0].duration_s == 3.0  # from the injected duration_of
    assert m.segments[1].duration_s == 5.0  # from the row (not recomputed)
    assert m.segments[0].recording_id == "a"  # defaults to id


def test_build_manifest_skips_empty_text_rows():
    rows = [
        {"id": "a", "audio": "a.wav", "text": "কথা"},
        {"id": "b", "audio": "b.wav", "text": "   "},  # unscorable → skipped
    ]
    m = build_manifest(
        name="m", source="openslr_slr53", split="train", domain="read",
        version="v", rows=rows, duration_of=_fake_duration,
    )
    assert [s.id for s in m.segments] == ["a"]


def test_build_manifest_validates_against_license_floor():
    m = build_manifest(
        name="m", source="common_voice_bn", split="test", domain="read",
        version="v", rows=[{"id": "a", "audio": "a.wav", "text": "কথা"}],
        duration_of=_fake_duration,
    )
    m.validate_against(LicenseRegistry.load(_LICENSES))  # source registered → no raise


# ---- Common Voice --------------------------------------------------------


def test_from_common_voice_tsv(tmp_path):
    tsv = tmp_path / "test.tsv"
    tsv.write_text(
        "client_id\tpath\tsentence\n"
        "spk1\tcv_bn_1.mp3\tআমি ভাত খাই\n"
        "spk1\tcv_bn_2.mp3\tতুমি কি করছ\n"
        "spk2\tcv_bn_3.mp3\t\n",  # empty sentence → skipped
        encoding="utf-8",
    )
    m = from_common_voice_tsv(
        tsv, split="test", name="cv-bn-test", version="cv-26",
        duration_of=_fake_duration,
    )
    assert len(m.segments) == 2
    assert m.segments[0].audio == "clips/cv_bn_1.mp3"
    assert m.segments[0].recording_id == "spk1"  # client_id is the CI block
    assert m.source == "common_voice_bn"


def test_from_common_voice_uses_durations_tsv(tmp_path):
    (tmp_path / "test.tsv").write_text(
        "client_id\tpath\tsentence\nspk1\tclip.mp3\tকথা\n", encoding="utf-8"
    )
    (tmp_path / "clip_durations.tsv").write_text(
        "clip\tduration[ms]\nclip.mp3\t4200\n", encoding="utf-8"
    )
    m = from_common_voice_tsv(
        tmp_path / "test.tsv", split="test", name="cv", version="v",
        durations_tsv=tmp_path / "clip_durations.tsv",
    )
    assert m.segments[0].duration_s == pytest.approx(4.2)  # from the durations file


# ---- FLEURS --------------------------------------------------------------


def test_collapse_to_recordings_joins_segments_per_recording():
    segs = (
        Segment(id="r1-1", audio="r1.wav", text="দ্বিতীয়", duration_s=2.0,
                recording_id="r1", start_s=4.0),
        Segment(id="r1-0", audio="r1.wav", text="প্রথম", duration_s=2.0,
                recording_id="r1", start_s=0.0),
        Segment(id="r2-0", audio="r2.wav", text="আলাদা", duration_s=3.0, recording_id="r2"),
    )
    m = Manifest(name="loop", source="bengali_loop", split="test", domain="long-form",
                 version="v", segments=segs)
    collapsed = collapse_to_recordings(m)
    assert len(collapsed.segments) == 2  # one entry per recording
    r1 = next(s for s in collapsed.segments if s.recording_id == "r1")
    assert r1.text == "প্রথম দ্বিতীয়"  # joined in start-time order
    assert r1.duration_s == 4.0  # summed
    assert r1.id == "r1"


def test_from_fleurs_tsv_derives_duration_from_num_samples(tmp_path):
    tsv = tmp_path / "test.tsv"
    # headerless: id  file_name  raw  transcription  num_samples  gender
    tsv.write_text(
        "1\t1.wav\tকাঁচা\tকাঁচা লেখা\t48000\tFEMALE\n"
        "2\t2.wav\t\t\t16000\tMALE\n",  # empty transcription → skipped
        encoding="utf-8",
    )
    m = from_fleurs_tsv(tsv, split="test", name="fleurs-bn-test")
    assert len(m.segments) == 1
    assert m.segments[0].duration_s == pytest.approx(48000 / 16000)  # 3.0 s
    assert m.segments[0].text == "কাঁচা লেখা"
    assert m.source == "fleurs_bn"
