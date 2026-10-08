"""The leakage audit finds id, text, and audio overlap : and stays honest when clean."""

from shono.data import Manifest, Segment, audit_leakage


def _m(name, split, segs):
    return Manifest(
        name=name,
        source="openslr_slr53",
        split=split,
        domain="read",
        version="v",
        segments=tuple(segs),
    )


def _seg(sid, text, *, rec=None, sha=None):
    return Segment(
        id=sid,
        audio=f"{sid}.wav",
        text=text,
        duration_s=2.0,
        recording_id=rec or sid,
        audio_sha256=sha,
    )


def test_clean_when_no_overlap():
    train = _m("train", "train", [_seg("t1", "আজকে ভালো দিন"), _seg("t2", "কেমন আছেন")])
    test = _m("test", "test", [_seg("e1", "নতুন বাক্য এখানে")])
    report = audit_leakage(train, test)
    assert report.is_clean
    assert "CLEAN" in report.summary()


def test_detects_shared_segment_id():
    train = _m("train", "train", [_seg("shared", "কথা এক")])
    test = _m("test", "test", [_seg("shared", "সম্পূর্ণ ভিন্ন বাক্য")])
    report = audit_leakage(train, test, check=["id"])
    ids = report.by_kind("id")
    assert len(ids) == 1 and ids[0].key == "openslr_slr53:shared"
    assert not report.is_clean


def test_detects_text_overlap_under_frozen_normalizer():
    train = _m("train", "train", [_seg("t1", "আমি ভাত খাই।")])
    test = _m("test", "test", [_seg("e1", "আমি ভাত খাই")])
    report = audit_leakage(train, test, check=["text"])
    text_overlaps = report.by_kind("text")
    assert len(text_overlaps) == 1
    assert "review each" in report.summary()


def test_detects_audio_checksum_overlap():
    train = _m("train", "train", [_seg("t1", "কথা এক", sha="d" * 64)])
    test = _m("test", "test", [_seg("e1", "কথা দুই", sha="d" * 64)])
    report = audit_leakage(train, test, check=["audio"])
    assert len(report.by_kind("audio")) == 1


def test_audio_coverage_reported_when_checksums_partial():
    train = _m("train", "train", [_seg("t1", "কথা", sha="a" * 64)])
    test = _m("test", "test", [_seg("e1", "ভিন্ন")])
    report = audit_leakage(train, test, check=["audio"])
    assert report.audio_coverage == 0.5


def test_audit_against_a_mix_of_train_manifests():
    a = _m("slr53", "train", [_seg("a1", "প্রথম")])
    b = _m("cv", "train", [_seg("b1", "দ্বিতীয় বাক্য")])
    test = _m("test", "test", [_seg("e1", "দ্বিতীয় বাক্য")])
    report = audit_leakage([a, b], test, check=["text"])
    assert not report.is_clean
    assert report.train_names == ("slr53", "cv")


def test_empty_train_list_rejected():
    test = _m("test", "test", [_seg("e1", "কিছু")])
    try:
        audit_leakage([], test)
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("expected ValueError for empty train list")
