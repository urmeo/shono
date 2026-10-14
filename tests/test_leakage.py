import hashlib
import os
import shutil
from dataclasses import replace

import pytest

from shono.data import Manifest, Segment, audit_leakage, leakage, require_no_leakage


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
    train = _m(
        "train",
        "train",
        [_seg("t1", "আজকে ভালো দিন", sha="a" * 64), _seg("t2", "কেমন আছেন", sha="b" * 64)],
    )
    test = _m("test", "test", [_seg("e1", "নতুন বাক্য এখানে", sha="c" * 64)])
    report = audit_leakage(train, test)
    assert report.is_complete and report.is_clean and report.is_approved
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
    assert not report.is_complete and not report.is_approved
    assert report.checked_kinds == ("audio",)
    assert report.summary().startswith("INCOMPLETE:")


def test_missing_hashes_cannot_approve_an_audit():
    train = _m("train", "train", [_seg("t1", "প্রথম")])
    test = _m("test", "test", [_seg("e1", "দ্বিতীয়")])
    report = audit_leakage(train, test)
    assert not report.is_clean and not report.is_approved
    assert report.audio_coverage == 0
    assert report.summary().startswith("INCOMPLETE:")
    with pytest.raises(ValueError, match="incomplete train/evaluation leakage audit"):
        require_no_leakage(train, test)


@pytest.mark.parametrize("check", [("id",), ("text",), ("audio",), ("id", "text")])
def test_subset_checks_cannot_approve_an_audit(check):
    train = _m("train", "train", [_seg("t1", "প্রথম", sha="a" * 64)])
    test = _m("test", "test", [_seg("e1", "দ্বিতীয়", sha="b" * 64)])
    report = audit_leakage(train, test, check=check)
    assert report.audio_coverage == 1
    assert report.checked_kinds == check
    assert not report.is_complete and not report.is_clean and not report.is_approved
    assert report.summary().startswith("INCOMPLETE:")


def test_copied_audio_without_hashes_is_hard_leakage(tmp_path):
    (tmp_path / "t1.wav").write_bytes(b"same recording bytes")
    shutil.copyfile(tmp_path / "t1.wav", tmp_path / "e1.wav")
    train = _m("train", "train", [_seg("t1", "প্রথম")])
    test = _m("test", "test", [_seg("e1", "দ্বিতীয়")])
    report = audit_leakage(train, test, audio_root=tmp_path)
    assert report.is_complete and not report.is_approved
    assert report.audio_coverage == 1
    assert report.train_hashed == report.eval_hashed == 1
    assert report.by_kind("audio")[0].key == hashlib.sha256(b"same recording bytes").hexdigest()
    assert not report.by_kind("file")
    with pytest.raises(ValueError, match="hard train/evaluation overlap"):
        require_no_leakage(train, test, audio_root=tmp_path)
    assert train.segments[0].audio_sha256 is None


def test_distinct_files_without_hashes_can_pass_a_complete_audit(tmp_path):
    for name in ("t1", "e1"):
        (tmp_path / f"{name}.wav").write_bytes(name.encode())
    train = _m("train", "train", [_seg("t1", "প্রথম")])
    test = _m("test", "test", [_seg("e1", "দ্বিতীয়")])
    report = require_no_leakage(train, test, audio_root=tmp_path)
    assert report.is_complete and report.is_clean and report.is_approved
    assert report.audio_coverage == 1
    assert report.checked_kinds == ("id", "text", "audio", "file")


def test_declared_hashes_are_verified_against_file_bytes(tmp_path):
    (tmp_path / "t1.wav").write_bytes(b"train bytes")
    (tmp_path / "e1.wav").write_bytes(b"eval bytes")
    train = _m("train", "train", [_seg("t1", "প্রথম", sha="a" * 64)])
    test = _m("test", "test", [_seg("e1", "দ্বিতীয়")])
    with pytest.raises(ValueError, match="audio_sha256 does not match file bytes"):
        audit_leakage(train, test, audio_root=tmp_path)


def test_hash_cache_covers_shared_files_and_hardlinks_only_within_one_audit(tmp_path, monkeypatch):
    (tmp_path / "t1.wav").write_bytes(b"one file")
    os.link(tmp_path / "t1.wav", tmp_path / "e1.wav")
    train = _m(
        "train",
        "train",
        [_seg("t1", "প্রথম"), replace(_seg("t2", "তৃতীয়"), audio="t1.wav")],
    )
    test = _m("test", "test", [_seg("e1", "দ্বিতীয়")])
    reads = []
    original = leakage.file_sha256

    def tracked_hash(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(leakage, "file_sha256", tracked_hash)
    for _ in range(2):
        report = audit_leakage(train, test, audio_root=tmp_path)
        assert report.train_hashed == 2 and report.eval_hashed == 1
        assert len(report.by_kind("audio")) == len(report.by_kind("file")) == 1
    assert reads == [tmp_path / "t1.wav", tmp_path / "t1.wav"]


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
