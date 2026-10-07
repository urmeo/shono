"""Real input boundaries, joins and immutable path contracts."""

import json
import os
import wave
from dataclasses import asdict, replace

import pytest
from data_fixtures import synthetic_registry

from shono.data import (
    LicenseRegistry,
    Manifest,
    Segment,
    build_manifest,
    collapse_to_recordings,
    from_loop_jsonl,
    from_mucs_kaldi,
    from_slr53_tsv,
    require_no_leakage,
    validate_audio_paths,
    validate_audio_window,
    validate_manifest_audio,
)
from shono.data.audio_paths import file_sha256
from shono.data.leakage import audit_leakage


def segment(**kwargs):
    return Segment(
        **(dict(id="a", audio="a.wav", text="কথা", duration_s=3, recording_id="r") | kwargs)
    )


def manifest(*segments, **kwargs):
    return Manifest(
        **(
            dict(
                name="m",
                source="synthetic",
                split="train",
                domain="read",
                version="fixture",
                segments=segments or (segment(),),
            )
            | kwargs
        )
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("duration_s", True),
        ("duration_s", complex(3, 0)),
        ("duration_s", float("nan")),
        ("duration_s", float("inf")),
        ("duration_s", 10**1000),
        ("start_s", False),
        ("start_s", float("nan")),
        ("start_s", float("inf")),
        ("audio", "../a.wav"),
        ("audio", "/a.wav"),
        ("audio", "C:/a.wav"),
        ("audio", "a\\b.wav"),
        ("audio", "./a.wav"),
        ("id", None),
        ("recording_id", " "),
        ("audio_sha256", "short"),
        ("language", None),
    ],
)
def test_segment_rejects_invalid_contract(field, value):
    with pytest.raises(ValueError):
        segment(**{field: value})


@pytest.mark.parametrize(
    "field,value", [("text", None), ("id", None), ("duration_s", True), ("start_s", True)]
)
def test_builder_validates_before_coercion(field, value):
    row = dict(id="a", audio="a.wav", text="কথা", duration_s=3)
    row[field] = value
    with pytest.raises(ValueError):
        build_manifest(
            name="m", source="synthetic", split="train", domain="read", version="v", rows=[row]
        )


def test_builder_preserves_hash_language_and_requires_aligned_duration():
    row = dict(
        id="a", audio="a.wav", text="কথা", duration_s=3, language="bn-en", audio_sha256="A" * 64
    )
    result = build_manifest(
        name="m", source="synthetic", split="train", domain="read", version="v", rows=[row]
    )
    assert result.segments[0].language == "bn-en"
    assert result.segments[0].audio_sha256 == "a" * 64
    row.pop("duration_s")
    row["start_s"] = 5
    with pytest.raises(ValueError, match="aligned span"):
        build_manifest(
            name="m",
            source="synthetic",
            split="train",
            domain="read",
            version="v",
            rows=[row],
            duration_of=lambda p: 10,
        )


@pytest.mark.parametrize(
    "header,record",
    [
        (None, {}),
        ({"segments": []}, {}),
        ({}, {"start_secs": 90}),
    ],
)
def test_jsonl_shape_and_unknown_fields_are_rejected(tmp_path, header, record):
    meta = dict(name="m", source="synthetic", split="train", domain="read", version="v")
    if header is not None:
        meta.update(header)
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps({"manifest": None if header is None else meta})
        + "\n\n"
        + json.dumps(asdict(segment()) | record)
        + "\n"
    )
    with pytest.raises(ValueError):
        Manifest.from_jsonl(path)


def test_jsonl_duplicate_fields_and_physical_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    head = json.dumps(
        {"manifest": dict(name="m", source="synthetic", split="train", domain="read", version="v")}
    )
    path.write_text(head + '\n\n{"id":"a","id":"b"}\n')
    with pytest.raises(ValueError, match="line 3.*duplicate"):
        Manifest.from_jsonl(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("hours", float("nan")),
        ("hours", True),
        ("hours", -1),
        ("redistributable_audio", "false"),
        ("share_alike", 0),
    ],
)
def test_registry_shapes_and_values(tmp_path, field, value):
    lic = asdict(next(iter(synthetic_registry("synthetic"))))
    lic[field] = value
    path = tmp_path / "licenses.json"
    path.write_text(json.dumps({"licenses": [lic]}))
    with pytest.raises(ValueError):
        LicenseRegistry.load(path)


def test_unknown_license_flags_are_honest():
    lic = next(iter(synthetic_registry("synthetic")))
    assert "unknown" in LicenseRegistry([replace(lic, redistributable_audio=None)]).render_table()


def test_real_wav_whole_file_and_explicit_crop(tmp_path):
    path = tmp_path / "a.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 16000 * 4)
    assert validate_audio_window(path) == 4
    assert validate_audio_window(path, 0, 3) == 3
    with pytest.raises(ValueError, match="whole-file duration"):
        validate_audio_window(path, None, 3)
    with pytest.raises(ValueError, match="past EOF"):
        validate_audio_window(path, 3, 2)
    with pytest.raises(ValueError, match="empty"):
        validate_audio_window(path, 4, 1)
    assert validate_audio_window(path, 3, 1.0005) == 1
    validate_manifest_audio(
        manifest(segment(duration_s=4, audio_sha256=file_sha256(path))), tmp_path
    )
    with pytest.raises(ValueError, match="does not match file bytes"):
        validate_manifest_audio(manifest(segment(duration_s=4, audio_sha256="f" * 64)), tmp_path)


def test_paths_preflight_all_sources_and_aliases(tmp_path):
    for directory in ("one", "two"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "a.wav").write_bytes(b"fixture")
    result = validate_audio_paths(
        manifest(segment(audio="one/a.wav"), segment(id="b", audio="two/a.wav")), tmp_path
    )
    assert result == (tmp_path / "one/a.wav", tmp_path / "two/a.wav")
    (tmp_path / "cycle").symlink_to("cycle")
    with pytest.raises(ValueError, match="resolve"):
        validate_audio_paths(manifest(segment(audio="cycle")), tmp_path)
    (tmp_path / "escape").symlink_to(tmp_path.parent)
    with pytest.raises(ValueError, match="escapes"):
        validate_audio_paths(manifest(segment(audio="escape/a.wav")), tmp_path)


def test_hardlinks_are_hard_leakage_without_hashes(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"fixture")
    os.link(tmp_path / "a.wav", tmp_path / "b.wav")
    train = manifest(segment(text="প্রথম"))
    evaluation = manifest(segment(id="b", audio="b.wav", text="দ্বিতীয়"), name="dev", split="dev")
    with pytest.raises(ValueError, match="hard train/evaluation"):
        require_no_leakage(train, evaluation, audio_root=tmp_path)


def test_ids_are_source_qualified_and_text_review_is_exact():
    train = manifest(segment(text="বাক্য"))
    evaluation = manifest(segment(text="অন্য"), name="dev", source="other", split="dev")
    assert require_no_leakage(train, evaluation).is_clean
    evaluation = replace(evaluation, segments=(segment(id="b", text="বাক্য।"),))
    with pytest.raises(ValueError, match="exact-key"):
        require_no_leakage(train, evaluation)
    report = require_no_leakage(
        train, evaluation, text_decisions={"বাক্য": "common synthetic sentence"}
    )
    assert report.is_approved and report.reviewed_text[0].reason
    with pytest.raises(ValueError, match="exact-key"):
        require_no_leakage(train, evaluation, text_decisions={"typo": "reviewed"})


def test_hash_coverage_does_not_hide_missing_eval():
    train = manifest(
        *(
            segment(id=str(i), audio=f"{i}.wav", text=f"train{i}", audio_sha256="a" * 64)
            for i in range(1000)
        )
    )
    evaluation = manifest(segment(id="eval", text="evaluation"), name="dev", split="dev")
    report = audit_leakage(train, evaluation)
    assert report.train_hashed == 1000 and report.eval_hashed == 0
    assert "eval 0/1" in report.summary() and "100%" not in report.summary()


@pytest.mark.parametrize("side", ["train", "evaluation"])
def test_leakage_boundary_rejects_detached_records(side):
    train = manifest()
    evaluation = manifest(segment(id="dev"), name="dev", split="dev")
    with pytest.raises(ValueError, match="Manifest records"):
        require_no_leakage(
            [None] if side == "train" else train, None if side == "evaluation" else evaluation
        )


def test_collapse_bound_full_duration_and_ambiguities(tmp_path):
    m = manifest(segment(id="a", start_s=0, duration_s=2), segment(id="b", start_s=4, duration_s=2))
    assert collapse_to_recordings(m).segments[0].duration_s == 6
    assert collapse_to_recordings(m, duration_of=lambda p: 7).segments[0].duration_s == 7
    with pytest.raises(ValueError, match="cover"):
        collapse_to_recordings(m, duration_of=lambda p: 5)
    for changed in (
        dict(audio="other.wav"),
        dict(start_s=None),
        dict(start_s=1),
        dict(language="en"),
        dict(audio_sha256="a" * 64),
    ):
        bad = replace(m, segments=(m.segments[0], replace(m.segments[1], **changed)))
        with pytest.raises(ValueError):
            collapse_to_recordings(bad)


def test_loop_published_recording_jsonl_is_not_training(tmp_path):
    path = tmp_path / "loop.jsonl"
    path.write_text(
        json.dumps(
            dict(id="recording", audio_filepath="a.wav", duration=3600, text="সম্পূর্ণ রেকর্ডিং")
        )
        + "\n"
    )
    m = from_loop_jsonl(path, name="loop", version="fixture", audio_prefix="loop-mount")
    assert m.segments[0].audio == "loop-mount/a.wav" and m.segments[0].start_s is None
    with pytest.raises(ValueError, match="aligned"):
        from_loop_jsonl(path, name="loop", version="fixture", split="train")


def test_kaldi_joins_preserve_full_text_times_and_speaker(tmp_path):
    (tmp_path / "audio space.wav").write_bytes(b"fixture")
    text = tmp_path / "text"
    text.write_text("u1 বাংলা and English words\nu2\tদ্বিতীয় বাক্য\n")
    wav = tmp_path / "wav.scp"
    wav.write_text("r audio space.wav\n")
    spans = tmp_path / "segments"
    spans.write_text("u1 r 0 3\nu2 r 4 7\n")
    speakers = tmp_path / "utt2spk"
    speakers.write_text("u1 speaker1\nu2 speaker2\n")
    kwargs = dict(
        name="mucs", split="train", audio_root=tmp_path, segments_path=spans, utt2spk_path=speakers
    )
    m = from_mucs_kaldi(text, wav, **kwargs)
    assert m.segments[0].text == "বাংলা and English words"
    assert m.segments[1].start_s == 4 and m.segments[1].speaker == "speaker2"
    assert m.segments[0].recording_id == "r"
    wav.write_text("r sox source.wav -t wav - |\n")
    with pytest.raises(ValueError, match="commands/pipes"):
        from_mucs_kaldi(text, wav, **kwargs)
    wav.write_text("r audio space.wav\n")
    speakers.write_text("u1 speaker1\n")
    with pytest.raises(ValueError, match="utt2spk"):
        from_mucs_kaldi(text, wav, **kwargs)


def test_slr53_published_three_column_index_and_file_mapping(tmp_path):
    local = tmp_path / "slr"
    local.mkdir()
    (local / "part-0").mkdir()
    (local / "part-0/file0.wav").write_bytes(b"fixture")
    index = local / "utt_spk_text.tsv"
    index.write_text("file0\tuser0\tবাংলা বাক্য\n")
    m = from_slr53_tsv(
        index,
        name="slr",
        split="train",
        audio_root=tmp_path,
        dataset_root=local,
        duration_of=lambda p: 3,
    )
    assert m.segments[0].audio == "slr/part-0/file0.wav"
    assert m.segments[0].text == "বাংলা বাক্য"
    assert m.segments[0].speaker == m.segments[0].recording_id == "user0"
    index.write_text("file0\tবাংলা বাক্য\n")
    with pytest.raises(ValueError, match="FileID, UserID"):
        from_slr53_tsv(
            index,
            name="slr",
            split="train",
            audio_root=tmp_path,
            dataset_root=local,
            duration_of=lambda p: 3,
        )
    index.write_text("missing\tuser0\tবাংলা বাক্য\n")
    with pytest.raises(ValueError, match="missing WAV"):
        from_slr53_tsv(
            index,
            name="slr",
            split="train",
            audio_root=tmp_path,
            dataset_root=local,
            duration_of=lambda p: 3,
        )
