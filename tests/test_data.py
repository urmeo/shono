import hashlib
import json
import os
import shutil
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import pytest
from support import TINY_MANIFEST, synthetic_registry, write_wav

from shono.checkpoints import hash_checkpoint
from shono.data import leakage
from shono.data.audio_paths import (
    file_sha256,
    validate_audio_paths,
    validate_audio_window,
    validate_manifest_audio,
)
from shono.data.build import (
    build_manifest,
    collapse_to_recordings,
    from_common_voice_tsv,
    from_fleurs_tsv,
    from_loop_jsonl,
    from_mucs_kaldi,
    from_slr53_tsv,
)
from shono.data.leakage import audit_leakage, require_no_leakage
from shono.data.license import DatasetLicense, LicenseRegistry
from shono.data.manifest import Manifest, Segment, load_manifest
from shono.data.prepare import DatasetMounts, prepare_manifests


def make_segment(**kwargs):
    return Segment(
        **(dict(id="a", audio="a.wav", text="কথা", duration_s=3, recording_id="r") | kwargs)
    )


_MANIFEST_FIELDS = dict(
    name="m", source="synthetic", split="train", domain="read", version="fixture"
)


def make_manifest(*segments, **kwargs):
    defaults = _MANIFEST_FIELDS | {"segments": segments or (make_segment(),)}
    return Manifest(**(defaults | kwargs))


def build_rows(rows, **kwargs):
    return build_manifest(**(_MANIFEST_FIELDS | dict(version="v", rows=rows) | kwargs))


def constant_duration(_path):
    return 3.0


def write_toml_registry(path, *records, as_of=None):
    header = f"as_of = {json.dumps(as_of)}\n" if as_of else ""
    entries = [
        "[[licenses]]\n"
        + "\n".join(f"{key} = {json.dumps(value)}" for key, value in record.items())
        for record in records
    ]
    path.write_text(header + "\n".join(entries), encoding="utf-8")


def write_jsonl(path, *records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def audit_manifest(name, split, segments):
    return make_manifest(*segments, name=name, source="openslr_slr53", split=split, version="v")


def audit_segment(segment_id, text, *, recording_id=None, audio_sha256=None):
    return Segment(
        id=segment_id,
        audio=f"{segment_id}.wav",
        text=text,
        duration_s=2.0,
        recording_id=recording_id or segment_id,
        audio_sha256=audio_sha256,
    )


def leakage_pair(
    *, train_text="প্রথম", evaluation_text="দ্বিতীয়", train_hash=None, evaluation_hash=None
):
    return (
        audit_manifest(
            "train", "train", [audit_segment("t1", train_text, audio_sha256=train_hash)]
        ),
        audit_manifest(
            "test", "test", [audit_segment("e1", evaluation_text, audio_sha256=evaluation_hash)]
        ),
    )


def write_common_voice_split(mounts, split, *, clip=None, text="কথা", value=0):
    clip = clip or f"{split}.wav"
    write_wav(mounts.common_voice / "clips" / clip, sample=value)
    (mounts.common_voice / f"{split}.tsv").write_text(
        f"client_id\tpath\tsentence\n{split}-speaker\t{clip}\t{text}\n",
        encoding="utf-8",
    )


def prepare_common_voice(mounts, output, **kwargs):
    return prepare_manifests(
        mounts,
        output,
        synthetic_registry("common_voice_bn"),
        loop_indexes={},
        **kwargs,
    )


@pytest.fixture
def common_voice(tmp_path):
    path = tmp_path / "test.tsv"
    return path, partial(
        from_common_voice_tsv,
        path,
        split="test",
        name="cv-bn-test",
        version="cv-26",
        duration_of=constant_duration,
    )


class TestCheckpointIdentity:
    @pytest.fixture
    def checkpoint(self, tmp_path):
        checkpoint = tmp_path / "checkpoint"
        (checkpoint / "weights").mkdir(parents=True)
        (checkpoint / "weights" / "model.bin").write_bytes(b"model weights")
        (checkpoint / "config.json").write_bytes(b"{}")
        return checkpoint

    def test_nested_files_have_complete_deterministic_identity(self, checkpoint):
        identity = hash_checkpoint(checkpoint)

        assert identity == {
            "source": "local",
            "path": str(checkpoint.resolve()),
            "files_sha256": {
                "config.json": hashlib.sha256(b"{}").hexdigest(),
                "weights/model.bin": hashlib.sha256(b"model weights").hexdigest(),
            },
        }
        assert list(identity["files_sha256"]) == ["config.json", "weights/model.bin"]
        assert hash_checkpoint(checkpoint) == identity

    @pytest.mark.parametrize("kind", ["missing", "file", "empty"])
    def test_invalid_checkpoint_directory_is_rejected(self, tmp_path, kind):
        checkpoint = tmp_path / "checkpoint"
        if kind == "file":
            checkpoint.write_bytes(b"weights")
        elif kind == "empty":
            checkpoint.mkdir()

        with pytest.raises(ValueError, match="empty|cannot hash"):
            hash_checkpoint(checkpoint)

    @pytest.mark.parametrize("kind", ["directory", "file"])
    def test_unreadable_checkpoint_contents_are_rejected(self, checkpoint, monkeypatch, kind):
        operation = "iterdir" if kind == "directory" else "open"
        target = checkpoint / "weights" if kind == "directory" else checkpoint / "weights/model.bin"
        original = getattr(Path, operation)

        def fail_read(path, *args, **kwargs):
            if path == target:
                raise PermissionError("cannot read model weights")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, operation, fail_read)
        with pytest.raises(ValueError, match="cannot hash") as error:
            hash_checkpoint(checkpoint)
        assert isinstance(error.value.__cause__, PermissionError)

    def test_symlinked_directory_is_rejected(self, checkpoint):
        (checkpoint / "linked-weights").symlink_to(checkpoint / "weights", target_is_directory=True)

        with pytest.raises(ValueError, match="symlinked directories"):
            hash_checkpoint(checkpoint)

    def test_broken_file_link_is_rejected(self, checkpoint):
        (checkpoint / "missing.bin").symlink_to(checkpoint / "nonexistent.bin")

        with pytest.raises(ValueError, match="unreadable file"):
            hash_checkpoint(checkpoint)

    def test_file_symlink_hashes_its_target_bytes(self, checkpoint, tmp_path):
        target = tmp_path / "shared-weights.bin"
        target.write_bytes(b"shared weights")
        (checkpoint / "shared.bin").symlink_to(target)

        identity = hash_checkpoint(checkpoint)

        assert (
            identity["files_sha256"]["shared.bin"] == hashlib.sha256(b"shared weights").hexdigest()
        )


class TestManifest:
    @pytest.fixture
    def manifest(self):
        return Manifest.from_text(TINY_MANIFEST)

    def test_tiny_manifest_loads(self, tmp_path):
        path = tmp_path / "tiny.jsonl"
        path.write_text(TINY_MANIFEST, encoding="utf-8")
        manifest = Manifest.from_jsonl(path)
        assert manifest.name == "tiny-longform-test"
        assert manifest.source == "bengali_loop"
        assert manifest.split == "test"
        assert len(manifest.segments) == 5
        assert manifest.recording_ids() == ["rec-loop-01", "rec-loop-02"]

    def test_total_hours_sums_durations(self, manifest):
        assert manifest.total_hours() == pytest.approx(16.8 / 3600.0)

    def test_jsonl_round_trip_is_lossless(self, tmp_path, manifest):
        output_path = tmp_path / "rt.jsonl"
        manifest.to_jsonl(output_path)
        restored_manifest = Manifest.from_jsonl(output_path)
        assert restored_manifest == manifest

    def test_manifest_snapshot_parses_the_bytes_used_for_its_hash(self, tmp_path, monkeypatch):
        path = tmp_path / "snapshot.jsonl"
        original = make_manifest(make_segment(text="original reference"))
        original.to_jsonl(path)
        raw = path.read_bytes()
        changed = raw.replace(b"original reference", b"different reference")
        parse_file = Manifest.from_jsonl

        def parse_other_revision(cls, source):
            path.write_bytes(changed)
            try:
                return parse_file(source)
            finally:
                path.write_bytes(raw)

        monkeypatch.setattr(Manifest, "from_jsonl", classmethod(parse_other_revision))
        manifest, digest = load_manifest(path)

        assert manifest == original
        assert digest == hashlib.sha256(raw).hexdigest()
        assert path.read_bytes() == raw

    def test_manifest_snapshot_hash_preserves_raw_line_endings(self, tmp_path):
        path = tmp_path / "snapshot.jsonl"
        original = make_manifest()
        original.to_jsonl(path)
        raw = path.read_bytes().replace(b"\n", b"\r\n")
        path.write_bytes(raw)

        manifest, digest = load_manifest(path)

        assert manifest == original
        assert digest == hashlib.sha256(raw).hexdigest()
        assert digest != hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()

    @pytest.mark.parametrize(
        "header,segment,error",
        [
            (None, asdict(make_segment(id="x", duration_s=1)), "header line"),
            (
                {"source": "openslr_slr53", "split": "test"},
                asdict(make_segment(id="s", duration_s=1)),
                "missing required field",
            ),
            (
                dict(name="m", source="openslr_slr53", split="test", domain="read", version="v"),
                dict(id="s", audio="a.wav", text="কথা"),
                "line 2",
            ),
        ],
    )
    def test_invalid_manifest_records(self, tmp_path, header, segment, error):
        records = [segment] if header is None else [{"manifest": header}, segment]
        path = write_jsonl(tmp_path / "bad.jsonl", *records)
        with pytest.raises(ValueError, match=error):
            Manifest.from_jsonl(path)

    @pytest.mark.parametrize(
        "changes,error",
        [
            ({"text": "   "}, "empty reference"),
            ({"duration_s": 0}, "non-positive duration"),
            ({"recording_id": ""}, "recording_id"),
            ({"start_s": -0.5}, "negative start_s"),
        ],
    )
    def test_segment_field_errors(self, changes, error):
        with pytest.raises(ValueError, match=error):
            make_segment(**(dict(id="s", duration_s=1) | changes))

    def test_duplicate_segment_ids_rejected(self):
        segment = make_segment()
        with pytest.raises(ValueError, match="duplicate segment id"):
            make_manifest(segment, segment)

    def test_unknown_split_rejected(self):
        with pytest.raises(ValueError, match="split"):
            make_manifest(split="validation")

    def test_empty_manifest_rejected(self):
        with pytest.raises(ValueError, match="no segments"):
            make_manifest(segments=())

    def test_records_with_joins_on_id(self, manifest):
        hypotheses = {
            segment.id: f"hypothesis {index}"
            for index, segment in reversed(list(enumerate(manifest.segments)))
        }

        assert manifest.records_with(hypotheses) == [
            (segment.recording_id, segment.text, f"hypothesis {index}")
            for index, segment in enumerate(manifest.segments)
        ]

    def test_records_with_missing_hypothesis_raises(self, manifest):
        hypotheses = {segment_id: "অনুমান" for segment_id in manifest.ids()[:-1]}
        with pytest.raises(KeyError, match="no hypothesis"):
            manifest.records_with(hypotheses)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("duration_s", True),
            ("duration_s", complex(3, 0)),
            ("duration_s", float("nan")),
            ("duration_s", float("inf")),
            pytest.param("duration_s", 10**1000, id="oversized-duration"),
            ("start_s", False),
            ("start_s", float("nan")),
            ("start_s", float("inf")),
            ("audio", "../a.wav"),
            ("audio", "/a.wav"),
            ("audio", "C:/a.wav"),
            ("audio", "a\\b.wav"),
            ("audio", "."),
            ("audio", "./"),
            ("audio", "./."),
            ("audio", "./a.wav"),
            ("id", None),
            ("recording_id", " "),
            ("audio_sha256", "short"),
            ("language", None),
        ],
    )
    def test_segment_rejects_invalid_contract(self, field, value):
        with pytest.raises(ValueError):
            make_segment(**{field: value})

    @pytest.mark.parametrize(
        "header,record",
        [
            (None, {}),
            ({"segments": []}, {}),
            ({}, {"start_secs": 90}),
        ],
    )
    def test_jsonl_shape_and_unknown_fields_are_rejected(self, tmp_path, header, record):
        meta = dict(name="m", source="synthetic", split="train", domain="read", version="v")
        if header is not None:
            meta.update(header)
        path = tmp_path / "bad.jsonl"
        path.write_text(
            json.dumps({"manifest": None if header is None else meta})
            + "\n\n"
            + json.dumps(asdict(make_segment()) | record)
            + "\n"
        )
        with pytest.raises(ValueError):
            Manifest.from_jsonl(path)

    @pytest.mark.parametrize("loader", ["file", "text", "snapshot"])
    def test_jsonl_duplicate_fields_and_physical_line_number(self, tmp_path, loader):
        path = tmp_path / "bad.jsonl"
        head = json.dumps({"manifest": _MANIFEST_FIELDS})
        text = head + '\n\n{"id":"a","id":"b"}\n'
        path.write_text(text)
        with pytest.raises(ValueError, match="line 3.*duplicate") as error:
            if loader == "text":
                Manifest.from_text(text, location=str(path))
            elif loader == "snapshot":
                load_manifest(path)
            else:
                Manifest.from_jsonl(path)
        assert str(path) in str(error.value)


class TestSourceRegistry:
    @pytest.fixture
    def registry(self) -> LicenseRegistry:
        return LicenseRegistry(
            [
                *synthetic_registry("fixture_active"),
                *synthetic_registry("fixture_excluded", status="excluded"),
            ]
        )

    def test_registry_indexes_supplied_sources(self, registry):
        for source in ("fixture_active", "fixture_excluded"):
            assert source in registry
            assert isinstance(registry[source], DatasetLicense)

    def test_toml_registry_preserves_nullable_metadata_and_json_compatibility(
        self, tmp_path, toml_license
    ):
        toml_path = tmp_path / "licenses.toml"
        write_toml_registry(
            toml_path,
            toml_license,
            toml_license | {"id": "known_hours", "hours": 1.5},
            as_of="fixture-date",
        )
        registry = LicenseRegistry.load(toml_path)
        assert registry.as_of == "fixture-date"
        assert len(registry) == 2
        assert registry["synthetic"].hours is None
        assert registry["synthetic"].redistributable_audio is None
        assert registry["synthetic"].attribution_required is None
        assert registry["synthetic"].share_alike is None
        assert registry["known_hours"].hours == 1.5
        path = tmp_path / "licenses.json"
        path.write_text(
            json.dumps(
                {"as_of": registry.as_of, "licenses": [asdict(record) for record in registry]}
            ),
            encoding="utf-8",
        )
        restored = LicenseRegistry.load(path)
        assert list(restored) == list(registry)
        assert restored.as_of == registry.as_of
        assert restored.render_table() == registry.render_table()

    def test_active_sources_exclude_blocked_sources(self, registry):
        assert registry.sources("active") == ["fixture_active"]
        assert registry["fixture_excluded"].status == "excluded"

    def test_require_unknown_source_is_actionable(self, registry):
        with pytest.raises(KeyError, match="not in the license registry"):
            registry.require("no_such_corpus")

    def test_render_table_lists_sources_and_licenses(self, registry):
        table = registry.render_table()
        assert "fixture-only" in table
        assert "Synthetic fixture" in table
        assert table.index("active") < table.index("excluded")

    def test_render_table_escapes_dynamic_cell_text(self):
        original = next(iter(synthetic_registry("synthetic")))
        license = replace(
            original,
            name="Dataset|extra\nrow",
            domain="read\rtest",
            license="license|name\ntext",
            url="",
        )

        table = LicenseRegistry([license]).render_table()

        assert len(table.splitlines()) == 3
        assert "Dataset\\|extra row" in table
        assert "read test" in table
        assert "license\\|name text" in table

    def test_excluded_source_cannot_be_referenced(self, registry):
        manifest = make_manifest(source="fixture_excluded", split="test")
        with pytest.raises(ValueError, match="excluded source"):
            manifest.validate_against(registry)

    def test_eval_only_source_cannot_be_training_data(self):
        registry = synthetic_registry("fixture_eval", status="eval-only")
        with pytest.raises(ValueError, match="eval-only"):
            make_manifest(source="fixture_eval").validate_against(registry)

    def test_active_source_validates_clean(self):
        manifest = Manifest.from_text(TINY_MANIFEST)
        manifest.validate_against(synthetic_registry("bengali_loop"))

    @pytest.mark.parametrize(
        "suffix,changes",
        [
            ("json", {"hours": float("nan")}),
            ("json", {"hours": True}),
            ("json", {"hours": -1}),
            ("json", {"redistributable_audio": "false"}),
            ("json", {"share_alike": 0}),
            ("toml", {"unexpected": "field"}),
            ("toml", {"hours": True}),
            ("toml", {"hours": -1}),
            ("toml", {"redistributable_audio": "false"}),
            ("toml", {"attribution_required": 0}),
            ("toml", {"share_alike": "unknown"}),
        ],
    )
    def test_registry_rejects_invalid_fields(self, tmp_path, toml_license, suffix, changes):
        path = tmp_path / f"licenses.{suffix}"
        if suffix == "toml":
            write_toml_registry(path, toml_license | changes)
        else:
            record = asdict(next(iter(synthetic_registry("synthetic")))) | changes
            path.write_text(json.dumps({"licenses": [record]}))
        with pytest.raises(ValueError):
            LicenseRegistry.load(path)

    @pytest.fixture
    def toml_license(self):
        return {
            key: value
            for key, value in asdict(next(iter(synthetic_registry("synthetic")))).items()
            if key not in {"hours", "redistributable_audio", "attribution_required", "share_alike"}
        }

    @pytest.mark.parametrize(
        "field",
        [
            "id",
            "name",
            "url",
            "license",
            "access",
            "status",
            "domain",
            "version",
            "notes",
            "confidence",
        ],
    )
    def test_toml_registry_requires_nonnullable_fields(self, tmp_path, toml_license, field):
        del toml_license[field]
        path = tmp_path / "licenses.toml"
        write_toml_registry(path, toml_license)
        with pytest.raises(ValueError, match="license entry 0 needs"):
            LicenseRegistry.load(path)

    @pytest.mark.parametrize(
        "field", ["hours", "redistributable_audio", "attribution_required", "share_alike"]
    )
    def test_json_registry_requires_explicit_unknown_fields(self, tmp_path, field):
        record = asdict(next(iter(synthetic_registry("synthetic"))))
        del record[field]
        path = tmp_path / "licenses.json"
        path.write_text(json.dumps({"licenses": [record]}), encoding="utf-8")
        with pytest.raises(ValueError, match="license entry 0 needs"):
            LicenseRegistry.load(path)

    def test_toml_registry_rejects_duplicate_fields(self, tmp_path, toml_license):
        path = tmp_path / "licenses.toml"
        write_toml_registry(path, toml_license)
        with path.open("a", encoding="utf-8") as stream:
            stream.write('\nid = "duplicate"\n')
        with pytest.raises(ValueError):
            LicenseRegistry.load(path)

    def test_render_table_marks_missing_redistribution_permission_unknown(self):
        license_record = next(iter(synthetic_registry("synthetic")))
        assert (
            "unknown"
            in LicenseRegistry([replace(license_record, redistributable_audio=None)]).render_table()
        )


class TestBuilders:
    def test_build_manifest_from_rows_uses_injected_duration(self):
        rows = [
            {"id": "a", "audio": "a.wav", "text": "আমি ভালো"},
            {"id": "b", "audio": "b.wav", "text": "তুমি কেমন", "duration_s": 5.0},
        ]
        manifest = build_rows(rows, source="openslr_slr53", duration_of=constant_duration)
        assert len(manifest.segments) == 2
        assert manifest.segments[0].duration_s == 3.0
        assert manifest.segments[1].duration_s == 5.0
        assert manifest.segments[0].recording_id == "a"

    def test_build_manifest_skips_empty_text_rows(self):
        rows = [
            {"id": "a", "audio": "a.wav", "text": "কথা"},
            {"id": "b", "audio": "b.wav", "text": "   "},
        ]
        manifest = build_rows(rows, source="openslr_slr53", duration_of=constant_duration)
        assert [s.id for s in manifest.segments] == ["a"]

    def test_built_manifest_accepts_registered_source(self):
        manifest = build_rows(
            [{"id": "a", "audio": "a.wav", "text": "কথা"}],
            source="common_voice_bn",
            split="test",
            duration_of=constant_duration,
        )
        manifest.validate_against(synthetic_registry("common_voice_bn"))

    @pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
    def test_from_common_voice_tsv(self, common_voice, encoding):
        tsv, load = common_voice
        tsv.write_text(
            "client_id\tpath\tsentence\n"
            "spk1\tcv_bn_1.mp3\tআমি ভাত খাই\n"
            "spk1\tcv_bn_2.mp3\tতুমি কি করছ\n"
            "spk2\tcv_bn_3.mp3\t\n",
            encoding=encoding,
        )
        manifest = load()
        assert len(manifest.segments) == 2
        assert manifest.segments[0].audio == "clips/cv_bn_1.mp3"
        assert manifest.segments[0].recording_id == "spk1"
        assert manifest.recording_ids() == ["spk1"]
        assert manifest.source == "common_voice_bn"

    @pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
    def test_from_common_voice_uses_durations_tsv(self, tmp_path, common_voice, encoding):
        path, load = common_voice
        path.write_text("client_id\tpath\tsentence\nspk1\tclip.mp3\tকথা\n", encoding="utf-8")
        (tmp_path / "clip_durations.tsv").write_text(
            "clip\tduration[ms]\nclip.mp3\t4200\n", encoding=encoding
        )
        manifest = load(durations_tsv=tmp_path / "clip_durations.tsv")
        assert manifest.segments[0].duration_s == pytest.approx(4.2)

    def test_collapse_to_recordings_joins_segments_per_recording(self):
        first = make_segment(audio="r1.wav", recording_id="r1", duration_s=2)
        segments = (
            replace(first, id="r1-1", text="দ্বিতীয়", start_s=4),
            replace(first, id="r1-0", text="প্রথম", start_s=0),
            make_segment(id="r2-0", audio="r2.wav", text="আলাদা", recording_id="r2"),
        )
        collapsed = collapse_to_recordings(make_manifest(*segments))
        assert len(collapsed.segments) == 2
        r1 = next(s for s in collapsed.segments if s.recording_id == "r1")
        assert r1.text == "প্রথম দ্বিতীয়"
        assert r1.duration_s == 6.0
        assert r1.id == "r1"

    def test_from_fleurs_tsv_derives_duration_from_num_samples(self, tmp_path):
        tsv = tmp_path / "test.tsv"
        tsv.write_text(
            "1\t1.wav\tকাঁচা\tকাঁচা লেখা\t48000\tFEMALE\n2\t2.wav\t\t\t16000\tMALE\n",
            encoding="utf-8",
        )
        manifest = from_fleurs_tsv(tsv, split="test", name="fleurs-bn-test")
        assert len(manifest.segments) == 1
        assert manifest.segments[0].duration_s == pytest.approx(48000 / 16000)
        assert manifest.segments[0].text == "কাঁচা লেখা"
        assert manifest.source == "fleurs_bn"

    @pytest.mark.parametrize(
        "field,value", [("text", None), ("id", None), ("duration_s", True), ("start_s", True)]
    )
    def test_builder_validates_before_coercion(self, field, value):
        row = dict(id="a", audio="a.wav", text="কথা", duration_s=3)
        row[field] = value
        with pytest.raises(ValueError):
            build_rows([row])

    def test_builder_preserves_hash_language_and_requires_aligned_duration(self):
        row = dict(
            id="a", audio="a.wav", text="কথা", duration_s=3, language="bn-en", audio_sha256="A" * 64
        )
        result = build_rows([row])
        assert result.segments[0].language == "bn-en"
        assert result.segments[0].audio_sha256 == "a" * 64
        row.pop("duration_s")
        row["start_s"] = 5
        with pytest.raises(ValueError, match="aligned span"):
            build_rows([row], duration_of=lambda p: 10)

    def test_recording_collapse_validates_duration_and_segment_consistency(self):
        manifest = make_manifest(
            make_segment(id="a", start_s=0, duration_s=2),
            make_segment(id="b", start_s=4, duration_s=2),
        )
        assert collapse_to_recordings(manifest).segments[0].duration_s == 6
        assert collapse_to_recordings(manifest, duration_of=lambda p: 7).segments[0].duration_s == 7
        with pytest.raises(ValueError, match="cover"):
            collapse_to_recordings(manifest, duration_of=lambda p: 5)
        for changed in (
            dict(audio="other.wav"),
            dict(start_s=None),
            dict(start_s=1),
            dict(language="en"),
            dict(audio_sha256="a" * 64),
        ):
            bad = replace(
                manifest, segments=(manifest.segments[0], replace(manifest.segments[1], **changed))
            )
            with pytest.raises(ValueError):
                collapse_to_recordings(bad)

    def test_loop_published_recording_jsonl_is_not_training(self, tmp_path):
        path = write_jsonl(
            tmp_path / "loop.jsonl",
            dict(id="recording", audio_filepath="a.wav", duration=3600, text="সম্পূর্ণ রেকর্ডিং"),
        )
        manifest = from_loop_jsonl(path, name="loop", version="fixture", audio_prefix="loop-mount")
        assert (
            manifest.segments[0].audio == "loop-mount/a.wav"
            and manifest.segments[0].start_s is None
        )
        with pytest.raises(ValueError, match="aligned"):
            from_loop_jsonl(path, name="loop", version="fixture", split="train")

    def test_kaldi_joins_preserve_full_text_times_and_speaker(self, tmp_path):
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
            name="mucs",
            split="train",
            audio_root=tmp_path,
            segments_path=spans,
            utt2spk_path=speakers,
        )
        manifest = from_mucs_kaldi(text, wav, **kwargs)
        assert manifest.segments[0].text == "বাংলা and English words"
        assert manifest.segments[1].start_s == 4 and manifest.segments[1].speaker == "speaker2"
        assert manifest.segments[0].recording_id == "r"
        wav.write_text("r sox source.wav -t wav - |\n")
        with pytest.raises(ValueError, match="commands/pipes"):
            from_mucs_kaldi(text, wav, **kwargs)
        wav.write_text("r audio space.wav\n")
        speakers.write_text("u1 speaker1\n")
        with pytest.raises(ValueError, match="utt2spk"):
            from_mucs_kaldi(text, wav, **kwargs)

    def test_slr53_published_three_column_index_and_file_mapping(self, tmp_path):
        local = tmp_path / "slr"
        (local / "part-0").mkdir(parents=True)
        (local / "part-0/file0.wav").write_bytes(b"fixture")
        index = local / "utt_spk_text.tsv"
        options = dict(
            name="slr",
            split="train",
            audio_root=tmp_path,
            dataset_root=local,
            duration_of=constant_duration,
        )
        index.write_text("file0\tuser0\tবাংলা বাক্য\n")
        manifest = from_slr53_tsv(index, **options)
        assert manifest.segments[0].audio == "slr/part-0/file0.wav"
        assert manifest.segments[0].text == "বাংলা বাক্য"
        assert manifest.segments[0].speaker == manifest.segments[0].recording_id == "user0"
        for content, error in [
            ("file0\tবাংলা বাক্য\n", "FileID, UserID"),
            ("missing\tuser0\tবাংলা বাক্য\n", "missing WAV"),
        ]:
            index.write_text(content)
            with pytest.raises(ValueError, match=error):
                from_slr53_tsv(index, **options)


class TestAudioPaths:
    def test_real_wav_whole_file_and_explicit_crop(self, tmp_path):
        path = write_wav(tmp_path / "a.wav", frames=4 * 16_000)
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
            make_manifest(make_segment(duration_s=4, audio_sha256=file_sha256(path))), tmp_path
        )
        with pytest.raises(ValueError, match="does not match file bytes"):
            validate_manifest_audio(
                make_manifest(make_segment(duration_s=4, audio_sha256="f" * 64)), tmp_path
            )

    def test_paths_preflight_all_sources_and_aliases(self, tmp_path):
        for directory in ("one", "two"):
            (tmp_path / directory).mkdir()
            (tmp_path / directory / "a.wav").write_bytes(b"fixture")
        result = validate_audio_paths(
            make_manifest(make_segment(audio="one/a.wav"), make_segment(id="b", audio="two/a.wav")),
            tmp_path,
        )
        assert result == (tmp_path / "one/a.wav", tmp_path / "two/a.wav")
        (tmp_path / "cycle").symlink_to("cycle")
        with pytest.raises(ValueError, match="resolve"):
            validate_audio_paths(make_manifest(make_segment(audio="cycle")), tmp_path)
        (tmp_path / "escape").symlink_to(tmp_path.parent)
        with pytest.raises(ValueError, match="escapes"):
            validate_audio_paths(make_manifest(make_segment(audio="escape/a.wav")), tmp_path)


class TestLeakageAudit:
    @pytest.mark.parametrize("check", [("id",), ("audio",)])
    def test_nontext_checks_skip_normalization(self, monkeypatch, check):
        normalized = []
        monkeypatch.setattr(leakage, "normalize", lambda text: normalized.append(text) or text)
        train, evaluation = leakage_pair()
        report = audit_leakage(train, evaluation, check=check)
        assert report.checked_kinds == check
        assert normalized == []

    def test_hardlinks_are_hard_leakage_without_hashes(self, tmp_path):
        (tmp_path / "a.wav").write_bytes(b"fixture")
        os.link(tmp_path / "a.wav", tmp_path / "b.wav")
        train = make_manifest(make_segment(text="প্রথম"))
        evaluation = make_manifest(
            make_segment(id="b", audio="b.wav", text="দ্বিতীয়"), name="dev", split="dev"
        )
        with pytest.raises(ValueError, match="hard train/evaluation"):
            require_no_leakage(train, evaluation, audio_root=tmp_path)

    def test_ids_are_source_qualified_and_text_review_is_exact(self):
        train = make_manifest(make_segment(text="বাক্য", audio_sha256="a" * 64))
        evaluation = make_manifest(
            make_segment(text="অন্য", audio_sha256="b" * 64), name="dev", source="other", split="dev"
        )
        assert require_no_leakage(train, evaluation).is_clean
        evaluation = replace(
            evaluation, segments=(make_segment(id="b", text="বাক্য।", audio_sha256="b" * 64),)
        )
        with pytest.raises(ValueError, match="exact-key"):
            require_no_leakage(train, evaluation)
        report = require_no_leakage(
            train, evaluation, text_decisions={"বাক্য": "common synthetic sentence"}
        )
        assert report.is_approved and report.reviewed_text[0].reason
        with pytest.raises(ValueError, match="exact-key"):
            require_no_leakage(train, evaluation, text_decisions={"typo": "reviewed"})

    def test_hash_coverage_does_not_hide_missing_eval(self):
        train = make_manifest(
            *(
                make_segment(id=str(i), audio=f"{i}.wav", text=f"train{i}", audio_sha256="a" * 64)
                for i in range(1000)
            )
        )
        evaluation = make_manifest(
            make_segment(id="eval", text="evaluation"), name="dev", split="dev"
        )
        report = audit_leakage(train, evaluation)
        assert report.train_hashed == 1000 and report.eval_hashed == 0
        assert "eval 0/1" in report.summary() and "100%" not in report.summary()
        assert not report.is_complete and not report.is_approved
        assert report.summary().startswith("INCOMPLETE:")

    @pytest.mark.parametrize("side", ["train", "evaluation"])
    def test_leakage_boundary_rejects_detached_records(self, side):
        train = make_manifest()
        evaluation = make_manifest(make_segment(id="dev"), name="dev", split="dev")
        with pytest.raises(ValueError, match="Manifest records"):
            require_no_leakage(
                [None] if side == "train" else train, None if side == "evaluation" else evaluation
            )

    def test_clean_when_no_overlap(self):
        train = audit_manifest(
            "train",
            "train",
            [
                audit_segment("t1", "আজকে ভালো দিন", audio_sha256="a" * 64),
                audit_segment("t2", "কেমন আছেন", audio_sha256="b" * 64),
            ],
        )
        test = audit_manifest(
            "test", "test", [audit_segment("e1", "নতুন বাক্য এখানে", audio_sha256="c" * 64)]
        )
        report = audit_leakage(train, test)
        assert report.is_complete and report.is_clean and report.is_approved
        assert "CLEAN" in report.summary()

    def test_detects_shared_segment_id(self):
        train = audit_manifest("train", "train", [audit_segment("shared", "কথা এক")])
        test = audit_manifest("test", "test", [audit_segment("shared", "সম্পূর্ণ ভিন্ন বাক্য")])
        report = audit_leakage(train, test, check=["id"])
        ids = report.by_kind("id")
        assert len(ids) == 1 and ids[0].key == "openslr_slr53:shared"
        assert not report.is_clean

    def test_detects_text_overlap_under_frozen_normalizer(self):
        train, test = leakage_pair(train_text="আমি ভাত খাই।", evaluation_text="আমি ভাত খাই")
        report = audit_leakage(train, test, check=["text"])
        text_overlaps = report.by_kind("text")
        assert len(text_overlaps) == 1
        assert "review each" in report.summary()

    def test_detects_audio_checksum_overlap(self):
        train, test = leakage_pair(
            train_text="কথা এক",
            train_hash="d" * 64,
            evaluation_text="কথা দুই",
            evaluation_hash="d" * 64,
        )
        report = audit_leakage(train, test, check=["audio"])
        assert len(report.by_kind("audio")) == 1

    def test_audio_coverage_reported_when_checksums_partial(self):
        train, test = leakage_pair(train_text="কথা", train_hash="a" * 64, evaluation_text="ভিন্ন")
        report = audit_leakage(train, test, check=["audio"])
        assert report.audio_coverage == 0.5
        assert not report.is_complete and not report.is_approved
        assert report.checked_kinds == ("audio",)
        assert report.summary().startswith("INCOMPLETE:")

    def test_missing_hashes_cannot_approve_an_audit(self):
        train, test = leakage_pair()
        report = audit_leakage(train, test)
        assert not report.is_clean and not report.is_approved
        assert report.audio_coverage == 0
        assert report.summary().startswith("INCOMPLETE:")
        with pytest.raises(ValueError, match="incomplete train/evaluation leakage audit"):
            require_no_leakage(train, test)

    @pytest.mark.parametrize("check", [("id",), ("text",), ("audio",), ("id", "text")])
    def test_subset_checks_cannot_approve_an_audit(self, check):
        train, test = leakage_pair(train_hash="a" * 64, evaluation_hash="b" * 64)
        report = audit_leakage(train, test, check=check)
        assert report.audio_coverage == 1
        assert report.checked_kinds == check
        assert not report.is_complete and not report.is_clean and not report.is_approved
        assert report.summary().startswith("INCOMPLETE:")

    def test_copied_audio_without_hashes_is_hard_leakage(self, tmp_path):
        (tmp_path / "t1.wav").write_bytes(b"same recording bytes")
        shutil.copyfile(tmp_path / "t1.wav", tmp_path / "e1.wav")
        train, test = leakage_pair()
        report = audit_leakage(train, test, audio_root=tmp_path)
        assert report.is_complete and not report.is_approved
        assert report.audio_coverage == 1
        assert report.train_hashed == report.eval_hashed == 1
        assert report.by_kind("audio")[0].key == hashlib.sha256(b"same recording bytes").hexdigest()
        assert not report.by_kind("file")
        with pytest.raises(ValueError, match="hard train/evaluation overlap"):
            require_no_leakage(train, test, audio_root=tmp_path)
        assert train.segments[0].audio_sha256 is None

    def test_distinct_files_without_hashes_can_pass_a_complete_audit(self, tmp_path):
        for name in ("t1", "e1"):
            (tmp_path / f"{name}.wav").write_bytes(name.encode())
        train, test = leakage_pair()
        report = require_no_leakage(train, test, audio_root=tmp_path)
        assert report.is_complete and report.is_clean and report.is_approved
        assert report.audio_coverage == 1
        assert report.checked_kinds == ("id", "text", "audio", "file")

    def test_declared_hashes_are_verified_against_file_bytes(self, tmp_path):
        (tmp_path / "t1.wav").write_bytes(b"train bytes")
        (tmp_path / "e1.wav").write_bytes(b"eval bytes")
        train, test = leakage_pair(train_hash="a" * 64)
        with pytest.raises(ValueError, match="audio_sha256 does not match file bytes"):
            audit_leakage(train, test, audio_root=tmp_path)

    def test_hash_cache_covers_shared_files_and_hardlinks_only_within_one_audit(
        self, tmp_path, monkeypatch
    ):
        (tmp_path / "t1.wav").write_bytes(b"one file")
        os.link(tmp_path / "t1.wav", tmp_path / "e1.wav")
        train = audit_manifest(
            "train",
            "train",
            [audit_segment("t1", "প্রথম"), replace(audit_segment("t2", "তৃতীয়"), audio="t1.wav")],
        )
        test = audit_manifest("test", "test", [audit_segment("e1", "দ্বিতীয়")])
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

    def test_audit_against_a_mix_of_train_manifests(self):
        a = audit_manifest("slr53", "train", [audit_segment("a1", "প্রথম")])
        b = audit_manifest("cv", "train", [audit_segment("b1", "দ্বিতীয় বাক্য")])
        test = audit_manifest("test", "test", [audit_segment("e1", "দ্বিতীয় বাক্য")])
        report = audit_leakage([a, b], test, check=["text"])
        assert not report.is_clean
        assert report.train_names == ("slr53", "cv")

    def test_empty_train_list_rejected(self):
        test = audit_manifest("test", "test", [audit_segment("e1", "কিছু")])
        with pytest.raises(ValueError, match="at least one"):
            audit_leakage([], test)


class TestPreparation:
    @pytest.fixture
    def mounts(self, tmp_path):
        audio = tmp_path / "audio"
        audio.mkdir()
        return DatasetMounts(
            audio_root=audio,
            common_voice=audio / "cv",
            fleurs=audio / "fleurs",
            slr53=audio / "slr53",
            mucs=audio / "mucs",
            loop=audio / "loop",
        )

    @pytest.mark.parametrize("split", ["train", "dev", "test"])
    def test_common_voice_splits_are_independent(self, tmp_path, mounts, split):
        write_common_voice_split(mounts, split)

        result = prepare_common_voice(mounts, tmp_path / "manifests")

        assert [manifest.name for manifest in result.manifests] == [f"cv-bn-{split}"]
        assert result.paths == (tmp_path / "manifests" / f"cv-bn-{split}.manifest.jsonl",)
        assert Manifest.from_jsonl(result.paths[0]) == result.manifests[0]
        assert not result.audits
        for missing in {"train", "dev", "test"} - {split}:
            assert f"Common Voice {missing}: local index missing" in result.pending

    @pytest.mark.parametrize(
        "dataset,name,source,version,audio",
        [
            ("common_voice", "cv-bn-test", "common_voice_bn", "cv-corpus-26.0", "clips/test.wav"),
            ("fleurs", "fleurs-bn-test", "fleurs_bn", "fleurs", "audio/recording.wav"),
            (
                "loop",
                "bengali-loop-test",
                "bengali_loop",
                "operator-recorded-release",
                "recording.wav",
            ),
        ],
    )
    def test_mount_can_equal_audio_root(
        self, tmp_path, mounts, dataset, name, source, version, audio
    ):
        mounts = replace(mounts, **{dataset: mounts.audio_root})
        indexes = {}
        if dataset == "common_voice":
            write_common_voice_split(mounts, "test")
        elif dataset == "fleurs":
            write_wav(mounts.fleurs / audio)
            (mounts.fleurs / "test.tsv").write_text(
                "1\trecording.wav\tকথা\tকথা\t16000\tFEMALE\n", encoding="utf-8"
            )
        else:
            write_wav(mounts.loop / audio)
            indexes[name] = write_jsonl(
                mounts.loop / "recordings.jsonl",
                dict(id="recording", audio_filepath=audio, duration=1, text="কথা"),
            )
        result = prepare_manifests(
            mounts,
            tmp_path / "manifests",
            synthetic_registry(source, status="eval-only" if dataset == "loop" else "active"),
            loop_indexes=indexes,
        )
        (manifest,) = result.manifests
        assert (manifest.name, manifest.source, manifest.split, manifest.version) == (
            name,
            source,
            "test",
            version,
        )
        assert manifest.segments[0].audio == audio
        assert manifest.segments[0].duration_s == 1
        if dataset == "loop":
            assert manifest.segments[0].start_s is None
        assert Manifest.from_jsonl(result.paths[0]) == manifest

    @pytest.mark.parametrize("dataset", ["common_voice", "fleurs", "loop"])
    def test_mount_outside_audio_root_is_rejected(self, tmp_path, mounts, dataset):
        mounts = replace(mounts, **{dataset: tmp_path / "outside"})

        with pytest.raises(ValueError, match="mount must be under audio_root"):
            prepare_manifests(
                mounts, tmp_path / "manifests", synthetic_registry("source"), loop_indexes={}
            )

    def test_repeated_preparation_replaces_current_split_without_duplicate_state(
        self, tmp_path, mounts
    ):
        write_common_voice_split(mounts, "train", text="প্রথম", value=1)
        write_common_voice_split(mounts, "test", text="দ্বিতীয়", value=2)
        output = tmp_path / "manifests"
        first = prepare_common_voice(mounts, output)
        assert len(first.audits) == 1
        assert first.audits[0].is_clean

        repeated = prepare_common_voice(mounts, output)
        assert repeated == first

        (mounts.common_voice / "train.tsv").unlink()
        write_common_voice_split(mounts, "test", clip="replacement.wav", text="নতুন", value=3)
        updated = prepare_common_voice(mounts, output)

        assert [manifest.name for manifest in updated.manifests] == ["cv-bn-test"]
        assert updated.manifests[0].references() == ["নতুন"]
        assert updated.manifests[0].segments[0].audio.endswith("replacement.wav")
        assert not updated.audits
        assert "Common Voice train: local index missing" in updated.pending
        assert Manifest.from_jsonl(updated.paths[0]) == updated.manifests[0]
        assert len(first.manifests) == 2

    @pytest.mark.parametrize("hash_audio", [False, True])
    def test_hash_setting_is_preserved_and_audits_read_actual_bytes(
        self, tmp_path, mounts, hash_audio
    ):
        write_common_voice_split(mounts, "train", text="প্রথম", value=1)
        write_common_voice_split(mounts, "test", text="দ্বিতীয়", value=2)

        result = prepare_common_voice(mounts, tmp_path / "manifests", hash_audio=hash_audio)

        assert result.audits[0].is_clean
        for manifest in result.manifests:
            segment = manifest.segments[0]
            expected = file_sha256(mounts.audio_root / segment.audio) if hash_audio else None
            assert segment.audio_sha256 == expected

    @pytest.mark.parametrize("hash_audio", [False, True])
    def test_loop_hash_setting(self, tmp_path, mounts, hash_audio):
        path = write_wav(mounts.loop / "recording.wav")
        index = write_jsonl(
            mounts.loop / "recordings.jsonl",
            dict(id="recording", audio_filepath=path.name, duration=1, text="কথা"),
        )
        result = prepare_manifests(
            mounts,
            tmp_path / "manifests",
            synthetic_registry("bengali_loop"),
            loop_indexes={"loop": index},
            hash_audio=hash_audio,
        )
        segment = result.manifests[0].segments[0]
        assert segment.audio_sha256 == (file_sha256(path) if hash_audio else None)

    def test_all_sources_validate_before_any_manifest_is_replaced(self, tmp_path, mounts):
        write_common_voice_split(mounts, "train")
        write_common_voice_split(mounts, "test")
        (mounts.common_voice / "clip_durations.tsv").write_text(
            "clip\tduration[ms]\ntrain.wav\t1000\ntest.wav\t5000\n", encoding="utf-8"
        )
        output = tmp_path / "manifests"
        output.mkdir()
        previous = output / "cv-bn-train.manifest.jsonl"
        previous.write_text("previous result", encoding="utf-8")

        with pytest.raises(ValueError, match="duration metadata"):
            prepare_common_voice(mounts, output)

        assert previous.read_text(encoding="utf-8") == "previous result"
        assert not (output / "cv-bn-test.manifest.jsonl").exists()

    def test_unregistered_source_is_rejected_before_outputs(self, tmp_path, mounts):
        write_common_voice_split(mounts, "test")
        output = tmp_path / "manifests"

        with pytest.raises(KeyError, match="supply its license record"):
            prepare_manifests(mounts, output, synthetic_registry("other"), loop_indexes={})

        assert not output.exists()

    def test_external_registry_destination_is_protected(self, tmp_path, mounts):
        write_common_voice_split(mounts, "test")
        output = tmp_path / "manifests"
        output.mkdir()
        registry_path = output / "cv-bn-test.manifest.jsonl"
        registry_path.write_text("external registry", encoding="utf-8")

        with pytest.raises(ValueError, match="overwrite input"):
            prepare_common_voice(mounts, output, protected_paths=(registry_path,))

        assert registry_path.read_text(encoding="utf-8") == "external registry"

    def test_no_available_indexes_returns_pending_without_creating_outputs(self, tmp_path, mounts):
        result = prepare_common_voice(mounts, tmp_path / "manifests")

        assert not result.manifests
        assert not result.paths
        assert not result.audits
        assert "Common Voice test: local index missing" in result.pending
        assert not (tmp_path / "manifests").exists()

    def test_preparation_preserves_dataset_specific_paths_and_annotations(self, tmp_path, mounts):
        write_common_voice_split(mounts, "test", text="প্রথম", value=1)
        write_wav(mounts.fleurs / "audio" / "fleurs.wav", sample=2)
        (mounts.fleurs / "test.tsv").write_text(
            "1\tfleurs.wav\tদ্বিতীয়\tদ্বিতীয়\t16000\tFEMALE\n", encoding="utf-8"
        )
        write_wav(mounts.slr53 / "part" / "slr.wav", sample=3)
        (mounts.slr53 / "utt_spk_text.tsv").write_text("slr\tslr-speaker\tতৃতীয়\n", encoding="utf-8")
        for split, value, text in (("train", 4, "চতুর্থ"), ("test", 5, "পঞ্চম")):
            local = mounts.mucs / split
            utterance = f"utterance-{split}"
            write_wav(local / "audio.wav", sample=value)
            (local / "text").write_text(f"{utterance} {text}\n", encoding="utf-8")
            (local / "wav.scp").write_text("recording audio.wav\n", encoding="utf-8")
            (local / "segments").write_text(f"{utterance} recording 0 1\n", encoding="utf-8")
            (local / "utt2spk").write_text(f"{utterance} mucs-speaker\n", encoding="utf-8")
        write_wav(mounts.loop / "loop.wav", sample=6)
        loop_index = write_jsonl(
            mounts.loop / "recordings.jsonl",
            dict(id="loop", audio_filepath="loop.wav", duration=1, text="ষষ্ঠ"),
        )

        result = prepare_manifests(
            mounts,
            tmp_path / "manifests",
            synthetic_registry(
                "common_voice_bn",
                "fleurs_bn",
                "openslr_slr53",
                "mucs_slr104",
                "bengali_loop",
            ),
            loop_indexes={"bengali-loop-test": loop_index},
        )

        manifests = {manifest.name: manifest for manifest in result.manifests}
        assert set(manifests) == {
            "cv-bn-test",
            "fleurs-bn-test",
            "slr53-train",
            "mucs-cs-train",
            "mucs-cs-test",
            "bengali-loop-test",
        }
        assert len(result.audits) == 4
        assert all(audit.is_clean for audit in result.audits)
        slr = manifests["slr53-train"].segments[0]
        assert slr.audio == "slr53/part/slr.wav"
        assert slr.recording_id == slr.speaker == "slr-speaker"
        mucs = manifests["mucs-cs-train"].segments[0]
        assert mucs.audio == "mucs/train/audio.wav"
        assert mucs.recording_id == "recording"
        assert mucs.speaker == "mucs-speaker"
        assert mucs.start_s == 0
        loop = manifests["bengali-loop-test"].segments[0]
        assert loop.audio == "loop/loop.wav"
        assert loop.start_s is None
        for path, manifest in zip(result.paths, result.manifests, strict=True):
            assert Manifest.from_jsonl(path) == manifest


class TestDelimitedInputValidation:
    @pytest.mark.parametrize("source", ["common_voice", "durations", "fleurs", "slr53"])
    def test_unclosed_quotes_are_rejected_with_file_and_line(self, tmp_path, source):
        path = tmp_path / "input.tsv"
        options = dict(name="fixture", split="train", duration_of=constant_duration)
        if source == "fleurs":
            path.write_text('1\tclip.wav\traw\t"unclosed transcript\t16000\tFEMALE\n')
            load = partial(from_fleurs_tsv, path, name="fixture", split="train")
            line = 1
        elif source == "slr53":
            path.write_text('clip\tspeaker\t"unclosed transcript\n')
            load = partial(
                from_slr53_tsv,
                path,
                audio_root=tmp_path,
                audio_paths={"clip": "clip.wav"},
                **options,
            )
            line = 1
        else:
            path.write_text('path\tsentence\nclip.wav\t"unclosed transcript\n')
            durations = None
            if source == "durations":
                path.write_text("path\tsentence\nclip.wav\ttranscript\n")
                durations = tmp_path / "durations.tsv"
                durations.write_text('clip\tduration[ms]\nclip.wav\t"1000\n')
            load = partial(
                from_common_voice_tsv, path, version="fixture", durations_tsv=durations, **options
            )
            path = durations or path
            line = 2
        with pytest.raises(ValueError, match=f"{path}:.*line {line}:.*unexpected end of data"):
            load()

    def test_quoted_multiline_tsv_preserves_text(self, common_voice):
        path, load = common_voice
        path.write_text('path\tsentence\nclip.wav\t"first\nsecond"\n', encoding="utf-8")
        manifest = load()
        assert manifest.references() == ["first\nsecond"]

    def test_tsv_width_errors_use_physical_line_after_multiline_field(self, common_voice):
        path, load = common_voice
        path.write_text('path\tsentence\nclip.wav\t"first\nsecond"\nmissing.wav\n')
        with pytest.raises(ValueError, match="line 4.*expected 2 columns, got 1"):
            load()

    def test_common_voice_and_duration_tables_keep_blank_line_compatibility(
        self, tmp_path, common_voice
    ):
        path, load = common_voice
        path.write_text("path\tsentence\n\nclip.wav\ttranscript\n\n")
        durations = tmp_path / "durations.tsv"
        durations.write_text("path\tduration\n\nclip.wav\t2000\n\n")
        manifest = load(durations_tsv=durations)
        assert manifest.references() == ["transcript"]
        assert manifest.segments[0].duration_s == 2
