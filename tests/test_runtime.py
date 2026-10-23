import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import wave
from contextlib import nullcontext
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from support import TINY_MANIFEST, synthetic_registry, write_wav

from shono.data.audio_paths import validate_manifest_audio
from shono.data.manifest import Manifest, Segment
from shono.eval.report import Predictions, ReportSpec
from shono.markdown import escape_cell
from shono.protocol import BASE_MODEL_ID, BASE_SYSTEM, report_spec, run_matrix
from shono.provenance import GitState, RunContext
from shono.train.audio import WhisperFineTuneDataset
from shono.train.checkpoint import ResumeDecision, decide_resume, find_latest_checkpoint
from shono.train.config import TrainConfig
from shono.train.data import TrainExample, build_examples, total_hours
from shono.train.experiment import render_experiment
from shono.train.trainer import DataCollatorSpeechSeq2Seq, run_training
from shono.transcribe.__main__ import _write_json, main
from shono.transcribe.batch import ManifestTranscriber, run_over_manifest
from shono.transcribe.chunking import plan_chunks
from shono.transcribe.convert import to_ct2, validate_ct2_checkpoint
from shono.transcribe.hallucination import HallucinationConfig, filter_hallucinations
from shono.transcribe.merge import merge_transcriptions
from shono.transcribe.pipeline import (
    FasterWhisperTranscriber,
    LongFormPipelineTranscriber,
    LongFormTranscriber,
    SileroVAD,
    real_time_factor,
)
from shono.transcribe.types import (
    ChunkTranscription,
    SpeechSegment,
    Transcript,
    TranscriptSegment,
    Word,
)
from shono.transcribe.whisper import ShortFormWhisperTranscriber

REPO = Path(__file__).resolve().parents[1]
MODEL_REVISION = "a" * 40


def stub_module(monkeypatch, name, **exports):
    monkeypatch.setitem(sys.modules, name, SimpleNamespace(**exports))


class FakeManifestTranscriber:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio_path, start_s=None, duration_s=None):
        self.calls.append((audio_path, start_s, duration_s))
        return f"অনুমান {len(self.calls)}"


def install_decoded_audio(monkeypatch, samples):
    stub_module(monkeypatch, "librosa", load=lambda *args, **kwargs: (samples, 16000))
    stub_module(
        monkeypatch,
        "numpy",
        isfinite=lambda data: SimpleNamespace(all=lambda: all(math.isfinite(x) for x in data)),
        ascontiguousarray=lambda data, **kwargs: data,
        float32="float32",
    )


def install_decoder(monkeypatch, segments):
    adapter = FasterWhisperTranscriber("local-checkpoint", device="cpu", compute_type="int8")
    monkeypatch.setattr(
        adapter,
        "_load",
        lambda: SimpleNamespace(transcribe=lambda *a, **kw: (iter(segments), None)),
    )
    return adapter


def write_conversion_checkpoint(tmp_path, name="model", ct2=False, slow=False):
    model = tmp_path / name
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "whisper"}))
    (model / ("model.bin" if ct2 else "model.safetensors")).write_bytes(b"fake weights")
    for filename in ("preprocessor_config.json", "tokenizer_config.json"):
        (model / filename).write_text("{}")
    if slow:
        (model / "vocab.json").write_text('{"a":1}')
        (model / "merges.txt").write_text("a b\n")
    else:
        (model / "tokenizer.json").write_text("{}")
    return model


def install_fake_converter(monkeypatch):
    calls = []

    class Converter:
        def __init__(self, source, *, copy_files):
            calls.append(("load", source, copy_files))

        def convert(self, output, *, quantization, force):
            calls.append(("convert", output, quantization, force))

    stub_module(monkeypatch, "ctranslate2.converters", TransformersConverter=Converter)
    return calls


def make_decoder_segment(start, text, *, timed=True, **scores):
    return SimpleNamespace(
        start=start,
        end=start + 1,
        text=text,
        words=[SimpleNamespace(start=start, end=start + 1, word=text, probability=0.9)]
        if timed
        else None,
        avg_logprob=scores.get("avg_logprob", -0.1),
        compression_ratio=scores.get("compression_ratio", 1.0),
        no_speech_prob=scores.get("no_speech_prob", 0.0),
    )


def install_hf_runtime(
    monkeypatch,
    *,
    revision=MODEL_REVISION,
    during_processor=None,
    model_factory=None,
    processor_factory=None,
    config=None,
):
    calls = []
    config = config or SimpleNamespace(_commit_hash=revision)
    model = SimpleNamespace(
        config=config,
        generation_config=SimpleNamespace(forced_decoder_ids=None, to_dict=lambda: {}),
        to=lambda device: None,
        eval=lambda: None,
    )
    processor = object()

    def load_config(source, **kwargs):
        calls.append(("config", source, kwargs))
        return config

    def load_model(source, **kwargs):
        calls.append(("model", source, kwargs))
        return model_factory() if model_factory else model

    def load_processor(source, **kwargs):
        calls.append(("processor", source, kwargs))
        if during_processor:
            during_processor()
        return processor_factory() if processor_factory else processor

    stub_module(monkeypatch, "torch", cuda=SimpleNamespace(is_available=lambda: False))
    stub_module(
        monkeypatch,
        "transformers",
        WhisperConfig=SimpleNamespace(from_pretrained=load_config),
        WhisperForConditionalGeneration=SimpleNamespace(from_pretrained=load_model),
        WhisperProcessor=SimpleNamespace(from_pretrained=load_processor),
    )
    return (calls, config, model, processor)


def training_config(**overrides) -> TrainConfig:
    settings = dict(
        model_id="base/whisper-medium-bn", model_revision="a" * 40, output_dir="/tmp/out"
    )
    settings.update(overrides)
    return TrainConfig(**settings)


def write_checkpoint_state(root, step, *, complete=True):
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir()
    if complete:
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": step}), encoding="utf-8"
        )
    return checkpoint


def make_selection_manifest(name, split, durations_s):
    segments = tuple(
        Segment(
            id=f"{name}-{index}",
            audio=f"{name}-{index}.wav",
            text=f"বাক্য {index}",
            duration_s=duration_s,
            recording_id=f"{name}-rec",
        )
        for index, duration_s in enumerate(durations_s)
    )
    return Manifest(
        name=name,
        source="openslr_slr53",
        split=split,
        domain="read",
        version="v",
        segments=segments,
    )


def experiment_run_context() -> RunContext:
    return RunContext.capture(
        42,
        {"report": "m3"},
        now=lambda: datetime(2026, 7, 23, tzinfo=UTC),
        git_state=GitState(sha="abc", dirty=False),
    )


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def item(self):
        return self.value


class FakeTensor:
    def __init__(self, rows):
        self.rows = rows

    def ne(self, value):
        return FakeTensor([[x != value for x in row] for row in self.rows])

    def masked_fill(self, mask, value):
        return FakeTensor(
            [
                [value if flag else x for (x, flag) in zip(row, flags, strict=True)]
                for (row, flags) in zip(self.rows, mask.rows, strict=True)
            ]
        )

    def __getitem__(self, index):
        (_, column) = index
        return FakeTensor([row[column] for row in self.rows])

    def __eq__(self, value):
        return FakeTensor([x == value for x in self.rows])

    def all(self):
        return FakeScalar(all(self.rows))

    def any(self):
        return FakeScalar(any(self.rows))


class FakeTrainingTokenizer:
    prefix_tokens = [1, 2, 3, 9]
    eos_token_id = 4
    pad_token_id = 0
    text_length = 2

    def convert_tokens_to_ids(self, text):
        return {"<|notimestamps|>": 9, "<|0.00|>": 100}[text]

    def __call__(self, text, add_special_tokens=True):
        prefix = self.prefix_tokens if add_special_tokens else []
        suffix = [self.eos_token_id] if add_special_tokens else []
        return SimpleNamespace(input_ids=[*prefix, *[5] * self.text_length, *suffix])

    def decode(self, ids, skip_special_tokens=True):
        return "target" if 5 in ids else ""

    def pad(self, rows, return_tensors):
        count = max(len(row["input_ids"]) for row in rows)
        ids = [row["input_ids"] + [0] * (count - len(row["input_ids"])) for row in rows]
        masks = [
            [1] * len(row["input_ids"]) + [0] * (count - len(row["input_ids"])) for row in rows
        ]
        result = {"input_ids": FakeTensor(ids), "attention_mask": FakeTensor(masks)}
        return AttributeDictionary(result)


class AttributeDictionary(dict):
    __getattr__ = dict.__getitem__


class FakeFeatureExtractor:
    def __call__(self, audio, **kwargs):
        assert kwargs["return_attention_mask"] is True
        return SimpleNamespace(input_features=[[1, 2, 3]], attention_mask=[[1, 1, 0]])

    def pad(self, rows, return_tensors):
        return {key: [row[key] for row in rows] for key in rows[0]}


def make_training_processor():
    return SimpleNamespace(
        tokenizer=FakeTrainingTokenizer(), feature_extractor=FakeFeatureExtractor()
    )


def make_training_manifest(
    name, split, *, segment_id=None, text=None, audio_path=None, duration_s=3
):
    segment_id = segment_id or name
    return Manifest(
        name=name,
        source="synthetic",
        split=split,
        domain="read",
        version="fixture",
        segments=(
            Segment(
                id=segment_id,
                audio=audio_path or f"{segment_id}.wav",
                text=text or name,
                duration_s=duration_s,
                recording_id=segment_id,
            ),
        ),
    )


class TrainingRun(SimpleNamespace):
    @property
    def output(self):
        return Path(self.cfg.output_dir)

    @property
    def identity_path(self):
        return self.output / "training-inputs.json"

    def identity(self):
        return json.loads(self.identity_path.read_text())

    def save_identity(self, identity):
        self.identity_path.write_text(json.dumps(identity))

    def run(self, **kwargs):
        return run_training(
            self.cfg,
            [self.train],
            [self.dev],
            registry=self.registry,
            audio_root=self.audio,
            duration_of=kwargs.pop("duration_of", lambda path: 3),
            **kwargs,
        )

    def local_checkpoint(self):
        checkpoint = self.audio.parent / "base"
        checkpoint.mkdir()
        for name in (
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "preprocessor_config.json",
        ):
            (checkpoint / name).write_bytes(b"original")
        self.cfg = replace(self.cfg, model_id=str(checkpoint), model_revision=None)
        return checkpoint

    def rejects(self, match=None, *, events=(), error=ValueError, **kwargs):
        with pytest.raises(error, match=match):
            self.run(**kwargs)
        assert [event[0] for event in self.events] == list(events)

    def change_segment(self, role, **changes):
        manifest = getattr(self, role)
        segment = replace(manifest.segments[0], **changes)
        setattr(self, role, replace(manifest, segments=(segment,)))


def confidence_chunk(text="কথা", *, avg_logprob=-0.2, compression_ratio=1.5, no_speech_prob=0.1):
    return ChunkTranscription(
        window_start_s=0.0,
        window_end_s=5.0,
        text=text,
        avg_logprob=avg_logprob,
        compression_ratio=compression_ratio,
        no_speech_prob=no_speech_prob,
    )


class PlannedSpeechVAD:
    def __init__(self, segments):
        self._segments = segments

    def detect(self, audio_path):
        return self._segments


class WindowDecoder:
    def transcribe_chunk(self, audio_path, start_s, end_s):
        if start_s >= 100:
            return (
                ChunkTranscription(
                    start_s, end_s, "নকল", no_speech_prob=0.95, words=(Word(0.0, 1.0, "নকল"),)
                ),
            )
        return (ChunkTranscription(start_s, end_s, "ভালো", words=(Word(0.0, 1.0, "ভালো"),)),)


class TestManifestTranscription:
    @pytest.fixture
    def transcribe(self, audio_root):
        def run(transcriber=None, *, system="fake", **kwargs):
            return run_over_manifest(
                transcriber or FakeManifestTranscriber(),
                Manifest.from_text(TINY_MANIFEST),
                system,
                audio_root=audio_root,
                **kwargs,
            )

        return run

    @pytest.fixture
    def audio_root(self, tmp_path):
        for name in ("rec-loop-01.wav", "rec-loop-02.wav"):
            write_wav(tmp_path / name, sample_rate=10, frames=200)
        return tmp_path

    def test_fake_satisfies_the_protocol(self):
        assert isinstance(FakeManifestTranscriber(), ManifestTranscriber)

    def test_run_over_manifest_covers_every_segment(self, transcribe, audio_root):
        manifest = Manifest.from_text(TINY_MANIFEST)
        transcriber = FakeManifestTranscriber()
        preds = transcribe(transcriber, system="fake-local")
        assert preds.system == "fake-local"
        assert preds.manifest == "tiny-longform-test"
        assert set(preds.hypotheses) == set(manifest.ids())
        assert len(transcriber.calls) == len(manifest.segments)
        assert Path(transcriber.calls[0][0]) == audio_root / "rec-loop-01.wav"
        assert preds.input_paths == tuple(call[0] for call in transcriber.calls)

    def test_preflight_reads_each_recording_duration_once(self, audio_root):
        inspected = []

        def read_duration(path):
            inspected.append(path)
            return 20

        paths = validate_manifest_audio(
            Manifest.from_text(TINY_MANIFEST), audio_root=audio_root, duration_of=read_duration
        )
        assert inspected == list(dict.fromkeys(paths))

    def test_late_missing_or_invalid_audio_rejects_before_calls(self, transcribe, audio_root):
        transcriber = FakeManifestTranscriber()
        (audio_root / "rec-loop-02.wav").unlink()
        with pytest.raises(ValueError, match="missing audio"):
            transcribe(transcriber)
        assert transcriber.calls == []

    def test_late_eof_rejects_before_calls(self, transcribe, audio_root):
        transcriber = FakeManifestTranscriber()
        with pytest.raises(ValueError, match="EOF"):
            transcribe(transcriber, duration_of=lambda p: 20 if p.name.endswith("01.wav") else 1)
        assert transcriber.calls == []

    def test_observed_runtime_metadata_is_recorded_and_redacted(self, transcribe, audio_root):
        transcriber = FakeManifestTranscriber()
        transcriber.observed_runtime_context = lambda: {"device": "cpu", "api_key": "private"}
        predictions = transcribe(transcriber)
        assert predictions.run_context == {
            "observed_runtime": {"device": "cpu", "api_key": "[redacted]"}
        }

    def test_longform_vad_identity_survives_prediction_export(
        self, audio_root, tmp_path, monkeypatch
    ):

        class Decoder:
            def transcribe_chunk(self, audio_path, start_s, end_s):
                return (ChunkTranscription(start_s, end_s, "অনুমান"),)

        vad = SileroVAD()
        monkeypatch.setattr(vad, "detect", lambda path: [SpeechSegment(0, 1)])
        adapter = LongFormPipelineTranscriber(LongFormTranscriber(vad, Decoder(), pad_s=0))
        manifest = Manifest(
            "synthetic",
            "synthetic",
            "test",
            "long-form",
            "fixture",
            (Segment("recording", "rec-loop-01.wav", "সঠিক", 20, "recording"),),
        )
        predictions = run_over_manifest(adapter, manifest, "synthetic", audio_root=audio_root)
        path = tmp_path / "predictions.jsonl"
        predictions.to_jsonl(path)
        metadata = Predictions.from_jsonl(path).run_context["observed_runtime"]
        assert metadata["vad"] == vad.observed_runtime_context()
        assert metadata["vad"]["revision"] == "1e261b036686cd0017d500ee96acd1c4ba572a9d"
        assert metadata["pad_s"] == 0


class TestRunMatrix:
    def test_builtin_protocols_cover_only_base_and_fine_tuned_models(self):
        assert {entry.system for entry in run_matrix()} == {
            BASE_SYSTEM,
            "shono whisper-medium (fine-tuned)",
        }
        baseline = report_spec("baselines")
        assert [system["model_id"] for system in baseline["systems"]] == [BASE_MODEL_ID]

    def test_canonical_base_is_not_generic_or_regional_checkpoint(self):
        assert BASE_MODEL_ID == "bengaliAI/tugstugi_bengaliai-asr_whisper-medium"
        assert BASE_SYSTEM == "tugstugi whisper-medium (base)"
        assert "regional" not in BASE_MODEL_ID
        for name in ("baselines", "longform", "codeswitch", "full"):
            spec = ReportSpec.from_dict(report_spec(name))
            base = next(system for system in spec.systems if system.name == BASE_SYSTEM)
            assert base.model_id == BASE_MODEL_ID

    @pytest.mark.parametrize(
        "mode",
        ["bounded-shrot", "provider-window", "provider-whole-recording"],
    )
    def test_unknown_inference_modes_are_rejected(self, mode):
        entry = run_matrix()[0]

        with pytest.raises(ValueError, match="unknown inference mode"):
            replace(entry, inference_mode=mode)

    def test_every_spec_cell_is_in_the_run_matrix_with_same_paths_and_modes(self):
        entries = {(entry.system, entry.slice_name): entry for entry in run_matrix()}
        for name in ("baselines", "longform", "codeswitch", "full"):
            spec = report_spec(name)
            for system in spec["systems"]:
                for item in spec["slices"]:
                    entry = entries[system["name"], item["name"]]
                    assert entry.predictions == system["predictions"][item["name"]]
                    assert entry.manifest == item["manifest"]
                    assert entry.inference_mode == system["inference_modes"][item["name"]]
        assert len(report_spec("full")["systems"]) == 2
        assert len(report_spec("full")["slices"]) == 3
        assert all(
            entry.inference_mode == "long-form-vad"
            for entry in entries.values()
            if entry.domain == "long-form"
        )
        assert all(
            entry.inference_mode == "bounded-short"
            for entry in entries.values()
            if entry.slug == "shono-medium" and entry.domain != "long-form"
        )


class TestPredictionProvenance:
    def test_prediction_writer_preserves_audio_inputs(self, tmp_path):
        source = tmp_path / "audio.wav"
        source.write_bytes(b"original")
        predictions = Predictions("s", "m", {"a": "text"}, input_paths=(str(source),))
        with pytest.raises(ValueError, match="overwrite input"):
            predictions.to_jsonl(source)
        assert source.read_bytes() == b"original"

    def test_unknown_git_status_does_not_claim_a_clean_tree(self, monkeypatch):

        def git(command, **kwargs):
            if command[1] == "rev-parse":
                return SimpleNamespace(returncode=0, stdout="a" * 40)
            return SimpleNamespace(returncode=128, stdout="")

        monkeypatch.setattr("shono.provenance.subprocess.run", git)
        context = RunContext.capture(0, {}, packages=())
        assert context.git.sha == "a" * 40
        assert context.to_dict()["git"]["dirty"] is None

    def test_backslashes_cannot_cancel_table_pipe_escaping(self):
        value = escape_cell("slice\\|extra\nrow")
        (prefix, suffix) = value.split("|")
        assert len(prefix) - len(prefix.rstrip("\\")) == 3
        assert suffix == "extra row"


class TestModelAdapters:
    def test_hf_whisper_fake_generation_and_observed_settings(self, tmp_path, monkeypatch):
        path = write_wav(tmp_path / "audio.wav", sample_rate=4, frames=8)
        install_decoded_audio(monkeypatch, [0] * 32000)
        features = Mock()
        features.to.return_value = features
        processor = Mock(
            return_value=SimpleNamespace(input_features=features, attention_mask=features)
        )
        processor.batch_decode.return_value = [" বাংলা "]
        config = SimpleNamespace(_commit_hash=MODEL_REVISION)
        model = Mock(config=config)
        model.generation_config = SimpleNamespace(
            forced_decoder_ids=[[1, 2]],
            to_dict=lambda: {"num_beams": 5, "do_sample": False},
        )
        model.generate.return_value = [1]
        load_processor = Mock(return_value=processor)
        load_model = Mock(return_value=model)
        stub_module(
            monkeypatch,
            "torch",
            no_grad=nullcontext,
            cuda=SimpleNamespace(is_available=lambda: False),
        )
        stub_module(
            monkeypatch,
            "transformers",
            WhisperConfig=SimpleNamespace(from_pretrained=lambda name: config),
            WhisperProcessor=SimpleNamespace(from_pretrained=load_processor),
            WhisperForConditionalGeneration=SimpleNamespace(from_pretrained=load_model),
        )
        adapter = ShortFormWhisperTranscriber("local-checkpoint")
        assert adapter.transcribe(str(path)) == "বাংলা"
        load_processor.assert_called_once_with("local-checkpoint", revision=MODEL_REVISION)
        load_model.assert_called_once_with(
            "local-checkpoint", config=config, revision=MODEL_REVISION
        )
        assert len(processor.call_args.args[0]) == 32000
        assert processor.call_args.kwargs == {
            "sampling_rate": 16000,
            "return_tensors": "pt",
            "return_attention_mask": True,
        }
        processor.batch_decode.assert_called_once_with([1], skip_special_tokens=True)
        model.generate.assert_called_once_with(
            features, attention_mask=features, language="bn", task="transcribe"
        )
        assert model.generation_config.forced_decoder_ids is None
        features.to.assert_called_with("cpu")
        assert adapter.observed_runtime_context()["generation_config"]["num_beams"] == 5
        assert adapter.observed_runtime_context()["model_revision"] == MODEL_REVISION

    def test_faster_whisper_does_not_invent_missing_word_timestamps(self, tmp_path, monkeypatch):
        path = write_wav(tmp_path / "audio.wav", sample_rate=4, frames=8)
        install_decoded_audio(monkeypatch, [0] * 16000)
        segment = make_decoder_segment(0, "কথা", timed=False, avg_logprob=-0.2)
        result = install_decoder(monkeypatch, [segment]).transcribe_chunk(str(path), 0, 1)
        assert len(result) == 1
        assert result[0].text == "কথা"
        assert result[0].words == ()

    def test_silero_model_load_is_cached_and_validates_input_first(self, tmp_path, monkeypatch):

        path = write_wav(tmp_path / "audio.wav", sample_rate=4, frames=8)
        install_decoded_audio(monkeypatch, [0] * 32000)
        loads = []

        def load(*args, **kwargs):
            loads.append((args, kwargs))
            return (
                "model",
                (
                    lambda *a, **kw: [{"start": 0, "end": 16000}],
                    None,
                    lambda *a, **kw: pytest.fail("Silero audio I/O must remain unused"),
                    None,
                ),
            )

        stub_module(
            monkeypatch, "torch", hub=SimpleNamespace(load=load), from_numpy=lambda audio: audio
        )
        vad = SileroVAD()
        with pytest.raises(ValueError):
            vad.detect(str(tmp_path / "missing.wav"))
        assert loads == []
        assert vad.detect(str(path))[0].duration_s == 1
        vad.detect(str(path))
        assert len(loads) == 1
        assert loads[0] == (
            ("snakers4/silero-vad:1e261b036686cd0017d500ee96acd1c4ba572a9d", "silero_vad"),
            {"trust_repo": True, "skip_validation": True},
        )
        assert vad.observed_runtime_context() == {
            "repository": "snakers4/silero-vad",
            "revision": "1e261b036686cd0017d500ee96acd1c4ba572a9d",
            "entrypoint": "silero_vad",
            "threshold": 0.5,
            "min_speech_ms": 250,
            "sampling_rate": 16000,
        }

    @pytest.mark.parametrize("samples", [[], [float("nan")], [float("inf")]])
    def test_silero_invalid_decoded_audio_rejects_before_weights(
        self, tmp_path, monkeypatch, samples
    ):

        path = write_wav(tmp_path / "audio.wav", sample_rate=4, frames=8)
        install_decoded_audio(monkeypatch, samples)
        stub_module(
            monkeypatch,
            "torch",
            hub=SimpleNamespace(load=lambda *args, **kwargs: pytest.fail("no model load")),
        )
        with pytest.raises(ValueError, match="finite samples"):
            SileroVAD().detect(str(path))

    @pytest.fixture
    def ct2_checkpoint(self, tmp_path):
        path = tmp_path / "checkpoint"
        path.mkdir()
        (path / "model.bin").write_bytes(b"weights")
        for name in ("config.json", "tokenizer.json", "preprocessor_config.json"):
            (path / name).write_text("{}")
        return path

    def test_ct2_observer_records_loaded_files_and_does_not_rehash_mutable_disk(
        self, ct2_checkpoint, monkeypatch
    ):
        model = SimpleNamespace(model=SimpleNamespace(compute_type="int8_float32"))
        stub_module(monkeypatch, "faster_whisper", WhisperModel=lambda *a, **kw: model)
        adapter = FasterWhisperTranscriber(str(ct2_checkpoint), device="cpu", compute_type="int8")
        assert adapter.observed_runtime_context()["checkpoint"] == "unknown"
        assert adapter._load() is model
        metadata = adapter.observed_runtime_context()
        assert (
            metadata["checkpoint"]["files_sha256"]["model.bin"]
            == hashlib.sha256(b"weights").hexdigest()
        )
        assert metadata["loaded_compute_type"] == "int8_float32"
        (ct2_checkpoint / "model.bin").write_bytes(b"replacement")
        assert adapter._load() is model
        assert adapter.observed_runtime_context() == metadata
        metadata["checkpoint"]["files_sha256"]["model.bin"] = "changed by caller"
        assert (
            adapter.observed_runtime_context()["checkpoint"]["files_sha256"]["model.bin"]
            == hashlib.sha256(b"weights").hexdigest()
        )
        json.dumps(adapter.observed_runtime_context(), allow_nan=False)

    def test_checkpoint_changed_during_load_is_rejected_without_caching(
        self, ct2_checkpoint, monkeypatch
    ):

        def load(*args, **kwargs):
            (ct2_checkpoint / "config.json").write_text('{"changed": true}')
            return object()

        stub_module(monkeypatch, "faster_whisper", WhisperModel=load)
        adapter = FasterWhisperTranscriber(str(ct2_checkpoint), device="cpu", compute_type="int8")
        with pytest.raises(ValueError, match="checkpoint changed while loading"):
            adapter._load()
        assert adapter._model is None
        assert adapter.observed_runtime_context()["checkpoint"] == "unknown"

    def test_shortform_pins_weights_generation_and_processor_to_the_resolved_config(
        self, monkeypatch
    ):
        (calls, config, model, processor) = install_hf_runtime(monkeypatch)
        adapter = ShortFormWhisperTranscriber("owner/mutable-model")
        assert adapter._load() == (model, processor)
        assert [name for (name, _, _) in calls] == ["config", "model", "processor"]
        assert calls[1][2] == {"config": config, "revision": "a" * 40}
        assert calls[2][2] == {"revision": "a" * 40}
        assert adapter.observed_runtime_context()["checkpoint"] == {
            "source": "hub",
            "model_id": "owner/mutable-model",
            "revision": "a" * 40,
        }

    def test_shortform_local_processor_and_weights_load_offline_and_share_recorded_hashes(
        self, ct2_checkpoint, monkeypatch
    ):
        (calls, config, model, processor) = install_hf_runtime(monkeypatch, revision=None)
        adapter = ShortFormWhisperTranscriber(str(ct2_checkpoint))
        assert adapter._load() == (model, processor)
        assert all(call[2]["local_files_only"] is True for call in calls)
        metadata = adapter.observed_runtime_context()
        assert (
            metadata["checkpoint"]["files_sha256"]["model.bin"]
            == hashlib.sha256(b"weights").hexdigest()
        )
        (ct2_checkpoint / "model.bin").write_bytes(b"changed")
        assert adapter._load() == (model, processor)
        assert adapter.observed_runtime_context() == metadata

    def test_shortform_rejects_local_checkpoint_changed_while_loading_processor(
        self, ct2_checkpoint, monkeypatch
    ):
        install_hf_runtime(
            monkeypatch,
            revision=None,
            during_processor=lambda: (ct2_checkpoint / "tokenizer.json").write_text(
                '{"changed": true}'
            ),
        )
        adapter = ShortFormWhisperTranscriber(str(ct2_checkpoint))
        with pytest.raises(ValueError, match="checkpoint changed while loading"):
            adapter._load()
        assert adapter._model is None and adapter._processor is None

    @pytest.mark.parametrize("revision", [None, "main", "a" * 7, "A" * 40])
    def test_shortform_requires_an_immutable_resolved_hub_revision(self, monkeypatch, revision):
        (calls, _, _, _) = install_hf_runtime(monkeypatch, revision=revision)
        adapter = ShortFormWhisperTranscriber("owner/model")
        with pytest.raises(ValueError, match="no immutable commit SHA"):
            adapter._load()
        assert [name for (name, _, _) in calls] == ["config"]
        assert adapter._model is None and adapter._processor is None

    def test_missing_local_checkpoint_rejects_before_loading_optional_runtime(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setitem(sys.modules, "transformers", None)
        adapter = ShortFormWhisperTranscriber(str(tmp_path / "missing"))
        with pytest.raises(ValueError, match="checkpoint directory does not exist"):
            adapter._load()

    def test_short_form_runtime_is_tested_without_model_downloads(self):

        adapter = ShortFormWhisperTranscriber("local-checkpoint", device="cpu")
        assert adapter.observed_runtime_context()["generation_config"] == "unknown"

    def test_failed_whisper_device_load_does_not_leave_a_cached_model(self, monkeypatch):
        models = []

        def make_model():
            model = Mock(config=config, generation_config=SimpleNamespace(forced_decoder_ids=None))
            if not models:
                model.to.side_effect = RuntimeError("device unavailable")
            models.append(model)
            return model

        calls, config, _, processor = install_hf_runtime(monkeypatch, model_factory=make_model)
        adapter = ShortFormWhisperTranscriber("local-checkpoint")
        with pytest.raises(RuntimeError, match="device unavailable"):
            adapter._load()
        assert adapter._model is None and adapter._processor is None
        assert adapter._load() == (models[1], processor)
        assert adapter._load() == (models[1], processor)
        models[1].to.assert_called_once_with("cpu")
        model_calls = [kwargs for name, source, kwargs in calls if name == "model"]
        assert model_calls == [{"config": config, "revision": MODEL_REVISION}] * 2
        assert len(models) == 2

    @pytest.mark.parametrize("untimed_words", [None, []])
    def test_mixed_word_timings_preserve_each_segments_precision(self, monkeypatch, untimed_words):
        monkeypatch.setattr("shono.transcribe.pipeline.validate_audio_window", lambda *a: 2)
        install_decoded_audio(monkeypatch, [0])
        segments = [make_decoder_segment(0, "প্রথম"), make_decoder_segment(1, "দ্বিতীয়")]
        segments[1].words = untimed_words
        adapter = install_decoder(monkeypatch, segments)
        segments = adapter.transcribe_chunk("recording.wav", 0, 2)
        transcript = merge_transcriptions(list(segments))
        assert transcript.text == "প্রথম দ্বিতীয়"
        assert [s.timing_precision for s in transcript.segments] == ["word", "chunk"]
        assert [(s.start_s, s.end_s) for s in transcript.segments] == [(0, 1), (1, 2)]
        assert segments[1].words == ()


class TestTranscriptionTypes:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, True, "1", 1j])
    def test_interval_and_clock_inputs_are_real_finite_and_nonnegative(self, bad):
        for make in (
            lambda: SpeechSegment(bad, 2),
            lambda: Word(bad, 2, "word"),
            lambda: TranscriptSegment(bad, 2, "text"),
            lambda: real_time_factor(bad, 10),
        ):
            with pytest.raises(ValueError):
                make()

    def test_word_local_bounds_and_confidence_guard(self):
        with pytest.raises(ValueError, match="within"):
            ChunkTranscription(10, 15, "text", words=(Word(1, 6, "text"),))
        with pytest.raises(ValueError, match="ordered"):
            ChunkTranscription(0, 5, "text", words=(Word(2, 3, "a"), Word(0, 1, "b")))
        for make in (
            lambda: Word(0, 1, "x", probability=1.1),
            lambda: HallucinationConfig(max_no_speech_prob=1.1),
            lambda: HallucinationConfig(min_avg_logprob=float("nan")),
            lambda: SileroVAD(min_speech_ms=True),
        ):
            with pytest.raises(ValueError):
                make()

    def test_speech_segment_rejects_nonpositive_duration(self):
        with pytest.raises(ValueError, match="end > start"):
            SpeechSegment(5.0, 5.0)


class TestChunkPlanning:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_chunk_s": 0},
            {"max_chunk_s": float("nan")},
            {"pad_s": -1},
            {"pad_s": 14},
            {"max_chunks": True},
            {"max_chunks": 0},
            {"max_chunks": 1000001},
        ],
    )
    def test_bad_chunk_parameters_reject_even_for_empty_input(self, kwargs):
        with pytest.raises(ValueError):
            plan_chunks([], **kwargs)

    def test_padding_is_within_hard_maximum_and_eof_without_gaps(self):
        chunks = plan_chunks([SpeechSegment(0, 70)], max_chunk_s=28, pad_s=0.2, audio_duration_s=70)
        assert chunks[0].start_s == 0 and chunks[-1].end_s == 70
        assert all(0 < c.duration_s <= 28 and c.forced_split for c in chunks)
        assert all(a.end_s >= b.start_s for (a, b) in zip(chunks, chunks[1:], strict=False))

    def test_bad_vad_eof_and_unbounded_plan_are_rejected(self):
        with pytest.raises(ValueError, match="beyond"):
            plan_chunks([SpeechSegment(0, 31)], audio_duration_s=30)
        with pytest.raises(ValueError, match="max_chunks"):
            plan_chunks([SpeechSegment(0, 1e308)], max_chunk_s=1e-300)
        with pytest.raises(ValueError, match="max_chunks"):
            plan_chunks([SpeechSegment(0, 60)], max_chunk_s=28, max_chunks=2)

    def test_subsample_large_timestamps_do_not_form_nonadvancing_chunks(self):
        with pytest.raises(ValueError, match="end > start|geometry"):
            plan_chunks([SpeechSegment(1e16, 1e16 + 4)], max_chunk_s=0.5)

    def test_chunks_pack_until_max_then_cut_at_silence(self):
        segs = [SpeechSegment(0, 10), SpeechSegment(11, 20), SpeechSegment(21, 40)]
        chunks = plan_chunks(segs, max_chunk_s=25.0)
        assert [(c.start_s, c.end_s) for c in chunks] == [(0, 20), (21, 40)]
        assert not any(c.forced_split for c in chunks)

    def test_single_segment_longer_than_max_is_force_split(self):
        chunks = plan_chunks([SpeechSegment(0, 70)], max_chunk_s=28.0)
        assert len(chunks) == 3
        assert all(c.forced_split for c in chunks)
        assert chunks[0].start_s == 0 and chunks[-1].end_s == pytest.approx(70)

    def test_plan_chunks_sorts_unordered_input(self):
        segs = [SpeechSegment(30, 40), SpeechSegment(0, 10)]
        chunks = plan_chunks(segs, max_chunk_s=100.0)
        assert chunks[0].start_s == 0

    def test_pad_extends_windows_without_going_negative(self):
        chunks = plan_chunks([SpeechSegment(0, 5)], max_chunk_s=28.0, pad_s=0.5)
        assert chunks[0].start_s == 0.0
        assert chunks[0].end_s == pytest.approx(5.5)

    def test_empty_input_yields_no_chunks(self):
        assert plan_chunks([], max_chunk_s=28.0) == []

    def test_nested_segment_does_not_shrink_coverage(self):
        chunks = plan_chunks([SpeechSegment(0, 20), SpeechSegment(5, 10)], max_chunk_s=25.0)
        assert [(c.start_s, c.end_s) for c in chunks] == [(0, 20)]


class TestTranscriptMerging:
    def test_missing_word_times_and_blank_chunks_preserve_new_text(self):
        result = merge_transcriptions(
            [
                ChunkTranscription(0, 10, ""),
                ChunkTranscription(5, 15, "new", words=(Word(1, 2, "new"),)),
            ]
        )
        assert result.text == "new" and result.segments[0].start_s == 6
        result = merge_transcriptions(
            [ChunkTranscription(0, 10, "first"), ChunkTranscription(8, 18, "last")]
        )
        assert result.text == "first last" and result.warnings
        assert all(s.timing_precision == "chunk" for s in result.segments)

    def test_timed_word_in_an_earlier_segment_gap_is_retained(self):
        first = ChunkTranscription(
            0, 10, "one three", words=(Word(0, 1, "one"), Word(9, 10, "three"))
        )
        second = ChunkTranscription(2, 8, "two", words=(Word(2, 3, "two"),))
        result = merge_transcriptions([first, second])
        assert result.text == "one two three"
        assert any(segment.start_s == 4 and segment.end_s == 5 for segment in result.segments)
        assert result.warnings

    def test_distinct_word_with_covered_timing_is_retained_with_warning(self):
        first = ChunkTranscription(0, 2, "one", words=(Word(0, 2, "one"),))
        second = ChunkTranscription(0.5, 2, "two", words=(Word(0, 1, "two"),))
        result = merge_transcriptions([first, second])
        assert result.text == "one two"
        assert result.warnings

    def test_matching_word_inside_prior_word_timing_is_removed(self):
        first = ChunkTranscription(0, 2, "one", words=(Word(0, 2, "one"),))
        second = ChunkTranscription(0.5, 2, "one", words=(Word(0, 1, "one"),))
        assert merge_transcriptions([first, second]).text == "one"

    def test_repeated_word_at_a_later_time_is_retained(self):
        first = ChunkTranscription(0, 2, "one", words=(Word(0, 1, "one"),))
        second = ChunkTranscription(0.5, 3, "one", words=(Word(1, 2, "one"),))
        assert merge_transcriptions([first, second]).text == "one one"

    def test_distinct_untimed_text_inside_a_prior_window_is_retained_with_warning(self):
        first = ChunkTranscription(0, 10, "first")
        second = ChunkTranscription(2, 8, "second")
        result = merge_transcriptions([first, second])
        assert result.text == "first second"
        assert result.warnings

    def test_matching_untimed_text_inside_a_prior_window_is_removed(self):
        first = ChunkTranscription(0, 10, "first")
        second = ChunkTranscription(2, 8, "first")
        assert merge_transcriptions([first, second]).text == "first"

    def test_equal_start_word_order_stays_in_decoder_order(self):
        chunk = ChunkTranscription(0, 3, "one two", words=(Word(0, 2, "one"), Word(0, 1, "two")))
        assert merge_transcriptions([chunk]).text == "one two"

    def test_merge_shifts_words_to_global_time(self):
        chunk = ChunkTranscription(10.0, 15.0, "শব্দ", words=(Word(1.0, 2.0, "শব্দ"),))
        transcript = merge_transcriptions([chunk])
        seg = transcript.segments[0]
        assert seg.start_s == pytest.approx(11.0) and seg.end_s == pytest.approx(12.0)
        assert transcript.text == "শব্দ"

    def test_merge_dedupes_overlapping_boundary_words(self):
        a = ChunkTranscription(0.0, 10.0, "", words=(Word(8.0, 9.0, "শেষ"),))
        b = ChunkTranscription(8.0, 18.0, "", words=(Word(0.5, 1.0, "শেষ"), Word(2.0, 3.0, "নতুন")))
        transcript = merge_transcriptions([a, b])
        assert transcript.text == "শেষ নতুন"

    def test_merge_wordless_chunks_concatenate_in_time_order(self):
        a = ChunkTranscription(0.0, 5.0, "প্রথম")
        b = ChunkTranscription(5.0, 10.0, "দ্বিতীয়")
        transcript = merge_transcriptions([b, a])
        assert transcript.text == "প্রথম দ্বিতীয়"

    def test_merge_wordless_overlap_keeps_new_tail_with_warning(self):
        a = ChunkTranscription(0.0, 10.0, "শেষ", words=(Word(8.0, 9.0, "শেষ"),))
        b = ChunkTranscription(8.0, 18.0, "শেষ নতুন")
        result = merge_transcriptions([a, b])
        assert result.text == "শেষ শেষ নতুন"
        assert result.warnings
        assert result.segments[-1].timing_precision == "chunk"


class TestLongFormPipeline:
    def test_invalid_pipeline_geometry_is_rejected_before_vad(self):
        vad = SimpleNamespace(detect=lambda *_: pytest.fail("VAD must not run"))
        decoder = SimpleNamespace(transcribe_chunk=lambda *_: pytest.fail("decoder must not run"))
        with pytest.raises(ValueError, match="30"):
            LongFormTranscriber(vad, decoder, max_chunk_s=31)
        pipeline = LongFormTranscriber(vad, decoder)
        with pytest.raises(ValueError):
            pipeline.transcribe("fake.wav", float("nan"))

    @pytest.mark.parametrize("start,end", [(0, 2), (0, 0.5)])
    def test_pipeline_rejects_segments_outside_the_requested_window(self, start, end):
        vad = SimpleNamespace(detect=lambda *_: [SpeechSegment(0.5, 1)])
        decoder = SimpleNamespace(
            transcribe_chunk=lambda *_: (ChunkTranscription(start, end, "text"),)
        )
        with pytest.raises(ValueError, match="outside its audio window"):
            LongFormTranscriber(vad, decoder, pad_s=0).transcribe("fake.wav", 2)

    def test_pipeline_clock_is_validated(self):
        vad = SimpleNamespace(detect=lambda *_: [SpeechSegment(0, 1)])
        decoder = SimpleNamespace(transcribe_chunk=lambda *_: (ChunkTranscription(0, 1, "text"),))
        clock = iter([2, 1])
        with pytest.raises(ValueError, match="processing_s"):
            LongFormTranscriber(vad, decoder, pad_s=0).transcribe(
                "fake.wav", 2, now=lambda: next(clock)
            )

    def test_longform_cropped_input_rejects_before_pipeline(self):
        adapter = LongFormPipelineTranscriber(
            SimpleNamespace(transcribe=lambda *_: pytest.fail("must not run"))
        )
        with pytest.raises(ValueError, match="complete recording"):
            adapter.transcribe("fake.wav", start_s=1, duration_s=2)

    def test_pipeline_observer_records_decoder_and_actual_filter_thresholds(self):
        decoder = SimpleNamespace(observed_runtime_context=lambda: {"model": "fixture"})
        guard = HallucinationConfig(min_avg_logprob=-0.5, max_compression_ratio=2)
        pipeline = LongFormTranscriber(SimpleNamespace(), decoder, hallucination=guard)
        metadata = pipeline.observed_runtime_context()
        assert metadata["decoder"] == {"model": "fixture"}
        assert metadata["hallucination"] == {
            "min_avg_logprob": -0.5,
            "max_compression_ratio": 2,
            "max_no_speech_prob": 0.6,
        }

    def test_pipeline_end_to_end_with_fakes(self):
        vad = PlannedSpeechVAD([SpeechSegment(0, 20), SpeechSegment(100, 105)])
        pipeline = LongFormTranscriber(vad, WindowDecoder(), max_chunk_s=28.0, pad_s=0.0)
        clock = iter([0.0, 12.0])
        result = pipeline.transcribe("rec.wav", audio_duration_s=200.0, now=lambda: next(clock))
        assert result.n_chunks == 2
        assert result.filtering.n_dropped == 1
        assert result.transcript.text == "ভালো"
        assert result.rtf == pytest.approx(12.0 / 200.0)
        assert result.rtf < 1.0

    def test_long_form_pipeline_transcriber_returns_merged_text(self, monkeypatch):

        pipeline = LongFormTranscriber(
            PlannedSpeechVAD([SpeechSegment(0, 20), SpeechSegment(100, 105)]),
            WindowDecoder(),
            pad_s=0.0,
        )
        adapter = LongFormPipelineTranscriber(pipeline)
        monkeypatch.setattr(
            "shono.transcribe.pipeline.validate_audio_window", lambda *a, **kw: 200.0
        )
        text = adapter.transcribe("rec.wav", duration_s=200.0)
        assert text == "ভালো"


class TestCheckpointConversion:
    def test_conversion_is_local_and_copies_processor_files(self, tmp_path, monkeypatch):
        model = write_conversion_checkpoint(tmp_path)
        calls = install_fake_converter(monkeypatch)
        output = tmp_path / "out"
        assert to_ct2(model, output, quantization="int8") == str(output)
        assert calls == [
            ("load", str(model), ["preprocessor_config.json", "tokenizer.json"]),
            ("convert", str(output), "int8", False),
        ]

    def test_slow_tokenizer_is_resolved_locally_and_saved(self, tmp_path, monkeypatch):
        model = write_conversion_checkpoint(tmp_path, slow=True)
        install_fake_converter(monkeypatch)
        calls = []

        def load(source, *, use_fast, local_files_only):
            calls.append((source, use_fast, local_files_only))
            return SimpleNamespace(
                is_fast=True, save_pretrained=lambda output: calls.append(output)
            )

        stub_module(
            monkeypatch, "transformers", AutoTokenizer=SimpleNamespace(from_pretrained=load)
        )
        output = tmp_path / "out"
        to_ct2(model, output)
        assert calls == [(str(model), True, True), str(output)]

    @pytest.mark.parametrize("options", [{"force": 1}, {"quantization": "wrong"}])
    def test_conversion_invalid_flags_precede_converter(self, tmp_path, monkeypatch, options):
        calls = install_fake_converter(monkeypatch)
        with pytest.raises(ValueError):
            to_ct2(write_conversion_checkpoint(tmp_path), tmp_path / "out", **options)
        assert calls == []

    def test_conversion_rejects_source_ancestors_descendants_and_aliases(
        self, tmp_path, monkeypatch
    ):
        model = write_conversion_checkpoint(tmp_path)
        calls = install_fake_converter(monkeypatch)
        for output in (model, model / "nested", tmp_path):
            with pytest.raises(ValueError, match="separate"):
                to_ct2(model, output, force=True)
        alias = tmp_path / "linked"
        alias.symlink_to(model, target_is_directory=True)
        with pytest.raises(ValueError, match="symlink"):
            to_ct2(model, alias, force=True)
        target = tmp_path / "target"
        target.mkdir()
        os.link(model / "model.safetensors", target / "alias.bin")
        with pytest.raises(ValueError, match="aliases"):
            to_ct2(model, target, force=True)
        assert calls == []

    def test_conversion_protects_resolved_source_file_inside_output(self, tmp_path, monkeypatch):
        model = write_conversion_checkpoint(tmp_path)
        output = tmp_path / "output"
        output.mkdir()
        weights = model / "model.safetensors"
        weights.rename(output / "weights")
        weights.symlink_to(output / "weights")
        calls = install_fake_converter(monkeypatch)
        with pytest.raises(ValueError, match="resolved checkpoint"):
            to_ct2(model, output, force=True)
        assert calls == []

    def test_conversion_protects_repo_sources_without_project_metadata(self, tmp_path, monkeypatch):
        import shono.transcribe.convert as module

        project = tmp_path / "project"
        package = project / "src" / "shono" / "transcribe"
        package.mkdir(parents=True)
        (project / "docs").mkdir()
        (project / "docs" / "notes.txt").write_text("keep")
        (project / "scripts").mkdir()
        monkeypatch.setattr(module, "__file__", str(package / "convert.py"))
        model = write_conversion_checkpoint(tmp_path)
        calls = install_fake_converter(monkeypatch)
        for output in (project / "docs", project / "scripts", project / "Docs" / "new", project):
            with pytest.raises(ValueError, match="repository sources"):
                to_ct2(model, output, force=True)
        assert calls == []
        to_ct2(model, project / "outputs" / "converted")
        assert calls[-1][0] == "convert"

    def test_installed_conversion_does_not_protect_unrelated_directories(
        self, tmp_path, monkeypatch
    ):
        import shono.transcribe.convert as module

        package = tmp_path / "lib" / "site-packages" / "shono" / "transcribe"
        package.mkdir(parents=True)
        monkeypatch.setattr(module, "__file__", str(package / "convert.py"))
        model = write_conversion_checkpoint(tmp_path)
        calls = install_fake_converter(monkeypatch)
        output = tmp_path / "lib" / "reports" / "converted"
        assert to_ct2(model, output) == str(output)
        assert calls[-1][0] == "convert"

    def test_incomplete_checkpoint_and_missing_runtime_have_clear_errors(
        self, tmp_path, monkeypatch
    ):
        model = write_conversion_checkpoint(tmp_path)
        (model / "model.safetensors").unlink()
        calls = install_fake_converter(monkeypatch)
        with pytest.raises(ValueError, match="weights"):
            to_ct2(model, tmp_path / "out")
        assert calls == []
        (model / "model.safetensors").write_bytes(b"fake")
        monkeypatch.setitem(sys.modules, "ctranslate2.converters", None)
        with pytest.raises(RuntimeError, match="requires.*runtimes"):
            to_ct2(model, tmp_path / "out")

    def test_conversion_cli_input2_and_missing_runtime1(self, tmp_path, monkeypatch, capsys):
        from shono.transcribe.convert import main as convert_main

        assert convert_main([str(tmp_path / "missing"), str(tmp_path / "out")]) == 2
        model = write_conversion_checkpoint(tmp_path)
        monkeypatch.setitem(sys.modules, "ctranslate2.converters", None)
        assert convert_main([str(model), str(tmp_path / "out")]) == 1
        assert "Traceback" not in capsys.readouterr().err

    def test_ct2_checkpoint_requires_processor_json_objects(self, tmp_path):
        model = write_conversion_checkpoint(tmp_path, ct2=True)
        assert validate_ct2_checkpoint(model) == model
        (model / "preprocessor_config.json").write_text("[]")
        with pytest.raises(ValueError, match="JSON object"):
            validate_ct2_checkpoint(model)


class TestTranscriptionCli:
    def test_json_export_protects_repo_sources_without_project_metadata(
        self, tmp_path, monkeypatch
    ):
        import shono.transcribe.__main__ as module

        project = tmp_path / "project"
        package = project / "src" / "shono" / "transcribe"
        package.mkdir(parents=True)
        source = project / "data" / "annotation.json"
        source.parent.mkdir()
        source.write_text("{}")
        monkeypatch.setattr(module, "__file__", str(package / "__main__.py"))
        model = write_conversion_checkpoint(tmp_path, ct2=True)
        audio = write_wav(tmp_path / "audio.wav", sample_rate=10)
        for target in (source, source.with_name("ANNOTATION.json")):
            with pytest.raises(ValueError, match="overwrite input"):
                module._output_path(target, audio, model)
        output = project / "outputs" / "transcript.json"
        assert module._output_path(output, audio, model) == output

    def test_installed_json_export_does_not_protect_unrelated_files(self, tmp_path, monkeypatch):
        import shono.transcribe.__main__ as module

        package = tmp_path / "lib" / "site-packages" / "shono" / "transcribe"
        package.mkdir(parents=True)
        monkeypatch.setattr(module, "__file__", str(package / "__main__.py"))
        model = write_conversion_checkpoint(tmp_path, ct2=True)
        audio = write_wav(tmp_path / "audio.wav", sample_rate=10)
        target = tmp_path / "lib" / "reports" / "transcript.json"
        target.parent.mkdir()
        target.write_text("{}")
        assert module._output_path(target, audio, model) == target

    def test_json_export_aliases_reject_before_pipeline(self, tmp_path, monkeypatch, capsys):
        import shono.transcribe.pipeline as module

        model = write_conversion_checkpoint(tmp_path, ct2=True)
        audio = write_wav(tmp_path / "audio.wav", sample_rate=10)
        original = audio.read_bytes()
        monkeypatch.setattr(module, "SileroVAD", lambda: pytest.fail("VAD must remain unloaded"))
        alias = tmp_path / "alias.json"
        os.link(audio, alias)
        assert main([str(audio), "--model", str(model), "--json", str(alias)]) == 1
        alias.unlink()
        alias.symlink_to(audio)
        assert main([str(audio), "--model", str(model), "--json", str(alias)]) == 1
        assert main([str(audio), "--model", str(model), "--json", str(model / "config.json")]) == 1
        assert audio.read_bytes() == original
        assert "Traceback" not in capsys.readouterr().err

    def test_incomplete_ct2_rejects_before_vad_and_module_help_is_clean(
        self, tmp_path, monkeypatch
    ):
        import shono.transcribe.pipeline as module

        model = tmp_path / "empty"
        model.mkdir()
        audio = write_wav(tmp_path / "audio.wav", sample_rate=10)
        monkeypatch.setattr(module, "SileroVAD", lambda: pytest.fail("VAD must remain unloaded"))
        assert main([str(audio), "--model", str(model)]) == 1
        for module_name in ("shono.transcribe", "shono.transcribe.convert"):
            result = subprocess.run(
                [sys.executable, "-W", "error", "-m", module_name, "--help"],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0 and "usage:" in result.stdout
            assert result.stderr == ""

    def test_atomic_json_failure_preserves_existing_file(self, tmp_path, monkeypatch):
        target = tmp_path / "output.json"
        target.write_text('{"old":true}\n')

        def fail(*args):
            raise OSError("fake replace failure")

        monkeypatch.setattr("shono.transcribe.__main__.os.replace", fail)
        with pytest.raises(OSError):
            _write_json(target, {"new": "বাংলা"})
        assert target.read_text() == '{"old":true}\n'
        assert list(tmp_path.iterdir()) == [target]

    def test_successful_cli_json_has_coarse_timing_and_warning(self, tmp_path, monkeypatch):
        import shono.transcribe.pipeline as module

        model = write_conversion_checkpoint(tmp_path, ct2=True)
        audio = write_wav(tmp_path / "audio.wav", sample_rate=10)
        original = audio.read_bytes()
        result = SimpleNamespace(
            transcript=Transcript(
                (TranscriptSegment(0, 1, "বাংলা", timing_precision="chunk"),), ("timing approximate",)
            ),
            rtf=0.5,
            audio_duration_s=1,
            processing_s=0.5,
            n_chunks=1,
            n_forced_splits=0,
            filtering=SimpleNamespace(n_dropped=0, summary=lambda: "kept all 1 decoder segments"),
        )
        monkeypatch.setattr(module, "SileroVAD", lambda: "fake VAD")
        decoder_settings = {}

        def make_decoder(*args, **kwargs):
            decoder_settings.update(kwargs)
            return "fake decoder"

        monkeypatch.setattr(module, "FasterWhisperTranscriber", make_decoder)
        inference = {"vad": {"repository": "fake/vad", "revision": "a" * 40}}
        monkeypatch.setattr(
            module,
            "LongFormTranscriber",
            lambda *a, **kw: SimpleNamespace(
                transcribe=lambda *a: result, observed_runtime_context=lambda: inference
            ),
        )
        target = tmp_path / "output" / "transcript.json"
        assert main([str(audio), "--model", str(model), "--json", str(target)]) == 0
        assert decoder_settings["device"] == "cpu"
        assert decoder_settings["compute_type"] == "int8"
        data = json.loads(target.read_text())
        assert data["text"] == "বাংলা" and data["segments"][0]["timing_precision"] == "chunk"
        assert data["warnings"] == ["timing approximate"]
        assert data["inference"] == inference
        assert audio.read_bytes() == original

    def test_cli_missing_audio_exits_2_without_loading_runtime(self, tmp_path, monkeypatch):
        import builtins

        original_import = builtins.__import__

        def core_imports_only(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "faster_whisper", "ctranslate2"}:
                pytest.fail(f"missing audio must be rejected before importing {name}")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", core_imports_only)
        assert main([str(tmp_path / "missing.wav"), "--model", "/some/ct2"]) == 2


class TestSegmentFiltering:
    @pytest.fixture
    def pipeline(self, monkeypatch):
        monkeypatch.setattr("shono.transcribe.pipeline.validate_audio_window", lambda *a: 4)
        install_decoded_audio(monkeypatch, [0])

        def make(segments, **kwargs):
            vad = SimpleNamespace(detect=lambda path: [SpeechSegment(10, 14)])
            return LongFormTranscriber(
                vad, install_decoder(monkeypatch, segments), pad_s=0, **kwargs
            )

        return make

    @pytest.mark.parametrize(
        "metric,value", [("avg_logprob", -1.8), ("compression_ratio", 3.5), ("no_speech_prob", 0.9)]
    )
    @pytest.mark.parametrize("bad_first", [True, False])
    @pytest.mark.parametrize("timed", [True, False])
    @pytest.mark.parametrize("bad_timed", [True, False])
    def test_good_segment_cannot_dilute_a_failing_segment(
        self, pipeline, metric, value, bad_first, timed, bad_timed
    ):
        (bad_start, good_start) = (0, 2) if bad_first else (2, 0)
        bad = make_decoder_segment(bad_start, "নকল", timed=bad_timed, **{metric: value})
        good = make_decoder_segment(good_start, "ভালো", timed=timed)
        segments = sorted([bad, good], key=lambda segment: segment.start)
        result = pipeline(segments).transcribe("recording.wav", 20)
        assert result.n_chunks == 1
        assert result.transcript.text == "ভালো"
        assert result.filtering.n_dropped == 1
        assert result.filtering.summary() == "kept 1, dropped 1 decoder segments"
        dropped = result.filtering.dropped[0]
        assert dropped.chunk.text == "নকল"
        assert getattr(dropped.chunk, metric) == value
        assert dropped.reasons[0].startswith(metric)
        kept = result.transcript.segments[0]
        assert (kept.start_s, kept.end_s) == (10 + good_start, 11 + good_start)
        assert kept.timing_precision == ("word" if timed else "chunk")

    def test_custom_thresholds_apply_to_each_segment(self, pipeline):
        segments = [
            make_decoder_segment(0, "নকল", avg_logprob=-0.8),
            make_decoder_segment(2, "ভালো"),
        ]
        strict = HallucinationConfig(min_avg_logprob=-0.5)
        result = pipeline(segments, hallucination=strict).transcribe("recording.wav", 20)
        assert result.transcript.text == "ভালো"
        assert result.filtering.n_dropped == 1
        relaxed = HallucinationConfig(min_avg_logprob=-2)
        segments[0].avg_logprob = -1.8
        result = pipeline(segments, hallucination=relaxed).transcribe("recording.wav", 20)
        assert result.transcript.text == "নকল ভালো"
        assert result.filtering.n_dropped == 0

    def test_all_failing_segments_leave_an_empty_transcript(self, pipeline):
        segments = [
            make_decoder_segment(0, "এক", avg_logprob=-2),
            make_decoder_segment(2, "দুই", compression_ratio=4),
        ]
        result = pipeline(segments).transcribe("recording.wav", 20)
        assert result.transcript.text == ""
        assert result.filtering.n_dropped == 2
        assert result.filtering.kept == ()

    def test_no_decoder_segments_does_not_count_an_accepted_segment(self, pipeline):
        result = pipeline([]).transcribe("recording.wav", 20)
        assert result.n_chunks == 1
        assert result.transcript.segments == ()
        assert result.filtering.kept == ()
        assert result.filtering.n_dropped == 0

    def test_word_alignment_outside_segment_bounds_keeps_correct_offsets(self, pipeline):
        segment = make_decoder_segment(1, "ভালো")
        (segment.start, segment.end) = (1.2, 1.8)
        result = pipeline([segment]).transcribe("recording.wav", 20)
        kept = result.transcript.segments[0]
        assert (kept.start_s, kept.end_s) == (11, 12)
        assert kept.text == "ভালো"
        assert kept.timing_precision == "word"

    @pytest.mark.parametrize(
        "field,value",
        [
            ("start", -0.1),
            ("end", 4.1),
            ("start", float("nan")),
            ("end", float("inf")),
            ("word_start", -0.1),
            ("word_end", 4.1),
        ],
    )
    def test_invalid_decoder_bounds_are_rejected(self, pipeline, field, value):
        segment = make_decoder_segment(0, "ভালো")
        if field.startswith("word_"):
            setattr(segment.words[0], field.removeprefix("word_"), value)
        else:
            setattr(segment, field, value)
        with pytest.raises(ValueError):
            pipeline([segment]).transcribe("recording.wav", 20)


class TestTrainingConfig:
    def test_config_defaults_are_the_recipe(self):
        c = training_config()
        assert c.optim == "adamw_bnb_8bit"
        assert c.freeze_encoder is False
        assert c.effective_batch_size == c.per_device_batch_size * c.gradient_accumulation_steps

    def test_config_requires_model_id(self):
        with pytest.raises(ValueError, match="model_id is required"):
            TrainConfig(model_id="", output_dir="/tmp/o")

    @pytest.mark.parametrize("revision", [None, "main", "v1.0", "a" * 7, "g" * 40, 123])
    def test_remote_model_requires_immutable_revision(self, revision):
        with pytest.raises(ValueError, match="model_revision.*40-character Hub commit"):
            training_config(model_revision=revision)

    @pytest.mark.parametrize(
        "model_id",
        [
            "http://example.invalid/model",
            "https://example.invalid/model",
            "https://huggingface.co/example/model",
        ],
    )
    def test_remote_urls_cannot_bypass_hub_revision_binding(self, model_id):
        with pytest.raises(ValueError, match="Hub repository ID"):
            training_config(model_id=model_id)

    def test_local_model_has_no_hub_revision(self, tmp_path):
        config = training_config(model_id=str(tmp_path), model_revision=None)
        assert config.model_revision is None
        with pytest.raises(ValueError, match="content hashes"):
            training_config(model_id=str(tmp_path))

    def test_missing_local_checkpoint_does_not_fall_back_to_hub(self, tmp_path):
        with pytest.raises(ValueError, match="directory does not exist"):
            training_config(model_id=str(tmp_path / "missing"))

    @pytest.mark.parametrize(
        "settings,field",
        [
            ({"optim": "sgd"}, "optim"),
            ({"learning_rate": 0}, "learning_rate"),
            ({"chunk_length_s": 5.0, "min_chunk_length_s": 10.0}, "min_chunk_length_s"),
            ({"timestamp_sample_fraction": 1.5}, "timestamp_sample_fraction"),
            ({"chunk_length_s": 45.0}, "chunk_length_s"),
        ],
    )
    def test_invalid_training_settings(self, settings, field):
        with pytest.raises(ValueError, match=field):
            training_config(**settings)


class TestCheckpointDiscovery:
    def test_resume_is_fresh_when_no_checkpoints(self, tmp_path):
        decision = decide_resume(tmp_path)
        assert decision.resume is False and decision.global_step == 0
        assert "fresh" in decision.summary

    def test_resume_picks_latest_complete_checkpoint(self, tmp_path):
        write_checkpoint_state(tmp_path, 500)
        write_checkpoint_state(tmp_path, 1000)
        write_checkpoint_state(tmp_path, 1500, complete=False)
        decision = decide_resume(tmp_path)
        assert decision.resume is True
        assert decision.checkpoint.name == "checkpoint-1000"
        assert decision.global_step == 1000

    def test_incomplete_checkpoint_never_selected(self, tmp_path):
        write_checkpoint_state(tmp_path, 700, complete=False)
        assert find_latest_checkpoint(tmp_path) is None

    @pytest.mark.parametrize(
        "state",
        [
            '{"global_step": 0, "global_step": 700}',
            '{"global_step": 700, "global_step": 700}',
            '{"global_step": 699}',
            '{"global_step": 700.0}',
            "[]",
            "{",
        ],
    )
    def test_invalid_checkpoint_state_never_selected(self, tmp_path, state):
        checkpoint = write_checkpoint_state(tmp_path, 700)
        (checkpoint / "trainer_state.json").write_text(state, encoding="utf-8")
        assert find_latest_checkpoint(tmp_path) is None

    @pytest.mark.parametrize("name", ["checkpoint-700\n", "checkpoint-700-old", "checkpoint--1"])
    def test_checkpoint_name_must_match_saved_step(self, tmp_path, name):
        checkpoint = tmp_path / name
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text('{"global_step": 700}', encoding="utf-8")
        assert find_latest_checkpoint(tmp_path) is None

    def test_resume_can_be_forced_off(self, tmp_path):
        write_checkpoint_state(tmp_path, 500)
        decision = decide_resume(tmp_path, enabled=False)
        assert decision.resume is False
        assert isinstance(decision, ResumeDecision)


class TestTrainingSelection:
    @pytest.fixture
    def select(self):
        def run(manifests, **overrides):
            settings = dict(
                chunk_length_s=30.0,
                min_chunk_length_s=1.0,
                timestamp_sample_fraction=0.0,
                seed=0,
                registry=synthetic_registry("openslr_slr53"),
            )
            return build_examples(manifests, **(settings | overrides))

        return run

    def test_build_examples_filters_by_duration(self, select):
        m = make_selection_manifest("m", "train", [0.5, 2.0, 30.0, 40.0])
        examples = select([m])
        kept = sorted(e.duration_s for e in examples)
        assert kept == [2.0, 30.0]

    def test_build_examples_preserves_scorable_original_target(self, select):
        manifest = make_selection_manifest("m", "train", [2.0])
        original = "হ্যাঁ... ১২!"
        manifest = replace(manifest, segments=(replace(manifest.segments[0], text=original),))
        examples = select([manifest])
        assert examples[0].text == original

    def test_duplicate_whole_file_and_zero_start_crop_are_rejected(self, select):
        manifest = make_selection_manifest("m", "train", [2.0])
        original = manifest.segments[0]
        manifest = replace(manifest, segments=(original, replace(original, id="crop", start_s=0.0)))
        with pytest.raises(ValueError, match="duplicate.*audio windows"):
            select([manifest])

    def test_build_examples_refuses_test_split(self, select):
        m = make_selection_manifest("m", "test", [2.0])
        with pytest.raises(ValueError, match="test data must never enter training"):
            select([m])

    def test_timestamp_fraction_is_deterministic_and_correctly_sized(self, select):
        m = make_selection_manifest("m", "train", [2.0] * 10)
        kw = dict(
            timestamp_sample_fraction=0.5,
            seed=7,
        )
        a = select([m], **kw)
        b = select([m], **kw)
        n_ts = sum(e.use_timestamps for e in a)
        assert n_ts == 5
        assert [e.use_timestamps for e in a] == [e.use_timestamps for e in b]

    def test_total_hours_sums_across_manifests(self, select):
        m1 = make_selection_manifest("a", "train", [30.0] * 120)
        m2 = make_selection_manifest("b", "train", [30.0] * 60)
        examples = select([m1, m2])
        assert total_hours(examples) == pytest.approx(1.5)


class TestExperimentRecords:
    def test_experiment_record_pre_run_says_not_run(self):
        text = render_experiment(
            "m3-v1",
            "fine-tuned model beats the base checkpoint",
            training_config(),
            experiment_run_context(),
        )
        assert "not run yet" in text
        assert "adamw_bnb_8bit" in text
        assert "fine-tuned model beats the base checkpoint" in text

    def test_experiment_record_distinguishes_unknown_git_status_from_clean(self):
        context = replace(experiment_run_context(), git=GitState(sha="abc", dirty=None))
        text = render_experiment("trial", "test hypothesis", training_config(), context)
        assert "git abc (working tree unknown)" in text

    def test_experiment_record_shows_delta_vs_baseline(self):
        text = render_experiment(
            "m3-v1",
            "beats baseline",
            training_config(),
            experiment_run_context(),
            metrics={"eval_wer": 0.25},
            baseline={"eval_wer": 0.34},
        )
        assert "eval_wer 0.2500" in text
        assert "-0.0900" in text


class TestTrainingIntegration:
    @pytest.mark.filterwarnings(
        "ignore:'(aifc|audioop|sunau)' is deprecated.*:DeprecationWarning:audioread.rawread"
    )
    @pytest.mark.filterwarnings(
        "ignore:mtime may not be reliable on this filesystem.*:UserWarning:transformers.trainer"
    )
    def test_real_cpu_training_exports_evaluates_and_resumes(self, tmp_path, monkeypatch):
        torch = pytest.importorskip("torch")
        pytest.importorskip("transformers", minversion="5.10.1")
        pytest.importorskip("accelerate")
        pytest.importorskip("librosa")
        import numpy as np
        from tokenizers.pre_tokenizers import ByteLevel
        from transformers import (
            Seq2SeqTrainer,
            WhisperConfig,
            WhisperFeatureExtractor,
            WhisperForConditionalGeneration,
            WhisperProcessor,
            WhisperTokenizer,
        )
        from transformers.models.whisper.tokenization_whisper import LANGUAGES

        from shono.train import trainer as training

        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
        checkpoint = tmp_path / "base"
        checkpoint.mkdir()
        vocabulary = {symbol: index for index, symbol in enumerate(sorted(ByteLevel.alphabet()))}
        special_tokens = [
            "<|endoftext|>",
            "<|startoftranscript|>",
            *(f"<|{language}|>" for language in LANGUAGES),
            "<|translate|>",
            "<|transcribe|>",
            "<|startoflm|>",
            "<|startofprev|>",
            "<|nospeech|>",
            "<|notimestamps|>",
        ]
        for token in [*special_tokens, *(f"<|{step * 0.02:.2f}|>" for step in range(1501))]:
            vocabulary[token] = len(vocabulary)
        tokenizer = WhisperTokenizer(
            vocab=vocabulary,
            merges=[],
            language="bn",
            task="transcribe",
            pad_token="<|endoftext|>",
            extra_special_tokens=special_tokens[1:],
        )
        processor = WhisperProcessor(WhisperFeatureExtractor(chunk_length=1), tokenizer)
        processor.save_pretrained(checkpoint)
        config = WhisperConfig(
            vocab_size=len(tokenizer),
            num_mel_bins=80,
            d_model=16,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=1,
            decoder_attention_heads=1,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            max_target_positions=16,
            max_source_positions=50,
            decoder_start_token_id=vocabulary["<|startoftranscript|>"],
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model = WhisperForConditionalGeneration(config)
        model.generation_config._from_model_config = False
        model.generation_config.suppress_tokens = []
        model.generation_config.begin_suppress_tokens = []
        model.generation_config.lang_to_id = {"<|bn|>": vocabulary["<|bn|>"]}
        model.generation_config.task_to_id = {"transcribe": vocabulary["<|transcribe|>"]}
        model.generation_config.no_timestamps_token_id = vocabulary["<|notimestamps|>"]
        model.generation_config.is_multilingual = True
        model.save_pretrained(checkpoint)
        audio_root = tmp_path / "audio"
        audio_root.mkdir()
        manifests = []
        for index, (split, text) in enumerate((("train", "alpha"), ("dev", "beta"))):
            samples = (np.sin(np.arange(4000) * (index + 1) * 0.1) * 1000).astype("<i2")
            with wave.open(str(audio_root / f"{split}.wav"), "wb") as stream:
                stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                stream.writeframes(samples.tobytes())
            manifests.append(
                Manifest(
                    name=split,
                    source="synthetic",
                    split=split,
                    domain="read",
                    version="fixture",
                    segments=(Segment(split, f"{split}.wav", text, 0.25, split),),
                )
            )
        settings = TrainConfig(
            model_id=str(checkpoint),
            output_dir=str(tmp_path / "trained"),
            optim="adamw_torch",
            fp16=False,
            gradient_checkpointing=False,
            per_device_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=2,
            save_steps=1,
            eval_steps=2,
            warmup_steps=0,
            chunk_length_s=1,
            min_chunk_length_s=0.1,
            timestamp_sample_fraction=0,
            resume=False,
        )
        for target in ("<|endoftext|>", "<|notimestamps|>", "<|0.00|>"):
            invalid = TrainExample("train.wav", target, 0.25, "train", "synthetic")
            dataset = WhisperFineTuneDataset(
                [invalid],
                processor,
                settings,
                audio_root=audio_root,
                max_target_positions=config.max_target_positions,
                decoder_start_token_id=config.decoder_start_token_id,
            )
            with pytest.raises(ValueError, match="empty after Whisper decoding"):
                dataset.validate_labels()
        training_arguments = training.to_training_arguments
        monkeypatch.setattr(
            training,
            "to_training_arguments",
            lambda config: training_arguments(
                config, use_cpu=True, disable_tqdm=True, dataloader_pin_memory=False
            ),
        )
        completed_runs = []
        train = Seq2SeqTrainer.train

        def record_training(trainer, **kwargs):
            result = train(trainer, **kwargs)
            completed_runs.append((kwargs["resume_from_checkpoint"], trainer.state.global_step))
            return result

        monkeypatch.setattr(Seq2SeqTrainer, "train", record_training)
        original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for iteration in range(2):
                metrics = run_training(
                    settings,
                    [manifests[0]],
                    [manifests[1]],
                    registry=synthetic_registry("synthetic"),
                    audio_root=audio_root,
                )
                assert all(
                    math.isfinite(metrics[key])
                    for key in ("train_loss", "eval_loss", "eval_wer", "eval_cer")
                )
                state_path = tmp_path / "trained" / "checkpoint-2" / "trainer_state.json"
                assert json.loads(state_path.read_text(encoding="utf-8"))["global_step"] == 2
                if iteration == 0:
                    shutil.rmtree(state_path.parent)
                    settings = replace(settings, resume=True)
        finally:
            torch.set_num_threads(original_threads)
        assert completed_runs == [(None, 2), (str(tmp_path / "trained" / "checkpoint-1"), 2)]
        restored = WhisperForConditionalGeneration.from_pretrained(settings.output_dir)
        restored_processor = WhisperProcessor.from_pretrained(
            settings.output_dir, language=settings.language, task=settings.task
        )
        assert restored.config.d_model == 16
        assert restored.generation_config.language == settings.language
        assert restored_processor.feature_extractor.chunk_length == 1
        assert (
            restored_processor.tokenizer("beta").input_ids == processor.tokenizer("beta").input_ids
        )
        transcriber = ShortFormWhisperTranscriber(settings.output_dir, device="cpu")
        assert isinstance(transcriber.transcribe(str(audio_root / "dev.wav")), str)


class TestTrainingLabels:
    def test_timestamped_label_omits_notimestamps_token(self, tmp_path):

        class Tokenizer(FakeTrainingTokenizer):
            def __call__(self, text, add_special_tokens=True):
                return SimpleNamespace(input_ids=[5, 6])

            def decode(self, ids, skip_special_tokens=True):
                return "আমি ভালো"

        processor = SimpleNamespace(tokenizer=Tokenizer())
        (tmp_path / "a.wav").write_bytes(b"synthetic")
        example = TrainExample(
            audio="a.wav",
            text="আমি ভালো",
            duration_s=3.0,
            recording_id="r",
            source="s",
            use_timestamps=True,
        )
        ds = WhisperFineTuneDataset(
            [example],
            processor,
            training_config(),
            audio_root=tmp_path,
            max_target_positions=448,
            decoder_start_token_id=1,
        )
        labels = ds._labels_for(example)
        no_ts = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        zero_ts = processor.tokenizer.convert_tokens_to_ids("<|0.00|>")
        assert no_ts not in labels
        assert zero_ts in labels


class TestTrainingRuntime:
    @pytest.fixture
    def runtime(self, tmp_path, monkeypatch):
        audio = tmp_path / "audio"
        audio.mkdir()
        train = make_training_manifest("train", "train")
        dev = make_training_manifest("dev", "dev")
        for manifest in (train, dev):
            (audio / manifest.segments[0].audio).write_bytes(manifest.name.encode())
        state = {}

        def load_processor():
            state["processor"] = make_training_processor()
            return state["processor"]

        def load_model():
            model = SimpleNamespace(
                config=SimpleNamespace(decoder_start_token_id=1, use_cache=True),
                generation_config=SimpleNamespace(),
                encoder=SimpleNamespace(requires_grad=True),
                decoder=SimpleNamespace(requires_grad=True),
            )

            def freeze_encoder():
                model.encoder.requires_grad = False
                events.append(("freeze",))

            model.freeze_encoder = freeze_encoder
            state["model"] = model
            return model

        events, *_ = install_hf_runtime(
            monkeypatch,
            processor_factory=load_processor,
            model_factory=load_model,
            config=SimpleNamespace(max_target_positions=32, decoder_start_token_id=1),
        )

        class Trainer:
            def __init__(self, **kwargs):
                state["trainer"] = kwargs
                events.append(("trainer",))

            def train(self, **kwargs):
                events.append(("train", kwargs))
                return SimpleNamespace(metrics={"train_loss": 0.25})

            def evaluate(self):
                events.append(("evaluate",))
                return {"eval_wer": 0.2, "eval_cer": 0.1}

            def save_model(self, path):
                events.append(("save", path))

        def arguments(**kwargs):
            state["args"] = kwargs
            return SimpleNamespace(**kwargs)

        transformers = sys.modules["transformers"]
        transformers.Seq2SeqTrainer = Trainer
        transformers.Seq2SeqTrainingArguments = arguments
        return TrainingRun(
            audio=audio,
            train=train,
            dev=dev,
            events=events,
            state=state,
            cfg=training_config(
                model_id="synthetic/base",
                output_dir=str(tmp_path / "outputs"),
                optim="adamw_torch",
                fp16=False,
                freeze_encoder=True,
            ),
            registry=synthetic_registry("synthetic"),
        )

    def test_trainer_evaluates_freezes_saves_and_returns_final_metrics(self, runtime):
        metrics = runtime.run()
        assert metrics == {"train_loss": 0.25, "eval_wer": 0.2, "eval_cer": 0.1}
        assert runtime.state["args"]["eval_strategy"] == "steps"
        assert runtime.state["args"]["predict_with_generate"] is True
        assert runtime.state["model"].encoder.requires_grad is False
        assert runtime.state["model"].decoder.requires_grad is True
        assert runtime.state["model"].config.use_cache is False
        assert [event[0] for event in runtime.events][-3:] == ["train", "evaluate", "save"]
        assert runtime.state["trainer"]["processing_class"] is runtime.state["processor"]
        identity = runtime.identity()
        assert identity["manifests"][0]["name"] == "train"
        assert identity["leakage"][0]["eval_total"] == 1

    def test_all_remote_loaders_use_same_persisted_immutable_revision(self, runtime):
        runtime.run()
        loaders = [
            event for event in runtime.events if event[0] in {"processor", "config", "model"}
        ]
        assert len(loaders) == 3
        assert all(event[1] == runtime.cfg.model_id for event in loaders)
        assert all(event[2]["revision"] == MODEL_REVISION for event in loaders)
        identity = runtime.identity()
        assert identity["config"]["model_revision"] == MODEL_REVISION
        assert identity["base_checkpoint"] == {
            "source": "hub",
            "model_id": runtime.cfg.model_id,
            "revision": MODEL_REVISION,
        }

    def test_changed_hub_revision_blocks_resume_before_loaders(self, runtime):
        runtime.run()
        runtime.events.clear()
        runtime.cfg = replace(runtime.cfg, model_revision="b" * 40)
        runtime.rejects(match="different training identity")

    @pytest.mark.parametrize("role", ["train", "dev"])
    def test_changed_audio_bytes_block_resume_before_loaders(self, runtime, role):
        runtime.run()
        manifest = getattr(runtime, role)
        path = runtime.audio / manifest.segments[0].audio
        path.write_bytes(b"changed audio with the same declared duration")
        runtime.events.clear()
        runtime.rejects(match="different training identity")

    def test_training_identity_records_undeclared_audio_hashes(self, runtime):
        runtime.run()
        identity = runtime.identity()
        assert identity["audio_files_sha256"] == {
            f"{name}.wav": hashlib.sha256(name.encode()).hexdigest() for name in ("train", "dev")
        }
        assert runtime.train.segments[0].audio_sha256 is None

    @pytest.mark.parametrize("resume", [False, True])
    def test_training_identity_without_audio_hashes_cannot_verify_restart(self, runtime, resume):
        runtime.run()
        identity = runtime.identity()
        del identity["audio_files_sha256"]
        runtime.save_identity(identity)
        runtime.cfg = replace(runtime.cfg, resume=resume)
        runtime.events.clear()
        runtime.rejects(match="no verifiable audio hashes")

    def test_training_identity_with_complete_declared_hashes_can_be_migrated(self, runtime):
        for role in ("train", "dev"):
            manifest = getattr(runtime, role)
            segment = manifest.segments[0]
            digest = hashlib.sha256((runtime.audio / segment.audio).read_bytes()).hexdigest()
            runtime.change_segment(role, audio_sha256=digest)
        runtime.run()
        identity = runtime.identity()
        del identity["audio_files_sha256"]
        runtime.save_identity(identity)
        runtime.events.clear()
        runtime.run()
        assert any(event[0] == "model" for event in runtime.events)
        assert "audio_files_sha256" in runtime.identity()

    @pytest.mark.parametrize("role", ["train", "dev"])
    @pytest.mark.parametrize("alias", ["symlink", "hardlink", "copy"])
    def test_duplicate_audio_windows_under_other_names_fail_preflight(self, runtime, role, alias):
        manifest = getattr(runtime, role)
        segment = manifest.segments[0]
        original = runtime.audio / segment.audio
        path = runtime.audio / f"{role}-alias.wav"
        if alias == "symlink":
            path.symlink_to(original.name)
        elif alias == "hardlink":
            os.link(original, path)
        else:
            shutil.copyfile(original, path)
        duplicate = replace(segment, id=f"{role}-alias", audio=path.name)
        setattr(runtime, role, replace(manifest, segments=(segment, duplicate)))
        runtime.rejects(match="duplicate.*audio windows")

    def test_distinct_crops_of_one_recording_remain_supported(self, runtime):
        segment = runtime.train.segments[0]
        crop = replace(segment, id="crop", start_s=0.0, duration_s=2.0)
        runtime.train = replace(runtime.train, segments=(segment, crop))
        runtime.run()
        assert len(runtime.state["trainer"]["train_dataset"]) == 2

    @pytest.mark.parametrize("role", ["train", "dev"])
    @pytest.mark.parametrize("alias", [False, True])
    def test_eof_tolerance_cannot_hide_duplicate_decoded_windows(self, runtime, role, alias):
        manifest = getattr(runtime, role)
        segment = manifest.segments[0]
        audio = segment.audio
        if alias:
            audio = f"{role}-copy.wav"
            shutil.copyfile(runtime.audio / segment.audio, runtime.audio / audio)
        duplicate = replace(
            segment, id=f"{role}-duplicate", audio=audio, start_s=0.0, duration_s=3.0005
        )
        setattr(runtime, role, replace(manifest, segments=(segment, duplicate)))
        runtime.rejects(match="duplicate.*audio windows")

    def test_local_checkpoint_loads_offline_and_unchanged_resume_works(self, runtime):
        checkpoint = runtime.local_checkpoint()
        runtime.run()
        loaders = [
            event for event in runtime.events if event[0] in {"processor", "config", "model"}
        ]
        assert all(event[1] == str(checkpoint.resolve()) for event in loaders)
        assert all(event[2].get("local_files_only") is True for event in loaders)
        assert all("revision" not in event[2] for event in loaders)
        identity = runtime.identity()
        assert set(identity["base_checkpoint"]["files_sha256"]) == {
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "preprocessor_config.json",
        }
        runtime.run()

    @pytest.mark.parametrize("file", ["config.json", "model.safetensors", "tokenizer.json"])
    def test_changed_local_checkpoint_blocks_resume_before_loaders(self, runtime, file):
        checkpoint = runtime.local_checkpoint()
        runtime.run()
        runtime.events.clear()
        (checkpoint / file).write_bytes(b"modified")
        runtime.rejects(match="different training identity")

    def test_local_checkpoint_change_between_loaders_blocks_training(self, runtime, monkeypatch):
        checkpoint = runtime.local_checkpoint()
        loader = sys.modules["transformers"].WhisperConfig
        original = loader.from_pretrained

        def load_and_change(*args, **kwargs):
            result = original(*args, **kwargs)
            (checkpoint / "tokenizer.json").write_bytes(b"modified")
            return result

        monkeypatch.setattr(loader, "from_pretrained", load_and_change)
        with pytest.raises(ValueError, match="changed while loading"):
            runtime.run()
        assert [event[0] for event in runtime.events] == ["processor", "config", "model"]
        assert not runtime.output.exists()

    @pytest.mark.parametrize("kind", ["child", "equal", "case-alias"])
    def test_training_output_cannot_overwrite_local_checkpoint(self, runtime, kind):
        checkpoint = runtime.local_checkpoint()
        output = checkpoint / "trained"
        if kind == "equal":
            output = checkpoint
        elif kind == "case-alias":
            output = checkpoint.with_name(checkpoint.name.upper()) / "trained"
        runtime.cfg = replace(runtime.cfg, output_dir=str(output))
        runtime.rejects(match="base checkpoint must be disjoint")

    @pytest.mark.parametrize(
        "kind", ["dev-train", "eval-only", "unknown", "missing", "overlap", "duration", "empty"]
    )
    def test_invalid_inputs_fail_before_pretrained(self, runtime, kind):
        if kind == "dev-train":
            runtime.train = replace(runtime.train, split="dev")
        elif kind == "eval-only":
            runtime.registry = synthetic_registry("synthetic", status="eval-only")
        elif kind == "unknown":
            runtime.registry = synthetic_registry("other")
        elif kind == "missing":
            (runtime.audio / "train.wav").unlink()
        elif kind == "overlap":
            runtime.change_segment("dev", id="train")
        elif kind == "duration":
            runtime.change_segment("train", duration_s=2)
        else:
            runtime.cfg = replace(runtime.cfg, min_chunk_length_s=4)
        runtime.rejects(error=(ValueError, KeyError))

    def test_duration_filtered_overlap_still_blocks(self, runtime):
        extra = replace(runtime.train.segments[0], id="shared", audio="long.wav", duration_s=40)
        runtime.train = replace(runtime.train, segments=(*runtime.train.segments, extra))
        runtime.change_segment("dev", id="shared")
        (runtime.audio / "long.wav").write_bytes(b"fixture")
        with pytest.raises(ValueError, match="hard train/evaluation"):
            runtime.run(duration_of=lambda p: 40 if p.name == "long.wav" else 3)
        assert runtime.events == []

    @pytest.mark.parametrize("kind", ["id", "same-file", "copy"])
    def test_development_and_held_out_data_cannot_share_ids_or_audio(self, runtime, kind):
        original = runtime.dev.segments[0]
        held_out = make_training_manifest("held-out", "test")
        path = runtime.audio / held_out.segments[0].audio
        path.write_bytes(b"held-out recording")
        if kind == "id":
            held_out = replace(held_out, segments=(replace(held_out.segments[0], id=original.id),))
        elif kind == "same-file":
            held_out = replace(
                held_out, segments=(replace(held_out.segments[0], audio=original.audio),)
            )
        else:
            shutil.copyfile(runtime.audio / original.audio, path)
        runtime.rejects(match="hard development/held-out overlap", held_out_manifests=[held_out])

    def test_distinct_held_out_audio_can_share_development_text(self, runtime):
        held_out = make_training_manifest("held-out", "test", text=runtime.dev.segments[0].text)
        (runtime.audio / held_out.segments[0].audio).write_bytes(b"held-out recording")
        runtime.run(held_out_manifests=[held_out])
        assert any(event[0] == "model" for event in runtime.events)

    def test_exact_text_decision_is_saved(self, runtime):
        runtime.change_segment("dev", text="train")
        runtime.run(
            text_decisions={"dev": {"train": "synthetic common reference, distinct recordings"}},
        )
        identity = runtime.identity()
        assert identity["leakage"][0]["reviewed_text"][0]["key"] == "train"

    def test_incompatible_resume_identity_and_missing_identity_fail_before_models(self, runtime):
        runtime.run()
        runtime.events.clear()
        out = runtime.output
        write_checkpoint_state(out, 5)
        runtime.cfg = replace(runtime.cfg, learning_rate=2e-05)
        runtime.rejects(match="different training identity")
        (out / "training-inputs.json").unlink()
        runtime.rejects(match="no declared training identity")

    def test_compatible_resume_passes_saved_step(self, runtime):
        runtime.run()
        runtime.events.clear()
        checkpoint = write_checkpoint_state(runtime.output, 5)
        runtime.run()
        assert ("train", {"resume_from_checkpoint": str(checkpoint)}) in runtime.events

    @pytest.mark.parametrize("legacy_identity", [False, True])
    def test_enabling_resume_preserves_training_identity(self, runtime, legacy_identity):
        runtime.cfg = replace(runtime.cfg, resume=False)
        runtime.run()
        out = runtime.output
        if legacy_identity:
            identity = runtime.identity()
            identity["config"]["resume"] = False
            runtime.save_identity(identity)
        checkpoint = write_checkpoint_state(out, 5)
        runtime.cfg = replace(runtime.cfg, resume=True)
        runtime.events.clear()
        runtime.run()
        assert ("train", {"resume_from_checkpoint": str(checkpoint)}) in runtime.events
        assert "resume" not in runtime.identity()["config"]

    def test_failed_identity_publication_preserves_previous_record(self, runtime, monkeypatch):
        from shono import output_paths

        runtime.run()
        path = runtime.identity_path
        original = path.read_bytes()
        files = set(runtime.output.iterdir())
        replace_file = output_paths.os.replace

        def fail_identity(source, destination):
            if Path(destination) == path:
                raise OSError("interrupted identity publication")
            return replace_file(source, destination)

        monkeypatch.setattr(output_paths.os, "replace", fail_identity)
        with pytest.raises(OSError, match="interrupted identity publication"):
            runtime.run()
        assert path.read_bytes() == original
        assert set(runtime.output.iterdir()) == files

    def test_prior_training_identity_rejects_duplicate_json_keys(self, runtime):
        runtime.run()
        identity_path = runtime.identity_path
        original = identity_path.read_text()
        identity_path.write_text('{"config": {}, ' + original.lstrip()[1:])
        runtime.events.clear()
        runtime.rejects(match="cannot read prior training identity")

    @pytest.mark.parametrize("change", ["config", "manifest", "checkpoint"])
    @pytest.mark.parametrize("resume", [False, True])
    def test_legacy_resume_record_still_requires_matching_training_inputs(
        self, runtime, change, resume
    ):
        runtime.cfg = replace(runtime.cfg, resume=False)
        runtime.run()
        identity = runtime.identity()
        identity["config"]["resume"] = False
        runtime.save_identity(identity)
        runtime.cfg = replace(runtime.cfg, resume=resume)
        if change == "config":
            runtime.cfg = replace(runtime.cfg, learning_rate=2e-05)
        elif change == "manifest":
            runtime.change_segment("dev", text="changed reference")
        else:
            runtime.cfg = replace(runtime.cfg, model_revision="b" * 40)
        runtime.events.clear()
        runtime.rejects(match="different training identity")

    def test_disabling_resume_starts_fresh_with_matching_training_inputs(self, runtime):
        runtime.run()
        write_checkpoint_state(runtime.output, 5)
        runtime.cfg = replace(runtime.cfg, resume=False)
        runtime.events.clear()
        runtime.run()
        assert ("train", {"resume_from_checkpoint": None}) in runtime.events

    @pytest.mark.parametrize("role", ["train", "dev"])
    @pytest.mark.parametrize("target", ["...", "।!?+", "\u200b\u200c\u200d\ufeff"])
    def test_unscorable_targets_fail_before_model_loading(self, runtime, role, target):
        runtime.change_segment(role, text=target)
        runtime.rejects(match="empty after evaluation normalization")

    @pytest.mark.parametrize("kind", ["inside", "ancestor", "hardlink", "symlink", "cycle"])
    def test_outputs_cannot_alias_inputs(self, runtime, kind):
        out = runtime.output
        if kind == "inside":
            runtime.cfg = replace(runtime.cfg, output_dir=str(runtime.audio / "outputs"))
        elif kind == "ancestor":
            runtime.cfg = replace(runtime.cfg, output_dir=str(runtime.audio.parent))
        elif kind == "cycle":
            out.symlink_to(out.name)
            runtime.cfg = replace(runtime.cfg, output_dir=str(out))
        else:
            out.mkdir()
            target = out / "training-inputs.json"
            if kind == "hardlink":
                os.link(runtime.audio / "train.wav", target)
            else:
                target.symlink_to(runtime.audio / "train.wav")
        with pytest.raises(ValueError):
            runtime.run()
        assert runtime.events == [] and (runtime.audio / "train.wav").read_bytes() == b"train"

    def test_label_capacity_rejects_before_model_weights(self, runtime, monkeypatch):
        monkeypatch.setattr(FakeTrainingTokenizer, "text_length", 40)
        runtime.rejects(match="model capacity", events=("processor", "config"))

    @pytest.mark.parametrize("role", ["train", "dev"])
    @pytest.mark.parametrize("target", ["<|endoftext|>", "<|notimestamps|>", "<|0.00|>"])
    def test_targets_removed_by_whisper_decoding_fail_before_model_weights(
        self, runtime, monkeypatch, role, target
    ):
        original = FakeTrainingTokenizer.__call__

        def tokenize(tokenizer, text, add_special_tokens=True):
            if text == target:
                prefix = tokenizer.prefix_tokens if add_special_tokens else []
                return SimpleNamespace(input_ids=[*prefix, tokenizer.eos_token_id])
            return original(tokenizer, text, add_special_tokens)

        monkeypatch.setattr(FakeTrainingTokenizer, "__call__", tokenize)
        runtime.change_segment(role, text=target)
        runtime.rejects(match="empty after Whisper decoding", events=("processor", "config"))

    def test_masks_padding_bos_and_capacity(self, tmp_path, monkeypatch):
        from shono.train import audio as module

        (tmp_path / "a.wav").write_bytes(b"fixture")
        example = TrainExample("a.wav", "বাক্য", 3, "r", "synthetic", use_timestamps=True)
        cfg = TrainConfig(
            model_id="synthetic", model_revision=MODEL_REVISION, output_dir=str(tmp_path / "out")
        )
        proc = make_training_processor()
        ds = WhisperFineTuneDataset(
            [example],
            proc,
            cfg,
            audio_root=tmp_path,
            max_target_positions=7,
            decoder_start_token_id=1,
        )
        monkeypatch.setattr(module, "_load_audio", lambda *args: [0.1, 0.2])
        item = ds[0]
        assert item["attention_mask"] == [1, 1, 0] and 9 not in item["labels"]
        batch = DataCollatorSpeechSeq2Seq(proc, 1)(
            [item, dict(item, labels=[1, 5, 4], attention_mask=[1, 0, 0])]
        )
        assert batch["attention_mask"] == [[1, 1, 0], [1, 0, 0]]
        assert batch["labels"].rows[1] == [5, 4, -100, -100, -100, -100, -100]
        ds.max_target_positions = 6
        with pytest.raises(ValueError, match="model capacity"):
            ds.validate_labels()

    @pytest.mark.parametrize(
        "field,value",
        [
            ("learning_rate", float("nan")),
            ("learning_rate", True),
            ("weight_decay", float("inf")),
            ("warmup_steps", -1),
            ("save_steps", False),
            ("eval_steps", 0),
            ("seed", -1),
            ("freeze_encoder", 1),
            ("per_device_batch_size", 10**1000),
        ],
    )
    def test_training_config_rejects_invalid_numbers(self, field, value):
        with pytest.raises(ValueError):
            TrainConfig(
                model_id="synthetic",
                model_revision=MODEL_REVISION,
                output_dir="out",
                **{field: value},
            )

    def test_selection_requires_role_license_unique_inputs(self, runtime):
        kwargs = dict(
            chunk_length_s=28,
            min_chunk_length_s=1,
            timestamp_sample_fraction=0,
            seed=42,
            registry=runtime.registry,
        )
        with pytest.raises(ValueError):
            build_examples([runtime.dev], **kwargs)
        assert build_examples([runtime.dev], role="eval", **kwargs)[0].split == "dev"
        with pytest.raises(ValueError, match="distinct"):
            build_examples([runtime.train, runtime.train], **kwargs)

    @pytest.mark.parametrize(
        "directory", ["src", "docs", "data", "tests", "notebooks", "scripts", "reports"]
    )
    def test_repository_source_outputs_fail_before_model_calls(self, runtime, directory):

        from shono.train import output

        repo = Path(output.__file__).resolve().parents[3]
        runtime.cfg = replace(runtime.cfg, output_dir=str(repo / directory / "synthetic-output"))
        runtime.rejects(match="source subtrees")


class TestHallucinationFiltering:
    def test_low_logprob_chunk_is_dropped(self):
        result = filter_hallucinations([confidence_chunk(avg_logprob=-2.0)])
        assert result.n_dropped == 1
        assert "avg_logprob" in result.dropped[0].reasons[0]

    def test_repetitive_chunk_dropped_by_compression_ratio(self):
        result = filter_hallucinations([confidence_chunk(compression_ratio=3.0)])
        assert result.n_dropped == 1

    def test_silence_chunk_dropped_by_no_speech_prob(self):
        result = filter_hallucinations([confidence_chunk(no_speech_prob=0.9)])
        assert result.n_dropped == 1

    def test_clean_chunk_kept_and_reported(self):
        result = filter_hallucinations([confidence_chunk()])
        assert result.n_dropped == 0
        assert "kept all 1" in result.summary()

    def test_custom_thresholds_respected(self):
        cfg = HallucinationConfig(min_avg_logprob=-3.0)
        assert filter_hallucinations([confidence_chunk(avg_logprob=-2.0)], cfg).n_dropped == 0


class TestRealTimeFactor:
    def test_real_time_factor(self):
        assert real_time_factor(30.0, 60.0) == 0.5
        with pytest.raises(ValueError):
            real_time_factor(1.0, 0.0)


class TestOutputPaths:
    @pytest.fixture
    def git_directory(self, tmp_path):
        directory = tmp_path / "project" / ".git"
        directory.mkdir(parents=True)
        (directory / "HEAD").write_text("ref: refs/heads/main\n")
        return directory

    @pytest.mark.parametrize(
        "location", [".git/HEAD", ".git/reports/result.json", ".GIT/result.json", "alias/HEAD"]
    )
    def test_file_destinations_reject_git_metadata(self, git_directory, location):
        from shono.output_paths import validate_output_destinations

        project = git_directory.parent
        (project / "alias").symlink_to(git_directory, target_is_directory=True)
        original = (git_directory / "HEAD").read_bytes()
        with pytest.raises(ValueError, match="Git metadata"):
            validate_output_destinations((project / location,))
        assert (git_directory / "HEAD").read_bytes() == original
        assert set(git_directory.iterdir()) == {git_directory / "HEAD"}

    def test_git_directory_symlink_cannot_hide_metadata(self, tmp_path):
        from shono.output_paths import validate_output_destinations

        metadata = tmp_path / "metadata"
        metadata.mkdir()
        git = tmp_path / ".git"
        git.symlink_to(metadata, target_is_directory=True)
        with pytest.raises(ValueError, match="Git metadata"):
            validate_output_destinations((git / "result.json",))
        assert list(metadata.iterdir()) == []

    @pytest.mark.parametrize("subdirectory", ["", "models"])
    def test_forced_conversion_rejects_git_metadata_before_loading(
        self, tmp_path, monkeypatch, git_directory, subdirectory
    ):
        checkpoint = write_conversion_checkpoint(tmp_path)
        calls = install_fake_converter(monkeypatch)
        with pytest.raises(ValueError, match="Git metadata"):
            to_ct2(checkpoint, git_directory / subdirectory, force=True)
        assert calls == []
        assert (git_directory / "HEAD").read_text() == "ref: refs/heads/main\n"

    @pytest.mark.parametrize("subdirectory", ["", "training"])
    def test_training_output_rejects_git_metadata(self, tmp_path, git_directory, subdirectory):
        from shono.train.output import validate_training_output

        audio = tmp_path / "audio"
        audio.mkdir()
        with pytest.raises(ValueError, match="Git metadata"):
            validate_training_output(git_directory / subdirectory, audio)
        assert (git_directory / "HEAD").read_text() == "ref: refs/heads/main\n"


class TestLinkedSourceProtection:
    @pytest.fixture
    def inputs(self, tmp_path, monkeypatch):
        import shono.train.output as training
        import shono.transcribe.__main__ as cli
        import shono.transcribe.convert as conversion

        project = tmp_path / "project"
        package = project / "src" / "shono"
        for name in ("train", "transcribe"):
            (package / name).mkdir(parents=True)
        mounted = tmp_path / "mounted-data"
        (mounted / "nested").mkdir(parents=True)
        reference = mounted / "nested" / "reference.json"
        reference.write_text('{"reference":"original"}')
        (mounted / "loop").symlink_to(mounted, target_is_directory=True)
        (project / "data").symlink_to(mounted, target_is_directory=True)
        model = write_conversion_checkpoint(tmp_path)
        (model / "model.bin").write_bytes(b"converted weights")
        audio_root = tmp_path / "audio"
        audio_root.mkdir()
        audio = write_wav(audio_root / "recording.wav", sample_rate=10)
        for module, relative in (
            (cli, "transcribe/__main__.py"),
            (conversion, "transcribe/convert.py"),
            (training, "train/output.py"),
        ):
            monkeypatch.setattr(module, "__file__", str(package / relative))
        return SimpleNamespace(
            project=project, mounted=mounted, reference=reference, model=model, audio=audio
        )

    @pytest.mark.parametrize("resolved", [False, True])
    def test_json_export_rejects_linked_input_tree_before_inference(
        self, inputs, resolved, monkeypatch
    ):
        import shono.transcribe.pipeline as pipeline

        monkeypatch.setattr(pipeline, "SileroVAD", lambda: pytest.fail("VAD must stay unloaded"))
        target = inputs.project / "data" / "nested" / "reference.json"
        if resolved:
            target = target.resolve()
        assert main([str(inputs.audio), "--model", str(inputs.model), "--json", str(target)]) == 1
        assert inputs.reference.read_text() == '{"reference":"original"}'

    @pytest.mark.parametrize("resolved", [False, True])
    def test_conversion_rejects_nested_linked_input_tree_before_loading(
        self, inputs, resolved, monkeypatch
    ):
        calls = install_fake_converter(monkeypatch)
        output = inputs.project / "data" / "nested"
        if resolved:
            output = output.resolve()
        with pytest.raises(ValueError, match="repository sources"):
            to_ct2(inputs.model, output, force=True)
        assert calls == []
        assert inputs.reference.read_text() == '{"reference":"original"}'

    @pytest.mark.parametrize("resolved", [False, True])
    def test_training_rejects_nested_linked_source_tree(self, inputs, resolved):
        from shono.train.output import validate_training_output

        output = inputs.project / "data" / "nested" / "trained"
        if resolved:
            output = output.resolve()
        with pytest.raises(ValueError, match="source subtrees"):
            validate_training_output(output, inputs.audio.parent)
        assert not output.exists()

    @pytest.mark.parametrize("location", ["outputs", "outside"])
    def test_linked_inputs_allow_normal_generated_destinations(self, inputs, location):
        from shono.train.output import validate_training_output
        from shono.transcribe.__main__ import _output_path
        from shono.transcribe.convert import _conversion_paths

        root = (
            inputs.project / "outputs"
            if location == "outputs"
            else inputs.project.parent / "outside"
        )
        target = root / "transcript.json"
        assert _output_path(target, inputs.audio, inputs.model) == target.resolve()
        assert (
            _conversion_paths(inputs.model, root / "converted", False)[1]
            == (root / "converted").resolve()
        )
        assert (
            validate_training_output(root / "trained", inputs.audio.parent)
            == (root / "trained").resolve()
        )

    @pytest.mark.parametrize("operation", ["json", "conversion", "training"])
    def test_cyclic_source_roots_fail_cleanly(self, inputs, operation):
        from shono.train.output import validate_training_output
        from shono.transcribe.__main__ import _output_path
        from shono.transcribe.convert import _conversion_paths

        (inputs.project / "docs").symlink_to("docs", target_is_directory=True)
        output = inputs.project / "outputs" / "result"
        operations = {
            "json": lambda: _output_path(output.with_suffix(".json"), inputs.audio, inputs.model),
            "conversion": lambda: _conversion_paths(inputs.model, output, False),
            "training": lambda: validate_training_output(output, inputs.audio.parent),
        }
        with pytest.raises(ValueError, match="resolve|validate conversion paths"):
            operations[operation]()

    def test_source_discovery_propagates_unreadable_directories(self, tmp_path, monkeypatch):
        from shono.output_paths import source_files

        assert source_files(tmp_path / "missing") == ()

        def unreadable(base, *, followlinks, onerror):
            assert followlinks is False
            onerror(PermissionError("unreadable source directory"))

        monkeypatch.setattr("shono.output_paths.os.walk", unreadable)
        with pytest.raises(PermissionError, match="unreadable source directory"):
            source_files(tmp_path)
