"""Regression checks for overlapping transcription content."""

import hashlib
import json
import sys
from types import SimpleNamespace as NS

import pytest

from shono.api import DeepgramTranscriber, GoogleChirpTranscriber
from shono.transcribe import (
    ChunkTranscription,
    FasterWhisperTranscriber,
    HallucinationConfig,
    LongFormTranscriber,
    ShortFormWhisperTranscriber,
    Word,
    merge_transcriptions,
)


def test_timed_word_in_an_earlier_segment_gap_is_retained():
    first = ChunkTranscription(0, 10, "one three", words=(Word(0, 1, "one"), Word(9, 10, "three")))
    second = ChunkTranscription(2, 8, "two", words=(Word(2, 3, "two"),))

    result = merge_transcriptions([first, second])

    assert result.text == "one two three"
    assert any(segment.start_s == 4 and segment.end_s == 5 for segment in result.segments)
    assert result.warnings


def test_distinct_word_with_covered_timing_is_retained_with_warning():
    first = ChunkTranscription(0, 2, "one", words=(Word(0, 2, "one"),))
    second = ChunkTranscription(0.5, 2, "two", words=(Word(0, 1, "two"),))

    result = merge_transcriptions([first, second])

    assert result.text == "one two"
    assert result.warnings


def test_matching_word_inside_prior_word_timing_is_removed():
    first = ChunkTranscription(0, 2, "one", words=(Word(0, 2, "one"),))
    second = ChunkTranscription(0.5, 2, "one", words=(Word(0, 1, "one"),))

    assert merge_transcriptions([first, second]).text == "one"


def test_repeated_word_at_a_later_time_is_retained():
    first = ChunkTranscription(0, 2, "one", words=(Word(0, 1, "one"),))
    second = ChunkTranscription(0.5, 3, "one", words=(Word(1, 2, "one"),))

    assert merge_transcriptions([first, second]).text == "one one"


def test_distinct_untimed_text_inside_a_prior_window_is_retained_with_warning():
    first = ChunkTranscription(0, 10, "first")
    second = ChunkTranscription(2, 8, "second")

    result = merge_transcriptions([first, second])

    assert result.text == "first second"
    assert result.warnings


def test_matching_untimed_text_inside_a_prior_window_is_removed():
    first = ChunkTranscription(0, 10, "first")
    second = ChunkTranscription(2, 8, "first")

    assert merge_transcriptions([first, second]).text == "first"


def test_equal_start_word_order_stays_in_decoder_order():
    chunk = ChunkTranscription(0, 3, "one two", words=(Word(0, 2, "one"), Word(0, 1, "two")))

    assert merge_transcriptions([chunk]).text == "one two"


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "checkpoint"
    path.mkdir()
    (path / "model.bin").write_bytes(b"weights")
    for name in ("config.json", "tokenizer.json", "preprocessor_config.json"):
        (path / name).write_text("{}")
    return path


def test_ct2_observer_records_loaded_files_and_does_not_rehash_mutable_disk(
    checkpoint, monkeypatch
):
    model = NS(model=NS(compute_type="int8_float32"))
    monkeypatch.setitem(sys.modules, "faster_whisper", NS(WhisperModel=lambda *a, **kw: model))
    adapter = FasterWhisperTranscriber(str(checkpoint), device="cpu", compute_type="int8")

    assert adapter.observed_runtime_context()["checkpoint"] == "unknown"
    assert adapter._load() is model
    metadata = adapter.observed_runtime_context()
    assert (
        metadata["checkpoint"]["files_sha256"]["model.bin"]
        == hashlib.sha256(b"weights").hexdigest()
    )
    assert metadata["loaded_compute_type"] == "int8_float32"
    (checkpoint / "model.bin").write_bytes(b"replacement")
    assert adapter._load() is model
    assert adapter.observed_runtime_context() == metadata
    metadata["checkpoint"]["files_sha256"]["model.bin"] = "changed by caller"
    assert (
        adapter.observed_runtime_context()["checkpoint"]["files_sha256"]["model.bin"]
        == hashlib.sha256(b"weights").hexdigest()
    )
    json.dumps(adapter.observed_runtime_context(), allow_nan=False)


def test_checkpoint_changed_during_load_is_rejected_without_caching(checkpoint, monkeypatch):
    def load(*args, **kwargs):
        (checkpoint / "config.json").write_text('{"changed": true}')
        return object()

    monkeypatch.setitem(sys.modules, "faster_whisper", NS(WhisperModel=load))
    adapter = FasterWhisperTranscriber(str(checkpoint), device="cpu", compute_type="int8")
    with pytest.raises(ValueError, match="checkpoint changed while loading"):
        adapter._load()
    assert adapter._model is None
    assert adapter.observed_runtime_context()["checkpoint"] == "unknown"


def test_pipeline_observer_records_decoder_and_actual_filter_thresholds():
    decoder = NS(observed_runtime_context=lambda: {"model": "fixture"})
    guard = HallucinationConfig(min_avg_logprob=-0.5, max_compression_ratio=2)
    pipeline = LongFormTranscriber(NS(), decoder, hallucination=guard)

    metadata = pipeline.observed_runtime_context()

    assert metadata["decoder"] == {"model": "fixture"}
    assert metadata["hallucination"] == {
        "min_avg_logprob": -0.5,
        "max_compression_ratio": 2,
        "max_no_speech_prob": 0.6,
    }


@pytest.mark.parametrize(
    "adapter,model,language",
    [
        (
            DeepgramTranscriber("private-key", model="selected-model", language="bn"),
            "selected-model",
            "bn",
        ),
        (
            GoogleChirpTranscriber("private-project", model="chirp_2", language="bn-BD"),
            "chirp_2",
            "bn-BD",
        ),
    ],
)
def test_provider_observer_records_requested_model_without_credentials(adapter, model, language):
    metadata = adapter.observed_runtime_context()
    assert metadata["model"] == model
    assert metadata["language"] == language
    assert metadata["model_revision"] == "unknown"
    assert "private" not in json.dumps(metadata)


def hf_runtime(monkeypatch, *, revision="a" * 40, during_processor=None):
    calls = []
    config = NS(_commit_hash=revision)
    model = NS(
        config=config,
        generation_config=NS(forced_decoder_ids=None, to_dict=lambda: {}),
        to=lambda device: None,
        eval=lambda: None,
    )
    processor = object()

    def load_config(source, **kwargs):
        calls.append(("config", source, kwargs))
        return config

    def load_model(source, **kwargs):
        calls.append(("model", source, kwargs))
        return model

    def load_processor(source, **kwargs):
        calls.append(("processor", source, kwargs))
        if during_processor:
            during_processor()
        return processor

    monkeypatch.setitem(sys.modules, "torch", NS(cuda=NS(is_available=lambda: False)))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        NS(
            WhisperConfig=NS(from_pretrained=load_config),
            WhisperForConditionalGeneration=NS(from_pretrained=load_model),
            WhisperProcessor=NS(from_pretrained=load_processor),
        ),
    )
    return calls, config, model, processor


def test_shortform_pins_weights_generation_and_processor_to_the_resolved_config(monkeypatch):
    calls, config, model, processor = hf_runtime(monkeypatch)
    adapter = ShortFormWhisperTranscriber("owner/mutable-model")

    assert adapter._load() == (model, processor)

    assert [name for name, _, _ in calls] == ["config", "model", "processor"]
    assert calls[1][2] == {"config": config, "revision": "a" * 40}
    assert calls[2][2] == {"revision": "a" * 40}
    assert adapter.observed_runtime_context()["checkpoint"] == {
        "source": "hub",
        "model_id": "owner/mutable-model",
        "revision": "a" * 40,
    }


def test_shortform_local_processor_and_weights_load_offline_and_share_recorded_hashes(
    checkpoint, monkeypatch
):
    calls, config, model, processor = hf_runtime(monkeypatch, revision=None)
    adapter = ShortFormWhisperTranscriber(str(checkpoint))

    assert adapter._load() == (model, processor)

    assert all(call[2]["local_files_only"] is True for call in calls)
    metadata = adapter.observed_runtime_context()
    assert (
        metadata["checkpoint"]["files_sha256"]["model.bin"]
        == hashlib.sha256(b"weights").hexdigest()
    )
    (checkpoint / "model.bin").write_bytes(b"changed")
    assert adapter._load() == (model, processor)
    assert adapter.observed_runtime_context() == metadata


def test_shortform_rejects_local_checkpoint_changed_while_loading_processor(
    checkpoint, monkeypatch
):
    hf_runtime(
        monkeypatch,
        revision=None,
        during_processor=lambda: (checkpoint / "tokenizer.json").write_text('{"changed": true}'),
    )
    adapter = ShortFormWhisperTranscriber(str(checkpoint))

    with pytest.raises(ValueError, match="checkpoint changed while loading"):
        adapter._load()

    assert adapter._model is None and adapter._processor is None


@pytest.mark.parametrize("revision", [None, "main", "a" * 7, "A" * 40])
def test_shortform_requires_an_immutable_resolved_hub_revision(monkeypatch, revision):
    calls, _, _, _ = hf_runtime(monkeypatch, revision=revision)
    adapter = ShortFormWhisperTranscriber("owner/model")

    with pytest.raises(ValueError, match="no immutable commit SHA"):
        adapter._load()

    assert [name for name, _, _ in calls] == ["config"]
    assert adapter._model is None and adapter._processor is None


def test_missing_local_checkpoint_rejects_before_loading_optional_runtime(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    adapter = ShortFormWhisperTranscriber(str(tmp_path / "missing"))

    with pytest.raises(ValueError, match="checkpoint directory does not exist"):
        adapter._load()
