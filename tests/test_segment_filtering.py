"""Confidence guards apply to decoder segments before transcript merging."""

import sys
from types import SimpleNamespace as NS

import pytest

from shono.transcribe import (
    FasterWhisperTranscriber,
    HallucinationConfig,
    LongFormTranscriber,
    SpeechSegment,
)


def decoder_segment(start, text, *, timed=True, **scores):
    return NS(
        start=start,
        end=start + 1,
        text=text,
        words=[NS(start=start, end=start + 1, word=text, probability=0.9)] if timed else None,
        avg_logprob=scores.get("avg_logprob", -0.1),
        compression_ratio=scores.get("compression_ratio", 1.0),
        no_speech_prob=scores.get("no_speech_prob", 0.0),
    )


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setattr("shono.transcribe.pipeline.validate_audio_window", lambda *a: 4)
    monkeypatch.setitem(sys.modules, "librosa", NS(load=lambda *a, **kw: ([0], 16000)))
    monkeypatch.setitem(sys.modules, "numpy", NS(isfinite=lambda audio: NS(all=lambda: True)))

    def make(segments, **kwargs):
        decoder = FasterWhisperTranscriber("local-checkpoint", device="cpu", compute_type="int8")
        monkeypatch.setattr(
            decoder, "_load", lambda: NS(transcribe=lambda *a, **kw: (iter(segments), None))
        )
        vad = NS(detect=lambda path: [SpeechSegment(10, 14)])
        return LongFormTranscriber(vad, decoder, pad_s=0, **kwargs)

    return make


@pytest.mark.parametrize(
    "metric,value",
    [("avg_logprob", -1.8), ("compression_ratio", 3.5), ("no_speech_prob", 0.9)],
)
@pytest.mark.parametrize("bad_first", [True, False])
@pytest.mark.parametrize("timed", [True, False])
@pytest.mark.parametrize("bad_timed", [True, False])
def test_good_segment_cannot_dilute_a_failing_segment(
    pipeline, metric, value, bad_first, timed, bad_timed
):
    bad_start, good_start = (0, 2) if bad_first else (2, 0)
    bad = decoder_segment(bad_start, "নকল", timed=bad_timed, **{metric: value})
    good = decoder_segment(good_start, "ভালো", timed=timed)
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


def test_custom_thresholds_apply_to_each_segment(pipeline):
    segments = [decoder_segment(0, "নকল", avg_logprob=-0.8), decoder_segment(2, "ভালো")]
    strict = HallucinationConfig(min_avg_logprob=-0.5)
    result = pipeline(segments, hallucination=strict).transcribe("recording.wav", 20)
    assert result.transcript.text == "ভালো"
    assert result.filtering.n_dropped == 1

    relaxed = HallucinationConfig(min_avg_logprob=-2)
    segments[0].avg_logprob = -1.8
    result = pipeline(segments, hallucination=relaxed).transcribe("recording.wav", 20)
    assert result.transcript.text == "নকল ভালো"
    assert result.filtering.n_dropped == 0


def test_all_failing_segments_leave_an_empty_transcript(pipeline):
    segments = [
        decoder_segment(0, "এক", avg_logprob=-2),
        decoder_segment(2, "দুই", compression_ratio=4),
    ]
    result = pipeline(segments).transcribe("recording.wav", 20)
    assert result.transcript.text == ""
    assert result.filtering.n_dropped == 2
    assert result.filtering.kept == ()


def test_no_decoder_segments_does_not_count_an_accepted_segment(pipeline):
    result = pipeline([]).transcribe("recording.wav", 20)
    assert result.n_chunks == 1
    assert result.transcript.segments == ()
    assert result.filtering.kept == ()
    assert result.filtering.n_dropped == 0


def test_word_alignment_outside_segment_bounds_keeps_correct_offsets(pipeline):
    segment = decoder_segment(1, "ভালো")
    segment.start, segment.end = 1.2, 1.8
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
def test_invalid_decoder_bounds_are_rejected(pipeline, field, value):
    segment = decoder_segment(0, "ভালো")
    if field.startswith("word_"):
        setattr(segment.words[0], field.removeprefix("word_"), value)
    else:
        setattr(segment, field, value)
    with pytest.raises(ValueError):
        pipeline([segment]).transcribe("recording.wav", 20)
