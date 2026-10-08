"""Long-form pipeline: chunk at silence, drop hallucinations, merge to global time."""

import pytest

from shono.transcribe import (
    ChunkTranscription,
    HallucinationConfig,
    LongFormTranscriber,
    SpeechSegment,
    Word,
    filter_hallucinations,
    merge_transcriptions,
    plan_chunks,
    real_time_factor,
)


def test_speech_segment_rejects_nonpositive_duration():
    with pytest.raises(ValueError, match="end > start"):
        SpeechSegment(5.0, 5.0)


def test_chunks_pack_until_max_then_cut_at_silence():
    segs = [SpeechSegment(0, 10), SpeechSegment(11, 20), SpeechSegment(21, 40)]
    chunks = plan_chunks(segs, max_chunk_s=25.0)
    assert [(c.start_s, c.end_s) for c in chunks] == [(0, 20), (21, 40)]
    assert not any(c.forced_split for c in chunks)


def test_single_segment_longer_than_max_is_force_split():
    chunks = plan_chunks([SpeechSegment(0, 70)], max_chunk_s=28.0)
    assert len(chunks) == 3
    assert all(c.forced_split for c in chunks)
    assert chunks[0].start_s == 0 and chunks[-1].end_s == pytest.approx(70)


def test_plan_chunks_sorts_unordered_input():
    segs = [SpeechSegment(30, 40), SpeechSegment(0, 10)]
    chunks = plan_chunks(segs, max_chunk_s=100.0)
    assert chunks[0].start_s == 0


def test_pad_extends_windows_without_going_negative():
    chunks = plan_chunks([SpeechSegment(0, 5)], max_chunk_s=28.0, pad_s=0.5)
    assert chunks[0].start_s == 0.0
    assert chunks[0].end_s == pytest.approx(5.5)


def test_empty_input_yields_no_chunks():
    assert plan_chunks([], max_chunk_s=28.0) == []


def test_nested_segment_does_not_shrink_coverage():
    chunks = plan_chunks([SpeechSegment(0, 20), SpeechSegment(5, 10)], max_chunk_s=25.0)
    assert [(c.start_s, c.end_s) for c in chunks] == [(0, 20)]


def _chunk(text="কথা", logprob=-0.2, comp=1.5, nosp=0.1, **kw):
    return ChunkTranscription(
        window_start_s=kw.get("start", 0.0),
        window_end_s=kw.get("end", 5.0),
        text=text,
        avg_logprob=logprob,
        compression_ratio=comp,
        no_speech_prob=nosp,
        words=kw.get("words", ()),
    )


def test_low_logprob_chunk_is_dropped():
    result = filter_hallucinations([_chunk(logprob=-2.0)])
    assert result.n_dropped == 1
    assert "avg_logprob" in result.dropped[0].reasons[0]


def test_repetitive_chunk_dropped_by_compression_ratio():
    result = filter_hallucinations([_chunk(comp=3.0)])
    assert result.n_dropped == 1


def test_silence_chunk_dropped_by_no_speech_prob():
    result = filter_hallucinations([_chunk(nosp=0.9)])
    assert result.n_dropped == 1


def test_clean_chunk_kept_and_reported():
    result = filter_hallucinations([_chunk()])
    assert result.n_dropped == 0
    assert "kept all 1" in result.summary()


def test_custom_thresholds_respected():
    cfg = HallucinationConfig(min_avg_logprob=-3.0)
    assert filter_hallucinations([_chunk(logprob=-2.0)], cfg).n_dropped == 0


def test_merge_shifts_words_to_global_time():
    chunk = ChunkTranscription(10.0, 15.0, "শব্দ", words=(Word(1.0, 2.0, "শব্দ"),))
    transcript = merge_transcriptions([chunk])
    seg = transcript.segments[0]
    assert seg.start_s == pytest.approx(11.0) and seg.end_s == pytest.approx(12.0)
    assert transcript.text == "শব্দ"


def test_merge_dedupes_overlapping_boundary_words():
    a = ChunkTranscription(0.0, 10.0, "", words=(Word(8.0, 9.0, "শেষ"),))
    b = ChunkTranscription(8.0, 18.0, "", words=(Word(0.5, 1.0, "শেষ"), Word(2.0, 3.0, "নতুন")))
    transcript = merge_transcriptions([a, b])
    assert transcript.text == "শেষ নতুন"


def test_merge_wordless_chunks_concatenate_in_time_order():
    a = ChunkTranscription(0.0, 5.0, "প্রথম")
    b = ChunkTranscription(5.0, 10.0, "দ্বিতীয়")
    transcript = merge_transcriptions([b, a])
    assert transcript.text == "প্রথম দ্বিতীয়"


def test_merge_skips_wordless_chunk_fully_covered():
    a = ChunkTranscription(0.0, 10.0, "সব")
    covered = ChunkTranscription(2.0, 8.0, "নকল")
    assert merge_transcriptions([a, covered]).text == "সব"


def test_merge_wordless_overlap_keeps_new_tail_with_warning():
    a = ChunkTranscription(0.0, 10.0, "শেষ", words=(Word(8.0, 9.0, "শেষ"),))
    b = ChunkTranscription(8.0, 18.0, "শেষ নতুন")
    result = merge_transcriptions([a, b])
    assert result.text == "শেষ শেষ নতুন"
    assert result.warnings
    assert result.segments[-1].timing_precision == "chunk"


def test_real_time_factor():
    assert real_time_factor(30.0, 60.0) == 0.5
    with pytest.raises(ValueError):
        real_time_factor(1.0, 0.0)


class _FakeVAD:
    def __init__(self, segments):
        self._segments = segments

    def detect(self, audio_path):
        return self._segments


class _FakeTranscriber:
    """Returns a canned transcription per window; one window is a hallucination."""

    def transcribe_chunk(self, audio_path, start_s, end_s):
        if start_s >= 100:
            return ChunkTranscription(
                start_s, end_s, "নকল", no_speech_prob=0.95, words=(Word(0.0, 1.0, "নকল"),)
            )
        return ChunkTranscription(start_s, end_s, "ভালো", words=(Word(0.0, 1.0, "ভালো"),))


def test_pipeline_end_to_end_with_fakes():
    vad = _FakeVAD([SpeechSegment(0, 20), SpeechSegment(100, 105)])
    pipeline = LongFormTranscriber(vad, _FakeTranscriber(), max_chunk_s=28.0, pad_s=0.0)
    clock = iter([0.0, 12.0])
    result = pipeline.transcribe("rec.wav", audio_duration_s=200.0, now=lambda: next(clock))

    assert result.n_chunks == 2
    assert result.filtering.n_dropped == 1
    assert result.transcript.text == "ভালো"
    assert result.rtf == pytest.approx(12.0 / 200.0)
    assert result.rtf < 1.0


def test_long_form_pipeline_transcriber_returns_merged_text(monkeypatch):
    from shono.transcribe import LongFormPipelineTranscriber

    pipeline = LongFormTranscriber(
        _FakeVAD([SpeechSegment(0, 20), SpeechSegment(100, 105)]),
        _FakeTranscriber(),
        pad_s=0.0,
    )
    adapter = LongFormPipelineTranscriber(pipeline)
    monkeypatch.setattr("shono.transcribe.pipeline.validate_audio_window", lambda *a, **kw: 200.0)
    text = adapter.transcribe("rec.wav", duration_s=200.0)
    assert text == "ভালো"


def test_cli_missing_audio_exits_2_without_loading_torch():
    from shono.transcribe.__main__ import main

    assert main(["/no/such/file.wav", "--model", "/some/ct2"]) == 2


def test_short_form_runtime_is_tested_without_model_downloads():
    from shono.transcribe import ShortFormWhisperTranscriber

    adapter = ShortFormWhisperTranscriber("local-checkpoint", device="cpu")
    assert adapter.observed_runtime_context()["generation_config"] == "unknown"
