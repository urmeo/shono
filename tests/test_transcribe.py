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

# ---- SpeechSegment -------------------------------------------------------


def test_speech_segment_rejects_nonpositive_duration():
    with pytest.raises(ValueError, match="end > start"):
        SpeechSegment(5.0, 5.0)


# ---- chunk planning ------------------------------------------------------


def test_chunks_pack_until_max_then_cut_at_silence():
    segs = [SpeechSegment(0, 10), SpeechSegment(11, 20), SpeechSegment(21, 40)]
    chunks = plan_chunks(segs, max_chunk_s=25.0)
    # 0..20 fits in 25 (packs the first two); 21..40 starts a new chunk.
    assert [(c.start_s, c.end_s) for c in chunks] == [(0, 20), (21, 40)]
    assert not any(c.forced_split for c in chunks)


def test_single_segment_longer_than_max_is_force_split():
    chunks = plan_chunks([SpeechSegment(0, 70)], max_chunk_s=28.0)
    assert len(chunks) == 3  # ceil(70/28)
    assert all(c.forced_split for c in chunks)
    assert chunks[0].start_s == 0 and chunks[-1].end_s == pytest.approx(70)


def test_plan_chunks_sorts_unordered_input():
    segs = [SpeechSegment(30, 40), SpeechSegment(0, 10)]
    chunks = plan_chunks(segs, max_chunk_s=100.0)
    assert chunks[0].start_s == 0  # merged into one window, ordered


def test_pad_extends_windows_without_going_negative():
    chunks = plan_chunks([SpeechSegment(0, 5)], max_chunk_s=28.0, pad_s=0.5)
    assert chunks[0].start_s == 0.0  # clamped at 0, not -0.5
    assert chunks[0].end_s == pytest.approx(5.5)


def test_empty_input_yields_no_chunks():
    assert plan_chunks([], max_chunk_s=28.0) == []


def test_nested_segment_does_not_shrink_coverage():
    # A segment nested inside the current window must not pull the window end back
    # (that would silently drop the audio past it).
    chunks = plan_chunks([SpeechSegment(0, 20), SpeechSegment(5, 10)], max_chunk_s=25.0)
    assert [(c.start_s, c.end_s) for c in chunks] == [(0, 20)]  # 10–20 s not lost


# ---- hallucination filter ------------------------------------------------


def _chunk(text="কথা", logprob=-0.2, comp=1.5, nosp=0.1, **kw):
    return ChunkTranscription(
        window_start_s=kw.get("start", 0.0), window_end_s=kw.get("end", 5.0),
        text=text, avg_logprob=logprob, compression_ratio=comp, no_speech_prob=nosp,
        words=kw.get("words", ()),
    )


def test_low_logprob_chunk_is_dropped():
    result = filter_hallucinations([_chunk(logprob=-2.0)])
    assert result.n_dropped == 1
    assert "avg_logprob" in result.dropped[0].reasons[0]


def test_repetitive_chunk_dropped_by_compression_ratio():
    result = filter_hallucinations([_chunk(comp=3.0)])  # gzip-compressible → repetition
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


# ---- merge ---------------------------------------------------------------


def test_merge_shifts_words_to_global_time():
    # A chunk whose window starts at 10 s, with a word at chunk-local 1..2 s.
    chunk = ChunkTranscription(10.0, 15.0, "শব্দ", words=(Word(1.0, 2.0, "শব্দ"),))
    transcript = merge_transcriptions([chunk])
    seg = transcript.segments[0]
    assert seg.start_s == pytest.approx(11.0) and seg.end_s == pytest.approx(12.0)
    assert transcript.text == "শব্দ"


def test_merge_dedupes_overlapping_boundary_words():
    a = ChunkTranscription(0.0, 10.0, "", words=(Word(8.0, 9.0, "শেষ"),))
    # b overlaps a; its first word (global 8.5) is already covered by a's word (ends 9.0).
    b = ChunkTranscription(8.0, 18.0, "", words=(Word(0.5, 1.0, "শেষ"), Word(2.0, 3.0, "নতুন")))
    transcript = merge_transcriptions([a, b])
    assert transcript.text == "শেষ নতুন"  # the duplicate "শেষ" at the boundary is dropped once


def test_merge_wordless_chunks_concatenate_in_time_order():
    a = ChunkTranscription(0.0, 5.0, "প্রথম")
    b = ChunkTranscription(5.0, 10.0, "দ্বিতীয়")
    transcript = merge_transcriptions([b, a])  # unordered
    assert transcript.text == "প্রথম দ্বিতীয়"


def test_merge_skips_wordless_chunk_fully_covered():
    a = ChunkTranscription(0.0, 10.0, "সব")
    covered = ChunkTranscription(2.0, 8.0, "নকল")  # window inside a → skipped
    assert merge_transcriptions([a, covered]).text == "সব"


def test_merge_wordless_overlap_is_dropped_not_duplicated():
    # A wordless chunk starting inside covered time is dropped, never re-emitted —
    # dropping is honest where duplicating would fabricate a repeat.
    a = ChunkTranscription(0.0, 10.0, "শেষ", words=(Word(8.0, 9.0, "শেষ"),))
    b = ChunkTranscription(8.0, 18.0, "শেষ নতুন")  # no word timings, overlaps a
    text = merge_transcriptions([a, b]).text
    assert "শেষ শেষ" not in text
    assert text == "শেষ"


# ---- real-time factor ----------------------------------------------------


def test_real_time_factor():
    assert real_time_factor(30.0, 60.0) == 0.5  # 2x faster than real time
    with pytest.raises(ValueError):
        real_time_factor(1.0, 0.0)


# ---- pipeline orchestration (fakes, no audio) ----------------------------


class _FakeVAD:
    def __init__(self, segments):
        self._segments = segments

    def detect(self, audio_path):
        return self._segments


class _FakeTranscriber:
    """Returns a canned transcription per window; one window is a hallucination."""

    def transcribe_chunk(self, audio_path, start_s, end_s):
        if start_s >= 100:  # the last chunk is silence hallucinated as text
            return ChunkTranscription(start_s, end_s, "নকল", no_speech_prob=0.95,
                                      words=(Word(0.0, 1.0, "নকল"),))
        return ChunkTranscription(start_s, end_s, "ভালো",
                                  words=(Word(0.0, 1.0, "ভালো"),))


def test_pipeline_end_to_end_with_fakes():
    vad = _FakeVAD([SpeechSegment(0, 20), SpeechSegment(100, 105)])
    pipeline = LongFormTranscriber(vad, _FakeTranscriber(), max_chunk_s=28.0, pad_s=0.0)
    clock = iter([0.0, 12.0])  # 12 s to process 200 s of audio
    result = pipeline.transcribe("rec.wav", audio_duration_s=200.0, now=lambda: next(clock))

    assert result.n_chunks == 2
    assert result.filtering.n_dropped == 1  # the silence chunk
    assert result.transcript.text == "ভালো"  # hallucination excluded from the transcript
    assert result.rtf == pytest.approx(12.0 / 200.0)
    assert result.rtf < 1.0


def test_long_form_pipeline_transcriber_returns_merged_text():
    # Adapts the pipeline to the transcribe(path)->str protocol for recording-level eval.
    from shono.transcribe import LongFormPipelineTranscriber

    pipeline = LongFormTranscriber(
        _FakeVAD([SpeechSegment(0, 20), SpeechSegment(100, 105)]), _FakeTranscriber(),
        pad_s=0.0,
    )
    adapter = LongFormPipelineTranscriber(pipeline)
    text = adapter.transcribe("rec.wav", duration_s=200.0)  # duration given → no audio read
    assert text == "ভালো"  # hallucinated silence chunk excluded from the merged text


def test_cli_missing_audio_exits_2_without_loading_torch():
    # The file check happens before any GPU import, so this runs torch-free.
    from shono.transcribe.__main__ import main

    assert main(["/no/such/file.wav", "--model", "/some/ct2"]) == 2


def test_short_form_whisper_transcriber_wiring(tmp_path):
    # Prove the baseline transcriber's load→features→generate→decode path works,
    # end to end, with a real (tiny) Whisper — runs where torch/transformers exist.
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("librosa")
    import numpy as np

    from shono.transcribe import ShortFormWhisperTranscriber

    wav = tmp_path / "clip.wav"
    sf.write(wav, np.zeros(16000, dtype="float32"), 16000)  # 1 s of silence
    transcriber = ShortFormWhisperTranscriber("openai/whisper-tiny", language="en")
    text = transcriber.transcribe(str(wav))
    assert isinstance(text, str)  # a real decode, not a crash
