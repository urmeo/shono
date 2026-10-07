"""Boundary regressions for chunk coverage, interval unions and DER reporting."""

import math
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from shono.demo.pipeline import format_transcript, transcribe_recording
from shono.diarize import (
    DERConfig,
    SpeakerSegment,
    attribute_speakers,
    corpus_der,
    der,
    der_bootstrap_ci,
    load_loop_csv,
    load_rttm,
    render_der_report,
    vad_intersection,
)
from shono.transcribe import (
    ChunkTranscription,
    HallucinationConfig,
    LongFormTranscriber,
    SpeechSegment,
    Transcript,
    TranscriptSegment,
    Word,
    merge_transcriptions,
    plan_chunks,
    real_time_factor,
)
from shono.transcribe.pipeline import LongFormPipelineTranscriber, SileroVAD


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, True, "1", 1j])
def test_interval_and_clock_inputs_are_real_finite_and_nonnegative(bad):
    for make in (
        lambda: SpeechSegment(bad, 2),
        lambda: SpeakerSegment(bad, 2, "A"),
        lambda: Word(bad, 2, "word"),
        lambda: TranscriptSegment(bad, 2, "text"),
        lambda: DERConfig(collar_s=bad),
        lambda: real_time_factor(bad, 10),
    ):
        with pytest.raises(ValueError):
            make()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_chunk_s": 0},
        {"max_chunk_s": float("nan")},
        {"pad_s": -1},
        {"pad_s": 14},
        {"max_chunks": True},
        {"max_chunks": 0},
        {"max_chunks": 1_000_001},
    ],
)
def test_bad_chunk_parameters_reject_even_for_empty_input(kwargs):
    with pytest.raises(ValueError):
        plan_chunks([], **kwargs)


def test_padding_is_within_hard_maximum_and_eof_without_gaps():
    chunks = plan_chunks([SpeechSegment(0, 70)], max_chunk_s=28, pad_s=0.2, audio_duration_s=70)
    assert chunks[0].start_s == 0 and chunks[-1].end_s == 70
    assert all(0 < c.duration_s <= 28 and c.forced_split for c in chunks)
    assert all(a.end_s >= b.start_s for a, b in zip(chunks, chunks[1:], strict=False))


def test_bad_vad_eof_and_unbounded_plan_are_rejected():
    with pytest.raises(ValueError, match="beyond"):
        plan_chunks([SpeechSegment(0, 31)], audio_duration_s=30)
    with pytest.raises(ValueError, match="max_chunks"):
        plan_chunks([SpeechSegment(0, 1e308)], max_chunk_s=1e-300)
    with pytest.raises(ValueError, match="max_chunks"):
        plan_chunks([SpeechSegment(0, 60)], max_chunk_s=28, max_chunks=2)


def test_subsample_large_timestamps_do_not_form_nonadvancing_chunks():
    with pytest.raises(ValueError, match="end > start|geometry"):
        plan_chunks([SpeechSegment(1e16, 1e16 + 4)], max_chunk_s=0.5)


def test_missing_word_times_and_blank_chunks_preserve_new_text():
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


def test_word_local_bounds_and_confidence_guard():
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


def test_invalid_pipeline_geometry_is_rejected_before_vad():
    vad = NS(detect=lambda *_: pytest.fail("VAD must not run"))
    decoder = NS(transcribe_chunk=lambda *_: pytest.fail("decoder must not run"))
    with pytest.raises(ValueError, match="30"):
        LongFormTranscriber(vad, decoder, max_chunk_s=31)
    pipeline = LongFormTranscriber(vad, decoder)
    with pytest.raises(ValueError):
        pipeline.transcribe("fake.wav", float("nan"))


def test_pipeline_clock_and_returned_window_are_validated():
    vad = NS(detect=lambda *_: [SpeechSegment(0, 1)])
    decoder = NS(transcribe_chunk=lambda *_: ChunkTranscription(0, 2, "text"))
    with pytest.raises(ValueError, match="different audio window"):
        LongFormTranscriber(vad, decoder, pad_s=0).transcribe("fake.wav", 2)
    decoder = NS(transcribe_chunk=lambda *_: ChunkTranscription(0, 1, "text"))
    clock = iter([2, 1])
    with pytest.raises(ValueError, match="processing_s"):
        LongFormTranscriber(vad, decoder, pad_s=0).transcribe(
            "fake.wav", 2, now=lambda: next(clock)
        )


def test_longform_cropped_input_rejects_before_pipeline():
    adapter = LongFormPipelineTranscriber(NS(transcribe=lambda *_: pytest.fail("must not run")))
    with pytest.raises(ValueError, match="complete recording"):
        adapter.transcribe("fake.wav", start_s=1, duration_s=2)


def test_overlapping_vad_and_speaker_turns_do_not_double_vote():
    speakers = [SpeakerSegment(0, 2, "A"), SpeakerSegment(0, 2, "A"), SpeakerSegment(2, 5, "B")]
    clipped = vad_intersection(speakers, [SpeechSegment(0, 2), SpeechSegment(0, 5)])
    assert [(s.start_s, s.end_s, s.speaker) for s in clipped] == [(0, 2, "A"), (2, 5, "B")]
    transcript = Transcript(
        (TranscriptSegment(0, 5, "text", timing_precision="chunk"),), ("boundary",)
    )
    attributed = attribute_speakers(transcript, clipped)
    assert attributed.segments[0].speaker == "B"
    assert attributed.segments[0].speaker_precision == "dominant-segment"
    assert attributed.segments[0].timing_precision == "chunk"
    assert attributed.warnings == ("boundary",)


def test_demo_uses_once_computed_vad_and_excludes_silent_turns():
    transcript = Transcript((TranscriptSegment(0, 3, "text", timing_precision="chunk"),))
    result = NS(transcript=transcript, speech=(SpeechSegment(2, 3),))
    pipeline = NS(transcribe=lambda *_: result)
    diarizer = NS(diarize=lambda *_: [SpeakerSegment(0, 2, "A"), SpeakerSegment(2, 3, "B")])
    output = transcribe_recording("fake.wav", 3, pipeline, diarizer)
    assert output.segments[0].speaker == "B"
    assert "complete chunks" in format_transcript(output)
    assert "dominant speaker" in format_transcript(output)


def test_formatter_does_not_shrink_overlapping_same_speaker_end():
    transcript = Transcript(
        (TranscriptSegment(0, 20, "a", "A"), TranscriptSegment(10, 11, "b", "A"))
    )
    assert "0:20" in format_transcript(transcript)


def test_large_finite_der_midpoint_and_undefined_corpus():
    result = der([SpeakerSegment(1e308, 1.1e308, "A")], [], DERConfig(collar_s=0))
    assert result.der == 1 and result.total_ref_s > 0
    reference = [SpeakerSegment(5, 5.2, "A")]
    result = corpus_der([(reference, [])], DERConfig(collar_s=0.25))
    assert math.isnan(result.der) and result.total_ref_s == 0
    assert "| empty | n/a |" in render_der_report([("empty", result, None)], DERConfig())


def test_overflowed_der_ratio_fails_clearly():
    with pytest.raises(ValueError, match="DER"):
        der(
            [SpeakerSegment(0, 1e-308, "A")], [SpeakerSegment(1, 1e308, "B")], DERConfig(collar_s=0)
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_resamples": True},
        {"n_resamples": 0},
        {"n_resamples": 1_000_001},
        {"confidence": float("nan")},
        {"confidence": 1},
        {"confidence": 0},
        {"seed": True},
    ],
)
def test_der_bootstrap_parameters_reject(kwargs):
    recordings = [([SpeakerSegment(0, 10, "A")], [])] * 2
    with pytest.raises(ValueError):
        der_bootstrap_ci(recordings, **kwargs)


def test_der_confidence_and_protocol_match_actual_population():
    config = DERConfig(collar_s=0)
    recordings = [
        ([SpeakerSegment(0, 10, "A")], []),
        ([SpeakerSegment(0, 10, "A")], [SpeakerSegment(0, 10, "B")]),
    ]
    result = corpus_der(recordings, config)
    interval = der_bootstrap_ci(recordings, config, n_resamples=100, confidence=0.8)
    assert "80% CI" in render_der_report([("fake", result, interval)], config)
    with pytest.raises(ValueError, match="protocol"):
        render_der_report([("fake", result, interval)], DERConfig(collar_s=0.25))
    with pytest.raises(ValueError, match="point"):
        render_der_report([("fake", result, replace(interval, point=0.9))], config)
    with pytest.raises(ValueError, match="scorable"):
        der_bootstrap_ci([([SpeakerSegment(0, 0.1, "A")], [])] * 2)


def test_rttm_rejects_nonfinite_rows_and_multiple_recordings(tmp_path):
    path = tmp_path / "ref.rttm"
    path.write_text("SPEAKER rec 1 0 NaN <NA> <NA> A\n")
    with pytest.raises(ValueError, match="line 1"):
        load_rttm(path)
    path.write_text("SPEAKER rec1 1 0 2 <NA> <NA> A\nSPEAKER rec2 1 0 2 <NA> <NA> B\n")
    with pytest.raises(ValueError, match="multiple recording"):
        load_rttm(path)


def test_loop_csv_clock_and_speaker_contract(tmp_path):
    path = tmp_path / "loop.csv"
    path.write_text("start_time,end_time,speaker_id,text\n00:00:01.5,00:00:03,2,বাংলা\n")
    assert load_loop_csv(path) == [SpeakerSegment(1.5, 3, "2")]
    path.write_text("start_time,end_time,speaker_id\n00:99:01,00:00:03,2\n")
    with pytest.raises(ValueError, match="line 2"):
        load_loop_csv(path)
