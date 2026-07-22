"""Diarization: DER under a stated protocol, VAD-intersection, speaker attribution."""

import pytest

from shono.diarize import (
    DERConfig,
    SpeakerSegment,
    attribute_speakers,
    corpus_der,
    der,
    der_bootstrap_ci,
    load_rttm,
    render_der_report,
    vad_intersection,
)
from shono.transcribe.types import SpeechSegment, Transcript, TranscriptSegment


def _s(start, end, spk):
    return SpeakerSegment(start, end, spk)


# ---- SpeakerSegment ------------------------------------------------------


def test_speaker_segment_validates():
    with pytest.raises(ValueError, match="end > start"):
        SpeakerSegment(1.0, 1.0, "A")
    with pytest.raises(ValueError, match="speaker label"):
        SpeakerSegment(0.0, 1.0, "")


# ---- DER: the load-bearing hand-computed cases ---------------------------


def test_perfect_diarization_is_zero():
    ref = [_s(0, 10, "A")]
    result = der(ref, [_s(0, 10, "spk0")], DERConfig(collar_s=0.0))
    assert result.der == 0.0
    assert result.mapping == {"spk0": "A"}


def test_optimal_mapping_fixes_swapped_labels():
    # Labels are swapped vs reference; optimal mapping must recover DER 0.
    ref = [_s(0, 5, "A"), _s(5, 10, "B")]
    hyp = [_s(0, 5, "Y"), _s(5, 10, "X")]
    result = der(ref, hyp, DERConfig(collar_s=0.0))
    assert result.der == 0.0
    assert result.mapping == {"Y": "A", "X": "B"}


def test_missed_speech_is_one_when_nothing_hypothesized():
    ref = [_s(0, 10, "A")]
    result = der(ref, [], DERConfig(collar_s=0.0))
    assert result.der == 1.0
    assert result.missed_s == pytest.approx(10.0)


def test_false_alarm_counts_hypothesized_nonspeech():
    ref = [_s(0, 5, "A")]
    hyp = [_s(0, 10, "A")]  # 5 s of speech invented past the reference
    result = der(ref, hyp, DERConfig(collar_s=0.0))
    assert result.false_alarm_s == pytest.approx(5.0)
    assert result.der == pytest.approx(1.0)  # 5 s error / 5 s ref


def test_confusion_counts_wrong_speaker():
    ref = [_s(0, 10, "A")]
    hyp = [_s(0, 5, "A"), _s(5, 10, "B")]  # second half attributed to a new speaker
    result = der(ref, hyp, DERConfig(collar_s=0.0))
    # [5,10] maps neither A nor the extra to A's reference correctly for B → confusion.
    assert result.confusion_s == pytest.approx(5.0)
    assert result.der == pytest.approx(0.5)


def test_collar_forgives_boundary_slop():
    ref = [_s(0, 5, "A"), _s(5, 10, "B")]
    hyp = [_s(0, 5.1, "A"), _s(5.1, 10, "B")]  # boundary off by 0.1 s
    with_collar = der(ref, hyp, DERConfig(collar_s=0.25))
    without = der(ref, hyp, DERConfig(collar_s=0.0))
    assert with_collar.der == pytest.approx(0.0)  # slop is inside the collar
    assert without.der > 0.0


def test_skip_overlap_excludes_overlapping_reference():
    ref = [_s(0, 10, "A"), _s(4, 6, "B")]  # A and B overlap on [4,6]
    hyp = [_s(0, 10, "A")]  # misses B entirely
    scored = der(ref, hyp, DERConfig(collar_s=0.0, skip_overlap=False))
    skipped = der(ref, hyp, DERConfig(collar_s=0.0, skip_overlap=True))
    assert scored.der > 0.0  # the missed overlapping B counts
    assert skipped.der == pytest.approx(0.0)  # [4,6] excluded, rest is perfect


def test_empty_reference_rejected():
    with pytest.raises(ValueError, match="empty reference"):
        der([], [_s(0, 1, "A")])


def test_der_is_nan_when_reference_fully_collar_excluded():
    import math

    # A 0.2 s reference vanishes inside the 0.25 s collar, but the hypothesis has a
    # 3 s false alarm outside it — DER is undefined, must be NaN, never a hiding 0.0.
    result = der([_s(5.0, 5.2, "A")], [_s(0.0, 3.0, "B")], DERConfig(collar_s=0.25))
    assert math.isnan(result.der)
    assert result.false_alarm_s == pytest.approx(3.0)
    assert result.total_ref_s == 0.0


# ---- corpus DER + CI -----------------------------------------------------


def test_corpus_der_weights_by_reference_time():
    # One perfect 100 s recording + one fully-missed 10 s recording.
    good = ([_s(0, 100, "A")], [_s(0, 100, "A")])
    bad = ([_s(0, 10, "A")], [])
    result = corpus_der([good, bad], DERConfig(collar_s=0.0))
    # 10 s error / 110 s reference = 0.0909, not the 0.5 a naive per-file mean gives.
    assert result.der == pytest.approx(10.0 / 110.0)


def test_der_bootstrap_ci_brackets_point():
    recs = [([_s(0, 10, "A")], [_s(0, 10, "A")]) for _ in range(3)]
    recs.append(([_s(0, 10, "A")], []))  # one bad recording
    ci = der_bootstrap_ci(recs, DERConfig(collar_s=0.0), n_resamples=200, seed=1)
    assert ci.lower <= ci.point <= ci.upper
    assert ci.n_recordings == 4


def test_der_bootstrap_needs_two_recordings():
    with pytest.raises(ValueError, match=">= 2 recordings"):
        der_bootstrap_ci([([_s(0, 10, "A")], [_s(0, 10, "A")])])


# ---- VAD-intersection ----------------------------------------------------


def test_vad_intersection_clips_turns_to_speech():
    turns = [_s(0, 10, "A")]
    speech = [SpeechSegment(2, 4), SpeechSegment(6, 8)]
    clipped = vad_intersection(turns, speech)
    assert [(s.start_s, s.end_s, s.speaker) for s in clipped] == [(2, 4, "A"), (6, 8, "A")]


def test_vad_intersection_drops_turns_over_silence():
    turns = [_s(0, 2, "A")]
    speech = [SpeechSegment(5, 8)]  # no overlap
    assert vad_intersection(turns, speech) == []


# ---- speaker attribution -------------------------------------------------


def test_attribute_assigns_dominant_speaker():
    transcript = Transcript(segments=(TranscriptSegment(0, 6, "কথা"),))
    speakers = [_s(0, 5, "A"), _s(5, 10, "B")]  # A overlaps 5 s, B overlaps 1 s
    out = attribute_speakers(transcript, speakers)
    assert out.segments[0].speaker == "A"


def test_attribute_leaves_none_when_no_overlap():
    transcript = Transcript(segments=(TranscriptSegment(0, 2, "কথা"),))
    out = attribute_speakers(transcript, [_s(5, 10, "A")])
    assert out.segments[0].speaker is None


# ---- RTTM loading + DER report -------------------------------------------


def test_load_rttm_parses_speaker_turns(tmp_path):
    rttm = tmp_path / "ref.rttm"
    rttm.write_text(
        "SPEAKER rec1 1 0.00 5.00 <NA> <NA> A <NA> <NA>\n"
        "SPEAKER rec1 1 5.00 3.00 <NA> <NA> B <NA> <NA>\n"
        "SPEAKER rec2 1 0.00 2.00 <NA> <NA> X <NA> <NA>\n",  # different recording
        encoding="utf-8",
    )
    turns = load_rttm(rttm, file_id="rec1")
    assert [(s.start_s, s.end_s, s.speaker) for s in turns] == [(0.0, 5.0, "A"), (5.0, 8.0, "B")]


def test_render_der_report_states_protocol_and_components():
    result = corpus_der([([_s(0, 10, "A")], [_s(0, 5, "A")])], DERConfig(collar_s=0.0))
    ci = der_bootstrap_ci(
        [([_s(0, 10, "A")], [_s(0, 5, "A")]), ([_s(0, 10, "A")], [_s(0, 10, "A")])],
        DERConfig(collar_s=0.0), n_resamples=100,
    )
    report = render_der_report([("shono", result, ci)], DERConfig(collar_s=0.25))
    assert "collar 0.25" in report  # protocol stated
    assert "| shono |" in report  # the system row
    assert "Missed" in report  # components broken out
