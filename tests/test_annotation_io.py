"""Annotation loaders reject ambiguous channels and CSV columns."""

import pytest

from shono.diarize import SpeakerSegment, load_loop_csv, load_rttm


@pytest.mark.parametrize("file_id", [None, "rec"])
def test_rttm_rejects_multiple_channels_in_selected_recording(tmp_path, file_id):
    path = tmp_path / "reference.rttm"
    path.write_text(
        "SPEAKER rec 1 0 2 <NA> <NA> A\nSPEAKER rec 2 0 2 <NA> <NA> B\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="multiple channels"):
        load_rttm(path, file_id=file_id)


def test_rttm_channel_check_applies_to_selected_recording(tmp_path):
    path = tmp_path / "reference.rttm"
    path.write_text(
        "SPEAKER selected 2 3 2 <NA> <NA> A\n"
        "SPEAKER selected 2 0 1 <NA> <NA> B\n"
        "SPEAKER other 1 0 2 <NA> <NA> X\n"
        "SPEAKER other 2 0 2 <NA> <NA> Y\n",
        encoding="utf-8",
    )

    assert load_rttm(path, file_id="selected") == [
        SpeakerSegment(0, 1, "B"),
        SpeakerSegment(3, 5, "A"),
    ]


@pytest.mark.parametrize(
    "header,row",
    [
        (
            "start_time,end_time,speaker_id,start_time",
            "00:00:01,00:00:05,1,00:00:03",
        ),
        (
            "start_time,end_time,speaker_id,text,text",
            "00:00:01,00:00:05,1,first,second",
        ),
    ],
)
def test_loop_csv_rejects_duplicate_headers(tmp_path, header, row):
    path = tmp_path / "reference.csv"
    path.write_text(f"{header}\n{row}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate.*header"):
        load_loop_csv(path)


@pytest.mark.parametrize(
    "row",
    ["00:00:01,00:00:05,1", "00:00:01,00:00:05,1,text,extra"],
)
def test_loop_csv_rejects_rows_with_wrong_width(tmp_path, row):
    path = tmp_path / "reference.csv"
    path.write_text(f"start_time,end_time,speaker_id,text\n{row}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 2.*4 columns.*got"):
        load_loop_csv(path)


def test_loop_csv_accepts_quoted_text_and_reordered_unique_headers(tmp_path):
    path = tmp_path / "reference.csv"
    path.write_text(
        'text,speaker_id,end_time,start_time\n"first, second\nthird",2,00:00:05,00:00:01\n',
        encoding="utf-8-sig",
    )

    assert load_loop_csv(path) == [SpeakerSegment(1, 5, "2")]
