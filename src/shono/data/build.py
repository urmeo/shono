"""Build manifests from explicit indexes and local audio mappings."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path

from shono.data.audio_paths import (
    DurationFn,
    hash_manifest_audio,
    resolved_path,
)
from shono.data.audio_paths import audio_duration as duration_from_audio
from shono.data.manifest import Manifest, Segment
from shono.data.tabular import delimited_rows, named_rows
from shono.validation import audio_reference, json_object, real_number, text_value

FLEURS_SAMPLE_RATE = 16_000


def build_manifest(
    *,
    name: str,
    source: str,
    split: str,
    domain: str,
    version: str,
    rows: Iterable[Mapping[str, object]],
    language: str = "bn",
    checksum: str | None = None,
    audio_root: str | Path = ".",
    duration_of: DurationFn | None = None,
    hash_audio: bool = False,
) -> Manifest:
    """Build metadata. Aligned rows need duration; blank references are omitted.

    audio_sha256 hashes complete source-file bytes, including for cropped spans.
    """
    if type(hash_audio) is not bool:
        raise ValueError("hash_audio must be a boolean")
    dur_fn = duration_of or duration_from_audio
    root = resolved_path(audio_root)
    segments = []
    for i, row in enumerate(rows, 1):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {i} must be an object")
        text = text_value(row.get("text"), f"row {i} text", empty=True).strip()
        if not text:
            continue
        audio = audio_reference(row.get("audio"))
        sid = text_value(row.get("id"), f"row {i} id")
        start = row.get("start_s")
        if start is not None:
            start = real_number(start, f"row {i} start_s", allow_string=True)
            if row.get("duration_s") is None:
                raise ValueError(f"row {i}: aligned span requires duration_s")
        duration = row.get("duration_s")
        if duration is None:
            path = resolved_path(root / audio)
            if not path.is_relative_to(root):
                raise ValueError(f"row {i}: audio escapes audio_root")
            duration = dur_fn(path)
        duration = real_number(duration, f"row {i} duration_s", allow_string=True)
        segments.append(
            Segment(
                id=sid,
                audio=audio,
                text=text,
                duration_s=duration,
                recording_id=text_value(row.get("recording_id", sid), f"row {i} recording_id"),
                start_s=start,
                speaker=row.get("speaker"),
                language=row.get("language", language),
                audio_sha256=row.get("audio_sha256"),
            )
        )
    manifest = Manifest(
        name=name,
        source=source,
        split=split,
        domain=domain,
        version=version,
        language=language,
        checksum=checksum,
        segments=tuple(segments),
    )
    return hash_manifest_audio(manifest, root) if hash_audio else manifest


def collapse_to_recordings(
    manifest: Manifest,
    *,
    duration_of: DurationFn | None = None,
    audio_root: str | Path = ".",
) -> Manifest:
    """Join nonoverlapping spans from one file per block.

    Without duration_of, duration is an annotated timeline bound. Whole-file
    inference must separately verify the complete reference and actual duration.
    """
    groups: dict[str, list[Segment]] = defaultdict(list)
    for seg in manifest.segments:
        groups[seg.recording_id].append(seg)
    segments = []
    root = resolved_path(audio_root)
    for rid, group in sorted(groups.items()):
        if len({s.audio for s in group}) != 1:
            raise ValueError(f"block {rid!r} contains multiple audio files")
        if len(group) > 1 and any(s.start_s is None for s in group):
            raise ValueError(f"block {rid!r} contains ambiguous untimed segments")
        if len({s.language for s in group}) != 1 or len({s.audio_sha256 for s in group}) != 1:
            raise ValueError(f"block {rid!r} contains conflicting language/hash metadata")
        spans = sorted(group, key=lambda s: (s.start_s or 0.0, s.id))
        previous_end = 0.0
        for seg in spans:
            start = seg.start_s or 0.0
            if start < previous_end:
                raise ValueError(f"block {rid!r} contains overlapping reference spans")
            previous_end = start + seg.duration_s
        duration = previous_end
        if duration_of is not None:
            path = resolved_path(root / spans[0].audio)
            if not path.is_relative_to(root):
                raise ValueError("recording audio escapes audio_root")
            duration = real_number(duration_of(path), "actual recording duration")
            if duration < previous_end:
                raise ValueError(f"block {rid!r}: actual duration does not cover reference spans")
        segments.append(
            replace(
                spans[0],
                id=rid,
                text=" ".join(s.text for s in spans),
                duration_s=duration,
                start_s=None,
                speaker=spans[0].speaker if len({s.speaker for s in spans}) == 1 else None,
            )
        )
    return replace(manifest, segments=tuple(segments))


def from_common_voice_tsv(
    tsv_path: str | Path,
    *,
    split: str,
    name: str,
    version: str,
    source: str = "common_voice_bn",
    audio_subdir: str = "clips",
    durations_tsv: str | Path | None = None,
    audio_root: str | Path = ".",
    duration_of: DurationFn | None = None,
    hash_audio: bool = False,
) -> Manifest:
    durations = _load_cv_durations(durations_tsv) if durations_tsv else {}
    rows = []
    for lineno, rec in named_rows(tsv_path, {"path", "sentence"}, skip_blank=True):
        clip = text_value(rec["path"], f"{tsv_path}: line {lineno} path")
        if not rec["sentence"].strip():
            continue
        clip_id = Path(clip).stem
        row = dict(
            id=clip_id,
            audio=f"{audio_subdir}/{clip}",
            text=rec["sentence"],
            recording_id=rec.get("client_id") or clip_id,
            duration_s=durations.get(clip),
        )
        rows.append(row)
    return build_manifest(
        name=name,
        source=source,
        split=split,
        domain="read",
        version=version,
        rows=rows,
        audio_root=audio_root,
        duration_of=duration_of,
        hash_audio=hash_audio,
    )


def _load_cv_durations(path: str | Path) -> dict[str, float]:
    rows = delimited_rows(path)
    _, columns = next(rows, (0, []))
    clip_key = "clip" if "clip" in columns else "path"
    duration_key = "duration[ms]" if "duration[ms]" in columns else "duration"
    if clip_key not in columns or duration_key not in columns or len(set(columns)) != len(columns):
        raise ValueError(f"{path}: missing/duplicate duration columns")
    durations = {}
    for lineno, values in rows:
        if not values:
            continue
        if len(values) != len(columns):
            raise ValueError(f"{path}: line {lineno}: malformed duration row")
        record = dict(zip(columns, values, strict=True))
        clip = text_value(record[clip_key], "duration clip")
        if clip in durations:
            raise ValueError(f"{path}: duplicate duration clip {clip!r}")
        value = real_number(record[duration_key], "duration milliseconds", allow_string=True)
        if value <= 0:
            raise ValueError("duration milliseconds must be positive")
        durations[clip] = value / 1000.0
    return durations


def from_fleurs_tsv(
    tsv_path: str | Path,
    *,
    split: str,
    name: str,
    audio_subdir: str = "audio",
    source: str = "fleurs_bn",
    version: str = "fleurs",
    audio_root: str | Path = ".",
    hash_audio: bool = False,
) -> Manifest:
    rows = []
    for lineno, parts in delimited_rows(tsv_path):
        if len(parts) != 6:
            raise ValueError(f"{tsv_path}: line {lineno}: expected six FLEURS fields")
        _id, file_name, _raw, transcription, num_samples, _gender = parts
        if not transcription.strip():
            continue
        samples = real_number(num_samples, "num_samples", allow_string=True)
        if samples <= 0 or not samples.is_integer():
            raise ValueError(f"{tsv_path}: line {lineno}: num_samples must be a positive integer")
        rows.append(
            dict(
                id=Path(file_name).stem,
                audio=f"{audio_subdir}/{file_name}",
                text=transcription,
                duration_s=samples / FLEURS_SAMPLE_RATE,
            )
        )
    return build_manifest(
        name=name,
        source=source,
        split=split,
        domain="read",
        version=version,
        rows=rows,
        audio_root=audio_root,
        hash_audio=hash_audio,
    )


def from_slr53_tsv(
    path: str | Path,
    *,
    name: str,
    split: str,
    audio_root: str | Path,
    dataset_root: str | Path | None = None,
    audio_paths: Mapping[str, str] | None = None,
    duration_of: DurationFn | None = None,
    hash_audio: bool = False,
    source: str = "openslr_slr53",
    version: str = "SLR53",
) -> Manifest:
    """Read published FileID/UserID/transcription TSV with an explicit WAV mapping."""
    root = resolved_path(audio_root)
    if (dataset_root is None) == (audio_paths is None):
        raise ValueError("supply either dataset_root or an explicit audio_paths mapping")
    mapping = dict(audio_paths) if audio_paths is not None else {}
    if dataset_root is not None:
        local = resolved_path(dataset_root)
        if not local.is_dir() or not local.is_relative_to(root):
            raise ValueError("SLR53 dataset_root must be an existing directory under audio_root")
        for candidate in sorted(local.rglob("*")):
            if candidate.suffix.lower() != ".wav" or not candidate.is_file():
                continue
            resolved = resolved_path(candidate)
            if not resolved.is_relative_to(local):
                raise ValueError("SLR53 WAV symlink escapes dataset_root")
            key = candidate.stem
            if key in mapping:
                raise ValueError(f"SLR53 duplicate WAV FileID {key!r}")
            mapping[key] = resolved.relative_to(root).as_posix()
    rows = []
    for lineno, parts in delimited_rows(path):
        if len(parts) != 3:
            raise ValueError(f"{path}: line {lineno}: expected FileID, UserID, transcription")
        uid, speaker, text = parts
        text_value(uid, "SLR53 FileID")
        text_value(speaker, "SLR53 UserID")
        if uid not in mapping:
            raise ValueError(f"{path}: line {lineno}: missing WAV for FileID {uid!r}")
        rows.append(
            dict(id=uid, audio=mapping[uid], text=text, recording_id=speaker, speaker=speaker)
        )
    return build_manifest(
        name=name,
        source=source,
        split=split,
        domain="read",
        version=version,
        rows=rows,
        audio_root=root,
        duration_of=duration_of,
        hash_audio=hash_audio,
    )


def from_loop_jsonl(
    path: str | Path,
    *,
    name: str,
    version: str,
    split: str = "test",
    audio_prefix: str = "",
    audio_root: str | Path = ".",
    source: str = "bengali_loop",
) -> Manifest:
    """Read published recording JSONL. It is not aligned training data."""
    if split == "train":
        raise ValueError(
            "Loop recording JSONL is evaluation-only; training requires aligned targets"
        )
    rows = []
    needed = {"id", "audio_filepath", "duration", "text"}
    for lineno, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        rec = json_object(raw, f"{path}: line {lineno}")
        if not needed.issubset(rec):
            raise ValueError(
                f"{path}: line {lineno}: missing Loop recording fields {needed - set(rec)}"
            )
        audio = text_value(rec["audio_filepath"], "audio_filepath")
        if Path(audio).is_absolute():
            resolved = resolved_path(audio)
            root = resolved_path(audio_root)
            if not resolved.is_relative_to(root):
                raise ValueError("Loop audio filepath escapes audio_root")
            audio = resolved.relative_to(root).as_posix()
        elif audio_prefix:
            audio = f"{audio_prefix}/{audio}"
        rows.append(
            dict(
                id=rec["id"],
                audio=audio,
                text=rec["text"],
                duration_s=rec["duration"],
                language=rec.get("language", "bn"),
                audio_sha256=rec.get("audio_sha256"),
            )
        )
    return build_manifest(
        name=name,
        source=source,
        split=split,
        domain="long-form",
        version=version,
        rows=rows,
        audio_root=audio_root,
    )


def _kaldi_pairs(path: str | Path) -> dict[str, str]:
    records = {}
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not parts[1].strip():
            raise ValueError(f"{path}: line {lineno}: expected ID and value")
        if parts[0] in records:
            raise ValueError(f"{path}: line {lineno}: duplicate ID {parts[0]!r}")
        records[parts[0]] = parts[1].strip()
    return records


def from_mucs_kaldi(
    text_path: str | Path,
    wav_scp_path: str | Path,
    *,
    name: str,
    split: str,
    version: str = "SLR104",
    segments_path: str | Path | None = None,
    utt2spk_path: str | Path | None = None,
    audio_root: str | Path,
    path_root: str | Path | None = None,
    duration_of: DurationFn | None = None,
    source: str = "mucs_slr104",
    hash_audio: bool = False,
) -> Manifest:
    """Read explicit Kaldi joins; wav.scp accepts literal filesystem paths only."""
    texts = _kaldi_pairs(text_path)
    wavs = _kaldi_pairs(wav_scp_path)
    speakers = _kaldi_pairs(utt2spk_path) if utt2spk_path is not None else {}
    if utt2spk_path is not None and set(speakers) != set(texts):
        raise ValueError("utt2spk IDs must exactly match text IDs")
    spans = {}
    if segments_path is not None:
        for lineno, line in enumerate(Path(segments_path).read_text().splitlines(), 1):
            parts = line.split()
            if len(parts) != 4 or parts[0] in spans:
                raise ValueError(f"{segments_path}: line {lineno}: malformed/duplicate segment")
            uid, rid, start, end = parts
            start = real_number(start, "segment start", allow_string=True)
            end = real_number(end, "segment end", allow_string=True)
            if start < 0 or end <= start:
                raise ValueError("Kaldi segments require 0 <= start < end")
            spans[uid] = (rid, start, end - start)
        if set(spans) != set(texts):
            raise ValueError("segments IDs must exactly match text IDs")
    elif set(wavs) != set(texts):
        raise ValueError("without segments, wav.scp IDs must exactly match text IDs")
    root = resolved_path(audio_root)
    relative_root = resolved_path(path_root) if path_root is not None else root
    audio_refs = {}
    for rid, raw_path in wavs.items():
        if any(token in raw_path for token in ("|", "$", "`", "\n")):
            raise ValueError(
                "wav.scp commands/pipes/substitutions are unsupported; supply file paths"
            )
        path = Path(raw_path)
        resolved = resolved_path(path if path.is_absolute() else relative_root / path)
        if not resolved.is_relative_to(root):
            raise ValueError(f"wav.scp recording {rid!r} escapes audio_root")
        audio_refs[rid] = resolved.relative_to(root).as_posix()
    rows = []
    for uid, text in texts.items():
        rid, start, duration = spans.get(uid, (uid, None, None))
        if rid not in audio_refs:
            raise ValueError(f"utterance {uid!r}: missing wav.scp recording {rid!r}")
        rows.append(
            dict(
                id=uid,
                audio=audio_refs[rid],
                text=text,
                recording_id=rid,
                start_s=start,
                duration_s=duration,
                speaker=speakers.get(uid),
            )
        )
    return build_manifest(
        name=name,
        source=source,
        split=split,
        domain="code-switch",
        version=version,
        rows=rows,
        audio_root=root,
        duration_of=duration_of,
        hash_audio=hash_audio,
    )
