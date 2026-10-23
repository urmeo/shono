"""Prepare validated dataset manifests from explicit local mounts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from shono.data.audio_paths import hash_manifest_audio, resolved_path, validate_manifest_audio
from shono.data.build import (
    from_common_voice_tsv,
    from_fleurs_tsv,
    from_loop_jsonl,
    from_mucs_kaldi,
    from_slr53_tsv,
)
from shono.data.leakage import LeakageReport, audit_leakage
from shono.data.license import LicenseRegistry
from shono.data.manifest import Manifest
from shono.output_paths import validate_output_destinations


@dataclass(frozen=True)
class DatasetMounts:
    audio_root: Path
    common_voice: Path
    fleurs: Path
    slr53: Path
    mucs: Path
    loop: Path


@dataclass(frozen=True)
class PreparationResult:
    manifests: tuple[Manifest, ...]
    paths: tuple[Path, ...]
    audits: tuple[LeakageReport, ...]
    pending: tuple[str, ...]


def prepare_manifests(
    mounts: DatasetMounts,
    output_dir: str | Path,
    registry: LicenseRegistry,
    *,
    loop_indexes: Mapping[str, Path],
    hash_audio: bool = False,
    protected_paths: Iterable[str | Path] = (),
) -> PreparationResult:
    """Read available splits, validate every input, then replace source manifests.

    Audits describe overlap; training must separately require an approved audit.
    Each call rebuilds its input and pending collections from the current mounts.
    """
    if type(hash_audio) is not bool:
        raise ValueError("hash_audio must be a boolean")
    audio_root = resolved_path(mounts.audio_root)
    if not audio_root.is_dir():
        raise ValueError(f"audio_root is not an existing directory: {audio_root}")
    roots = {
        name: resolved_path(getattr(mounts, name))
        for name in ("common_voice", "fleurs", "slr53", "mucs", "loop")
    }
    for name, root in roots.items():
        if not root.is_relative_to(audio_root):
            raise ValueError(f"{name} mount must be under audio_root")
    manifests = []
    pending = []
    protected = list(protected_paths)

    def collect(index, missing, reader, *, indexes=(), **options):
        if not index.is_file():
            pending.append(missing)
            return
        manifest = reader(index, audio_root=audio_root, **options)
        manifest.validate_against(registry)
        protected.extend(validate_manifest_audio(manifest, audio_root))
        if hash_audio:
            manifest = hash_manifest_audio(manifest, audio_root)
        protected.extend(path for path in (index, *indexes) if path is not None)
        manifests.append(manifest)

    cv_root = roots["common_voice"]
    duration_index = cv_root / "clip_durations.tsv"
    durations = duration_index if duration_index.is_file() else None
    for split in ("train", "dev", "test"):
        collect(
            cv_root / f"{split}.tsv",
            f"Common Voice {split}: local index missing",
            from_common_voice_tsv,
            name=f"cv-bn-{split}",
            split=split,
            version="cv-corpus-26.0",
            audio_subdir=(cv_root.relative_to(audio_root) / "clips").as_posix(),
            durations_tsv=durations,
            indexes=(durations,),
        )
    collect(
        roots["fleurs"] / "test.tsv",
        "FLEURS: local test index missing",
        from_fleurs_tsv,
        name="fleurs-bn-test",
        split="test",
        audio_subdir=(roots["fleurs"].relative_to(audio_root) / "audio").as_posix(),
    )
    collect(
        roots["slr53"] / "utt_spk_text.tsv",
        "SLR53: local published index missing",
        from_slr53_tsv,
        name="slr53-train",
        split="train",
        dataset_root=roots["slr53"],
    )
    for split in ("train", "test"):
        local = roots["mucs"] / split
        wavs = local / "wav.scp"
        segments, speakers = (
            path if path.is_file() else None for path in (local / "segments", local / "utt2spk")
        )
        collect(
            local / "text",
            f"MUCS {split}: local indexes missing",
            from_mucs_kaldi,
            wav_scp_path=wavs,
            name=f"mucs-cs-{split}",
            split=split,
            path_root=local,
            segments_path=segments,
            utt2spk_path=speakers,
            indexes=(wavs, segments, speakers),
        )
    if "bengali_loop" not in registry or registry.require("bengali_loop").status == "excluded":
        pending.append("Bengali-Loop: release permissions are unconfirmed")
    else:
        prefix = (
            "" if roots["loop"] == audio_root else roots["loop"].relative_to(audio_root).as_posix()
        )
        for name, index in loop_indexes.items():
            collect(
                Path(index),
                f"{name}: local recording index missing",
                from_loop_jsonl,
                name=name,
                version="operator-recorded-release",
                audio_prefix=prefix,
            )

    trains = [manifest for manifest in manifests if manifest.split == "train"]
    audits = tuple(
        audit_leakage(trains, manifest, audio_root=audio_root)
        for manifest in manifests
        if trains and manifest.split in ("dev", "test")
    )
    output_dir = Path(output_dir)
    paths = validate_output_destinations(
        (output_dir / f"{manifest.name}.manifest.jsonl" for manifest in manifests),
        protected_paths=protected,
        output_root=output_dir,
    )
    if paths:
        output_dir.mkdir(parents=True, exist_ok=True)
        for manifest, path in zip(manifests, paths, strict=True):
            manifest.to_jsonl(path)
    return PreparationResult(tuple(manifests), paths, audits, tuple(pending))
