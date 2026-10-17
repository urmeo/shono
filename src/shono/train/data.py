"""Select licensed, role-specific examples without loading model weights."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, replace

from shono.data.license import LicenseRegistry
from shono.data.manifest import Manifest, Segment
from shono.data.validation import real_number, text_value
from shono.eval.normalize import NORMALIZER_VERSION, normalize


@dataclass(frozen=True)
class TrainExample:
    """An audio span, target text and optional timestamp supervision."""

    audio: str
    text: str
    duration_s: float
    recording_id: str
    source: str
    start_s: float | None = None
    use_timestamps: bool = False
    id: str | None = None
    split: str = "train"
    language: str = "bn"
    audio_sha256: str | None = None

    def __post_init__(self) -> None:
        text_value(self.source, "source")
        if type(self.use_timestamps) is not bool:
            raise ValueError("use_timestamps must be a boolean")
        text_value(self.split, "example split")
        if self.split not in {"train", "dev", "test"}:
            raise ValueError("example split must be train, dev or test")
        Segment(
            id=self.id if self.id is not None else self.recording_id,
            audio=self.audio,
            text=self.text,
            duration_s=self.duration_s,
            recording_id=self.recording_id,
            start_s=self.start_s,
            language=self.language,
            audio_sha256=self.audio_sha256,
        )


def build_examples(
    manifests: Sequence[Manifest],
    *,
    chunk_length_s: float,
    min_chunk_length_s: float,
    timestamp_sample_fraction: float,
    seed: int,
    registry: LicenseRegistry,
    role: str = "train",
) -> list[TrainExample]:
    """Select licensed train or eval examples in the supported duration range."""
    for key, value in (
        ("chunk_length_s", chunk_length_s),
        ("min_chunk_length_s", min_chunk_length_s),
        ("timestamp_sample_fraction", timestamp_sample_fraction),
    ):
        real_number(value, key)
    if not 0 < min_chunk_length_s < chunk_length_s <= 30:
        raise ValueError("need 0 < min_chunk_length_s < chunk_length_s <= 30")
    if not 0 <= timestamp_sample_fraction <= 1:
        raise ValueError("timestamp_sample_fraction must be in [0, 1]")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if role not in {"train", "eval"}:
        raise ValueError("role must be train or eval")
    if not isinstance(registry, LicenseRegistry):
        raise ValueError("registry must be a LicenseRegistry")
    if any(not isinstance(manifest, Manifest) for manifest in manifests):
        raise ValueError("example selection requires Manifest records")
    if not manifests or len({manifest.name for manifest in manifests}) != len(manifests):
        raise ValueError("need distinct named manifests")
    for manifest in manifests:
        if role == "train" and manifest.split != "train":
            raise ValueError(f"manifest {manifest.name!r}: dev/test data must never enter training")
        if role == "eval" and manifest.split not in {"dev", "test"}:
            raise ValueError("evaluation examples require dev or test manifests")
        manifest.validate_against(registry, training=role == "train")

    kept: list[TrainExample] = []
    for manifest in manifests:
        for segment in manifest.segments:
            if not min_chunk_length_s <= segment.duration_s <= chunk_length_s:
                continue
            if not normalize(segment.text):
                raise ValueError(
                    f"manifest {manifest.name!r}, segment {segment.id!r}: target is empty after "
                    f"evaluation normalization v{NORMALIZER_VERSION}"
                )
            kept.append(
                TrainExample(
                    audio=segment.audio,
                    text=segment.text,
                    duration_s=segment.duration_s,
                    recording_id=segment.recording_id,
                    source=manifest.source,
                    start_s=segment.start_s,
                    id=segment.id,
                    split=manifest.split,
                    language=segment.language,
                    audio_sha256=segment.audio_sha256,
                )
            )

    if not kept:
        raise ValueError(f"no {role} examples survive the duration bounds")
    identities = [(e.source, e.id) for e in kept]
    windows = [(e.audio, e.start_s or 0.0, e.duration_s) for e in kept]
    if len(set(identities)) != len(identities) or len(set(windows)) != len(windows):
        raise ValueError("duplicate training/evaluation items or audio windows")
    kept.sort(key=lambda e: (e.source, e.recording_id, e.audio, e.start_s or 0.0, e.id))
    n_timestamped = round(timestamp_sample_fraction * len(kept))
    if n_timestamped:
        rng = random.Random(seed)
        chosen = set(rng.sample(range(len(kept)), n_timestamped))
        kept = [
            replace(example, use_timestamps=True) if i in chosen else example
            for i, example in enumerate(kept)
        ]
    return kept


def total_hours(examples: Sequence[TrainExample]) -> float:
    """Selected training duration in hours."""
    duration = sum(e.duration_s for e in examples)
    if not math.isfinite(duration):
        raise ValueError("total training duration must be finite")
    return duration / 3600.0
