"""Turning manifests into training examples — the torch-free half of the dataloader.

This module decides *what* to train on: it filters manifest segments to the
trainable duration range, merges the source mix, and marks which examples carry
timestamped targets (so Whisper does not forget timestamps during the fine-tune).
Loading the actual audio into tensors is the torch-dependent half and lives in
``shono.train.trainer``; keeping the selection logic here makes it unit-testable
without a GPU and guarantees the leakage guard runs on every build.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from shono.data.manifest import Manifest


@dataclass(frozen=True)
class TrainExample:
    """One training item: a span of audio, its target text, and whether to train timestamps."""

    audio: str
    text: str
    duration_s: float
    recording_id: str
    source: str
    start_s: float | None = None
    use_timestamps: bool = False


def build_examples(
    manifests: Sequence[Manifest],
    *,
    chunk_length_s: float,
    min_chunk_length_s: float,
    timestamp_sample_fraction: float,
    seed: int,
) -> list[TrainExample]:
    """Build the training example list from a mix of manifests.

    Segments outside ``[min_chunk_length_s, chunk_length_s]`` are dropped (too
    short to be useful, too long for the model's receptive field). A deterministic
    ``timestamp_sample_fraction`` of the survivors is marked to train timestamped
    targets. Raises if any manifest is a ``test`` split — one specific guard against
    the most obvious leakage; the license floor (``Manifest.validate_against``, which
    also blocks eval-only sources) and the leakage audit cover the rest.
    """
    for m in manifests:
        if m.split == "test":
            raise ValueError(
                f"manifest {m.name!r} is a 'test' split; test data must never enter training"
            )

    kept: list[TrainExample] = []
    for m in manifests:
        for seg in m.segments:
            if min_chunk_length_s <= seg.duration_s <= chunk_length_s:
                kept.append(
                    TrainExample(
                        audio=seg.audio,
                        text=seg.text,
                        duration_s=seg.duration_s,
                        recording_id=seg.recording_id,
                        source=m.source,
                        start_s=seg.start_s,
                    )
                )

    kept.sort(key=lambda e: (e.source, e.recording_id, e.audio, e.start_s or 0.0))
    n_timestamped = int(round(timestamp_sample_fraction * len(kept)))
    if n_timestamped:
        rng = random.Random(seed)
        chosen = set(rng.sample(range(len(kept)), n_timestamped))
        kept = [
            (e if i not in chosen else _with_timestamps(e)) for i, e in enumerate(kept)
        ]
    return kept


def _with_timestamps(example: TrainExample) -> TrainExample:
    return TrainExample(
        audio=example.audio,
        text=example.text,
        duration_s=example.duration_s,
        recording_id=example.recording_id,
        source=example.source,
        start_s=example.start_s,
        use_timestamps=True,
    )


def total_hours(examples: Sequence[TrainExample]) -> float:
    """Total training audio, in hours — for the experiment record."""
    return sum(e.duration_s for e in examples) / 3600.0
