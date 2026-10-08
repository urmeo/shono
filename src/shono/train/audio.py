"""Load aligned audio, preserve masks and bound decoder targets."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from shono.data.audio_paths import resolved_path, validate_audio_window
from shono.data.validation import real_number
from shono.train.config import TrainConfig
from shono.train.data import TrainExample

TARGET_SAMPLE_RATE = 16_000
_TIMESTAMP_RESOLUTION_S = 0.02


def _load_audio(path: Path, start_s: float | None, duration_s: float):
    import librosa
    import numpy as np

    selected = validate_audio_window(path, start_s, duration_s)
    offset = 0.0 if start_s is None else start_s
    audio, _ = librosa.load(
        path, sr=TARGET_SAMPLE_RATE, offset=offset, duration=selected, mono=True
    )
    if audio.ndim != 1 or not len(audio) or np.iscomplexobj(audio) or not np.isfinite(audio).all():
        raise ValueError("decoded audio must contain finite mono samples")
    if abs(len(audio) / TARGET_SAMPLE_RATE - selected) > 0.001:
        raise ValueError("decoded audio sample count does not match the selected window")
    return audio


def _timestamp_token_id(tokenizer: Any, seconds: float) -> int:
    """Whisper timestamp token id for ``seconds`` (0.02 s grid from the ``<|0.00|>`` base)."""
    seconds = real_number(seconds, "timestamp seconds")
    if not 0 <= seconds <= 30:
        raise ValueError("timestamp seconds must be in [0, 30]")
    base = tokenizer.convert_tokens_to_ids("<|0.00|>")
    if type(base) is not int or base == getattr(tokenizer, "unk_token_id", None):
        raise ValueError("tokenizer does not support Whisper timestamp tokens")
    steps = int(round(seconds / _TIMESTAMP_RESOLUTION_S))
    return base + max(steps, 0)


class WhisperFineTuneDataset:
    """A torch-style dataset yielding ``{"input_features", "labels"}`` per example."""

    def __init__(
        self,
        examples: Sequence[TrainExample],
        processor: Any,
        config: TrainConfig,
        *,
        audio_root: str | Path = ".",
        max_target_positions: int,
        decoder_start_token_id: int,
    ) -> None:
        self.examples = list(examples)
        self.processor = processor
        self.config = config
        self.audio_root = resolved_path(audio_root)
        if type(max_target_positions) is not int or max_target_positions < 1:
            raise ValueError("max_target_positions must be a positive integer")
        self.max_target_positions = max_target_positions
        self.decoder_start_token_id = decoder_start_token_id
        self.paths = []
        for example in self.examples:
            if not config.min_chunk_length_s <= example.duration_s <= config.chunk_length_s:
                raise ValueError("example duration is outside the supported training window")
            path = resolved_path(self.audio_root / example.audio)
            if not path.is_relative_to(self.audio_root) or not path.is_file():
                raise ValueError(f"example {example.id!r}: missing/escaping audio file")
            self.paths.append(path)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        audio = _load_audio(self.paths[index], example.start_s, example.duration_s)
        encoded = self.processor.feature_extractor(
            audio, sampling_rate=TARGET_SAMPLE_RATE, return_attention_mask=True
        )
        features = encoded.input_features[0]
        labels = self._labels_for(example)
        return {
            "input_features": features,
            "attention_mask": encoded.attention_mask[0],
            "labels": labels,
        }

    def _labels_for(self, example: TrainExample) -> list[int]:
        tokenizer = self.processor.tokenizer
        if not example.use_timestamps:
            labels = tokenizer(example.text).input_ids
            return self._validate_labels(labels, example)
        no_ts = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        prefix = [t for t in tokenizer.prefix_tokens if t != no_ts]
        text_ids = tokenizer(example.text, add_special_tokens=False).input_ids
        end = example.duration_s
        if end > self.config.chunk_length_s:
            raise ValueError("example duration exceeds training chunk_length_s")
        start_tok = _timestamp_token_id(tokenizer, 0.0)
        end_tok = _timestamp_token_id(tokenizer, end)
        return self._validate_labels(
            [*prefix, start_tok, *text_ids, end_tok, tokenizer.eos_token_id], example
        )

    def _validate_labels(self, labels: list[int], example: TrainExample) -> list[int]:
        if not labels or any(type(token) is not int or token < 0 for token in labels):
            raise ValueError("target must contain nonnegative integer token IDs")
        count = len(labels) - (labels[0] == self.decoder_start_token_id)
        if count > self.max_target_positions:
            raise ValueError(
                f"example {example.source}:{example.id}: target has {count} tokens; "
                f"model capacity is {self.max_target_positions}"
            )
        return labels

    def validate_labels(self) -> None:
        for example in self.examples:
            self._labels_for(example)
