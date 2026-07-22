"""Audio → log-mel features + label ids — the torch-dependent half of the dataloader.

Loads each example's audio (resampling to 16 kHz), extracts Whisper's log-mel
features, and tokenizes the target. When an example is marked ``use_timestamps``
(a fraction are, so the model does not forget timestamps), the target is wrapped
with **segment-level** timestamp tokens ``<|0.00|> … <|end|>`` — word-level
alignment is out of scope here. torch/transformers/librosa are imported lazily,
so importing this module never requires them; only constructing the dataset does.

The first real run must sanity-check a few timestamped targets (a known Whisper
fine-tuning failure mode is timestamp forgetting) — that check lives in the
training notebook.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from shono.train.config import TrainConfig
from shono.train.data import TrainExample

TARGET_SAMPLE_RATE = 16_000
_TIMESTAMP_RESOLUTION_S = 0.02  # Whisper emits one timestamp token per 20 ms


def _load_audio(path: Path, start_s: float | None, duration_s: float):
    import librosa

    offset = start_s or 0.0
    audio, _ = librosa.load(
        path, sr=TARGET_SAMPLE_RATE, offset=offset, duration=duration_s, mono=True
    )
    return audio


def _timestamp_token_id(tokenizer: Any, seconds: float) -> int:
    """Whisper timestamp token id for ``seconds`` (0.02 s grid from the ``<|0.00|>`` base)."""
    base = tokenizer.convert_tokens_to_ids("<|0.00|>")
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
    ) -> None:
        self.examples = list(examples)
        self.processor = processor
        self.config = config
        self.audio_root = Path(audio_root)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        audio = _load_audio(
            self.audio_root / example.audio, example.start_s, example.duration_s
        )
        features = self.processor.feature_extractor(
            audio, sampling_rate=TARGET_SAMPLE_RATE
        ).input_features[0]
        labels = self._labels_for(example)
        return {"input_features": features, "labels": labels}

    def _labels_for(self, example: TrainExample) -> list[int]:
        tokenizer = self.processor.tokenizer
        if not example.use_timestamps:
            return tokenizer(example.text).input_ids
        # Segment-level timestamps: prefix tokens + <|0.00|> text <|end|> + eos.
        prefix = list(tokenizer.prefix_tokens)
        text_ids = tokenizer(example.text, add_special_tokens=False).input_ids
        end = min(example.duration_s, self.config.chunk_length_s)
        start_tok = _timestamp_token_id(tokenizer, 0.0)
        end_tok = _timestamp_token_id(tokenizer, end)
        return [*prefix, start_tok, *text_ids, end_tok, tokenizer.eos_token_id]
