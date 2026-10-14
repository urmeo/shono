"""Lazy HF Whisper inference over validated windows no longer than 30 seconds."""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path

from shono.checkpoints import hash_checkpoint
from shono.data.audio_paths import validate_audio_window

TARGET_SAMPLE_RATE = 16_000


class ShortFormWhisperTranscriber:
    def __init__(
        self,
        model_id: str,
        *,
        language: str = "bn",
        task: str = "transcribe",
        device: str | None = None,
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a Hub repository ID or local checkpoint directory")
        self.model_id = model_id
        self.language = language
        self.task = task
        self.device = device
        self._model = None
        self._processor = None
        self._checkpoint = None

    def observed_runtime_context(self) -> dict:
        """Return the loaded checkpoint identity and decoding settings."""
        model = self._model
        generation = getattr(model, "generation_config", None)
        config = getattr(model, "config", None)
        return {
            "model_id": self.model_id,
            "model_revision": getattr(config, "_commit_hash", None) or "unknown",
            "checkpoint": deepcopy(self._checkpoint) if self._checkpoint is not None else "unknown",
            "device": self.device or "unknown",
            "language": self.language,
            "task": self.task,
            "generation_config": generation.to_dict() if generation is not None else "unknown",
        }

    def _load(self):
        if self._model is None:
            path = Path(self.model_id).expanduser()
            local = path.exists()
            if local and not path.is_dir():
                raise ValueError("local model_id must be a checkpoint directory")
            if not local and (path.is_absolute() or self.model_id.startswith(("./", "../", "~"))):
                raise ValueError("local model_id checkpoint directory does not exist")
            if not local and "://" in self.model_id:
                raise ValueError(
                    "model_id must be a Hub repository ID or local checkpoint directory"
                )
            source = str(path.resolve()) if local else self.model_id
            identity = hash_checkpoint(path) if local else None
            load_options = {"local_files_only": True} if local else {}
            import torch
            from transformers import (
                WhisperConfig,
                WhisperForConditionalGeneration,
                WhisperProcessor,
            )

            config = WhisperConfig.from_pretrained(source, **load_options)
            if not local:
                revision = getattr(config, "_commit_hash", None)
                if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
                    raise ValueError("Hub checkpoint configuration has no immutable commit SHA")
                load_options = {"revision": revision}
                identity = {"source": "hub", "model_id": self.model_id, "revision": revision}
            model = WhisperForConditionalGeneration.from_pretrained(
                source, config=config, **load_options
            )
            processor = WhisperProcessor.from_pretrained(source, **load_options)
            if local and hash_checkpoint(path) != identity:
                raise ValueError("local checkpoint changed while loading")
            model.generation_config.forced_decoder_ids = None
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            self._processor = processor
            self._model = model
            self._checkpoint = identity
            self.device = device
        return self._model, self._processor

    def transcribe(
        self, audio_path: str, start_s: float | None = None, duration_s: float | None = None
    ) -> str:
        actual = validate_audio_window(audio_path, start_s, duration_s)
        if actual > 30:
            raise ValueError("short-form Whisper requires audio windows <= 30 s")
        import librosa
        import numpy as np
        import torch

        audio, _ = librosa.load(
            audio_path,
            sr=TARGET_SAMPLE_RATE,
            offset=start_s if start_s is not None else 0,
            duration=actual,
            mono=True,
        )
        if len(audio) == 0 or not np.isfinite(audio).all():
            raise ValueError("decoded audio must contain finite samples")
        model, processor = self._load()
        inputs = processor(
            audio,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
        )
        features = inputs.input_features.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        with torch.no_grad():
            ids = model.generate(
                features, attention_mask=attention_mask, language=self.language, task=self.task
            )
        return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
