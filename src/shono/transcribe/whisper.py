"""Lazy HF Whisper inference over validated windows no longer than 30 seconds."""

from __future__ import annotations

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
        self.model_id = model_id
        self.language = language
        self.task = task
        self.device = device
        self._model = None
        self._processor = None

    def observed_runtime_context(self) -> dict:
        """Return loaded decoding settings; missing revisions remain unknown."""
        model = self._model
        generation = getattr(model, "generation_config", None)
        config = getattr(model, "config", None)
        return {
            "model_id": self.model_id,
            "model_revision": getattr(config, "_commit_hash", None) or "unknown",
            "device": self.device or "unknown",
            "language": self.language,
            "task": self.task,
            "generation_config": generation.to_dict() if generation is not None else "unknown",
        }

    def _load(self):
        if self._model is None:
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            processor = WhisperProcessor.from_pretrained(self.model_id)
            model = WhisperForConditionalGeneration.from_pretrained(self.model_id)
            model.generation_config.forced_decoder_ids = None
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            self._processor = processor
            self._model = model
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
