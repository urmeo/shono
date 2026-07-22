"""Whisper over one audio window (HF transformers) — the baseline transcriber.

Zero-shot large-v3/turbo, the tugstugi Bengali base, or our own fine-tune: each is
just a Whisper checkpoint transcribing a window. This satisfies the same
``transcribe(audio, start_s, duration_s) -> str`` protocol the API adapters use, so
``shono.api.run_over_manifest`` turns any of them into a Predictions set — one
prediction path for baselines, ours, and commercial APIs alike.

torch/transformers/librosa are imported lazily; the model loads once on first use.
"""

from __future__ import annotations

TARGET_SAMPLE_RATE = 16_000


class ShortFormWhisperTranscriber:
    """Transcribe an audio window with a Whisper checkpoint (per-segment, short-form)."""

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

    def _load(self):
        if self._model is None:
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            self._processor = WhisperProcessor.from_pretrained(self.model_id)
            self._model = WhisperForConditionalGeneration.from_pretrained(self.model_id)
            self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self.device)
            self._model.eval()
        return self._model, self._processor

    def transcribe(
        self, audio_path: str, start_s: float | None = None, duration_s: float | None = None
    ) -> str:
        import librosa
        import torch

        model, processor = self._load()
        audio, _ = librosa.load(
            audio_path, sr=TARGET_SAMPLE_RATE, offset=start_s or 0.0,
            duration=duration_s, mono=True,
        )
        features = processor(
            audio, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(self.device)
        with torch.no_grad():
            ids = model.generate(features, language=self.language, task=self.task)
        return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
