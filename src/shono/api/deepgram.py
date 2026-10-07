"""Lazy Deepgram SDK 7 prerecorded transcription adapter."""

from __future__ import annotations

from shono.api.audio import window_wav_bytes


class DeepgramTranscriber:
    requires_budget = True

    def __init__(self, api_key: str, *, model: str = "nova-3", language: str = "bn") -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("a Deepgram API key is required")
        self.api_key = api_key
        self.model = model
        self.language = language
        self._client = None

    def _get_client(self):
        if self._client is None:
            from deepgram import DeepgramClient

            self._client = DeepgramClient(api_key=self.api_key)
        return self._client

    def transcribe(
        self, audio_path: str, start_s: float | None = None, duration_s: float | None = None
    ) -> str:
        content, _sr = window_wav_bytes(audio_path, start_s, duration_s)
        response = self._get_client().listen.v1.media.transcribe_file(
            request=content,
            model=self.model,
            language=self.language,
            smart_format=True,
            request_options={"max_retries": 0},
        )
        try:
            text = response.results.channels[0].alternatives[0].transcript
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError("Deepgram returned no transcription result") from exc
        if not isinstance(text, str):
            raise RuntimeError("Deepgram returned a non-text transcription")
        return text.strip()
