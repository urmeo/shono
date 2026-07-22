"""Deepgram (Nova-3) adapter — Bengali added in 2026, generous free credit.

Gated: needs a Deepgram API key. The SDK is imported lazily, so this module loads
without ``deepgram-sdk`` installed. Deepgram gives a sizeable free credit and
needs no card to start — confirm the current allowance and keep the
:class:`BudgetGuard` in front of any run.
"""

from __future__ import annotations

from shono.api.audio import window_wav_bytes


class DeepgramTranscriber:
    """Transcribe via Deepgram's pre-recorded API with the Nova-3 model."""

    def __init__(self, api_key: str, *, model: str = "nova-3", language: str = "bn") -> None:
        self.api_key = api_key
        self.model = model
        self.language = language
        self._client = None

    def _get_client(self):
        if self._client is None:
            from deepgram import DeepgramClient

            self._client = DeepgramClient(self.api_key)
        return self._client

    def transcribe(
        self, audio_path: str, start_s: float | None = None, duration_s: float | None = None
    ) -> str:
        from deepgram import FileSource, PrerecordedOptions

        content, _sr = window_wav_bytes(audio_path, start_s, duration_s)
        source: FileSource = {"buffer": content, "mimetype": "audio/wav"}
        options = PrerecordedOptions(model=self.model, language=self.language, smart_format=True)
        response = self._get_client().listen.rest.v("1").transcribe_file(source, options)
        return response.results.channels[0].alternatives[0].transcript.strip()
