"""Google Cloud Speech-to-Text (Chirp) adapter — the only major cloud with bn-BD.

Gated: needs a Google Cloud project with the Speech-to-Text API enabled and
credentials in the environment. The client is imported lazily, so this module
loads without ``google-cloud-speech`` installed. Free tier is ~60 min/month plus
the new-account credit — confirm current pricing before a run and keep the
:class:`BudgetGuard` in front of it.
"""

from __future__ import annotations

from shono.api.audio import window_wav_bytes


class GoogleChirpTranscriber:
    """Transcribe via Chirp (Speech-to-Text v2) with the Bengali (Bangladesh) locale."""

    def __init__(
        self,
        project_id: str,
        *,
        language: str = "bn-BD",
        model: str = "chirp_2",
        location: str = "us-central1",
    ) -> None:
        self.project_id = project_id
        self.language = language
        self.model = model
        self.location = location
        self._client = None

    def _client_and_recognizer(self):
        if self._client is None:
            from google.api_core.client_options import ClientOptions
            from google.cloud.speech_v2 import SpeechClient

            self._client = SpeechClient(
                client_options=ClientOptions(api_endpoint=f"{self.location}-speech.googleapis.com")
            )
        recognizer = f"projects/{self.project_id}/locations/{self.location}/recognizers/_"
        return self._client, recognizer

    def transcribe(
        self, audio_path: str, start_s: float | None = None, duration_s: float | None = None
    ) -> str:
        from google.cloud.speech_v2.types import cloud_speech

        client, recognizer = self._client_and_recognizer()
        content, _sr = window_wav_bytes(audio_path, start_s, duration_s)
        config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=[self.language],
            model=self.model,
        )
        request = cloud_speech.RecognizeRequest(
            recognizer=recognizer, config=config, content=content
        )
        response = client.recognize(request=request)
        return " ".join(
            r.alternatives[0].transcript for r in response.results if r.alternatives
        ).strip()
