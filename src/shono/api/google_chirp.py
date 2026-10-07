"""Lazy Chirp 2 synchronous adapter for audio windows shorter than 60 seconds."""

from __future__ import annotations

from shono.api.audio import window_wav_bytes


class GoogleChirpTranscriber:
    requires_budget = True

    def __init__(
        self,
        project_id: str,
        *,
        language: str = "bn-BD",
        model: str = "chirp_2",
        location: str = "us-central1",
    ) -> None:
        if not isinstance(project_id, str) or not project_id.strip():
            raise ValueError("a Google Cloud project ID is required")
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
        content, _sr = window_wav_bytes(audio_path, start_s, duration_s, max_duration_s=60)
        from google.cloud.speech_v2.types import cloud_speech

        client, recognizer = self._client_and_recognizer()
        config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=[self.language],
            model=self.model,
        )
        request = cloud_speech.RecognizeRequest(
            recognizer=recognizer,
            config=config,
            content=content,
        )
        response = client.recognize(request=request, retry=None)
        return " ".join(
            r.alternatives[0].transcript for r in response.results if r.alternatives
        ).strip()
