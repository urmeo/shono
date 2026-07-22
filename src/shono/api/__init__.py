"""Commercial ASR API benchmark: one interface, a $0-spend guard, real adapters.

The protocol, budget guard, and manifest runner are pure and imported eagerly; the
Google Chirp and Deepgram adapters keep their SDK imports inside their methods, so
this package loads without any cloud client installed.
"""

from shono.api.base import (
    ApiTranscriber,
    BudgetError,
    BudgetGuard,
    run_over_manifest,
)
from shono.api.deepgram import DeepgramTranscriber
from shono.api.google_chirp import GoogleChirpTranscriber

__all__ = [
    "ApiTranscriber",
    "BudgetError",
    "BudgetGuard",
    "DeepgramTranscriber",
    "GoogleChirpTranscriber",
    "run_over_manifest",
]
