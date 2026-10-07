"""Speaker diarization: DER scoring, the diarizer interface, and speaker attribution.

DER scoring, VAD-intersection, and speaker attribution are pure and imported
eagerly; the pyannote implementation keeps its torch import inside the class, so
this package loads without a GPU.
"""

from shono.diarize.attribute import attribute_speakers
from shono.diarize.der import (
    DERConfig,
    DERInterval,
    DERResult,
    corpus_der,
    der,
    der_bootstrap_ci,
    render_der_report,
)
from shono.diarize.diarizer import Diarizer, PyannoteDiarizer, vad_intersection
from shono.diarize.rttm import load_loop_csv, load_rttm
from shono.diarize.types import SpeakerSegment

__all__ = [
    "DERConfig",
    "DERInterval",
    "DERResult",
    "Diarizer",
    "PyannoteDiarizer",
    "SpeakerSegment",
    "attribute_speakers",
    "corpus_der",
    "der",
    "der_bootstrap_ci",
    "load_rttm",
    "load_loop_csv",
    "render_der_report",
    "vad_intersection",
]
