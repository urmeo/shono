"""Bengali ASR evaluation: versioned normalization, WER/CER, and bootstrap intervals."""

from importlib import import_module

from shono.eval.ci import BootstrapCI, blockwise_bootstrap_ci
from shono.eval.codeswitch import (
    CS_NORMALIZER_VERSION,
    bengali_fraction,
    code_switch_normalize,
    is_code_switched,
)
from shono.eval.normalize import NORMALIZER_VERSION, normalize
from shono.eval.score import ScoreReport, cer, score, wer

__all__ = [
    "CS_NORMALIZER_VERSION",
    "NORMALIZER_VERSION",
    "BootstrapCI",
    "Predictions",
    "bengali_fraction",
    "code_switch_normalize",
    "is_code_switched",
    "Report",
    "ReportSpec",
    "ScoreReport",
    "SliceScore",
    "blockwise_bootstrap_ci",
    "build_report",
    "cer",
    "normalize",
    "score",
    "score_slice",
    "wer",
    "write_report",
]


def __getattr__(name: str):
    # Report metadata depends on provenance, which also imports the normalizer.
    if name in {
        "Predictions",
        "Report",
        "ReportSpec",
        "SliceScore",
        "build_report",
        "score_slice",
        "write_report",
    }:
        return getattr(import_module("shono.eval.report"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
