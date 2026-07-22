"""Honest evaluation for Bengali ASR: frozen normalization, WER/CER, bootstrap CIs."""

from shono.eval.ci import BootstrapCI, blockwise_bootstrap_ci
from shono.eval.codeswitch import (
    CS_NORMALIZER_VERSION,
    bengali_fraction,
    code_switch_normalize,
    is_code_switched,
)
from shono.eval.normalize import NORMALIZER_VERSION, normalize
from shono.eval.report import (
    Predictions,
    Report,
    ReportSpec,
    SliceScore,
    build_report,
    score_slice,
    write_report,
)
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
