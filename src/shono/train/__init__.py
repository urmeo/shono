"""Training configuration, preparation and lazily imported runtime helpers."""

from shono.train.checkpoint import (
    ResumeDecision,
    decide_resume,
    find_latest_checkpoint,
    is_complete_checkpoint,
    list_checkpoints,
)
from shono.train.config import TrainConfig
from shono.train.data import TrainExample, build_examples, total_hours
from shono.train.experiment import render_experiment
from shono.train.trainer import run_training, smoke_step, to_training_arguments

__all__ = [
    "ResumeDecision",
    "TrainConfig",
    "TrainExample",
    "build_examples",
    "decide_resume",
    "find_latest_checkpoint",
    "is_complete_checkpoint",
    "list_checkpoints",
    "render_experiment",
    "run_training",
    "smoke_step",
    "to_training_arguments",
    "total_hours",
]
