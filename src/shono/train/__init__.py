"""Fine-tuning: the recipe, dataloading, checkpoint-resume, and the experiment record.

The config, checkpoint logic, example selection, and experiment template are pure
Python and imported eagerly. The torch/transformers training glue
(:func:`run_training`, :func:`smoke_step`, :class:`WhisperFineTuneDataset`) keeps
its heavy imports inside functions, so importing this package never needs a GPU.
"""

from shono.train.checkpoint import (
    ResumeDecision,
    decide_resume,
    find_latest_checkpoint,
    is_complete_checkpoint,
    list_checkpoints,
)
from shono.train.config import AugmentationConfig, TrainConfig
from shono.train.data import TrainExample, build_examples, total_hours
from shono.train.experiment import render_experiment
from shono.train.trainer import run_training, smoke_step, to_training_arguments

__all__ = [
    "AugmentationConfig",
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
