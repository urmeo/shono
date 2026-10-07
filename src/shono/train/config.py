"""Validated training settings; hardware fit and convergence are unmeasured."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from shono.data.validation import real_number, text_value

# Supported optimizer and scheduler identifiers.
_OPTIMIZERS = frozenset({"adamw_bnb_8bit", "adamw_torch", "adafactor"})
_SCHEDULERS = frozenset({"cosine", "linear", "constant_with_warmup"})


@dataclass(frozen=True)
class TrainConfig:
    """Declared settings with an explicit base checkpoint."""

    model_id: str
    output_dir: str
    language: str = "bn"
    task: str = "transcribe"

    # optimization
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 500
    weight_decay: float = 0.0
    optim: str = "adamw_bnb_8bit"

    # batch / schedule
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    num_train_epochs: float = 5.0
    max_steps: int = -1  # -1 = epoch-based

    # audio / features
    chunk_length_s: float = 28.0
    min_chunk_length_s: float = 1.0
    # Fraction of aligned examples given timestamp targets.
    timestamp_sample_fraction: float = 0.5

    # Precision and memory options.
    fp16: bool = True
    gradient_checkpointing: bool = True
    freeze_encoder: bool = False  # full fine-tune, not LoRA/frozen

    # Saved-step selection and evaluation schedule.
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 25
    save_total_limit: int = 3
    resume: bool = True

    seed: int = 42

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id is required (the base checkpoint to fine-tune from)")
        if not self.output_dir:
            raise ValueError("output_dir is required (where checkpoints are written)")
        text_value(self.model_id, "model_id")
        text_value(self.output_dir, "output_dir")
        text_value(self.language, "language")
        for key in ("task", "optim", "lr_scheduler_type"):
            text_value(getattr(self, key), key)
        if self.task not in {"transcribe", "translate"}:
            raise ValueError("task must be transcribe or translate")
        for key in ("fp16", "gradient_checkpointing", "freeze_encoder", "resume"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be a boolean")
        for key in (
            "warmup_steps",
            "per_device_batch_size",
            "gradient_accumulation_steps",
            "max_steps",
            "save_steps",
            "eval_steps",
            "logging_steps",
            "save_total_limit",
            "seed",
        ):
            if type(getattr(self, key)) is not int:
                raise ValueError(f"{key} must be an integer")
        if self.warmup_steps < 0 or not 0 <= self.seed < 2**32:
            raise ValueError("warmup_steps must be nonnegative and seed in [0, 2**32)")
        if any(
            getattr(self, key) < 1
            for key in ("save_steps", "eval_steps", "logging_steps", "save_total_limit")
        ):
            raise ValueError("save/eval/logging steps and save_total_limit must be positive")
        if self.max_steps != -1 and self.max_steps < 1:
            raise ValueError("max_steps must be -1 or a positive integer")
        for key in (
            "learning_rate",
            "weight_decay",
            "num_train_epochs",
            "chunk_length_s",
            "min_chunk_length_s",
            "timestamp_sample_fraction",
        ):
            real_number(getattr(self, key), key)
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.optim not in _OPTIMIZERS:
            raise ValueError(f"optim {self.optim!r} not in {sorted(_OPTIMIZERS)}")
        if self.lr_scheduler_type not in _SCHEDULERS:
            raise ValueError(
                f"lr_scheduler_type {self.lr_scheduler_type!r} not in {sorted(_SCHEDULERS)}"
            )
        if self.per_device_batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("batch size and gradient_accumulation_steps must be >= 1")
        real_number(self.effective_batch_size, "effective_batch_size")
        if self.weight_decay < 0 or self.num_train_epochs < 0:
            raise ValueError("weight_decay and num_train_epochs must be nonnegative")
        if self.num_train_epochs <= 0 and self.max_steps <= 0:
            raise ValueError("set num_train_epochs > 0 or max_steps > 0")
        # Supported Whisper audio/timestamp window.
        if not 0.0 < self.min_chunk_length_s < self.chunk_length_s <= 30.0:
            raise ValueError(
                f"need 0 < min_chunk_length_s ({self.min_chunk_length_s}) "
                f"< chunk_length_s ({self.chunk_length_s}) <= 30.0"
            )
        if not 0.0 <= self.timestamp_sample_fraction <= 1.0:
            raise ValueError("timestamp_sample_fraction must be in [0, 1]")

    @property
    def effective_batch_size(self) -> int:
        """Global batch size = per-device × accumulation (single-GPU assumed)."""
        return self.per_device_batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> dict:
        """A JSON-serializable view for the run's provenance and experiment record."""
        return asdict(self)
