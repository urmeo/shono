"""The fine-tuning recipe, made explicit and validated.

Whisper-medium is fine-tuned *fully* (not LoRA) with an 8-bit AdamW optimizer —
the configuration that fit a free T4 and beat LoRA on Bengali long-form in the
2026 literature. This module holds that recipe as a validated dataclass so the
training notebook is a thin driver and every hyperparameter is one reviewed,
version-controlled value rather than a magic number buried in a cell.

The heavy training code (``shono.train.trainer``) imports torch/transformers
lazily; this config module is pure Python and fully testable without a GPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Optimizers we support; 8-bit AdamW (bitsandbytes) is the default — it halves
# optimizer-state memory, which is what lets full fine-tuning fit a 16 GB T4.
_OPTIMIZERS = frozenset({"adamw_bnb_8bit", "adamw_torch", "adafactor"})
_SCHEDULERS = frozenset({"cosine", "linear", "constant_with_warmup"})


@dataclass(frozen=True)
class TrainConfig:
    """Everything that defines a training run, reproducibly.

    ``model_id`` is required and has no default on purpose: the base checkpoint is
    a decision (the Bengali whisper-medium fine-tune), and defaulting it risks
    silently training from the wrong weights.
    """

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
    gradient_accumulation_steps: int = 2  # effective batch 16 on one T4
    num_train_epochs: float = 5.0
    max_steps: int = -1  # -1 = epoch-based

    # audio / features
    chunk_length_s: float = 28.0
    min_chunk_length_s: float = 1.0
    # Keep a fraction of timestamped targets in the mix: fine-tuning on
    # untimed text alone makes Whisper forget timestamps, which breaks long-form.
    timestamp_sample_fraction: float = 0.5

    # memory / precision — the T4-fitting knobs
    fp16: bool = True
    gradient_checkpointing: bool = True
    freeze_encoder: bool = False  # full fine-tune, not LoRA/frozen

    # checkpointing (resume is mandatory to survive session limits)
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
        if self.num_train_epochs <= 0 and self.max_steps <= 0:
            raise ValueError("set num_train_epochs > 0 or max_steps > 0")
        # Upper bound 30 s: Whisper's receptive field, and past it the segment-level
        # timestamp token (<|30.00|>) would collide into non-timestamp vocabulary.
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
