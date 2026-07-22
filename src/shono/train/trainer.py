"""The torch/transformers training glue — imported lazily, run on the GPU.

Everything here needs torch, transformers, and (for the optimizer) bitsandbytes,
none of which are installed in the scoring environment; they live in the Kaggle
training image. So every heavy import is *inside* a function: ``import
shono.train.trainer`` succeeds on the local Mac, and only calling ``run_training``
(or the smoke step) pulls torch in. That keeps ``./verify`` green while the same
code drives the real run.

Two deliberate choices:
    * evaluation during training scores with the **frozen** ``shono.eval``
      normalizer + WER — never Whisper's built-in normalizer, which inflates
      Bengali accuracy by stripping vowel signs.
    * the run is resumable (D-0001): it consults :mod:`shono.train.checkpoint`
      and hands ``resume_from_checkpoint`` to the trainer.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shono.train.checkpoint import decide_resume
from shono.train.config import TrainConfig
from shono.train.data import TrainExample

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime here
    pass

TARGET_SAMPLE_RATE = 16_000


def to_training_arguments(config: TrainConfig, **overrides: Any):
    """Map :class:`TrainConfig` to a ``Seq2SeqTrainingArguments`` (transformers required)."""
    from transformers import Seq2SeqTrainingArguments

    args = {
        "output_dir": config.output_dir,
        "per_device_train_batch_size": config.per_device_batch_size,
        "per_device_eval_batch_size": config.per_device_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "lr_scheduler_type": config.lr_scheduler_type,
        "warmup_steps": config.warmup_steps,
        "weight_decay": config.weight_decay,
        "num_train_epochs": config.num_train_epochs,
        "max_steps": config.max_steps,
        "fp16": config.fp16,
        "gradient_checkpointing": config.gradient_checkpointing,
        "optim": config.optim,
        "save_steps": config.save_steps,
        "eval_steps": config.eval_steps,
        "logging_steps": config.logging_steps,
        "save_total_limit": config.save_total_limit,
        "predict_with_generate": True,
        "seed": config.seed,
        "report_to": [],
    }
    args.update(overrides)
    return Seq2SeqTrainingArguments(**args)


class DataCollatorSpeechSeq2Seq:
    """Pad log-mel features and label ids into a batch (the standard Whisper collator)."""

    def __init__(self, processor: Any, decoder_start_token_id: int) -> None:
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features: list[dict]) -> dict:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def build_compute_metrics(processor: Any):
    """A ``compute_metrics`` that scores with the frozen normalizer, not Whisper's."""
    from shono.eval import score

    def compute_metrics(pred) -> dict:
        import numpy as np

        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        ref_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        report = score(ref_str, pred_str)
        return {"wer": report.wer_normalized, "cer": report.cer_normalized}

    return compute_metrics


def run_training(
    config: TrainConfig,
    train_examples: Sequence[TrainExample],
    eval_examples: Sequence[TrainExample],
    *,
    audio_root: str | Path = ".",
):
    """Fine-tune ``config.model_id`` on the given examples, resuming if possible.

    Runs on a GPU with torch/transformers/bitsandbytes installed. Returns the
    trainer's final metrics. The heavy lifting (model load, audio decoding,
    training loop) happens here; the plumbing that shapes it is tested elsewhere.
    """
    from transformers import (
        Seq2SeqTrainer,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    from shono.train.audio import WhisperFineTuneDataset

    processor = WhisperProcessor.from_pretrained(
        config.model_id, language=config.language, task=config.task
    )
    model = WhisperForConditionalGeneration.from_pretrained(config.model_id)
    model.generation_config.language = config.language
    model.generation_config.task = config.task
    if config.gradient_checkpointing:
        model.config.use_cache = False

    train_ds = WhisperFineTuneDataset(train_examples, processor, config, audio_root=audio_root)
    eval_ds = WhisperFineTuneDataset(eval_examples, processor, config, audio_root=audio_root)
    collator = DataCollatorSpeechSeq2Seq(processor, model.config.decoder_start_token_id)

    resume = decide_resume(config.output_dir, enabled=config.resume)
    trainer = Seq2SeqTrainer(
        model=model,
        args=to_training_arguments(config),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=build_compute_metrics(processor),
        processing_class=processor,
    )
    result = trainer.train(
        resume_from_checkpoint=str(resume.checkpoint) if resume.resume else None
    )
    trainer.save_model(config.output_dir)
    return result.metrics


def smoke_step(output_dir: str | Path) -> dict:
    """Run one training step on CPU with a tiny model — no download, no audio files.

    Exercises the real plumbing (config → training args, model forward/backward on
    log-mel inputs, optimizer step, checkpoint save+reload) in seconds, so a broken
    training loop is caught before a multi-hour GPU run. Requires torch +
    transformers; skipped where they are absent.
    """
    import torch
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tiny = WhisperConfig(
        vocab_size=64,
        num_mel_bins=80,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=1,
        decoder_attention_heads=1,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_target_positions=48,
        max_source_positions=1500,
        decoder_start_token_id=1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = WhisperForConditionalGeneration(tiny)
    model.train()

    input_features = torch.randn(2, 80, 3000)
    labels = torch.randint(3, 64, (2, 8))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    out_before = model(input_features=input_features, labels=labels)
    loss = out_before.loss
    loss.backward()
    optimizer.step()

    ckpt = out / "checkpoint-1"
    model.save_pretrained(ckpt)
    reloaded = WhisperForConditionalGeneration.from_pretrained(ckpt)

    return {
        "loss": float(loss.detach()),
        "reloaded_params": sum(p.numel() for p in reloaded.parameters()),
        "checkpoint": str(ckpt),
    }
