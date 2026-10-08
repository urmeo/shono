"""Training orchestration with manifest preflight and generated evaluation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from shono.data.audio_paths import DurationFn, resolved_path, validate_manifest_audio
from shono.data.leakage import require_no_leakage
from shono.data.license import LicenseRegistry
from shono.data.manifest import Manifest
from shono.train.checkpoint import decide_resume
from shono.train.config import TrainConfig
from shono.train.data import build_examples
from shono.train.output import validate_training_output

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
        "eval_strategy": "steps",
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
        if not features or any(not f["labels"] for f in features):
            raise ValueError("collator requires nonempty items and labels")
        input_features = [
            {"input_features": f["input_features"], "attention_mask": f["attention_mask"]}
            for f in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        starts = labels[:, 0] == self.decoder_start_token_id
        if starts.any().cpu().item() and not starts.all().cpu().item():
            raise ValueError("batch mixes decoder-start conventions")
        if starts.all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def build_compute_metrics(processor: Any):
    """A ``compute_metrics`` that scores with the frozen normalizer, not Whisper's."""
    from shono.eval import score

    def compute_metrics(pred) -> dict:
        import numpy as np

        pred_ids = np.where(
            pred.predictions == -100, processor.tokenizer.pad_token_id, pred.predictions
        )
        label_ids = pred.label_ids
        label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        ref_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        report = score(ref_str, pred_str)
        return {"wer": report.wer_normalized, "cer": report.cer_normalized}

    return compute_metrics


def run_training(
    config: TrainConfig,
    train_manifests: Sequence[Manifest],
    eval_manifests: Sequence[Manifest],
    *,
    registry: LicenseRegistry,
    audio_root: str | Path,
    held_out_manifests: Sequence[Manifest] = (),
    text_decisions: Mapping[str, Mapping[str, str]] | None = None,
    duration_of: DurationFn | None = None,
):
    """Validate source manifests, train, evaluate and save. No pretrained calls occur before data
    preflight."""
    if not isinstance(config, TrainConfig):
        raise ValueError("config must be TrainConfig")
    all_manifests = [*train_manifests, *eval_manifests, *held_out_manifests]
    if not all_manifests or any(not isinstance(m, Manifest) for m in all_manifests):
        raise ValueError("training requires source manifests, not detached examples")
    if not eval_manifests or any(m.split != "dev" for m in eval_manifests):
        raise ValueError("training evaluation requires development manifests")
    if len({m.name for m in all_manifests}) != len(all_manifests):
        raise ValueError("training/development/held-out manifest names must be distinct")
    selection = dict(
        chunk_length_s=config.chunk_length_s,
        min_chunk_length_s=config.min_chunk_length_s,
        timestamp_sample_fraction=config.timestamp_sample_fraction,
        seed=config.seed,
        registry=registry,
    )
    train_examples = build_examples(train_manifests, **selection)
    selection["timestamp_sample_fraction"] = 0.0
    eval_examples = build_examples(eval_manifests, role="eval", **selection)
    decisions = {} if text_decisions is None else text_decisions
    evaluation = [*eval_manifests, *held_out_manifests]
    if not isinstance(decisions, Mapping) or set(decisions) - {m.name for m in evaluation}:
        raise ValueError("text_decisions has unknown evaluation manifest names")
    for manifest in all_manifests:
        manifest.validate_against(registry, training=manifest in train_manifests)
        if any(seg.language != config.language for seg in manifest.segments):
            raise ValueError(f"manifest {manifest.name!r}: language conflicts with training config")
        validate_manifest_audio(manifest, audio_root, duration_of=duration_of)
    audits = [
        require_no_leakage(
            train_manifests,
            manifest,
            text_decisions=decisions.get(manifest.name),
            audio_root=audio_root,
        )
        for manifest in evaluation
    ]
    out = validate_training_output(config.output_dir, audio_root)
    resume = decide_resume(out, enabled=config.resume)
    identity = json.loads(
        json.dumps(
            {
                "config": config.to_dict(),
                "manifests": [asdict(m) for m in all_manifests],
                "leakage": [asdict(report) for report in audits],
            },
            allow_nan=False,
        )
    )
    identity_path = out / "training-inputs.json"
    if not resolved_path(identity_path).is_relative_to(out):
        raise ValueError("training identity path escapes output_dir")
    if config.resume and identity_path.exists():
        try:
            prior = json.loads(identity_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise ValueError("cannot read prior training identity") from exc
        if prior != identity:
            raise ValueError(
                "output_dir contains a different training identity; choose a new output"
            )
    elif resume.resume:
        raise ValueError("checkpoint has no declared training identity; choose a new output")
    from transformers import (
        Seq2SeqTrainer,
        WhisperConfig,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    from shono.train.audio import WhisperFineTuneDataset

    processor = WhisperProcessor.from_pretrained(
        config.model_id, language=config.language, task=config.task
    )
    model_config = WhisperConfig.from_pretrained(config.model_id)
    dataset_options = dict(
        audio_root=audio_root,
        max_target_positions=model_config.max_target_positions,
        decoder_start_token_id=model_config.decoder_start_token_id,
    )
    train_ds = WhisperFineTuneDataset(train_examples, processor, config, **dataset_options)
    eval_ds = WhisperFineTuneDataset(eval_examples, processor, config, **dataset_options)
    train_ds.validate_labels()
    eval_ds.validate_labels()
    model = WhisperForConditionalGeneration.from_pretrained(config.model_id)
    model.generation_config.language = config.language
    model.generation_config.task = config.task
    model.generation_config.return_timestamps = False
    model.generation_config.max_length = model_config.max_target_positions
    model.generation_config.forced_decoder_ids = None
    if config.gradient_checkpointing:
        model.config.use_cache = False
    if config.freeze_encoder:
        if not callable(getattr(model, "freeze_encoder", None)):
            raise ValueError("model does not support freeze_encoder")
        model.freeze_encoder()
    collator = DataCollatorSpeechSeq2Seq(processor, model.config.decoder_start_token_id)

    out.mkdir(parents=True, exist_ok=True)
    identity_path.write_text(
        json.dumps(identity, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=to_training_arguments(config),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=build_compute_metrics(processor),
        processing_class=processor,
    )
    result = trainer.train(resume_from_checkpoint=str(resume.checkpoint) if resume.resume else None)
    evaluation_metrics = trainer.evaluate()
    trainer.save_model(config.output_dir)
    return {**result.metrics, **evaluation_metrics}


def smoke_step(output_dir: str | Path) -> dict:
    """Check CPU plumbing with a randomly initialized tiny model."""
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
