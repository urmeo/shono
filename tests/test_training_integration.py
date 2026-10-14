"""Exercise the real CPU Trainer with local random weights and synthetic audio."""

import json
import math
import shutil
import wave
from dataclasses import replace

import pytest
from data_fixtures import synthetic_registry

from shono.data import Manifest, Segment
from shono.train import TrainConfig, TrainExample, run_training
from shono.train.audio import WhisperFineTuneDataset
from shono.transcribe import ShortFormWhisperTranscriber


@pytest.mark.filterwarnings(
    "ignore:'(aifc|audioop|sunau)' is deprecated.*:DeprecationWarning:audioread.rawread"
)
@pytest.mark.filterwarnings(
    "ignore:mtime may not be reliable on this filesystem.*:UserWarning:transformers.trainer"
)
def test_real_cpu_training_exports_evaluates_and_resumes(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers", minversion="4.52.4")
    pytest.importorskip("accelerate")
    pytest.importorskip("librosa")
    import numpy as np
    from transformers import (
        Seq2SeqTrainer,
        WhisperConfig,
        WhisperFeatureExtractor,
        WhisperForConditionalGeneration,
        WhisperProcessor,
        WhisperTokenizer,
    )
    from transformers.models.whisper.tokenization_whisper import LANGUAGES, bytes_to_unicode

    from shono.train import trainer as training

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    checkpoint = tmp_path / "base"
    checkpoint.mkdir()
    vocabulary = {symbol: index for index, symbol in enumerate(bytes_to_unicode().values())}
    special_tokens = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *(f"<|{language}|>" for language in LANGUAGES),
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
    ]
    for token in [*special_tokens, *(f"<|{step * 0.02:.2f}|>" for step in range(1501))]:
        vocabulary[token] = len(vocabulary)
    vocabulary_path = checkpoint / "vocab.json"
    vocabulary_path.write_text(json.dumps(vocabulary), encoding="utf-8")
    merges_path = checkpoint / "merges.txt"
    merges_path.write_text("#version: 0.2\n", encoding="utf-8")
    tokenizer = WhisperTokenizer(
        str(vocabulary_path),
        str(merges_path),
        language="bn",
        task="transcribe",
        pad_token="<|endoftext|>",
        additional_special_tokens=special_tokens[1:],
    )
    processor = WhisperProcessor(WhisperFeatureExtractor(chunk_length=1), tokenizer)
    processor.save_pretrained(checkpoint)
    config = WhisperConfig(
        vocab_size=len(tokenizer),
        num_mel_bins=80,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=1,
        decoder_attention_heads=1,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_target_positions=16,
        max_source_positions=50,
        decoder_start_token_id=vocabulary["<|startoftranscript|>"],
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = WhisperForConditionalGeneration(config)
    model.generation_config._from_model_config = False
    model.generation_config.suppress_tokens = []
    model.generation_config.begin_suppress_tokens = []
    model.generation_config.lang_to_id = {"<|bn|>": vocabulary["<|bn|>"]}
    model.generation_config.task_to_id = {"transcribe": vocabulary["<|transcribe|>"]}
    model.generation_config.no_timestamps_token_id = vocabulary["<|notimestamps|>"]
    model.generation_config.is_multilingual = True
    model.save_pretrained(checkpoint)
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    manifests = []
    for index, (split, text) in enumerate((("train", "alpha"), ("dev", "beta"))):
        samples = (np.sin(np.arange(4000) * (index + 1) * 0.1) * 1000).astype("<i2")
        with wave.open(str(audio_root / f"{split}.wav"), "wb") as stream:
            stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            stream.writeframes(samples.tobytes())
        manifests.append(
            Manifest(
                name=split,
                source="synthetic",
                split=split,
                domain="read",
                version="fixture",
                segments=(Segment(split, f"{split}.wav", text, 0.25, split),),
            )
        )
    settings = TrainConfig(
        model_id=str(checkpoint),
        output_dir=str(tmp_path / "trained"),
        optim="adamw_torch",
        fp16=False,
        gradient_checkpointing=False,
        per_device_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=2,
        save_steps=1,
        eval_steps=2,
        warmup_steps=0,
        chunk_length_s=1,
        min_chunk_length_s=0.1,
        timestamp_sample_fraction=0,
        resume=False,
    )
    for target in ("<|endoftext|>", "<|notimestamps|>", "<|0.00|>"):
        invalid = TrainExample("train.wav", target, 0.25, "train", "synthetic")
        dataset = WhisperFineTuneDataset(
            [invalid],
            processor,
            settings,
            audio_root=audio_root,
            max_target_positions=config.max_target_positions,
            decoder_start_token_id=config.decoder_start_token_id,
        )
        with pytest.raises(ValueError, match="empty after Whisper decoding"):
            dataset.validate_labels()
    training_arguments = training.to_training_arguments
    monkeypatch.setattr(
        training,
        "to_training_arguments",
        lambda config: training_arguments(
            config, use_cpu=True, disable_tqdm=True, dataloader_pin_memory=False
        ),
    )
    completed_runs = []
    train = Seq2SeqTrainer.train

    def record_training(trainer, **kwargs):
        result = train(trainer, **kwargs)
        completed_runs.append((kwargs["resume_from_checkpoint"], trainer.state.global_step))
        return result

    monkeypatch.setattr(Seq2SeqTrainer, "train", record_training)
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for iteration in range(2):
            metrics = run_training(
                settings,
                [manifests[0]],
                [manifests[1]],
                registry=synthetic_registry("synthetic"),
                audio_root=audio_root,
            )
            assert all(math.isfinite(metrics[key]) for key in ("eval_loss", "eval_wer", "eval_cer"))
            state_path = tmp_path / "trained" / "checkpoint-2" / "trainer_state.json"
            assert json.loads(state_path.read_text(encoding="utf-8"))["global_step"] == 2
            if iteration == 0:
                shutil.rmtree(state_path.parent)
                settings = replace(settings, resume=True)
    finally:
        torch.set_num_threads(original_threads)
    assert completed_runs == [(None, 2), (str(tmp_path / "trained" / "checkpoint-1"), 2)]
    restored = WhisperForConditionalGeneration.from_pretrained(settings.output_dir)
    restored_processor = WhisperProcessor.from_pretrained(
        settings.output_dir, language=settings.language, task=settings.task
    )
    assert restored.config.d_model == 16
    assert restored.generation_config.language == settings.language
    assert restored_processor.feature_extractor.chunk_length == 1
    assert restored_processor.tokenizer("beta").input_ids == processor.tokenizer("beta").input_ids
    transcriber = ShortFormWhisperTranscriber(settings.output_dir, device="cpu")
    assert isinstance(transcriber.transcribe(str(audio_root / "dev.wav")), str)
