"""Training plumbing: recipe validation, resume logic, example selection, experiment record.

Everything torch-free is tested here; the one-step CPU smoke runs only where torch
and transformers are installed (the Kaggle image), and skips cleanly otherwise.
"""

from datetime import UTC, datetime

import pytest

from shono.data.manifest import Manifest, Segment
from shono.provenance import GitState, RunContext
from shono.train import (
    ResumeDecision,
    TrainConfig,
    build_examples,
    decide_resume,
    find_latest_checkpoint,
    render_experiment,
    total_hours,
)


def _config(**over) -> TrainConfig:
    base = dict(model_id="base/whisper-medium-bn", output_dir="/tmp/out")
    base.update(over)
    return TrainConfig(**base)


# ---- TrainConfig ---------------------------------------------------------


def test_config_defaults_are_the_recipe():
    c = _config()
    assert c.optim == "adamw_bnb_8bit"  # 8-bit AdamW is what fits a T4
    assert c.freeze_encoder is False  # full fine-tune, not LoRA
    assert c.effective_batch_size == c.per_device_batch_size * c.gradient_accumulation_steps


def test_config_requires_model_id():
    with pytest.raises(ValueError, match="model_id is required"):
        TrainConfig(model_id="", output_dir="/tmp/o")


def test_config_rejects_unknown_optimizer():
    with pytest.raises(ValueError, match="optim"):
        _config(optim="sgd")


def test_config_rejects_bad_learning_rate():
    with pytest.raises(ValueError, match="learning_rate"):
        _config(learning_rate=0)


def test_config_rejects_inverted_chunk_bounds():
    with pytest.raises(ValueError, match="min_chunk_length_s"):
        _config(chunk_length_s=5.0, min_chunk_length_s=10.0)


def test_config_rejects_out_of_range_timestamp_fraction():
    with pytest.raises(ValueError, match="timestamp_sample_fraction"):
        _config(timestamp_sample_fraction=1.5)


def test_config_rejects_chunk_over_30s():
    # Past 30 s, the segment-level timestamp token would collide into normal vocab.
    with pytest.raises(ValueError, match="chunk_length_s"):
        _config(chunk_length_s=45.0)


# ---- checkpoint / resume -------------------------------------------------


def _make_checkpoint(root, step, *, complete=True):
    d = root / f"checkpoint-{step}"
    d.mkdir()
    if complete:
        (d / "trainer_state.json").write_text("{}", encoding="utf-8")
    return d


def test_resume_is_fresh_when_no_checkpoints(tmp_path):
    decision = decide_resume(tmp_path)
    assert decision.resume is False and decision.global_step == 0
    assert "fresh" in decision.summary


def test_resume_picks_latest_complete_checkpoint(tmp_path):
    _make_checkpoint(tmp_path, 500)
    _make_checkpoint(tmp_path, 1000)
    _make_checkpoint(tmp_path, 1500, complete=False)  # half-written — must be skipped
    decision = decide_resume(tmp_path)
    assert decision.resume is True
    assert decision.checkpoint.name == "checkpoint-1000"
    assert decision.global_step == 1000


def test_incomplete_checkpoint_never_selected(tmp_path):
    _make_checkpoint(tmp_path, 700, complete=False)
    assert find_latest_checkpoint(tmp_path) is None


def test_resume_can_be_forced_off(tmp_path):
    _make_checkpoint(tmp_path, 500)
    decision = decide_resume(tmp_path, enabled=False)
    assert decision.resume is False
    assert isinstance(decision, ResumeDecision)


# ---- example selection ---------------------------------------------------


def _manifest(name, split, durations):
    segs = tuple(
        Segment(id=f"{name}-{i}", audio=f"{name}.wav", text=f"বাক্য {i}",
                duration_s=d, recording_id=f"{name}-rec")
        for i, d in enumerate(durations)
    )
    return Manifest(name=name, source="openslr_slr53", split=split, domain="read",
                    version="v", segments=segs)


def test_build_examples_filters_by_duration():
    m = _manifest("m", "train", [0.5, 2.0, 30.0, 40.0])  # keep only 2.0 and 30.0
    examples = build_examples([m], chunk_length_s=30.0, min_chunk_length_s=1.0,
                              timestamp_sample_fraction=0.0, seed=0)
    kept = sorted(e.duration_s for e in examples)
    assert kept == [2.0, 30.0]


def test_build_examples_refuses_test_split():
    m = _manifest("m", "test", [2.0])
    with pytest.raises(ValueError, match="test data must never enter training"):
        build_examples([m], chunk_length_s=30.0, min_chunk_length_s=1.0,
                       timestamp_sample_fraction=0.0, seed=0)


def test_timestamp_fraction_is_deterministic_and_correctly_sized():
    m = _manifest("m", "train", [2.0] * 10)
    kw = dict(chunk_length_s=30.0, min_chunk_length_s=1.0, timestamp_sample_fraction=0.5, seed=7)
    a = build_examples([m], **kw)
    b = build_examples([m], **kw)
    n_ts = sum(e.use_timestamps for e in a)
    assert n_ts == 5  # round(0.5 * 10)
    assert [e.use_timestamps for e in a] == [e.use_timestamps for e in b]  # deterministic


def test_total_hours_sums_across_manifests():
    m1 = _manifest("a", "train", [3600.0])
    m2 = _manifest("b", "train", [1800.0])
    examples = build_examples([m1, m2], chunk_length_s=4000.0, min_chunk_length_s=1.0,
                              timestamp_sample_fraction=0.0, seed=0)
    assert total_hours(examples) == pytest.approx(1.5)


# ---- experiment record ---------------------------------------------------


def _run_context() -> RunContext:
    return RunContext.capture(
        42, {"report": "m3"}, now=lambda: datetime(2026, 7, 23, tzinfo=UTC),
        git_state=GitState(sha="abc", dirty=False),
    )


def test_experiment_record_pre_run_says_not_run():
    text = render_experiment("m3-v1", "medium full-FT beats zero-shot large-v3",
                             _config(), _run_context())
    assert "not run yet" in text
    assert "adamw_bnb_8bit" in text
    assert "medium full-FT beats zero-shot" in text


def test_experiment_record_shows_delta_vs_baseline():
    text = render_experiment(
        "m3-v1", "beats baseline", _config(), _run_context(),
        metrics={"eval_wer": 0.25}, baseline={"eval_wer": 0.34},
    )
    assert "eval_wer 0.2500" in text
    assert "-0.0900" in text  # improvement over baseline


# ---- CPU smoke (torch required) ------------------------------------------


def test_smoke_step_runs_one_step_on_cpu(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from shono.train import smoke_step

    result = smoke_step(tmp_path)
    assert result["loss"] == result["loss"]  # a real float (not NaN)
    assert result["reloaded_params"] > 0
    assert (tmp_path / "checkpoint-1").is_dir()


def test_timestamped_label_omits_notimestamps_token():
    # A timestamped target must NOT contain <|notimestamps|> — otherwise the model
    # is trained on "no timestamps" immediately followed by timestamp tokens.
    transformers = pytest.importorskip("transformers")
    from shono.train.audio import WhisperFineTuneDataset
    from shono.train.data import TrainExample

    processor = transformers.WhisperProcessor.from_pretrained(
        "openai/whisper-tiny", language="bn", task="transcribe"
    )
    example = TrainExample(
        audio="a.wav", text="আমি ভালো", duration_s=3.0, recording_id="r",
        source="s", use_timestamps=True,
    )
    ds = WhisperFineTuneDataset([example], processor, _config())
    labels = ds._labels_for(example)
    no_ts = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    zero_ts = processor.tokenizer.convert_tokens_to_ids("<|0.00|>")
    assert no_ts not in labels  # the bug this pins: notimestamps must be gone
    assert zero_ts in labels  # a real segment-start timestamp token is present
