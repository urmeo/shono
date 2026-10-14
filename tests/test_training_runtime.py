"""Offline fakes test the actual training boundary without downloading weights."""

import hashlib
import json
import os
import shutil
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
from data_fixtures import synthetic_registry

from shono.data import Manifest, Segment
from shono.train import TrainConfig, TrainExample, build_examples, run_training
from shono.train.audio import WhisperFineTuneDataset
from shono.train.trainer import DataCollatorSpeechSeq2Seq

MODEL_REVISION = "a" * 40


class Scalar:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def item(self):
        return self.value


class Tensor:
    def __init__(self, rows):
        self.rows = rows

    def ne(self, value):
        return Tensor([[x != value for x in row] for row in self.rows])

    def masked_fill(self, mask, value):
        return Tensor(
            [
                [value if flag else x for x, flag in zip(row, flags, strict=True)]
                for row, flags in zip(self.rows, mask.rows, strict=True)
            ]
        )

    def __getitem__(self, index):
        _, column = index
        return Tensor([row[column] for row in self.rows])

    def __eq__(self, value):
        return Tensor([x == value for x in self.rows])

    def all(self):
        return Scalar(all(self.rows))

    def any(self):
        return Scalar(any(self.rows))


class Tokenizer:
    prefix_tokens = [1, 2, 3, 9]
    eos_token_id = 4
    pad_token_id = 0
    text_length = 2

    def convert_tokens_to_ids(self, text):
        return {"<|notimestamps|>": 9, "<|0.00|>": 100}[text]

    def __call__(self, text, add_special_tokens=True):
        prefix = self.prefix_tokens if add_special_tokens else []
        suffix = [self.eos_token_id] if add_special_tokens else []
        return SimpleNamespace(input_ids=[*prefix, *([5] * self.text_length), *suffix])

    def decode(self, ids, skip_special_tokens=True):
        return "target" if 5 in ids else ""

    def pad(self, rows, return_tensors):
        count = max(len(row["input_ids"]) for row in rows)
        ids = [row["input_ids"] + [0] * (count - len(row["input_ids"])) for row in rows]
        masks = [
            [1] * len(row["input_ids"]) + [0] * (count - len(row["input_ids"])) for row in rows
        ]
        result = {"input_ids": Tensor(ids), "attention_mask": Tensor(masks)}
        return AttrDict(result)


class AttrDict(dict):
    __getattr__ = dict.__getitem__


class Extractor:
    def __call__(self, audio, **kwargs):
        assert kwargs["return_attention_mask"] is True
        return SimpleNamespace(input_features=[[1, 2, 3]], attention_mask=[[1, 1, 0]])

    def pad(self, rows, return_tensors):
        return {key: [row[key] for row in rows] for key in rows[0]}


def processor():
    return SimpleNamespace(tokenizer=Tokenizer(), feature_extractor=Extractor())


def source_manifest(name, split, *, sid=None, text=None, audio=None, duration=3):
    sid = sid or name
    return Manifest(
        name=name,
        source="synthetic",
        split=split,
        domain="read",
        version="fixture",
        segments=(
            Segment(
                id=sid,
                audio=audio or f"{sid}.wav",
                text=text or name,
                duration_s=duration,
                recording_id=sid,
            ),
        ),
    )


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    audio = tmp_path / "audio"
    audio.mkdir()
    train = source_manifest("train", "train")
    dev = source_manifest("dev", "dev")
    for manifest in (train, dev):
        (audio / manifest.segments[0].audio).write_bytes(manifest.name.encode())
    cfg = TrainConfig(
        model_id="synthetic/base",
        model_revision=MODEL_REVISION,
        output_dir=str(tmp_path / "outputs"),
        optim="adamw_torch",
        fp16=False,
        freeze_encoder=True,
    )
    events = []
    state = {}

    class Processor:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            events.append(("processor", model_id, kwargs))
            state["processor"] = processor()
            return state["processor"]

    class Config:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            events.append(("config", model_id, kwargs))
            return SimpleNamespace(max_target_positions=32, decoder_start_token_id=1)

    class Model:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            events.append(("model", model_id, kwargs))

            class Loaded:
                config = SimpleNamespace(decoder_start_token_id=1, use_cache=True)
                generation_config = SimpleNamespace()
                encoder = SimpleNamespace(requires_grad=True)
                decoder = SimpleNamespace(requires_grad=True)

                def freeze_encoder(self):
                    self.encoder.requires_grad = False
                    events.append(("freeze",))

            state["model"] = Loaded()
            return state["model"]

    class Trainer:
        def __init__(self, **kwargs):
            state["trainer"] = kwargs
            events.append(("trainer",))

        def train(self, **kwargs):
            events.append(("train", kwargs))
            return SimpleNamespace(metrics={"train_loss": 0.25})

        def evaluate(self):
            events.append(("evaluate",))
            return {"eval_wer": 0.2, "eval_cer": 0.1}

        def save_model(self, path):
            events.append(("save", path))

    def arguments(**kwargs):
        state["args"] = kwargs
        return SimpleNamespace(**kwargs)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            WhisperProcessor=Processor,
            WhisperConfig=Config,
            WhisperForConditionalGeneration=Model,
            Seq2SeqTrainer=Trainer,
            Seq2SeqTrainingArguments=arguments,
        ),
    )
    return SimpleNamespace(
        audio=audio,
        train=train,
        dev=dev,
        cfg=cfg,
        events=events,
        state=state,
        registry=synthetic_registry("synthetic"),
    )


def run(runtime, **kwargs):
    return run_training(
        runtime.cfg,
        [runtime.train],
        [runtime.dev],
        registry=runtime.registry,
        audio_root=runtime.audio,
        duration_of=lambda p: 3,
        **kwargs,
    )


def test_trainer_evaluates_freezes_saves_and_returns_final_metrics(runtime):
    metrics = run(runtime)
    assert metrics == {"train_loss": 0.25, "eval_wer": 0.2, "eval_cer": 0.1}
    assert runtime.state["args"]["eval_strategy"] == "steps"
    assert runtime.state["args"]["predict_with_generate"] is True
    assert runtime.state["model"].encoder.requires_grad is False
    assert runtime.state["model"].decoder.requires_grad is True
    assert runtime.state["model"].config.use_cache is False
    assert [event[0] for event in runtime.events][-3:] == ["train", "evaluate", "save"]
    assert runtime.state["trainer"]["processing_class"] is runtime.state["processor"]
    identity = json.loads((runtime.audio.parent / "outputs/training-inputs.json").read_text())
    assert identity["manifests"][0]["name"] == "train"
    assert identity["leakage"][0]["eval_total"] == 1


def test_all_remote_loaders_use_same_persisted_immutable_revision(runtime):
    run(runtime)
    loaders = [event for event in runtime.events if event[0] in {"processor", "config", "model"}]
    assert len(loaders) == 3
    assert all(event[1] == runtime.cfg.model_id for event in loaders)
    assert all(event[2]["revision"] == MODEL_REVISION for event in loaders)
    identity = json.loads((runtime.audio.parent / "outputs/training-inputs.json").read_text())
    assert identity["config"]["model_revision"] == MODEL_REVISION
    assert identity["base_checkpoint"] == {
        "source": "hub",
        "model_id": runtime.cfg.model_id,
        "revision": MODEL_REVISION,
    }


def test_changed_hub_revision_blocks_resume_before_loaders(runtime):
    run(runtime)
    runtime.events.clear()
    runtime.cfg = replace(runtime.cfg, model_revision="b" * 40)
    with pytest.raises(ValueError, match="different training identity"):
        run(runtime)
    assert runtime.events == []


@pytest.mark.parametrize("role", ["train", "dev"])
def test_changed_audio_bytes_block_resume_before_loaders(runtime, role):
    run(runtime)
    manifest = getattr(runtime, role)
    path = runtime.audio / manifest.segments[0].audio
    path.write_bytes(b"changed audio with the same declared duration")
    runtime.events.clear()

    with pytest.raises(ValueError, match="different training identity"):
        run(runtime)

    assert runtime.events == []


def test_training_identity_records_undeclared_audio_hashes(runtime):
    run(runtime)
    identity = json.loads((runtime.audio.parent / "outputs/training-inputs.json").read_text())

    assert identity["audio_files_sha256"] == {
        f"{name}.wav": hashlib.sha256(name.encode()).hexdigest() for name in ("train", "dev")
    }
    assert runtime.train.segments[0].audio_sha256 is None


@pytest.mark.parametrize("resume", [False, True])
def test_training_identity_without_audio_hashes_cannot_verify_restart(runtime, resume):
    run(runtime)
    identity_path = runtime.audio.parent / "outputs/training-inputs.json"
    identity = json.loads(identity_path.read_text())
    del identity["audio_files_sha256"]
    identity_path.write_text(json.dumps(identity))
    runtime.cfg = replace(runtime.cfg, resume=resume)
    runtime.events.clear()

    with pytest.raises(ValueError, match="no verifiable audio hashes"):
        run(runtime)

    assert runtime.events == []


def test_training_identity_with_complete_declared_hashes_can_be_migrated(runtime):
    for role in ("train", "dev"):
        manifest = getattr(runtime, role)
        segment = manifest.segments[0]
        digest = hashlib.sha256((runtime.audio / segment.audio).read_bytes()).hexdigest()
        setattr(runtime, role, replace(manifest, segments=(replace(segment, audio_sha256=digest),)))
    run(runtime)
    identity_path = runtime.audio.parent / "outputs/training-inputs.json"
    identity = json.loads(identity_path.read_text())
    del identity["audio_files_sha256"]
    identity_path.write_text(json.dumps(identity))
    runtime.events.clear()

    run(runtime)

    assert any(event[0] == "model" for event in runtime.events)
    assert "audio_files_sha256" in json.loads(identity_path.read_text())


@pytest.mark.parametrize("role", ["train", "dev"])
@pytest.mark.parametrize("alias", ["symlink", "hardlink", "copy"])
def test_duplicate_audio_windows_under_other_names_fail_preflight(runtime, role, alias):
    manifest = getattr(runtime, role)
    segment = manifest.segments[0]
    original = runtime.audio / segment.audio
    path = runtime.audio / f"{role}-alias.wav"
    if alias == "symlink":
        path.symlink_to(original.name)
    elif alias == "hardlink":
        os.link(original, path)
    else:
        shutil.copyfile(original, path)
    duplicate = replace(segment, id=f"{role}-alias", audio=path.name)
    setattr(runtime, role, replace(manifest, segments=(segment, duplicate)))

    with pytest.raises(ValueError, match="duplicate.*audio windows"):
        run(runtime)

    assert runtime.events == []


def test_distinct_crops_of_one_recording_remain_supported(runtime):
    segment = runtime.train.segments[0]
    crop = replace(segment, id="crop", start_s=0.0, duration_s=2.0)
    runtime.train = replace(runtime.train, segments=(segment, crop))

    run(runtime)

    assert len(runtime.state["trainer"]["train_dataset"]) == 2


@pytest.mark.parametrize("role", ["train", "dev"])
@pytest.mark.parametrize("alias", [False, True])
def test_eof_tolerance_cannot_hide_duplicate_decoded_windows(runtime, role, alias):
    manifest = getattr(runtime, role)
    segment = manifest.segments[0]
    audio = segment.audio
    if alias:
        audio = f"{role}-copy.wav"
        shutil.copyfile(runtime.audio / segment.audio, runtime.audio / audio)
    duplicate = replace(
        segment, id=f"{role}-duplicate", audio=audio, start_s=0.0, duration_s=3.0005
    )
    setattr(runtime, role, replace(manifest, segments=(segment, duplicate)))

    with pytest.raises(ValueError, match="duplicate.*audio windows"):
        run(runtime)

    assert runtime.events == []


def use_local_checkpoint(runtime):
    checkpoint = runtime.audio.parent / "base"
    checkpoint.mkdir()
    for name in ("config.json", "model.safetensors", "tokenizer.json", "preprocessor_config.json"):
        (checkpoint / name).write_bytes(b"original")
    runtime.cfg = replace(runtime.cfg, model_id=str(checkpoint), model_revision=None)
    return checkpoint


def test_local_checkpoint_loads_offline_and_unchanged_resume_works(runtime):
    checkpoint = use_local_checkpoint(runtime)
    run(runtime)
    loaders = [event for event in runtime.events if event[0] in {"processor", "config", "model"}]
    assert all(event[1] == str(checkpoint.resolve()) for event in loaders)
    assert all(event[2].get("local_files_only") is True for event in loaders)
    assert all("revision" not in event[2] for event in loaders)
    identity = json.loads((runtime.audio.parent / "outputs/training-inputs.json").read_text())
    assert set(identity["base_checkpoint"]["files_sha256"]) == {
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "preprocessor_config.json",
    }
    run(runtime)


@pytest.mark.parametrize("file", ["config.json", "model.safetensors", "tokenizer.json"])
def test_changed_local_checkpoint_blocks_resume_before_loaders(runtime, file):
    checkpoint = use_local_checkpoint(runtime)
    run(runtime)
    runtime.events.clear()
    (checkpoint / file).write_bytes(b"modified")
    with pytest.raises(ValueError, match="different training identity"):
        run(runtime)
    assert runtime.events == []


def test_local_checkpoint_change_between_loaders_blocks_training(runtime, monkeypatch):
    checkpoint = use_local_checkpoint(runtime)
    loader = sys.modules["transformers"].WhisperConfig
    original = loader.from_pretrained

    def load_and_change(*args, **kwargs):
        result = original(*args, **kwargs)
        (checkpoint / "tokenizer.json").write_bytes(b"modified")
        return result

    monkeypatch.setattr(loader, "from_pretrained", load_and_change)
    with pytest.raises(ValueError, match="changed while loading"):
        run(runtime)
    assert [event[0] for event in runtime.events] == ["processor", "config", "model"]
    assert not (runtime.audio.parent / "outputs").exists()


@pytest.mark.parametrize("kind", ["child", "equal", "case-alias"])
def test_training_output_cannot_overwrite_local_checkpoint(runtime, kind):
    checkpoint = use_local_checkpoint(runtime)
    output = checkpoint / "trained"
    if kind == "equal":
        output = checkpoint
    elif kind == "case-alias":
        output = checkpoint.with_name(checkpoint.name.upper()) / "trained"
    runtime.cfg = replace(runtime.cfg, output_dir=str(output))
    with pytest.raises(ValueError, match="base checkpoint must be disjoint"):
        run(runtime)
    assert runtime.events == []


@pytest.mark.parametrize(
    "kind", ["dev-train", "eval-only", "unknown", "missing", "overlap", "duration", "empty"]
)
def test_invalid_inputs_fail_before_pretrained(runtime, kind):
    if kind == "dev-train":
        runtime.train = replace(runtime.train, split="dev")
    elif kind == "eval-only":
        runtime.registry = synthetic_registry("synthetic", status="eval-only")
    elif kind == "unknown":
        runtime.registry = synthetic_registry("other")
    elif kind == "missing":
        (runtime.audio / "train.wav").unlink()
    elif kind == "overlap":
        runtime.dev = replace(runtime.dev, segments=(replace(runtime.dev.segments[0], id="train"),))
    elif kind == "duration":
        runtime.train = replace(
            runtime.train, segments=(replace(runtime.train.segments[0], duration_s=2),)
        )
    else:
        runtime.cfg = replace(runtime.cfg, min_chunk_length_s=4)
    with pytest.raises((ValueError, KeyError)):
        run(runtime)
    assert runtime.events == []


def test_duration_filtered_overlap_still_blocks(runtime):
    extra = replace(runtime.train.segments[0], id="shared", audio="long.wav", duration_s=40)
    runtime.train = replace(runtime.train, segments=(*runtime.train.segments, extra))
    runtime.dev = replace(runtime.dev, segments=(replace(runtime.dev.segments[0], id="shared"),))
    (runtime.audio / "long.wav").write_bytes(b"fixture")
    with pytest.raises(ValueError, match="hard train/evaluation"):
        run_training(
            runtime.cfg,
            [runtime.train],
            [runtime.dev],
            registry=runtime.registry,
            audio_root=runtime.audio,
            duration_of=lambda p: 40 if p.name == "long.wav" else 3,
        )
    assert runtime.events == []


@pytest.mark.parametrize("kind", ["id", "same-file", "copy"])
def test_development_and_held_out_data_cannot_share_ids_or_audio(runtime, kind):
    original = runtime.dev.segments[0]
    held_out = source_manifest("held-out", "test")
    path = runtime.audio / held_out.segments[0].audio
    path.write_bytes(b"held-out recording")
    if kind == "id":
        held_out = replace(held_out, segments=(replace(held_out.segments[0], id=original.id),))
    elif kind == "same-file":
        held_out = replace(
            held_out, segments=(replace(held_out.segments[0], audio=original.audio),)
        )
    else:
        shutil.copyfile(runtime.audio / original.audio, path)

    with pytest.raises(ValueError, match="hard development/held-out overlap"):
        run(runtime, held_out_manifests=[held_out])

    assert runtime.events == []


def test_distinct_held_out_audio_can_share_development_text(runtime):
    held_out = source_manifest("held-out", "test", text=runtime.dev.segments[0].text)
    (runtime.audio / held_out.segments[0].audio).write_bytes(b"held-out recording")

    run(runtime, held_out_manifests=[held_out])

    assert any(event[0] == "model" for event in runtime.events)


def test_exact_text_decision_is_saved(runtime):
    runtime.dev = replace(runtime.dev, segments=(replace(runtime.dev.segments[0], text="train"),))
    run(
        runtime,
        text_decisions={"dev": {"train": "synthetic common reference, distinct recordings"}},
    )
    identity = json.loads((runtime.audio.parent / "outputs/training-inputs.json").read_text())
    assert identity["leakage"][0]["reviewed_text"][0]["key"] == "train"


def test_incompatible_resume_identity_and_missing_identity_fail_before_models(runtime):
    run(runtime)
    runtime.events.clear()
    out = runtime.audio.parent / "outputs"
    checkpoint = out / "checkpoint-5"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text('{"global_step":5}')
    runtime.cfg = replace(runtime.cfg, learning_rate=2e-5)
    with pytest.raises(ValueError, match="different training identity"):
        run(runtime)
    assert runtime.events == []
    (out / "training-inputs.json").unlink()
    with pytest.raises(ValueError, match="no declared training identity"):
        run(runtime)
    assert runtime.events == []


def test_compatible_resume_passes_saved_step(runtime):
    run(runtime)
    runtime.events.clear()
    checkpoint = runtime.audio.parent / "outputs/checkpoint-5"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text('{"global_step":5}')
    run(runtime)
    assert ("train", {"resume_from_checkpoint": str(checkpoint)}) in runtime.events


@pytest.mark.parametrize("legacy_identity", [False, True])
def test_enabling_resume_preserves_training_identity(runtime, legacy_identity):
    runtime.cfg = replace(runtime.cfg, resume=False)
    run(runtime)
    out = runtime.audio.parent / "outputs"
    identity_path = out / "training-inputs.json"
    if legacy_identity:
        identity = json.loads(identity_path.read_text())
        identity["config"]["resume"] = False
        identity_path.write_text(json.dumps(identity))
    checkpoint = out / "checkpoint-5"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text('{"global_step":5}')
    runtime.cfg = replace(runtime.cfg, resume=True)
    runtime.events.clear()

    run(runtime)

    assert ("train", {"resume_from_checkpoint": str(checkpoint)}) in runtime.events
    assert "resume" not in json.loads(identity_path.read_text())["config"]


def test_prior_training_identity_rejects_duplicate_json_keys(runtime):
    run(runtime)
    identity_path = runtime.audio.parent / "outputs/training-inputs.json"
    original = identity_path.read_text()
    identity_path.write_text('{"config": {}, ' + original.lstrip()[1:])
    runtime.events.clear()

    with pytest.raises(ValueError, match="cannot read prior training identity"):
        run(runtime)

    assert runtime.events == []


@pytest.mark.parametrize("change", ["config", "manifest", "checkpoint"])
@pytest.mark.parametrize("resume", [False, True])
def test_legacy_resume_record_still_requires_matching_training_inputs(runtime, change, resume):
    runtime.cfg = replace(runtime.cfg, resume=False)
    run(runtime)
    identity_path = runtime.audio.parent / "outputs/training-inputs.json"
    identity = json.loads(identity_path.read_text())
    identity["config"]["resume"] = False
    identity_path.write_text(json.dumps(identity))
    runtime.cfg = replace(runtime.cfg, resume=resume)
    if change == "config":
        runtime.cfg = replace(runtime.cfg, learning_rate=2e-5)
    elif change == "manifest":
        runtime.dev = replace(
            runtime.dev, segments=(replace(runtime.dev.segments[0], text="changed reference"),)
        )
    else:
        runtime.cfg = replace(runtime.cfg, model_revision="b" * 40)
    runtime.events.clear()

    with pytest.raises(ValueError, match="different training identity"):
        run(runtime)

    assert runtime.events == []


def test_disabling_resume_starts_fresh_with_matching_training_inputs(runtime):
    run(runtime)
    checkpoint = runtime.audio.parent / "outputs/checkpoint-5"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text('{"global_step":5}')
    runtime.cfg = replace(runtime.cfg, resume=False)
    runtime.events.clear()

    run(runtime)

    assert ("train", {"resume_from_checkpoint": None}) in runtime.events


@pytest.mark.parametrize("role", ["train", "dev"])
@pytest.mark.parametrize("target", ["...", "।!?+", "\u200b\u200c\u200d\ufeff"])
def test_unscorable_targets_fail_before_model_loading(runtime, role, target):
    manifest = getattr(runtime, role)
    setattr(
        runtime, role, replace(manifest, segments=(replace(manifest.segments[0], text=target),))
    )

    with pytest.raises(ValueError, match="empty after evaluation normalization"):
        run(runtime)

    assert runtime.events == []


@pytest.mark.parametrize("kind", ["inside", "ancestor", "hardlink", "symlink", "cycle"])
def test_outputs_cannot_alias_inputs(runtime, kind):
    out = runtime.audio.parent / "outputs"
    if kind == "inside":
        runtime.cfg = replace(runtime.cfg, output_dir=str(runtime.audio / "outputs"))
    elif kind == "ancestor":
        runtime.cfg = replace(runtime.cfg, output_dir=str(runtime.audio.parent))
    elif kind == "cycle":
        out.symlink_to(out.name)
        runtime.cfg = replace(runtime.cfg, output_dir=str(out))
    else:
        out.mkdir()
        target = out / "training-inputs.json"
        if kind == "hardlink":
            os.link(runtime.audio / "train.wav", target)
        else:
            target.symlink_to(runtime.audio / "train.wav")
    with pytest.raises(ValueError):
        run(runtime)
    assert runtime.events == [] and (runtime.audio / "train.wav").read_bytes() == b"train"


def test_label_capacity_rejects_before_model_weights(runtime, monkeypatch):
    monkeypatch.setattr(Tokenizer, "text_length", 40)
    with pytest.raises(ValueError, match="model capacity"):
        run(runtime)
    assert [event[0] for event in runtime.events] == ["processor", "config"]


@pytest.mark.parametrize("role", ["train", "dev"])
@pytest.mark.parametrize("target", ["<|endoftext|>", "<|notimestamps|>", "<|0.00|>"])
def test_targets_removed_by_whisper_decoding_fail_before_model_weights(
    runtime, monkeypatch, role, target
):
    original = Tokenizer.__call__

    def tokenize(tokenizer, text, add_special_tokens=True):
        if text == target:
            prefix = tokenizer.prefix_tokens if add_special_tokens else []
            return SimpleNamespace(input_ids=[*prefix, tokenizer.eos_token_id])
        return original(tokenizer, text, add_special_tokens)

    monkeypatch.setattr(Tokenizer, "__call__", tokenize)
    manifest = getattr(runtime, role)
    setattr(
        runtime, role, replace(manifest, segments=(replace(manifest.segments[0], text=target),))
    )

    with pytest.raises(ValueError, match="empty after Whisper decoding"):
        run(runtime)

    assert [event[0] for event in runtime.events] == ["processor", "config"]


def test_masks_padding_bos_and_capacity(tmp_path, monkeypatch):
    from shono.train import audio as module

    (tmp_path / "a.wav").write_bytes(b"fixture")
    example = TrainExample("a.wav", "বাক্য", 3, "r", "synthetic", use_timestamps=True)
    cfg = TrainConfig(
        model_id="synthetic", model_revision=MODEL_REVISION, output_dir=str(tmp_path / "out")
    )
    proc = processor()
    ds = WhisperFineTuneDataset(
        [example], proc, cfg, audio_root=tmp_path, max_target_positions=7, decoder_start_token_id=1
    )
    monkeypatch.setattr(module, "_load_audio", lambda *args: [0.1, 0.2])
    item = ds[0]
    assert item["attention_mask"] == [1, 1, 0] and 9 not in item["labels"]
    batch = DataCollatorSpeechSeq2Seq(proc, 1)(
        [item, dict(item, labels=[1, 5, 4], attention_mask=[1, 0, 0])]
    )
    assert batch["attention_mask"] == [[1, 1, 0], [1, 0, 0]]
    assert batch["labels"].rows[1] == [5, 4, -100, -100, -100, -100, -100]
    ds.max_target_positions = 6
    with pytest.raises(ValueError, match="model capacity"):
        ds.validate_labels()


@pytest.mark.parametrize(
    "field,value",
    [
        ("learning_rate", float("nan")),
        ("learning_rate", True),
        ("weight_decay", float("inf")),
        ("warmup_steps", -1),
        ("save_steps", False),
        ("eval_steps", 0),
        ("seed", -1),
        ("freeze_encoder", 1),
        ("per_device_batch_size", 10**1000),
    ],
)
def test_training_config_rejects_invalid_numbers(field, value):
    with pytest.raises(ValueError):
        TrainConfig(
            model_id="synthetic", model_revision=MODEL_REVISION, output_dir="out", **{field: value}
        )


def test_selection_requires_role_license_unique_inputs(runtime):
    kwargs = dict(
        chunk_length_s=28,
        min_chunk_length_s=1,
        timestamp_sample_fraction=0,
        seed=42,
        registry=runtime.registry,
    )
    with pytest.raises(ValueError):
        build_examples([runtime.dev], **kwargs)
    assert build_examples([runtime.dev], role="eval", **kwargs)[0].split == "dev"
    with pytest.raises(ValueError, match="distinct"):
        build_examples([runtime.train, runtime.train], **kwargs)


@pytest.mark.parametrize(
    "directory", ["src", "docs", "data", "tests", "notebooks", "scripts", "reports"]
)
def test_repository_source_outputs_fail_before_model_calls(runtime, directory):
    from pathlib import Path

    from shono.train import output

    repo = Path(output.__file__).resolve().parents[3]
    runtime.cfg = replace(runtime.cfg, output_dir=str(repo / directory / "synthetic-output"))
    with pytest.raises(ValueError, match="source subtrees"):
        run(runtime)
    assert runtime.events == []
