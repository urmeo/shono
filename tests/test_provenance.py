"""RunContext captures a reproducible, JSON-safe snapshot; seeding is real."""

import json
import random
from datetime import UTC, datetime

from shono.eval import NORMALIZER_VERSION
from shono.provenance import GitState, RunContext, seed_everything

_FIXED = datetime(2026, 7, 23, 9, 30, tzinfo=UTC)


def _capture(**over):
    base = dict(
        seed=13,
        config={"model": "whisper-medium", "lr": 5e-6},
        command="python -m shono.eval --report baselines",
        now=lambda: _FIXED,
        git_state=GitState(sha="abc123", dirty=False),
    )
    base.update(over)
    return RunContext.capture(**base)


def test_capture_records_seed_config_command():
    ctx = _capture()
    assert ctx.seed == 13
    assert ctx.config["model"] == "whisper-medium"
    assert ctx.command.startswith("python -m shono.eval")


def test_capture_stamps_normalizer_version():
    # A report's numbers are only meaningful against the normalizer that made them.
    assert _capture().normalizer_version == NORMALIZER_VERSION


def test_injected_clock_and_git_make_capture_deterministic():
    a = _capture()
    b = _capture()
    assert a.timestamp == b.timestamp == "2026-07-23T09:30:00+00:00"
    assert a.git.sha == "abc123" and a.git.dirty is False


def test_to_dict_is_json_serializable():
    payload = json.dumps(_capture().to_dict())
    round_trip = json.loads(payload)
    assert round_trip["seed"] == 13
    assert round_trip["git"] == {"sha": "abc123", "dirty": False}
    assert round_trip["normalizer_version"] == NORMALIZER_VERSION


def test_packages_include_shono_and_scoring_deps():
    pkgs = _capture().packages
    # These are always installed in this repo; torch/cuda deps are legitimately absent locally.
    assert "jiwer" in pkgs
    assert "bnunicodenormalizer" in pkgs


def test_dirty_tree_is_recorded():
    ctx = _capture(git_state=GitState(sha="deadbeef", dirty=True))
    assert ctx.to_dict()["git"]["dirty"] is True


def test_seed_everything_makes_python_random_reproducible():
    seed_everything(7)
    first = [random.random() for _ in range(5)]
    seed_everything(7)
    second = [random.random() for _ in range(5)]
    assert first == second


def test_unknown_packages_are_explicit():
    assert _capture(packages=("shono-nonexistent-test-package",)).packages == {
        "shono-nonexistent-test-package": "unknown"
    }


def test_named_credentials_are_redacted_without_changing_ordinary_arguments():
    context = _capture(
        command="HF_TOKEN=abc python run.py --api-key secret --max-new-tokens 20 "
        '--text "ordinary answer"',
        config={"api_key": "secret", "max_new_tokens": 20, "nested": {"hf_token": "abc"}},
        extra={"password": "p"},
    )
    text = json.dumps(context.to_dict())
    assert "secret" not in text and "abc" not in context.command
    assert "ordinary answer" in context.command and "--max-new-tokens 20" in context.command
    assert context.config["max_new_tokens"] == 20
    assert context.extra["password"] == "[redacted]"


def test_metadata_nonfinite_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="finite"):
        _capture(config={"lr": float("nan")})


def test_naive_capture_clock_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="aware"):
        _capture(now=lambda: datetime(2026, 1, 1))


def test_invalid_seed_never_seeds_python():
    import pytest

    with pytest.raises(ValueError, match="seed"):
        seed_everything(True)


def test_extra_named_command_and_api_aliases_do_not_retain_credentials():
    context = _capture(
        extra={"command": "run --hf-token abc", "apiKey": "private", "DEEPGRAM_KEY": "private"}
    )
    text = json.dumps(context.extra)
    assert "abc" not in text and "private" not in text
