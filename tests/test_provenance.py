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
