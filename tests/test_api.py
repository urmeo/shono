"""Manifest and allowance checks use synthetic local audio and fake providers."""

import wave
from pathlib import Path

import pytest

from shono.api import ApiTranscriber, BudgetError, BudgetGuard, run_over_manifest
from shono.data.manifest import Manifest

_TINY = Path(__file__).parent / "fixtures" / "tiny_manifest.jsonl"


class _FakeApi:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio_path, start_s=None, duration_s=None):
        self.calls.append((audio_path, start_s, duration_s))
        return f"অনুমান {len(self.calls)}"


@pytest.fixture
def audio_root(tmp_path):
    for name in ("rec-loop-01.wav", "rec-loop-02.wav"):
        with wave.open(str(tmp_path / name), "wb") as source:
            source.setparams((1, 2, 10, 0, "NONE", "not compressed"))
            source.writeframes(b"\0\0" * 200)
    return tmp_path


def test_fake_satisfies_the_protocol():
    assert isinstance(_FakeApi(), ApiTranscriber)


def test_run_over_manifest_covers_every_segment(audio_root):
    manifest = Manifest.from_jsonl(_TINY)
    api = _FakeApi()
    preds = run_over_manifest(api, manifest, "fake-api", audio_root=audio_root)
    assert preds.system == "fake-api"
    assert preds.manifest == "tiny-longform-test"
    assert set(preds.hypotheses) == set(manifest.ids())
    assert len(api.calls) == len(manifest.segments)
    assert Path(api.calls[0][0]) == audio_root / "rec-loop-01.wav"
    assert preds.input_paths == tuple(call[0] for call in api.calls)


def test_budget_guard_allows_within_free_allowance(audio_root):
    manifest = Manifest.from_jsonl(_TINY)
    preds = run_over_manifest(
        _FakeApi(), manifest, "fake", audio_root=audio_root, budget=BudgetGuard(free_hours=1.0)
    )
    assert len(preds.hypotheses) == 5


def test_budget_guard_refuses_paid_spend_before_any_call(audio_root):
    api = _FakeApi()
    with pytest.raises(BudgetError, match="paid spend requires an explicit decision"):
        run_over_manifest(
            api, Manifest.from_jsonl(_TINY), "fake", audio_root=audio_root, budget=BudgetGuard()
        )
    assert api.calls == []


def test_budget_guard_allows_paid_when_opted_in(audio_root):
    preds = run_over_manifest(
        _FakeApi(),
        Manifest.from_jsonl(_TINY),
        "fake",
        audio_root=audio_root,
        budget=BudgetGuard(allow_paid=True),
    )
    assert len(preds.hypotheses) == 5


def test_allowance_is_cumulative_and_check_does_not_reserve():
    guard = BudgetGuard(free_hours=1)
    guard.check(0.75)
    assert guard.used_hours == 0
    guard.reserve(0.75)
    with pytest.raises(BudgetError):
        guard.reserve(0.5)
    assert guard.used_hours == 0.75


def test_failed_provider_consumes_entire_manifest_allowance(audio_root):
    class Failed(_FakeApi):
        def transcribe(self, *args):
            raise RuntimeError("fake failed request")

    manifest = Manifest.from_jsonl(_TINY)
    guard = BudgetGuard(free_hours=1)
    with pytest.raises(RuntimeError, match="fake failed"):
        run_over_manifest(Failed(), manifest, "fake", audio_root=audio_root, budget=guard)
    assert guard.used_hours == pytest.approx(16.8 / 3600)


def test_commercial_adapter_defaults_to_zero_allowance(audio_root):
    api = _FakeApi()
    api.requires_budget = True
    with pytest.raises(BudgetError):
        run_over_manifest(api, Manifest.from_jsonl(_TINY), "fake", audio_root=audio_root)
    assert api.calls == []


def test_late_missing_or_invalid_audio_rejects_before_calls(audio_root):
    manifest = Manifest.from_jsonl(_TINY)
    api = _FakeApi()
    (audio_root / "rec-loop-02.wav").unlink()
    with pytest.raises(ValueError, match="missing audio"):
        run_over_manifest(api, manifest, "fake", audio_root=audio_root)
    assert api.calls == []


def test_late_eof_rejects_before_calls(audio_root):
    manifest = Manifest.from_jsonl(_TINY)
    api = _FakeApi()
    with pytest.raises(ValueError, match="EOF"):
        run_over_manifest(
            api,
            manifest,
            "fake",
            audio_root=audio_root,
            duration_of=lambda p: 20 if p.name.endswith("01.wav") else 1,
        )
    assert api.calls == []


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, True, "1"])
def test_budget_invalid_inputs(bad):
    with pytest.raises(ValueError):
        BudgetGuard(free_hours=bad)
    with pytest.raises(ValueError):
        BudgetGuard(allow_paid=True).reserve(bad)


def test_overflowing_cumulative_allowance_is_clear():
    guard = BudgetGuard(allow_paid=True)
    guard.reserve(1e308)
    with pytest.raises(ValueError, match="finite"):
        guard.reserve(1e308)
    assert guard.used_hours == 1e308


def test_observed_runtime_metadata_is_recorded_and_redacted(audio_root):
    api = _FakeApi()
    api.observed_runtime_context = lambda: {"device": "cpu", "api_key": "private"}
    predictions = run_over_manifest(api, Manifest.from_jsonl(_TINY), "fake", audio_root=audio_root)
    assert predictions.run_context == {
        "observed_runtime": {"device": "cpu", "api_key": "[redacted]"}
    }
