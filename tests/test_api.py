"""Commercial-API benchmark plumbing: interface, $0 budget guard, manifest runner.

No real API is called — a fake transcriber stands in, so these tests are free and
offline. The adapters' SDK imports are lazy and never triggered here.
"""

from pathlib import Path

import pytest

from shono.api import ApiTranscriber, BudgetError, BudgetGuard, run_over_manifest
from shono.data.manifest import Manifest

_TINY = Path(__file__).parent / "fixtures" / "tiny_manifest.jsonl"


class _FakeApi:
    """Records calls; returns a canned transcript per segment."""

    def __init__(self):
        self.calls = []

    def transcribe(self, audio_path, start_s=None, duration_s=None):
        self.calls.append((audio_path, start_s, duration_s))
        return f"অনুমান {len(self.calls)}"


def test_fake_satisfies_the_protocol():
    assert isinstance(_FakeApi(), ApiTranscriber)


def test_run_over_manifest_covers_every_segment():
    manifest = Manifest.from_jsonl(_TINY)
    api = _FakeApi()
    preds = run_over_manifest(api, manifest, "fake-api", audio_root="/audio")
    assert preds.system == "fake-api"
    assert preds.manifest == "tiny-longform-test"
    assert set(preds.hypotheses) == set(manifest.ids())  # a hypothesis for each segment
    assert len(api.calls) == len(manifest.segments)
    assert api.calls[0][0].endswith("rec-loop-01.wav")  # audio_root joined


def test_budget_guard_allows_within_free_allowance():
    manifest = Manifest.from_jsonl(_TINY)  # ~0.0047 h of audio
    api = _FakeApi()
    preds = run_over_manifest(api, manifest, "fake", budget=BudgetGuard(free_hours=1.0))
    assert len(preds.hypotheses) == 5


def test_budget_guard_refuses_paid_spend_before_any_call():
    manifest = Manifest.from_jsonl(_TINY)
    api = _FakeApi()
    with pytest.raises(BudgetError, match="paid spend requires an explicit decision"):
        run_over_manifest(api, manifest, "fake", budget=BudgetGuard(free_hours=0.0))
    assert api.calls == []  # refused before a single (paid) call was made


def test_budget_guard_allows_paid_when_opted_in():
    manifest = Manifest.from_jsonl(_TINY)
    api = _FakeApi()
    preds = run_over_manifest(
        api, manifest, "fake", budget=BudgetGuard(free_hours=0.0, allow_paid=True)
    )
    assert len(preds.hypotheses) == 5  # explicit opt-in proceeds
