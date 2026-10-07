import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script", ["check_evidence.py", "check_notebooks.py"])
@pytest.mark.parametrize("argument,status", [("--help", 0), ("--invalid", 2)])
def test_helper_cli_handles_options_without_work(script, argument, status):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), argument], capture_output=True, text=True
    )
    assert result.returncode == status
    assert "Traceback" not in result.stderr


def test_hand_computed_artifact_is_current():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_evidence.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "model accuracy remains unmeasured" in result.stdout
