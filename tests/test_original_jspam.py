"""Checks that the original JSPAM Fortran still compiles and runs the same on this machine."""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
JSPAM_CLONE = REPO_ROOT.parent / "JSPAM"
SCRIPT = "examples/spam/check_original_jspam.sh"


def docker_is_running() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


@pytest.mark.slow
def test_original_jspam():
    for tool in ["bash", "git"]:
        if shutil.which(tool) is None:
            pytest.skip(f"{tool} not found")
    if not docker_is_running():
        pytest.skip("Docker is not running")
    if not JSPAM_CLONE.is_dir():
        pytest.skip(f"no JSPAM clone at {JSPAM_CLONE}")

    # A path relative to cwd, since Windows may find a bash that doesn't read Windows paths.
    result = subprocess.run(["bash", SCRIPT], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
