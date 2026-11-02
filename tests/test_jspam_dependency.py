"""Checks that this project's jspam dependency installs and runs."""

import pytest

# Skipped where the dependencies aren't installed, such as native Windows.
np = pytest.importorskip("numpy")
jspam = pytest.importorskip("jspam")


def test_jspam_runs_a_collision():
    # Model SDSS 587722984435351614 from JSPAM's archive/data.txt
    params = np.array(
        [
            -9.93853, -4.5805, 15.43348, -1.54025, -2.97899, 4.69234, 34.41671, 29.49607, 2.97913, 4.2501, 301.0,
            35.5, 310.80745, 321.98757, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]
    )  # fmt: skip
    final = jspam.run_collision(params, n1=10, n2=5)
    assert final.shape == (16, 6)
    assert np.isfinite(final).all()
