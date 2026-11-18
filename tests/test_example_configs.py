"""The example configs in examples/configs/ stay valid, and defaults.json stays equal to the code's defaults.

docs/configuration.md shows these files, so a change to a config field or default fails here until the
examples (and the doc) are updated.
"""

import json
from dataclasses import fields
from pathlib import Path

import pytest

# Skipped where the dependencies aren't installed, such as native Windows.
np = pytest.importorskip("numpy")
jspam = pytest.importorskip("jspam")

from galactic_tidal_constraints.simr import simr  # noqa: E402
from galactic_tidal_constraints.simr.__main__ import _read_config  # noqa: E402
from galactic_tidal_constraints.simr.image_creator import ImageConfig  # noqa: E402
from galactic_tidal_constraints.simr.simulation import SimulatorConfig  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "examples" / "configs"


def load(name: str) -> dict:
    return json.loads((CONFIGS / name).read_text(encoding="utf-8"))


def test_defaults_lists_every_field():
    """Every SimulatorConfig and ImageConfig field is written out, so the file is a complete reference."""
    request = load("defaults.json")
    assert set(request) == set(simr.REQUEST_KEYS)
    assert list(request["run"]) == list(simr.RUN_KEYS)
    assert list(request["simulator"]) == [f.name for f in fields(SimulatorConfig)]
    assert list(request["image"]) == [f.name for f in fields(ImageConfig)]
    assert request["sim_metrics"] == [] and request["img_metrics"] == []  # the default: no metrics


def test_defaults_equal_the_code():
    """Read the way simr reads a request, the file gives exactly the default configs."""
    returned, saved, simulators, images, sim_metrics, img_metrics = simr._read_request(load("defaults.json"))
    assert returned == saved == []
    assert simulators == [SimulatorConfig()]
    assert images == [ImageConfig()]
    assert sim_metrics == [] and img_metrics == []


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.json")), ids=lambda p: p.name)
def test_every_example_config_is_valid(path, caplog):
    """Each example loads as the command line loads it (with no warning) and passes simr's checks."""
    request = _read_config(str(path))
    assert not caplog.records
    simr._read_request(request)


def test_compare_lnl_runs_every_combination():
    _, saved, simulators, images, _, _ = simr._read_request(load("compare_lnl.json"))
    assert saved == ["sim.orbit", "sim.particles", "img.model"]
    assert [s.lnl for s in simulators] == [0.1, 0.001]
    assert images[0] == ImageConfig()
    assert (images[1].blur_sigma, images[1].brightness) == (1.0, "log")


def test_configuration_doc_lists_every_data_name():
    """docs/configuration.md's Data names table names everything run.return and run.save accept."""
    doc = (REPO / "docs" / "configuration.md").read_text(encoding="utf-8")
    missing = [name for name in [*simr.SIM_DATA, *simr.IMAGE_DATA] if f"| `{name}` |" not in doc]
    assert not missing, f"add to docs/configuration.md (Data names): {missing}"


def test_example_models_file():
    models = simr.read_models(CONFIGS / "models.txt")
    assert models.shape == (2, 14)
    assert models[1, 0] == -9.93853  # SDSS 587722984435351614, as in the other tests
