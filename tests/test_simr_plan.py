"""Tests for simr.plan (MDRN-4 step 10): what a run would do, worked out without simulating."""

import json

import pytest

np = pytest.importorskip("numpy")
jspam = pytest.importorskip("jspam")

from galactic_tidal_constraints.simr import simr  # noqa: E402
from galactic_tidal_constraints.simr import simulation as sim  # noqa: E402
from galactic_tidal_constraints.simr.simulation import SimulatorConfig  # noqa: E402

# Model SDSS 587722984435351614 (as in test_simr.py) and a second, heavier secondary.
PARAMS = np.array(
    [-9.93853, -4.5805, 15.43348, -1.54025, -2.97899, 4.69234, 34.41671, 29.49607, 2.97913, 4.2501,
     301.0, 35.5, 310.80745, 321.98757]
)  # fmt: skip
HEAVIER = PARAMS.copy()
HEAVIER[sim.PARAM_INDEX["mass2"]] *= 2
MODELS = np.stack([PARAMS, HEAVIER])
SMALL = {"n1": 20, "n2": 10}

TARGET = sim.search_orbit(PARAMS, SimulatorConfig(**SMALL)).astype(np.float32)[::-1][::50].copy()
TARGET_IMAGE = np.random.default_rng(9).random((128, 128)).astype(np.float32)
EXTERNALS = {"target": TARGET, "target_image": TARGET_IMAGE}
FULL = {
    "run": {"return": ["sim.particles", "img.model", "drv.win", "drv.tidal"], "save": ["sim.orbit", "ext.target"]},
    "simulator": [SMALL, {**SMALL, "lnl": 0.01}],
    "image": [{}, {"blur_sigma": 1.0}, {"width": 64, "height": 32, "primary_center": [16, 16],
                                         "secondary_center": [48, 16]}],
    "derived": [
        {"name": "win", "method": "recent_approach_cosine_window", "inputs": ["ext.target"]},
        {"name": "model_win", "method": "recent_approach_cosine_window", "inputs": ["sim.search_orbit"]},
        {"name": "tidal", "method": "detect_edges", "inputs": ["img.unperturbed"]},
    ],
    "sim_metrics": [{"method": "orbit_time_error", "inputs": ["ext.target", "sim.search_orbit", "drv.win"]},
                    {"method": "closest_approach", "inputs": ["sim.search_orbit"]}],
    "img_metrics": [{"method": "absolute_difference", "inputs": ["img.model", "img.unperturbed"]}],
    "composite": {},
}
ORBIT_BYTES = simr.ORBIT_ROWS_AT_MOST * 7 * 4


def _never(*args, **kwargs):
    raise AssertionError("a plan must not simulate or make images")


@pytest.fixture
def no_simulating(monkeypatch):
    for item in simr.SIM_DATA:
        monkeypatch.setitem(simr.SIM_DATA, item, _never)
    for item, (needs, _) in simr.IMAGE_DATA.items():
        monkeypatch.setitem(simr.IMAGE_DATA, item, (needs, _never))


def test_a_plan_counts_everything_without_simulating(no_simulating):
    plan = simr.plan(MODELS, FULL, externals=EXTERNALS)
    assert (plan["models"], plan["simulators"], plan["image_entries"]) == (2, 2, 3)
    assert plan["simulations"] == 4 and plan["images"] == 12
    assert plan["sim_items"] == ["sim.particles", "sim.orbit", "sim.search_orbit", "sim.unperturbed"]
    assert plan["image_items"] == ["img.model", "img.unperturbed"]
    assert plan["derived"] == {"run": 1, "sim": 4, "img": 12}
    assert plan["metric_cells"] == {"sim_metrics": 8, "img_metrics": 12}
    assert plan["composite"] is True
    assert json.loads(json.dumps(plan)) == plan  # plain data


def test_sizes_are_exact_for_particles_and_images(no_simulating):
    returned = simr.plan(MODELS, FULL, externals=EXTERNALS)["return"]
    particles = (20 + 10 + 1) * 6 * 4
    assert returned["items"]["sim.particles"] == {"arrays": 4, "bytes": 4 * particles, "upper_bound": False}
    images = 2 * 2 * (128 * 128 + 128 * 128 + 32 * 64) * 4  # models x simulators x the three image entries
    assert returned["items"]["img.model"] == {"arrays": 12, "bytes": images, "upper_bound": False}
    assert returned["items"]["drv.tidal"] == {"arrays": 12, "bytes": images, "upper_bound": False}
    assert returned["items"]["drv.win"] == {"arrays": 1, "bytes": len(TARGET) * 4, "upper_bound": False}
    assert returned["bytes"] == 4 * particles + 2 * images + len(TARGET) * 4 and returned["upper_bound"] is False


def test_orbits_are_an_upper_bound_and_externals_are_counted_once(no_simulating):
    plan = simr.plan(MODELS, FULL, externals=EXTERNALS)
    saved = plan["save"]
    assert saved["items"]["sim.orbit"] == {"arrays": 4, "bytes": 4 * ORBIT_BYTES, "upper_bound": True}
    assert saved["items"]["ext.target"] == {"arrays": 1, "bytes": TARGET.nbytes, "upper_bound": False}
    assert saved["upper_bound"] is True
    assert plan["externals"]["target"] == {"source": "memory", "shape": list(TARGET.shape),
                                           "hash": simr.array_hash(TARGET), "bytes": TARGET.nbytes}
    assert "target_image" not in plan["externals"]  # given, but no entry uses it


def test_the_sizes_match_what_a_run_returns():
    request = {**FULL, "run": {"return": ["sim.particles", "sim.orbit", "sim.search_orbit", "img.model",
                                          "drv.model_win", "drv.tidal", "drv.win"]}}
    plan = simr.plan(MODELS, request, externals=EXTERNALS)
    result = simr.run(MODELS, request, externals=EXTERNALS)
    for name, item in plan["return"]["items"].items():
        data = result.data[name]
        cells = [data] if name == "drv.win" else list(data.ravel())
        actual = sum(cell.nbytes for cell in cells)
        assert item["arrays"] == len(cells), name
        if item["upper_bound"]:
            assert actual <= item["bytes"], name
        else:
            assert actual == item["bytes"], name
    assert plan["simulations"] == (result.status["sim"] == "ran").sum()
    assert plan["images"] == (result.status["img"] == "ran").sum()
    assert plan["metric_cells"]["sim_metrics"] == result.sim_metrics.size


def test_only_what_is_asked_for_is_planned(no_simulating):
    orbit_only = simr.plan(MODELS, {"simulator": SMALL, "sim_metrics": [FULL["sim_metrics"][1]]})
    assert orbit_only["simulations"] == 2 and orbit_only["sim_items"] == ["sim.search_orbit"]
    assert orbit_only["images"] == 0 and orbit_only["image_items"] == []
    assert orbit_only["return"] == {"bytes": 0, "upper_bound": False, "items": {}, "over_cap": False}
    nothing = simr.plan(MODELS, {"simulator": SMALL})
    assert nothing["simulations"] == 0 and nothing["externals"] == {}


def test_a_plan_makes_the_same_checks_as_a_run(no_simulating, tmp_path):
    with pytest.raises(ValueError, match="unknown keys in request"):
        simr.plan(MODELS, {"simulater": SMALL})
    with pytest.raises(ValueError, match="names an external the run wasn't given"):
        simr.plan(MODELS, FULL)
    with pytest.raises(ValueError, match="must be a 'orbit' array"):
        simr.plan(MODELS, FULL, externals={"target": TARGET[:, :3]})
    with pytest.raises(ValueError, match="run.save: ext.spare isn't an external"):
        simr.plan(MODELS, {**FULL, "run": {"save": ["ext.spare"]}}, externals=EXTERNALS)
    with pytest.raises(ValueError, match="models must have shape"):
        simr.plan(MODELS[:, :5], FULL, externals=EXTERNALS)


def test_the_output_is_only_checked_by_name(no_simulating, tmp_path):
    with pytest.raises(ValueError, match="run.save names data to save, which needs an output folder"):
        simr.plan(MODELS, FULL, externals=EXTERNALS, output=tmp_path / "runs.json")
    with pytest.raises(ValueError):
        simr.plan(MODELS, FULL, externals=EXTERNALS, output=tmp_path / "runs.txt")
    simr.plan(MODELS, FULL, externals=EXTERNALS, output=tmp_path / "runs")
    assert not list(tmp_path.iterdir())  # nothing read or written
    simr.plan(MODELS, FULL, externals=EXTERNALS)  # without an output, run.save's sizes are still planned


def test_format_plan_reads_as_a_few_lines(no_simulating):
    text = simr.format_plan(simr.plan(MODELS, FULL, externals=EXTERNALS))
    assert text.splitlines()[0] == "2 models x 2 simulator entries x 3 image entries"
    assert "  simulations: 4 (each computes sim.particles, sim.orbit, sim.search_orbit, sim.unperturbed)" in text
    assert "  images:      12 (each makes img.model, img.unperturbed)" in text
    assert "1 once for the run, 4 per simulation, 12 per image" in text
    assert "8 simulation-level cells, 12 image-level cells, and one composite per model" in text
    assert "external target: shape (601, 7)" in text and "from memory" in text
    assert "saved to disk: up to " in text and "    sim.orbit: 4 arrays, up to 3.20 MB" in text
    assert "returned in memory: 0 bytes (nothing)" in simr.format_plan(simr.plan(MODELS, {"simulator": SMALL}))
