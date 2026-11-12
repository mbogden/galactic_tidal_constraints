"""Tests for the SIMR workflow (simr/simr.py). Each section covers one step of the rebuild."""

import pytest

# Skipped where the dependencies aren't installed, such as native Windows.
np = pytest.importorskip("numpy")
jspam = pytest.importorskip("jspam")

from galactic_tidal_constraints.simr import sim, simr  # noqa: E402
from galactic_tidal_constraints.simr.sim import SimulatorConfig  # noqa: E402
from galactic_tidal_constraints.simr.simr import SimrResult  # noqa: E402

# Model SDSS 587722984435351614 (as in test_sim.py) and a second, heavier secondary.
PARAMS = np.array(
    [-9.93853, -4.5805, 15.43348, -1.54025, -2.97899, 4.69234, 34.41671, 29.49607, 2.97913, 4.2501,
     301.0, 35.5, 310.80745, 321.98757]
)  # fmt: skip
HEAVIER = PARAMS.copy()
HEAVIER[sim.PARAM_INDEX["mass2"]] *= 2
MODELS = np.stack([PARAMS, HEAVIER])

SMALL = {"n1": 20, "n2": 10}


def request(names, simulator=SMALL):
    return {"run": {"return": names}, "simulator": simulator}


# --- Step 8a: requests and returned data -------------------------------------------------------------


def test_returns_only_the_requested_names():
    result = simr.run(MODELS, request(["sim.orbit", "sim.timing"]))
    assert isinstance(result, SimrResult)
    assert list(result.data) == ["sim.orbit", "sim.timing"]


def test_result_holds_the_models_and_filled_in_simulators():
    result = simr.run(MODELS, request([], simulator=[SMALL, {**SMALL, "lnl": 0.01}]))
    np.testing.assert_array_equal(result.models, MODELS)
    assert result.simulators == [SimulatorConfig(**SMALL), SimulatorConfig(**SMALL, lnl=0.01)]
    assert result.simulators[0].heat_mode == "centered"  # a default the request didn't give
    assert simr.run(PARAMS, request([])).models.shape == (1, 14)


def test_nothing_requested_returns_no_data():
    assert simr.run(MODELS, request([])).data == {}
    assert simr.run(MODELS, {}).data == {}


def test_data_is_indexed_by_model_and_simulator():
    result = simr.run(MODELS, request(["sim.particles"], simulator=[SMALL, {**SMALL, "lnl": 0.01}]))
    particles = result.data["sim.particles"]
    assert particles.shape == (2, 2)
    assert particles[1, 0].shape == (31, 6)


def test_simulated_arrays_are_float32():
    result = simr.run(MODELS, request(["sim.particles", "sim.unperturbed", "sim.orbit"]))
    for name in ("sim.particles", "sim.unperturbed", "sim.orbit"):
        assert result.data[name][0, 0].dtype == np.float32, name


def test_cells_match_calling_sim_directly():
    config = SimulatorConfig(**SMALL)
    result = simr.run(MODELS, request(["sim.particles", "sim.orbit", "sim.timing"]))
    expected = sim.simulate_particles(HEAVIER, config).astype(np.float32)
    np.testing.assert_array_equal(result.data["sim.particles"][1, 0], expected)
    np.testing.assert_array_equal(
        result.data["sim.orbit"][1, 0], sim.simulate_orbit(HEAVIER, config).astype(np.float32)
    )
    assert result.data["sim.timing"][1, 0]["t_start"] == sim.get_timing(HEAVIER, config)["t_start"]


def test_entries_with_the_same_seed_start_from_the_same_particles():
    # Two entries that differ only in lnl: the disks start from the same particles.
    result = simr.run(MODELS[:1], request(["sim.unperturbed"], simulator=[SMALL, {**SMALL, "lnl": 0.01}]))
    np.testing.assert_array_equal(result.data["sim.unperturbed"][0, 0], result.data["sim.unperturbed"][0, 1])


def test_a_models_particles_dont_depend_on_the_other_models():
    alone = simr.run(MODELS[:1], request(["sim.particles"]))
    together = simr.run(MODELS, request(["sim.particles"]))
    np.testing.assert_array_equal(alone.data["sim.particles"][0, 0], together.data["sim.particles"][0, 0])


def test_entries_that_differ_only_in_seed_give_different_particles():
    result = simr.run(MODELS[:1], request(["sim.particles"], simulator=[SMALL, {**SMALL, "seed": 1}]))
    assert result.simulators[0].seed == 0  # the default
    assert result.simulators[1].seed == 1
    assert not np.array_equal(result.data["sim.particles"][0, 0], result.data["sim.particles"][0, 1])


def test_runs_repeat_exactly():
    first = simr.run(MODELS, request(["sim.particles"]))
    again = simr.run(MODELS, request(["sim.particles"]))
    for m in range(len(MODELS)):
        np.testing.assert_array_equal(first.data["sim.particles"][m, 0], again.data["sim.particles"][m, 0])


def test_one_model_and_one_simulator_object_are_accepted():
    result = simr.run(PARAMS, request(["sim.orbit"], simulator=SimulatorConfig(**SMALL)))
    assert result.data["sim.orbit"].shape == (1, 1)


def test_default_simulator_is_simulator_config():
    result = simr.run(PARAMS, {"run": {"return": ["sim.timing"]}})
    assert result.data["sim.timing"][0, 0] == sim.get_timing(PARAMS, SimulatorConfig())


@pytest.mark.parametrize(
    "bad, message",
    [
        ({"runn": {}}, "unknown keys in request"),
        ({"run": {"save": ["sim.orbit"]}}, "unknown keys in run"),
        ({"run": {"return": ["sim.orbits"]}}, "unknown names in run.return"),
        ({"run": {"return": "sim.orbit"}}, "must be a list"),
        ({"run": {"seed": 3}}, "unknown keys in run"),
        ({"simulator": {"seed": 1.5}}, r"simulator\[0\]: seed"),
        ({"simulator": []}, "empty"),
        ({"simulator": [SMALL, {"n_1": 5}]}, r"simulator\[1\]"),
        ({"simulator": [{"lnl": -1}]}, r"simulator\[0\]: lnl"),
    ],
)
def test_bad_requests_name_the_problem(bad, message):
    with pytest.raises(ValueError, match=message):
        simr.run(MODELS, bad)


@pytest.mark.parametrize("bad", [MODELS[:, :13], np.empty((0, 14)), np.full((1, 14), np.nan)])
def test_bad_models_are_rejected(bad):
    with pytest.raises(ValueError):
        simr.run(bad, request([]))
