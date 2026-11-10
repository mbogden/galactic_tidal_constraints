"""Tests for the Simulator stage (simr/sim.py). Each section covers one step of the rebuild."""

from dataclasses import FrozenInstanceError, asdict, replace

import pytest

# Skipped where the dependencies aren't installed, such as native Windows.
np = pytest.importorskip("numpy")
pytest.importorskip("jspam")

from galactic_tidal_constraints.simr import sim  # noqa: E402
from galactic_tidal_constraints.simr.sim import SimulatorConfig  # noqa: E402

# Model SDSS 587722984435351614, a full 34-value line from JSPAM's archive/data.txt.
MODEL_STRING = (
    "-9.93853,-4.5805,15.43348,-1.54025,-2.97899,4.69234,34.41671,29.49607,2.97913,4.2501,301.0,35.5,"
    "310.80745,321.98757,0.3,0.3,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0,0,0.24008,-2.41999,5.62715,0.0,1.0,0.0,1.0,0.0,0.0"
)
PARAMS = np.array(MODEL_STRING.split(",")[:22], dtype=float)

# Last row of a.101 from the original `basic_run "<model string>"`: the secondary's center at the end.
EXPECTED_SECONDARY = [-9.9400701, -4.5834789, 15.438172, -1.5399158, -2.9788359, 4.6918210]

SMALL = SimulatorConfig(n1=20, n2=10)  # quick runs, 31 rows
MEDIUM = SimulatorConfig(n1=50, n2=50)


# --- Step 1: one run ---------------------------------------------------------------------------------


def test_simulate_shape_and_finite():
    final = sim.simulate(PARAMS, SMALL, seed=1)
    assert final.shape == (31, 6)
    assert np.isfinite(final).all()


def test_simulate_matches_original_program():
    final = sim.simulate(PARAMS, SMALL, seed=1)
    np.testing.assert_allclose(final[-1], EXPECTED_SECONDARY, rtol=1e-5, atol=1e-6)


def test_simulate_accepts_a_list():
    final = sim.simulate(list(PARAMS), SMALL, seed=1)
    assert final.shape == (31, 6)


def test_simulate_rejects_wrong_number_of_parameters():
    with pytest.raises(ValueError):
        sim.simulate(PARAMS[:21], SMALL)


# --- Step 2: parameter names and model strings -------------------------------------------------------


def test_param_names_match_spam_order():
    assert len(sim.PARAM_NAMES) == 22
    assert len(set(sim.PARAM_NAMES)) == 22
    assert sim.PARAM_NAMES[:8] == ("x", "y", "z", "vx", "vy", "vz", "mass1", "mass2")
    assert sim.PARAM_INDEX["mass1"] == 6
    assert sim.PARAM_INDEX["epsilon2"] == 15
    assert all(sim.PARAM_NAMES[i] == name for name, i in sim.PARAM_INDEX.items())


def test_parse_model_string_reads_first_22_values():
    params = sim.parse_model_string(MODEL_STRING)
    assert params.shape == (22,)
    np.testing.assert_array_equal(params, PARAMS)


def test_parse_model_string_exactly_22_and_whitespace():
    exact = ",".join(MODEL_STRING.split(",")[:22])
    np.testing.assert_array_equal(sim.parse_model_string(f"  {exact}\n"), PARAMS)


def test_parse_model_string_rejects_too_few_values():
    with pytest.raises(ValueError):
        sim.parse_model_string("1,2,3")


# --- Step 3: seeds -----------------------------------------------------------------------------------


def test_same_seed_gives_identical_particles():
    first = sim.simulate(PARAMS, MEDIUM, seed=7)
    second = sim.simulate(PARAMS, MEDIUM, seed=7)
    np.testing.assert_array_equal(first, second)


def test_different_seeds_change_particles_not_orbit():
    first = sim.simulate(PARAMS, MEDIUM, seed=7)
    second = sim.simulate(PARAMS, MEDIUM, seed=8)
    assert not np.allclose(first[:-1], second[:-1])
    np.testing.assert_allclose(first[-1], second[-1], rtol=1e-12)


def test_no_seed_differs_each_run():
    first = sim.simulate(PARAMS, MEDIUM)
    second = sim.simulate(PARAMS, MEDIUM)
    assert not np.allclose(first[:-1], second[:-1])


# --- Step 4a: the starting state ---------------------------------------------------------------------


def test_initial_state_shape_and_differs_from_end():
    start = sim.initial_state(PARAMS, SMALL, seed=3)
    end = sim.simulate(PARAMS, SMALL, seed=3)
    assert start.shape == end.shape == (31, 6)
    assert not np.allclose(start, end)


def test_initial_state_is_repeatable():
    first = sim.initial_state(PARAMS, SMALL, seed=3)
    second = sim.initial_state(PARAMS, SMALL, seed=3)
    np.testing.assert_array_equal(first, second)


# --- Steps 4a and 4b: disk heat ----------------------------------------------------------------------

HEAT = 0.4
N = 2000


def heat_kick(heat_mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Starting states with and without heat, one seed. Returns (kick per disk particle, position change)."""
    cold = sim.initial_state(PARAMS, SimulatorConfig(n1=N, n2=N), seed=5)
    hot_config = SimulatorConfig(n1=N, n2=N, heat1=HEAT, heat2=HEAT, heat_mode=heat_mode)
    hot = sim.initial_state(PARAMS, hot_config, seed=5)
    return hot[:-1, 3:] - cold[:-1, 3:], hot[:, :3] - cold[:, :3]


@pytest.mark.parametrize("heat_mode", ["wallin", "centered"])
def test_heat_only_changes_disk_velocities(heat_mode):
    _, moved = heat_kick(heat_mode)
    np.testing.assert_array_equal(moved, 0.0)


def test_wallin_heat_kicks_are_positive_and_drift():
    kick, _ = heat_kick("wallin")
    assert kick.min() >= -1e-12
    assert kick.max() < HEAT + 1e-12
    np.testing.assert_allclose(kick.mean(axis=0), HEAT / 2, rtol=0.05)


def test_centered_heat_kicks_have_no_drift():
    kick, _ = heat_kick("centered")
    assert kick.min() >= -HEAT / 2 - 1e-12
    assert kick.max() < HEAT / 2 + 1e-12
    np.testing.assert_allclose(kick.mean(axis=0), 0.0, atol=0.02 * HEAT)


def test_heat_modes_differ_only_by_the_drift():
    wallin, _ = heat_kick("wallin")
    centered, _ = heat_kick("centered")
    np.testing.assert_allclose(wallin - centered, HEAT / 2, rtol=1e-9)


def test_no_heat_gives_the_same_run_in_both_modes():
    wallin = sim.simulate(PARAMS, replace(SMALL, heat_mode="wallin"), seed=5)
    centered = sim.simulate(PARAMS, replace(SMALL, heat_mode="centered"), seed=5)
    np.testing.assert_array_equal(wallin, centered)


# --- Step 4c: SimulatorConfig ------------------------------------------------------------------------


def test_config_defaults():
    config = SimulatorConfig()
    assert asdict(config) == {"n1": 1000, "n2": 1000, "heat1": 0.0, "heat2": 0.0, "heat_mode": "centered"}


def test_simulate_uses_default_config():
    default = sim.initial_state(PARAMS, seed=5)
    explicit = sim.initial_state(PARAMS, SimulatorConfig(), seed=5)
    assert default.shape == (2001, 6)
    np.testing.assert_array_equal(default, explicit)


def test_config_is_frozen():
    with pytest.raises(FrozenInstanceError):
        SMALL.n1 = 5


def test_config_round_trips_through_a_dict():
    config = SimulatorConfig(n1=300, n2=200, heat1=0.1, heat2=0.2, heat_mode="wallin")
    assert SimulatorConfig(**asdict(config)) == config


def test_config_accepts_numpy_integers():
    assert SimulatorConfig(n1=np.int64(10)).n1 == 10


@pytest.mark.parametrize(
    "bad",
    [{"n1": 0}, {"n2": -5}, {"n1": 10.5}, {"heat1": -0.1}, {"heat2": -1}, {"heat_mode": "centred"}],
)
def test_config_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        SimulatorConfig(**bad)


def test_config_rejects_unknown_fields():
    with pytest.raises(TypeError):
        SimulatorConfig(potential="point_mass")
