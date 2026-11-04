"""Checks the Simulator stage."""

import numpy as np
import pytest

jspam = pytest.importorskip("jspam")

from galactic_tidal_constraints.simr import sim  # noqa: E402
from galactic_tidal_constraints.simr.sim import CollisionParams, Endings, SimulatorConfig  # noqa: E402

# Model SDSS 587722984435351614 from JSPAM's archive/data.txt
PARAMS = np.array(
    [
        -9.93853, -4.5805, 15.43348, -1.54025, -2.97899, 4.69234, 34.41671, 29.49607, 2.97913, 4.2501, 301.0,
        35.5, 310.80745, 321.98757, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
)  # fmt: skip

SMALL = SimulatorConfig(n1=20, n2=10)
MANY_ENDINGS = tuple(round(0.8 + 0.04 * i, 2) for i in range(11))


def test_params_survive_to_array_and_back():
    params = CollisionParams.from_array(PARAMS)
    assert params.mass2 == 29.49607
    assert params.theta1 == 310.80745
    np.testing.assert_array_equal(params.to_array(), PARAMS)
    with pytest.raises(ValueError):
        CollisionParams.from_array(PARAMS[:5])


def test_params_object_and_array_give_the_same_run():
    a = sim.simulate(PARAMS, SMALL, seed=1).snapshots
    b = sim.simulate(CollisionParams.from_array(PARAMS), SMALL, seed=1).snapshots
    np.testing.assert_array_equal(a, b)


def test_same_seed_same_particles():
    config = SimulatorConfig(n1=20, n2=10, heat1=0.1, heat2=0.1)
    a = sim.simulate(PARAMS, config, seed=1).snapshots
    np.testing.assert_array_equal(a, sim.simulate(PARAMS, config, seed=1).snapshots)
    assert not np.array_equal(a, sim.simulate(PARAMS, config, seed=2).snapshots)


def test_particle_run_needs_a_seed_but_an_orbit_does_not():
    with pytest.raises(ValueError):
        sim.simulate(PARAMS, SMALL)
    assert sim.simulate(PARAMS, SimulatorConfig(kind="orbit")).orbit is not None


def test_last_row_matches_the_orbit():
    particles = sim.simulate(PARAMS, SMALL, seed=1)
    orbit = sim.simulate(PARAMS, SimulatorConfig(kind="orbit"))
    np.testing.assert_array_equal(particles.snapshots[-1, -1], orbit.orbit[-1])
    assert orbit.t_start == particles.t_start
    assert orbit.times[-1] == particles.times[-1]
    assert orbit.orbit.shape == (particles.n_steps + 1, 6)
    assert orbit.snapshots is None


def test_two_mass_models_do_not_affect_each_other():
    other = SimulatorConfig(n1=20, n2=10, lnl=0.05, m_halo=3.0)
    a = sim.simulate(PARAMS, SMALL, seed=1).snapshots
    sim.simulate(PARAMS, other, seed=1)
    np.testing.assert_array_equal(a, sim.simulate(PARAMS, SMALL, seed=1).snapshots)


def test_old_many_endings_as_fractions():
    # The old many_endings_run's 11 outputs: 80, 84, ..., 120% of the run's steps.
    config = SimulatorConfig(n1=20, n2=10, endings=Endings("fraction", MANY_ENDINGS))
    result = sim.simulate(PARAMS, config, seed=1)
    assert result.snapshots.shape == (11, 31, 6)
    np.testing.assert_array_equal(result.steps, [round(f * result.n_steps) for f in MANY_ENDINGS])
    np.testing.assert_allclose(result.times, result.t_start + result.steps * sim.STEP_SIZE)


def test_every_unit_reaches_the_same_steps():
    timing = jspam.get_timing(PARAMS)
    steps = np.array([0, 1000, timing.n_steps])
    times = timing.t_start + steps * sim.STEP_SIZE
    assert sim.resolve_end_steps(Endings("step", tuple(map(float, steps))), timing) == list(steps)
    assert sim.resolve_end_steps(Endings("time", tuple(times)), timing) == list(steps)
    assert sim.resolve_end_steps(Endings("gyr", tuple(sim.to_gyr(times))), timing) == list(steps)
    assert sim.resolve_end_steps(Endings("fraction", tuple(steps / timing.n_steps)), timing) == list(steps)


def test_time_unit_is_about_87_myr():
    assert sim.to_gyr(1.0) == pytest.approx(0.0866, abs=1e-4)


def test_fraction_one_is_the_normal_end():
    a = sim.simulate(PARAMS, SMALL, seed=1)
    b = sim.simulate(PARAMS, SimulatorConfig(n1=20, n2=10, endings=Endings("fraction", (0.0, 1.0))), seed=1)
    np.testing.assert_array_equal(a.snapshots[0], b.snapshots[1])
    assert b.steps[0] == 0
    assert b.times[0] == a.t_start


@pytest.mark.parametrize(
    "endings",
    [
        Endings("fraction", (1.0, 0.5)),  # not ascending
        Endings("fraction", (0.5, 0.5)),  # repeated
        Endings("fraction", (0.5, 0.5000001)),  # two values land on one step
        Endings("gyr", (-100.0,)),  # before the start
        Endings("step", (10.5,)),  # not a whole step
    ],
)
def test_bad_endings_stop_the_run(endings):
    with pytest.raises(ValueError):
        sim.resolve_end_steps(endings, jspam.get_timing(PARAMS))


def test_bad_settings_are_rejected():
    with pytest.raises(ValueError):
        Endings(unit="days")
    for bad in (
        {"kind": "image"},
        {"potential": "mond"},
        {"heat_mode": "x"},
        {"normalization": "x"},
        {"lnl": -1.0},
    ):
        with pytest.raises(ValueError):
            SimulatorConfig(**bad)


def test_start_fallback_is_reported(caplog):
    wide = PARAMS.copy()
    wide[3:6] *= 0.2
    result = sim.simulate(wide, SimulatorConfig(kind="orbit"))
    assert result.start_fallback is True
    assert result.t_start == -5.0
    assert "start-time" in caplog.text
