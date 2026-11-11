"""Tests for the Simulator stage (simr/sim.py). Each section covers one step of the rebuild."""

from dataclasses import FrozenInstanceError, asdict, replace

import pytest

# Skipped where the dependencies aren't installed, such as native Windows.
np = pytest.importorskip("numpy")
jspam = pytest.importorskip("jspam")

from galactic_tidal_constraints.simr import sim  # noqa: E402
from galactic_tidal_constraints.simr.sim import SimulatorConfig  # noqa: E402

# Model SDSS 587722984435351614, a full 34-value line from JSPAM's archive/data.txt.
MODEL_STRING = (
    "-9.93853,-4.5805,15.43348,-1.54025,-2.97899,4.69234,34.41671,29.49607,2.97913,4.2501,301.0,35.5,"
    "310.80745,321.98757,0.3,0.3,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0,0,0.24008,-2.41999,5.62715,0.0,1.0,0.0,1.0,0.0,0.0"
)
PARAMS = np.array(MODEL_STRING.split(",")[:14], dtype=float)

# Last row of a.101 from the original `basic_run "<model string>"`: the secondary's center at the end.
EXPECTED_SECONDARY = [-9.9400701, -4.5834789, 15.438172, -1.5399158, -2.9788359, 4.6918210]

SMALL = SimulatorConfig(n1=20, n2=10)  # quick runs, 31 rows
MEDIUM = SimulatorConfig(n1=50, n2=50)


# --- Step 1: one run ---------------------------------------------------------------------------------


def test_simulate_shape_and_finite():
    final = sim.simulate_particles(PARAMS, SMALL, seed=1)
    assert final.shape == (31, 6)
    assert np.isfinite(final).all()


def test_simulate_matches_original_program():
    original = replace(SMALL, normalization="wallin", start="wallin")  # the original's physics and start time
    final = sim.simulate_particles(PARAMS, original, seed=1)
    np.testing.assert_allclose(final[-1], EXPECTED_SECONDARY, rtol=1e-5, atol=1e-6)


def test_simulate_accepts_a_list():
    final = sim.simulate_particles(list(PARAMS), SMALL, seed=1)
    assert final.shape == (31, 6)


def test_simulate_rejects_wrong_number_of_parameters():
    with pytest.raises(ValueError):
        sim.simulate_particles(PARAMS[:13], SMALL)


# --- Step 2: parameter names and model strings -------------------------------------------------------


def test_param_names_match_spam_order():
    assert len(sim.PARAM_NAMES) == 14
    assert len(set(sim.PARAM_NAMES)) == 14
    assert sim.PARAM_NAMES[:8] == ("x", "y", "z", "vx", "vy", "vz", "mass1", "mass2")
    assert sim.PARAM_INDEX["mass1"] == 6
    assert sim.PARAM_INDEX["theta2"] == 13
    assert all(sim.PARAM_NAMES[i] == name for name, i in sim.PARAM_INDEX.items())


def test_parse_model_string_reads_first_14_values():
    params = sim.parse_model_string(MODEL_STRING)
    assert params.shape == (14,)
    np.testing.assert_array_equal(params, PARAMS)


def test_parse_model_string_exactly_14_or_22_and_whitespace():
    for n in (14, 22):
        exact = ",".join(MODEL_STRING.split(",")[:n])
        np.testing.assert_array_equal(sim.parse_model_string(f"  {exact}\n"), PARAMS)


def test_parse_model_string_rejects_too_few_values():
    with pytest.raises(ValueError):
        sim.parse_model_string("1,2,3")


# --- Step 3: seeds -----------------------------------------------------------------------------------


def test_same_seed_gives_identical_particles():
    first = sim.simulate_particles(PARAMS, MEDIUM, seed=7)
    second = sim.simulate_particles(PARAMS, MEDIUM, seed=7)
    np.testing.assert_array_equal(first, second)


def test_different_seeds_change_particles_not_orbit():
    first = sim.simulate_particles(PARAMS, MEDIUM, seed=7)
    second = sim.simulate_particles(PARAMS, MEDIUM, seed=8)
    assert not np.allclose(first[:-1], second[:-1])
    np.testing.assert_allclose(first[-1], second[-1], rtol=1e-12)


def test_no_seed_differs_each_run():
    first = sim.simulate_particles(PARAMS, MEDIUM)
    second = sim.simulate_particles(PARAMS, MEDIUM)
    assert not np.allclose(first[:-1], second[:-1])


# --- Step 4a: the starting state ---------------------------------------------------------------------


def test_initial_state_shape_and_differs_from_end():
    start = sim.initial_state(PARAMS, SMALL, seed=3)
    end = sim.simulate_particles(PARAMS, SMALL, seed=3)
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
    wallin = sim.simulate_particles(PARAMS, replace(SMALL, heat_mode="wallin"), seed=5)
    centered = sim.simulate_particles(PARAMS, replace(SMALL, heat_mode="centered"), seed=5)
    np.testing.assert_array_equal(wallin, centered)


# --- Step 4c: SimulatorConfig ------------------------------------------------------------------------


def test_config_defaults():
    config = SimulatorConfig()
    assert asdict(config) == {
        "n1": 1000, "n2": 1000, "heat1": 0.0, "heat2": 0.0, "heat_mode": "centered",
        "lnl": 0.1, "r_scale": 10.0, "m_halo": 5.8, "m_bulge": 0.3333, "h_bulge": 0.2, "h_disk": 1.0,
        "normalization": "total", "start": "pass", "start_pass": 1, "no_pass": -5.0,
    }  # fmt: skip


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
    [
        {"n1": 0}, {"n2": -5}, {"n1": 10.5}, {"heat1": -0.1}, {"heat2": -1}, {"heat_mode": "centred"},
        {"lnl": -0.1}, {"r_scale": 0}, {"m_halo": -1}, {"m_bulge": 0}, {"h_bulge": -0.2}, {"h_disk": 0},
        {"normalization": "unit"},
    ],  # fmt: skip
)
def test_config_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        SimulatorConfig(**bad)


def test_config_rejects_unknown_fields():
    with pytest.raises(TypeError):
        SimulatorConfig(potential="point_mass")


# --- Step 5a: the mass model -------------------------------------------------------------------------

WALLIN = SimulatorConfig(normalization="wallin")
WALLIN_NORM = 7.13333

# Line 3 of JSPAM's archive/data.txt: the secondary passes close enough for dynamical friction to matter.
CLOSE_PARAMS = np.array(
    [
        1.08119, -1.66798, -2.06826, -0.18438, -0.39693, -0.02919, 0.51654, 0.36192, 0.94043, 0.43877, 12.97415,
        135.11902, 97.54035, 305.01225,
    ]
)  # fmt: skip


def test_mass_profile_shape_and_total_normalization():
    profile = sim.mass_profile()
    assert set(profile) == {"radius", "halo", "disk", "bulge", "total", "force"}
    assert all(values.shape == (10000,) for values in profile.values())
    np.testing.assert_allclose(profile["total"][-1], 1.0, rtol=1e-12)  # "total" is the default


def test_wallin_normalization_gives_0_9955():
    np.testing.assert_allclose(sim.mass_profile(WALLIN)["total"][-1], 0.99549, rtol=1e-4)


def test_total_normalization_keeps_mass_one_when_masses_change():
    profile = sim.mass_profile(SimulatorConfig(m_halo=11.6, m_bulge=1.0))
    np.testing.assert_allclose(profile["total"][-1], 1.0, rtol=1e-12)


def test_components_sum_to_total():
    profile = sim.mass_profile()
    np.testing.assert_allclose(profile["halo"] + profile["disk"] + profile["bulge"], profile["total"], rtol=1e-12)


def test_each_config_mass_value_reaches_its_component():
    """Each field changes the part of the galaxy it names, so a swap between them would fail here."""
    base = sim.mass_profile(WALLIN)
    np.testing.assert_allclose(sim.mass_profile(replace(WALLIN, m_bulge=0.6))["bulge"][-1], 0.6 / WALLIN_NORM, rtol=1e-6)
    np.testing.assert_allclose(sim.mass_profile(replace(WALLIN, m_halo=11.6))["halo"], 2 * base["halo"], rtol=1e-12)
    np.testing.assert_allclose(sim.mass_profile(replace(WALLIN, h_disk=2.0))["disk"][-1], 2.0 / WALLIN_NORM, rtol=1e-2)
    wide_bulge = sim.mass_profile(replace(WALLIN, h_bulge=0.5))
    np.testing.assert_allclose(wide_bulge["bulge"][-1], base["bulge"][-1], rtol=1e-6)
    assert wide_bulge["bulge"][100] < base["bulge"][100]
    stretched = sim.mass_profile(replace(WALLIN, r_scale=5.0))
    for component in ("halo", "disk", "bulge"):
        assert not np.allclose(stretched[component], base[component])


def test_lnl_changes_a_close_encounter():
    config = SimulatorConfig(n1=5, n2=5)
    base = sim.simulate_particles(CLOSE_PARAMS, replace(config, lnl=0.0), seed=1)
    friction = sim.simulate_particles(CLOSE_PARAMS, replace(config, lnl=0.5), seed=1)
    assert np.abs(base[-1, :3] - friction[-1, :3]).max() > 1e-3


def test_normalization_changes_the_run_slightly():
    """At the default masses the two differ by 0.45% in force (0.9955 vs 1), so the orbit moves a little."""
    total = sim.simulate_particles(PARAMS, SMALL, seed=1)
    wallin = sim.simulate_particles(PARAMS, replace(SMALL, normalization="wallin"), seed=1)
    assert not np.array_equal(total[-1], wallin[-1])
    np.testing.assert_allclose(total[-1], wallin[-1], rtol=0.05)


# --- Step 5c: 14-value models ------------------------------------------------------------------------


def test_simulate_appends_the_unused_parameters():
    """A 14-value model gives exactly the run of the original 22-value line (whose last 8 differ)."""
    original = np.array(MODEL_STRING.split(",")[:22], dtype=float)
    assert not np.array_equal(original[14:], sim.UNUSED_PARAMS)  # so the test means something
    ours = sim.simulate_particles(PARAMS, SMALL, seed=1)
    spam = jspam.run_collision(original, seed=1, **sim._jspam_args(PARAMS, SMALL))
    np.testing.assert_array_equal(ours, spam)


@pytest.mark.parametrize("index", range(14, 22))
def test_last_8_parameters_do_nothing(index):
    """Each of SPAM's last 8 parameters, changed on its own, leaves the run bit-identical."""
    full = np.concatenate([PARAMS, sim.UNUSED_PARAMS])
    changed = full.copy()
    changed[index] = 7.5
    args = sim._jspam_args(PARAMS, SMALL)
    np.testing.assert_array_equal(
        jspam.run_collision(full, seed=1, **args), jspam.run_collision(changed, seed=1, **args)
    )


def test_simulate_rejects_22_values():
    with pytest.raises(ValueError, match="parse_model_string"):
        sim.simulate_particles(np.concatenate([PARAMS, sim.UNUSED_PARAMS]), SMALL)


# --- Step 6a: timing and orbits ----------------------------------------------------------------------

WALLIN_START = SimulatorConfig(start="wallin")


def test_get_timing():
    timing = sim.get_timing(PARAMS, WALLIN_START)
    assert set(timing) == {
        "t_start", "n_steps", "start_fallback", "t_start_found", "t_closest", "closest_distance", "closest_speed",
        "start", "start_pass", "fallback_reason",
    }  # fmt: skip
    assert timing["start"] == "wallin"
    assert timing["t_start"] < timing["t_closest"] < 0
    assert timing["n_steps"] == int(-timing["t_start"] / 0.001) + 2
    assert not timing["start_fallback"]
    # Not bound here, so the start is the closest approach minus 10 x rout1 / speed
    lead = 10 * PARAMS[sim.PARAM_INDEX["rout1"]] / timing["closest_speed"]
    np.testing.assert_allclose(timing["t_start"], timing["t_closest"] - lead, atol=0.002)


def test_orbit_starts_and_ends_with_a_particle_run():
    timing = sim.get_timing(PARAMS)
    orbit = sim.simulate_orbit(PARAMS)
    assert orbit.shape == (timing["n_steps"] + 1, 7)
    np.testing.assert_allclose(orbit[0, 6], timing["t_start"])
    np.testing.assert_array_equal(orbit[0, :6], sim.initial_state(PARAMS, SMALL, seed=1)[-1])
    np.testing.assert_array_equal(orbit[-1, :6], sim.simulate_particles(PARAMS, SMALL, seed=1)[-1])


def test_orbit_closest_approach_matches_timing():
    timing = sim.get_timing(PARAMS, WALLIN_START)
    orbit = sim.simulate_orbit(PARAMS, WALLIN_START)
    separation = np.linalg.norm(orbit[:, :3], axis=1)
    closest = separation.argmin()
    np.testing.assert_allclose(separation[closest], timing["closest_distance"], rtol=1e-6)
    np.testing.assert_allclose(orbit[closest, 6], timing["t_closest"], atol=0.002)


def test_orbit_ignores_particles_and_heat_but_not_the_mass_model():
    orbit = sim.simulate_orbit(PARAMS)
    hot = SimulatorConfig(n1=5, n2=7, heat1=0.5, heat2=0.5)
    np.testing.assert_array_equal(sim.simulate_orbit(PARAMS, hot), orbit)
    # Every run ends at the observed position, so a weaker force changes where (and when) the orbit starts
    weaker = sim.simulate_orbit(PARAMS, SimulatorConfig(r_scale=12.0))
    np.testing.assert_allclose(weaker[-1, :3], orbit[-1, :3], atol=0.01)
    assert weaker[0, 6] < orbit[0, 6]
    assert not np.allclose(weaker[0, :3], orbit[0, :3], atol=0.5)


def test_search_can_pick_an_older_closest_approach():
    """With r_scale 9, SPAM's search picks an older pass (at the same distance as the recent one)."""
    default = sim.get_timing(PARAMS, WALLIN_START)
    stronger = sim.get_timing(PARAMS, replace(WALLIN_START, r_scale=9.0))
    assert default["t_closest"] > -4
    assert stronger["t_closest"] < -20
    assert stronger["start_fallback"]


def test_start_fallback_can_start_after_the_encounter():
    """With r_scale = 5 the search's start is before -12, so SPAM starts at -5, after the closest approach."""
    timing = sim.get_timing(PARAMS, replace(WALLIN_START, r_scale=5.0))
    assert timing["start_fallback"]
    assert timing["t_start"] == -5.0
    assert timing["t_start_found"] < -12
    assert timing["t_closest"] < timing["t_start"]


# --- Step 6b: choosing the start ---------------------------------------------------------------------


def test_search_orbit_runs_back_from_now():
    path = sim.search_orbit(PARAMS)
    assert abs(len(path) - 30001) <= 2
    np.testing.assert_allclose(path[0, :6], PARAMS[:6])
    assert path[0, 6] == 0.0 and path[-1, 6] < -29.99


def test_find_passes():
    passes = sim.find_passes(PARAMS)
    assert [p["pass"] for p in passes] == list(range(1, len(passes) + 1))
    assert all(a["t"] > b["t"] for a, b in zip(passes, passes[1:]))  # most recent first
    np.testing.assert_allclose(passes[0]["t"], -2.914, atol=0.002)
    distance = np.linalg.norm(sim.search_orbit(PARAMS)[:, :3], axis=1)
    np.testing.assert_allclose(passes[0]["distance"], distance.min(), rtol=1e-6)


def test_pass_start_uses_spams_lead_on_the_chosen_pass():
    timing = sim.get_timing(PARAMS)  # start="pass", start_pass=1 are the defaults
    assert timing["start"] == "pass" and timing["start_pass"] == 1
    lead = sim.LEAD_FACTOR * PARAMS[sim.PARAM_INDEX["rout1"]] / timing["closest_speed"]
    assert timing["t_start"] == timing["t_closest"] - lead
    # Where SPAM's search finds the same pass, the two starts agree to about a step
    np.testing.assert_allclose(timing["t_start"], sim.get_timing(PARAMS, WALLIN_START)["t_start"], atol=0.002)


def test_pass_start_fixes_spams_older_pass():
    """With r_scale 9, SPAM falls back to -5; pass 1 is the recent pass, and pass 2 the older one SPAM found."""
    config = SimulatorConfig(r_scale=9.0)
    recent = sim.get_timing(PARAMS, config)
    older = sim.get_timing(PARAMS, replace(config, start_pass=2))
    spam = sim.get_timing(PARAMS, replace(config, start="wallin"))
    assert recent["t_closest"] > -4 and recent["t_start"] < recent["t_closest"]
    np.testing.assert_allclose(older["t_closest"], spam["t_closest"], atol=0.002)
    assert older["t_start"] < older["t_closest"] < -20
    assert spam["t_start"] == -5.0


FIRST_APPROACH = PARAMS.copy()  # reversed, faster velocities: approaching now, unbound, no pass before now
FIRST_APPROACH[3:6] *= -3
OLD_PASS = PARAMS.copy()  # a light, slow, distant pair: separating now, but its pass was before t = -30
OLD_PASS[:3] *= 4
OLD_PASS[3:6] *= 0.2
OLD_PASS[6:8] *= 0.1
ERROR = SimulatorConfig(no_pass="error")


def test_missing_pass_falls_back_to_minus_5():
    for params in (FIRST_APPROACH, OLD_PASS):
        timing = sim.get_timing(params)
        assert timing["start"] == "fallback" and timing["start_fallback"]
        assert timing["t_start"] == -5.0 and timing["n_steps"] == 5002
        assert np.isnan(timing["t_closest"]) and "0 close passes" in timing["fallback_reason"]
    orbit = sim.simulate_orbit(FIRST_APPROACH, SMALL)
    assert orbit[0, 6] == -5.0
    np.testing.assert_array_equal(orbit[-1, :6], sim.simulate_particles(FIRST_APPROACH, SMALL, seed=1)[-1])
    assert sim.get_timing(FIRST_APPROACH, SimulatorConfig(no_pass=-4.0))["t_start"] == -4.0
    found = sim.get_timing(PARAMS)
    assert found["fallback_reason"] is None and not found["start_fallback"]


def test_missing_pass_is_an_error_when_asked():
    count = len(sim.find_passes(PARAMS))
    with pytest.raises(ValueError, match=f"start_pass={count + 1}, but the centers make {count} close passes back"):
        sim.get_timing(PARAMS, replace(ERROR, start_pass=count + 1))


def test_first_approach_reason():
    assert FIRST_APPROACH[:3] @ FIRST_APPROACH[3:6] < 0  # approaching
    with pytest.raises(ValueError, match="0 close passes .* approaching now, with no earlier pass"):
        sim.simulate_particles(FIRST_APPROACH, replace(ERROR, n1=20, n2=10), seed=1)  # fails before any particles


def test_pass_older_than_the_window_reason():
    assert OLD_PASS[:3] @ OLD_PASS[3:6] > 0  # separating
    with pytest.raises(ValueError, match="0 close passes .* still shrinking at t = -30, so an older pass"):
        sim.get_timing(OLD_PASS, ERROR)
    sim.get_timing(OLD_PASS, SimulatorConfig(start="wallin"))  # SPAM's search still picks something


def test_fixed_start():
    config = replace(SMALL, start=-8.0)
    timing = sim.get_timing(PARAMS, config)
    assert timing["start"] == "fixed" and timing["t_start"] == -8.0 and timing["n_steps"] == 8002
    orbit = sim.simulate_orbit(PARAMS, config)
    assert orbit[0, 6] == -8.0
    np.testing.assert_array_equal(orbit[-1, :6], sim.simulate_particles(PARAMS, config, seed=1)[-1])
    np.testing.assert_allclose(orbit[0, :6], sim.search_orbit(PARAMS)[8000, :6], rtol=1e-9, atol=1e-9)


def test_every_start_lies_on_the_search_path():
    # r_scale 9, pass 2: the older pass, starting at -27.1 (within the path, which ends at -30)
    for config in [SimulatorConfig(), WALLIN_START, SimulatorConfig(r_scale=9.0, start_pass=2)]:
        path = sim.search_orbit(PARAMS, config)
        start = sim.simulate_orbit(PARAMS, config)[0]
        row = int(np.ceil(-start[6] / 0.001 - 1e-9))  # placement steps back while t_start < t
        np.testing.assert_allclose(start[:6], path[row, :6], rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("bad", [{"start": "later"}, {"start": 0.5}, {"start": 0}, {"start_pass": 0}, {"start_pass": 1.5},
                                 {"no_pass": "skip"}, {"no_pass": 1.0}])
def test_start_settings_are_checked(bad):
    with pytest.raises(ValueError):
        SimulatorConfig(**bad)


def test_start_settings_survive_a_json_round_trip():
    import json

    for config in [SimulatorConfig(start=-8.5), SimulatorConfig(start_pass=2), WALLIN_START, ERROR]:
        assert SimulatorConfig(**json.loads(json.dumps(asdict(config)))) == config
