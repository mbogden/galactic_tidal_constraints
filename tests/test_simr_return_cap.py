"""Tests for run.max_return_gb (MDRN-4 step 10c): a run whose returned data would be too big for the one
process that holds it is stopped before anything is computed."""

import pytest

np = pytest.importorskip("numpy")
jspam = pytest.importorskip("jspam")

from galactic_tidal_constraints.simr import simr  # noqa: E402

# Model SDSS 587722984435351614 (as in test_simr.py).
PARAMS = np.array(
    [-9.93853, -4.5805, 15.43348, -1.54025, -2.97899, 4.69234, 34.41671, 29.49607, 2.97913, 4.2501,
     301.0, 35.5, 310.80745, 321.98757]
)  # fmt: skip
SMALL = {"n1": 20, "n2": 10}
PARTICLE_BYTES = (20 + 10 + 1) * 6 * 4  # one sim.particles array


def request(cap="default", returned=("sim.particles",), **more):
    run = {"return": list(returned)}
    if cap != "default":
        run["max_return_gb"] = cap
    return {"run": run, "simulator": SMALL, **more}


def _never(*args, **kwargs):
    raise AssertionError("simulated, though the run should have stopped first")


def test_the_default_is_one_gb_and_ordinary_runs_are_far_below_it():
    result = simr.run(PARAMS, request())
    assert simr.DEFAULT_MAX_RETURN_GB == 1.0 and result.config["run"]["max_return_gb"] == 1.0
    assert result.data["sim.particles"][0, 0].nbytes == PARTICLE_BYTES


def test_a_run_over_the_cap_stops_before_anything_is_computed(monkeypatch, tmp_path):
    monkeypatch.setitem(simr.SIM_DATA, "sim.particles", _never)
    just_under = 2 * PARTICLE_BYTES / 2**30
    models = np.stack([PARAMS, PARAMS * 1.01, PARAMS * 1.02])
    with pytest.raises(ValueError) as error:
        simr.run(models, request(cap=just_under), output=tmp_path / "runs")
    message = str(error.value)
    assert "run.return would hold 2.18 KB in this process's memory" in message
    assert "sim.particles 2.18 KB (3 arrays)" in message
    assert "Save it instead" in message and "null for no cap" in message
    assert not (tmp_path / "runs").exists()  # not even the catalog


def test_exactly_at_the_cap_runs_and_null_means_no_cap():
    exactly = PARTICLE_BYTES / 2**30
    assert simr.run(PARAMS, request(cap=exactly)).status["sim"].tolist() == [["ran"]]
    with pytest.raises(ValueError, match="more than run.max_return_gb"):
        simr.run(PARAMS, request(cap=exactly * 0.99))
    unlimited = simr.run(PARAMS, request(cap=None))
    assert unlimited.status["sim"].tolist() == [["ran"]] and unlimited.config["run"]["max_return_gb"] is None


def test_orbits_count_at_their_longest_and_the_message_says_up_to():
    one_orbit = simr.ORBIT_ROWS_AT_MOST * 7 * 4
    with pytest.raises(ValueError, match="run.return would hold up to 820.39 KB"):
        simr.run(PARAMS, request(cap=0.9 * one_orbit / 2**30, returned=["sim.orbit"]))
    result = simr.run(PARAMS, request(cap=one_orbit / 2**30, returned=["sim.orbit"]))
    assert result.data["sim.orbit"][0, 0].nbytes < one_orbit  # the real one is shorter


def test_saving_and_metrics_dont_count_toward_the_cap(tmp_path):
    saving = {"run": {"save": ["sim.particles"], "max_return_gb": 1e-9}, "simulator": SMALL,
              "sim_metrics": [{"method": "closest_approach", "inputs": ["sim.search_orbit"]}]}
    result = simr.run(PARAMS, saving, output=tmp_path)
    assert result.status["sim_metrics"].tolist() == [[["ran"]]]


def test_a_plan_reports_the_cap_without_raising():
    over = simr.plan(PARAMS, request(cap=0.5 * PARTICLE_BYTES / 2**30))
    assert over["return"]["over_cap"] is True and over["max_return_gb"] == 0.5 * PARTICLE_BYTES / 2**30
    assert "OVER THE CAP, so the run would not start" in simr.format_plan(over)
    under = simr.plan(PARAMS, request())
    assert under["return"]["over_cap"] is False and under["max_return_gb"] == 1.0
    assert "returned in memory: 744 bytes (run.max_return_gb allows 1 GB)" in simr.format_plan(under)
    assert simr.plan(PARAMS, request(cap=None))["return"]["over_cap"] is False


@pytest.mark.parametrize("cap", [0, -1, "1", True, float("inf"), float("nan")])
def test_a_bad_cap_is_a_config_error(cap):
    with pytest.raises(ValueError, match="run.max_return_gb must be a number above 0, or null for no cap"):
        simr.run(PARAMS, request(cap=cap))


def test_a_recorded_run_keeps_its_cap(tmp_path):
    simr.run(PARAMS, {"run": {"max_return_gb": 0.25}, "simulator": SMALL}, output=tmp_path / "runs.json")
    assert simr.load_catalog(tmp_path / "runs.json")["runs"][0]["run"]["max_return_gb"] == 0.25
    assert simr.load_run(tmp_path / "runs.json")["request"]["run"]["max_return_gb"] == 0.25
