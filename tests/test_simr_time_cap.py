"""Tests for run.max_hours (MDRN-4 step 10): a run whose simulations could take longer than the cap, at
the worst case and spread over all its workers, is stopped before anything is computed."""

import builtins
from concurrent.futures import ProcessPoolExecutor

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
SPEED = {"seconds_per_particle_step": 1e-4, "seconds_per_orbit": 0.0}  # a slow machine, in round numbers
# One simulation at the worst case on that machine: 31 particles x 30,002 steps x 1e-4 s
ONE_RUN = 31 * simr.STEPS_AT_MOST * 1e-4


def request(hours="default", **run):
    if hours != "default":
        run["max_hours"] = hours
    return {"run": {"save": ["sim.particles"], **run}, "simulator": SMALL}


def _never(*args, **kwargs):
    raise AssertionError("simulated, though the run should have stopped first")


@pytest.fixture
def slow_machine(monkeypatch):
    monkeypatch.setattr(simr, "_SPEED", dict(SPEED))


def test_the_default_is_four_hours_and_ordinary_runs_never_time_anything(monkeypatch, tmp_path):
    monkeypatch.setattr(simr, "measure_speed", _never)  # far under the cap: let through without measuring
    result = simr.run(PARAMS, request(), output=tmp_path)
    assert simr.DEFAULT_MAX_HOURS == 4.0 and result.config["run"]["max_hours"] == 4.0
    assert result.status["sim"].tolist() == [["ran"]]


def test_a_run_over_the_cap_stops_before_anything_is_computed(slow_machine, monkeypatch, tmp_path):
    monkeypatch.setitem(simr.SIM_DATA, "sim.particles", _never)
    models = np.stack([PARAMS * (1 + 0.001 * i) for i in range(4)])
    with pytest.raises(ValueError) as error:
        simr.run(models, request(hours=3.5 * ONE_RUN / 3600), output=tmp_path / "runs")
    message = str(error.value)
    assert "the run's 4 simulations could take about 6.2 minutes on 1 worker" in message
    assert "every run counted at its longest" in message and "more than run.max_hours" in message
    assert "null for no cap" in message and 'time="exact"' in message
    assert not (tmp_path / "runs").exists()  # not even the catalog


def test_at_or_under_the_cap_runs_and_null_means_no_cap(slow_machine, tmp_path):
    models = np.stack([PARAMS, PARAMS * 1.001])
    assert simr.run(models, request(hours=2.01 * ONE_RUN / 3600), output=tmp_path / "a").status["sim"].tolist() \
        == [["ran"], ["ran"]]
    with pytest.raises(ValueError, match="more than run.max_hours"):
        simr.run(models, request(hours=1.99 * ONE_RUN / 3600), output=tmp_path / "b")
    unlimited = simr.run(models, request(hours=None), output=tmp_path / "c")
    assert unlimited.status["sim"].tolist() == [["ran"], ["ran"]] and unlimited.config["run"]["max_hours"] is None


def test_all_the_runs_workers_count(slow_machine, tmp_path):
    models = np.stack([PARAMS * (1 + 0.001 * i) for i in range(4)])
    hours = 2.5 * ONE_RUN / 3600  # four runs: too long for one worker, fine for two
    with pytest.raises(ValueError, match="on 1 worker "):
        simr.run(models, request(hours=hours), output=tmp_path / "solo")
    with ProcessPoolExecutor(max_workers=2) as pool:
        result = simr.run(models, request(hours=hours), output=tmp_path / "pool", map=pool.map)
        assert result.status["sim"].ravel().tolist() == ["ran"] * 4
        with pytest.raises(ValueError, match="on 2 workers "):
            simr.run(models, request(hours=1.5 * ONE_RUN / 3600), output=tmp_path / "pool2", map=pool.map)


def test_the_worker_count_comes_from_the_map(monkeypatch):
    assert simr._workers(builtins.map) == 1
    with ProcessPoolExecutor(max_workers=3) as pool:
        assert simr._workers(pool.map) == 3
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "64")  # under mpirun: 63 workers and the rank running the command
    assert simr._workers(lambda function, tasks: None) == 63
    monkeypatch.delenv("OMPI_COMM_WORLD_SIZE")
    assert simr._workers(lambda function, tasks: None) == simr.available_cpus()


def test_orbit_only_runs_and_a_fixed_start_are_counted_as_they_are(slow_machine, monkeypatch):
    orbit_only = {"run": {"max_hours": 1e-9}, "simulator": SMALL,
                  "sim_metrics": [{"method": "closest_approach", "inputs": ["sim.search_orbit"]}]}
    assert simr.run(PARAMS, orbit_only).status["sim_metrics"].tolist() == [[["ran"]]]  # no particles: no time
    short = {"run": {"return": ["sim.particles"], "max_hours": 1.1 * 31 * 2000 * 1e-4 / 3600},
             "simulator": {**SMALL, "start": -2.0}}
    assert simr.run(PARAMS, short).status["sim"].tolist() == [["ran"]]  # 2,000 steps, not 30,002


def test_a_plan_reports_the_time_cap_without_raising():
    over = simr.plan(PARAMS, request(hours=0.5 * ONE_RUN / 3600), time="exact", speed=SPEED)
    assert over["max_hours"] == 0.5 * ONE_RUN / 3600
    assert over["time"]["over_cap"] is True and over["time"]["worst_wall_seconds"] == pytest.approx(ONE_RUN)
    assert over["time"]["wall_seconds"] < over["time"]["worst_wall_seconds"]  # the real start is later than -30
    assert "OVER THE CAP, so the run would not start" in simr.format_plan(over)
    spread = simr.plan(PARAMS, request(hours=0.5 * ONE_RUN / 3600), time="worst", speed=SPEED, workers=4)
    assert spread["time"]["over_cap"] is False  # the same run over four workers
    line = [line for line in simr.format_plan(spread).splitlines() if "worst case on" in line][0]
    assert line.startswith("  worst case on 4 workers: about 23 seconds (run.max_hours allows ")
    assert "worst case on" not in simr.format_plan(simr.plan(PARAMS, request(hours=None), time="worst", speed=SPEED))


@pytest.mark.parametrize("hours", [0, -1, "4", "auto", True, float("inf"), float("nan")])
def test_a_bad_cap_is_a_config_error(hours):
    with pytest.raises(ValueError, match="run.max_hours must be a number above 0, or null for no cap"):
        simr.run(PARAMS, request(hours=hours))


def test_a_recorded_run_keeps_its_cap(tmp_path):
    simr.run(PARAMS, {"run": {"max_hours": 12}, "simulator": SMALL}, output=tmp_path / "runs.json")
    assert simr.load_catalog(tmp_path / "runs.json")["runs"][0]["run"]["max_hours"] == 12
    assert simr.load_run(tmp_path / "runs.json")["request"]["run"]["max_hours"] == 12
