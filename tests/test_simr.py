"""Checks the SIMR workflow: saving, loading, reuse, failures and parallel runs."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

pytest.importorskip("jspam")

from galactic_tidal_constraints.simr import config as config_module  # noqa: E402
from galactic_tidal_constraints.simr import sim, simr  # noqa: E402

MODEL_A = [1.5, 1.0, 0.5, -0.5, 0.5, 0.3, 1.0, 1.0, 1.0, 1.0, 30.0, 60.0, 20.0, 40.0, 0.3, 0.3] + [0.0] * 6
MODEL_B = [2.0, 0.5, 0.5, -0.4, 0.6, 0.2, 1.0, 0.5, 1.0, 0.8, 10.0, 20.0, 30.0, 40.0, 0.3, 0.3] + [0.0] * 6
# Passes the config's checks but SPAM refuses it, because the secondary has no velocity.
BAD_MODEL = [1.5, 1.0, 0.5, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 30.0, 60.0, 20.0, 40.0, 0.3, 0.3] + [0.0] * 6


def make_config(models=(MODEL_A, MODEL_B), **run):
    simulators = [
        {"n1": 8, "n2": 4, "endings": {"unit": "fraction", "values": [0.0, 1.0]}},
        {"n1": 8, "n2": 4, "lnl": 0.01},
        {"kind": "orbit"},
    ]
    return config_module.from_dict(
        {"models": list(models), "simulator": simulators, "run": {"seed": 5, **run}}
    )


def read_all(output, config):
    out = {}
    for m in range(len(config.models)):
        for s in range(len(config.simulator)):
            _, simulation = simr.load_simulation(simr.simulation_folder(output, m, s))
            out[m, s] = simulation.orbit if simulation.kind == "orbit" else simulation.snapshots
    return out


def assert_same_runs(a, b):
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_save_and_load_round_trip(tmp_path):
    config = sim.SimulatorConfig(n1=8, n2=4, endings=sim.Endings("fraction", (0.0, 1.0)))
    simulation = sim.simulate(np.array(MODEL_A), config, seed=1)
    settings = {"params": MODEL_A, "seed": 1}
    simr.save_simulation(tmp_path / "x", settings, simulation)

    loaded_settings, loaded = simr.load_simulation(tmp_path / "x")
    assert loaded_settings == settings
    assert loaded.kind == "particles"
    assert loaded.t_start == simulation.t_start
    assert loaded.n_steps == simulation.n_steps
    assert loaded.start_fallback == simulation.start_fallback
    assert loaded.orbit is None
    np.testing.assert_array_equal(loaded.snapshots, simulation.snapshots)
    np.testing.assert_array_equal(loaded.steps, simulation.steps)
    np.testing.assert_array_equal(loaded.times, simulation.times)
    assert sorted(p.name for p in (tmp_path / "x").iterdir()) == [simr.SETTINGS_FILE, simr.SIMULATION_FILE]


def test_orbit_round_trip(tmp_path):
    simulation = sim.simulate(np.array(MODEL_A), sim.SimulatorConfig(kind="orbit"))
    simr.save_simulation(tmp_path / "x", {}, simulation)
    _, loaded = simr.load_simulation(tmp_path / "x")
    assert loaded.snapshots is None
    np.testing.assert_array_equal(loaded.orbit, simulation.orbit)


def test_failed_write_leaves_no_file(tmp_path):
    def fail(_):
        raise OSError("disk full")

    with pytest.raises(OSError):
        simr._write_atomic(tmp_path / "file", fail)
    assert list(tmp_path.iterdir()) == []


def test_run_saves_everything_and_records_what_ran(tmp_path):
    config = make_config()
    summaries = simr.run(config, tmp_path)
    assert [s["status"] for s in summaries] == ["ran"] * 6
    assert [(s["model"], s["simulator"]) for s in summaries] == [(m, s) for m in range(2) for s in range(3)]

    saved = json.loads((tmp_path / simr.RUN_CONFIG_FILE).read_text())
    assert config_module.from_dict(saved["config"]) == config
    assert set(saved["provenance"]) == {"galactic_tidal_constraints", "jspam"}
    assert set(saved["provenance"]["jspam"]) == {"commit", "uncommitted_changes"}
    assert len(json.loads((tmp_path / simr.SUMMARY_FILE).read_text())) == 6


def test_second_run_reuses_and_gets_the_same_numbers(tmp_path):
    config = make_config()
    simr.run(config, tmp_path)
    first = read_all(tmp_path, config)
    summaries = simr.run(config, tmp_path)
    assert [s["status"] for s in summaries] == ["reused"] * 6
    assert_same_runs(first, read_all(tmp_path, config))


def test_a_lost_simulation_runs_again_and_comes_back_the_same(tmp_path):
    config = make_config()
    simr.run(config, tmp_path)
    first = read_all(tmp_path, config)
    (simr.simulation_folder(tmp_path, 1, 0) / simr.SIMULATION_FILE).unlink()
    statuses = [s["status"] for s in simr.run(config, tmp_path)]
    assert statuses.count("ran") == 1 and statuses.count("reused") == 5
    assert_same_runs(first, read_all(tmp_path, config))


def test_different_settings_stop_the_run(tmp_path):
    simr.run(make_config(), tmp_path)
    before = (tmp_path / simr.RUN_CONFIG_FILE).read_text()
    with pytest.raises(simr.SettingsMismatch):
        simr.run(make_config(seed=6), tmp_path)
    assert (tmp_path / simr.RUN_CONFIG_FILE).read_text() == before


def test_a_task_checks_its_own_folder_too(tmp_path):
    task = simr.make_tasks(make_config())[0]
    simr.run_task(task, 5, str(tmp_path))
    with pytest.raises(simr.SettingsMismatch):
        simr.run_task(task, 6, str(tmp_path))


def test_failed_model_does_not_stop_the_run(tmp_path, caplog):
    config = make_config(models=(MODEL_A, BAD_MODEL, MODEL_B))
    summaries = simr.run(config, tmp_path)
    statuses = {(s["model"], s["simulator"]): s["status"] for s in summaries}
    assert all(statuses[1, s] == "failed" for s in range(3))
    assert all(statuses[m, s] == "ran" for m in (0, 2) for s in range(3))
    assert "Model 1 on simulator 0 failed" in caplog.text
    assert not (simr.simulation_folder(tmp_path, 1, 0) / simr.SIMULATION_FILE).exists()


def test_results_do_not_depend_on_batching(tmp_path):
    config = make_config()
    simr.run(config, tmp_path / "a")
    tasks = simr.make_tasks(config)
    for task in tasks[::-1]:  # one at a time, in the opposite order
        simr.run_task(task, config.run.seed, str(tmp_path / "b"))
    assert_same_runs(read_all(tmp_path / "a", config), read_all(tmp_path / "b", config))


def test_save_none_returns_the_simulations_and_writes_nothing(tmp_path):
    config = make_config(save="none")
    summaries = simr.run(config, tmp_path / "unused")
    assert not (tmp_path / "unused").exists()
    assert summaries[0]["simulation"].snapshots.shape == (2, 13, 6)


def test_save_all_needs_an_output_folder():
    with pytest.raises(ValueError):
        simr.run(make_config())


def test_seeds_differ_by_model_and_repeat():
    assert simr.model_seed(1, 0) == simr.model_seed(1, 0)
    assert simr.model_seed(1, 0) != simr.model_seed(1, 1)
    assert simr.model_seed(1, 0) != simr.model_seed(2, 0)


@pytest.mark.parametrize("n_tasks,n_workers", [(1, 4), (10, 1), (100, 4), (7, 16), (1000, 32)])
def test_batches_cover_every_task_once_and_shrink(n_tasks, n_workers):
    tasks = [simr.Task(i, 0, (), {}) for i in range(n_tasks)]
    batches = simr.make_batches(tasks, n_workers, 0, None)
    assert [t.model_index for b in batches for t in b.tasks] == list(range(n_tasks))
    sizes = [len(b.tasks) for b in batches]
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[-1] == 1


def test_local_pool_matches_solo(tmp_path):
    config = make_config()
    simr.run(config, tmp_path / "solo")
    with ProcessPoolExecutor(3) as pool:
        simr.run(config, tmp_path / "pool", pool.map, 3)
    assert_same_runs(read_all(tmp_path / "solo", config), read_all(tmp_path / "pool", config))


def test_command_line(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(make_config().to_dict()))
    simr.main(["--config", str(path), "--output", str(tmp_path / "out"), "run.name=cli"])
    saved = json.loads((tmp_path / "out" / simr.RUN_CONFIG_FILE).read_text())
    assert saved["config"]["run"]["name"] == "cli"
    assert (simr.simulation_folder(tmp_path / "out", 1, 2) / simr.SIMULATION_FILE).exists()


@pytest.mark.slow
@pytest.mark.skipif(
    importlib.util.find_spec("mpi4py") is None or shutil.which("mpirun") is None,
    reason="needs mpi4py and mpirun",
)
def test_mpi_matches_solo(tmp_path):
    config = make_config()
    path = tmp_path / "run.json"
    path.write_text(json.dumps(config.to_dict()))
    simr.run(config, tmp_path / "solo")

    # --oversubscribe: this test may run with fewer CPUs than ranks.
    command = ["mpirun", "--oversubscribe", "-n", "3", sys.executable, "-m", "mpi4py.futures"]
    command += [
        "-m",
        "galactic_tidal_constraints.simr",
        "--config",
        str(path),
        "--output",
        str(tmp_path / "mpi"),
    ]
    command += ["--mode", "mpi"]
    env = {**os.environ, "OMPI_ALLOW_RUN_AS_ROOT": "1", "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1"}
    subprocess.run(command, check=True, env=env, timeout=300)

    assert_same_runs(read_all(tmp_path / "solo", config), read_all(tmp_path / "mpi", config))
    summaries = json.loads((tmp_path / "mpi" / simr.SUMMARY_FILE).read_text())
    assert [s["status"] for s in summaries] == ["ran"] * 6
