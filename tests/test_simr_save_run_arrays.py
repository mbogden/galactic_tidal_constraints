"""Tests for saving once-per-run arrays (MDRN-4 step 7b): run.save takes "ext.<key>" and the "drv.<name>"
of derived entries made only from externals. Each is written once to the output folder, named by its
data's hash, and the run's record says where."""

import pytest

np = pytest.importorskip("numpy")
jspam = pytest.importorskip("jspam")

from galactic_tidal_constraints.simr import derived, simr  # noqa: E402
from galactic_tidal_constraints.simr import simulation as sim  # noqa: E402
from galactic_tidal_constraints.simr.simulation import SimulatorConfig  # noqa: E402

# Model SDSS 587722984435351614 (as in test_simr.py).
PARAMS = np.array(
    [-9.93853, -4.5805, 15.43348, -1.54025, -2.97899, 4.69234, 34.41671, 29.49607, 2.97913, 4.2501,
     301.0, 35.5, 310.80745, 321.98757]
)  # fmt: skip
SMALL = {"n1": 20, "n2": 10}

# A stand-in target: the model's own path, every 50th row, in increasing time.
TARGET = sim.search_orbit(PARAMS, SimulatorConfig(**SMALL)).astype(np.float32)[::-1][::50].copy()
WINDOW = {"name": "win", "method": "recent_approach_cosine_window", "inputs": ["ext.target"]}
METRIC = {"method": "orbit_time_error", "inputs": ["ext.target", "sim.search_orbit", "drv.win"]}


def request(*saved, derived_entries=(WINDOW,), sim_metrics=(METRIC,)):
    return {"run": {"save": list(saved)}, "simulator": SMALL, "derived": list(derived_entries),
            "sim_metrics": list(sim_metrics)}


def test_an_external_is_saved_by_hash_and_the_record_says_where(tmp_path):
    result = simr.run(PARAMS, request("ext.target"), externals={"target": TARGET}, output=tmp_path)
    digest = result.externals["target"]["hash"]
    [record] = simr.load_catalog(tmp_path)["runs"]
    assert record["files"] == {"ext.target": f"externals/{digest}.npy"}
    saved = np.load(tmp_path / record["files"]["ext.target"])
    assert saved.dtype == np.float32
    np.testing.assert_array_equal(saved, TARGET)
    assert simr.array_hash(saved) == digest
    assert record["externals"]["target"]["source"] == "memory"  # the record still says where it came from
    assert not list(tmp_path.rglob("*.tmp"))


def test_a_derived_array_made_from_externals_is_saved_the_same_way(tmp_path):
    result = simr.run(PARAMS, request("drv.win", "ext.target"), externals={"target": TARGET}, output=tmp_path)
    digest = result.derived_configs[0]["hash"]
    [record] = simr.load_catalog(tmp_path)["runs"]
    assert record["files"]["drv.win"] == f"derived/{digest}.npy"
    np.testing.assert_array_equal(np.load(tmp_path / record["files"]["drv.win"]),
                                  derived.recent_approach_cosine_window(TARGET))
    assert sorted(path.parent.name for path in tmp_path.rglob("*.npy")) == ["derived", "externals"]


def test_the_same_data_is_stored_once_across_runs_and_keys(tmp_path):
    simr.run(PARAMS, request("ext.target"), externals={"target": TARGET}, output=tmp_path)
    [file] = (tmp_path / "externals").iterdir()
    written = file.stat().st_mtime_ns
    renamed = {"run": {"save": ["ext.same_data"]}, "simulator": SMALL,
               "sim_metrics": [{"method": "orbit_time_error", "inputs": ["ext.same_data", "sim.search_orbit"]}]}
    simr.run(PARAMS, renamed, externals={"same_data": TARGET.copy()}, output=tmp_path)
    assert [path.name for path in (tmp_path / "externals").iterdir()] == [file.name]
    assert file.stat().st_mtime_ns == written  # not written again
    first, second = simr.load_catalog(tmp_path)["runs"]
    assert first["files"]["ext.target"] == second["files"]["ext.same_data"]


def test_an_external_given_by_path_keeps_its_path_and_is_copied_only_when_asked(tmp_path):
    np.save(tmp_path / "target.npy", TARGET)
    externals = {"target": str(tmp_path / "target.npy")}
    simr.run(PARAMS, request(), externals=externals, output=tmp_path / "runs")
    [record] = simr.load_catalog(tmp_path / "runs")["runs"]
    assert record["externals"]["target"]["source"] == str(tmp_path / "target.npy")
    assert record["files"] == {} and not (tmp_path / "runs" / "externals").exists()
    simr.run(PARAMS, request("ext.target"), externals=externals, output=tmp_path / "runs")
    record = simr.load_catalog(tmp_path / "runs")["runs"][-1]
    assert record["externals"]["target"]["source"] == str(tmp_path / "target.npy")  # the path, with the copy
    assert (tmp_path / "runs" / record["files"]["ext.target"]).exists()


def test_saving_works_through_a_pool(tmp_path):
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=2) as pool:
        simr.run(PARAMS, request("ext.target", "drv.win"), externals={"target": TARGET}, output=tmp_path,
                 map=pool.map)
    [record] = simr.load_catalog(tmp_path)["runs"]
    np.testing.assert_array_equal(np.load(tmp_path / record["files"]["ext.target"]), TARGET)
    assert (tmp_path / record["files"]["drv.win"]).exists()


def test_a_run_that_saves_none_records_no_files(tmp_path):
    simr.run(PARAMS, request(), externals={"target": TARGET}, output=tmp_path / "runs.json")
    assert simr.load_catalog(tmp_path / "runs.json")["runs"][0]["files"] == {}


def test_bad_names_stop_the_run_before_anything_is_written(tmp_path):
    output = tmp_path / "runs"
    with pytest.raises(ValueError, match="run.save: ext.unused isn't an external a metric or derived entry uses"):
        simr.run(PARAMS, request("ext.unused"), externals={"target": TARGET, "unused": TARGET}, output=output)
    with pytest.raises(ValueError, match="run.save: ext.missing isn't an external"):
        simr.run(PARAMS, request("ext.missing"), externals={"target": TARGET}, output=output)
    with pytest.raises(ValueError, match=r"unknown names in run.save: \['ext.'\]"):
        simr.run(PARAMS, request("ext."), externals={"target": TARGET}, output=output)
    with pytest.raises(ValueError, match=r"unknown names in run.return: \['ext.target'\]"):
        simr.run(PARAMS, {**request(), "run": {"return": ["ext.target"]}}, externals={"target": TARGET})
    assert not output.exists()


@pytest.mark.parametrize("output", [None, "runs.json"])
def test_saving_an_external_needs_an_output_folder(tmp_path, output):
    with pytest.raises(ValueError, match="run.save names data to save, which needs an output folder"):
        simr.run(PARAMS, request("ext.target"), externals={"target": TARGET},
                 output=None if output is None else tmp_path / output)
