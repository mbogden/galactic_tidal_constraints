"""Tests for simr.load_run (MDRN-4 step 7d): a recorded run's models, full config and externals, rebuilt
from its catalog record so it can run again."""

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

# A stand-in target: the first model's own path, every 50th row, in increasing time.
TARGET = sim.search_orbit(PARAMS, SimulatorConfig(**SMALL)).astype(np.float32)[::-1][::50].copy()
INPUTS = ["ext.target", "sim.search_orbit", "drv.win"]


def request(*saved):
    return {
        "run": {"save": list(saved)},
        "simulator": SMALL,
        "derived": [{"name": "win", "method": "recent_approach_cosine_window", "inputs": ["ext.target"]}],
        "sim_metrics": [{"method": "orbit_time_error", "inputs": INPUTS},
                        {"method": "orbit_position_error", "inputs": INPUTS}],
        "composite": {"sim_weights": [[0.5, 0.5]]},
    }


def test_a_run_that_saved_its_external_runs_again_from_the_catalog_alone(tmp_path):
    first = simr.run(MODELS, request("ext.target"), externals={"target": TARGET}, output=tmp_path)
    loaded = simr.load_run(tmp_path)
    assert set(loaded) == {"models", "request", "externals"}
    np.testing.assert_array_equal(loaded["models"], MODELS)
    assert loaded["request"] == first.config
    digest = first.externals["target"]["hash"]
    assert loaded["externals"] == {"target": {"path": str((tmp_path / "externals" / f"{digest}.npy").absolute()),
                                              "hash": digest}}
    again = simr.run(**loaded, output=tmp_path / "again")  # its config saves, so it needs an output folder
    np.testing.assert_array_equal(again.composite, first.composite)
    assert again.config == first.config and again.externals["target"]["hash"] == digest


def test_models_come_back_in_the_runs_row_order(tmp_path):
    simr.run(MODELS, request(), externals={"target": TARGET}, output=tmp_path / "runs.json")
    simr.run(MODELS[::-1], request(), externals={"target": TARGET}, output=tmp_path / "runs.json")
    np.testing.assert_array_equal(simr.load_run(tmp_path / "runs.json", externals={"target": TARGET})["models"],
                                  MODELS[::-1])
    np.testing.assert_array_equal(
        simr.load_run(tmp_path / "runs.json", run=0, externals={"target": TARGET})["models"], MODELS)


def test_an_external_given_by_path_comes_back_as_its_path_with_its_hash(tmp_path):
    np.save(tmp_path / "target.npy", TARGET)
    first = simr.run(PARAMS, request(), externals={"target": str(tmp_path / "target.npy")}, output=tmp_path / "runs")
    loaded = simr.load_run(tmp_path / "runs")
    assert loaded["externals"] == {"target": {"path": str(tmp_path / "target.npy"),
                                              "hash": first.externals["target"]["hash"]}}
    assert simr.run(**loaded).composite.tolist() == first.composite.tolist()
    # If the file's data changes, the recorded hash stops the rerun
    np.save(tmp_path / "target.npy", TARGET[::2])
    with pytest.raises(ValueError, match="not the expected"):
        simr.run(**simr.load_run(tmp_path / "runs"))


def test_a_saved_copy_is_used_before_the_recorded_path(tmp_path):
    np.save(tmp_path / "target.npy", TARGET)
    simr.run(PARAMS, request("ext.target"), externals={"target": str(tmp_path / "target.npy")},
             output=tmp_path / "runs")
    (tmp_path / "target.npy").unlink()  # the original is gone; the output's copy is enough
    loaded = simr.load_run(tmp_path / "runs")
    assert "externals" in loaded["externals"]["target"]["path"].replace("\\", "/").split("/")
    assert simr.run(**loaded, output=tmp_path / "runs").status["sim_metrics"].tolist() == [[["ran", "ran"]]]


def test_an_in_memory_external_that_wasnt_saved_must_be_given(tmp_path):
    first = simr.run(PARAMS, request(), externals={"target": TARGET}, output=tmp_path)
    with pytest.raises(ValueError, match=r"given the externals \['target'\] in memory and didn't save them"):
        simr.load_run(tmp_path)
    loaded = simr.load_run(tmp_path, externals={"target": TARGET})
    assert loaded["externals"]["target"] is TARGET
    assert simr.run(**loaded).composite.tolist() == first.composite.tolist()


def test_externals_given_to_load_run_are_checked_against_the_record(tmp_path):
    simr.run(PARAMS, request(), externals={"target": TARGET}, output=tmp_path)
    with pytest.raises(ValueError, match="the data's hash is .* but run -1 of .* used data with the hash"):
        simr.load_run(tmp_path, externals={"target": TARGET[::2]})
    with pytest.raises(ValueError, match=r"externals \['other'\] weren't used by run -1"):
        simr.load_run(tmp_path, externals={"target": TARGET, "other": TARGET})
    # A path given here gets the recorded hash, so `run` checks the file
    np.save(tmp_path / "wrong.npy", TARGET[::2])
    loaded = simr.load_run(tmp_path, externals={"target": tmp_path / "wrong.npy"})
    assert loaded["externals"]["target"]["path"] == str(tmp_path / "wrong.npy")
    with pytest.raises(ValueError, match="not the expected"):
        simr.run(**loaded)


def test_a_run_without_externals_or_metrics_loads_too(tmp_path):
    first = simr.run(MODELS, {"run": {"save": ["sim.orbit"]}, "simulator": SMALL}, output=tmp_path)
    loaded = simr.load_run(tmp_path)
    assert loaded["externals"] == {} and loaded["request"] == first.config
    again = simr.run(**loaded, output=tmp_path)  # into the same output: everything is reused
    assert again.status["sim"].tolist() == [["reused"], ["reused"]]


def test_an_older_record_without_the_newer_keys_still_loads(tmp_path):
    import json

    simr.run(PARAMS, {"simulator": SMALL}, output=tmp_path)
    catalog = simr.load_catalog(tmp_path)
    for key in ("composite", "derived", "files", "externals", "values"):
        del catalog["runs"][0][key]  # as a run recorded before these existed
    (tmp_path / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    loaded = simr.load_run(tmp_path)
    assert "composite" not in loaded["request"] and loaded["externals"] == {}
    assert simr.run(**loaded).config["composite"] is None  # a missing section takes its default
