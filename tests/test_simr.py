"""Tests for the SIMR workflow (simr/simr.py). Each section covers one step of the rebuild."""

import json
import math
from dataclasses import asdict
from pathlib import Path

import pytest

# Skipped where the dependencies aren't installed, such as native Windows.
np = pytest.importorskip("numpy")
jspam = pytest.importorskip("jspam")

from galactic_tidal_constraints.simr import image_creator as img  # noqa: E402
from galactic_tidal_constraints.simr import simr  # noqa: E402
from galactic_tidal_constraints.simr import simulation as sim  # noqa: E402
from galactic_tidal_constraints.simr.image_creator import ImageConfig  # noqa: E402
from galactic_tidal_constraints.simr.simulation import SimulatorConfig  # noqa: E402
from galactic_tidal_constraints.simr.simr import SimrResult  # noqa: E402

# Model SDSS 587722984435351614 (as in test_simulation.py) and a second, heavier secondary.
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
        ({"run": {"return": ["sim.orbits"]}}, "unknown names in run.return"),
        ({"run": {"return": "sim.orbit"}}, "must be a list"),
        ({"run": {"seed": 3}}, "unknown keys in run"),
        ({"simulator": {"seed": 1.5}}, r"simulator\[0\]\.seed must be a whole number"),
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


# --- Step 8b: saving and loading ---------------------------------------------------------------------


def save_request(save, returned=(), simulator=SMALL):
    return {"run": {"return": list(returned), "save": list(save)}, "simulator": simulator}


def test_save_writes_one_folder_per_model(tmp_path):
    simr.run(MODELS, save_request(["sim.particles", "sim.orbit"]), output=tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["catalog.json", "model_000000", "model_000001"]
    folder = tmp_path / "model_000001"
    assert sorted(p.name for p in folder.iterdir()) == ["model.json", "simulations"]
    assert sorted(p.name for p in (folder / "simulations").iterdir()) == ["sim_00_orbit.npy", "sim_00_particles.npy"]


def test_saved_arrays_load_back_exactly_as_returned(tmp_path):
    names = ["sim.particles", "sim.unperturbed", "sim.orbit"]
    result = simr.run(MODELS, save_request(names, returned=names), output=tmp_path)
    loaded = simr.load_simulation(tmp_path / "model_000001", "sim_00")
    assert list(loaded) == names
    for name in names:
        assert loaded[name].dtype == np.float32
        np.testing.assert_array_equal(loaded[name], result.data[name][1, 0])


def test_saving_without_returning_keeps_nothing_in_memory(tmp_path):
    result = simr.run(MODELS, save_request(["sim.particles"]), output=tmp_path)
    assert result.data == {}
    assert simr.load_simulation(tmp_path / "model_000000", "sim_00")["sim.particles"].shape == (31, 6)


def test_model_json_records_model_config_timing_and_code(tmp_path):
    simr.run(MODELS, save_request(["sim.orbit"], simulator=[SMALL, {**SMALL, "seed": 4}]), output=tmp_path)
    record = simr.load_model(tmp_path / "model_000001")
    assert record["model"] == dict(zip(sim.PARAM_NAMES, HEAVIER.tolist()))
    assert list(record["simulations"]) == ["sim_00", "sim_01"]
    entry = record["simulations"]["sim_01"]
    assert entry["config"] == {**asdict(SimulatorConfig(**SMALL)), "seed": 4}
    assert entry["timing"]["t_start"] == sim.get_timing(HEAVIER, SimulatorConfig(**SMALL))["t_start"]
    assert entry["files"] == {"sim.orbit": "simulations/sim_01_orbit.npy"}
    assert set(entry["code"]) == {"galactic_tidal_constraints", "jspam"}


def test_jspam_version_is_the_pinned_commit():
    pin = Path(__file__).parents[1].joinpath("pyproject.toml").read_text().split("JSPAM.git@")[1][:40]
    assert sim.jspam_version()["commit"] == pin
    assert simr.code_versions()["jspam"] == {**sim.jspam_version(), "uncommitted_changes": False}


def test_timing_only_saves_no_array_file(tmp_path):
    simr.run(MODELS[:1], save_request(["sim.timing"]), output=tmp_path)
    entry = simr.load_model(tmp_path / "model_000000")["simulations"]["sim_00"]
    assert entry["files"] == {}
    assert "t_start" in entry["timing"]
    assert not (tmp_path / "model_000000" / "simulations").exists()


def test_no_temporary_files_are_left(tmp_path):
    simr.run(MODELS, save_request(["sim.particles"]), output=tmp_path)
    assert not list(tmp_path.rglob("*.tmp"))


def test_save_needs_an_output_folder():
    with pytest.raises(ValueError, match="needs an output folder"):
        simr.run(MODELS, save_request(["sim.orbit"]))


def test_a_saved_array_is_left_alone_and_the_run_moves_on(tmp_path):
    simr.run(MODELS[:1], save_request(["sim.particles"]), output=tmp_path)
    folder = tmp_path / "model_000000"
    before = simr.load_model(folder)
    particles = folder / "simulations" / "sim_00_particles.npy"
    file_time = particles.stat().st_mtime_ns
    # Again with a different sim_00 and a new sim_01: sim_00 stays as it was, sim_01 is added.
    changed = {**SMALL, "lnl": 0.01}
    result = simr.run(MODELS[:1], save_request(["sim.particles"], returned=["sim.particles"],
                                               simulator=[changed, changed]), output=tmp_path)
    after = simr.load_model(folder)
    assert after["simulations"]["sim_00"] == before["simulations"]["sim_00"]
    assert particles.stat().st_mtime_ns == file_time
    assert after["simulations"]["sim_01"]["config"]["lnl"] == 0.01
    assert result.data["sim.particles"][0, 0] is not None  # still computed, because it was returned


def test_a_saved_simulation_gains_missing_arrays_from_the_same_config(tmp_path):
    simr.run(MODELS[:1], save_request(["sim.particles"]), output=tmp_path)
    folder = tmp_path / "model_000000"
    particles = folder / "simulations" / "sim_00_particles.npy"
    file_time = particles.stat().st_mtime_ns
    simr.run(MODELS[:1], save_request(["sim.particles", "sim.orbit"]), output=tmp_path)
    entry = simr.load_model(folder)["simulations"]["sim_00"]
    assert entry["files"] == {"sim.particles": "simulations/sim_00_particles.npy",
                              "sim.orbit": "simulations/sim_00_orbit.npy"}
    assert particles.stat().st_mtime_ns == file_time  # the particles weren't rewritten
    np.testing.assert_array_equal(simr.load_simulation(folder, "sim_00")["sim.orbit"],
                                  sim.simulate_orbit(PARAMS, SimulatorConfig(**SMALL)).astype(np.float32))


def test_a_different_config_never_adds_to_a_saved_simulation(tmp_path):
    simr.run(MODELS[:1], save_request(["sim.particles"]), output=tmp_path)
    simr.run(MODELS[:1], save_request(["sim.orbit"], simulator={**SMALL, "lnl": 0.01}), output=tmp_path)
    entry = simr.load_model(tmp_path / "model_000000")["simulations"]["sim_00"]
    assert list(entry["files"]) == ["sim.particles"]
    assert not (tmp_path / "model_000000" / "simulations" / "sim_00_orbit.npy").exists()


def test_numpy_integers_in_a_config_save(tmp_path):
    simr.run(MODELS[:1], save_request(["sim.orbit"], simulator=SimulatorConfig(n1=np.int64(20), n2=10)),
             output=tmp_path)
    assert simr.load_model(tmp_path / "model_000000")["simulations"]["sim_00"]["config"]["n1"] == 20


def test_a_folder_holding_another_model_than_the_catalog_says_is_an_error(tmp_path):
    simr.run(MODELS[:1], save_request(["sim.orbit"]), output=tmp_path)
    catalog = simr.load_catalog(tmp_path)
    catalog["models"]["000000"] = simr.model_parameters(HEAVIER)  # the catalog and the folder now disagree
    (tmp_path / "catalog.json").write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="different model"):
        simr.run(MODELS[1:], save_request(["sim.orbit"]), output=tmp_path)


def test_only_listed_simulations_count_as_saved(tmp_path):
    simr.run(MODELS[:1], save_request(["sim.orbit"]), output=tmp_path)
    folder = tmp_path / "model_000000"
    (folder / "simulations" / "sim_05_orbit.npy").write_bytes(b"left by a crash")
    with pytest.raises(KeyError, match="sim_05"):
        simr.load_simulation(folder, "sim_05")


def test_bad_save_names_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown names in run.save"):
        simr.run(MODELS, save_request(["sim.orbits"]), output=tmp_path)


def _not_computed(*args):
    raise AssertionError("computed, but it was saved")


def test_saved_data_is_loaded_instead_of_computed(tmp_path, monkeypatch):
    first = simr.run(MODELS, save_request(["sim.particles", "sim.orbit"], returned=["sim.particles"]), output=tmp_path)
    for item in ("sim.particles", "sim.orbit", "sim.timing"):
        monkeypatch.setitem(simr.SIM_DATA, item, _not_computed)
    again = simr.run(MODELS, request(["sim.particles", "sim.orbit", "sim.timing"]), output=tmp_path)
    np.testing.assert_array_equal(again.data["sim.particles"][1, 0], first.data["sim.particles"][1, 0])
    assert again.data["sim.particles"][1, 0].dtype == np.float32
    assert again.data["sim.timing"][1, 0]["t_start"] == sim.get_timing(HEAVIER, SimulatorConfig(**SMALL))["t_start"]


def test_only_missing_data_is_computed(tmp_path, monkeypatch):
    simr.run(MODELS[:1], save_request(["sim.particles"]), output=tmp_path)
    monkeypatch.setitem(simr.SIM_DATA, "sim.particles", _not_computed)
    result = simr.run(MODELS[:1], request(["sim.particles", "sim.orbit"]), output=tmp_path)
    assert result.data["sim.orbit"][0, 0].shape[1] == 7  # computed: it wasn't saved


def test_a_different_config_is_computed_not_loaded(tmp_path):
    simr.run(MODELS[:1], save_request(["sim.particles"]), output=tmp_path)
    changed = {**SMALL, "seed": 1}
    result = simr.run(MODELS[:1], request(["sim.particles"], simulator=changed), output=tmp_path)
    expected = sim.simulate_particles(PARAMS, SimulatorConfig(**changed)).astype(np.float32)
    np.testing.assert_array_equal(result.data["sim.particles"][0, 0], expected)


def test_a_run_without_save_writes_only_the_catalog(tmp_path):
    simr.run(MODELS, request(["sim.orbit"]), output=tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["catalog.json"]
    [record] = simr.load_catalog(tmp_path)["runs"]
    assert record["run"] == {"return": ["sim.orbit"], "save": []}
    assert record["status"] == {"sim": [["ran"], ["ran"]], "img": [[["empty"]], [["empty"]]],  # no images asked for
                                "sim_metrics": [[[]], [[]]], "img_metrics": [[[[]]], [[[]]]]}  # no metric entries


def test_nothing_requested_is_empty(tmp_path):
    result = simr.run(MODELS, request([]), output=tmp_path)
    assert result.status["sim"].tolist() == [["empty"], ["empty"]]
    assert simr.load_catalog(tmp_path)["runs"][0]["status"] == {
        "sim": [["empty"], ["empty"]], "img": [[["empty"]], [["empty"]]],
        "sim_metrics": [[[]], [[]]], "img_metrics": [[[[]]], [[[]]]],
    }


def test_a_model_the_folder_doesnt_have_is_computed_and_added_to_the_catalog(tmp_path):
    simr.run(MODELS[:1], save_request(["sim.orbit"]), output=tmp_path)
    result = simr.run(MODELS, request(["sim.orbit"]), output=tmp_path)
    assert result.ids == ["000000", "000001"]
    assert result.status["sim"][:, 0].tolist() == ["reused", "ran"]
    assert not (tmp_path / "model_000001").exists()  # nothing to save, so no model folder


def test_catalog_lists_the_models_and_the_run(tmp_path):
    result = simr.run(MODELS, save_request(["sim.orbit"], simulator=[SMALL, {**SMALL, "lnl": 0.01}]), output=tmp_path)
    assert result.ids == ["000000", "000001"]
    catalog = simr.load_catalog(tmp_path)
    assert catalog["models"] == {"000000": simr.model_parameters(PARAMS), "000001": simr.model_parameters(HEAVIER)}
    [record] = catalog["runs"]
    assert record["started"] <= record["finished"]
    assert record["run"] == {"return": [], "save": ["sim.orbit"]}
    assert record["simulator"] == [asdict(SimulatorConfig(**SMALL)), asdict(SimulatorConfig(**SMALL, lnl=0.01))]
    assert record["image"] == [json.loads(json.dumps(asdict(ImageConfig())))]  # the default entry
    assert record["sim_metrics"] == [] and record["img_metrics"] == []
    assert "fitness" not in record  # replaced by the metric sections
    assert record["model_ids"] == ["000000", "000001"]
    assert record["code"] == simr.code_versions()


def test_ids_follow_the_model_not_its_row(tmp_path):
    third = PARAMS.copy()
    third[sim.PARAM_INDEX["vx"]] += 0.1
    simr.run(MODELS, save_request(["sim.orbit"]), output=tmp_path)
    # Reordered, with a new model in the middle: known models keep their ids, the new one goes at the bottom.
    result = simr.run(np.stack([HEAVIER, third, PARAMS]), save_request(["sim.orbit"]), output=tmp_path)
    assert result.ids == ["000001", "000002", "000000"]
    catalog = simr.load_catalog(tmp_path)
    assert list(catalog["models"]) == ["000000", "000001", "000002"]
    assert [run["model_ids"] for run in catalog["runs"]] == [["000000", "000001"], ["000001", "000002", "000000"]]
    assert simr.load_model(tmp_path / "model_000002")["model"] == simr.model_parameters(third)


def test_a_reordered_run_reuses_each_models_saved_data(tmp_path, monkeypatch):
    first = simr.run(MODELS, save_request(["sim.particles"], returned=["sim.particles"]), output=tmp_path)
    monkeypatch.setitem(simr.SIM_DATA, "sim.particles", _not_computed)
    again = simr.run(MODELS[::-1], request(["sim.particles"]), output=tmp_path)
    np.testing.assert_array_equal(again.data["sim.particles"][0, 0], first.data["sim.particles"][1, 0])


def test_the_same_model_twice_gets_one_id(tmp_path):
    result = simr.run(np.stack([PARAMS, PARAMS]), save_request(["sim.orbit"]), output=tmp_path)
    assert result.ids == ["000000", "000000"]
    assert list(simr.load_catalog(tmp_path)["models"]) == ["000000"]


def test_ids_are_none_without_an_output_folder():
    assert simr.run(MODELS, request([])).ids == [None, None]


def test_every_run_with_an_output_is_recorded(tmp_path):
    simr.run(MODELS[:1], save_request(["sim.orbit"]), output=tmp_path)
    simr.run(MODELS, request(["sim.orbit"]), output=tmp_path)
    catalog = simr.load_catalog(tmp_path)
    assert list(catalog["models"]) == ["000000", "000001"]
    assert [run["model_ids"] for run in catalog["runs"]] == [["000000"], ["000000", "000001"]]


def test_an_unfinished_run_shows_in_the_catalog(tmp_path, monkeypatch):
    def disk_full(*args):
        raise OSError("No space left on device")

    monkeypatch.setattr(simr, "_save_simulation", disk_full)  # an error while writing stops the run
    with pytest.raises(OSError):
        simr.run(MODELS, save_request(["sim.orbit"]), output=tmp_path)
    [record] = simr.load_catalog(tmp_path)["runs"]
    assert record["started"] is not None and record["finished"] is None



# --- Step 9: failures and status ---------------------------------------------------------------------

# Approaching now and unbound: it has no pass before now, so no_pass="error" makes the simulation fail.
FIRST_APPROACH = PARAMS.copy()
FIRST_APPROACH[3:6] *= -3
FAILING = {**SMALL, "no_pass": "error"}


def nan_for_heavier(params, config):
    orbit = simr.SIM_DATA_BEFORE_TEST["sim.orbit"](params, config)
    if params[sim.PARAM_INDEX["mass2"]] == HEAVIER[sim.PARAM_INDEX["mass2"]]:
        orbit[5, 0] = np.nan
    return orbit


def test_a_failure_doesnt_stop_the_run():
    models = np.stack([PARAMS, FIRST_APPROACH, HEAVIER])
    result = simr.run(models, request(["sim.orbit"], simulator=FAILING))
    assert result.status["sim"][:, 0].tolist() == ["ran", "failed", "ran"]
    assert result.data["sim.orbit"][1, 0] is None
    assert result.data["sim.orbit"][2, 0].shape[1] == 7
    assert result.errors["sim"][1, 0].startswith("ValueError: ")
    assert result.errors["sim"][0, 0] is None


def test_nan_or_inf_in_an_array_is_a_failure(monkeypatch):
    monkeypatch.setattr(simr, "SIM_DATA_BEFORE_TEST", dict(simr.SIM_DATA), raising=False)
    monkeypatch.setitem(simr.SIM_DATA, "sim.orbit", nan_for_heavier)
    result = simr.run(MODELS, request(["sim.orbit", "sim.timing"]))
    assert result.status["sim"][:, 0].tolist() == ["ran", "failed"]
    assert result.errors["sim"][1, 0] == "ValueError: NaN or inf in sim.orbit"
    assert result.data["sim.timing"][1, 0] is None  # the whole simulation fails, not just the orbit


def test_nan_in_timing_isnt_a_failure():
    result = simr.run(MODELS[:1], request(["sim.timing"], simulator={**SMALL, "start": -5.0}))
    assert np.isnan(result.data["sim.timing"][0, 0]["closest_distance"])  # a fixed start has no pass
    assert result.status["sim"][0, 0] == "ran"


def test_a_failure_is_not_saved_and_is_retried(tmp_path):
    models = np.stack([PARAMS, FIRST_APPROACH])
    simr.run(models, save_request(["sim.orbit"], simulator=FAILING), output=tmp_path)
    assert simr.load_model(tmp_path / "model_000000")["simulations"]["sim_00"]["files"]
    assert not (tmp_path / "model_000001" / "model.json").exists()
    again = simr.run(models, save_request(["sim.orbit"], simulator=FAILING), output=tmp_path)
    assert again.status["sim"][:, 0].tolist() == ["reused", "failed"]


def test_reused_only_when_everything_came_from_disk(tmp_path):
    simr.run(MODELS[:1], save_request(["sim.particles"]), output=tmp_path)
    loaded = simr.run(MODELS[:1], request(["sim.particles"]), output=tmp_path)
    partly = simr.run(MODELS[:1], request(["sim.particles", "sim.orbit"]), output=tmp_path)
    assert loaded.status["sim"][0, 0] == "reused"
    assert partly.status["sim"][0, 0] == "ran"


def test_status_seconds_and_errors_have_the_datas_shape():
    result = simr.run(MODELS, request(["sim.orbit"], simulator=[SMALL, SMALL]))
    for field in (result.status, result.seconds, result.errors):
        assert list(field) == ["sim", "img", "sim_metrics", "img_metrics"] and field["sim"].shape == (2, 2)
        assert field["img"].shape == (2, 2, 1)  # one default image entry
        assert field["sim_metrics"].shape == (2, 2, 0)  # no metric entries
        assert field["img_metrics"].shape == (2, 2, 1, 0)
    assert (result.seconds["sim"] > 0).all()


def test_the_catalog_records_each_simulations_status(tmp_path):
    simr.run(np.stack([PARAMS, FIRST_APPROACH]), save_request(["sim.orbit"], simulator=FAILING), output=tmp_path)
    [record] = simr.load_catalog(tmp_path)["runs"]
    assert record["status"] == {"sim": [["ran"], ["failed"]], "img": [[["empty"]], [["empty"]]],
                                "sim_metrics": [[[]], [[]]], "img_metrics": [[[[]]], [[[]]]]}
    assert record["errors"]["sim"][0] == [None] and record["errors"]["sim"][1][0].startswith("ValueError")
    assert len(record["seconds"]["sim"]) == 2 and record["finished"] is not None


def test_failures_are_logged_with_a_count(caplog):
    with caplog.at_level("INFO", logger="galactic_tidal_constraints.simr.simr"):
        simr.run(np.stack([PARAMS, FIRST_APPROACH]), request(["sim.orbit"], simulator=FAILING))
    failure = next(r for r in caplog.records if r.levelname == "ERROR")
    assert "model 1" in failure.getMessage() and failure.exc_info is not None
    assert "2 simulations: 1 ran, 0 reused, 1 failed, 0 empty" in caplog.text



# --- Step 9b: every output gets a catalog --------------------------------------------------------------


def test_a_json_output_is_a_catalog_file_alone(tmp_path):
    catalog_file = tmp_path / "gen_07.json"
    result = simr.run(MODELS, request(["sim.orbit"]), output=catalog_file)
    assert [p.name for p in tmp_path.iterdir()] == ["gen_07.json"]
    assert result.ids == ["000000", "000001"]
    catalog = simr.load_catalog(catalog_file)
    assert list(catalog["models"]) == ["000000", "000001"]
    assert catalog["runs"][0]["status"]["sim"] == [["ran"], ["ran"]]


def test_a_json_output_collects_runs_like_a_folder(tmp_path):
    catalog_file = tmp_path / "runs.json"
    simr.run(MODELS, request([]), output=catalog_file)
    again = simr.run(MODELS[::-1], request([]), output=catalog_file)
    assert again.ids == ["000001", "000000"]
    assert len(simr.load_catalog(catalog_file)["runs"]) == 2


def test_saving_needs_a_folder_not_a_catalog_file(tmp_path):
    with pytest.raises(ValueError, match="is a catalog file, not a folder"):
        simr.run(MODELS, save_request(["sim.orbit"]), output=tmp_path / "gen_07.json")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ["runs.txt", "runs.v2", "catalog.JSON.bak"])
def test_other_extensions_are_errors(tmp_path, name):
    with pytest.raises(ValueError, match="a folder .no extension. or a .json catalog file"):
        simr.run(MODELS, request([]), output=tmp_path / name)
    assert list(tmp_path.iterdir()) == []


def test_output_paths():
    assert simr.output_paths("runs") == (Path("runs/catalog.json"), Path("runs"))
    assert simr.output_paths("runs/gen.json") == (Path("runs/gen.json"), None)
    assert simr.output_paths(".hidden") == (Path(".hidden/catalog.json"), Path(".hidden"))



# --- Step 11: reading models, and checking types -----------------------------------------------------------


def test_read_models_takes_the_first_14_values(tmp_path, caplog):
    path = tmp_path / "models.txt"
    archive_line = ",".join(str(v) for v in HEAVIER) + ",0.3,0.3,0,0,0,0,0,0,1.0,0,0 -6.9 24.9 SDSS_587722984435351614"
    path.write_text("# a comment\n" + ",".join(str(v) for v in PARAMS) + "\n\n" + archive_line + "\n")
    with caplog.at_level("INFO", logger="galactic_tidal_constraints.simr.simr"):
        models = simr.read_models(path)
    np.testing.assert_array_equal(models, MODELS)
    assert "1 of 2 lines had more than 14 values" in caplog.text


def test_read_models_reads_jspams_archive_line_format(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text(
        "-9.93853,-4.5805,15.43348,-1.54025,-2.97899,4.69234,34.41671,29.49607,2.97913,4.2501,301.0,35.5,"
        "310.80745,321.98757,0.3,0.3,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0,0,0.24008,-2.41999,5.62715,0.0,1.0,0.0,1.0,"
        "0.0,0.0 -6.904942981293307 24.945037352311584 SDSS_587722984435351614\n"
    )
    np.testing.assert_array_equal(simr.read_models(path)[0], PARAMS)


@pytest.mark.parametrize(
    "text, message",
    [
        ("1,2,3\n", "line 1: expected at least 14"),
        ("# only a comment\n\n", "has no models"),
        ("a," + "1," * 13 + "\n", "line 1: the first 14 values must be numbers"),
    ],
)
def test_bad_models_files_name_the_line(tmp_path, text, message):
    path = tmp_path / "models.txt"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        simr.read_models(path)


@pytest.mark.parametrize(
    "entry, message",
    [
        ({"lnl": "0.1"}, r"simulator\[0\]\.lnl must be a number: '0.1'"),
        ({"n1": 10.0}, r"simulator\[0\]\.n1 must be a whole number"),
        ({"n1": True}, r"simulator\[0\]\.n1 must be a whole number"),
        ({"heat_mode": 1}, r"simulator\[0\]\.heat_mode must be text"),
        ({"start": [1]}, r"simulator\[0\]\.start must be text or a number"),
    ],
)
def test_wrong_types_name_the_field(entry, message):
    with pytest.raises(ValueError, match=message):
        simr.run(MODELS, {"simulator": entry})


def test_json_style_numbers_are_accepted():
    # JSON writes 1.0 as 1, so an int is fine for a float field; a number is fine for start.
    result = simr.run(MODELS[:1], request([], simulator={**SMALL, "lnl": 1, "start": -6, "seed": np.int64(2)}))
    assert result.simulators[0].lnl == 1 and result.simulators[0].start == -6


# --- Step 12a: one task per model, through a map function ----------------------------------------------


def counting_map(calls):
    """A map that records each task it's given, then runs it like built-in map."""

    def run_all(function, tasks):
        tasks = list(tasks)
        calls.extend(tasks)
        return [function(task) for task in tasks]

    return run_all


def test_each_model_is_one_task():
    calls = []
    simr.run(MODELS, request(["sim.orbit"], simulator=[SMALL, {**SMALL, "lnl": 0.01}]), map=counting_map(calls))
    assert [task.row for task in calls] == [0, 1]
    assert all(len(task.configs) == 2 for task in calls)  # a task holds every simulator entry of its model


def test_another_map_gives_the_same_results():
    plain = simr.run(MODELS, request(["sim.particles", "sim.timing"]))
    mapped = simr.run(MODELS, request(["sim.particles", "sim.timing"]), map=counting_map([]))
    for m in range(len(MODELS)):
        np.testing.assert_array_equal(mapped.data["sim.particles"][m, 0], plain.data["sim.particles"][m, 0])
    assert mapped.status["sim"].tolist() == plain.status["sim"].tolist()


def test_the_same_model_twice_is_one_task(tmp_path):
    calls = []
    twice = np.stack([PARAMS, HEAVIER, PARAMS])
    result = simr.run(twice, save_request(["sim.orbit"], returned=["sim.orbit"]), output=tmp_path,
                      map=counting_map(calls))
    assert [task.model_id for task in calls] == ["000000", "000001"]
    assert result.ids == ["000000", "000001", "000000"]
    np.testing.assert_array_equal(result.data["sim.orbit"][2, 0], result.data["sim.orbit"][0, 0])
    assert result.status["sim"][:, 0].tolist() == ["ran", "ran", "ran"]


def test_tasks_and_the_worker_can_be_sent_to_other_processes(tmp_path):
    import pickle

    calls = []
    simr.run(MODELS, save_request(["sim.orbit"]), output=tmp_path, map=counting_map(calls))
    task = pickle.loads(pickle.dumps(calls[0]))
    assert task.model_id == "000000" and task.folder == tmp_path / "model_000000"
    assert pickle.loads(pickle.dumps(simr._run_model)) is simr._run_model


# --- Step 12b: local pools, catalog writes while running, a worker that dies ---------------------------


def test_a_local_pool_gives_the_same_results_as_one_process(tmp_path):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    req = save_request(["sim.particles"], returned=["sim.particles", "sim.timing"], simulator=[SMALL, {**SMALL, "seed": 1}])
    solo = simr.run(MODELS, req, output=tmp_path / "solo")
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        pooled = simr.run(MODELS, req, output=tmp_path / "pool", map=pool.map)
    for m in range(len(MODELS)):
        for s in range(2):
            np.testing.assert_array_equal(pooled.data["sim.particles"][m, s], solo.data["sim.particles"][m, s])
            np.testing.assert_array_equal(
                simr.load_simulation(tmp_path / "pool" / f"model_{pooled.ids[m]}", f"sim_{s:02d}")["sim.particles"],
                solo.data["sim.particles"][m, s])
    assert pooled.status["sim"].tolist() == solo.status["sim"].tolist()
    assert simr.load_catalog(tmp_path / "pool")["runs"][0]["finished"] is not None


def test_statuses_are_written_while_results_arrive(tmp_path, monkeypatch):
    monkeypatch.setattr(simr, "CATALOG_WRITE_SECONDS", 0.0)  # write after every model
    seen = []

    def peeking_map(function, tasks):
        for task in tasks:
            yield function(task)
            seen.append(simr.load_catalog(tmp_path)["runs"][0]["status"]["sim"])

    simr.run(MODELS, request(["sim.timing"]), output=tmp_path, map=peeking_map)
    # The catalog was written after model 0 (seen once model 1 is handed out), with model 1 not yet run.
    assert seen[0] == [["ran"], [None]]


def test_a_broken_pool_records_what_finished_and_raises(tmp_path, caplog):
    from concurrent.futures.process import BrokenProcessPool

    def dying_map(function, tasks):
        tasks = iter(tasks)
        yield function(next(tasks))
        raise BrokenProcessPool("a worker died")

    with pytest.raises(simr.SimrRunError, match="1 models unfinished"):
        simr.run(MODELS, save_request(["sim.orbit"]), output=tmp_path, map=dying_map)
    assert "1 of 2 models unfinished: ['000001']" in caplog.text
    [record] = simr.load_catalog(tmp_path)["runs"]
    assert record["finished"] is None
    assert record["status"] == {"sim": [["ran"], [None]], "img": [[["empty"]], [[None]]],
                                "sim_metrics": [[[]], [[]]], "img_metrics": [[[[]]], [[[]]]]}
    again = simr.run(MODELS, save_request(["sim.orbit"]), output=tmp_path)  # a rerun resumes
    assert again.status["sim"][:, 0].tolist() == ["reused", "ran"]


def _exit_hard(params, config):
    import os

    os._exit(1)  # what a segfault in SPAM's Fortran does to a worker


def test_a_worker_that_really_dies(tmp_path, monkeypatch):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    monkeypatch.setitem(simr.SIM_DATA, "sim.orbit", _exit_hard)  # "fork" workers inherit this
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("fork")) as pool:
        with pytest.raises(simr.SimrRunError, match="a worker died"):
            simr.run(MODELS, save_request(["sim.orbit"]), output=tmp_path, map=pool.map)
    assert simr.load_catalog(tmp_path)["runs"][0]["finished"] is None


def test_available_cpus():
    assert simr.available_cpus() >= 1


# --- Step 12c: the MPI rank in log lines ----------------------------------------------------------------
@pytest.mark.parametrize("variable", simr.MPI_RANK_VARIABLES)
def test_log_lines_name_the_mpi_rank(monkeypatch, variable):
    for name in simr.MPI_RANK_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(variable, "3")
    seen = {}
    monkeypatch.setattr(simr.logging, "basicConfig", lambda **kwargs: seen.update(kwargs))
    simr.configure_logging("DEBUG")
    assert seen == {"level": "DEBUG", "format": simr.LOG_FORMAT.replace("%(processName)s", "rank 3")}


def test_log_lines_name_the_process_without_mpi(monkeypatch):
    for name in simr.MPI_RANK_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    seen = {}
    monkeypatch.setattr(simr.logging, "basicConfig", lambda **kwargs: seen.update(kwargs))
    simr.configure_logging()
    assert seen["format"] == simr.LOG_FORMAT


# --- After Step 12d: a local pool under mpirun names its processes -----------------------------------
def test_log_lines_can_ignore_the_mpi_rank(monkeypatch):
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")  # as mpirun sets it for a single process
    seen = {}
    monkeypatch.setattr(simr.logging, "basicConfig", lambda **kwargs: seen.update(kwargs))
    simr.configure_logging("INFO", use_rank=False)
    assert seen["format"] == simr.LOG_FORMAT


# --- MDRN-3 step 2: images from simr.run, in memory -------------------------------------------------------


def test_images_are_indexed_by_model_simulator_and_image():
    images = [{"blur_sigma": 1.0}, {"width": 64, "height": 32, "brightness": "log"}]
    result = simr.run(MODELS, {**request(["img.model", "img.unperturbed"]), "image": images})
    assert result.images == [ImageConfig(blur_sigma=1.0), ImageConfig(width=64, height=32, brightness="log")]
    for name in ("img.model", "img.unperturbed"):
        assert result.data[name].shape == (2, 1, 2)
        for m in range(2):
            assert result.data[name][m, 0, 0].shape == (128, 128)
            assert result.data[name][m, 0, 1].shape == (32, 64)
            assert result.data[name][m, 0, 1].dtype == np.float32
    assert result.status["img"].tolist() == [[["ran", "ran"]], [["ran", "ran"]]]
    assert (result.seconds["img"] > 0).all()


def test_an_image_is_create_image_on_the_simulations_particles():
    config = ImageConfig(primary_center=(32, 64), secondary_center=(96, 64), blur_sigma=0.5)
    result = simr.run(MODELS, {**request(["sim.particles", "sim.unperturbed", "img.model", "img.unperturbed"]),
                               "image": config})
    for m, params in enumerate(MODELS):
        for sim_item, image_item in (("sim.particles", "img.model"), ("sim.unperturbed", "img.unperturbed")):
            expected = img.create_image(result.data[sim_item][m, 0], params, SMALL["n1"], SMALL["n2"], config)
            np.testing.assert_array_equal(result.data[image_item][m, 0, 0], expected)


def test_images_compute_the_particles_they_need_without_returning_them():
    result = simr.run(MODELS, request(["img.model"]))
    assert list(result.data) == ["img.model"]
    assert result.status["sim"].tolist() == [["ran"], ["ran"]]  # the particles were computed for the images
    assert result.images == [ImageConfig()]  # no image section: one default entry


def test_image_entries_take_centers_as_lists():
    result = simr.run(PARAMS, {**request(["img.model"]), "image": {"primary_center": [10, 20],
                                                                      "secondary_center": [30.5, 20]}})
    assert result.images[0].primary_center == (10.0, 20.0) and result.images[0].secondary_center == (30.5, 20.0)


@pytest.mark.parametrize(
    ("image", "message"),
    [
        ({"zoom": 2}, r"unknown keys in image\[0\]"),
        ({"primary_center": "middle"}, r"image\[0\].primary_center must be a list of 2"),
        ({"primary_center": [1, 2, 3]}, r"image\[0\].primary_center must be a list of 2"),
        ([{}, {"primary_center": [5, 5], "secondary_center": [5, 5]}], r"image\[1\]: .*can't be equal"),
        ([], "image is an empty list"),
    ],
)
def test_bad_image_entries_are_errors(image, message):
    with pytest.raises(ValueError, match=message):
        simr.run(PARAMS, {**request(["img.model"]), "image": image})


def test_a_failed_simulation_skips_its_images():
    result = simr.run(np.stack([PARAMS, FIRST_APPROACH]), {**request(["img.model"], simulator=FAILING),
                                                         "image": [{}, {}]})
    assert result.status["sim"][:, 0].tolist() == ["ran", "failed"]
    assert result.status["img"].tolist() == [[["ran", "ran"]], [["skipped", "skipped"]]]
    assert result.data["img.model"][1, 0, 0] is None


def test_a_failed_image_is_recorded_and_the_run_moves_on(monkeypatch, tmp_path, caplog):
    caplog.set_level("INFO", logger="galactic_tidal_constraints.simr.simr")
    def broken(values, params, sim_config, image_config):
        if image_config.blur_sigma == 1.0:
            raise ValueError("no frame")
        return img.create_image(values["sim.particles"], params, sim_config.n1, sim_config.n2, image_config)

    monkeypatch.setitem(simr.IMAGE_DATA, "img.model", (("sim.particles",), broken))
    result = simr.run(PARAMS, {**request(["img.model"]), "image": [{}, {"blur_sigma": 1.0}]}, output=tmp_path)
    assert result.status["sim"].tolist() == [["ran"]]
    assert result.status["img"].tolist() == [[["ran", "failed"]]]
    assert result.errors["img"][0, 0, 1] == "ValueError: no frame"
    assert result.data["img.model"][0, 0, 1] is None and result.data["img.model"][0, 0, 0] is not None
    assert "2 images: 1 ran, 0 reused, 1 failed, 0 skipped" in caplog.text
    [record] = simr.load_catalog(tmp_path)["runs"]
    assert record["status"]["img"] == [[["ran", "failed"]]]
    assert record["errors"]["img"][0][0][1] == "ValueError: no frame"


def test_images_use_saved_particles(tmp_path):
    simr.run(MODELS, save_request(["sim.particles"]), output=tmp_path)
    result = simr.run(MODELS, {**request(["img.model"]), "image": {"blur_sigma": 1.0}}, output=tmp_path)
    assert result.status["sim"].tolist() == [["reused"], ["reused"]]
    saved = simr.load_simulation(tmp_path / "model_000000", "sim_00")["sim.particles"]
    expected = img.create_image(saved, PARAMS, SMALL["n1"], SMALL["n2"], ImageConfig(blur_sigma=1.0))
    np.testing.assert_array_equal(result.data["img.model"][0, 0, 0], expected)


def test_the_catalog_records_the_image_entries(tmp_path):
    simr.run(MODELS, {**request(["img.model"]), "image": [{}, {"brightness": "linear"}]}, output=tmp_path)
    [record] = simr.load_catalog(tmp_path)["runs"]
    assert [entry["brightness"] for entry in record["image"]] == ["log", "linear"]
    assert record["image"][0]["primary_center"] == [32.0, 64.0]
    assert record["status"]["img"] == [[["ran", "ran"]], [["ran", "ran"]]]


def test_images_come_back_the_same_through_a_pool():
    from concurrent.futures import ProcessPoolExecutor

    req = {**request(["img.model"]), "image": [{}, {"blur_sigma": 1.0}]}
    solo = simr.run(MODELS, req)
    with ProcessPoolExecutor(2) as pool:
        pooled = simr.run(MODELS, req, map=pool.map)
    for index in np.ndindex(solo.data["img.model"].shape):
        np.testing.assert_array_equal(pooled.data["img.model"][index], solo.data["img.model"][index])
    assert pooled.status["img"].tolist() == solo.status["img"].tolist()


# --- MDRN-3 step 3: saving and loading images -------------------------------------------------------------

TWO_IMAGES = [{"primary_center": [32, 64], "secondary_center": [96, 64]},
              {"primary_center": [32, 64], "secondary_center": [96, 64], "blur_sigma": 0.75}]


def image_request(save, returned=(), simulator=SMALL, image=TWO_IMAGES):
    return {"run": {"save": list(save), "return": list(returned)}, "simulator": simulator, "image": image}


def test_names_and_files():
    assert simr.simulation_name(3) == "sim_03" and simr.image_name(12) == "img_0012"
    assert simr.image_file("sim_00", "img_0001", "img.model") == "images/sim_00_img_0001_model.npy"


def test_saved_images_layout_and_model_json(tmp_path):
    save = ["sim.particles", "img.model", "img.unperturbed"]
    simr.run(PARAMS, image_request(save, simulator=[SMALL, {**SMALL, "lnl": 0.01}]), output=tmp_path)
    folder = tmp_path / "model_000000"
    assert sorted(p.name for p in (folder / "images").iterdir()) == [
        f"{s}_{i}_{item}.npy" for s in ("sim_00", "sim_01") for i in ("img_0000", "img_0001")
        for item in ("model", "unperturbed")
    ]
    record = simr.load_model(folder)
    assert list(record["images"]) == ["img_0000", "img_0001"]
    entry = record["images"]["img_0001"]
    assert entry["config"]["blur_sigma"] == 0.75 and entry["config"]["primary_center"] == [32.0, 64.0]
    assert entry["files"] == {
        s: {"img.model": f"images/{s}_img_0001_model.npy", "img.unperturbed": f"images/{s}_img_0001_unperturbed.npy"}
        for s in ("sim_00", "sim_01")
    }
    assert record["simulations"]["sim_00"]["files"] == {"sim.particles": "simulations/sim_00_particles.npy"}


def test_saved_images_are_what_run_returns(tmp_path):
    result = simr.run(MODELS, image_request(["img.model"], returned=["img.model"]), output=tmp_path)
    for m, model_id in enumerate(result.ids):
        loaded = simr.load_images(tmp_path / f"model_{model_id}")
        assert list(loaded) == [("sim_00", "img_0000", "img.model"), ("sim_00", "img_0001", "img.model")]
        for i in range(2):
            np.testing.assert_array_equal(loaded["sim_00", simr.image_name(i), "img.model"],
                                          result.data["img.model"][m, 0, i])
            assert loaded["sim_00", simr.image_name(i), "img.model"].dtype == np.float32


def test_saving_only_images_records_the_simulation(tmp_path):
    simr.run(PARAMS, image_request(["img.model"]), output=tmp_path)
    record = simr.load_model(tmp_path / "model_000000")
    assert not (tmp_path / "model_000000" / "simulations").exists()
    assert record["simulations"]["sim_00"]["files"] == {}
    assert record["simulations"]["sim_00"]["config"] == asdict(SimulatorConfig(**SMALL))
    assert record["simulations"]["sim_00"]["timing"]["n_steps"] > 0


def test_saved_images_are_reused_without_simulating(tmp_path, monkeypatch):
    simr.run(MODELS, image_request(["img.model"]), output=tmp_path)

    def no_simulation(*args, **kwargs):
        raise AssertionError("simulated")

    monkeypatch.setitem(simr.SIM_DATA, "sim.particles", no_simulation)
    again = simr.run(MODELS, image_request(["img.model"], returned=["img.model"]), output=tmp_path)
    assert again.status["sim"].tolist() == [["reused"], ["reused"]]
    assert again.status["img"].tolist() == [[["reused", "reused"]], [["reused", "reused"]]]
    saved = simr.load_images(tmp_path / "model_000000", image="img_0001")
    np.testing.assert_array_equal(again.data["img.model"][0, 0, 1], saved["sim_00", "img_0001", "img.model"])


def test_a_missing_item_is_made_and_added(tmp_path):
    simr.run(PARAMS, image_request(["img.model"]), output=tmp_path)
    again = simr.run(PARAMS, image_request(["img.model", "img.unperturbed"]), output=tmp_path)
    assert again.status["sim"].tolist() == [["ran"]]  # the unperturbed particles had to be made
    assert again.status["img"].tolist() == [[["ran", "ran"]]]  # each image entry made its unperturbed image
    files = simr.load_model(tmp_path / "model_000000")["images"]["img_0000"]["files"]["sim_00"]
    assert sorted(files) == ["img.model", "img.unperturbed"]


def test_a_changed_image_config_is_neither_used_nor_changed(tmp_path, caplog):
    simr.run(PARAMS, image_request(["img.model"]), output=tmp_path)
    before = simr.load_images(tmp_path / "model_000000")
    changed = [TWO_IMAGES[0], {**TWO_IMAGES[1], "blur_sigma": 1.5}]
    again = simr.run(PARAMS, image_request(["img.model"], returned=["img.model"], image=changed), output=tmp_path)
    assert again.status["img"].tolist() == [[["reused", "ran"]]]
    assert "img_0001 was saved with a different image config" in caplog.text
    record = simr.load_model(tmp_path / "model_000000")
    assert record["images"]["img_0001"]["config"]["blur_sigma"] == 0.75  # unchanged
    after = simr.load_images(tmp_path / "model_000000")
    for key in before:
        np.testing.assert_array_equal(after[key], before[key])
    assert not np.array_equal(again.data["img.model"][0, 0, 1], before["sim_00", "img_0001", "img.model"])


def test_a_changed_simulation_config_leaves_its_images_alone(tmp_path):
    simr.run(PARAMS, image_request(["img.model"]), output=tmp_path)
    before = simr.load_model(tmp_path / "model_000000")
    again = simr.run(PARAMS, image_request(["img.model"], simulator={**SMALL, "lnl": 0.01}), output=tmp_path)
    assert again.status["sim"].tolist() == [["ran"]] and again.status["img"].tolist() == [[["ran", "ran"]]]
    assert simr.load_model(tmp_path / "model_000000") == before  # nothing used, nothing changed


def test_a_failed_image_saves_nothing_and_is_retried(tmp_path, monkeypatch):
    def broken(values, params, sim_config, image_config):
        raise ValueError("no frame")

    with monkeypatch.context() as patch:
        patch.setitem(simr.IMAGE_DATA, "img.model", (("sim.particles",), broken))
        failed = simr.run(PARAMS, image_request(["img.model"]), output=tmp_path)
    assert failed.status["img"].tolist() == [[["failed", "failed"]]]
    assert simr.load_images(tmp_path / "model_000000") == {}
    retried = simr.run(PARAMS, image_request(["img.model"]), output=tmp_path)
    assert retried.status["img"].tolist() == [[["ran", "ran"]]]
    assert len(simr.load_images(tmp_path / "model_000000")) == 2


def test_a_failed_simulation_saves_no_images(tmp_path):
    result = simr.run(np.stack([PARAMS, FIRST_APPROACH]), image_request(["img.model"], simulator=FAILING),
                      output=tmp_path)
    assert result.status["img"][1].tolist() == [["skipped", "skipped"]]
    assert not (tmp_path / "model_000001").exists()


def test_load_images_filters(tmp_path):
    save = ["img.model", "img.unperturbed"]
    simr.run(PARAMS, image_request(save, simulator=[SMALL, {**SMALL, "lnl": 0.01}]), output=tmp_path)
    folder = tmp_path / "model_000000"
    assert len(simr.load_images(folder)) == 8
    assert {key[0] for key in simr.load_images(folder, simulation="sim_01")} == {"sim_01"}
    assert len(simr.load_images(folder, simulation="sim_01")) == 4
    assert {key[1] for key in simr.load_images(folder, image="img_0000")} == {"img_0000"}
    assert list(simr.load_images(folder, "sim_00", "img_0001")) == [
        ("sim_00", "img_0001", "img.model"), ("sim_00", "img_0001", "img.unperturbed")]
    models_only = simr.load_images(folder, items="img.model")
    assert len(models_only) == 4 and {key[2] for key in models_only} == {"img.model"}
    assert len(simr.load_images(folder, items=["img.model", "img.unperturbed"])) == 8
    with pytest.raises(KeyError, match="sim_05"):
        simr.load_images(folder, simulation="sim_05")
    with pytest.raises(KeyError, match="img_0007"):
        simr.load_images(folder, image="img_0007")
    with pytest.raises(ValueError, match="unknown image items"):
        simr.load_images(folder, items="img.galaxies")


def test_load_images_returns_nothing_when_none_match(tmp_path):
    simr.run(PARAMS, image_request(["sim.particles"]), output=tmp_path)  # no images saved
    assert simr.load_images(tmp_path / "model_000000") == {}


def test_a_stopped_run_resumes_its_images(tmp_path):
    from concurrent.futures.process import BrokenProcessPool

    def dying_map(function, tasks):
        tasks = iter(tasks)
        yield function(next(tasks))
        raise BrokenProcessPool("a worker died")

    with pytest.raises(simr.SimrRunError):
        simr.run(MODELS, image_request(["img.model"]), output=tmp_path, map=dying_map)
    again = simr.run(MODELS, image_request(["img.model"]), output=tmp_path)
    assert again.status["img"].tolist() == [[["reused", "reused"]], [["ran", "ran"]]]


# --- MDRN-4 step 1: sim.search_orbit as an item -------------------------------------------------------


def test_search_orbit_matches_calling_sim_directly():
    result = simr.run(MODELS, request(["sim.search_orbit"]))
    assert result.data["sim.search_orbit"].shape == (2, 1)
    expected = sim.search_orbit(HEAVIER, SimulatorConfig(**SMALL)).astype(np.float32)
    np.testing.assert_array_equal(result.data["sim.search_orbit"][1, 0], expected)
    assert result.data["sim.search_orbit"][0, 0].dtype == np.float32


def test_search_orbit_spans_now_back_to_minus_30():
    path = simr.run(PARAMS, request(["sim.search_orbit"])).data["sim.search_orbit"][0, 0]
    assert path.shape[1] == 7
    assert path[0, 6] == 0.0  # row 0 is now
    np.testing.assert_allclose(path[-1, 6], -30.0, atol=0.01)


def test_search_orbit_saves_and_loads_back(tmp_path):
    simr.run(PARAMS, save_request(["sim.search_orbit"]), output=tmp_path)
    folder = tmp_path / "model_000000"
    assert (folder / "simulations/sim_00_search_orbit.npy").exists()
    loaded = simr.load_simulation(folder, "sim_00")["sim.search_orbit"]
    np.testing.assert_array_equal(loaded, sim.search_orbit(PARAMS, SimulatorConfig(**SMALL)).astype(np.float32))


# --- MDRN-4 step 2b: the metrics stage in simr ---------------------------------------------------------

from galactic_tidal_constraints.simr import metrics  # noqa: E402

SIM_METRICS = [
    {"method": "closest_approach", "inputs": ["sim.search_orbit"]},
    {"method": "closest_approach", "inputs": ["sim.search_orbit"], "select": "distance"},
]
IMG_METRICS = [{"method": "absolute_difference", "inputs": ["img.model", "img.unperturbed"]}]


def metric_request(returned=(), sim_metrics=SIM_METRICS, img_metrics=IMG_METRICS, simulator=SMALL):
    return {"run": {"return": list(returned)}, "simulator": simulator,
            "sim_metrics": sim_metrics, "img_metrics": img_metrics}


def test_metric_values_match_calling_metrics_directly():
    result = simr.run(MODELS, metric_request(returned=["img.model", "img.unperturbed"]))
    assert result.sim_metrics.shape == (2, 1, 2)
    assert result.img_metrics.shape == (2, 1, 1, 1)
    path = sim.search_orbit(HEAVIER, SimulatorConfig(**SMALL)).astype(np.float32)
    expected = metrics.closest_approach(path)
    value = result.sim_metrics[1, 0, 0]
    assert value["t"] == expected["t"] and value["distance"] == expected["distance"]
    assert result.sim_metrics[1, 0, 1] == expected["distance"]
    assert isinstance(result.sim_metrics[1, 0, 1], float)
    difference = metrics.absolute_difference(
        result.data["img.model"][1, 0, 0], result.data["img.unperturbed"][1, 0, 0]
    )
    assert result.img_metrics[1, 0, 0, 0] == difference


def test_filled_in_entries_record_all_four_keys():
    result = simr.run(PARAMS, metric_request())
    assert result.sim_metric_configs == [
        {"method": "closest_approach", "inputs": ["sim.search_orbit"], "select": None, "params": {}},
        {"method": "closest_approach", "inputs": ["sim.search_orbit"], "select": "distance", "params": {}},
    ]
    assert result.img_metric_configs == [
        {"method": "absolute_difference", "inputs": ["img.model", "img.unperturbed"], "select": None, "params": {}},
    ]


def test_metrics_force_computation_without_returning_data():
    result = simr.run(PARAMS, metric_request(img_metrics=[]))
    assert result.data == {}
    assert result.status["sim"].tolist() == [["ran"]]  # not "empty": the metrics needed the search orbit
    assert result.status["img"].tolist() == [[["empty"]]]  # no img metrics, no images
    assert result.img_metrics.shape == (1, 1, 1, 0)
    assert result.sim_metrics[0, 0, 0]["distance"] > 0


def test_img_metrics_alone_compute_their_images():
    result = simr.run(PARAMS, metric_request(sim_metrics=[]))
    assert result.data == {}
    assert result.status["img"].tolist() == [[["ran"]]]
    assert result.status["img_metrics"].tolist() == [[[["ran"]]]]
    assert 0.0 <= result.img_metrics[0, 0, 0, 0] <= 1.0


def test_empty_metric_sections_are_allowed():
    result = simr.run(PARAMS, {"run": {"return": ["sim.orbit"]}, "simulator": SMALL,
                               "sim_metrics": [], "img_metrics": []})
    assert result.sim_metrics.shape == (1, 1, 0)
    assert result.img_metrics.shape == (1, 1, 1, 0)
    assert result.status["sim_metrics"].size == 0 and result.status["img_metrics"].size == 0


def test_a_failed_simulation_skips_its_metrics():
    request = metric_request(simulator=[SMALL, FAILING])
    result = simr.run(np.stack([FIRST_APPROACH, FIRST_APPROACH]), request)
    assert result.status["sim"][0].tolist() == ["ran", "failed"]
    assert result.status["sim_metrics"][0].tolist() == [["ran", "ran"], ["skipped", "skipped"]]
    assert result.status["img_metrics"][0].tolist() == [[["ran"]], [["skipped"]]]
    # A skipped cell: None for a descriptive entry, NaN for one giving a float (a select, a scalar method)
    assert result.sim_metrics[0, 1, 0] is None
    assert math.isnan(result.sim_metrics[0, 1, 1]) and math.isnan(result.img_metrics[0, 1, 0, 0])


def test_a_failed_metric_is_recorded_and_everything_else_moves_on(monkeypatch):
    from dataclasses import replace

    def broken(orbit):
        raise RuntimeError("boom")

    monkeypatch.setitem(metrics.METHODS, "closest_approach",
                        replace(metrics.METHODS["closest_approach"], function=broken))
    result = simr.run(PARAMS, metric_request(returned=["sim.orbit"]))
    assert result.status["sim"].tolist() == [["ran"]]  # the simulation is unaffected
    assert result.status["sim_metrics"].tolist() == [[["failed", "failed"]]]
    assert result.errors["sim_metrics"][0, 0, 0] == "RuntimeError: boom"
    assert result.sim_metrics[0, 0, 0] is None  # descriptive: None
    assert math.isnan(result.sim_metrics[0, 0, 1])  # select: one float, so NaN
    assert result.status["img_metrics"].tolist() == [[[["ran"]]]]  # the image metric still ran


def test_a_nan_metric_value_is_a_failure(monkeypatch):
    from dataclasses import replace

    monkeypatch.setitem(metrics.METHODS, "absolute_difference",
                        replace(metrics.METHODS["absolute_difference"], function=lambda a, b: float("nan")))
    result = simr.run(PARAMS, metric_request(sim_metrics=[]))
    assert result.status["img"].tolist() == [[["ran"]]]  # the image is unaffected
    assert result.status["img_metrics"].tolist() == [[[["failed"]]]]
    assert "nan" in result.errors["img_metrics"][0, 0, 0, 0]
    assert math.isnan(result.img_metrics[0, 0, 0, 0])


def test_float_columns_stay_floats_through_a_pool():
    from concurrent.futures import ProcessPoolExecutor

    request = metric_request(simulator=[SMALL, FAILING])
    with ProcessPoolExecutor(max_workers=2) as pool:
        result = simr.run(np.stack([FIRST_APPROACH, FIRST_APPROACH]), request, map=pool.map)
    assert result.status["sim"][:, 1].tolist() == ["failed", "failed"]
    column = result.sim_metrics[..., 1].astype(float)  # the select entry: usable as a float array
    assert np.isfinite(column[:, 0]).all() and np.isnan(column[:, 1]).all()


def test_the_result_holds_the_catalogs_config_and_code(tmp_path):
    result = simr.run(PARAMS, metric_request(), output=tmp_path)
    [record] = simr.load_catalog(tmp_path)["runs"]
    assert set(result.config) == set(simr.REQUEST_KEYS)
    assert result.config == {key: record[key] for key in result.config}
    assert result.code == record["code"]


def test_without_an_output_the_result_still_says_what_ran():
    result = simr.run(PARAMS, metric_request(img_metrics=[]))
    assert result.config["simulator"] == [json.loads(json.dumps(asdict(SimulatorConfig(**SMALL))))]
    assert result.config["image"] == [json.loads(json.dumps(asdict(ImageConfig())))]
    assert result.config["sim_metrics"] == result.sim_metric_configs
    assert result.config["composite"] is None
    for repo in ("galactic_tidal_constraints", "jspam"):
        assert {"commit", "uncommitted_changes"} <= set(result.code[repo])


def test_the_config_is_its_own_copy():
    result = simr.run(PARAMS, metric_request(img_metrics=[]))
    result.config["sim_metrics"][0]["method"] = "changed"
    assert result.sim_metric_configs[0]["method"] == "closest_approach"


def test_the_full_config_runs_again_as_a_request():
    first = simr.run(PARAMS, {**metric_request(), "composite": {}})
    again = simr.run(PARAMS, first.config)
    assert again.config == first.config
    assert again.composite.tolist() == first.composite.tolist()


def test_a_full_config_with_derived_entries_runs_again():
    target = sim.search_orbit(PARAMS, SimulatorConfig(**SMALL)).astype(np.float32)[::-1][::50].copy()
    request = {
        "simulator": SMALL,
        "derived": [{"name": "win", "method": "recent_approach_cosine_window", "inputs": ["ext.target"]},
                    {"name": "model_win", "method": "recent_approach_cosine_window", "inputs": ["sim.search_orbit"]}],
        "sim_metrics": [{"method": "orbit_time_error", "inputs": ["ext.target", "sim.search_orbit", "drv.win"]}],
    }
    first = simr.run(PARAMS, request, externals={"target": target})
    assert {"level", "shape", "hash"} <= set(first.config["derived"][0])  # recorded, not given
    again = simr.run(PARAMS, first.config, externals={"target": target})
    assert again.config == first.config
    assert again.sim_metrics[0, 0, 0] == first.sim_metrics[0, 0, 0]
    # The recorded keys are worked out again, never taken from the request
    tampered = {**first.config, "derived": [{**first.config["derived"][0], "level": "img", "hash": "x"},
                                            first.config["derived"][1]]}
    assert simr.run(PARAMS, tampered, externals={"target": target}).config == first.config


def test_the_catalog_records_the_metric_entries_and_statuses(tmp_path):
    simr.run(PARAMS, metric_request(), output=tmp_path)
    [record] = simr.load_catalog(tmp_path)["runs"]
    assert "fitness" not in record
    assert [entry["method"] for entry in record["sim_metrics"]] == ["closest_approach", "closest_approach"]
    assert record["img_metrics"][0]["inputs"] == ["img.model", "img.unperturbed"]
    assert record["status"]["sim_metrics"] == [[["ran", "ran"]]]
    assert record["status"]["img_metrics"] == [[[["ran"]]]]


def test_metrics_come_back_the_same_through_a_pool():
    from concurrent.futures import ProcessPoolExecutor

    solo = simr.run(MODELS, metric_request())
    with ProcessPoolExecutor(max_workers=2) as pool:
        pooled = simr.run(MODELS, metric_request(), map=pool.map)
    for m in range(len(MODELS)):
        assert solo.sim_metrics[m, 0, 1] == pooled.sim_metrics[m, 0, 1]
        assert solo.img_metrics[m, 0, 0, 0] == pooled.img_metrics[m, 0, 0, 0]
    assert pooled.status["sim_metrics"].tolist() == solo.status["sim_metrics"].tolist()


def test_metrics_use_saved_arrays_instead_of_computing(tmp_path, monkeypatch):
    simr.run(PARAMS, save_request(["sim.search_orbit"]), output=tmp_path)
    fresh = simr.run(PARAMS, metric_request(img_metrics=[]))
    for item in ("sim.particles", "sim.unperturbed", "sim.orbit", "sim.search_orbit"):
        monkeypatch.setitem(simr.SIM_DATA, item, _not_computed)
    reused = simr.run(PARAMS, metric_request(img_metrics=[]), output=tmp_path)
    assert reused.status["sim"].tolist() == [["reused"]]
    assert reused.status["sim_metrics"].tolist() == [[["ran", "ran"]]]
    assert reused.sim_metrics[0, 0, 1] == fresh.sim_metrics[0, 0, 1]


def test_a_metric_cell_is_timed_on_its_own():
    result = simr.run(PARAMS, metric_request())
    assert result.seconds["sim_metrics"].shape == (1, 1, 2)
    assert (result.seconds["sim_metrics"] >= 0).all()
    assert (result.seconds["img_metrics"] >= 0).all()


@pytest.mark.parametrize(
    ("section", "entries", "message"),
    [
        ("sim_metrics", [{"method": "nope", "inputs": []}], "unknown method"),
        ("sim_metrics", [{"inputs": ["sim.orbit"]}], 'needs a "method"'),
        ("sim_metrics", [{"method": "closest_approach"}], "takes 1 input"),
        ("sim_metrics", [{"method": "closest_approach", "inputs": ["sim.orbit", "sim.orbit"]}], "takes 1 input"),
        ("sim_metrics", [{"method": "closest_approach", "inputs": ["sim.timing"]}], "isn't a metric input"),
        ("sim_metrics", [{"method": "closest_approach", "inputs": ["ext.test_orbit"]}], "no simr item among the inputs"),
        ("sim_metrics", [{"method": "pca_difference", "inputs": ["ext.test_particles", "sim.particles"]}],
         "names an external the run wasn't given: no externals were given"),
        ("sim_metrics", [{"method": "closest_approach", "inputs": ["sim.particles"]}],
         "takes a orbit here, but sim.particles is a particles"),
        ("sim_metrics", [{"method": "absolute_difference", "inputs": ["img.model", "img.unperturbed"]}],
         "belongs in img_metrics"),
        ("img_metrics", [{"method": "closest_approach", "inputs": ["sim.search_orbit"]}],
         "belongs in sim_metrics"),
        ("sim_metrics", [{"method": "closest_approach", "inputs": ["sim.orbit"], "select": "position"}],
         "isn't a single-float output"),
        ("sim_metrics", [{"method": "closest_approach", "inputs": ["sim.orbit"], "select": "nope"}],
         "isn't a single-float output"),
        ("img_metrics", [{"method": "absolute_difference", "inputs": ["img.model", "img.model"],
                          "select": "distance"}], "is a scalar method"),
        ("sim_metrics", [{"method": "closest_approach", "inputs": ["sim.orbit"], "params": {"h": 1}}],
         "unknown settings"),
        ("sim_metrics", [{"method": "closest_approach", "inputs": ["sim.orbit"], "extra": 1}],
         "unknown keys in sim_metrics"),
        ("sim_metrics", ["closest_approach"], "must be a dict"),
        ("sim_metrics", "closest_approach", "must be a metric entry"),
    ],
)
def test_bad_metric_entries_name_the_problem(section, entries, message):
    with pytest.raises(ValueError, match=message):
        simr.run(PARAMS, {"simulator": SMALL, section: entries})


# --- Externals: arrays from outside the run, as metric inputs (MDRN-4 step 3) -------------------------

EXTERNAL_PARTICLES = np.random.default_rng(3).normal(size=(40, 6))  # float64 on purpose: simr converts
EXTERNAL_IMAGE = np.random.default_rng(4).random((128, 128)).astype(np.float32)
PARTICLES_ENTRY = {"method": "pca_difference", "inputs": ["ext.test_particles", "sim.particles"]}
IMAGE_ENTRY = {"method": "absolute_difference", "inputs": ["ext.test_image", "img.model"]}


def externals_request(sim_metrics=(PARTICLES_ENTRY,), img_metrics=(), returned=()):
    return {"run": {"return": list(returned)}, "simulator": SMALL,
            "sim_metrics": list(sim_metrics), "img_metrics": list(img_metrics)}


def test_an_external_array_is_a_metric_input_at_both_levels():
    result = simr.run(PARAMS, externals_request(img_metrics=[IMAGE_ENTRY], returned=["sim.particles", "img.model"]),
                      externals={"test_particles": EXTERNAL_PARTICLES, "test_image": EXTERNAL_IMAGE})
    assert result.status["sim_metrics"].tolist() == [[["ran"]]]
    assert result.status["img_metrics"].tolist() == [[[["ran"]]]]
    assert result.sim_metrics[0, 0, 0] == metrics.pca_difference(
        EXTERNAL_PARTICLES.astype(np.float32), result.data["sim.particles"][0, 0]
    )
    assert result.img_metrics[0, 0, 0, 0] == metrics.absolute_difference(
        EXTERNAL_IMAGE, result.data["img.model"][0, 0, 0]
    )


def test_an_external_given_as_a_path_matches_the_array(tmp_path):
    np.save(tmp_path / "particles.npy", EXTERNAL_PARTICLES)
    from_array = simr.run(PARAMS, externals_request(), externals={"test_particles": EXTERNAL_PARTICLES})
    from_path = simr.run(PARAMS, externals_request(), externals={"test_particles": tmp_path / "particles.npy"})
    from_string = simr.run(PARAMS, externals_request(), externals={"test_particles": str(tmp_path / "particles.npy")})
    assert from_array.sim_metrics[0, 0, 0] == from_path.sim_metrics[0, 0, 0] == from_string.sim_metrics[0, 0, 0]


def test_externals_come_back_the_same_through_a_pool():
    from concurrent.futures import ProcessPoolExecutor

    externals = {"test_particles": EXTERNAL_PARTICLES}
    solo = simr.run(MODELS, externals_request(), externals=externals)
    with ProcessPoolExecutor(max_workers=2) as pool:
        pooled = simr.run(MODELS, externals_request(), externals=externals, map=pool.map)
    assert pooled.sim_metrics.tolist() == solo.sim_metrics.tolist()


def test_array_hash_is_sha256_of_the_shape_and_the_float32_values():
    import hashlib

    as_float32 = EXTERNAL_PARTICLES.astype("<f4")
    expected = hashlib.sha256(b"float32(40, 6)" + as_float32.tobytes()).hexdigest()
    assert simr.array_hash(as_float32) == expected
    assert simr.array_hash(EXTERNAL_PARTICLES) == expected  # float64 in: hashed as the float32 simr uses
    assert simr.array_hash(np.asfortranarray(as_float32)) == expected  # memory layout doesn't matter
    assert simr.array_hash(as_float32.reshape(6, 40)) != expected  # the shape does
    assert simr.array_hash(as_float32 + 1) != expected


def test_the_result_and_the_catalog_record_each_external(tmp_path, caplog):
    np.save(tmp_path / "image.npy", EXTERNAL_IMAGE)
    digest = simr.array_hash(EXTERNAL_PARTICLES)
    with caplog.at_level("INFO"):
        result = simr.run(PARAMS, externals_request(img_metrics=[IMAGE_ENTRY]), output=tmp_path / "runs",
                          externals={"test_particles": EXTERNAL_PARTICLES, "test_image": tmp_path / "image.npy",
                                     "spare": EXTERNAL_IMAGE})
    assert result.externals == {  # only the used ones
        "test_particles": {"source": "memory", "shape": [40, 6], "hash": digest},
        "test_image": {"source": str(tmp_path / "image.npy"), "shape": [128, 128],
                      "hash": simr.array_hash(EXTERNAL_IMAGE)},
    }
    [record] = simr.load_catalog(tmp_path / "runs")["runs"]
    assert record["externals"] == result.externals
    assert f"external 'test_particles': memory, shape (40, 6), hash {digest}" in caplog.text


def test_a_run_without_externals_records_none(tmp_path):
    result = simr.run(PARAMS, metric_request(), output=tmp_path)
    assert result.externals == {}
    assert simr.load_catalog(tmp_path)["runs"][0]["externals"] == {}


def test_an_expected_hash_must_match_the_data(tmp_path, monkeypatch):
    np.save(tmp_path / "particles.npy", EXTERNAL_PARTICLES)
    digest = simr.array_hash(EXTERNAL_PARTICLES)
    given = {"test_particles": {"path": tmp_path / "particles.npy", "hash": digest}}
    assert simr.run(PARAMS, externals_request(), externals=given).externals["test_particles"]["hash"] == digest
    np.save(tmp_path / "particles.npy", EXTERNAL_PARTICLES + 1)  # the file changed since the hash was taken
    monkeypatch.setitem(simr.SIM_DATA, "sim.particles", _not_computed)
    with pytest.raises(ValueError, match=f"the data's hash is [0-9a-f]{{64}}, not the expected {digest}"):
        simr.run(PARAMS, externals_request(), externals=given)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"hash": "abc"}, 'a dict holds "path" and, optionally, "hash"'),
        ({"path": "/x.npy", "sha": "abc"}, 'a dict holds "path" and, optionally, "hash"'),
        ({"path": "/x.npy", "hash": 12}, "the hash must be a string"),
        ({"path": EXTERNAL_PARTICLES}, "the path must be a string or Path"),
    ],
)
def test_a_bad_external_dict_names_the_problem(value, message):
    with pytest.raises(ValueError, match=message):
        simr.run(PARAMS, externals_request(), externals={"test_particles": value})


def _shared_folders():
    root = simr.SHARED_MEMORY if simr.SHARED_MEMORY.is_dir() else Path(__import__("tempfile").gettempdir())
    return sorted(root.glob("simr_externals_*"))


def test_a_pool_shares_one_copy_and_removes_it_when_the_run_ends(caplog):
    from concurrent.futures import ProcessPoolExecutor

    before = _shared_folders()
    with ProcessPoolExecutor(max_workers=2) as pool, caplog.at_level("INFO"):
        simr.run(MODELS, externals_request(), externals={"test_particles": EXTERNAL_PARTICLES}, map=pool.map)
    assert "externals ['ext.test_particles'] shared with the workers through" in caplog.text
    assert _shared_folders() == before  # the run's folder is gone


def test_the_shared_copy_is_removed_when_the_run_fails(tmp_path):
    from concurrent.futures import ProcessPoolExecutor

    before = _shared_folders()
    with ProcessPoolExecutor(max_workers=2) as pool, pytest.raises(ValueError, match="needs an output folder"):
        simr.run(MODELS, {**externals_request(), "run": {"save": ["sim.orbit"]}},
                 externals={"test_particles": EXTERNAL_PARTICLES}, map=pool.map)
    assert _shared_folders() == before


def test_a_solo_run_shares_nothing(caplog):
    with caplog.at_level("INFO"):
        simr.run(PARAMS, externals_request(), externals={"test_particles": EXTERNAL_PARTICLES})
    assert "shared with the workers" not in caplog.text


def test_a_float32_file_is_mapped_as_it_is_and_others_get_a_shared_copy(tmp_path):
    np.save(tmp_path / "f32.npy", EXTERNAL_PARTICLES.astype(np.float32))
    np.save(tmp_path / "f64.npy", EXTERNAL_PARTICLES)
    uses = {"a": ("particles", "here"), "b": ("particles", "here"), "c": ("particles", "here")}
    arrays, files, records = simr._read_externals(
        {"a": tmp_path / "f32.npy", "b": tmp_path / "f64.npy", "c": EXTERNAL_PARTICLES}, uses, []
    )
    assert files == {"ext.a": tmp_path / "f32.npy", "ext.b": None, "ext.c": None}
    named = {f"ext.{key}": record for key, record in records.items()}
    paths, folder = simr._share_externals(arrays, files, named)
    try:
        assert paths["ext.a"] == str(tmp_path / "f32.npy")  # the caller's own file: no copy
        assert paths["ext.b"] == paths["ext.c"] == str(folder / f"{records['b']['hash']}.npy")  # same data: once
        assert len(list(folder.iterdir())) == 1
        opened = simr._open_externals(paths)
        for key in uses:
            name = f"ext.{key}"
            assert opened[name].dtype == np.float32 and not opened[name].flags.writeable
            assert (opened[name] == EXTERNAL_PARTICLES.astype(np.float32)).all()
        assert simr._open_externals(paths)["ext.b"] is opened["ext.b"]  # mapped once per process
        simr._open_externals({"ext.a": paths["ext.a"]})
        assert len(simr._OPEN_EXTERNALS) == 1  # only the latest task's are kept
    finally:
        simr._OPEN_EXTERNALS.clear()
        __import__("shutil").rmtree(folder)
    only_file, no_folder = simr._share_externals({"ext.a": arrays["ext.a"]}, {"ext.a": files["ext.a"]}, named)
    assert no_folder is None and only_file == {"ext.a": str(tmp_path / "f32.npy")}


def test_too_little_shared_memory_stops_the_run_before_anything_is_written(monkeypatch):
    from collections import namedtuple
    from concurrent.futures import ProcessPoolExecutor

    monkeypatch.setattr(simr.shutil, "disk_usage", lambda path: namedtuple("usage", "total used free")(100, 90, 10))
    before = _shared_folders()
    with ProcessPoolExecutor(max_workers=2) as pool, pytest.raises(ValueError, match="need 0.0 MB in .* but 0.0 MB is free"):
        simr.run(MODELS, externals_request(), externals={"test_particles": EXTERNAL_PARTICLES}, map=pool.map)
    assert _shared_folders() == before


MULTI_NODE_MPI = [
    {"OMPI_COMM_WORLD_LOCAL_SIZE": "2", "OMPI_COMM_WORLD_SIZE": "4"},  # OpenMPI: 4 ranks, 2 on this node
    {"MPI_LOCALNRANKS": "1", "PMI_SIZE": "3"},  # MPICH / Intel MPI
]


@pytest.mark.parametrize("variables", MULTI_NODE_MPI)
def test_a_multi_node_mpi_job_with_shared_externals_stops_before_anything_is_written(
        tmp_path, monkeypatch, variables):
    from concurrent.futures import ProcessPoolExecutor  # stands in for the MPI pool: any pool shares

    for name, value in variables.items():
        monkeypatch.setenv(name, value)
    before = _shared_folders()
    with ProcessPoolExecutor(max_workers=2) as pool, pytest.raises(
            ValueError, match=r"externals \['ext.test_particles'\] are shared through this node's memory .* only "):
        simr.run(MODELS, externals_request(), output=tmp_path / "runs",
                 externals={"test_particles": EXTERNAL_PARTICLES}, map=pool.map)
    assert not (tmp_path / "runs").exists()  # no catalog
    assert _shared_folders() == before  # nothing in /dev/shm


def test_float32_files_need_no_shared_copy_so_a_multi_node_job_may_use_them(tmp_path, monkeypatch):
    from concurrent.futures import ProcessPoolExecutor

    monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_SIZE", "2")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    np.save(tmp_path / "particles.npy", EXTERNAL_PARTICLES.astype(np.float32))
    with ProcessPoolExecutor(max_workers=2) as pool:
        result = simr.run(MODELS, externals_request(), externals={"test_particles": tmp_path / "particles.npy"},
                          map=pool.map)
    assert result.status["sim_metrics"].tolist() == [[["ran"]], [["ran"]]]


@pytest.mark.parametrize("pooled", [False, True])
def test_one_node_mpi_jobs_and_solo_runs_are_not_stopped(monkeypatch, pooled):
    from concurrent.futures import ProcessPoolExecutor

    monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_SIZE", "4")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")  # every rank on this node
    if not pooled:
        monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_SIZE", "1")  # solo shares nothing, so nothing to check
    externals = {"test_particles": EXTERNAL_PARTICLES}
    if pooled:
        with ProcessPoolExecutor(max_workers=2) as pool:
            result = simr.run(MODELS, externals_request(), externals=externals, map=pool.map)
    else:
        result = simr.run(MODELS, externals_request(), externals=externals)
    assert result.status["sim_metrics"].tolist() == [[["ran"]], [["ran"]]]


def test_a_worker_that_cannot_see_the_shared_copy_says_why():
    with pytest.raises(simr.SimrRunError, match="every worker must run on the node that started the run"):
        simr._open_externals({"test_particles": "/dev/shm/simr_externals_gone/external_0.npy"})


def test_a_metric_gets_its_external_read_only_and_the_callers_array_is_untouched(monkeypatch):
    from dataclasses import replace

    def writes(a, b, **settings):
        a[0, 0] = 1.0
        return 0.0

    monkeypatch.setitem(metrics.METHODS, "pca_difference",
                        replace(metrics.METHODS["pca_difference"], function=writes))
    mine = EXTERNAL_PARTICLES.astype(np.float32)
    result = simr.run(PARAMS, externals_request(), externals={"test_particles": mine})
    assert result.status["sim_metrics"].tolist() == [[["failed"]]]
    assert "read-only" in result.errors["sim_metrics"][0, 0, 0]
    assert mine.flags.writeable and (mine == EXTERNAL_PARTICLES.astype(np.float32)).all()


def test_externals_no_metric_uses_are_ignored_and_logged(caplog):
    with caplog.at_level("INFO"):
        result = simr.run(PARAMS, externals_request(),
                          externals={"test_particles": EXTERNAL_PARTICLES, "spare": "/nowhere/not_loaded.npy"})
    assert result.status["sim_metrics"].tolist() == [[["ran"]]]
    assert "ignored: ['spare']" in caplog.text


def _missing(tmp_path):
    return tmp_path / "missing.npy"


def _not_an_array(tmp_path):
    (tmp_path / "text.npy").write_text("not an array")
    return tmp_path / "text.npy"


@pytest.mark.parametrize(
    ("external", "message"),
    [
        (lambda tmp_path: np.zeros((5, 7)), r"externals\['test_particles'\] \(used by sim_metrics\[0\].inputs\[0\]\) "
                                            r"must be a 'particles' array, shape \(None, 6\): got \(5, 7\)"),
        (lambda tmp_path: np.full((5, 6), np.nan), "holds NaN or inf"),
        (lambda tmp_path: np.full((5, 6), 1e300), "holds NaN or inf"),  # too big for float32
        (lambda tmp_path: np.zeros((5, 6), dtype=bool), "must be a numeric numpy array.*got bool array"),
        (lambda tmp_path: [[0.0] * 6] * 5, "must be a numeric numpy array.*got list"),
        (lambda tmp_path: "particles.npy", "a path must be absolute and name a .npy file"),
        (lambda tmp_path: tmp_path / "particles.txt", "a path must be absolute and name a .npy file"),
        (_missing, "can't load"),
        (_not_an_array, "can't load"),
    ],
)
def test_a_bad_external_stops_the_run_before_anything_is_simulated(external, message, tmp_path, monkeypatch):
    monkeypatch.setitem(simr.SIM_DATA, "sim.particles", _not_computed)
    with pytest.raises(ValueError, match=message):
        simr.run(PARAMS, externals_request(), externals={"test_particles": external(tmp_path)})


@pytest.mark.parametrize(
    ("externals", "message"),
    [
        ({"other": EXTERNAL_PARTICLES}, r"ext.test_particles names an external the run wasn't given: externals has \['other'\]"),
        ({"test_particles": EXTERNAL_PARTICLES, "sim.mine": EXTERNAL_PARTICLES}, "a key must be letters, digits and _"),
        ({"test_particles": EXTERNAL_PARTICLES, "img.mine": EXTERNAL_PARTICLES}, "a key must be letters, digits and _"),
        ([EXTERNAL_PARTICLES], "externals must be a dict"),
    ],
)
def test_bad_externals_keys_name_the_problem(externals, message):
    with pytest.raises(ValueError, match=message):
        simr.run(PARAMS, externals_request(), externals=externals)


def test_an_external_image_must_match_every_image_entry():
    request = {**externals_request(sim_metrics=[], img_metrics=[IMAGE_ENTRY]), "image": [{}, {"width": 64}]}
    with pytest.raises(ValueError, match=r"is 128 x 128 \(height x width\), but image\[1\] makes 128 x 64 images"):
        simr.run(PARAMS, request, externals={"test_image": EXTERNAL_IMAGE})


def test_one_external_has_one_kind():
    request = externals_request(img_metrics=[{"method": "absolute_difference", "inputs": ["ext.test_particles", "img.model"]}])
    with pytest.raises(ValueError, match="used as a image here but as a particles in sim_metrics"):
        simr.run(PARAMS, request, externals={"test_particles": EXTERNAL_PARTICLES})


def test_a_method_with_a_setup_step_is_not_supported_yet():
    from dataclasses import replace

    original = metrics.METHODS["closest_approach"]
    metrics.METHODS["closest_approach"] = replace(original, setup=lambda: None)
    try:
        with pytest.raises(NotImplementedError, match="setup step"):
            simr.run(PARAMS, metric_request(img_metrics=[]))
    finally:
        metrics.METHODS["closest_approach"] = original
