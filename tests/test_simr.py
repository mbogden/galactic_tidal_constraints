"""Tests for the SIMR workflow (simr/simr.py). Each section covers one step of the rebuild."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

# Skipped where the dependencies aren't installed, such as native Windows.
np = pytest.importorskip("numpy")
jspam = pytest.importorskip("jspam")

from galactic_tidal_constraints.simr import simr  # noqa: E402
from galactic_tidal_constraints.simr import simulation as sim  # noqa: E402
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
    with pytest.raises(ValueError, match="no output folder"):
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


def test_reading_alone_writes_nothing(tmp_path):
    simr.run(MODELS, request(["sim.orbit"]), output=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_reading_a_model_the_folder_doesnt_have_computes_it(tmp_path):
    simr.run(MODELS[:1], save_request(["sim.orbit"]), output=tmp_path)
    result = simr.run(MODELS, request(["sim.orbit"]), output=tmp_path)
    assert result.ids == ["000000", None]
    assert result.data["sim.orbit"][1, 0].shape[1] == 7



# --- Step 8c: the output catalog -----------------------------------------------------------------------


def test_catalog_lists_the_models_and_the_run(tmp_path):
    result = simr.run(MODELS, save_request(["sim.orbit"], simulator=[SMALL, {**SMALL, "lnl": 0.01}]), output=tmp_path)
    assert result.ids == ["000000", "000001"]
    catalog = simr.load_catalog(tmp_path)
    assert catalog["models"] == {"000000": simr.model_parameters(PARAMS), "000001": simr.model_parameters(HEAVIER)}
    [record] = catalog["runs"]
    assert record["started"] <= record["finished"]
    assert record["run"] == {"return": [], "save": ["sim.orbit"]}
    assert record["simulator"] == [asdict(SimulatorConfig(**SMALL)), asdict(SimulatorConfig(**SMALL, lnl=0.01))]
    assert record["image"] == [] and record["fitness"] == []
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


def test_a_run_that_only_reads_adds_nothing_to_the_catalog(tmp_path):
    simr.run(MODELS[:1], save_request(["sim.orbit"]), output=tmp_path)
    before = simr.load_catalog(tmp_path)
    simr.run(MODELS, request(["sim.orbit"]), output=tmp_path)
    assert simr.load_catalog(tmp_path) == before


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
        assert list(field) == ["sim"] and field["sim"].shape == (2, 2)
    assert (result.seconds["sim"] > 0).all()


def test_the_catalog_records_each_simulations_status(tmp_path):
    simr.run(np.stack([PARAMS, FIRST_APPROACH]), save_request(["sim.orbit"], simulator=FAILING), output=tmp_path)
    [record] = simr.load_catalog(tmp_path)["runs"]
    assert record["status"] == {"sim": [["ran"], ["failed"]]}
    assert record["errors"]["sim"][0] == [None] and record["errors"]["sim"][1][0].startswith("ValueError")
    assert len(record["seconds"]["sim"]) == 2 and record["finished"] is not None


def test_failures_are_logged_with_a_count(caplog):
    with caplog.at_level("INFO", logger="galactic_tidal_constraints.simr.simr"):
        simr.run(np.stack([PARAMS, FIRST_APPROACH]), request(["sim.orbit"], simulator=FAILING))
    failure = next(r for r in caplog.records if r.levelname == "ERROR")
    assert "model 1" in failure.getMessage() and failure.exc_info is not None
    assert "2 simulations: 1 ran, 0 reused, 1 failed" in caplog.text
