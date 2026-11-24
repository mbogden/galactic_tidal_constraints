"""Tests for saving derived arrays made from a model's data (MDRN-4 step 7c): run.save takes the
"drv.<name>" of per-simulation and per-image entries, which go in the model's folder and are listed in
model.json."""

import pytest

np = pytest.importorskip("numpy")
jspam = pytest.importorskip("jspam")

from galactic_tidal_constraints.simr import derived, simr  # noqa: E402
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
# Approaching now and unbound: its only pass is too long ago for a window, so that derived entry fails.
APPROACHING = PARAMS.copy()
APPROACHING[3:6] *= -3

TARGET_IMAGE = np.random.default_rng(9).random((128, 128)).astype(np.float32)
MODEL_WIN = {"name": "model_win", "method": "recent_approach_cosine_window", "inputs": ["sim.search_orbit"]}
TIDAL = {"name": "tidal", "method": "difference_weights", "inputs": ["ext.target_image", "img.unperturbed"]}
EXTERNALS = {"target_image": TARGET_IMAGE}


def request(*saved, derived_entries=(MODEL_WIN, TIDAL), simulator=SMALL, image=None):
    config = {"run": {"save": list(saved)}, "simulator": simulator, "derived": list(derived_entries)}
    return config if image is None else {**config, "image": image}


def test_a_per_simulation_array_goes_in_the_models_folder(tmp_path):
    simr.run(MODELS, request("drv.model_win"), externals=EXTERNALS, output=tmp_path)
    for model_id, params in (("000000", PARAMS), ("000001", HEAVIER)):
        folder = tmp_path / f"model_{model_id}"
        record = simr.load_model(folder)
        listed = record["derived"]["model_win"]
        assert listed["files"] == {"sim_00": "derived/sim_00_drv_model_win.npy"}
        assert listed["config"]["method"] == "recent_approach_cosine_window" and listed["config"]["level"] == "sim"
        assert listed["config"]["params"] == {"k": 1.0, "flat": 0.0, "pass_number": 1}
        assert listed["made_from"] == {}
        path = sim.search_orbit(params, SimulatorConfig(**SMALL)).astype(np.float32)
        saved = np.load(folder / listed["files"]["sim_00"])
        np.testing.assert_array_equal(saved, derived.recent_approach_cosine_window(path))
        # Its simulation is recorded too (config, timing), with no arrays of its own
        assert record["simulations"]["sim_00"]["files"] == {} and "t_start" in record["simulations"]["sim_00"]["timing"]
    assert not list(tmp_path.rglob("*.tmp")) and not (tmp_path / "derived").exists()


def test_a_per_image_array_is_listed_by_simulation_and_image(tmp_path):
    images = [{}, {"blur_sigma": 1.0}]
    result = simr.run(PARAMS, {**request("drv.tidal", image=images), "run": {"save": ["drv.tidal"],
                                                                              "return": ["drv.tidal"]}},
                      externals=EXTERNALS, output=tmp_path)
    folder = tmp_path / "model_000000"
    record = simr.load_model(folder)
    listed = record["derived"]["tidal"]
    assert listed["files"] == {"sim_00": {"img_0000": "derived/sim_00_img_0000_drv_tidal.npy",
                                          "img_0001": "derived/sim_00_img_0001_drv_tidal.npy"}}
    for i in range(2):
        np.testing.assert_array_equal(np.load(folder / listed["files"]["sim_00"][f"img_000{i}"]),
                                      result.data["drv.tidal"][0, 0, i])
    # What it was made from: the external, by hash. And its image entries are recorded, with no images.
    assert listed["made_from"] == {"ext.target_image": {"shape": [128, 128],
                                                        "hash": result.externals["target_image"]["hash"]}}
    assert record["images"]["img_0001"]["config"]["blur_sigma"] == 1.0
    assert record["images"]["img_0001"]["files"] == {}


def test_load_derived_reads_them_back(tmp_path):
    result = simr.run(PARAMS, {**request("drv.model_win", "drv.tidal"),
                               "run": {"save": ["drv.model_win", "drv.tidal"], "return": ["drv.model_win", "drv.tidal"]}},
                      externals=EXTERNALS, output=tmp_path)
    loaded = simr.load_derived(tmp_path / "model_000000")
    assert list(loaded) == [("model_win", "sim_00"), ("tidal", "sim_00", "img_0000")]
    np.testing.assert_array_equal(loaded[("model_win", "sim_00")], result.data["drv.model_win"][0, 0])
    np.testing.assert_array_equal(loaded[("tidal", "sim_00", "img_0000")], result.data["drv.tidal"][0, 0, 0])
    assert list(simr.load_derived(tmp_path / "model_000000", "tidal")) == [("tidal", "sim_00", "img_0000")]
    with pytest.raises(KeyError, match="no saved derived entry 'nope'"):
        simr.load_derived(tmp_path / "model_000000", "nope")


def test_an_entry_made_from_another_records_the_chain(tmp_path):
    edges = {"name": "edges", "method": "detect_edges", "inputs": ["drv.tidal"]}
    simr.run(PARAMS, request("drv.edges", derived_entries=(TIDAL, edges)), externals=EXTERNALS, output=tmp_path)
    listed = simr.load_model(tmp_path / "model_000000")["derived"]
    assert list(listed) == ["edges"]  # only what run.save named
    made_from = listed["edges"]["made_from"]
    assert made_from["drv.tidal"]["method"] == "difference_weights"
    assert made_from["ext.target_image"]["hash"] == simr.array_hash(TARGET_IMAGE)


def test_a_failed_entry_saves_nothing_and_the_rest_carry_on(tmp_path):
    result = simr.run(np.stack([APPROACHING, PARAMS]), request("drv.model_win", "drv.tidal"),
                      externals=EXTERNALS, output=tmp_path)
    assert result.status["sim"].tolist() == [["ran"], ["ran"]]
    failed = simr.load_model(tmp_path / "model_000000")["derived"]
    assert list(failed) == ["tidal"]  # its window couldn't be made; its weight image could
    assert sorted(simr.load_model(tmp_path / "model_000001")["derived"]) == ["model_win", "tidal"]


def test_a_changed_entry_starts_its_record_again(tmp_path, caplog):
    simr.run(PARAMS, request("drv.model_win"), externals=EXTERNALS, output=tmp_path)
    wider = {**MODEL_WIN, "params": {"k": 0.5}}
    simr.run(PARAMS, request("drv.model_win", derived_entries=(wider, TIDAL)), externals=EXTERNALS, output=tmp_path)
    assert "derived 'model_win' was saved with a different config" in caplog.text
    listed = simr.load_model(tmp_path / "model_000000")["derived"]["model_win"]
    assert listed["config"]["params"]["k"] == 0.5
    path = sim.search_orbit(PARAMS, SimulatorConfig(**SMALL)).astype(np.float32)
    np.testing.assert_array_equal(simr.load_derived(tmp_path / "model_000000")[("model_win", "sim_00")],
                                  derived.recent_approach_cosine_window(path, k=0.5))


def test_a_changed_external_also_starts_the_record_again(tmp_path):
    simr.run(PARAMS, request("drv.tidal"), externals=EXTERNALS, output=tmp_path)
    other = np.random.default_rng(10).random((128, 128)).astype(np.float32)
    simr.run(PARAMS, request("drv.tidal"), externals={"target_image": other}, output=tmp_path)
    listed = simr.load_model(tmp_path / "model_000000")["derived"]["tidal"]
    assert listed["made_from"]["ext.target_image"]["hash"] == simr.array_hash(other)


def test_nothing_is_added_under_a_simulation_saved_with_another_config(tmp_path):
    simr.run(PARAMS, request("drv.model_win"), externals=EXTERNALS, output=tmp_path)
    simr.run(PARAMS, request("drv.tidal", simulator={**SMALL, "lnl": 0.01}), externals=EXTERNALS, output=tmp_path)
    assert list(simr.load_model(tmp_path / "model_000000")["derived"]) == ["model_win"]


def test_a_second_simulator_entry_adds_its_own_files(tmp_path):
    simr.run(PARAMS, request("drv.model_win", simulator=[SMALL, {**SMALL, "lnl": 0.01}]), externals=EXTERNALS,
             output=tmp_path)
    assert list(simr.load_derived(tmp_path / "model_000000")) == [("model_win", "sim_00"), ("model_win", "sim_01")]


def test_saving_works_through_a_pool(tmp_path):
    from concurrent.futures import ProcessPoolExecutor

    solo = simr.run(MODELS, request("drv.model_win", "drv.tidal"), externals=EXTERNALS, output=tmp_path / "solo")
    with ProcessPoolExecutor(max_workers=2) as pool:
        simr.run(MODELS, request("drv.model_win", "drv.tidal"), externals=EXTERNALS, output=tmp_path / "pool",
                 map=pool.map)
    for model_id in solo.ids:
        one = simr.load_derived(tmp_path / "solo" / f"model_{model_id}")
        other = simr.load_derived(tmp_path / "pool" / f"model_{model_id}")
        assert list(one) == list(other)
        for key in one:
            np.testing.assert_array_equal(one[key], other[key])


def test_every_kind_of_run_save_name_together(tmp_path):
    window = {"name": "win", "method": "detect_edges", "inputs": ["ext.target_image"]}
    simr.run(PARAMS, request("sim.orbit", "img.model", "ext.target_image", "drv.win", "drv.model_win", "drv.tidal",
                             derived_entries=(window, MODEL_WIN, TIDAL)), externals=EXTERNALS, output=tmp_path)
    folders = sorted({path.parent.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*.npy")})
    assert folders == ["derived", "externals", "model_000000/derived", "model_000000/images",
                       "model_000000/simulations"]
