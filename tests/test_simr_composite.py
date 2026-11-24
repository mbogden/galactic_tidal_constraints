"""Tests for the composite section in simr (MDRN-4 step 5b): one weighted mean per model of its scalar
metric cells, checked before anything runs and recorded in the catalog."""

import json
import math

import pytest

np = pytest.importorskip("numpy")
jspam = pytest.importorskip("jspam")

from galactic_tidal_constraints.simr import derived, metrics, simr  # noqa: E402
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
# A simulator entry the first model fails on: it has no pass with a first-approach velocity.
FIRST_APPROACH = PARAMS.copy()
FIRST_APPROACH[3:6] *= -3
FAILING = {**SMALL, "no_pass": "error"}

# A stand-in target: the first model's own path, every 50th row, in increasing time.
PATH = sim.search_orbit(PARAMS, SimulatorConfig(**SMALL)).astype(np.float32)
TARGET = PATH[::-1][::50].copy()
EXTERNALS = {"test_orbit": TARGET, "test_weights": derived.recent_approach_cosine_window(TARGET)}
ORBIT_INPUTS = ["ext.test_orbit", "sim.search_orbit", "ext.test_weights"]
# The old code's 50/50 mix: the time and the position error, then a descriptive entry (a dict).
ORBIT_METRICS = [
    {"method": "orbit_time_error", "inputs": ORBIT_INPUTS},
    {"method": "orbit_position_error", "inputs": ORBIT_INPUTS},
    {"method": "closest_approach", "inputs": ["sim.search_orbit"]},
]
IMG_METRICS = [{"method": "absolute_difference", "inputs": ["img.model", "img.unperturbed"]}]


def request(composite, sim_metrics=ORBIT_METRICS, img_metrics=(), simulator=SMALL):
    return {"simulator": simulator, "sim_metrics": list(sim_metrics), "img_metrics": list(img_metrics),
            "composite": composite}


def test_the_50_50_orbit_mix():
    result = simr.run(MODELS, request({"sim_weights": [[0.5, 0.5, 0.0]]}), externals=EXTERNALS)
    assert result.composite.dtype == np.float64 and result.composite.shape == (2,)
    expected = 0.5 * result.sim_metrics[:, 0, 0].astype(float) + 0.5 * result.sim_metrics[:, 0, 1].astype(float)
    np.testing.assert_allclose(result.composite, expected)
    assert result.composite[0] == pytest.approx(0, abs=1e-9) and result.composite[1] > 0  # the target's own model wins
    assert result.composite_weights == {"sim_weights": [[0.5, 0.5, 0.0]]}


def test_the_composite_is_metrics_composite_on_the_returned_arrays():
    weights = {"sim_weights": [[1.0, 3.0, 0.0]], "img_weights": [[[2.0]]]}
    result = simr.run(MODELS, request(weights, img_metrics=IMG_METRICS), externals=EXTERNALS)
    expected = metrics.composite(result.sim_metrics, result.img_metrics, **weights)
    np.testing.assert_array_equal(result.composite, expected)


def test_an_empty_section_weighs_every_float_entry_equally_and_dicts_zero():
    result = simr.run(MODELS, request({}, img_metrics=IMG_METRICS), externals=EXTERNALS)
    assert result.composite_weights == {"sim_weights": [[1.0, 1.0, 0.0]], "img_weights": [[[1.0]]]}
    cells = [result.sim_metrics[:, 0, 0], result.sim_metrics[:, 0, 1], result.img_metrics[:, 0, 0, 0]]
    np.testing.assert_allclose(result.composite, np.mean([c.astype(float) for c in cells], axis=0))


def test_weights_cover_every_simulator_and_image_entry():
    simulators = [SMALL, {**SMALL, "lnl": 0.01}]
    images = [{}, {"blur_sigma": 1.0}]
    weights = {"img_weights": [[[1.0], [0.0]], [[0.0], [3.0]]]}  # (simulator, image, entry)
    result = simr.run(PARAMS, {"simulator": simulators, "image": images, "img_metrics": IMG_METRICS,
                               "composite": weights})
    values = result.img_metrics[0, :, :, 0].astype(float)
    assert result.composite[0] == pytest.approx((values[0, 0] + 3 * values[1, 1]) / 4)


def test_no_section_or_null_means_no_composite(tmp_path):
    for composite in (None, "absent"):
        config = request(composite)
        if composite == "absent":
            del config["composite"]
        result = simr.run(PARAMS, config, externals=EXTERNALS, output=tmp_path / f"{composite}.json")
        assert result.composite is None and result.composite_weights is None
        assert simr.load_catalog(tmp_path / f"{composite}.json")["runs"][0]["composite"] is None


def test_a_failed_simulation_gives_its_model_a_nan_composite():
    models = np.stack([FIRST_APPROACH, PARAMS])
    config = {**request({}, simulator=FAILING), "run": {"return": ["sim.orbit"]}}  # sim.orbit needs the start
    result = simr.run(models, config, externals=EXTERNALS)
    assert result.status["sim"].tolist() == [["failed"], ["ran"]]
    assert math.isnan(result.composite[0]) and math.isfinite(result.composite[1])


def test_a_pool_gives_the_same_composite():
    from concurrent.futures import ProcessPoolExecutor

    solo = simr.run(MODELS, request({}), externals=EXTERNALS)
    with ProcessPoolExecutor(max_workers=2) as pool:
        pooled = simr.run(MODELS, request({}), externals=EXTERNALS, map=pool.map)
    np.testing.assert_array_equal(solo.composite, pooled.composite)


def test_the_catalog_records_the_filled_in_weights(tmp_path):
    simr.run(MODELS, request({}, img_metrics=IMG_METRICS), externals=EXTERNALS, output=tmp_path)
    [record] = simr.load_catalog(tmp_path)["runs"]
    assert record["composite"] == {"sim_weights": [[1.0, 1.0, 0.0]], "img_weights": [[[1.0]]]}
    assert "composite" not in record["status"]


# --- MDRN-4 step 7a: metric values and the composite in the run record ---------------------------------


def test_the_record_holds_the_values_and_load_metrics_gives_them_back(tmp_path):
    result = simr.run(MODELS, request({}, img_metrics=IMG_METRICS), externals=EXTERNALS, output=tmp_path)
    [record] = simr.load_catalog(tmp_path)["runs"]
    values = record["values"]
    assert values["sim_metrics"][1][0][0] == result.sim_metrics[1, 0, 0]  # [model][simulator][entry]
    assert values["img_metrics"][1][0][0][0] == result.img_metrics[1, 0, 0, 0]
    assert values["composite"] == result.composite.tolist()
    assert values["sim_metrics"][0][0][2]["position"] == result.sim_metrics[0, 0, 2]["position"].tolist()

    loaded = simr.load_metrics(tmp_path)
    assert loaded["sim_metrics"].shape == (2, 1, 3) and loaded["img_metrics"].shape == (2, 1, 1, 1)
    np.testing.assert_array_equal(loaded["sim_metrics"][..., :2].astype(float),
                                  result.sim_metrics[..., :2].astype(float))
    np.testing.assert_array_equal(loaded["img_metrics"].astype(float), result.img_metrics.astype(float))
    np.testing.assert_array_equal(loaded["composite"], result.composite)
    assert loaded["composite"].dtype == np.float64
    approach, expected = loaded["sim_metrics"][1, 0, 2], result.sim_metrics[1, 0, 2]
    assert set(approach) == set(expected) and approach["t"] == expected["t"]
    np.testing.assert_array_equal(approach["position"], expected["position"])
    assert isinstance(approach["position"], np.ndarray)


def test_a_failed_scalar_is_null_on_disk_and_nan_when_loaded(tmp_path):
    models = np.stack([FIRST_APPROACH, PARAMS])
    config = {**request({}, simulator=FAILING), "run": {"return": ["sim.orbit"]}}
    simr.run(models, config, externals=EXTERNALS, output=tmp_path / "runs.json")
    text = (tmp_path / "runs.json").read_text(encoding="utf-8")
    assert "NaN" not in text  # valid JSON for every reader
    values = simr.load_catalog(tmp_path / "runs.json")["runs"][0]["values"]
    assert values["sim_metrics"][0][0] == [None, None, None] and values["composite"][0] is None
    loaded = simr.load_metrics(tmp_path / "runs.json")
    assert math.isnan(loaded["sim_metrics"][0, 0, 0]) and math.isnan(loaded["sim_metrics"][0, 0, 1])
    assert loaded["sim_metrics"][0, 0, 2] is None  # a descriptive entry's failed cell stays None
    assert math.isnan(loaded["composite"][0]) and math.isfinite(loaded["composite"][1])


def test_load_metrics_picks_a_run_and_handles_no_composite(tmp_path):
    simr.run(PARAMS, request(None), externals=EXTERNALS, output=tmp_path)
    simr.run(MODELS, request({}), externals=EXTERNALS, output=tmp_path)
    first, last = simr.load_metrics(tmp_path, run=0), simr.load_metrics(tmp_path)
    assert first["composite"] is None and first["sim_metrics"].shape == (1, 1, 3)
    assert last["composite"].shape == (2,) and last["img_metrics"].shape == (2, 1, 1, 0)


def test_load_metrics_says_when_a_run_recorded_no_values(tmp_path):
    simr.run(PARAMS, request(None), externals=EXTERNALS, output=tmp_path)
    catalog = simr.load_catalog(tmp_path)
    del catalog["runs"][0]["values"]  # as a run from before values were saved
    (tmp_path / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    with pytest.raises(ValueError, match="recorded no metric values"):
        simr.load_metrics(tmp_path)


def _never_simulate(*args):
    raise AssertionError("simulated before the config was checked")


@pytest.mark.parametrize(
    "composite, img_metrics, match",
    [
        ([0.5, 0.5], (), "composite must be a dict of weights"),
        ({"weights": [[1, 1, 0]]}, (), r"unknown keys in composite: \['weights'\]"),
        ({"sim_weights": [[1, 1, 0]], "img_weights": [[[1.0]]]}, (),
         "composite.img_weights: img_metrics has no entries"),
        ({"sim_weights": [[1, 1, 0]]}, IMG_METRICS, "composite.img_weights is missing"),
        ({"sim_weights": [1, 1, 0]}, (), r"shape \(simulator, sim_metrics entry\) = \(1, 3\): got shape \(3,\)"),
        ({"sim_weights": [[1, 1]]}, (), r"= \(1, 3\): got shape \(1, 2\)"),
        ({"sim_weights": [[1, 1], [1]]}, (), "got"),
        ({"sim_weights": [[1, True, 0]]}, (), r"composite.sim_weights\[0, 1\] must be a number: True"),
        ({"sim_weights": [[1, "1", 0]]}, (), "must be a number"),
        ({"sim_weights": [[1, -1, 0]]}, (), "finite and not negative"),
        ({"sim_weights": [[0, 0, 0]]}, (), "the weights are all zero"),
        ({"sim_weights": [[1, 1, 1]]}, (), r"sim_metrics\[2\] \(closest_approach\) gives a dict.*must be 0"),
    ],
)
def test_bad_composite_sections_stop_the_run_before_anything_is_simulated(composite, img_metrics, match,
                                                                          monkeypatch):
    monkeypatch.setitem(simr.SIM_DATA, "sim.search_orbit", _never_simulate)
    with pytest.raises(ValueError, match=match):
        simr.run(PARAMS, request(composite, img_metrics=img_metrics), externals=EXTERNALS)


def test_a_composite_needs_metric_entries():
    with pytest.raises(ValueError, match="no metric entries to combine"):
        simr.run(PARAMS, {"simulator": SMALL, "composite": {}})


def test_an_empty_section_with_only_dict_entries_is_an_error():
    with pytest.raises(ValueError, match="no metric entry gives one float"):
        simr.run(PARAMS, request({}, sim_metrics=[ORBIT_METRICS[2]]))
