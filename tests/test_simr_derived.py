"""Tests for the derived section in simr: entries made from externals, computed once for the run
(MDRN-4 step 4c)."""

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

# A stand-in target: the first model's own path, every 50th row, in increasing time.
TARGET = sim.search_orbit(PARAMS, SimulatorConfig(**SMALL)).astype(np.float32)[::-1][::50].copy()
TARGET_IMAGE = np.random.default_rng(9).random((128, 128)).astype(np.float32)
OTHER_IMAGE = np.random.default_rng(10).random((128, 128)).astype(np.float32)

WINDOW_ENTRY = {"name": "target_win", "method": "recent_approach_cosine_window", "inputs": ["ext.test_orbit"]}
ORBIT_METRIC = {"method": "orbit_time_error", "inputs": ["ext.test_orbit", "sim.search_orbit", "drv.target_win"]}


def request(derived_entries, sim_metrics=(), img_metrics=()):
    return {"simulator": SMALL, "derived": derived_entries, "sim_metrics": list(sim_metrics),
            "img_metrics": list(img_metrics)}


def test_a_derived_window_weighs_an_orbit_metric():
    result = simr.run(MODELS, request([WINDOW_ENTRY], [ORBIT_METRIC]), externals={"test_orbit": TARGET})
    window = derived.recent_approach_cosine_window(TARGET)
    heavier_path = sim.search_orbit(HEAVIER, SimulatorConfig(**SMALL)).astype(np.float32)
    assert result.status["sim_metrics"].tolist() == [[["ran"]], [["ran"]]]
    assert result.sim_metrics[0, 0, 0] == pytest.approx(0, abs=1e-9)
    assert result.sim_metrics[1, 0, 0] == metrics.orbit_time_error(TARGET, heavier_path, window)
    # The same values as giving the weights as an external
    as_external = simr.run(
        MODELS, {"simulator": SMALL, "sim_metrics": [{**ORBIT_METRIC, "inputs": ORBIT_METRIC["inputs"][:2] + ["ext.w"]}]},
        externals={"test_orbit": TARGET, "w": window},
    )
    assert as_external.sim_metrics.tolist() == result.sim_metrics.tolist()


def test_the_filled_in_entry_records_its_settings_level_shape_and_hash(tmp_path):
    entry = {**WINDOW_ENTRY, "params": {"k": 0.5}}
    result = simr.run(PARAMS, request([entry], [ORBIT_METRIC]), output=tmp_path,
                      externals={"test_orbit": TARGET})
    window = derived.recent_approach_cosine_window(TARGET, k=0.5)
    assert result.derived_configs == [{
        "name": "target_win", "method": "recent_approach_cosine_window", "inputs": ["ext.test_orbit"],
        "params": {"k": 0.5, "flat": 0.0, "pass_number": 1}, "level": "run",
        "shape": [len(TARGET)], "hash": simr.array_hash(window),
    }]
    [record] = simr.load_catalog(tmp_path)["runs"]
    assert record["derived"] == result.derived_configs
    assert list(result.externals) == ["test_orbit"]  # the derived array isn't listed as an external


def test_no_derived_section_means_none(tmp_path):
    result = simr.run(PARAMS, {"simulator": SMALL}, output=tmp_path)
    assert result.derived_configs == []
    assert simr.load_catalog(tmp_path)["runs"][0]["derived"] == []
    assert simr.run(PARAMS, {"simulator": SMALL, "derived": []}).derived_configs == []


def test_an_entry_can_use_entries_above_it_and_serve_as_data_or_weights():
    entries = [
        {"name": "changed", "method": "difference_weights", "inputs": ["ext.test_image", "ext.test_other"]},
        {"name": "changed_edges", "method": "detect_edges", "inputs": ["drv.changed"]},
        {"name": "target_edges", "method": "detect_edges", "inputs": ["ext.test_image"]},
    ]
    img_metrics = [
        {"method": "absolute_difference", "inputs": ["ext.test_image", "img.model", "drv.changed"]},  # as weights
        {"method": "absolute_difference", "inputs": ["drv.target_edges", "img.model"]},  # as data
        {"method": "absolute_difference", "inputs": ["drv.changed_edges", "img.model"]},
    ]
    request_ = {**request(entries, img_metrics=img_metrics), "run": {"return": ["img.model"]}}
    result = simr.run(PARAMS, request_, externals={"test_image": TARGET_IMAGE, "test_other": OTHER_IMAGE})
    image = result.data["img.model"][0, 0, 0]
    changed = derived.difference_weights(TARGET_IMAGE, OTHER_IMAGE)
    assert result.img_metrics[0, 0, 0].tolist() == [
        metrics.absolute_difference(TARGET_IMAGE, image, changed),
        metrics.absolute_difference(derived.detect_edges(TARGET_IMAGE), image),
        metrics.absolute_difference(derived.detect_edges(changed), image),
    ]
    assert [entry["level"] for entry in result.derived_configs] == ["run", "run", "run"]


def test_derived_arrays_reach_a_pools_workers(caplog):
    from concurrent.futures import ProcessPoolExecutor

    solo = simr.run(MODELS, request([WINDOW_ENTRY], [ORBIT_METRIC]), externals={"test_orbit": TARGET})
    with ProcessPoolExecutor(max_workers=2) as pool, caplog.at_level("INFO"):
        pooled = simr.run(MODELS, request([WINDOW_ENTRY], [ORBIT_METRIC]), externals={"test_orbit": TARGET},
                          map=pool.map)
    assert pooled.sim_metrics.tolist() == solo.sim_metrics.tolist()
    assert "derived 'target_win' (recent_approach_cosine_window): shape" in caplog.text
    assert "'drv.target_win'" in caplog.text and "shared with the workers" in caplog.text


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ("recent_approach_cosine_window", "derived must be an entry .a dict. or a list of them"),
        (["target_win"], r"derived\[0\] must be a dict with a name, a method and inputs"),
        ([{"method": "detect_edges", "inputs": ["x"]}], r'derived\[0\] needs a "name"'),
        ([{**WINDOW_ENTRY, "name": "target win"}], r'derived\[0\] needs a "name"'),
        ([{**WINDOW_ENTRY, "name": "drv.x"}], r'derived\[0\] needs a "name"'),
        ([WINDOW_ENTRY, WINDOW_ENTRY], r"derived\[1\]: the name 'target_win' is already used by an entry above"),
        ([{**WINDOW_ENTRY, "method": "window"}], r"derived\[0\].method: unknown method 'window'. Known: "),
        ([{**WINDOW_ENTRY, "inputs": []}], r"recent_approach_cosine_window takes 1 input\(s\)"),
        ([{**WINDOW_ENTRY, "params": {"width": 2}}], r"derived\[0\].params: unknown settings \['width'\]"),
        ([{**WINDOW_ENTRY, "extra": 1}], r"unknown keys in derived\[0\]"),
        ([{"name": "e", "method": "detect_edges", "inputs": ["drv.later"]},
          {"name": "later", "method": "detect_edges", "inputs": ["ext.test_image"]}],
         r"derived\[0\].inputs\[0\]: 'drv.later' isn't a derived entry above this one. Above: \[\]"),
        ([{"name": "e", "method": "detect_edges", "inputs": ["drv.e"]}], "isn't a derived entry above this one"),
        ([WINDOW_ENTRY, {"name": "e", "method": "detect_edges", "inputs": ["drv.target_win"]}],
         r"derived\[1\].inputs\[0\]: detect_edges takes a image here, but drv.target_win is a weights"),
        ([{"name": "e", "method": "detect_edges", "inputs": ["sim.timing"]}], "'sim.timing' isn't an input"),
    ],
)
def test_bad_derived_entries_name_the_problem(entries, message):
    with pytest.raises(ValueError, match=message):
        simr.run(PARAMS, request(entries), externals={"test_orbit": TARGET, "test_image": TARGET_IMAGE})


@pytest.mark.parametrize(
    ("metric", "message"),
    [
        ({**ORBIT_METRIC, "inputs": ORBIT_METRIC["inputs"][:2] + ["drv.missing"]},
         r"'drv.missing' isn't an entry in the derived section, which has \['drv.target_win'\]"),
        ({"method": "orbit_time_error", "inputs": ["drv.target_win", "sim.search_orbit"]},
         "orbit_time_error takes a orbit here, but drv.target_win is a weights"),
    ],
)
def test_a_metric_naming_a_derived_entry_wrongly_names_the_problem(metric, message):
    with pytest.raises(ValueError, match=message):
        simr.run(PARAMS, request([WINDOW_ENTRY], [metric]), externals={"test_orbit": TARGET})


def test_a_derived_method_that_fails_stops_the_run_before_anything_is_simulated(monkeypatch):
    monkeypatch.setitem(simr.SIM_DATA, "sim.search_orbit", lambda params, config: 1 / 0)
    early_part = TARGET[:200].copy()  # only an old pass, too near the orbit's start for the window to fit
    with pytest.raises(ValueError, match=r"derived\[0\] \('target_win'\): the window would start at t = "):
        simr.run(PARAMS, request([WINDOW_ENTRY], [ORBIT_METRIC]), externals={"test_orbit": early_part})
    with pytest.raises(ValueError, match=r"derived\[0\].inputs\[0\]: ext.test_orbit names an external the run wasn't given"):
        simr.run(PARAMS, request([WINDOW_ENTRY], [ORBIT_METRIC]))


def test_a_derived_image_a_metric_uses_must_match_the_image_entries():
    small = np.random.default_rng(11).random((64, 64)).astype(np.float32)
    entries = [{"name": "edges", "method": "detect_edges", "inputs": ["ext.test_small"]}]
    metric = {"method": "absolute_difference", "inputs": ["drv.edges", "img.model"]}
    with pytest.raises(ValueError, match="64 x 64 .height x width., but image.0. makes 128 x 128 images"):
        simr.run(PARAMS, request(entries, img_metrics=[metric]), externals={"test_small": small})


# --- Step 4d: entries made from simulated data, and run.return ------------------------------------------

TIDAL = {"name": "tidal", "method": "difference_weights", "inputs": ["ext.test_image", "img.unperturbed"]}
MODEL_WIN = {"name": "model_win", "method": "recent_approach_cosine_window", "inputs": ["sim.search_orbit"]}
# The first model with its velocity reversed and tripled: its only pass is too long ago for the window to fit.
APPROACHING = PARAMS.copy()
APPROACHING[3:6] *= -3


def returning(request_, *names):
    return {**request_, "run": {"return": list(names)}}


def test_a_per_image_entry_weighs_an_image_metric_with_the_models_own_unperturbed_image():
    metric = {"method": "absolute_difference", "inputs": ["ext.test_image", "img.model", "drv.tidal"]}
    result = simr.run(MODELS, returning(request([TIDAL], img_metrics=[metric]),
                                        "img.model", "img.unperturbed", "drv.tidal"),
                      externals={"test_image": TARGET_IMAGE})
    assert result.derived_configs[0]["level"] == "img" and "hash" not in result.derived_configs[0]
    assert result.data["drv.tidal"].shape == (2, 1, 1)  # indexed like the images it's made from
    for m in range(2):
        model, unperturbed = result.data["img.model"][m, 0, 0], result.data["img.unperturbed"][m, 0, 0]
        weights = derived.difference_weights(TARGET_IMAGE, unperturbed)
        assert (result.data["drv.tidal"][m, 0, 0] == weights).all()
        assert result.img_metrics[m, 0, 0, 0] == metrics.absolute_difference(TARGET_IMAGE, model, weights)


def test_a_per_simulation_entry_comes_back_indexed_like_the_simulations():
    result = simr.run(MODELS, returning(request([MODEL_WIN]), "sim.search_orbit", "drv.model_win"))
    assert result.derived_configs[0]["level"] == "sim"
    assert result.data["drv.model_win"].shape == (2, 1)
    for m in range(2):
        path = result.data["sim.search_orbit"][m, 0]
        assert (result.data["drv.model_win"][m, 0] == derived.recent_approach_cosine_window(path)).all()
    assert result.status["sim"].tolist() == [["ran"], ["ran"]]  # computed even with no metric using it


def test_a_run_level_entry_comes_back_as_the_one_array():
    from concurrent.futures import ProcessPoolExecutor

    request_ = returning(request([WINDOW_ENTRY], [ORBIT_METRIC]), "drv.target_win")
    solo = simr.run(MODELS, request_, externals={"test_orbit": TARGET})
    window = derived.recent_approach_cosine_window(TARGET)
    assert solo.data["drv.target_win"].shape == window.shape and (solo.data["drv.target_win"] == window).all()
    with ProcessPoolExecutor(max_workers=2) as pool:
        pooled = simr.run(MODELS, request_, externals={"test_orbit": TARGET}, map=pool.map)
    assert (pooled.data["drv.target_win"] == window).all()


def test_entries_chain_across_levels_and_serve_as_data():
    entries = [
        {"name": "target_edges", "method": "detect_edges", "inputs": ["ext.test_image"]},  # once
        {"name": "model_edges", "method": "detect_edges", "inputs": ["img.model"]},  # per image
        TIDAL,
        {"name": "tidal_edges", "method": "detect_edges", "inputs": ["drv.tidal"]},  # per image, from an entry above
    ]
    metric = {"method": "absolute_difference", "inputs": ["drv.target_edges", "drv.model_edges", "drv.tidal_edges"]}
    result = simr.run(PARAMS, returning(request(entries, img_metrics=[metric]), "img.model", "img.unperturbed"),
                      externals={"test_image": TARGET_IMAGE})
    assert [entry["level"] for entry in result.derived_configs] == ["run", "img", "img", "img"]
    model, unperturbed = result.data["img.model"][0, 0, 0], result.data["img.unperturbed"][0, 0, 0]
    expected = metrics.absolute_difference(
        derived.detect_edges(TARGET_IMAGE), derived.detect_edges(model),
        derived.detect_edges(derived.difference_weights(TARGET_IMAGE, unperturbed)),
    )
    assert result.img_metrics[0, 0, 0, 0] == expected


def test_per_model_entries_match_through_a_pool():
    from concurrent.futures import ProcessPoolExecutor

    metric = {"method": "absolute_difference", "inputs": ["ext.test_image", "img.model", "drv.tidal"]}
    request_ = returning(request([TIDAL], img_metrics=[metric]), "drv.tidal")
    solo = simr.run(MODELS, request_, externals={"test_image": TARGET_IMAGE})
    with ProcessPoolExecutor(max_workers=2) as pool:
        pooled = simr.run(MODELS, request_, externals={"test_image": TARGET_IMAGE}, map=pool.map)
    assert pooled.img_metrics.tolist() == solo.img_metrics.tolist()
    for m in range(2):
        assert (pooled.data["drv.tidal"][m, 0, 0] == solo.data["drv.tidal"][m, 0, 0]).all()


def test_a_failed_entry_fails_only_what_uses_it_for_that_model():
    metrics_ = [
        {"method": "orbit_time_error", "inputs": ["sim.search_orbit", "sim.search_orbit", "drv.model_win"]},
        {"method": "closest_approach", "inputs": ["sim.search_orbit"], "select": "distance"},
    ]
    result = simr.run(np.stack([PARAMS, APPROACHING]), returning(request([MODEL_WIN], metrics_), "drv.model_win"))
    assert result.status["sim"].tolist() == [["ran"], ["ran"]]  # the simulations are unaffected
    assert result.data["drv.model_win"][0, 0] is not None and result.data["drv.model_win"][1, 0] is None
    assert result.status["sim_metrics"][1, 0].tolist() == ["failed", "ran"]  # only the metric that uses it
    assert "its input drv.model_win failed (ValueError: the window would start" in result.errors["sim_metrics"][1, 0, 0]


@pytest.mark.parametrize(
    ("section", "metric", "message"),
    [
        ("sim_metrics", {"method": "absolute_difference", "inputs": ["ext.test_image", "drv.tidal"]},
         "drv.tidal is made from img.* items, so it belongs in img_metrics, not sim_metrics"),
        ("img_metrics", {"method": "absolute_difference", "inputs": ["ext.test_image", "drv.target_edges"]},
         "no simr item among the inputs"),  # both are fixed for the run: nothing depends on the model
        ("img_metrics", {"method": "orbit_time_error",
                         "inputs": ["ext.test_orbit", "sim.search_orbit", "drv.model_win"]},
         "no img.* input; a metric that uses only sim.* items belongs in sim_metrics"),
    ],
)
def test_a_metrics_level_follows_what_its_derived_inputs_are_made_from(section, metric, message):
    entries = [TIDAL, MODEL_WIN, {"name": "target_edges", "method": "detect_edges", "inputs": ["ext.test_image"]}]
    with pytest.raises(ValueError, match=message.replace("*", r"\*")):
        simr.run(PARAMS, {"simulator": SMALL, "derived": entries, section: [metric]},
                 externals={"test_image": TARGET_IMAGE, "test_orbit": TARGET})


def test_a_metric_made_only_from_per_model_derived_entries_is_allowed():
    entries = [{"name": "model_edges", "method": "detect_edges", "inputs": ["img.model"]},
               {"name": "plain_edges", "method": "detect_edges", "inputs": ["img.unperturbed"]}]
    metric = {"method": "absolute_difference", "inputs": ["drv.model_edges", "drv.plain_edges"]}
    result = simr.run(PARAMS, request(entries, img_metrics=[metric]))
    assert result.status["img_metrics"].tolist() == [[[["ran"]]]]
    assert 0 < result.img_metrics[0, 0, 0, 0] <= 1


def test_derived_names_in_run_return_and_run_save_are_checked():
    with pytest.raises(ValueError, match=r"unknown names in run.return: \['drv.nope'\]"):
        simr.run(PARAMS, returning(request([WINDOW_ENTRY]), "drv.nope"), externals={"test_orbit": TARGET})
    with pytest.raises(ValueError, match="run.save: derived arrays can't be saved yet"):
        simr.run(PARAMS, {**request([WINDOW_ENTRY]), "run": {"save": ["drv.target_win"]}},
                 externals={"test_orbit": TARGET})


def test_an_externals_key_is_letters_digits_and_underscores():
    with pytest.raises(ValueError, match=r"a key must be letters, digits and _ .* 'drv.mine'"):
        simr.run(PARAMS, request([WINDOW_ENTRY], [ORBIT_METRIC]),
                 externals={"test_orbit": TARGET, "drv.mine": TARGET})
