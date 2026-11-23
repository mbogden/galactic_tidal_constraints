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

WINDOW_ENTRY = {"name": "target_win", "method": "recent_approach_cosine_window", "inputs": ["test_ext_orbit"]}
ORBIT_METRIC = {"method": "orbit_time_error", "inputs": ["test_ext_orbit", "sim.search_orbit", "drv.target_win"]}


def request(derived_entries, sim_metrics=(), img_metrics=()):
    return {"simulator": SMALL, "derived": derived_entries, "sim_metrics": list(sim_metrics),
            "img_metrics": list(img_metrics)}


def test_a_derived_window_weighs_an_orbit_metric():
    result = simr.run(MODELS, request([WINDOW_ENTRY], [ORBIT_METRIC]), externals={"test_ext_orbit": TARGET})
    window = derived.recent_approach_cosine_window(TARGET)
    heavier_path = sim.search_orbit(HEAVIER, SimulatorConfig(**SMALL)).astype(np.float32)
    assert result.status["sim_metrics"].tolist() == [[["ran"]], [["ran"]]]
    assert result.sim_metrics[0, 0, 0] == pytest.approx(0, abs=1e-9)
    assert result.sim_metrics[1, 0, 0] == metrics.orbit_time_error(TARGET, heavier_path, window)
    # The same values as giving the weights as an external
    as_external = simr.run(
        MODELS, {"simulator": SMALL, "sim_metrics": [{**ORBIT_METRIC, "inputs": ORBIT_METRIC["inputs"][:2] + ["w"]}]},
        externals={"test_ext_orbit": TARGET, "w": window},
    )
    assert as_external.sim_metrics.tolist() == result.sim_metrics.tolist()


def test_the_filled_in_entry_records_its_settings_level_shape_and_hash(tmp_path):
    entry = {**WINDOW_ENTRY, "params": {"k": 0.5}}
    result = simr.run(PARAMS, request([entry], [ORBIT_METRIC]), output=tmp_path,
                      externals={"test_ext_orbit": TARGET})
    window = derived.recent_approach_cosine_window(TARGET, k=0.5)
    assert result.derived_configs == [{
        "name": "target_win", "method": "recent_approach_cosine_window", "inputs": ["test_ext_orbit"],
        "params": {"k": 0.5, "flat": 0.0, "pass_number": 1}, "level": "run",
        "shape": [len(TARGET)], "hash": simr.array_hash(window),
    }]
    [record] = simr.load_catalog(tmp_path)["runs"]
    assert record["derived"] == result.derived_configs
    assert list(result.externals) == ["test_ext_orbit"]  # the derived array isn't listed as an external


def test_no_derived_section_means_none(tmp_path):
    result = simr.run(PARAMS, {"simulator": SMALL}, output=tmp_path)
    assert result.derived_configs == []
    assert simr.load_catalog(tmp_path)["runs"][0]["derived"] == []
    assert simr.run(PARAMS, {"simulator": SMALL, "derived": []}).derived_configs == []


def test_an_entry_can_use_entries_above_it_and_serve_as_data_or_weights():
    entries = [
        {"name": "changed", "method": "difference_weights", "inputs": ["test_ext_image", "test_ext_other"]},
        {"name": "changed_edges", "method": "detect_edges", "inputs": ["drv.changed"]},
        {"name": "target_edges", "method": "detect_edges", "inputs": ["test_ext_image"]},
    ]
    img_metrics = [
        {"method": "absolute_difference", "inputs": ["test_ext_image", "img.model", "drv.changed"]},  # as weights
        {"method": "absolute_difference", "inputs": ["drv.target_edges", "img.model"]},  # as data
        {"method": "absolute_difference", "inputs": ["drv.changed_edges", "img.model"]},
    ]
    request_ = {**request(entries, img_metrics=img_metrics), "run": {"return": ["img.model"]}}
    result = simr.run(PARAMS, request_, externals={"test_ext_image": TARGET_IMAGE, "test_ext_other": OTHER_IMAGE})
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

    solo = simr.run(MODELS, request([WINDOW_ENTRY], [ORBIT_METRIC]), externals={"test_ext_orbit": TARGET})
    with ProcessPoolExecutor(max_workers=2) as pool, caplog.at_level("INFO"):
        pooled = simr.run(MODELS, request([WINDOW_ENTRY], [ORBIT_METRIC]), externals={"test_ext_orbit": TARGET},
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
          {"name": "later", "method": "detect_edges", "inputs": ["test_ext_image"]}],
         r"derived\[0\].inputs\[0\]: 'drv.later' isn't a derived entry above this one. Above: \[\]"),
        ([{"name": "e", "method": "detect_edges", "inputs": ["drv.e"]}], "isn't a derived entry above this one"),
        ([WINDOW_ENTRY, {"name": "e", "method": "detect_edges", "inputs": ["drv.target_win"]}],
         r"derived\[1\].inputs\[0\]: detect_edges takes a image here, but drv.target_win is a weights"),
        ([{"name": "e", "method": "detect_edges", "inputs": ["sim.timing"]}], "'sim.timing' isn't an input"),
    ],
)
def test_bad_derived_entries_name_the_problem(entries, message):
    with pytest.raises(ValueError, match=message):
        simr.run(PARAMS, request(entries), externals={"test_ext_orbit": TARGET, "test_ext_image": TARGET_IMAGE})


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
        simr.run(PARAMS, request([WINDOW_ENTRY], [metric]), externals={"test_ext_orbit": TARGET})


def test_a_derived_method_that_fails_stops_the_run_before_anything_is_simulated(monkeypatch):
    monkeypatch.setitem(simr.SIM_DATA, "sim.search_orbit", lambda params, config: 1 / 0)
    early_part = TARGET[:200].copy()  # only an old pass, too near the orbit's start for the window to fit
    with pytest.raises(ValueError, match=r"derived\[0\] \('target_win'\): the window would start at t = "):
        simr.run(PARAMS, request([WINDOW_ENTRY], [ORBIT_METRIC]), externals={"test_ext_orbit": early_part})
    with pytest.raises(ValueError, match=r"derived\[0\].inputs\[0\]: 'test_ext_orbit' isn't a simr item, so it must "
                                         r"be a key in externals"):
        simr.run(PARAMS, request([WINDOW_ENTRY], [ORBIT_METRIC]))


def test_a_derived_image_a_metric_uses_must_match_the_image_entries():
    small = np.random.default_rng(11).random((64, 64)).astype(np.float32)
    entries = [{"name": "edges", "method": "detect_edges", "inputs": ["test_ext_small"]}]
    metric = {"method": "absolute_difference", "inputs": ["drv.edges", "img.model"]}
    with pytest.raises(ValueError, match="64 x 64 .height x width., but image.0. makes 128 x 128 images"):
        simr.run(PARAMS, request(entries, img_metrics=[metric]), externals={"test_ext_small": small})


def test_entries_made_from_simulated_data_are_the_next_step():
    entry = {"name": "model_win", "method": "recent_approach_cosine_window", "inputs": ["sim.search_orbit"]}
    with pytest.raises(NotImplementedError, match=r"derived\[0\] \('model_win'\) is made from sim.\* or img.\* items"):
        simr.run(PARAMS, request([entry]))


def test_an_externals_key_cannot_start_with_drv():
    with pytest.raises(ValueError, match="a key must be a string that doesn't start with"):
        simr.run(PARAMS, request([WINDOW_ENTRY], [ORBIT_METRIC]),
                 externals={"test_ext_orbit": TARGET, "drv.mine": TARGET})
