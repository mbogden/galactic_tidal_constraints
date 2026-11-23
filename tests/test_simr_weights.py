"""Tests for metrics with weights through simr: the orbit metrics against an external target, and
weights given as externals (MDRN-4 step 4b)."""

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
PATH = sim.search_orbit(PARAMS, SimulatorConfig(**SMALL)).astype(np.float32)
TARGET = PATH[::-1][::50].copy()
WINDOW = derived.recent_approach_cosine_window(TARGET)
INPUTS = ["test_ext_orbit", "sim.search_orbit"]


def request(*entries, section="sim_metrics", returned=()):
    return {"run": {"return": list(returned)}, "simulator": SMALL, section: list(entries)}


def test_the_orbit_metrics_score_the_targets_own_model_zero_and_another_above():
    result = simr.run(
        MODELS,
        request({"method": "orbit_time_error", "inputs": INPUTS + ["test_ext_weights"]},
                {"method": "orbit_position_error", "inputs": INPUTS + ["test_ext_weights"]}),
        externals={"test_ext_orbit": TARGET, "test_ext_weights": WINDOW},
    )
    assert result.status["sim_metrics"].tolist() == [[["ran", "ran"]], [["ran", "ran"]]]
    own, other = result.sim_metrics[0, 0], result.sim_metrics[1, 0]
    assert own[0] == pytest.approx(0, abs=1e-9) and own[1] == pytest.approx(0, abs=1e-9)
    assert other[0] > 0.01 and other[1] > 0.001
    assert other[0] >= other[1]  # being in the wrong place at each time costs at least the path's distance
    heavier_path = sim.search_orbit(HEAVIER, SimulatorConfig(**SMALL)).astype(np.float32)
    assert other[0] == metrics.orbit_time_error(TARGET, heavier_path, WINDOW)


def test_weights_are_optional_and_change_the_value():
    result = simr.run(
        HEAVIER,
        request({"method": "orbit_time_error", "inputs": INPUTS},
                {"method": "orbit_time_error", "inputs": INPUTS + ["test_ext_weights"]}),
        externals={"test_ext_orbit": TARGET, "test_ext_weights": WINDOW},
    )
    plain, weighted = result.sim_metrics[0, 0]
    assert plain > 0 and weighted > 0 and plain != weighted
    assert result.sim_metric_configs[0]["inputs"] == INPUTS
    assert result.sim_metric_configs[1]["inputs"] == INPUTS + ["test_ext_weights"]
    assert result.externals["test_ext_weights"]["shape"] == [len(TARGET)]


def test_weights_of_the_wrong_length_stop_the_run_before_anything_is_simulated(monkeypatch):
    monkeypatch.setitem(simr.SIM_DATA, "sim.search_orbit", lambda params, config: 1 / 0)
    with pytest.raises(ValueError, match=r"sim_metrics\[0\]: the weights 'test_ext_weights' \(for 'test_ext_orbit'\) "
                                         r"must be one per row of the first input, \d+: got \(5,\)"):
        simr.run(PARAMS, request({"method": "orbit_time_error", "inputs": INPUTS + ["test_ext_weights"]}),
                 externals={"test_ext_orbit": TARGET, "test_ext_weights": np.ones(5)})


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"method": "orbit_time_error", "inputs": INPUTS + ["sim.particles"]},
         "takes a weights here, but sim.particles is a particles"),
        ({"method": "orbit_time_error", "inputs": INPUTS + ["test_ext_weights", "test_ext_weights"]},
         "takes 2 input.s., one per kind: .'orbit', 'orbit'., and optionally its weights .a weights."),
        ({"method": "closest_approach", "inputs": ["sim.search_orbit", "test_ext_weights"]},
         r"closest_approach takes 1 input\(s\), one per kind: \['orbit'\]$"),
    ],
)
def test_bad_weight_inputs_name_the_problem(entry, message):
    with pytest.raises(ValueError, match=message):
        simr.run(PARAMS, request(entry), externals={"test_ext_orbit": TARGET, "test_ext_weights": WINDOW})


def test_an_external_used_as_weights_and_as_an_orbit_has_two_kinds():
    with pytest.raises(ValueError, match="used as a weights here but as a orbit"):
        simr.run(PARAMS, request({"method": "orbit_time_error", "inputs": INPUTS + ["test_ext_orbit"]}),
                 externals={"test_ext_orbit": TARGET})


def test_an_image_metric_takes_a_weight_image():
    target_image = np.random.default_rng(8).random((128, 128)).astype(np.float32)
    mask = np.zeros((128, 128), dtype=np.float32)
    mask[40:90, 20:110] = 1.0
    result = simr.run(
        PARAMS,
        request({"method": "absolute_difference", "inputs": ["test_ext_image", "img.model"]},
                {"method": "absolute_difference", "inputs": ["test_ext_image", "img.model", "test_ext_mask"]},
                section="img_metrics", returned=["img.model"]),
        externals={"test_ext_image": target_image, "test_ext_mask": mask},
    )
    plain, weighted = result.img_metrics[0, 0, 0]
    image = result.data["img.model"][0, 0, 0]
    assert plain == metrics.absolute_difference(target_image, image)
    assert weighted == metrics.absolute_difference(target_image, image, mask) != plain


def test_all_zero_weights_known_before_the_run_stop_it_before_anything_is_simulated(monkeypatch):
    # Both the weights and the target they weigh are externals, so every model's cell would fail the same way.
    monkeypatch.setitem(simr.SIM_DATA, "sim.search_orbit", lambda params, config: 1 / 0)
    with pytest.raises(ValueError, match=r"sim_metrics\[0\]: the weights 'test_ext_weights' \(for 'test_ext_orbit'\) "
                                         r"are all zero"):
        simr.run(
            PARAMS,
            request({"method": "orbit_time_error", "inputs": INPUTS + ["test_ext_weights"]},
                    {"method": "orbit_time_error", "inputs": INPUTS}),
            externals={"test_ext_orbit": TARGET, "test_ext_weights": np.zeros(len(TARGET))},
        )


def test_all_zero_weights_made_per_model_fail_that_cell_and_nothing_else():
    # A weight image made from the model's own image (zero everywhere: it differs from itself nowhere).
    entry = {"name": "nothing", "method": "difference_weights", "inputs": ["img.model", "img.model"],
             "params": {"floor": 0.0}}
    result = simr.run(
        PARAMS,
        {"simulator": SMALL, "derived": [entry],
         "img_metrics": [{"method": "absolute_difference", "inputs": ["img.model", "img.unperturbed", "drv.nothing"]},
                         {"method": "absolute_difference", "inputs": ["img.model", "img.unperturbed"]}]},
    )
    assert result.status["img"].tolist() == [[["ran"]]]
    assert result.status["img_metrics"].tolist() == [[[["failed", "ran"]]]]
    assert "the weights are all zero" in result.errors["img_metrics"][0, 0, 0, 0]
