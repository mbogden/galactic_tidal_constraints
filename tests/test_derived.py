"""Tests for the Derived stage (simr/derived.py): the registry and each method on its own."""

import pytest

np = pytest.importorskip("numpy")

from galactic_tidal_constraints.simr import derived, metrics  # noqa: E402


def orbit_from(times, distances):
    """An orbit along x with the given times and distances from the primary: shape (n, 7), float32."""
    orbit = np.zeros((len(times), 7), dtype=np.float32)
    orbit[:, 0] = distances
    orbit[:, 6] = times
    return orbit


# One pass at t = -2 (distance 1), from t = -6 to now.
TIMES = np.linspace(-6, 0, 61)
ONE_PASS = orbit_from(TIMES, 1 + (TIMES + 2) ** 2)
window = derived.recent_approach_cosine_window


# --- The registry ----------------------------------------------------------------------------------------


def test_the_table_lists_each_method_with_its_kinds_and_settings():
    assert derived.table() == {
        "recent_approach_cosine_window": {"inputs": ["orbit"], "output": "weights",
                                          "settings": {"k": 1.0, "flat": 0.0, "pass_number": 1}},
        "difference_weights": {"inputs": ["image", "image"], "output": "image",
                               "settings": {"floor": 0.05, "threshold": None}},
        "detect_edges": {"inputs": ["image"], "output": "image", "settings": {"threshold": None}},
    }


def test_every_declared_setting_is_an_argument_with_that_default():
    import inspect

    for name, method in derived.METHODS.items():
        parameters = list(inspect.signature(method.function).parameters.values())
        assert len(parameters) == len(method.inputs) + len(method.settings), name
        assert {p.name: p.default for p in parameters[len(method.inputs):]} == method.settings, name


def test_a_declaration_with_an_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown kind 'picture'"):
        derived.derived(inputs=("picture",), output="image")(lambda a: a)
    assert "<lambda>" not in derived.METHODS


def test_compute_runs_a_method_by_name_and_returns_float32():
    weights = derived.compute("recent_approach_cosine_window", [ONE_PASS], {"k": 1.0})
    assert weights.dtype == np.float32 and weights.shape == (61,)
    assert (weights == window(ONE_PASS)).all()


def test_compute_checks_the_inputs_and_the_result(monkeypatch):
    from dataclasses import replace

    with pytest.raises(ValueError, match=r"detect_edges: input 0 must be a 'image' array.*got \(7,\)"):
        derived.compute("detect_edges", [ONE_PASS[0]])  # one row: 1-D, not an image
    original = derived.METHODS["detect_edges"]
    monkeypatch.setitem(derived.METHODS, "detect_edges", replace(original, function=lambda image, **s: image[0]))
    with pytest.raises(ValueError, match=r"detect_edges: the result must be a 'image' array.*got \(4,\)"):
        derived.compute("detect_edges", [np.zeros((4, 4), dtype=np.float32)])
    monkeypatch.setitem(derived.METHODS, "detect_edges", replace(original, function=lambda image, **s: image * np.nan))
    with pytest.raises(ValueError, match="the result holds NaN or inf"):
        derived.compute("detect_edges", [np.zeros((4, 4), dtype=np.float32)])


def test_weights_are_an_input_kind_one_value_per_row():
    metrics.check_input("weights", np.ones(5, dtype=np.float32), "w")
    with pytest.raises(ValueError, match=r"w must be a 'weights' array, shape \(None,\): got \(5, 1\)"):
        metrics.check_input("weights", np.ones((5, 1), dtype=np.float32), "w")


# --- recent_approach_cosine_window -----------------------------------------------------------------------


def test_the_window_is_1_at_the_pass_and_0_at_both_ends():
    weights = window(ONE_PASS)  # peak -2, now 0, k = 1: start -4
    assert weights.dtype == np.float32 and weights.shape == (61,)
    at = {round(float(t), 1): float(w) for t, w in zip(TIMES, weights)}
    assert at[-2.0] == 1.0
    assert at[0.0] == pytest.approx(0, abs=1e-6) and at[-4.0] == pytest.approx(0, abs=1e-6)
    assert at[-1.0] == pytest.approx(0.5) and at[-3.0] == pytest.approx(0.5)  # halfway down each side
    assert all(at[t] == 0 for t in (-6.0, -5.0, -4.5))  # nothing before the start
    assert ((weights >= 0) & (weights <= 1)).all()


def test_k_stretches_only_the_early_side():
    weights = window(ONE_PASS, k=2.0)  # start -6
    at = {round(float(t), 1): float(w) for t, w in zip(TIMES, weights)}
    assert at[-2.0] == 1.0 and at[-4.0] == pytest.approx(0.5) and at[-6.0] == pytest.approx(0, abs=1e-6)
    assert at[-1.0] == pytest.approx(0.5)  # the late side is unchanged


def test_flat_keeps_a_fraction_of_each_side_at_1():
    weights = window(ONE_PASS, flat=0.5)  # each side is 2 long: 1 flat, then 1 of cosine
    at = {round(float(t), 1): float(w) for t, w in zip(TIMES, weights)}
    assert all(at[t] == 1.0 for t in (-3.0, -2.5, -2.0, -1.5, -1.0))
    assert at[-0.5] == pytest.approx(0.5) and at[-3.5] == pytest.approx(0.5)
    assert at[0.0] == pytest.approx(0, abs=1e-6)
    box = window(ONE_PASS, flat=1.0)
    assert set(np.unique(box)) == {0.0, 1.0} and box[TIMES >= -4].all() and not box[TIMES < -4 - 1e-6].any()


def test_the_peak_is_the_most_recent_pass_and_pass_number_picks_an_older_one():
    times = np.linspace(-20, 0, 201)
    # Passes near -15 and -3; the older one is the closer of the two, and still isn't the default.
    distance = np.minimum(0.5 + (times + 15) ** 2, 2 + (times + 3) ** 2)
    two_passes = orbit_from(times, distance)
    recent = window(two_passes)
    assert times[np.argmax(recent)] == pytest.approx(-3)
    older = window(two_passes, pass_number=2, k=0.2)
    assert times[np.argmax(older)] == pytest.approx(-15)
    with pytest.raises(ValueError, match="pass 3 isn't in the orbit: it has 2 pass"):
        window(two_passes, pass_number=3)


def test_the_weights_follow_the_rows_own_order():
    backward = ONE_PASS[::-1].copy()  # as sim.search_orbit runs: from now back in time
    assert (window(backward) == window(ONE_PASS)[::-1]).all()


def test_uneven_spacing_is_fine():
    times = np.array([-6, -5, -4, -3.5, -3, -2.5, -2.25, -2, -1.5, -1, -0.2, 0])
    weights = window(orbit_from(times, 1 + (times + 2) ** 2))
    assert weights[7] == 1.0 and weights[8] == pytest.approx(0.5 * (1 + np.cos(np.pi * 0.25)))


@pytest.mark.parametrize(
    ("orbit", "settings", "message"),
    [
        (ONE_PASS, {"k": 0}, "k must be positive"),
        (ONE_PASS, {"flat": 1.5}, "flat must be between 0 and 1"),
        (ONE_PASS, {"pass_number": 0}, "pass_number must be a whole number of at least 1"),
        (ONE_PASS, {"pass_number": 1.0}, "pass_number must be a whole number of at least 1"),
        (ONE_PASS, {"k": 3.0}, r"the window would start at t = -8 \(peak -2, k = 3\), before the orbit's first"),
        (orbit_from(TIMES, 10 - TIMES), {}, "pass 1 isn't in the orbit: it has 0 pass"),  # still approaching
        (orbit_from(TIMES + 4, 1 + (TIMES + 2) ** 2), {}, "the pass is at t = 2, not before now"),
        (orbit_from([-3, -2, -2, -1], [3, 1, 1, 2]), {}, "times running strictly one way"),
        (ONE_PASS[:2], {}, "at least 3 rows"),
    ],
)
def test_bad_windows_name_the_problem(orbit, settings, message):
    with pytest.raises(ValueError, match=message):
        window(orbit, **settings)


# --- difference_weights ----------------------------------------------------------------------------------

A = np.zeros((4, 5), dtype=np.float32)
B = A.copy()
B[1, 2], B[2, 3] = 0.8, 0.2  # the two images differ in two pixels


def test_difference_weights_are_the_scaled_difference_with_a_floor():
    weights = derived.difference_weights(A, B)
    assert weights.dtype == np.float32 and weights.shape == A.shape
    assert weights[1, 2] == 1.0 and weights[2, 3] == pytest.approx(0.25)
    assert weights[0, 0] == pytest.approx(0.05)  # the default floor, where the images agree
    assert (derived.difference_weights(B, A) == weights).all()  # the order doesn't matter
    assert derived.difference_weights(A, B, floor=0.0)[0, 0] == 0.0


def test_a_threshold_makes_the_weights_on_off_before_the_floor():
    weights = derived.difference_weights(A, B, threshold=0.5)
    assert weights[1, 2] == 1.0 and weights[2, 3] == pytest.approx(0.05) and weights[0, 0] == pytest.approx(0.05)
    assert set(np.unique(derived.difference_weights(A, B, threshold=0.1, floor=0.0))) == {0.0, 1.0}


def test_identical_images_give_the_floor_everywhere():
    assert (derived.difference_weights(A, A) == np.float32(0.05)).all()
    assert not derived.difference_weights(A, A, floor=0.0).any()  # all zeros: for a metric to reject as weights


@pytest.mark.parametrize(
    ("settings", "message"),
    [({"floor": -0.1}, "floor must be between 0 and 1"), ({"threshold": 2}, "threshold must be None or between"),
     ({"threshold": True}, "threshold must be None or between")],
)
def test_bad_difference_settings_name_the_problem(settings, message):
    with pytest.raises(ValueError, match=message):
        derived.difference_weights(A, B, **settings)


def test_difference_weights_need_images_of_one_shape():
    with pytest.raises(ValueError, match=r"same shape: \(4, 5\) vs \(5, 4\)"):
        derived.difference_weights(A, A.T)


# --- detect_edges ----------------------------------------------------------------------------------------

STEP = np.zeros((6, 8), dtype=np.float32)
STEP[:, 4:] = 1.0  # dark on the left, bright on the right: one vertical edge


def test_edges_mark_where_brightness_changes():
    edges = derived.detect_edges(STEP)
    assert edges.dtype == np.float32 and edges.shape == STEP.shape
    assert (edges[:, 3:5] == 1.0).all()  # the two columns either side of the step
    assert not edges[:, :3].any() and not edges[:, 5:].any()  # flat regions, and the image's border
    assert (derived.detect_edges(STEP.T) == edges.T).all()  # rows and columns are treated alike


def test_a_flat_image_has_no_edges():
    assert not derived.detect_edges(np.full((5, 5), 0.3, dtype=np.float32)).any()


def test_edge_strength_is_scaled_and_a_threshold_makes_it_on_off():
    image = STEP.copy()
    image[:, 6:] = 1.2  # a second, weaker edge
    edges = derived.detect_edges(image)
    assert edges.max() == 1.0 and 0 < edges[2, 5] < 1
    on_off = derived.detect_edges(image, threshold=0.5)
    assert set(np.unique(on_off)) == {0.0, 1.0} and on_off[2, 3] == 1.0 and on_off[2, 5] == 0.0
    with pytest.raises(ValueError, match="threshold must be None or between"):
        derived.detect_edges(image, threshold=-1)
