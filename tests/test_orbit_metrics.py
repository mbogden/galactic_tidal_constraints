"""Tests for the orbit metrics and for optional weights (simr/metrics.py; MDRN-4 step 4b)."""

import pytest

np = pytest.importorskip("numpy")

from galactic_tidal_constraints.simr import derived, metrics  # noqa: E402


def orbit(times, x, y=0.0):
    """An orbit in the x-y plane at the given times: shape (n, 7), float32."""
    times = np.asarray(times, dtype=float)
    out = np.zeros((len(times), 7), dtype=np.float32)
    out[:, 0], out[:, 1], out[:, 6] = x, y, times
    return out


T = np.linspace(-4, 0, 41)  # the target's times: every 0.1
DENSE = np.linspace(-6, 0, 6001)  # a model's times: every 0.001, as SPAM steps
TARGET = orbit(T, T)  # moving along x: at x = t
ON_TARGET = orbit(DENSE, DENSE)


# --- The declarations and weights through compute -------------------------------------------------------


def test_the_orbit_metrics_declare_optional_row_weights():
    for name in ("orbit_time_error", "orbit_position_error"):
        method = metrics.METHODS[name]
        assert (method.kind, method.inputs, method.weights) == ("scalar", ("orbit", "orbit"), "weights")
        assert (method.best, method.range, method.settings) == ("low", (0.0, np.inf), {})
    assert metrics.METHODS["absolute_difference"].weights == "image"
    assert metrics.METHODS["closest_approach"].weights is None


def test_compute_passes_one_extra_input_as_the_weights():
    weights = np.zeros(41, dtype=np.float32)
    weights[20] = 1.0  # only t = -2 counts, where the model below is 1 away in y
    model = orbit(DENSE, DENSE, y=(DENSE + 3))
    assert metrics.compute("orbit_time_error", [TARGET, model, weights]) == pytest.approx(1.0, abs=1e-5)
    assert metrics.compute("orbit_time_error", [TARGET, model]) != pytest.approx(1.0, abs=1e-2)


@pytest.mark.parametrize(
    ("method", "inputs", "message"),
    [
        ("orbit_time_error", [TARGET], r"takes 2 input\(s\) \['orbit', 'orbit'\], and optionally 'weights' weights"),
        ("orbit_time_error", [TARGET, ON_TARGET, np.ones(41, np.float32), np.ones(41, np.float32)], "got 4"),
        ("closest_approach", [TARGET, np.ones(41, np.float32)], r"takes 1 input\(s\) \['orbit'\]: got 2"),
        ("orbit_time_error", [TARGET, ON_TARGET, np.ones((41, 1), np.float32)],
         r"the weights must be a 'weights' array, shape \(None,\)"),
        ("orbit_time_error", [TARGET, ON_TARGET, np.zeros(41, np.float32)], "the weights are all zero"),
        ("orbit_time_error", [TARGET, ON_TARGET, -np.ones(41, np.float32)], "must be finite and not negative"),
        ("orbit_time_error", [TARGET, ON_TARGET, np.ones(40, np.float32)], "one per target row, 41: got"),
    ],
)
def test_compute_rejects_bad_weights_and_input_counts(method, inputs, message):
    with pytest.raises(ValueError, match=message):
        metrics.compute(method, inputs)


# --- The two orbit errors --------------------------------------------------------------------------------


def test_a_model_on_the_target_scores_zero_on_both():
    assert metrics.orbit_time_error(TARGET, ON_TARGET) == pytest.approx(0, abs=1e-9)
    assert metrics.orbit_position_error(TARGET, ON_TARGET) == pytest.approx(0, abs=1e-9)


def test_a_constant_offset_is_its_square_on_both():
    beside = orbit(DENSE, DENSE, y=2.0)
    assert metrics.orbit_time_error(TARGET, beside) == pytest.approx(4.0)
    assert metrics.orbit_position_error(TARGET, beside) == pytest.approx(4.0)


def test_the_model_rows_may_run_backward_in_time():
    beside = orbit(DENSE, DENSE, y=(0.5 * DENSE))
    backward = beside[::-1].copy()  # as sim.search_orbit runs
    assert metrics.orbit_time_error(TARGET, backward) == metrics.orbit_time_error(TARGET, beside)
    assert metrics.orbit_position_error(TARGET, backward) == metrics.orbit_position_error(TARGET, beside)


def test_the_result_is_a_time_average_whatever_the_targets_spacing():
    # The model drifts away in y as 0.5 t, so the squared distance is 0.25 t^2: its average over
    # t = -4..0 is 4/3. A target sampled densely near t = -4, where the error is largest, gives the
    # same answer as an even one; a plain mean over its rows would not.
    model = orbit(DENSE, DENSE, y=(0.5 * DENSE))
    uneven_times = np.concatenate([np.linspace(-4, -3, 101), np.linspace(-2.9, 0, 30)])
    uneven = orbit(uneven_times, uneven_times)
    assert metrics.orbit_time_error(TARGET, model) == pytest.approx(4 / 3, rel=0.005)
    assert metrics.orbit_time_error(uneven, model) == pytest.approx(4 / 3, rel=0.005)
    plain_mean = float(np.mean(0.25 * uneven_times**2))
    assert plain_mean > 1.2 * 4 / 3  # what the old row-count average would have said


def test_a_model_that_is_late_fails_on_time_and_nearly_passes_on_position():
    # Both go round one circle of radius 5; the model runs 0.2 rad behind. With the window's weights,
    # the path is the same (position) but the model is in the wrong place at each time (time).
    times = np.linspace(-6, 0, 121)
    target = orbit(times, 5 * np.cos(times), 5 * np.sin(times))
    target[:, 0] += 6  # off-centre, so the distance from the primary has a pass (near t = -pi)
    late = orbit(DENSE, 6 + 5 * np.cos(DENSE - 0.2), 5 * np.sin(DENSE - 0.2))
    weights = derived.recent_approach_cosine_window(target, k=0.9)
    by_time = metrics.orbit_time_error(target, late, weights)
    by_position = metrics.orbit_position_error(target, late, weights)
    assert by_time == pytest.approx((2 * 5 * np.sin(0.1)) ** 2, rel=0.01)  # the chord between them
    assert by_position < 0.02 * by_time


def test_weights_choose_which_target_times_count():
    model = orbit(DENSE, DENSE, y=np.where(DENSE < -2, 3.0, 0.0))  # off by 3 before t = -2, on it after
    late_only = (T > -1.95).astype(np.float32)
    early_only = (T < -2.05).astype(np.float32)
    assert metrics.orbit_time_error(TARGET, model, late_only) == pytest.approx(0, abs=1e-9)
    assert metrics.orbit_time_error(TARGET, model, early_only) == pytest.approx(9.0)
    assert metrics.orbit_position_error(TARGET, model, early_only) == pytest.approx(9.0)


def test_the_position_search_stays_inside_the_counted_times():
    # The model passes through the target's late points only at early times, far from them later.
    # Counting only late target times, those early model rows are not searched.
    model = orbit(DENSE, np.where(DENSE < -3, DENSE + 3.5, 50.0))
    late_only = (T >= -0.5).astype(np.float32)  # target x from -0.5 to 0: the model was there near t = -3.7
    assert metrics.orbit_position_error(TARGET, model, late_only) > 2000


def test_the_model_only_needs_to_cover_the_target_times_that_count():
    short = orbit(np.linspace(-1.5, 0, 1501), np.linspace(-1.5, 0, 1501))
    with pytest.raises(ValueError, match=r"model orbit runs from t = -1.5 to 0, but the target rows that count "
                                         r"run from t = -4 to 0"):
        metrics.orbit_time_error(TARGET, short)
    late_only = (T >= -1.0).astype(np.float32)
    assert metrics.orbit_time_error(TARGET, short, late_only) == pytest.approx(0, abs=1e-9)
    assert metrics.orbit_position_error(TARGET, short, late_only) == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("method", [metrics.orbit_time_error, metrics.orbit_position_error])
def test_bad_orbits_name_the_problem(method):
    with pytest.raises(ValueError, match="target orbit needs at least 2 rows, with times strictly increasing"):
        method(TARGET[::-1].copy(), ON_TARGET)
    with pytest.raises(ValueError, match="model orbit needs at least 2 rows, with times running strictly one way"):
        method(TARGET, orbit([-4, -2, -2, 0], [0, 0, 0, 0]))


def test_the_search_works_in_chunks(monkeypatch):
    beside = orbit(DENSE, DENSE, y=(0.5 * DENSE))
    whole = metrics.orbit_position_error(TARGET, beside)
    monkeypatch.setattr(metrics, "ORBIT_SEARCH_CHUNK", 5000)  # about one target row at a time
    assert metrics.orbit_position_error(TARGET, beside) == whole


# --- absolute_difference with weights --------------------------------------------------------------------

A = np.zeros((2, 3), dtype=np.float32)
B = np.array([[0.6, 0.0, 0.0], [0.0, 0.0, 0.3]], dtype=np.float32)


def test_absolute_difference_takes_a_weight_per_pixel():
    assert metrics.absolute_difference(A, B) == pytest.approx(0.15)
    even = np.ones((2, 3), dtype=np.float32)
    assert metrics.absolute_difference(A, B, even) == pytest.approx(0.15)  # even weights: the plain mean
    mostly_first = np.array([[3.0, 0, 0], [0, 0, 1.0]], dtype=np.float32)
    assert metrics.absolute_difference(A, B, mostly_first) == pytest.approx((3 * 0.6 + 0.3) / 4)
    assert metrics.compute("absolute_difference", [A, B, mostly_first]) == pytest.approx((3 * 0.6 + 0.3) / 4)


def test_image_weights_must_match_the_images_and_not_be_all_zero():
    with pytest.raises(ValueError, match=r"weights must have the images' shape \(2, 3\): got \(3, 2\)"):
        metrics.absolute_difference(A, B, np.ones((3, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="all zero"):
        metrics.absolute_difference(A, B, np.zeros((2, 3), dtype=np.float32))
