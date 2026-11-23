"""Tests for the Metrics stage (simr/metrics.py): the registry, the table and the first methods.

The stage is numpy-only, so these tests never need jspam: methods get small synthetic arrays.
"""

import json
from pathlib import Path

import pytest

# Skipped where the dependencies aren't installed, such as native Windows.
np = pytest.importorskip("numpy")

from galactic_tidal_constraints.simr import metrics  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


# --- MDRN-4 step 2a: the registry -----------------------------------------------------------------------


def test_the_first_methods_are_registered():
    assert set(metrics.METHODS) == {"closest_approach", "absolute_difference", "orbit_time_error",
                                    "orbit_position_error", "pca_shape", "pca_difference"}
    assert metrics.METHODS["closest_approach"].kind == "descriptive"
    assert metrics.METHODS["absolute_difference"].kind == "scalar"


def test_declarations_are_complete():
    approach = metrics.METHODS["closest_approach"]
    assert approach.inputs == ("orbit",)
    assert approach.outputs == {"t": (), "distance": (), "position": (3,), "velocity": (3,)}
    assert approach.best is None and approach.range is None
    difference = metrics.METHODS["absolute_difference"]
    assert difference.inputs == ("image", "image")
    assert difference.outputs is None
    assert (difference.best, difference.range) == ("low", (0.0, 1.0))
    assert approach.settings == {} and difference.settings == {}
    for method in metrics.METHODS.values():
        assert method.setup is None  # plain methods: no one-time setup step


def test_scalar_outputs_are_the_float_keys():
    assert metrics.METHODS["closest_approach"].scalar_outputs == ("t", "distance")
    assert metrics.METHODS["absolute_difference"].scalar_outputs == ()


def test_table_is_json_ready_and_matches_the_declarations():
    table = json.loads(json.dumps(metrics.table()))
    assert set(table) == set(metrics.METHODS)
    assert table["closest_approach"] == {
        "kind": "descriptive",
        "inputs": ["orbit"],
        "weights": False,
        "settings": {},
        "best": None,
        "range": None,
        "outputs": {"t": [], "distance": [], "position": [3], "velocity": [3]},
    }
    assert table["absolute_difference"] == {
        "kind": "scalar",
        "inputs": ["image", "image"],
        "weights": True,
        "settings": {},
        "best": "low",
        "range": [0.0, 1.0],
        "outputs": None,
    }


@pytest.mark.parametrize(
    ("declaration", "message"),
    [
        ({"kind": "scala", "inputs": ("image",), "best": "low", "range": (0, 1)}, "kind must be one of"),
        ({"kind": "scalar", "inputs": ("picture",), "best": "low", "range": (0, 1)}, "unknown input kinds"),
        ({"kind": "scalar", "inputs": ("image",), "range": (0, 1)}, 'best="low" or "high"'),
        ({"kind": "scalar", "inputs": ("image",), "best": "low"}, "declares its range"),
        ({"kind": "scalar", "inputs": ("image",), "best": "low", "range": (0, 1), "outputs": {"x": ()}},
         "declares no outputs"),
        ({"kind": "descriptive", "inputs": ("orbit",)}, "declares its output keys"),
        ({"kind": "descriptive", "inputs": ("orbit",), "outputs": {"x": ()}, "best": "low"},
         "a scalar method's value only"),
    ],
)
def test_bad_declarations_fail_at_registration(declaration, message):
    with pytest.raises(ValueError, match=message):

        @metrics.metric(**declaration)
        def throwaway():  # pragma: no cover - registration fails first
            pass

    assert "throwaway" not in metrics.METHODS


def test_registering_a_name_twice_is_an_error():
    with pytest.raises(ValueError, match="already registered"):

        @metrics.metric(kind="scalar", inputs=("image", "image"), best="low", range=(0, 1))
        def absolute_difference(a, b):  # pragma: no cover - registration fails first
            pass

    assert metrics.METHODS["absolute_difference"].function is metrics.absolute_difference


# --- MDRN-4 step 2a: closest_approach -------------------------------------------------------------------

# A small orbit with its closest row in the middle: distances 5, 2, 3.
ORBIT = np.array(
    [
        [5.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0],
        [0.0, 2.0, 0.0, -1.5, 0.5, 2.0, -1.0],
        [0.0, 0.0, 3.0, 0.25, -1.0, 0.75, -2.0],
    ],
    dtype=np.float32,
)


def test_closest_approach_finds_the_smallest_distance():
    value = metrics.closest_approach(ORBIT)
    assert set(value) == {"t", "distance", "position", "velocity"}
    np.testing.assert_allclose(value["t"], -1.0)
    np.testing.assert_allclose(value["distance"], 2.0)
    np.testing.assert_allclose(value["position"], [0.0, 2.0, 0.0])
    np.testing.assert_allclose(value["velocity"], [-1.5, 0.5, 2.0])
    assert isinstance(value["t"], float) and isinstance(value["distance"], float)


def test_closest_approach_matches_its_declared_outputs():
    value = metrics.closest_approach(ORBIT)
    declared = metrics.METHODS["closest_approach"].outputs
    assert set(value) == set(declared)
    for key, shape in declared.items():
        assert np.shape(value[key]) == shape, key


def test_closest_approach_ignores_row_order():
    # sim.search_orbit's rows run backward in time; the smallest distance is the same either way.
    forward, backward = metrics.closest_approach(ORBIT), metrics.closest_approach(ORBIT[::-1])
    for key in forward:
        np.testing.assert_array_equal(forward[key], backward[key], err_msg=key)


def test_closest_approach_returns_a_copy_not_a_view():
    orbit = ORBIT.copy()
    value = metrics.closest_approach(orbit)
    orbit[:] = 0
    np.testing.assert_array_equal(value["position"], [0.0, 2.0, 0.0])


# --- MDRN-4 step 2a: absolute_difference ----------------------------------------------------------------


def test_absolute_difference_is_the_mean_pixel_difference():
    a = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
    b = np.zeros((2, 2), dtype=np.float32)
    value = metrics.absolute_difference(a, b)
    assert value == pytest.approx(0.4375)
    assert isinstance(value, float)
    assert metrics.absolute_difference(b, a) == value  # symmetric


def test_identical_images_differ_by_zero():
    image = np.random.default_rng(0).random((4, 3), dtype=np.float32)
    assert metrics.absolute_difference(image, image) == 0.0


def test_mismatched_shapes_are_an_error_naming_them():
    with pytest.raises(ValueError, match=r"same shape.*\(2, 2\).*\(2, 3\)"):
        metrics.absolute_difference(np.zeros((2, 2), dtype=np.float32), np.zeros((2, 3), dtype=np.float32))


# --- MDRN-4 step 2b: compute ----------------------------------------------------------------------------

from contextlib import contextmanager  # noqa: E402


@contextmanager
def registered(function, **declaration):
    """A method registered for one test, never left in METHODS."""
    metrics.metric(**declaration)(function)
    try:
        yield
    finally:
        metrics.METHODS.pop(function.__name__, None)


def test_compute_runs_a_scalar_method():
    a = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
    b = np.zeros((2, 2), dtype=np.float32)
    value = metrics.compute("absolute_difference", [a, b])
    assert value == metrics.absolute_difference(a, b)
    assert isinstance(value, float)


def test_compute_select_returns_one_float():
    full = metrics.compute("closest_approach", [ORBIT])
    assert set(full) == {"t", "distance", "position", "velocity"}
    value = metrics.compute("closest_approach", [ORBIT], select="distance")
    assert value == full["distance"]
    assert isinstance(value, float)


@pytest.mark.parametrize(
    ("method", "inputs", "message"),
    [
        # (3, 6) is a particles shape, not an orbit's (None, 7)
        ("closest_approach", [ORBIT[:, :6]], r"input 0 must be a 'orbit' array, shape \(None, 7\): got \(3, 6\)"),
        ("closest_approach", [ORBIT[0]], r"got \(7,\)"),  # one row: 1-D, not an orbit
        ("closest_approach", [ORBIT[:, :3].tolist()], "got list"),  # not an array at all
        ("absolute_difference", [np.zeros((2, 2), dtype=np.float32), np.zeros((2, 2, 2), dtype=np.float32)],
         r"input 1 must be a 'image' array.*got \(2, 2, 2\)"),
    ],
)
def test_compute_checks_each_input_against_its_kind(method, inputs, message):
    with pytest.raises(ValueError, match=message):
        metrics.compute(method, inputs)


def test_check_input_names_what_it_was_given():
    metrics.check_input("particles", np.zeros((5, 6), dtype=np.float32), "test_particles")  # fits: no error
    with pytest.raises(ValueError, match=r"test_particles must be a 'particles' array, shape \(None, 6\): got \(5, 7\)"):
        metrics.check_input("particles", np.zeros((5, 7), dtype=np.float32), "test_particles")


def test_compute_rejects_a_nonfinite_scalar():
    def bad_scalar(a, b):
        return float("nan")

    with registered(bad_scalar, kind="scalar", inputs=("image", "image"), best="low", range=(0, 1)):
        with pytest.raises(ValueError, match="returned nan"):
            metrics.compute("bad_scalar", [ORBIT, ORBIT])


def test_compute_rejects_a_scalar_that_isnt_a_number():
    def not_a_number(a, b):
        return {"oops": 1.0}

    with registered(not_a_number, kind="scalar", inputs=("image", "image"), best="low", range=(0, 1)):
        with pytest.raises(ValueError, match="not one float"):
            metrics.compute("not_a_number", [ORBIT, ORBIT])


def test_compute_accepts_numpy_scalars():
    def numpy_scalar(a, b):
        return np.float32(0.5)

    with registered(numpy_scalar, kind="scalar", inputs=("image", "image"), best="low", range=(0, 1)):
        value = metrics.compute("numpy_scalar", [ORBIT, ORBIT])
    assert value == 0.5 and isinstance(value, float)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"t": 1.0}, "exactly the keys"),  # a missing key
        ({"t": 1.0, "position": np.zeros(2), "extra": 0.0}, "exactly the keys"),  # an extra key
        (3.0, "exactly the keys"),  # not a dict at all
        ({"t": 1.0, "position": np.zeros(3)}, "must have shape"),
        ({"t": float("inf"), "position": np.zeros(2)}, "holds NaN or inf"),
        ({"t": 1.0, "position": np.array([0.0, np.nan])}, "holds NaN or inf"),
    ],
)
def test_compute_checks_a_descriptive_value_against_its_declaration(value, message):
    def bad_descriptor(orbit):
        return value

    with registered(bad_descriptor, kind="descriptive", inputs=("orbit",), outputs={"t": (), "position": (2,)}):
        with pytest.raises(ValueError, match=message):
            metrics.compute("bad_descriptor", [ORBIT])


# --- MDRN-4 step 2a: the module doc's table -------------------------------------------------------------


def test_metrics_doc_table_matches_the_registry():
    """docs/modules/metrics.md's methods table stays current: one row per method, with its kind,
    input kinds, output keys (descriptive) and best and range (scalar)."""
    doc = (REPO / "docs" / "modules" / "metrics.md").read_text(encoding="utf-8")
    section = doc.split("## The methods", 1)[1]
    rows = {line.split("|")[1].strip(" `"): line for line in section.splitlines() if line.startswith("| `")}
    assert set(rows) == set(metrics.METHODS), "docs/modules/metrics.md's methods table is out of date"
    for name, method in metrics.METHODS.items():
        row = rows[name]
        assert method.kind in row, name
        for kind in method.inputs:
            assert kind in row, name
        for key in method.outputs or {}:
            assert f"`{key}`" in row, name
        if method.kind == "scalar":
            assert method.best in row, name


# --- PCA shape: pca_shape and pca_difference ------------------------------------------------------------

RNG = np.random.default_rng(7)
# A triaxial Gaussian cloud: semi-axes 4, 2, 1 along x, y, z, with velocities alongside.
BLOB = np.hstack([RNG.normal(size=(20000, 3)) * [4.0, 2.0, 1.0], RNG.normal(size=(20000, 3))]).astype(np.float32)


def rotation(angle_z, angle_x):
    cz, sz, cx, sx = np.cos(angle_z), np.sin(angle_z), np.cos(angle_x), np.sin(angle_x)
    return np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])


def moved(blob, matrix=None, scale=1.0, shift=(0.0, 0.0, 0.0)):
    out = blob.astype(np.float64).copy()
    out[:, :3] = scale * out[:, :3] @ (np.eye(3) if matrix is None else matrix).T + shift
    return out


def test_pca_declarations():
    shape, difference = metrics.METHODS["pca_shape"], metrics.METHODS["pca_difference"]
    assert shape.inputs == ("particles",) and difference.inputs == ("particles", "particles")
    assert shape.settings == {"columns": "position", "center": "mean", "trim_quantile": None}
    assert difference.settings == {
        **shape.settings, "descriptor": "ratios", "distance": "l2", "orientation_weight": 0.0,
    }
    assert shape.scalar_outputs == ("b_a", "c_a")
    assert (difference.best, difference.range) == ("low", (0.0, float("inf")))


def test_pca_shape_finds_the_axes():
    value = metrics.compute("pca_shape", [BLOB])
    np.testing.assert_allclose(value["axes"], [4.0, 2.0, 1.0], rtol=0.03)
    np.testing.assert_allclose(np.abs(value["vectors"]), np.eye(3), atol=0.02)
    np.testing.assert_allclose(value["center"], 0.0, atol=0.06)
    np.testing.assert_allclose([value["b_a"], value["c_a"]], [0.5, 0.25], rtol=0.03)


def test_pca_shape_vectors_have_a_canonical_sign():
    for blob in (BLOB, moved(BLOB, matrix=-np.eye(3))):
        vectors = metrics.pca_shape(blob)["vectors"]
        assert (vectors[np.arange(3), np.argmax(np.abs(vectors), axis=1)] > 0).all()


def test_pca_shape_velocity_columns():
    np.testing.assert_allclose(metrics.pca_shape(BLOB, columns="velocity")["axes"], 1.0, rtol=0.03)


def test_pca_shape_does_not_change_its_input():
    blob = BLOB.copy()
    metrics.pca_shape(blob, center="median", trim_quantile=0.9)
    np.testing.assert_array_equal(blob, BLOB)


def test_pca_trimming_drops_the_far_outliers():
    with_outliers = np.vstack([BLOB, np.tile([[1000.0, 0, 0, 0, 0, 0]], (100, 1))])
    trimmed = metrics.pca_shape(with_outliers, trim_quantile=0.99)["axes"]
    np.testing.assert_allclose(trimmed, [4.0, 2.0, 1.0], rtol=0.1)
    assert metrics.pca_shape(with_outliers)["axes"][0] > 50


@pytest.mark.parametrize(
    "points",
    [
        np.zeros((5, 3)),  # one point, repeated
        np.outer(np.arange(10.0), [1.0, 2.0, 3.0]),  # a straight line
        np.column_stack([RNG.normal(size=(100, 2)), np.zeros(100)]),  # a flat sheet in 3D
    ],
)
def test_pca_degenerate_clouds_stay_finite(points):
    particles = np.hstack([points, np.zeros_like(points)])
    metrics.compute("pca_shape", [particles])  # checks every output is finite
    assert metrics.compute("pca_difference", [particles, BLOB]) >= 0.0


def test_pca_needs_three_particles():
    with pytest.raises(ValueError, match="at least 3 particles"):
        metrics.pca_shape(BLOB[:2])


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"columns": "pos"}, "columns must be"),
        ({"center": "mode"}, "center must be"),
        ({"trim_quantile": 0.0}, "trim_quantile must be"),
        ({"descriptor": "raw"}, "descriptor must be"),
        ({"distance": "cosine"}, "distance must be"),
        ({"orientation_weight": -1.0}, "orientation_weight must be"),
    ],
)
def test_pca_bad_settings_are_errors(params, message):
    with pytest.raises(ValueError, match=message):
        metrics.pca_difference(BLOB, BLOB, **params)


def test_pca_difference_of_a_cloud_with_itself_is_zero():
    for descriptor in metrics.PCA_DESCRIPTORS:
        value = metrics.pca_difference(BLOB, BLOB, descriptor=descriptor, orientation_weight=1.0)
        assert value == pytest.approx(0.0, abs=1e-9)


def test_pca_difference_ignores_rotation_and_shift():
    other = moved(BLOB, matrix=rotation(0.7, 1.1), shift=(30.0, -5.0, 2.0))
    assert metrics.pca_difference(BLOB, other) == pytest.approx(0.0, abs=1e-6)
    assert metrics.pca_difference(BLOB, other, descriptor="geometric", distance="l1") == pytest.approx(0.0, abs=1e-6)


def test_pca_difference_ignores_size_only_with_ratios():
    bigger = moved(BLOB, scale=3.0)
    assert metrics.pca_difference(BLOB, bigger) == pytest.approx(0.0, abs=1e-6)
    assert metrics.pca_difference(BLOB, bigger, descriptor="axes") > 1.0


def test_pca_difference_orientation_term():
    turned = moved(BLOB, matrix=rotation(np.pi / 2, 0.0))  # the long axis now points along y
    assert metrics.pca_difference(BLOB, turned, orientation_weight=2.0) == pytest.approx(2.0, abs=1e-3)


def test_pca_difference_known_shapes():
    sphere = np.hstack([RNG.normal(size=(20000, 3)), np.zeros((20000, 3))])
    # ratios [0.5, 0.25] vs [1, 1]: l2 = sqrt(0.25 + 0.5625), l1 = 1.25
    assert metrics.pca_difference(BLOB, sphere) == pytest.approx(np.sqrt(0.8125), rel=0.03)
    assert metrics.pca_difference(BLOB, sphere, distance="l1") == pytest.approx(1.25, rel=0.03)
