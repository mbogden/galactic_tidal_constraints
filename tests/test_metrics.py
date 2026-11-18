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
    assert set(metrics.METHODS) == {"closest_approach", "absolute_difference"}
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
    for method in metrics.METHODS.values():
        assert method.settings == {}  # none of the first methods has settings
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
        "settings": {},
        "best": None,
        "range": None,
        "outputs": {"t": [], "distance": [], "position": [3], "velocity": [3]},
    }
    assert table["absolute_difference"] == {
        "kind": "scalar",
        "inputs": ["image", "image"],
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
