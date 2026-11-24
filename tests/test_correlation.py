"""Tests for the correlation image metric (metrics.correlation, MDRN-4 step 8): the Pearson correlation of
two images' pixels, with optional per-pixel weights. Numpy only: small synthetic images."""

import math

import pytest

# Skipped where the dependencies aren't installed, such as native Windows.
np = pytest.importorskip("numpy")

from galactic_tidal_constraints.simr import metrics  # noqa: E402

RNG = np.random.default_rng(11)
IMAGE = RNG.random((16, 24)).astype(np.float32)
OTHER = RNG.random((16, 24)).astype(np.float32)


def test_its_declaration():
    method = metrics.METHODS["correlation"]
    assert (method.kind, method.inputs, method.best, method.range) == ("scalar", ("image", "image"), "high", (-1.0, 1.0))
    assert method.weights is True and metrics.weights_kind(method) == "image" and method.settings == {}


def test_an_image_correlates_fully_with_itself_and_its_negative():
    assert metrics.correlation(IMAGE, IMAGE) == pytest.approx(1.0)
    assert metrics.correlation(IMAGE, 1.0 - IMAGE) == pytest.approx(-1.0)


def test_brightness_and_contrast_dont_matter():
    dimmer = (0.3 * IMAGE + 0.1).astype(np.float32)
    assert metrics.correlation(IMAGE, dimmer) == pytest.approx(1.0, abs=1e-6)
    assert metrics.absolute_difference(IMAGE, dimmer) > 0.1  # where the difference metric sees a gap


def test_it_matches_numpy_and_is_symmetric():
    expected = np.corrcoef(IMAGE.ravel().astype(float), OTHER.ravel().astype(float))[0, 1]
    assert metrics.correlation(IMAGE, OTHER) == pytest.approx(expected, abs=1e-12)
    assert metrics.correlation(OTHER, IMAGE) == pytest.approx(expected, abs=1e-12)
    assert abs(expected) < 0.2  # two unrelated noise images


def test_the_value_never_leaves_its_range():
    for scale in (1e-3, 1.0, 1e3):
        value = metrics.correlation(IMAGE * scale, IMAGE * scale)
        assert -1.0 <= value <= 1.0


def test_weights_pick_where_the_pattern_counts():
    mixed = IMAGE.copy()
    mixed[:, 12:] = OTHER[:, 12:]  # the left half matches IMAGE, the right half doesn't
    left = np.zeros(IMAGE.shape, dtype=np.float32)
    left[:, :12] = 1.0
    assert metrics.correlation(IMAGE, mixed, weights=left) == pytest.approx(1.0)
    assert metrics.correlation(IMAGE, mixed) < 0.9
    assert metrics.correlation(IMAGE, mixed, weights=1.0 - left) == pytest.approx(
        np.corrcoef(IMAGE[:, 12:].ravel().astype(float), OTHER[:, 12:].ravel().astype(float))[0, 1], abs=1e-12)


def test_uniform_weights_change_nothing():
    even = np.full(IMAGE.shape, 3.0, dtype=np.float32)
    assert metrics.correlation(IMAGE, OTHER, weights=even) == pytest.approx(metrics.correlation(IMAGE, OTHER))


@pytest.mark.parametrize("flat, which", [(0, "the first"), (1, "the second")])
def test_a_flat_image_is_undefined_not_zero(flat, which):
    images = [IMAGE, OTHER]
    images[flat] = np.full(IMAGE.shape, 0.7, dtype=np.float32)
    with pytest.raises(ValueError, match=f"the correlation is undefined: {which} image has no variation"):
        metrics.correlation(*images)


def test_an_image_flat_where_the_weights_count_is_undefined_too():
    half_flat = IMAGE.copy()
    half_flat[:, :12] = 0.5
    left = np.zeros(IMAGE.shape, dtype=np.float32)
    left[:, :12] = 1.0
    assert math.isfinite(metrics.correlation(half_flat, OTHER))
    with pytest.raises(ValueError, match="the first image has no variation where the pixels count"):
        metrics.correlation(half_flat, OTHER, weights=left)


def test_mismatched_shapes_and_bad_weights_are_errors():
    with pytest.raises(ValueError, match=r"the same shape: \(16, 24\) vs \(8, 24\)"):
        metrics.correlation(IMAGE, OTHER[:8])
    with pytest.raises(ValueError, match="the weights must have the images' shape"):
        metrics.correlation(IMAGE, OTHER, weights=np.ones((8, 24)))
    with pytest.raises(ValueError, match="all zero"):
        metrics.correlation(IMAGE, OTHER, weights=np.zeros(IMAGE.shape))


def test_compute_runs_it_with_and_without_weights():
    weights = RNG.random(IMAGE.shape).astype(np.float32)
    assert metrics.compute("correlation", [IMAGE, OTHER]) == metrics.correlation(IMAGE, OTHER)
    assert metrics.compute("correlation", [IMAGE, OTHER, weights]) == metrics.correlation(IMAGE, OTHER, weights)
    with pytest.raises(ValueError, match="undefined"):
        metrics.compute("correlation", [IMAGE, np.zeros(IMAGE.shape, dtype=np.float32)])
