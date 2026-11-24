"""Tests for the composite score (metrics.composite, MDRN-4 step 5a): a weighted mean of each model's
scalar metric cells across both levels.

Numpy only: the metric arrays are small hand-made ones shaped like SimrResult's.
"""

import math

import pytest

# Skipped where the dependencies aren't installed, such as native Windows.
np = pytest.importorskip("numpy")

from galactic_tidal_constraints.simr import metrics  # noqa: E402

# Two models, one simulator entry, two sim-level entries (say the two orbit errors).
SIM = np.array([[[1.0, 3.0]], [[2.0, 6.0]]])
# The same two models, one simulator entry, two image entries, one image-level entry.
IMG = np.array([[[[0.1], [0.3]]], [[[0.2], [0.4]]]])


def test_the_50_50_orbit_mix():
    got = metrics.composite(sim_metrics=SIM, sim_weights=[[0.5, 0.5]])
    assert got.dtype == np.float64 and got.shape == (2,)
    assert got.tolist() == [2.0, 4.0]


def test_weights_need_not_sum_to_one():
    assert metrics.composite(sim_metrics=SIM, sim_weights=[[1, 3]]).tolist() == \
        metrics.composite(sim_metrics=SIM, sim_weights=[[0.25, 0.75]]).tolist() == [2.5, 5.0]


def test_no_weights_means_every_cell_counts_equally():
    assert metrics.composite(sim_metrics=SIM).tolist() == [2.0, 4.0]


def test_both_levels_make_one_weighted_mean():
    got = metrics.composite(SIM, IMG, sim_weights=[[1.0, 1.0]], img_weights=[[[2.0], [0.0]]])
    # (1*1 + 1*3 + 2*0.1) / (1 + 1 + 2), and the same for model 1
    assert got == pytest.approx([(1 + 3 + 0.2) / 4, (2 + 6 + 0.4) / 4])


def test_an_image_level_alone():
    assert metrics.composite(img_metrics=IMG).tolist() == pytest.approx([0.2, 0.3])


def test_weight_zero_skips_a_cell_whatever_it_holds():
    values = np.empty((2, 1, 3), dtype=object)  # as simr returns them: a dict, a float, NaN
    values[:, 0, 0] = [{"t": -1.0}, None]       # a descriptive entry (None: it failed)
    values[:, 0, 1] = [1.0, 2.0]
    values[:, 0, 2] = [math.nan, 5.0]
    assert metrics.composite(sim_metrics=values, sim_weights=[[0, 1, 0]]).tolist() == [1.0, 2.0]


def test_nan_with_a_nonzero_weight_makes_only_that_model_nan():
    values = SIM.copy()
    values[1, 0, 1] = math.nan
    got = metrics.composite(sim_metrics=values, sim_weights=[[0.5, 0.5]])
    assert got[0] == 2.0 and math.isnan(got[1])


def test_object_arrays_of_floats_work_like_float_arrays():
    assert metrics.composite(sim_metrics=SIM.astype(object)).tolist() == [2.0, 4.0]


@pytest.mark.parametrize("value", [{"t": 1.0}, None, True, "1.0"])
def test_a_cell_that_isnt_a_number_needs_weight_zero(value):
    values = SIM.astype(object)
    values[0, 0, 0] = value
    with pytest.raises(ValueError, match="not one number"):
        metrics.composite(sim_metrics=values)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({}, "needs sim_metrics, img_metrics or both"),
        ({"sim_weights": [[1, 1]]}, "sim_weights were given without sim_metrics"),
        ({"sim_metrics": SIM, "img_weights": [[[1], [1]]]}, "img_weights were given without img_metrics"),
        ({"sim_metrics": SIM, "sim_weights": [1, 1]}, r"shape \(simulator, entry\) = \(1, 2\): got \(2,\)"),
        ({"sim_metrics": SIM, "sim_weights": [[1, -1]]}, "finite and not negative"),
        ({"sim_metrics": SIM, "sim_weights": [[1, math.nan]]}, "finite and not negative"),
        ({"sim_metrics": SIM, "sim_weights": [[0, 0]]}, "all zero"),
        ({"sim_metrics": SIM[:, 0]}, r"must be \[model, simulator, entry\]"),
        ({"sim_metrics": SIM, "img_metrics": IMG[:1]}, "different numbers of models"),
    ],
)
def test_bad_calls_are_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        metrics.composite(**kwargs)


def test_zero_weights_on_one_level_are_fine_when_the_other_counts():
    got = metrics.composite(SIM, IMG, sim_weights=[[0, 0]], img_weights=[[[1], [1]]])
    assert got.tolist() == pytest.approx([0.2, 0.3])
