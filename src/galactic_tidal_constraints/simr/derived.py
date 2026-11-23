"""Derived stage: arrays made from other arrays, for metrics to use as data or as weights.

A derived method takes arrays (orbits, particles, images, from a simulation or from outside the run)
and returns an array about the size of its input: a window of weights along an orbit, a weight image,
an edge map. That is the line between this stage and the Metrics stage (`simr/metrics.py`): a metric
reduces its inputs to one float or a small dict; a derived method returns an array. So a metric never
needs another metric's value, and anything array-shaped that several metrics could share lives here.

Like the other stages, a derived method takes data in and returns data out; it never touches disk,
pools or MPI. Each is registered with the `derived` decorator where it's defined, which records its
input kinds, the kind of array it returns and its settings in `METHODS`. Kinds are the Metrics stage's
(`metrics.INPUT_SHAPES`), so a derived array can go wherever a metric takes that kind:

    "weights"  one value per row of another array, shape (n,): weights along an orbit's rows
    "image"    (height, width): a weight image, a mask, an edge map

Call a method directly, or by name through `compute`, which also checks the inputs and the result.
"""

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from galactic_tidal_constraints.simr import metrics


@dataclass(frozen=True)
class Derived:
    """One derived method's declaration.

    Attributes:
        name: the method's name (the function's name).
        function: the method itself: `function(*inputs, **settings)`.
        inputs: each input's kind (`metrics.INPUT_KINDS`), in the order the function takes them.
        output: the kind of array it returns.
        settings: the method's settings and their defaults.
    """

    name: str
    function: Callable
    inputs: tuple[str, ...]
    output: str
    settings: dict


METHODS: dict[str, Derived] = {}


def derived(inputs: tuple[str, ...], output: str, settings: dict | None = None):
    """Register a derived method: declare its input kinds, the kind it returns and its settings."""

    def register(function: Callable) -> Callable:
        name = function.__name__
        for kind in (*inputs, output):
            if kind not in metrics.INPUT_KINDS:
                raise ValueError(f"{name}: unknown kind {kind!r}. Known: {list(metrics.INPUT_KINDS)}")
        if not inputs:
            raise ValueError(f"{name}: a derived method takes at least one input")
        METHODS[name] = Derived(name=name, function=function, inputs=tuple(inputs), output=output,
                                settings=dict(settings or {}))
        return function

    return register


def table() -> dict[str, dict]:
    """The registered methods as plain, JSON-ready data: name -> inputs, output and settings."""
    return {
        name: {"inputs": list(m.inputs), "output": m.output, "settings": dict(m.settings)}
        for name, m in METHODS.items()
    }


def compute(method: str, inputs: list[np.ndarray], params: dict | None = None) -> np.ndarray:
    """Run one registered method on its input arrays, and check both sides against its declaration.

    Args:
        method: a name from `METHODS`.
        inputs: the input arrays, one per declared kind, in order.
        params: settings passed to the method as keyword arguments.

    Returns:
        The derived array, float32, of the method's declared kind.

    Raises:
        ValueError: an input doesn't fit its kind's shape, or the result isn't an array of the declared
        kind, or holds NaN or inf. Whatever the method itself raises is not caught.
    """
    m = METHODS[method]
    for j, (kind, value) in enumerate(zip(m.inputs, inputs)):
        metrics.check_input(kind, value, f"{method}: input {j}")
    value = m.function(*inputs, **(params or {}))
    metrics.check_input(m.output, value, f"{method}: the result")
    if not np.isfinite(value).all():
        raise ValueError(f"{method}: the result holds NaN or inf")
    return value.astype(np.float32, copy=False)


@derived(inputs=("orbit",), output="weights", settings={"k": 1.0, "flat": 0.0, "pass_number": 1})
def recent_approach_cosine_window(orbit: np.ndarray, k=1.0, flat=0.0, pass_number=1) -> np.ndarray:
    """Weights along an orbit's rows: a raised-cosine window centered on a close approach.

    The peak is at a pass: a row where the distance between the centers stops falling and starts
    rising. By default the most recent one (the pass nearest to now, t = 0). The window is 1 at the
    peak and falls as half a cosine on each side, to 0 at now and at `start = peak - k * (0 - peak)`;
    outside those it's 0. The peak is taken at a row as given, with no interpolation: whoever made the
    orbit chose its spacing.

    Args:
        orbit: shape (n, 7), columns x, y, z, vx, vy, vz, t: the secondary's center, primary at the
            origin. Times must run strictly one way (either order), with the pass before t = 0.
        k: how far back the window reaches, as a multiple of the time from the peak to now. 1 is
            symmetric; larger reaches further into the past.
        flat: the fraction of each side that stays at 1 before the cosine starts, 0 to 1. 0 (the
            default) is a pure cosine; 1 is a box.
        pass_number: which pass is the peak: 1 the most recent, 2 the one before it, and so on.

    Returns:
        One weight per row, shape (n,), float32 in [0, 1], in the rows' own order.

    Raises:
        ValueError: a bad setting; times that don't run one way; no such pass in the orbit; a pass
        at or after t = 0; or a window that would start before the orbit's first time.
    """
    if not k > 0:
        raise ValueError(f"k must be positive: {k!r}")
    if not 0 <= flat <= 1:
        raise ValueError(f"flat must be between 0 and 1: {flat!r}")
    if isinstance(pass_number, bool) or not isinstance(pass_number, (int, np.integer)) or pass_number < 1:
        raise ValueError(f"pass_number must be a whole number of at least 1: {pass_number!r}")
    t = orbit[:, 6].astype(float)
    steps = np.diff(t)
    if len(t) < 3 or not ((steps > 0).all() or (steps < 0).all()):
        raise ValueError("the orbit needs at least 3 rows, with times running strictly one way")
    distance = np.linalg.norm(orbit[:, :3].astype(float), axis=1)
    if steps[0] < 0:  # work forward in time; the weights go back to the rows' own order at the end
        t, distance = t[::-1], distance[::-1]
    passes = np.flatnonzero((distance[1:-1] < distance[:-2]) & (distance[1:-1] <= distance[2:])) + 1
    if len(passes) < pass_number:
        raise ValueError(
            f"pass {pass_number} isn't in the orbit: it has {len(passes)} pass(es) "
            f"(local minima of the distance between the centers) from t = {t[0]:g} to {t[-1]:g}"
        )
    peak, end = t[passes[-pass_number]], 0.0
    if not peak < end:
        raise ValueError(f"the pass is at t = {peak:g}, not before now (t = 0)")
    start = peak - k * (end - peak)
    if start < t[0] - 1e-9 * max(1.0, abs(t[0])):
        raise ValueError(
            f"the window would start at t = {start:g} (peak {peak:g}, k = {k:g}), before the orbit's "
            f"first time, {t[0]:g}"
        )
    weights = np.zeros(len(t))
    for edge in (start, end):  # one side at a time: from the peak out to where the weight reaches 0
        side = abs(edge - peak)
        away = (t - peak) / (edge - peak)  # 0 at the peak, 1 at the edge, negative on the other side
        on_side = (away >= 0) & (away <= 1)
        if flat >= 1:
            weights[on_side] = 1.0
        elif side > 0:
            along = np.clip((away[on_side] - flat) / (1 - flat), 0, 1)  # 0 through the flat part, then 0 -> 1
            weights[on_side] = 0.5 * (1 + np.cos(math.pi * along))
    if steps[0] < 0:
        weights = weights[::-1]
    return weights.astype(np.float32)


@derived(inputs=("image", "image"), output="image", settings={"floor": 0.05, "threshold": None})
def difference_weights(a: np.ndarray, b: np.ndarray, floor=0.05, threshold=None) -> np.ndarray:
    """A weight image that is large where two images differ: |a - b|, scaled so its largest value is 1.

    With a target and a model's unperturbed image, it marks where the target departs from undisturbed
    disks: the tidal features, and where the disks themselves changed. The model can't shrink weights
    made this way, since neither image is the model's.

    Args:
        a, b: two greyscale images of the same shape.
        floor: the smallest weight anywhere, 0 to 1. Above 0, material in a region where the two
            images agree (e.g. both empty) still counts a little.
        threshold: None, or 0 to 1: make the weights on/off, 1 where the scaled difference is above
            it and 0 elsewhere, before the floor is applied.

    Returns:
        The weight image, a's shape, float32 in [floor, 1]. Two identical images give `floor`
        everywhere.

    Raises:
        ValueError: the shapes differ, or a setting is out of range.
    """
    if a.shape != b.shape:
        raise ValueError(f"the images must have the same shape: {a.shape} vs {b.shape}")
    if not 0 <= floor <= 1:
        raise ValueError(f"floor must be between 0 and 1: {floor!r}")
    weights = _scaled(np.abs(a.astype(float) - b.astype(float)))
    weights = _on_off(weights, threshold)
    return np.maximum(weights, floor).astype(np.float32)


@derived(inputs=("image",), output="image", settings={"threshold": None})
def detect_edges(image: np.ndarray, threshold=None) -> np.ndarray:
    """An edge map: how fast an image's brightness changes at each pixel, scaled so the strongest edge is 1.

    The Sobel operator: the brightness gradient across columns and across rows, each smoothed over the
    three pixels alongside, combined as their length. The image's border is extended by its own edge
    pixels, so the border itself isn't an edge.

    Args:
        image: a greyscale image, shape (height, width).
        threshold: None, or 0 to 1: make the map on/off, 1 where the scaled edge strength is above it
            and 0 elsewhere.

    Returns:
        The edge map, the image's shape, float32 in [0, 1]. A flat image gives all zeros.

    Raises:
        ValueError: `threshold` is out of range.
    """
    p = np.pad(image.astype(float), 1, mode="edge")
    across = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    down = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    return _on_off(_scaled(np.hypot(across, down)), threshold).astype(np.float32)


def _scaled(values: np.ndarray) -> np.ndarray:
    """Divide by the largest value, so it becomes 1. All zeros stay zeros."""
    largest = values.max() if values.size else 0.0
    return values / largest if largest > 0 else values


def _on_off(values: np.ndarray, threshold) -> np.ndarray:
    """1 where a value is above `threshold` and 0 elsewhere; the values unchanged when it's None."""
    if threshold is None:
        return values
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float, np.integer, np.floating)) \
            or not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be None or between 0 and 1: {threshold!r}")
    return (values > threshold).astype(float)
