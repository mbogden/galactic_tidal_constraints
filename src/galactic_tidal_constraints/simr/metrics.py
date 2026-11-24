"""Metrics stage: one function per metric, computing a value from simulated or external arrays.

Like the other stages, it takes data in and returns data out; it never touches disk, pools or MPI.
Each method is registered with the `metric` decorator, which records everything about it in `METHODS`
(the metrics table, one source of truth): its kind, inputs, settings and their defaults, direction and
range, and for descriptive methods its output keys and shapes. `simr` looks methods up there and calls
them by name, and `table()` hands the same information to higher-level code.

Two kinds of metrics:
  scalar:      returns exactly one float (a fitness value, a perturbation measure, an orbit error).
  descriptive: returns a dict with everything about one thing (e.g. closest approach). Each declares
               its keys and their shapes; a key with shape () holds one float (`Metric.scalar_outputs`),
               and only those may be picked out by a config entry's `select`.

Methods return their natural, raw values (an error in SPAM units, a difference in [0, 1]); each
declares its direction (`best`) and `range` so a later normalization step can scale any method to a
chosen range. A method may also register a one-time `setup` step (load a network, pick a device),
run the first time it's used in a process; plain methods have none.

Inputs are declared by kind, not by item, so the same method works on arrays from any source (a
`simr` item, an external array, or a derived one), as long as the shape fits:
  orbit:     (n, 7) float32: x, y, z, vx, vy, vz, t (`sim.orbit`, `sim.search_orbit`)
  particles: (n, 6) float32: x, y, z, vx, vy, vz (`sim.particles`, `sim.unperturbed`)
  image:     (height, width) float32 in [0, 1] (`img.*`)
  weights:   (n,) float32: one value per row of the array it weighs

Weights are optional. A method that can use them says so (`Metric.weights` is True) and has a
`weights=None` argument; they come as one extra input after the declared ones. Without them every
row or pixel counts equally. Weights must be non-negative and not all zero, and match the method's
first input (the target, when there is one): one value per row of an orbit or of particles, or an
image of the same shape (`check_weights`, the one place this rule lives).

`composite` is the one place metric values are combined: a weighted mean of a model's scalar metric
cells across both levels, one number per model.
"""

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

# Each input kind's array shape (None: any length). `compute` checks every input against its declared
# kind before the method runs, so a wrong array fails cleanly wherever it came from (a simr item today,
# an external array later).
INPUT_SHAPES = {"orbit": (None, 7), "particles": (None, 6), "image": (None, None), "weights": (None,)}
INPUT_KINDS = tuple(INPUT_SHAPES)
KINDS = ("scalar", "descriptive")


@dataclass(frozen=True)
class Metric:
    """One method's row in the metrics table: everything declared about it, plus the function itself.

    Attributes:
        name: the method's name, as config entries give it.
        kind: "scalar" (returns one float) or "descriptive" (returns a dict).
        inputs: each input's kind, in the order the function takes them (from `INPUT_KINDS`).
        settings: the method's settings and their defaults, passed as keyword arguments after the
            inputs. {} for a method with none.
        best: "low" or "high": which raw value is better. None for a descriptive method.
        range: the raw value's (smallest, largest), inf allowed. None for a descriptive method.
        outputs: a descriptive method's keys and their shapes; () is one float. None for a scalar.
        function: the method itself: `function(*inputs, **settings)`.
        setup: an optional one-time step, run the first time the method is used in a process (e.g.
            load a network and its weights). None for plain methods.
        weights: True if the method can take weights, as one more input after `inputs`; they must
            match its first input (`weights_kind`, `check_weights`). False: it takes none.
    """

    name: str
    kind: str
    inputs: tuple[str, ...]
    settings: dict
    best: str | None
    range: tuple[float, float] | None
    outputs: dict[str, tuple[int, ...]] | None
    function: Callable
    setup: Callable | None = None
    weights: bool = False

    @property
    def scalar_outputs(self) -> tuple[str, ...]:
        """The descriptive keys holding one float: what a config entry's `select` may name."""
        return tuple(key for key, shape in (self.outputs or {}).items() if shape == ())


# Every registered method, by name: the metrics table, built by the `metric` decorator.
METHODS: dict[str, Metric] = {}


def metric(
    kind: str,
    inputs: tuple[str, ...] | list[str],
    *,
    settings: dict | None = None,
    best: str | None = None,
    range: tuple[float, float] | None = None,
    outputs: dict[str, tuple[int, ...]] | None = None,
    setup: Callable | None = None,
    weights: bool = False,
) -> Callable:
    """Register a metric method in `METHODS`, checking its declaration as the module is imported.

    A scalar method declares `best` and `range` and no `outputs`; a descriptive method declares
    `outputs` (key -> shape, () for one float) and neither `best` nor `range`. A method that can
    use weights declares `weights=True` and takes them as a `weights=None` argument.
    """

    def register(function: Callable) -> Callable:
        name = function.__name__
        if name in METHODS:
            raise ValueError(f"metric {name!r} is already registered")
        if kind not in KINDS:
            raise ValueError(f"{name}: kind must be one of {list(KINDS)}: {kind!r}")
        unknown = [k for k in inputs if k not in INPUT_KINDS]
        if not isinstance(weights, bool) or (weights and not inputs):
            raise ValueError(f"{name}: weights is True or False, and a method that takes them has an input to match")
        if unknown:
            raise ValueError(f"{name}: unknown input kinds {unknown}. Known: {list(INPUT_KINDS)}")
        if kind == "scalar":
            if outputs is not None:
                raise ValueError(f"{name}: a scalar method returns one float and declares no outputs")
            if best not in ("low", "high"):
                raise ValueError(f'{name}: a scalar method declares best="low" or "high": {best!r}')
            if range is None:
                raise ValueError(f"{name}: a scalar method declares its range (smallest, largest)")
        else:
            if not outputs:
                raise ValueError(f"{name}: a descriptive method declares its output keys and shapes")
            if best is not None or range is not None:
                raise ValueError(f"{name}: best and range describe a scalar method's value only")
        METHODS[name] = Metric(
            name=name,
            kind=kind,
            inputs=tuple(inputs),
            settings=dict(settings or {}),
            best=best,
            range=None if range is None else (float(range[0]), float(range[1])),
            outputs=None if outputs is None else {key: tuple(shape) for key, shape in outputs.items()},
            function=function,
            setup=setup,
            weights=weights,
        )
        return function

    return register


def table() -> dict[str, dict]:
    """The metrics table as plain data (JSON-ready), one entry per method, for checks and higher-level
    code: kind, inputs, weights (whether it can take them), settings, best, range and outputs, as
    declared."""
    return {
        name: {
            "kind": m.kind,
            "inputs": list(m.inputs),
            "weights": m.weights,
            "settings": dict(m.settings),
            "best": m.best,
            "range": None if m.range is None else list(m.range),
            "outputs": None if m.outputs is None else {key: list(shape) for key, shape in m.outputs.items()},
        }
        for name, m in METHODS.items()
    }


def check_input(kind: str, value, where: str) -> None:
    """Check one array against its input kind's shape (`INPUT_SHAPES`). `compute` runs it on every
    input; `simr` runs it on each external array before anything is simulated, so both give the same
    message.

    Args:
        kind: one of `INPUT_KINDS`.
        value: the array to check.
        where: what the array is, to start the error message (e.g. "pca_shape: input 0").

    Raises:
        ValueError: `value` isn't a numpy array of that kind's shape.
    """
    shape = INPUT_SHAPES[kind]
    if not isinstance(value, np.ndarray) or value.ndim != len(shape) or any(
        want is not None and got != want for got, want in zip(value.shape, shape)
    ):
        got = value.shape if isinstance(value, np.ndarray) else type(value).__name__
        raise ValueError(f"{where} must be a {kind!r} array, shape {shape}: got {got}")


def compute(method: str, inputs: list[np.ndarray], select: str | None = None, params: dict | None = None):
    """Run one registered method on its input arrays and check the value against its declaration.

    Args:
        method: a name from `METHODS`.
        inputs: the input arrays, one per declared kind, in order. A method that takes weights
            (`Metric.weights`) accepts one more at the end: its weights.
        select: for a descriptive method, the single-float output to return instead of the dict.
        params: settings passed to the method as keyword arguments (defaults for those left out come
            from the entry that was filled in against the declaration, not from here).

    Returns:
        The method's float (a scalar method, or `select`), or its dict (a descriptive method).

    Raises:
        ValueError: the wrong number of inputs; an input doesn't fit its kind's shape
        (`INPUT_SHAPES`); weights that are negative, not finite or all zero; the value doesn't match
        the declaration, or it holds NaN or inf — a metric failure. Whatever the method itself raises
        is not caught.
    """
    m = METHODS[method]
    allowed = (len(m.inputs), len(m.inputs) + 1) if m.weights else (len(m.inputs),)
    if len(inputs) not in allowed:
        raise ValueError(
            f"{method} takes {len(m.inputs)} input(s) {list(m.inputs)}"
            + (f", and optionally {weights_kind(m)!r} weights after them" if m.weights else "")
            + f": got {len(inputs)}"
        )
    for j, (kind, value) in enumerate(zip(m.inputs, inputs)):
        check_input(kind, value, f"{method}: input {j}")
    keywords = dict(params or {})
    if len(inputs) > len(m.inputs):
        check_input(weights_kind(m), inputs[-1], f"{method}: the weights")
        keywords["weights"] = check_weights(inputs[-1], inputs[0], m.inputs[0], f"{method}: the weights")
    value = m.function(*inputs[: len(m.inputs)], **keywords)
    if m.kind == "scalar":
        if not isinstance(value, (int, float, np.floating, np.integer)) or isinstance(value, bool):
            raise ValueError(f"{method} returned {type(value).__name__}, not one float")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{method} returned {value}")
        return value
    if not isinstance(value, dict) or set(value) != set(m.outputs or {}):
        raise ValueError(f"{method} must return a dict with exactly the keys {sorted(m.outputs or {})}")
    for key, shape in (m.outputs or {}).items():
        if np.shape(value[key]) != shape:
            raise ValueError(f"{method}[{key!r}] must have shape {shape}: got {np.shape(value[key])}")
        if not np.isfinite(np.asarray(value[key], dtype=float)).all():
            raise ValueError(f"{method}[{key!r}] holds NaN or inf")
    return float(value[select]) if select is not None else value


def weights_kind(method: Metric) -> str | None:
    """The kind of weights a method takes, which follows from its first input: an image for an image,
    else "weights" (one value per row). None for a method that takes no weights."""
    if not method.weights:
        return None
    return "image" if method.inputs[0] == "image" else "weights"


def check_weights(weights: np.ndarray, first: np.ndarray, kind: str, where: str) -> np.ndarray:
    """The one rule for every metric's weights: they match the method's first input, and are usable.

    Args:
        weights: the weights given.
        first: the method's first input (the target, when there is one), which they weigh.
        kind: `first`'s kind: for an "image" the weights have its shape; for an "orbit" or
            "particles", one value per row.
        where: what the weights are, to start the error message.

    Returns:
        The weights as float64.

    Raises:
        ValueError: they don't match `first`; or they aren't finite, hold a negative value, or are
        all zero. A derived array may be all zeros; it can't be used as weights.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if kind == "image":
        if weights.shape != first.shape:
            raise ValueError(f"{where} must have the images' shape {first.shape}: got {weights.shape}")
    elif weights.shape != (len(first),):
        raise ValueError(f"{where} must be one per row of the first input, {len(first)}: got {weights.shape}")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError(f"{where} must be finite and not negative")
    if not (weights > 0).any():
        raise ValueError(f"{where} are all zero: nothing would be compared")
    return weights


@metric(
    kind="descriptive",
    inputs=("orbit",),
    outputs={"t": (), "distance": (), "position": (3,), "velocity": (3,)},
)
def closest_approach(orbit: np.ndarray) -> dict:
    """The point along an orbit where the two centers are closest.

    The smallest distance in whatever orbit it's given, so the entry's input decides the window:
    `sim.search_orbit` is the whole window from now back to t = -30, `sim.orbit` is the run itself
    (the simulator config's `start` decides). Row order doesn't matter (`sim.search_orbit`'s rows run
    backward in time).

    Args:
        orbit: the secondary's center over time, (n, 7): x, y, z, vx, vy, vz, t.

    Returns:
        "t" and "distance" (floats), "position" and "velocity" (arrays of shape (3,)) at that row.
    """
    distance = np.linalg.norm(orbit[:, :3], axis=1)
    row = int(np.argmin(distance))
    return {
        "t": float(orbit[row, 6]),
        "distance": float(distance[row]),
        "position": orbit[row, 0:3].copy(),
        "velocity": orbit[row, 3:6].copy(),
    }


@metric(kind="scalar", inputs=("image", "image"), best="low", range=(0.0, 1.0), weights=True)
def absolute_difference(a: np.ndarray, b: np.ndarray, weights: np.ndarray | None = None) -> float:
    """The mean |a - b| over the pixels of two greyscale images of the same shape.

    0 means identical, 1 the largest possible difference between images in [0, 1]. It works for any
    two matching images, wherever they come from: two `simr` images of one simulation (model vs
    unperturbed reads as how perturbed the model is), or a `simr` image against an external one (a
    target).

    Args:
        a, b: greyscale images, (height, width), values in [0, 1].
        weights: None (every pixel counts equally), or a weight per pixel, the images' shape: the
            value is then the weighted mean, sum(w * |a - b|) / sum(w). With a target, a model and
            weights from `derived.difference_weights(target, unperturbed)`, differences count most
            where the target shows tidal features.
    """
    if a.shape != b.shape:
        raise ValueError(f"the images must have the same shape: {a.shape} vs {b.shape}")
    difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
    if weights is None:
        return float(np.mean(difference))
    weights = check_weights(weights, a, "image", "the weights")
    return float((weights * difference).sum() / weights.sum())


# The smallest (weighted) pixel variance a correlation accepts: below it the image counts as flat, so
# rounding noise in a constant image is never mistaken for a pattern. Images are in [0, 1].
CORRELATION_MIN_VARIANCE = 1e-20


@metric(kind="scalar", inputs=("image", "image"), best="high", range=(-1.0, 1.0), weights=True)
def correlation(a: np.ndarray, b: np.ndarray, weights: np.ndarray | None = None) -> float:
    """The Pearson correlation of two greyscale images' pixels: how alike their patterns are, whatever
    their overall brightness or contrast.

    1 means one image is the other scaled and shifted (b = c * a + d, c > 0), 0 no linear relation,
    -1 the negative. Unlike `absolute_difference`, a model image that's uniformly dimmer than the
    target still scores 1. Best is high, where the difference metrics are best low.

    Args:
        a, b: greyscale images of the same shape, (height, width).
        weights: None (every pixel counts equally), or a weight per pixel, the images' shape: the
            means, variances and covariance are then weighted ones, so the pattern counts most where
            the weights are large (e.g. `derived.difference_weights(target, unperturbed)`).

    Raises:
        ValueError: the shapes differ; or an image has no variation (where the weights count), so the
        correlation is undefined: a metric failure, not 0.
    """
    if a.shape != b.shape:
        raise ValueError(f"the images must have the same shape: {a.shape} vs {b.shape}")
    share = np.ones(a.shape) if weights is None else check_weights(weights, a, "image", "the weights")
    share = share / share.sum()
    from_mean_a = a.astype(np.float64) - (share * a).sum()
    from_mean_b = b.astype(np.float64) - (share * b).sum()
    variance_a, variance_b = (share * from_mean_a**2).sum(), (share * from_mean_b**2).sum()
    if variance_a < CORRELATION_MIN_VARIANCE or variance_b < CORRELATION_MIN_VARIANCE:
        flat = " and ".join(name for name, variance in (("the first", variance_a), ("the second", variance_b))
                            if variance < CORRELATION_MIN_VARIANCE)
        raise ValueError(f"the correlation is undefined: {flat} image has no variation where the pixels count")
    value = (share * from_mean_a * from_mean_b).sum() / math.sqrt(variance_a * variance_b)
    return float(min(1.0, max(-1.0, value)))  # rounding can step a hair outside [-1, 1]


# --- Orbit errors: a model's orbit against a target's ---------------------------------------------------

# The most squared distances held at once while searching for nearest points (about 32 MB as float64).
ORBIT_SEARCH_CHUNK = 4_000_000


def _orbit_pair(target: np.ndarray, model: np.ndarray, weights: np.ndarray | None):
    """What both orbit errors need, checked: the target rows that count, each one's share of the
    weighted time, and the model's rows in time order.

    Each target row stands for the time around it (half the gap to the row before plus half the gap to
    the row after: the trapezoid rule), so the result is a weighted average over time, and doesn't
    depend on how the target was sampled.

    Returns (positions, times, shares, model): the counted target rows' x, y, z and t, their shares
    (weight x time, summing to 1), and the model orbit sorted by time.
    """
    times = target[:, 6].astype(np.float64)
    if len(times) < 2 or not (np.diff(times) > 0).all():
        raise ValueError("the target orbit needs at least 2 rows, with times strictly increasing")
    if weights is None:
        weights = np.ones(len(times))
    weights = check_weights(weights, target, "orbit", "the weights")
    gaps = np.diff(times)
    spans = np.concatenate(([gaps[0] / 2], (gaps[:-1] + gaps[1:]) / 2, [gaps[-1] / 2]))
    shares = weights * spans
    counted = shares > 0
    model_times = model[:, 6].astype(np.float64)
    steps = np.diff(model_times)
    if len(model_times) < 2 or not ((steps > 0).all() or (steps < 0).all()):
        raise ValueError("the model orbit needs at least 2 rows, with times running strictly one way")
    if steps[0] < 0:
        model = model[::-1]
        model_times = model_times[::-1]
    first, last = times[counted][0], times[counted][-1]
    slack = np.abs(steps).max() / 2  # a target time may sit half a model step past the model's ends
    if first < model_times[0] - slack or last > model_times[-1] + slack:
        raise ValueError(
            f"the model orbit runs from t = {model_times[0]:g} to {model_times[-1]:g}, but the target "
            f"rows that count run from t = {first:g} to {last:g}"
        )
    return target[counted, :3].astype(np.float64), times[counted], shares[counted] / shares[counted].sum(), model


@metric(kind="scalar", inputs=("orbit", "orbit"), best="low", range=(0.0, math.inf), weights=True)
def orbit_time_error(target: np.ndarray, model: np.ndarray, weights: np.ndarray | None = None) -> float:
    """How far a model's orbit is from a target's at the same times: the weighted time average of the
    squared distance between the two, in SPAM length units squared.

    For each target row, the model's position at the nearest model time is used. 0 means the model is
    where the target is, when the target is there. A model that follows the right path at the wrong
    times scores badly here and well on `orbit_position_error`.

    Args:
        target: the orbit to match, (n, 7): x, y, z, vx, vy, vz, t, with times strictly increasing and
            any spacing.
        model: the model's orbit, (m, 7), e.g. `sim.search_orbit`; rows in either time order. It must
            cover the target times that count.
        weights: None (every target time counts equally), or one weight per target row, e.g. from
            `derived.recent_approach_cosine_window(target)`.
    """
    positions, times, shares, model = _orbit_pair(target, model, weights)
    model_times = model[:, 6].astype(np.float64)
    after = np.clip(np.searchsorted(model_times, times), 1, len(model_times) - 1)
    nearest = np.where(times - model_times[after - 1] <= model_times[after] - times, after - 1, after)
    squared = ((positions - model[nearest, :3].astype(np.float64)) ** 2).sum(axis=1)
    return float((shares * squared).sum())


@metric(kind="scalar", inputs=("orbit", "orbit"), best="low", range=(0.0, math.inf), weights=True)
def orbit_position_error(target: np.ndarray, model: np.ndarray, weights: np.ndarray | None = None) -> float:
    """How far a target's path is from a model's path, whatever the timing: the weighted time average
    of the squared distance from each target point to the nearest model point, in SPAM length units
    squared.

    Only the model's rows inside the span of the target times that count are searched, so on a
    repeating orbit a target point can't match an earlier lap. 0 means every counted target point
    lies on the model's path. The nearest model row is used, with no interpolation between rows.

    Args:
        target, model, weights: as in `orbit_time_error`.
    """
    positions, times, shares, model = _orbit_pair(target, model, weights)
    model_times = model[:, 6].astype(np.float64)
    slack = np.abs(np.diff(model_times)).max() / 2
    inside = (model_times >= times[0] - slack) & (model_times <= times[-1] + slack)
    path = model[inside, :3].astype(np.float64)
    squared = np.empty(len(positions))
    rows = max(1, ORBIT_SEARCH_CHUNK // max(1, len(path)))
    for start in range(0, len(positions), rows):
        block = positions[start:start + rows]
        squared[start:start + rows] = ((block[:, None, :] - path[None, :, :]) ** 2).sum(axis=2).min(axis=1)
    return float((shares * squared).sum())


# --- PCA shape: the particles' covariance ellipsoid ------------------------------------------------------

PCA_COLUMNS = {"position": slice(0, 3), "velocity": slice(3, 6)}
PCA_CENTERS = ("mean", "median")
PCA_DESCRIPTORS = ("ratios", "axes", "geometric")
PCA_DISTANCES = ("l1", "l2")
PCA_EPS = 1e-12  # floor on each eigenvalue, so a line, a sheet or a single point never divides by zero
PCA_SETTINGS = {"columns": "position", "center": "mean", "trim_quantile": None}


def _pca(particles: np.ndarray, columns: str, center: str, trim_quantile: float | None) -> dict:
    """The principal axes of a particle cloud: semi-axes, unit vectors, center. Never changes its input."""
    if columns not in PCA_COLUMNS:
        raise ValueError(f"columns must be one of {list(PCA_COLUMNS)}: {columns!r}")
    if center not in PCA_CENTERS:
        raise ValueError(f"center must be one of {list(PCA_CENTERS)}: {center!r}")
    if particles.ndim != 2 or particles.shape[1] != 6:
        raise ValueError(f"particles must have shape (n, 6): got {particles.shape}")
    points = particles[:, PCA_COLUMNS[columns]].astype(np.float64)

    def middle(p):
        return p.mean(axis=0) if center == "mean" else np.median(p, axis=0)

    if trim_quantile is not None:
        if not 0.0 < trim_quantile <= 1.0:
            raise ValueError(f"trim_quantile must be in (0, 1] or None: {trim_quantile!r}")
        distance = np.linalg.norm(points - middle(points), axis=1)
        points = points[distance <= np.quantile(distance, trim_quantile)]
    if len(points) < 3:
        raise ValueError(f"PCA needs at least 3 particles: got {len(points)}")
    c = middle(points)
    offsets = points - c
    values, vectors = np.linalg.eigh(offsets.T @ offsets / len(points))  # ascending
    order = np.argsort(values)[::-1]
    axes = np.sqrt(np.maximum(values[order], PCA_EPS))
    vectors = vectors[:, order].T  # rows, largest axis first
    # An eigenvector's sign is arbitrary: make each one's largest component positive, so the output
    # is the same on every run and machine.
    signs = np.sign(vectors[np.arange(3), np.argmax(np.abs(vectors), axis=1)])
    vectors = vectors * signs[:, None]
    return {"axes": axes, "vectors": vectors, "center": c}


@metric(
    kind="descriptive",
    inputs=("particles",),
    settings=PCA_SETTINGS,
    outputs={"axes": (3,), "vectors": (3, 3), "center": (3,), "b_a": (), "c_a": ()},
)
def pca_shape(particles: np.ndarray, columns="position", center="mean", trim_quantile=None) -> dict:
    """The shape of a particle cloud as its covariance ellipsoid (principal component analysis).

    The whole array as given: for `sim.particles` that's both galaxies together.

    Args:
        particles: (n, 6): x, y, z, vx, vy, vz. At least 3 particles (after trimming).
        columns: "position" (columns 0-2) or "velocity" (3-5).
        center: "mean" or "median" of the points.
        trim_quantile: drop the particles farther from the center than this quantile of the
            distances before the PCA (e.g. 0.95 drops the farthest 5%). None keeps them all: tidal
            tails are the outliers.

    Returns:
        "axes" (3,): the semi-axes (standard deviation along each principal axis), largest first.
        "vectors" (3, 3): the matching unit vectors as rows, each with its largest component positive.
        "center" (3,): the center used.
        "b_a", "c_a": the axis ratios (middle / largest, smallest / largest), in [0, 1].
    """
    pca = _pca(particles, columns, center, trim_quantile)
    a, b, c = pca["axes"]
    return {**pca, "b_a": float(b / a), "c_a": float(c / a)}


@metric(
    kind="scalar",
    inputs=("particles", "particles"),
    settings={**PCA_SETTINGS, "descriptor": "ratios", "distance": "l2", "orientation_weight": 0.0},
    best="low",
    range=(0.0, math.inf),
)
def pca_difference(
    a: np.ndarray,
    b: np.ndarray,
    columns="position",
    center="mean",
    trim_quantile=None,
    descriptor="ratios",
    distance="l2",
    orientation_weight=0.0,
) -> float:
    """How different two particle clouds' shapes are, from their PCA ellipsoids (`pca_shape`).

    0 means the same shape. Both clouds get the same `columns`, `center` and `trim_quantile`; they
    may hold different numbers of particles.

    Args:
        a, b: (n, 6) each: x, y, z, vx, vy, vz.
        columns, center, trim_quantile: as in `pca_shape`.
        descriptor: what's compared. "ratios": [b/a, c/a], blind to size, position and rotation.
            "axes": the semi-axes themselves, so size counts. "geometric": [linearity, planarity,
            sphericity] = [(a-b)/a, (b-c)/a, c/a].
        distance: "l2" (Euclidean) or "l1" between the two descriptors.
        orientation_weight: w >= 0 adds w * (1 - |v1_a . v1_b|), how far apart the two largest axes
            point (0 aligned, 1 perpendicular). 0 compares shape alone.

    Returns:
        The distance. With the defaults it lies in [0, sqrt(2)].
    """
    if descriptor not in PCA_DESCRIPTORS:
        raise ValueError(f"descriptor must be one of {list(PCA_DESCRIPTORS)}: {descriptor!r}")
    if distance not in PCA_DISTANCES:
        raise ValueError(f"distance must be one of {list(PCA_DISTANCES)}: {distance!r}")
    if not orientation_weight >= 0.0:
        raise ValueError(f"orientation_weight must be >= 0: {orientation_weight!r}")
    pcas = [_pca(p, columns, center, trim_quantile) for p in (a, b)]

    def describe(axes):
        big, mid, small = axes
        if descriptor == "ratios":
            return np.array([mid / big, small / big])
        if descriptor == "axes":
            return axes
        return np.array([(big - mid) / big, (mid - small) / big, small / big])

    gap = describe(pcas[0]["axes"]) - describe(pcas[1]["axes"])
    value = np.abs(gap).sum() if distance == "l1" else np.sqrt((gap**2).sum())
    if orientation_weight:
        value += orientation_weight * (1.0 - abs(float(pcas[0]["vectors"][0] @ pcas[1]["vectors"][0])))
    return float(value)


# --- Composite: one number per model from the scalar metrics ---------------------------------------------


def composite(
    sim_metrics: np.ndarray | None = None,
    img_metrics: np.ndarray | None = None,
    sim_weights: np.ndarray | None = None,
    img_weights: np.ndarray | None = None,
) -> np.ndarray:
    """One composite score per model: the weighted mean of its metric cells, across both levels.

        composite[m] = (sum(sim_weights * sim_metrics[m]) + sum(img_weights * img_metrics[m]))
                       / (sum(sim_weights) + sum(img_weights))

    The weights needn't sum to 1. A cell with weight 0 is skipped entirely, so it may hold anything (a
    descriptive entry's dict or None, or a failed scalar's NaN). A NaN in a cell with a nonzero weight
    makes that model's composite NaN, and only that model's. Whether a mix makes sense (directions,
    scales) is the caller's choice; this only applies the weights.

    Args:
        sim_metrics: [model, simulator, entry], as `SimrResult.sim_metrics` (an object array is fine).
        img_metrics: [model, simulator, image, entry], as `SimrResult.img_metrics`.
        sim_weights: one weight per cell of a model, shape (simulator, entry). None: 1 for every cell.
        img_weights: likewise, shape (simulator, image, entry).
        At least one level must be given; weights for a level that isn't given are an error.

    Returns:
        float64, shape (model,).

    Raises:
        ValueError: no level given; weights without their level, of the wrong shape, negative, not
        finite, or all zero across both levels; the levels disagree on the number of models; a cell
        with a nonzero weight holds something other than one number (e.g. a dict).
    """
    levels = []
    for level, values, weights, axes in (("sim", sim_metrics, sim_weights, "(simulator, entry)"),
                                          ("img", img_metrics, img_weights, "(simulator, image, entry)")):
        if values is None:
            if weights is not None:
                raise ValueError(f"{level}_weights were given without {level}_metrics")
            continue
        values = np.asarray(values)
        if values.ndim != len(axes.split(",")) + 1:
            raise ValueError(f"{level}_metrics must be [model, {axes[1:-1]}]: got shape {values.shape}")
        weights = np.ones(values.shape[1:]) if weights is None else np.asarray(weights, dtype=np.float64)
        if weights.shape != values.shape[1:]:
            raise ValueError(
                f"{level}_weights must have one weight per cell of a model, shape {axes} = {values.shape[1:]}: "
                f"got {weights.shape}"
            )
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError(f"{level}_weights must be finite and not negative")
        levels.append((level, values, weights))
    if not levels:
        raise ValueError("composite needs sim_metrics, img_metrics or both")
    if len({len(values) for _, values, _ in levels}) > 1:
        raise ValueError(f"the levels have different numbers of models: {[len(v) for _, v, _ in levels]}")
    total = sum(float(weights.sum()) for _, _, weights in levels)
    if total <= 0:
        raise ValueError("the weights are all zero: nothing would be combined")
    result = np.zeros(len(levels[0][1]))
    for level, values, weights in levels:
        used = weights > 0
        result += _cells_as_floats(values[:, used], level) @ weights[used]
    return result / total


def _cells_as_floats(cells: np.ndarray, level: str) -> np.ndarray:
    """The cells a composite uses, (model, cell), as float64; each must hold one number (NaN allowed)."""
    if cells.dtype != object:
        return cells.astype(np.float64)
    floats = np.empty(cells.shape)
    for index, value in np.ndenumerate(cells):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(
                f"{level}_metrics: a cell with a nonzero weight holds {type(value).__name__}, not one number "
                f"(a descriptive entry's dict, or its None, must have weight 0)"
            )
        floats[index] = value
    return floats
