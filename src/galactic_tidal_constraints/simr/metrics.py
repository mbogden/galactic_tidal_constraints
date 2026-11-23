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
`simr` item, or an external array once those exist), as long as the shape fits:
  orbit:     (n, 7) float32: x, y, z, vx, vy, vz, t (`sim.orbit`, `sim.search_orbit`)
  particles: (n, 6) float32: x, y, z, vx, vy, vz (`sim.particles`, `sim.unperturbed`)
  image:     (height, width) float32 in [0, 1] (`img.*`)
"""

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

# Each input kind's array shape (None: any length). `compute` checks every input against its declared
# kind before the method runs, so a wrong array fails cleanly wherever it came from (a simr item today,
# an external array later).
INPUT_SHAPES = {"orbit": (None, 7), "particles": (None, 6), "image": (None, None)}
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
) -> Callable:
    """Register a metric method in `METHODS`, checking its declaration as the module is imported.

    A scalar method declares `best` and `range` and no `outputs`; a descriptive method declares
    `outputs` (key -> shape, () for one float) and neither `best` nor `range`.
    """

    def register(function: Callable) -> Callable:
        name = function.__name__
        if name in METHODS:
            raise ValueError(f"metric {name!r} is already registered")
        if kind not in KINDS:
            raise ValueError(f"{name}: kind must be one of {list(KINDS)}: {kind!r}")
        unknown = [k for k in inputs if k not in INPUT_KINDS]
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
        )
        return function

    return register


def table() -> dict[str, dict]:
    """The metrics table as plain data (JSON-ready), one entry per method, for checks and higher-level
    code: kind, inputs, settings, best, range and outputs, as declared."""
    return {
        name: {
            "kind": m.kind,
            "inputs": list(m.inputs),
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
        inputs: the input arrays, one per declared kind, in order.
        select: for a descriptive method, the single-float output to return instead of the dict.
        params: settings passed to the method as keyword arguments (defaults for those left out come
            from the entry that was filled in against the declaration, not from here).

    Returns:
        The method's float (a scalar method, or `select`), or its dict (a descriptive method).

    Raises:
        ValueError: an input doesn't fit its kind's shape (`INPUT_SHAPES`), the value doesn't match
        the declaration, or it holds NaN or inf — a metric failure. Whatever the method itself raises
        is not caught.
    """
    m = METHODS[method]
    for j, (kind, value) in enumerate(zip(m.inputs, inputs)):
        check_input(kind, value, f"{method}: input {j}")
    value = m.function(*inputs, **(params or {}))
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


@metric(kind="scalar", inputs=("image", "image"), best="low", range=(0.0, 1.0))
def absolute_difference(a: np.ndarray, b: np.ndarray) -> float:
    """The mean |a - b| over the pixels of two greyscale images of the same shape.

    0 means identical, 1 the largest possible difference between images in [0, 1]. It works for any
    two matching images, wherever they come from: two `simr` images of one simulation (model vs
    unperturbed reads as how perturbed the model is), or a `simr` image against an external one (a
    target) once externals exist.

    Args:
        a, b: greyscale images, (height, width), values in [0, 1].
    """
    if a.shape != b.shape:
        raise ValueError(f"the images must have the same shape: {a.shape} vs {b.shape}")
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


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
