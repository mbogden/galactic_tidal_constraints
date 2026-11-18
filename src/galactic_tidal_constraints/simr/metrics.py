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

from dataclasses import dataclass
from typing import Callable

import numpy as np

INPUT_KINDS = ("orbit", "particles", "image")
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
