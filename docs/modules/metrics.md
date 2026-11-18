# Metrics stage (`simr/metrics.py`)

The Metrics stage computes one value per method from simulated or external arrays. Like the other
stages it takes data in and returns data out; it never touches disk, pools or MPI. SIMR is a pipeline
for generating many relevant metrics; "fitness" keeps its meaning as what's optimized, usually a
composite of metrics (a later step).

Only what exists today is described here. Decisions behind it are in
`Experiments/MDRN-4` (the vault) and [PLAN.md](../../PLAN.md).

## Two kinds of metrics
- **Scalar**: returns exactly one float (a fitness value, a perturbation measure, an orbit error).
- **Descriptive**: returns a dict with everything about one thing (e.g. closest approach). Each
  declares its keys and their shapes; a key with shape `()` holds one float
  (`Metric.scalar_outputs`), and only those may be picked out by a config entry's `select` (used by
  `simr`'s metric entries, step 2b).

Methods return their natural, **raw** values; each scalar method declares its direction (`best`) and
`range`, so a later normalization step can scale any method to a chosen range.

## The registry: the metrics table in code
Each method is registered with the `metric` decorator, where it's defined — one source of truth. The
declaration is checked at import (a scalar method must declare `best` and `range` and no `outputs`;
a descriptive method the reverse), and recorded in `metrics.METHODS` as a `Metric`:

| Field | What it declares |
|---|---|
| `kind` | `"scalar"` or `"descriptive"` |
| `inputs` | Each input's kind, in the order the function takes them |
| `settings` | The method's settings and their defaults (`{}` for none), passed as keyword arguments |
| `best`, `range` | A scalar method's direction (`"low"` or `"high"`) and raw `(smallest, largest)` |
| `outputs` | A descriptive method's keys and shapes; `()` is one float |
| `function` | The method itself: `function(*inputs, **settings)` |
| `setup` | An optional one-time step (e.g. load a network), run the first time the method is used in a process; `None` for plain methods |

`metrics.table()` returns the same information as plain, JSON-ready data, for `simr`'s early checks
and for higher-level programs.

## Input kinds
A method declares its inputs by kind, not by item, so the same method works on arrays from any
source (a `simr` item, or an external array once those exist), as long as the shape fits:

| Kind | Shape | `simr` items |
|---|---|---|
| `orbit` | `(n, 7)` float32: x, y, z, vx, vy, vz, t | `sim.orbit`, `sim.search_orbit` |
| `particles` | `(n, 6)` float32: x, y, z, vx, vy, vz | `sim.particles`, `sim.unperturbed` |
| `image` | `(height, width)` float32 in [0, 1] | `img.*` |

## The methods
This table is guarded by a test (`tests/test_metrics.py`) against `metrics.table()`.

| Method | Kind | Inputs | Returns | Best | Range | Settings |
|---|---|---|---|---|---|---|
| `closest_approach` | descriptive | orbit | `t` (), `distance` (), `position` (3,), `velocity` (3,) | — | — | — |
| `absolute_difference` | scalar | image, image | one float | low | [0, 1] | — |

- **`closest_approach`**: the point along an orbit where the two centers are closest — the smallest
  distance in whatever orbit it's given, so the entry's input decides the window (`sim.search_orbit`:
  the whole window from now back to t = −30; `sim.orbit`: the run itself, so the simulator config's
  `start` decides). Row order doesn't matter.
- **`absolute_difference`**: the mean |a − b| over the pixels of two greyscale images of the same
  shape; 0 is identical. Any two matching images work: model vs unperturbed reads as how perturbed
  the model is, and an image vs an external target (once externals exist) as how far from the target.
  Images of another shape are an error; matching frames is the caller's job.
