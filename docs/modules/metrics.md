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

`metrics.compute(method, inputs, select=None, params=None)` runs one method on its input arrays and
checks the value against the declaration: a scalar must be one finite float; a descriptive dict must
have exactly the declared keys with the declared shapes and no NaN or inf anywhere. A value that fails
is a `ValueError` — in `simr`, a metric failure. With `select`, the named single-float output comes
back instead of the dict.

## In `simr`
The `sim_metrics` and `img_metrics` config sections list entries (`method`, `inputs`, optional
`select` and `params`), checked against the table before anything runs; the values come back in
`SimrResult.sim_metrics` `[model, simulator, entry]` and `.img_metrics` `[model, simulator, image,
entry]`, with per-cell status, time and error. Details:
[simr.md](simr.md#the-result) and [configuration.md](../configuration.md#metric-entries).

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
| `pca_shape` | descriptive | particles | `axes` (3,), `vectors` (3, 3), `center` (3,), `b_a` (), `c_a` () | — | — | `columns="position"`, `center="mean"`, `trim_quantile=None` |
| `pca_difference` | scalar | particles, particles | one float | low | [0, ∞) | as `pca_shape`, plus `descriptor="ratios"`, `distance="l2"`, `orientation_weight=0.0` |

- **`closest_approach`**: the point along an orbit where the two centers are closest — the smallest
  distance in whatever orbit it's given, so the entry's input decides the window (`sim.search_orbit`:
  the whole window from now back to t = −30; `sim.orbit`: the run itself, so the simulator config's
  `start` decides). Row order doesn't matter.
- **`absolute_difference`**: the mean |a − b| over the pixels of two greyscale images of the same
  shape; 0 is identical. Any two matching images work: model vs unperturbed reads as how perturbed
  the model is, and an image vs an external target (once externals exist) as how far from the target.
  Images of another shape are an error; matching frames is the caller's job.
- **`pca_shape`**: the shape of a particle cloud as its covariance ellipsoid (3D principal component
  analysis), on the whole array as given (for `sim.particles`, both galaxies together). `columns`
  picks positions or velocities; `center` is the mean or median; `trim_quantile` (e.g. 0.95) drops
  the farthest particles first, off by default because tidal tails are the outliers. `axes` are the
  semi-axes (standard deviations along each principal axis), largest first; `vectors` are the
  matching unit vectors as rows, each with its largest component positive so the output is
  deterministic; `b_a` and `c_a` are the axis ratios. Needs at least 3 particles; lines, sheets and
  repeated points stay finite (each eigenvalue has a tiny floor).
- **`pca_difference`**: how different two clouds' shapes are, from their `pca_shape` ellipsoids;
  0 is the same shape, and the clouds may hold different numbers of particles. `descriptor`:
  `"ratios"` `[b/a, c/a]` (blind to size, position and rotation), `"axes"` (size counts) or
  `"geometric"` `[linearity, planarity, sphericity]`; `distance` is `"l2"` or `"l1"`. With
  `orientation_weight` w > 0 it adds w·(1 − |v1_a·v1_b|), how far apart the two longest axes point
  (SPAM's frame is the sky's, so orientation is real information). With the defaults the value lies
  in [0, √2].

## Where to see it
- `examples/notebooks/metrics_tour.ipynb`: a hands-on tour, each method on real arrays.
- `tests/test_metrics.py`: the registry's checks and the methods on small synthetic arrays.
