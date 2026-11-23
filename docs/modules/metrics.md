# Metrics stage (`simr/metrics.py`)

The Metrics stage computes one value per method from simulated or external arrays. Like the other
stages it takes data in and returns data out; it never touches disk, pools or MPI. SIMR is a pipeline
for generating many relevant metrics; "fitness" keeps its meaning as what's optimized, usually a
composite of metrics: a weighted mean of the scalar metrics, one per model (`metrics.composite`,
below; `simr` returns it when the config has a `composite` section).

Only what exists today is described here. Decisions behind it are in
`Experiments/MDRN-4` (the vault) and [PLAN.md](../../PLAN.md).

## Metric or derived?
A metric deliberately reduces its inputs to something small. The size of what a function returns
decides where it belongs:

| It returns | It's a |
|---|---|
| One float, or a small dict | Metric (this stage) |
| An array about the size of its input (a mask, an edge map, a window, weights) | Derived method ([derived.md](derived.md)) |

So a metric never takes another metric's value as an input, and data only moves one way: items and
externals → derived arrays → metrics → the composite, the one place metric values are combined.

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
| `weights` | `True` if the method can take weights as one more input, which must match its first input; `False` (the default) if it takes none |

`metrics.table()` returns the same information as plain, JSON-ready data, for `simr`'s early checks
and for higher-level programs.

`metrics.compute(method, inputs, select=None, params=None)` runs one method on its input arrays and
checks both sides against the declaration. Each input must fit its kind's shape
(`metrics.INPUT_SHAPES`; the table under Input kinds; the check itself is `metrics.check_input`), so a wrong array fails cleanly before the
method runs, wherever it came from. The value: a scalar must be one finite float; a descriptive dict must
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
source (a `simr` item, or an external array given to `simr.run`; see Externals in
[configuration.md](../configuration.md)), as long as the shape fits:

| Kind | Shape | `simr` items |
|---|---|---|
| `orbit` | `(n, 7)` float32: x, y, z, vx, vy, vz, t | `sim.orbit`, `sim.search_orbit` |
| `particles` | `(n, 6)` float32: x, y, z, vx, vy, vz | `sim.particles`, `sim.unperturbed` |
| `image` | `(height, width)` float32 in [0, 1] | `img.*` |
| `weights` | `(n,)` float32: one value per row of the array it weighs | none yet; made by the Derived stage ([derived.md](derived.md)) or given as an external |

## Weights
A method that can use weights says so (`weights=True` in its declaration) and has a `weights=None`
argument; a method that doesn't say so takes none. In a config entry they're one more name after the
method's inputs:

```json
{"method": "orbit_time_error", "inputs": ["ext.tng_orbit", "sim.search_orbit", "drv.target_win"]}
```

- **Optional:** without them, every row or pixel counts equally.
- **They match the method's first input** (the target, when there is one): one value per row of an
  orbit or of particles, or an image of the same shape. One function, `metrics.check_weights`, holds
  this rule for every method.
- **They must be usable:** finite, none negative, not all zero; otherwise that metric cell fails.
- **Where they come from:** a derived entry (`drv.<name>`; [derived.md](derived.md)) or an external
  array.
- When the weights and the input they weigh are both known before the run (externals, or derived from
  them), they're matched before anything is simulated.

## The methods
This table is guarded by a test (`tests/test_metrics.py`) against `metrics.table()`.

| Method | Kind | Inputs | Returns | Best | Range | Settings |
|---|---|---|---|---|---|---|
| `closest_approach` | descriptive | orbit | `t` (), `distance` (), `position` (3,), `velocity` (3,) | — | — | — |
| `absolute_difference` | scalar | image, image; optional image weights | one float | low | [0, 1] | — |
| `orbit_time_error` | scalar | orbit (target), orbit (model); optional weights | one float | low | [0, ∞) | — |
| `orbit_position_error` | scalar | orbit (target), orbit (model); optional weights | one float | low | [0, ∞) | — |
| `pca_shape` | descriptive | particles | `axes` (3,), `vectors` (3, 3), `center` (3,), `b_a` (), `c_a` () | — | — | `columns="position"`, `center="mean"`, `trim_quantile=None` |
| `pca_difference` | scalar | particles, particles | one float | low | [0, ∞) | as `pca_shape`, plus `descriptor="ratios"`, `distance="l2"`, `orientation_weight=0.0` |

- **`closest_approach`**: the point along an orbit where the two centers are closest — the smallest
  distance in whatever orbit it's given, so the entry's input decides the window (`sim.search_orbit`:
  the whole window from now back to t = −30; `sim.orbit`: the run itself, so the simulator config's
  `start` decides). Row order doesn't matter.
- **`absolute_difference`**: the mean |a − b| over the pixels of two greyscale images of the same
  shape; 0 is identical. Any two matching images work: model vs unperturbed reads as how perturbed
  the model is, and an image vs an external target as how far from the target.
  Images of another shape are an error; matching frames is the caller's job. With a weight image
  it's the weighted mean, Σ w·|a − b| / Σ w: with a target, a model and weights from
  `derived.difference_weights(target, unperturbed)`, differences count most where the target shows
  tidal features.
- **`orbit_time_error`**: how far a model's orbit is from a target's **at the same times**: the
  weighted time average of the squared distance between them, in SPAM length units squared. For each
  target row the model's position at the nearest model time is used. 0 means the model is where the
  target is, when the target is there.
- **`orbit_position_error`**: how far the target's **path** is from the model's, whatever the timing:
  the same average of the squared distance from each target point to the nearest model point. Only
  the model's rows inside the span of the target times that count are searched, so on a repeating
  orbit a target point can't match an earlier lap. A model on the right path at the wrong times scores
  well here and badly on `orbit_time_error`; because of that span, a late model also loses a little
  here near the window's ends, where the weights are small.
- **Both orbit errors:**
  - The target comes first, the model second (e.g. `sim.search_orbit`, in either time order).
  - The target's times must increase, with any spacing. Each row stands for the time around it (the
    trapezoid rule), so the value is an average over time, not over rows, and doesn't depend on how
    the target was sampled.
  - The model must cover the target times that count, or the metric fails; nothing is padded.
  - Weights are one per target row, e.g. `derived.recent_approach_cosine_window(target)`. They replace
    the old code's Gaussian weight function; the old 50/50 mix of the two errors is a composite.
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

## The composite
`metrics.composite(sim_metrics=None, img_metrics=None, sim_weights=None, img_weights=None)` combines a
model's metric cells into one number, the weighted mean across both levels:

```
composite[m] = (Σ sim_weights · sim_metrics[m] + Σ img_weights · img_metrics[m]) / (Σ sim_weights + Σ img_weights)
```

```python
# The old 50/50 mix of the two orbit errors, sim_metrics entries 0 and 1, one simulator entry:
fitness = metrics.composite(sim_metrics=result.sim_metrics, sim_weights=[[0.5, 0.5]])   # shape (model,)
```

- **Inputs** are shaped like `SimrResult`'s: `sim_metrics` `[model, simulator, entry]`, `img_metrics`
  `[model, simulator, image, entry]`. Give either level or both; object arrays are fine.
- **Weights** have one value per cell of a model: `sim_weights` `(simulator, entry)`, `img_weights`
  `(simulator, image, entry)`. They needn't sum to 1, must be finite and not negative, and not all
  zero across both levels. A level given without weights counts every cell equally; weights for a
  level that isn't given are an error.
- **Weight 0 skips a cell** entirely, so it may hold anything: a descriptive entry's dict (or its
  `None`), or a failed scalar's NaN. A cell with a nonzero weight must hold one number.
- **NaN** in a cell with a nonzero weight (a failed or skipped scalar) makes that model's composite
  NaN, and only that model's.
- **Mixing is the caller's choice.** The composite only applies the weights; it doesn't check
  directions (`best`) or scales. Raw values differ a lot (orbit errors are often 0.1 to 1, image
  differences 0.02 to 0.06), so a mix is dominated by its largest terms unless the weights compensate.
- **In `simr`**, the config's `composite` section holds the weights and the result's `composite` the
  values ([configuration.md](../configuration.md#composite)).

## Where to see it
- `examples/notebooks/metrics_tour.ipynb`: a hands-on tour: each method on real arrays, externals,
  the derived arrays, weights and the orbit metrics, and one config that uses them all.
- `tests/test_metrics.py`: the registry's checks and the methods on small synthetic arrays;
  `tests/test_orbit_metrics.py`: the orbit metrics and weights; `tests/test_simr_weights.py`: through
  `simr`; `tests/test_composite.py`: the composite.
