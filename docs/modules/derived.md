# Derived stage (`simr/derived.py`)

The Derived stage makes arrays from other arrays, for metrics to use as data or as weights: a window
of weights along an orbit, a weight image, an edge map. Like the other stages it takes data in and
returns data out; it never touches disk, pools or MPI.

```python
from galactic_tidal_constraints.simr import derived

weights = derived.recent_approach_cosine_window(target_orbit, k=1.0)
weights = derived.compute("recent_approach_cosine_window", [target_orbit], {"k": 1.0})   # the same, by name
```

Only what exists today is described here. `simr` runs these methods from a `derived` config section:
once for the run, per simulation or per image, depending on what each entry is made from
([configuration.md](../configuration.md#derived-entries)). A derived array can also be made in Python
and given to a run as an external, e.g. as a metric's weights ([metrics.md](metrics.md#weights)). Decisions behind
it are in [PLAN.md](../../PLAN.md).

## Derived or metric?
The size of what a function returns decides where it belongs:

| It returns | It's a |
|---|---|
| An array about the size of its input (a mask, an edge map, a window, weights) | Derived method |
| One float, or a small dict | Metric ([metrics.md](metrics.md)) |

So a metric never needs another metric's value, and data only moves one way: items and externals →
derived arrays → metrics.

## The registry
Each method is registered with the `derived` decorator where it's defined, and recorded in
`derived.METHODS` as a `Derived`:

| Field | What it declares |
|---|---|
| `inputs` | Each input's kind, in the order the function takes them |
| `output` | The kind of array it returns |
| `settings` | Its settings and their defaults, passed as keyword arguments |
| `function` | The method itself: `function(*inputs, **settings)` |

Kinds are the Metrics stage's input kinds ([metrics.md](metrics.md#input-kinds)), so a derived array
can go wherever a metric takes that kind. `derived.table()` gives the declarations as plain data.

`derived.compute(method, inputs, params=None)` runs one method by name. It checks each input against
its kind's shape, and that the result is a finite array of the declared kind, and returns it as
`float32`. A failure is a `ValueError`.

## The methods

| Method | Inputs | Returns | Settings |
|---|---|---|---|
| `recent_approach_cosine_window` | orbit | weights, one per row | `k` = 1.0, `flat` = 0.0, `pass_number` = 1 |
| `difference_weights` | image, image | image | `floor` = 0.05, `threshold` = None |
| `detect_edges` | image | image | `threshold` = None |

### `recent_approach_cosine_window(orbit, k=1.0, flat=0.0, pass_number=1)`
Weights along an orbit's rows, centered on a close approach, for metrics that compare orbits.

- **The peak** is a pass: a row where the distance between the centers stops falling and starts
  rising. By default the most recent one, nearest to now (t = 0); `pass_number` 2 is the one before
  it, and so on. This is the most recent pass, not the closest one anywhere in the orbit. The peak is
  taken at a row as given, with no interpolation: whoever made the orbit chose its spacing.
- **The shape:** 1 at the peak, falling as half a cosine on each side, to 0 at now and at
  `start = peak − k × (0 − peak)`; 0 outside. `k` = 1 is symmetric; larger reaches further back.
- **`flat`** (0 to 1) is the fraction of each side that stays at 1 before the cosine starts: 0 is a
  pure cosine, 1 is a box.
- Times may be unevenly spaced, and may run forward or backward (as `sim.search_orbit` does); the
  weights come back in the rows' own order.
- **Errors:** no such pass in the orbit (e.g. a pair still approaching); a pass at or after t = 0;
  a window that would start before the orbit's first time; times that don't run strictly one way.

### `difference_weights(a, b, floor=0.05, threshold=None)`
A weight image that is large where two images differ: `|a − b|`, scaled so its largest value is 1.

- With a target image and a model's unperturbed image, it marks where the target departs from
  undisturbed disks: the tidal features, and where the disks themselves changed. The model can't
  shrink weights made this way, since neither image is the model's.
- **`floor`** is the smallest weight anywhere. Without it, material in a region where the two images
  agree (e.g. both empty) would cost nothing; the default 0.05 makes it count a little.
- **`threshold`** (None, or 0 to 1) makes the weights on/off: 1 where the scaled difference is above
  it, 0 elsewhere, before the floor is applied.
- Two identical images give `floor` everywhere.

### `detect_edges(image, threshold=None)`
An edge map: how fast the brightness changes at each pixel (the Sobel operator), scaled so the
strongest edge is 1.

- The image's border is extended by its own edge pixels, so the border itself isn't an edge.
- **`threshold`** (None, or 0 to 1) makes the map on/off.
- A flat image gives all zeros. There is no blur setting: blur the image where it's made (the image
  config's `blur_sigma`, or your own code for a target).
- On `log` brightness images, faint features give relatively strong edges; that's the image's choice.

## As weights
A derived array may be all zeros, and that's not an error here. It becomes one only when a metric is
given it as weights: weights must be non-negative and not all zero.

## Where to see it
- `tests/test_derived.py`: each method on small hand-made arrays.
