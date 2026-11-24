# Configuration

How to tell `simr` what to simulate and image: the config's sections, how its lists combine, what's
checked and what's recorded. The fields themselves are listed in each stage's doc, linked below.
Decisions behind it are in [PLAN.md](../PLAN.md) (Configuration).

## What a config is
A run has two inputs:
- **The models** say *what* to simulate: one model per line in a text file (`--models`), or an
  `(N, 14)` array in Python ([modules/simr.md](modules/simr.md#models)).
- **The config** says *how*: a JSON file (`--config`), or the same thing as a dict in Python (the
  "request"). It lists **only what differs from the defaults**; everything missing takes its default.

Either can change while the other stays fixed. There are no command-line overrides: the config file is
the whole run, and a small change is a new config file. Flags that don't change results stay outside
it: `--models`, `--output`, `--externals`, `--mode`, `--workers` and `--log-level`
([modules/simr.md](modules/simr.md#the-command-line)).

## The sections
| Section | Holds | Missing means | Fields |
|---|---|---|---|
| `run` | `return`: data to hand back in memory (Python only). `save`: data to write to the output folder. Both are lists of names (below) | Nothing returned or saved | [modules/simr.md](modules/simr.md#data-names) |
| `simulator` | One `SimulatorConfig` entry, or a list of them | One default entry | [modules/simulation.md](modules/simulation.md#simulatorconfig) |
| `image` | One `ImageConfig` entry, or a list of them | One default entry | [modules/image_creator.md](modules/image_creator.md#imageconfig) |
| `derived` | Named entries (below), in order: arrays made from other arrays, for metrics to use as data or weights | None | [modules/derived.md](modules/derived.md) |
| `sim_metrics` | One metric entry (below), or a list of them: metrics whose inputs are `sim.*` items | No metrics (an empty list `[]` also means that) | [modules/metrics.md](modules/metrics.md) |
| `img_metrics` | The same, for metrics with at least one `img.*` input | No metrics | [modules/metrics.md](modules/metrics.md) |
| `composite` | The weights for one composite score per model: a weighted mean of the scalar metrics, returned with the metrics (below) | No composite (`null` also means that) | [modules/metrics.md](modules/metrics.md#the-composite) |

What each simulator setting does physically is in
[spam_physics.md](spam_physics.md#the-settings-we-expose-simulatorconfig).

### Data names
`run.return` and `run.save` take any of these. `simr` computes only what they name (and what that
needs), so asking only for `sim.orbit` never runs particles.

| Name | Holds | Notes |
|---|---|---|
| `sim.particles` | Every particle at the end of the run | |
| `sim.unperturbed` | The starting disks placed around today's centers, as if the galaxies never met | |
| `sim.orbit` | The secondary's center at every step | Fast: no particles |
| `sim.search_orbit` | The secondary's center from now back to t = −30, the path every run's start lies on | Fast: no particles. About 30,001 rows (~840 KB saved) |
| `sim.timing` | When the run starts, and why (a dict) | Always recorded in `model.json` when a simulation is saved; it has no `.npy` file |
| `img.model` | The model image | Needs `sim.particles`, which is made (or loaded) even if not asked for |
| `img.unperturbed` | The unperturbed image, in the same frame | Needs `sim.unperturbed` |

Shapes and details: [modules/simr.md](modules/simr.md#data-names).

On the command line `run.return` is ignored, as nothing comes back in memory there, so one config works
in a notebook and on the command line. A non-empty `run.return` gets a warning; an empty one doesn't.

### Metric entries
A metric entry names a method from the metrics table ([modules/metrics.md](modules/metrics.md)) and the
items it takes, in the order and kinds the method declares:

```json
"sim_metrics": [
  {"method": "closest_approach", "inputs": ["sim.search_orbit"]},
  {"method": "closest_approach", "inputs": ["sim.search_orbit"], "select": "distance"}
],
"img_metrics": [
  {"method": "absolute_difference", "inputs": ["img.model", "img.unperturbed"]}
]
```

| Key | Meaning |
|---|---|
| `method` | A name from the metrics table |
| `inputs` | The items the method takes, one per declared kind, in order. A method that can use weights takes them as one more name at the end ([modules/metrics.md](modules/metrics.md#weights)) |
| `select` (optional) | One single-float output of a descriptive method: the entry's value becomes that float instead of the dict |
| `params` (optional) | The method's settings; everything left out takes the method's default |

Metrics are computed whenever their sections list entries and always come back in the result; they
aren't part of `run.return`. With an output, their values are written to the run's record in the
catalog ([modules/simr.md](modules/simr.md#the-output-catalog)). `simr` computes (or loads) the items an entry needs even when nothing else asks
for them. An entry with an `img.*` input goes in `img_metrics` (one per simulation and image entry);
an entry using only `sim.*` items goes in `sim_metrics` (one per simulation). `sim.timing` isn't a
metric input.

### Composite
The `composite` section turns a model's scalar metrics into one number, the weighted mean of its
metric cells over both levels (`metrics.composite`), returned as `result.composite` (shape
`(model,)`). It holds the weights, one per cell of a model:

```json
"sim_metrics": [
  {"method": "orbit_time_error",     "inputs": ["ext.tng_orbit", "sim.search_orbit", "drv.target_win"]},
  {"method": "orbit_position_error", "inputs": ["ext.tng_orbit", "sim.search_orbit", "drv.target_win"]},
  {"method": "closest_approach",     "inputs": ["sim.search_orbit"]}
],
"composite": {"sim_weights": [[0.5, 0.5, 0.0]]}
```

| Key | Shape | Meaning |
|---|---|---|
| `sim_weights` | `(simulator, sim_metrics entry)` | One weight per `sim_metrics` cell of a model |
| `img_weights` | `(simulator, image, img_metrics entry)` | One weight per `img_metrics` cell of a model |

- **One key per level that has entries**, and none for a level without them. To leave a level out,
  give it zeros.
- **`"composite": {}`** gives weight 1 to every entry that gives one float, and 0 to descriptive
  entries (dicts). No section, or `null`, means no composite.
- **Weights** needn't sum to 1. They're numbers (never `true` or `false`), finite and not negative,
  not all zero, and **0 for a descriptive entry** (one without `select`).
- **A failed or skipped cell** with a nonzero weight makes that model's composite NaN; the others are
  unaffected.
- **Checked before anything runs**, like the metric entries. The run's record holds the filled-in
  weights (`"composite"`, also `result.composite_weights`), and the composite values with the metric
  values (`"values"`).
- Mixing directions or scales is the config's choice: `simr` only applies the weights.

### Derived entries
A derived entry makes an array from other arrays: weights along an orbit, a weight image, an edge map
([modules/derived.md](modules/derived.md)). Each has a name, and a metric uses it as `drv.<name>`,
as data or as weights:

```json
"derived": [
  {"name": "target_win",   "method": "recent_approach_cosine_window", "inputs": ["ext.tng_orbit"],
   "params": {"k": 1.0}},
  {"name": "target_edges", "method": "detect_edges", "inputs": ["ext.tng_image"]}
],
"sim_metrics": [
  {"method": "orbit_time_error", "inputs": ["ext.tng_orbit", "sim.search_orbit", "drv.target_win"]}
],
"img_metrics": [
  {"method": "absolute_difference", "inputs": ["drv.target_edges", "img.model"]}
]
```

| Key | Meaning |
|---|---|
| `name` | Letters, digits and `_`, not starting with a digit; unique in the section. Metrics and later entries call it `drv.<name>` |
| `method` | A derived method |
| `inputs` | What it's made from, one per declared kind: externals, `sim.*` or `img.*` items, or entries **above it** in the list |
| `params` (optional) | The method's settings; everything left out takes the method's default |

- **Order matters only for entries built from other entries:** an entry may use those above it, so a
  loop is impossible.
- **How often an entry is computed follows from what it's made from** (its `level`):

  | Made from | Computed | Returned by `run.return` as |
  |---|---|---|
  | Externals only | Once, before anything is simulated (`"run"`) | The one array |
  | Any `sim.*` item | Once per simulation (`"sim"`) | `[model, simulator]`, like `sim.*` data |
  | Any `img.*` item | Once per image (`"img"`) | `[model, simulator, image]`, like `img.*` data |

  An entry built from another takes the more specific of their levels.
- **A metric's section follows its inputs:** an entry made from `img.*` items counts as an image
  input, so a metric using it goes in `img_metrics`. A metric still needs something that depends on
  the model: an item, or an entry made from one.
- **No new combinations:** an entry adds one array per run, simulation or image, and metrics use it
  only where they name it. Every listed entry is computed, used or not; one made from `img.*` items
  makes `simr` create the images.
- **Checked before anything runs:** names, methods, kinds, order and levels; a method that fails on
  its externals; and a once-per-run image that a metric uses but that isn't the size of the image
  entries.
- **A failure during the run** (e.g. a model whose orbit has no pass to center a window on) is that
  entry's alone, for that model: metrics that use it fail with a message naming it, and everything
  else carries on.
- **Recorded:** the run's record lists every entry with its full settings and level, and for one
  computed once, its array's shape and hash, so the config says how the weights were made. A config
  may carry those recorded keys (`level`, `shape`, `hash`), e.g. one saved from `result.config`; they're
  ignored and worked out again.
- **Not saved yet:** `run.save` doesn't take `drv.*` names.

### Externals
An input named `ext.<key>` is an **external**: an array from outside the run, such as a target's orbit,
image or particles. The config holds only the name; the array itself is given to the run under that
key, like the models:

```json
"sim_metrics": [{"method": "pca_difference", "inputs": ["ext.tng_particles", "sim.particles"]}]
```
```python
result = simr.run(models, config, externals={"tng_particles": array})
```

Every input name has a prefix that says where it comes from, so a name without one is an error:

| Prefix | Comes from |
|---|---|
| `sim.` | The simulator |
| `img.` | The Image Creator |
| `drv.` | The `derived` section |
| `ext.` | Outside the run: the `externals` given with it |

- **A value** is a numeric numpy array, or the absolute path of a `.npy` file, which `simr` loads once.
- **Each used external is hashed and recorded.** The hash is SHA-256 of the data as `float32`, with its
  shape (`simr.array_hash(array)`), so the same data gives the same hash from memory or a file, on any
  machine. Each one's source (the path, or `"memory"`), shape and hash are logged, returned in
  `result.externals`, and written to the run's record in the catalog (`"externals"`). The catalog
  records which data a run used, not the data itself.
- **An expected hash** can be given with a path: `{"path": "/data/tng_123/image.npy", "hash": "a3f9..."}`.
  If the file's data has another hash, the run stops before anything is simulated.
- **Its kind** (orbit, particles or image) is the kind of the input place it's used in. One external
  has one kind.
- **Checked before anything is simulated:** every `ext.<key>` a metric or derived entry names is in `externals`; each array is
  turned into `float32` and must hold no NaN or inf, have its kind's shape, and, for an image, the
  height and width of every image entry. Any problem stops the run with a message naming the key.
- **A key** is letters, digits and `_`, not starting with a digit (a Python identifier, in plain
  ASCII): `tng_particles` is named `ext.tng_particles`. So a key has no dots and never looks like a
  number; a bare id needs a letter in front (`sdss_587722984435351614`). The
  `ext.` belongs to the config only; the dict's keys, the `--externals` file and the run's record use
  the plain key. Keys no metric uses are ignored, with one log message.
- **Every metric entry needs an input that depends on the model** (a `sim.*` or `img.*` item, or a
  derived entry made from one), or its value would be the same for every model.
- **Metrics get externals read-only**, so a method can't change a target by accident.

**Parallel runs share externals on one node.** With a pool (`map=pool.map`), the workers don't each get
a copy. `simr` writes each external once to `/dev/shm`, a folder that lives in the node's memory, not
on disk; every worker maps that one copy read-only, and `simr` removes it when the run ends. So:

- **One node only, for externals shared through `/dev/shm`.** `/dev/shm` belongs to one node, so a
  pool that shares externals there must have all its workers on the node that started the run. An MPI
  job that spans more nodes stops **before anything is written**, with a message saying so (`simr`
  reads the launcher's count of ranks per node). The intent: large external data stays on one node,
  and different data sets, such as different targets' particles, get their own jobs.
- **`float32` `.npy` files on a disk every node reads** (e.g. `/home` on Babbage) aren't copied to
  `/dev/shm`, so a run that uses only those can span nodes.
- **Memory:** the node holds each external once, however many workers use it.
- **A `float32` `.npy` given by path is mapped from your own file**, with no copy at all. Saving targets
  as `float32` avoids the copy; a file of another type gets a converted copy in `/dev/shm`.
- **A run without a pool** uses the arrays straight from memory and writes nothing.
- **In Docker**, `/dev/shm` is 64 MB unless the container is started with a larger `--shm-size` (e.g.
  `--shm-size=4g`). Too little room stops the run before anything is simulated. Apptainer uses the
  node's own `/dev/shm` (on Babbage, half the node's memory).
- If a run is killed outright, its `simr_externals_*` folder can be left behind in `/dev/shm`; it's
  safe to delete.

**On the command line**, `--externals externals.json` gives the same dict as a JSON file. Each key maps
to a `.npy` path, or to a path with its expected hash:

```json
{
  "tng_particles": "targets/tng_123_particles.npy",
  "tng_image": {"path": "targets/tng_123_image.npy", "hash": "7a3be7f6..."}
}
```

A relative path is relative to the JSON file's folder, not to where the command runs, so an experiment
folder holding its configs and targets can be moved or copied to Babbage whole. The run records the
full path it used. Any problem exits with code 1 before anything is simulated or written.

## Lists and combinations
- **Every model runs on every simulator entry, and every simulation is imaged with every image entry.**
  So 2 models, 2 simulator entries and 2 image entries make 4 simulations and 8 images.
- A list can appear only directly under a section; anything deeper is an ordinary value, like an image
  center.
- **Every metric entry runs on every cell:** each `sim_metrics` entry on every simulation, each
  `img_metrics` entry on every (simulation, image) pair, with no per-entry filters.
- **An entry's position is its name:** simulator entry 0 is `sim_00`, image entry 1 is `img_0001`. Data
  is indexed the same way: `[model, simulator]` for `sim.*`, `[model, simulator, image]` for `img.*`,
  and the metric values add an entry axis: `sim_metrics[model, simulator, entry]`,
  `img_metrics[model, simulator, image, entry]`
  (see [architecture.md](architecture.md#metric-arrays-two-levels)).
- Combinations multiply quickly; keeping them to a sensible size is up to whoever writes the config.

## Checks
Everything is checked before anything runs, and an error says where the problem is:

| Mistake | Error |
|---|---|
| An unknown section or field | `unknown keys in simulator[1]: ['sed']. Known: ['seed', 'n1', …]` |
| An unknown data name | `unknown names in run.save: ['sim.orbits']. Known: ['sim.particles', …]` |
| A name instead of a list | `run.save must be a list of names, e.g. ["sim.orbit"]` |
| A wrong type | `simulator[0].lnl must be a number: '0.1'` |
| A bad value | `simulator[0]: lnl can't be negative: -1` |
| An empty list | `simulator is an empty list` (an empty **metric** section is fine: no metrics) |
| An unknown method | `sim_metrics[0].method: unknown method 'closest'. Known: […]` |
| A wrong metric input | `sim_metrics[0].inputs[0]: closest_approach takes a orbit here, but sim.particles is a particles` |
| An `img.*` input at the wrong level | `sim_metrics[0].inputs[0]: an img.* input belongs in img_metrics, not sim_metrics` |
| A `select` that isn't one float | `sim_metrics[1].select: 'position' isn't a single-float output of closest_approach. Choices: ['t', 'distance']` |
| Composite weights of the wrong shape | `composite.sim_weights must have one weight per cell of a model, shape (simulator, sim_metrics entry) = (1, 3): got shape (3,)` |
| A weight on a descriptive entry | `composite.sim_weights: sim_metrics[2] (closest_approach) gives a dict, not one float, so its weight must be 0` |

Types follow JSON:
- A whole number is fine for a number field (JSON may write `10.0` as `10`), but a field that takes a
  whole number (`n1`, `seed`) rejects `1000.0`.
- `true` and `false` are never numbers.
- A pair, such as an image center, is a list of 2: `[32, 64]`.

## What's recorded
- With an output, the run's record in `catalog.json` holds every entry's **full** config, defaults
  filled in, so the output says exactly what ran even when the config file listed almost nothing.
- **The result holds the same**, with or without an output: `result.config` (the full config, as plain
  JSON-ready data) and `result.code` (each repo's commit and whether it had uncommitted changes).
  Saved as JSON, `result.config` runs again as a config; a derived entry's recorded `level`, `shape`
  and `hash` are accepted there and worked out again, never taken from the file.
- It also records each external the metrics used: its source, shape and hash (see Externals).
- Each model's `model.json` records the full config of every simulation and image entry it saved.
- A saved simulation or image is reused only when its config matches **exactly**; a different config
  is neither used nor changed ([modules/simr.md](modules/simr.md#reuse)).

## Examples
The files are in [examples/configs/](../examples/configs/). A test checks that each one passes `simr`'s
checks, and that `defaults.json` still equals the code's defaults (`tests/test_example_configs.py`).

**Compare two dynamical-friction strengths**, saving orbits, particles and two kinds of images
([compare_lnl.json](../examples/configs/compare_lnl.json)):
```json
{
  "run": {"save": ["sim.orbit", "sim.particles", "img.model"]},
  "simulator": [
    {"lnl": 0.1},
    {"lnl": 0.001}
  ],
  "image": [
    {},
    {"blur_sigma": 1.0, "brightness": "log"}
  ]
}
```
```bash
python -m galactic_tidal_constraints.simr --models examples/configs/models.txt \
  --config examples/configs/compare_lnl.json --output runs
```
With the two models in [models.txt](../examples/configs/models.txt): 4 simulations (`sim_00` with
`lnl` 0.1, `sim_01` with Wallin's 0.001) and 8 images (`img_0000` plain, `img_0001` blurred, log
brightness).

**Every field at its default** ([defaults.json](../examples/configs/defaults.json)). Running with it
is the same as running with no config at all; it's here to show what can be set. The empty `run` lists
take the names [above](#data-names):
```json
{
  "run": {"return": [], "save": []},
  "simulator": {
    "seed": 0, "n1": 1000, "n2": 1000, "heat1": 0.0, "heat2": 0.0, "heat_mode": "centered",
    "lnl": 0.1, "r_scale": 10.0, "m_halo": 5.8, "m_bulge": 0.3333, "h_bulge": 0.2, "h_disk": 1.0,
    "normalization": "total", "start": "pass", "start_pass": 1, "no_pass": -5.0
  },
  "image": {
    "width": 128, "height": 128, "primary_center": [32.0, 64.0], "secondary_center": [96.0, 64.0],
    "blur_sigma": 0.0, "blur_truncate": 4.0, "brightness": "log"
  },
  "derived": [], "sim_metrics": [], "img_metrics": [], "composite": null
}
```

## From Python
The same config, as a dict, is `simr.run`'s request. Entries can also be `SimulatorConfig` or
`ImageConfig` objects, and `run.return` works here:
```python
from galactic_tidal_constraints.simr import simr
from galactic_tidal_constraints.simr.simulation import SimulatorConfig

models = simr.read_models("examples/configs/models.txt")
result = simr.run(models, {
    "run": {"return": ["sim.orbit"]},
    "simulator": [SimulatorConfig(lnl=0.1), {"lnl": 0.001}],
})
result.data["sim.orbit"][1, 1]      # model 1's orbit with lnl 0.001
```
