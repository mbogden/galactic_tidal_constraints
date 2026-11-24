# SIMR workflow (`simr/simr.py`)

`simr` runs the simulations a request asks for, for many models, and hands back only the data the
request names. With an output it records every run in a catalog, and it can save data to an output
folder and load it back. It sits one level
above the Simulator stage ([`simr/simulation.py`](simulation.md), imported as `sim`), which runs one simulation at a
time and knows nothing about disk.

```python
from galactic_tidal_constraints.simr import simr

result = simr.run(models, request, output=None, map=map, externals=None)
```

Only what exists today is described here. Decisions behind it are in [PLAN.md](../../PLAN.md).

## Inputs

### Models
A NumPy array of shape `(N, 14)`, one model per row, in `sim.PARAM_NAMES` order (`x` to `theta2`).
One model of shape `(14,)` also works. Rows must be finite.

### The request
A plain dict, the same as a config file ([configuration.md](../configuration.md) has the sections, checks and examples):

```python
request = {
    "run": {
        "return": ["sim.orbit", "sim.timing"],   # data to hand back in memory
        "save": ["sim.particles"],               # data to write to the output folder
    },
    "simulator": [{"lnl": 0.1}, {"lnl": 0.01}],  # one dict or a list; each becomes a SimulatorConfig
    "image": [{}, {"blur_sigma": 1.0}],          # one dict or a list; each becomes an ImageConfig
    "sim_metrics": [{"method": "closest_approach", "inputs": ["sim.search_orbit"], "select": "distance"}],
    "img_metrics": [{"method": "absolute_difference", "inputs": ["img.model", "img.unperturbed"]}],
}
```

| Key | Default | Meaning |
|---|---|---|
| `run.return` | `[]` | Names of the data to hand back |
| `run.save` | `[]` | Names of the data to save. Independent of `return`: a run can do both |
| `simulator` | one `SimulatorConfig()` | A dict (keys are `SimulatorConfig` fields) or a `SimulatorConfig`, or a list of them |
| `image` | one `ImageConfig()` | A dict (keys are `ImageConfig` fields) or an `ImageConfig`, or a list of them. Centers are lists in JSON (`[column, row]`). See [image_creator.md](image_creator.md) |
| `derived` | none | Named entries, in order: arrays made from other arrays, which metrics name as `drv.<name>`. See [derived.md](derived.md) and [configuration.md](../configuration.md#derived-entries) |
| `sim_metrics` | none | Metric entries whose inputs are `sim.*` items: one dict or a list ([] is also none). See [metrics.md](metrics.md) and [configuration.md](../configuration.md#metric-entries) |
| `img_metrics` | none | Metric entries with at least one `img.*` input, likewise |

**Every model runs on every simulator entry, and every simulation is imaged with every image
entry.** Every `sim_metrics` entry runs on every simulation, and every `img_metrics` entry on every
(simulation, image) pair; metrics always come back in the result (they aren't part of `return`), and
`simr` computes (or loads) the items they name even when nothing else asks for them, so an orbit-only
request with `sim_metrics` never builds an image. Entries are checked against the metrics table
(`metrics.METHODS`) before anything runs; an entry naming a method with a `setup` step is a
`NotImplementedError` until the DL step wires setup in. Unknown keys, unknown names, wrong types and bad values are errors that say where the
problem is (e.g. `simulator[0].lnl must be a number: '0.1'`, `simulator[1]: lnl can't be negative`,
`image[0].primary_center must be a list of 2 (a number, a number)`), raised before anything runs.
Types are checked against the config's type hints; an integer is fine for a number field (JSON writes
`1.0` as `1`), `true` / `false` are never numbers, and a tuple field takes a list of that length.

### Data names
The Simulator stage can hand back:

| Name | Each cell holds |
|---|---|
| `sim.particles` | `sim.simulate_particles`: every particle at the end, `(n1 + n2 + 1, 6)` |
| `sim.unperturbed` | `sim.unperturbed_particles`: the same disks at today's centers, `(n1 + n2 + 1, 6)` |
| `sim.orbit` | `sim.simulate_orbit`: the secondary's center at every step, `(n_steps + 1, 7)` |
| `sim.search_orbit` | `sim.search_orbit`: the secondary's center from now back to t = −30, the path every run's start lies on, `(about 30001, 7)` |
| `sim.timing` | the dict from `sim.get_timing` (start time, steps, which pass) |

The Image Creator stage can hand back (`simr.IMAGE_DATA`; each made with `img.create_image`, from the
simulation's own particles and that simulator's `n1` and `n2`):

| Name | Made from | Each cell holds |
|---|---|---|
| `img.model` | `sim.particles` | The model image, `float32` `(height, width)` in [0, 1] |
| `img.unperturbed` | `sim.unperturbed` | The unperturbed image, in the same frame |

`run.return` also takes derived entries, as `drv.<name>`
([configuration.md](../configuration.md#derived-entries)): one made per simulation or per image comes
back indexed like that stage's data, and one computed once for the run comes back as the array
itself.

`run.save` also takes `ext.<key>`, an external some entry uses, and any derived entry's `drv.<name>`
(see Saving). Where each goes follows from what it's made from:

| Name | Saved in |
|---|---|
| `ext.<key>` | `<output>/externals/<hash>.npy`: once for the output |
| `drv.<name>`, made only from externals | `<output>/derived/<hash>.npy`: once for the output |
| `drv.<name>`, made from a model's data (`sim.*`, `img.*`) | That model's folder, `model_<id>/derived/` |

**Only what `return` and `save` name is computed.** Asking only for `sim.orbit` never runs
particles. An image needs the `sim` items it's made from, so `simr` computes them (or loads them, if
saved) even when they aren't returned or saved. Simulated arrays are converted to `float32` as soon
as a simulation is done; everything after expects `float32`.

### Seeds
The seed is part of each simulator entry (`SimulatorConfig.seed`, 0 by default), and `simr` uses it
for every model. SPAM draws the same random numbers for any model, so with one seed every model's
disks share one pattern, stretched by `rout` and tilted by `theta` and `phi`: metric differences
between models come only from their parameters. To compare draws, list entries that differ only in
`seed`. Details: [spam_physics.md](../spam_physics.md), `seed`.

### Externals
`externals` is a dict of arrays from outside the run (e.g. a target's orbit, image or particles), by
key: each value an array, a `.npy` path, or `{"path": ..., "hash": ...}`. A metric or derived entry
names one as `ext.<key>`. Like the models, they aren't part of the request. Each
external a metric uses is loaded once, converted to `float32`, checked and hashed before anything
runs, and handed to metrics read-only. Keys, checks, hashes and the command line's `--externals`:
[configuration.md](../configuration.md#externals). With a pool they're shared through `/dev/shm`
(see Spreading the work).

### The output
`output` is a runtime argument, not part of the request, because it doesn't change any result. With
an output, **every run is recorded in a catalog**, whether or not it saves data (see The output
catalog). It's a folder or a `.json` file, told apart by name, so it works before anything exists:

| `output` | What it is |
|---|---|
| `runs/` (no extension) | A folder: `runs/catalog.json`, plus one folder per model for the data `run.save` names. Saved data is loaded from it instead of computed (see Reuse) |
| `gen_07.json` | A catalog file on its own, with no model data. `run.save` with it is an error |
| anything else, e.g. `runs.txt`, `runs.v2/` | An error |

`run.save` needs a folder; otherwise `output` is optional.

## The result
`simr.run` returns a `SimrResult`:

| Field | What it is |
|---|---|
| `models` | The models that ran, `(N, 14)`: row m is model m (the result's own copy) |
| `simulators` | The simulator entries as `SimulatorConfig`s, every default filled in: entry s is simulator s |
| `images` | The image entries as `ImageConfig`s, likewise: entry i is image i |
| `data` | One entry per name in `return`, and nothing else |
| `sim_metrics` | Each `sim_metrics` entry's value per simulation, an object array `[model, simulator, entry]`: a float (a scalar method, or `select`) or a dict (a descriptive method). A failed or skipped cell holds NaN where the entry gives a float, `None` where it gives a dict. Shape `(N, S, 0)` with no entries |
| `img_metrics` | Likewise per image: `[model, simulator, image, entry]` |
| `composite` | With a `composite` section: one `float64` per model, shape `(N,)`, the weighted mean of its metric cells (`metrics.composite`); NaN for a model whose weighted cell failed or was skipped. `None` without the section ([configuration.md](../configuration.md#composite)) |
| `sim_metric_configs`, `img_metric_configs` | The filled-in entries (`method`, `inputs`, `select` — `None` when absent — and `params` with every default), so `sim_metrics[..., k]` is entry `sim_metric_configs[k]` |
| `composite_weights` | The weights the composite used, filled in: `{"sim_weights", "img_weights"}` as nested lists, a key per level with entries. `None` without the section |
| `derived_configs` | The filled-in `derived` entries, in order: `name`, `method`, `inputs`, `params` with every default, `level` (`"run"`: computed once; `"sim"`: per simulation; `"img"`: per image), and for `"run"` ones the array's `shape` and `hash` |
| `externals` | Each external a metric used, by key: `{"source"` (its path, or `"memory"`), `"shape"`, `"hash"` (`simr.array_hash`)`}`. Empty when none was used |
| `config` | The whole request with every default filled in, as plain JSON-ready data: `run`, `simulator`, `image`, `derived`, `sim_metrics`, `img_metrics`, `composite`. Exactly what the catalog's run record holds, so a run without an output still says what ran, and it runs again as a request (`simr.run(models, result.config, externals=...)`) |
| `code` | The code that ran: each repo's commit and whether it had uncommitted changes (`simr.code_versions()`, the catalog's `"code"`). Read on every run, with or without an output |
| `ids` | Each row's model id in the output's catalog, or `None` without an output (see The output catalog) |
| `status`, `seconds`, `errors` | Per stage (`"sim"`, `"img"`, `"sim_metrics"`, `"img_metrics"`), indexed like its data or values; see Failures and status |

Each `data` entry is a NumPy object array of cells, indexed by the levels its stage depends on:
`[model, simulator]` for `sim.*`, `[model, simulator, image]` for `img.*`. A cell holds that
simulation's (or image's) own array, so orbits of different lengths fit side by side:

```python
result.data["sim.orbit"][12, 1]         # model 12's orbit from simulator entry 1
result.data["img.model"][12, 1, 0]    # its image from image entry 0
result.simulators[1].lnl                # what entry 1 was
```

## Planning a run: `simr.plan`
A config multiplies quickly (models × simulator entries × image entries × metric entries), and
everything `run.return` names comes back into the memory of the one process that called `run`.
`simr.plan` says what a run would do before it's started:

```python
plan = simr.plan(models, request, externals=None, output=None)
print(simr.format_plan(plan))
```
```
1000 models x 2 simulator entries x 1 image entries
  simulations: 2,000 (each computes sim.particles, sim.search_orbit)
  images:      2,000 (each makes img.model)
  derived:     1 once for the run, 0 per simulation, 0 per image
  metrics:     4,000 simulation-level cells, 2,000 image-level cells, and one composite per model
  external tng_orbit: shape (601, 7), 16.43 KB, from /data/targets/tng_123/orbit.npy
  returned in memory: 216.60 MB
    sim.particles: 2,000 arrays, 91.60 MB
    img.model: 2,000 arrays, 125.00 MB
  saved to disk: up to 1.56 GB
    sim.search_orbit: 2,000 arrays, up to 1.56 GB
```

- **It makes every check `run` makes before simulating** (the models, the request, the externals, the
  derived entries made from them), through the same code, so a plan that comes back is a run that
  would start. Externals are loaded and hashed, and once-per-run derived arrays computed, as `run` does
  first. **Nothing is simulated and nothing is written.**
- **`plan`** is a plain, JSON-ready dict: the counts (`models`, `simulations`, `images`, `derived`,
  `metric_cells`), what each simulation and image computes (`sim_items`, `image_items`), the
  `externals` used, and `return` and `save`, each with its total `bytes` and its `items`.
- **Sizes** are the arrays as `float32`. They're exact for particles and images, which follow from
  the config. An orbit's length depends on each model's start, which isn't known until it runs, so
  every orbit is counted at the longest one can be (`simr.ORBIT_ROWS_AT_MOST` rows, about 840 KB) and
  marked `upper_bound` ("up to" in the text). `sim.timing` and the metric values are a few numbers each
  and count as 0.
- **`output`** is only checked by name (`run.save` needs a folder). What's already saved there isn't
  read, so a plan counts every simulation, though a run into that folder would reuse some.

## Spreading the work: `map`
`simr.run(..., map=...)` takes a `map` function, built-in `map` by default. Each call handles **one
model** with all its simulator entries: it loads what's saved, computes the rest, saves what's asked
for and catches each simulation's failure, touching only that model's folder. The same model twice in
one run is one call, so no two calls write the same folder. Only the process that called `run` touches
the catalog: it assigns ids and writes the run record before any model is handed out, and collects
the results in order.

With built-in `map`, models run one after another in this process. A pool's `map` (e.g.
`ProcessPoolExecutor().map`) runs them in parallel; the pool is created once by the caller, never by
`simr`. Results don't depend on the `map`: each simulator entry's seed comes from its config. Returned
data travels back to the calling process and stays in its memory, so at scale, save instead.

```python
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=simr.available_cpus(), mp_context=multiprocessing.get_context("spawn"),
                         initializer=simr.configure_logging) as pool:
    result = simr.run(models, request, output="runs", map=pool.map)
```

- **While results arrive**, the statuses so far are written to the catalog at most once a minute
  (`simr.CATALOG_WRITE_SECONDS`) and again at the end; cells not yet run are `null`.
- **A worker that dies** (a broken pool, e.g. a crash in SPAM's Fortran, which Python can't catch)
  stops the run: the unfinished models are logged, the catalog keeps the statuses so far with
  `"finished": null`, and `simr.SimrRunError` is raised. Everything already saved is reused by a rerun.
- `simr.available_cpus()` is the CPUs this process may use (under Slurm, the job's, not the
  machine's). `simr.configure_logging(level)` sets up logging with the process name in each line, for
  the main process and, as the pool's initializer, each worker. Under `mpirun` it names the MPI rank
  instead (below); `use_rank=False` keeps process names, which `--mode local` uses, since a local pool
  started under `mpirun` (as Babbage's job scripts do) would otherwise read rank 0 in every process.
- **Externals with a pool** aren't copied into each model's task. `simr` writes each one once to
  `/dev/shm` (memory, not disk; a `float32` `.npy` given by path is used as it is), every worker maps
  that copy read-only, and the folder is removed when the run ends. `/dev/shm` belongs to one node, so
  **a pool that shares externals there must have all its workers on the calling process's node**. Under
  an MPI launcher, `simr` reads how many ranks are on this node (`OMPI_COMM_WORLD_LOCAL_SIZE` vs
  `OMPI_COMM_WORLD_SIZE`; `MPI_LOCALNRANKS` vs `PMI_SIZE` for MPICH) and stops a multi-node job with a
  `ValueError` before anything is written. Without those variables, a worker that can't see the copy
  raises `SimrRunError` when it starts. `float32` files on a shared disk need no copy, so they work
  across nodes. Details: [configuration.md](../configuration.md#externals).
- "spawn" starts workers fresh, as MPI does. A script that uses it needs the usual
  `if __name__ == "__main__":` guard.

### With MPI
An MPI pool's `map` works the same way, with MPI ranks as the workers, so a run can span nodes.
`mpi4py` is an optional group (`uv sync --group mpi`), built from source so it uses the installed MPI.
It's launched with `mpi4py.futures`: rank 0 runs the script (or the command line) and every other rank
waits for tasks.

```bash
mpirun -n 64 python -m mpi4py.futures -m galactic_tidal_constraints.simr --mode mpi --models ... --output ...
```
```python
from mpi4py.futures import MPIPoolExecutor   # in a script started with: mpirun -n N python -m mpi4py.futures script.py

with MPIPoolExecutor(initializer=simr.configure_logging) as pool:   # N - 1 workers
    result = simr.run(models, request, output="runs", map=pool.map)
```

- **Log lines** name the rank (`[rank 3]`) instead of the process. `configure_logging` reads it from
  the variable the launcher sets (`OMPI_COMM_WORLD_RANK` for OpenMPI, `PMI_RANK` for MPICH), so `simr`
  never imports `mpi4py`.
- **A worker that crashes** under MPI usually makes `mpirun` end the whole job, rank 0 included, so no
  `SimrRunError` is raised. The catalog keeps its last write (at most a minute old) with
  `"finished": null`, and a rerun reuses everything already saved.

## Failures and status
A simulation (one model on one simulator entry) **fails as a whole** when computing any requested item
raises an error, or when any simulated array holds NaN or inf. `sim.timing` isn't checked: it
legitimately holds NaN (e.g. `closest_distance` for a fixed start). A failure doesn't stop the run:

- its cells in `result.data` hold `None`;
- nothing is saved for it, because every item is computed and checked before anything is written;
- nothing is recorded in `model.json`, so a later run retries it;
- the error is logged with its traceback.

Errors while **writing** files (a full disk, no permission) aren't caught: they stop the run.

An image (one simulation with one image entry) fails on its own, the same way: its cells hold
`None`, the error is logged, and its simulation is unaffected. When a simulation fails, its images are
`"skipped"`.

A metric cell that fails or is skipped holds **NaN** if its entry gives one float (a scalar method, or
`select`), so a float column stays all floats and can be used as a float array, and `None` if it gives a
dict.

Each outcome comes back per stage, indexed like the data:

| Field | Each cell |
|---|---|
| `result.status["sim"][m, s]` | `"ran"`, `"reused"` (saved, and nothing had to be computed), `"failed"`, or `"empty"` (nothing was asked of it: nothing returned or saved, and no image or metric needs it) |
| `result.status["img"][m, s, i]` | `"ran"`, `"reused"` (saved, and nothing had to be made), `"failed"`, `"skipped"` (its simulation failed), or `"empty"` (no image was asked for) |
| `result.status["sim_metrics"][m, s, k]` | `"ran"`, `"failed"` (the method raised, returned NaN or inf, or didn't match its declaration; the simulation is unaffected), or `"skipped"` (its simulation failed). With no entries there are no cells |
| `result.status["img_metrics"][m, s, i, k]` | The same per image; `"skipped"` also when the image it needs failed |
| `result.seconds[stage][...]` | Wall time, loading included. A simulation's time doesn't include its images or metrics; each metric cell is timed on its own |
| `result.errors[stage][...]` | `"Type: message"` for a failure, else `None` |

```python
(result.status["sim"] == "failed").sum()     # how many failed
```

The run ends with a log line counting how many simulations ran, were reused, failed and were empty,
and, when images were asked for, one counting how many images ran, were reused, failed and were
skipped. The exit
code (see The command line) depends on the simulations only.

## Saving

### Folder layout
```
<output>/
    catalog.json                  "models": id -> 14 parameters; "runs": one record per run here
    externals/
        <hash>.npy                an external `run.save` named (`ext.<key>`), stored once by its hash
    derived/
        <hash>.npy                a once-per-run derived array `run.save` named (`drv.<name>`)
    model_000000/                 one folder per model id
        model.json                the model's record
        simulations/
            sim_00_particles.npy  one file per saved array: <simulation>_<item>.npy
            sim_00_orbit.npy
        images/
            sim_00_img_0000_model.npy        one file per saved image: <simulation>_<image>_<item>.npy
            sim_00_img_0000_unperturbed.npy
        derived/                  derived arrays made from this model's data that `run.save` named
            sim_00_drv_model_win.npy         one made per simulation: <simulation>_drv_<name>.npy
            sim_00_img_0000_drv_tidal.npy    one made per image: <simulation>_<image>_drv_<name>.npy
```

A simulation's name is `sim_<s>` for simulator entry s (`sim_00`, `sim_01`, …; `simr.simulation_name`),
and an image entry's is `img_<i>` with four digits (`img_0000`, …; `simr.image_name`). Arrays are
`.npy` files, one per array, `float32`, uncompressed, so one array can be added or deleted without
touching the others.

### Once-per-run arrays: `externals/` and `derived/`
An external is copied into the output **only when `run.save` names it** (`"save": ["ext.tng_orbit"]`),
and so is a derived array made only from externals (`drv.<name>`). Large files that stay where they
are needn't be copied: the record keeps an external's path either way (`externals` → `source`).

- **Named by hash:** the file is `<hash>.npy`, the hash being `simr.array_hash` of its data (the one
  the record already holds). The same data is stored once, whatever key or run it came under, and two
  different arrays can never collide.
- **Written before anything is simulated**, as `float32`, atomically. A file of that hash already
  there is left alone.
- **The run's record lists them** under `files`: name → path relative to the output folder, e.g.
  `{"ext.tng_orbit": "externals/6cfc….npy"}`.
- The external must be one a metric or derived entry uses (only those are loaded, checked and
  hashed); naming another is an error before anything is written.

### A model's derived arrays: `model_<id>/derived/`
A derived entry made from a model's data (any `sim.*` or `img.*` input, directly or through another
entry) is saved in that model's folder when `run.save` names it, and listed in `model.json` under
`derived`, by name:

```json
"derived": {
  "tidal": {
    "config": {"name": "tidal", "method": "difference_weights", "inputs": ["ext.tng_image", "img.unperturbed"],
               "params": {"floor": 0.05, "threshold": null}, "level": "img"},
    "made_from": {"ext.tng_image": {"shape": [128, 128], "hash": "7a3b…"}},
    "files": {"sim_00": {"img_0000": "derived/sim_00_img_0000_drv_tidal.npy"}}
  }
}
```

- **`config`** is the filled-in entry. **`made_from`** is everything behind it besides the model's own
  data: each external (shape and hash) and each derived entry it uses, directly or through others.
- **`files`** are by simulation for an entry made per simulation (`"sim_00": "derived/…"`), and by
  simulation then image for one made per image.
- **A record, not reused.** Unlike simulations and images, a saved derived array is never loaded by a
  later run: each run that names it writes its own, replacing the file. If the entry, or anything in
  `made_from`, differs from what's listed, the listing starts again with this run's (a warning is
  logged; the earlier files stay on disk, unlisted).
- Its simulation (and image entry) is recorded too, so the file's configs are on record. Nothing is
  added under a simulation or image entry that was saved with a different config.
- A derived entry that fails for a model saves nothing for it.
- `simr.load_derived(folder)` reads them back.

### `model.json`
One per model folder: the model's 14 named parameters, each saved simulation by name, and each saved
image entry by name. `simulations` and `images` mirror the request's `simulator` and `image` lists:
each entry's full config, plus `files` pointing to what's on disk.

```json
{
  "model": {"x": 1.08119, "...": "...", "theta2": 305.01225},
  "simulations": {
    "sim_00": {
      "config": {"seed": 0, "n1": 1500, "...": "every SimulatorConfig field"},
      "timing": {"t_start": -12.68, "n_steps": 12681, "start": "pass", "...": "..."},
      "files": {"sim.particles": "simulations/sim_00_particles.npy"},
      "code": {"galactic_tidal_constraints": {"commit": "...", "uncommitted_changes": false},
               "jspam": {"url": "...", "commit": "...", "uncommitted_changes": false}}
    }
  },
  "images": {
    "img_0000": {
      "config": {"width": 128, "height": 128, "primary_center": [32.0, 64.0], "...": "every ImageConfig field"},
      "files": {
        "sim_00": {"img.model":       "images/sim_00_img_0000_model.npy",
                   "img.unperturbed": "images/sim_00_img_0000_unperturbed.npy"}
      }
    }
  }
}
```

- An image's address is (simulation, image, item): `images[image]["files"][simulation][item]`, the same
  three names `load_images` uses as its keys.
- Saving only images still records their simulation (config, timing, code, `"files": {}`), so it's
  clear what made them.
- `timing` is always recorded (it's small); `sim.timing` has no `.npy`.
- `code` is the code the simulation was made with: each repo's commit, and whether it had
  uncommitted changes. For this repo only changes to `src/`, `pyproject.toml` and `uv.lock` count;
  `jspam`'s commit comes from its install record (`sim.jspam_version()`).

### Write order
Every file is written to a temporary name, flushed to disk and renamed into place, so it's either
complete or absent, never half-written. Each `.npy` goes in first; only then is `model.json`
updated to list it. **An array counts as saved only if `model.json` lists it**, so a crash costs
only unfinished work, and a stray file is never mistaken for a result.

### Reuse
With an output folder, a simulation that `model.json` lists under the same name with **exactly the
same config** counts as already saved:
- anything requested that it has is **loaded instead of computed**;
- nothing it has is written again;
- arrays it's missing (e.g. `sim.unperturbed` asked for later) are computed and added.

A saved simulation with a different config is neither used nor changed, and a warning is logged.
Code versions are recorded but not compared; higher-level code can check them.

Images work the same way. A saved image is reused when **both** its image entry's config and its
simulation's config match this run's:
- it's **loaded instead of made**, and no simulation runs just to make it again (its status is
  `"reused"`);
- missing items (e.g. `img.unperturbed` asked for later) are made and added; that may need the
  simulation;
- an image entry saved with a different config is neither used nor changed (a warning is logged), and
  neither is any image of a simulation saved with a different config. Nothing listed is ever
  overwritten, so a listed file always matches the configs it's listed with;
- a failed image saves nothing and isn't listed, so a later run retries it.

### The output catalog
The catalog (a folder's `catalog.json`, or a `.json` output) is the output's table of contents:

- **`"models"`**: id → 14 parameters, for every model a run there has used. A run's rows are
  matched against it exactly (all 14 values). A model already there uses its folder; a new one gets
  the next id at the bottom. So reordered or overlapping model lists, such as GA generations, land in
  the same folders and reuse what's saved. The same model twice in one run gets one id.
  `result.ids[m]` is row m's id.
- **`"runs"`**: one record per run:

  | Key | What it is |
  |---|---|
  | `started`, `finished` | UTC times. Written when the run starts and again when it ends, so a run that crashed shows `"finished": null` |
  | `code` | The code versions, as in `model.json` |
  | `run` | Its `return` and `save` lists |
  | `simulator` | The full simulator configs |
  | `image` | The full image configs |
  | `derived` | The filled-in derived entries, as in the result's `derived_configs` |
  | `sim_metrics`, `img_metrics` | The filled-in metric entries (their values are under `values`) |
  | `composite` | The filled-in composite weights, or `null` without the section |
  | `externals` | Each used external's source (its path, or `"memory"`), shape and hash, as in the result |
  | `files` | Each once-per-run array `run.save` named (`ext.<key>`, `drv.<name>`) → its file, relative to the output folder. `{}` when none was |
  | `model_ids` | Row m's model id |
  | `status`, `seconds`, `errors` | As in the result, as lists; added when the run ends |
  | `values` | The metric values: `sim_metrics` and `img_metrics` as nested lists indexed like the result's arrays (`values["sim_metrics"][m][s][k]`), and `composite` (one per model, or `null`). A descriptive entry's cell is its dict, arrays as lists. JSON has no NaN, so **a failed or skipped scalar cell is `null`**; `simr.load_metrics` gives the arrays back with NaN |

A run without `save` writes only the catalog: its models and its record, no model folders.
**One run at a time per output.**

### Errors that protect the folder
- `run.save` with no `output`, or with a `.json` output, is an error.
- An output with an extension other than `.json` is an error.
- A model folder holding a different model than `catalog.json` says is an error, so two models'
  simulations never mix.
- Loading a simulation or array that `model.json` doesn't list is an error, even if a file exists.

## The command line
```bash
python -m galactic_tidal_constraints.simr --models models.txt --output runs/ [--config config.json] \
    [--externals externals.json] [--mode solo|local|mpi] [--workers N] [--log-level INFO] [--dry-run]
mpirun -n N python -m mpi4py.futures -m galactic_tidal_constraints.simr --mode mpi ...
```

| Flag | |
|---|---|
| `--models` | Required. The models file (below) |
| `--output` | Required: a folder or a `.json` catalog file, so every run is recorded |
| `--config` | Optional. The request as JSON, listing only what differs from the defaults; without it, one default simulator entry. `run.return` is ignored, with a warning if it names anything (nothing comes back in memory), so one config works in a notebook and here |
| `--externals` | Optional. A JSON file mapping each external key a metric names to a `.npy` path, or to `{"path": ..., "hash": ...}`; relative paths are relative to this file's folder. See Externals in [configuration.md](../configuration.md). With `--mode local` or `mpi`, every worker must be on the node that runs the command |
| `--mode` | `solo` (the default): one process. `local`: a pool of processes on this machine. `mpi`: MPI ranks, launched as above: rank 0 runs the command, the other N-1 are workers. Without `mpi4py`, with one rank, or without `-m mpi4py.futures` it stops with exit code 1 |
| `--workers` | With `--mode local`: how many processes (default: the CPUs this job may use) |
| `--log-level` | `DEBUG`, `INFO` (the default), `WARNING` or `ERROR` |
| `--dry-run` | Check the inputs as a run checks them and log the plan (`simr.plan`: counts and sizes; see Planning a run), without simulating or writing anything. Exit code 0 if the run would start, 1 if something would stop it. Worth running before submitting a job |

**The models file:** one model per line; the first 14 comma-separated values on the line are the
model. Longer lines (SPAM's 22-value strings, JSPAM's `archive/data.txt`) are fine, and one log
message counts them. Blank lines and `#` comments are skipped; fewer than 14 values, or a value that
isn't a number, is an error naming the line. `simr.read_models(path)` reads one in Python too.

**Exit codes:**

| Code | Meaning |
|---|---|
| 0 | The run ran, even if some simulations failed (they're in the catalog and the log). With `--dry-run`: nothing would stop the run |
| 1 | It couldn't run or finish: a bad models file, config or output, or a worker that died (a short message, no traceback). Under MPI a crashed worker usually ends the job with `mpirun`'s own error code |
| 2 | It ran, but every simulation failed: almost always a setup mistake |

A command-line run computes what `run.save` and the metric sections ask for; with neither, it records
its models and configs but simulates nothing (each simulation's status is `"empty"`). Metric values
and the composite are written to the run's record in the catalog; read them back with
`simr.load_metrics(output)`.

## Loading
| Function | Returns |
|---|---|
| `simr.read_models(path)` | A models file's models, `(N, 14)` |
| `simr.plan(models, request, externals=None, output=None)` | What a run would do, without simulating or writing: counts, the externals used, and the bytes `return` would hold and `save` would write (see Planning a run). `simr.format_plan(plan)` gives it as text |
| `simr.load_catalog(output)` | The catalog (of a folder or a `.json` file) as a dict |
| `simr.load_metrics(output, run=-1)` | A recorded run's metric values and composite, as the run returned them: `{"sim_metrics", "img_metrics", "composite"}`, arrays indexed like the result's, with NaN for failed scalar cells. `run` picks the run by position (-1: the last) |
| `simr.load_run(output, run=-1, externals=None)` | A recorded run's inputs, ready to run again: `{"models", "request", "externals"}` (below) |
| `simr.load_model(folder)` | A model folder's `model.json` as a dict |
| `simr.load_simulation(folder, "sim_00")` | That simulation's saved arrays, `{name: array}`, `float32`, exactly as `run` returns them |
| `simr.load_derived(folder, name=None)` | A model folder's saved derived arrays: `{(name, simulation): array}` for entries made per simulation, `{(name, simulation, image): array}` for those made per image, sorted; `name` picks one entry |
| `simr.load_images(folder, simulation=None, image=None, items=None)` | Saved images, `{(simulation, image, item): array}`, sorted, `float32`; each filter left `None` means all (below) |
| `simr.code_versions()` | The current code versions, in the form `model.json` records |

`load_run` rebuilds a run's inputs from its catalog record, under `simr.run`'s own argument names:

```python
simr.run(**simr.load_run("runs"), output="runs")           # the last run there, again
simr.run(**simr.load_run("runs", run=3), output="again")   # the fourth, into another folder
```

- **`models`**: the run's rows, in order, from the catalog's `models`.
- **`request`**: its full config, every default filled in (`result.config`). Its `run.save` list is
  included, so a run that saved data needs an output folder again; the same folder reuses what's saved.
- **`externals`**: each one as `{"path", "hash"}`, from its copy in the output folder if the run saved
  one (`run.save` `ext.<key>`), else from the path the run was given. The recorded hash goes with it,
  so the rerun stops if the file's data has changed.
- **An external given in memory and not saved can't be rebuilt.** `load_run` raises a `ValueError`
  naming each one, with its recorded shape and hash; give those as `externals={key: array or path}`.
  An array given there must hold exactly the recorded data (its hash is checked).

`load_images` picks by any mix of simulation, image entry and item:

```python
simr.load_images(folder)                                  # every saved image
simr.load_images(folder, simulation="sim_00")             # all of sim_00's images
simr.load_images(folder, image="img_0001")                # all images made with img_0001
simr.load_images(folder, "sim_00", "img_0001")            # one simulation and image entry
simr.load_images(folder, items="img.model")             # model images only (or a list of items)
# {("sim_00", "img_0001", "img.model"): array (128, 128), ...}
```

A simulation or image entry that `model.json` doesn't have is a `KeyError`, an unknown item a
`ValueError`; filters that match nothing saved give `{}`.

## Where to see it
- `examples/notebooks/simr_tour.ipynb`: a hands-on tour.
- `examples/notebooks/image_tour.ipynb`: images through `simr`, saved and loaded (its last sections).
- `tests/test_simr.py`: one section per step of the rebuild, ending with MDRN-3's image steps.
