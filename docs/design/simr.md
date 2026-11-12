# SIMR workflow (`simr/simr.py`)

`simr` runs the simulations a request asks for, for many models, and hands back only the data the
request names. With an output it records every run in a catalog, and it can save data to an output
folder and load it back. It sits one level
above the Simulator stage (`simr/simulation.py`, imported as `sim`), which runs one simulation at a
time and knows nothing about disk.

```python
from galactic_tidal_constraints.simr import simr

result = simr.run(models, request, output=None)
```

Only what exists today is described here. Decisions behind it are in [PLAN.md](../../PLAN.md).

## Inputs

### Models
A NumPy array of shape `(N, 14)`, one model per row, in `sim.PARAM_NAMES` order (`x` to `theta2`).
One model of shape `(14,)` also works. Rows must be finite.

### The request
A plain dict, shaped like the config file to come:

```python
request = {
    "run": {
        "return": ["sim.orbit", "sim.timing"],   # data to hand back in memory
        "save": ["sim.particles"],               # data to write to the output folder
    },
    "simulator": [{"lnl": 0.1}, {"lnl": 0.01}],  # one dict or a list; each becomes a SimulatorConfig
}
```

| Key | Default | Meaning |
|---|---|---|
| `run.return` | `[]` | Names of the data to hand back |
| `run.save` | `[]` | Names of the data to save. Independent of `return`: a run can do both |
| `simulator` | one `SimulatorConfig()` | A dict (keys are `SimulatorConfig` fields) or a `SimulatorConfig`, or a list of them |

**Every model runs on every simulator entry.** Unknown keys, unknown names, wrong types and bad
values are errors that say where the problem is (e.g. `simulator[0].lnl must be a number: '0.1'`,
`simulator[1]: lnl can't be negative`), raised before anything runs. Types are checked against
`SimulatorConfig`'s type hints; an integer is fine for a number field (JSON writes `1.0` as `1`), and
`true` / `false` are never numbers.

### Data names
The Simulator stage can hand back:

| Name | Each cell holds |
|---|---|
| `sim.particles` | `sim.simulate_particles`: every particle at the end, `(n1 + n2 + 1, 6)` |
| `sim.unperturbed` | `sim.unperturbed_particles`: the same disks at today's centers, `(n1 + n2 + 1, 6)` |
| `sim.orbit` | `sim.simulate_orbit`: the secondary's center at every step, `(n_steps + 1, 7)` |
| `sim.timing` | the dict from `sim.get_timing` (start time, steps, which pass) |

**Only what `return` and `save` name is computed.** Asking only for `sim.orbit` never runs
particles. Simulated arrays are converted to `float32` as soon as a simulation is done; everything
after expects `float32`.

### Seeds
The seed is part of each simulator entry (`SimulatorConfig.seed`, 0 by default), and `simr` uses it
for every model. SPAM draws the same random numbers for any model, so with one seed every model's
disks share one pattern, stretched by `rout` and tilted by `theta` and `phi`: score differences
between models come only from their parameters. To compare draws, list entries that differ only in
`seed`. Details: [spam_physics.md](../spam_physics.md), `seed`.

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
| `data` | One entry per name in `return`, and nothing else |
| `ids` | Each row's model id in the output's catalog, or `None` without an output (see The output catalog) |
| `status`, `seconds`, `errors` | Per stage (`"sim"` so far), indexed like its data; see Failures and status |

Each `data` entry is a NumPy object array of cells, indexed by the levels its stage depends on:
`[model, simulator]` for `sim.*`. A cell holds that simulation's own array, so orbits of different
lengths fit side by side:

```python
result.data["sim.orbit"][12, 1]         # model 12's orbit from simulator entry 1
result.simulators[1].lnl                # what entry 1 was
```

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

## Failures and status
A simulation (one model on one simulator entry) **fails as a whole** when computing any requested item
raises an error, or when any simulated array holds NaN or inf. `sim.timing` isn't checked: it
legitimately holds NaN (e.g. `closest_distance` for a fixed start). A failure doesn't stop the run:

- its cells in `result.data` hold `None`;
- nothing is saved for it, because every item is computed and checked before anything is written;
- nothing is recorded in `model.json`, so a later run retries it;
- the error is logged with its traceback.

Errors while **writing** files (a full disk, no permission) aren't caught: they stop the run.

Each simulation's outcome comes back per stage, indexed like the data:

| Field | Each cell |
|---|---|
| `result.status["sim"][m, s]` | `"ran"`, `"reused"` (saved, and nothing had to be computed), `"failed"`, or `"empty"` (the request asked for nothing: no `return`, no `save`) |
| `result.seconds["sim"][m, s]` | Wall time, loading included |
| `result.errors["sim"][m, s]` | `"Type: message"` for a failure, else `None` |

```python
(result.status["sim"] == "failed").sum()     # how many failed
```

The run ends with a log line counting how many simulations ran, were reused, failed and were empty.

## Saving

### Folder layout
```
<output>/
    catalog.json                  "models": id -> 14 parameters; "runs": one record per run that saved here
    model_000000/                 one folder per model id
        model.json                the model's record
        simulations/
            sim_00_particles.npy  one file per saved array: <simulation>_<item>.npy
            sim_00_orbit.npy
```

A simulation's name is `sim_<s>` for simulator entry s (`sim_00`, `sim_01`, …). Arrays are `.npy`
files, one per array, `float32`, uncompressed, so one array can be added or deleted without
touching the others.

### `model.json`
One per model folder: the model's 14 named parameters, and each saved simulation by name.

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
  }
}
```

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
  | `image`, `fitness` | Empty until those stages exist |
  | `model_ids` | Row m's model id |
  | `status`, `seconds`, `errors` | As in the result, as lists; added when the run ends |

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
python -m galactic_tidal_constraints.simr --models models.txt --output runs/ [--config config.json] [--log-level INFO]
```

| Flag | |
|---|---|
| `--models` | Required. The models file (below) |
| `--output` | Required: a folder or a `.json` catalog file, so every run is recorded |
| `--config` | Optional. The request as JSON, listing only what differs from the defaults; without it, one default simulator entry. `run.return` is ignored with a warning (nothing comes back in memory), so one config works in a notebook and here |
| `--log-level` | `DEBUG`, `INFO` (the default), `WARNING` or `ERROR` |

**The models file:** one model per line; the first 14 comma-separated values on the line are the
model. Longer lines (SPAM's 22-value strings, JSPAM's `archive/data.txt`) are fine, and one log
message counts them. Blank lines and `#` comments are skipped; fewer than 14 values, or a value that
isn't a number, is an error naming the line. `simr.read_models(path)` reads one in Python too.

**Exit codes:**

| Code | Meaning |
|---|---|
| 0 | The run ran, even if some simulations failed (they're in the catalog and the log) |
| 1 | It couldn't run: a bad models file, config or output (a short message, no traceback) |
| 2 | It ran, but every simulation failed: almost always a setup mistake |

Until fitness exists, a run computes only what `run.save` asks for, so a command-line run without
`run.save` records its models and configs but simulates nothing: each simulation's status is
`"empty"`.

## Loading
| Function | Returns |
|---|---|
| `simr.read_models(path)` | A models file's models, `(N, 14)` |
| `simr.load_catalog(output)` | The catalog (of a folder or a `.json` file) as a dict |
| `simr.load_model(folder)` | A model folder's `model.json` as a dict |
| `simr.load_simulation(folder, "sim_00")` | That simulation's saved arrays, `{name: array}`, `float32`, exactly as `run` returns them |
| `simr.code_versions()` | The current code versions, in the form `model.json` records |

## Where to see it
- `examples/notebooks/simr_tour.ipynb`: a hands-on tour.
- `tests/test_simr.py`: one section per step of the rebuild.
