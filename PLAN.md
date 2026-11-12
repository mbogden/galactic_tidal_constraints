# Modernization Plan

The `modern` branch is a rewrite of galaxyJSPAM as an installable, reproducible package,
renamed `galactic_tidal_constraints`.
This file records what has been decided. Items are added one at a time as they are decided.

## Scope
- Start by building around TNG → SPAM: simulate and optimize, for the TNG vs SPAM paper.
- Port only what that paper needs. The scope grows with the code. Image creation and
  machine scoring come later.
- New code is defined as it is written, starting with MDRN-1. It isn't based on an audit of the
  old code's use cases, or compared against the old code.

## Structure
How these fit together is described in [docs/architecture.md](docs/architecture.md).
- **Levels:** experiment, application / run, workflow and stage. "Pipeline" isn't a structural term.
- **Stages** do one thing per call, take data in and return data out, and never touch disk or
  parallelization.
- **Disk:** the `simr` module is in charge of reads and writes, including saving and resuming.
  `info_module` is not ported as a whole.
- **Precision:** `simr` converts simulated arrays (particles, orbits) to `float32` as soon as a
  simulation is done;
  everything after it (returned data, images, saved files) expects `float32`. That precision is
  enough once a simulation is done.
- **Saved arrays** are `.npy` files, one per array, written with `np.save` (no compression; it
  saves little on simulated floats). Separate files let one array be added or deleted without
  touching the others.
- **One folder per model:** `<output>/model_000000/` holds `model.json` and a subfolder per stage:
  `simulations/` (`<simulation>_<item>.npy`, e.g. `simulations/sim_00_particles.npy`), later
  `images/`. `model.json` is the model's record: its 14 named parameters, and each simulation by
  name with its full config (seed included), timing, files and the code that made it (both repos'
  commits and whether either had uncommitted changes); later images and fitness scores.
- **Write order:** each `.npy` is written atomically (to a temporary name, then renamed); only once
  it's in place is `model.json` updated to list it. An array counts as saved only if `model.json`
  lists it.
- **Output catalog:** every run given an output writes a catalog, whether or not it saves data.
  It holds "models" (id → 14 parameters, every model a run there has used) and "runs" (one record
  per run: times, code versions, what was returned and saved, the full stage configs, the model id
  of each row, each simulation's status, time and error, and later the scores). A run's models are
  matched exactly against the catalog; new ones get the next id at the bottom, so model folders
  (`model_<id>`) stay the same however runs order or overlap their models. The record is written
  when a run starts and again when it ends. One run at a time per output.
- **The output is a folder or a `.json` file,** told apart by name: no extension is a folder
  (`<folder>/catalog.json`, plus model folders for `run.save`), `.json` is a catalog file on its
  own (no model data, so `run.save` with a `.json` output is an error), and any other extension is
  an error.
- **Reuse:** with an output folder, a simulation `model.json` lists under the same name with exactly
  the same config counts as already saved. Anything requested that it has is loaded instead of
  computed, nothing it has is written again, and arrays it's missing are computed and added. A
  different config is neither used nor changed. Code versions are recorded but not compared;
  higher-level code can check them.
- **Applications** live in the repo. Each owns its run's tables and final scores, and chooses how to
  parallelize. Meta scores are computed here.
- **Failures:** a failed model returns `NaN` and writes the error to the log. In `simr`, a
  simulation (one model on one simulator entry) fails as a whole when computing any requested item
  raises an error or a simulated array holds NaN or inf; the run moves on. Its cells hold `None`,
  nothing is saved for it, and nothing is recorded in `model.json`, so a later run retries it.
  Errors while writing stop the run.
- **Status:** each stage reports a status, time and error for each of its cells, indexed like its
  data (`result.status["sim"][m, s]`: `"ran"`, `"reused"` when nothing had to be computed, or
  `"failed"`; later stages mark work that depends on a failure `"skipped"`). A run that saves adds
  them to its record in `catalog.json` when it ends.
- **One door to SPAM:** only one module imports `jspam`. Each run's saved config records the commit
  of both repos and whether either has uncommitted changes.
- **Physics fixes:** a fix that changes SPAM's physics is an option, and Wallin's behavior stays
  available. `jspam`'s defaults reproduce the original program; `sim` chooses ours.
  - Disk heat: `"centered"` by default in `sim` (Wallin's kicks are always positive, so heated
    disks drift).
  - Mass normalization: `"total"` by default in `sim` (divide the unit galaxy by its actual total
    mass, so it's always 1; Wallin's fixed 7.13333 makes the mass-model values change the total).
  - Start time: `"pass"` by default in `sim`: start SPAM's lead (10 × `rout1` / speed) before the most
    recent close pass, with older passes as options and a fixed start time allowed. SPAM's search picks
    the closest pass back to −30, which on a repeating orbit is decided by rounding, then replaces a
    start before −12 with −5. With no such pass back to −30, start at −5 (reported as a fallback), so the
    model still gets a score.
  - Dynamical friction: `lnl` = 0.1 by default in `sim` (Wallin's 0.001 makes it almost nothing).

## Configuration
One config tree for everything that can change a result, plus a few runtime flags.
- Config tree: one section per component (`run`, `simulator`, `image`, `fitness`, `ga`, …); `run`
  holds per-run values like the target and what to return or save.
- Precedence: defaults in code (dataclasses) < JSON config file. No command-line overrides
  (dropped 2026-11-12): the config file is the whole run, and small changes are new config files.
- Unknown keys or wrong types are an error. Each run saves its full merged config with its
  output.
- Parameter studies are generated config files, one run per file.
- Runtime flags (`argparse`) don't change results and stay outside the tree: `--config`,
  `--log-level`, `--mode`, output folder.
- Models are their own input, separate from the config: the config says how to simulate (and
  later image and score), and the models say what. Either can then change while the other stays
  fixed. On the command line they come from a text file (`--models models.txt`): one model per
  line, and the first 14 comma-separated values on the line are the model. Longer lines (SPAM's
  22-value strings, JSPAM's `archive/data.txt`) are accepted, and one log message says how many
  lines had more than 14 values. Fewer than 14 numbers is an error naming the line. Blank lines and
  `#` comments are ignored. The catalog records each run's models with its configs, so the output
  folder still says what was run.
- On the command line an output (folder or `.json` file) is required, so every run leaves its
  models, configs and statuses on disk; `run.save` is optional.

### Models and stage lists
- Encounter parameters are the one hard input every run needs, and they're their own list. Each
  stage can have one entry or a list, directly under the stage.
- A model is SPAM's first 14 parameters (`x` to `theta2`). We only use the extended potential, which
  never reads the last 8 (`epsilon1/2`, the six `rscale`), so they aren't tracked anywhere. `sim`
  appends fixed values for them when it calls `jspam`. Why: [docs/spam_physics.md](docs/spam_physics.md).
- SIMR and the workflow always return a 4D array, with no automatic reduction:
  `scores[model, simulator, image, fitness]`.
- Other data is kept only when a run asks for it by name. The `run` section has two lists of
  `stage.item` names (e.g. `sim.particles`, `sim.orbit`, later `image.model`): `return` (passed back
  in memory) and `save` (written to disk). They're independent; a run can do both. Each stage
  publishes its names, and an unknown name is a config error. `simr` computes only what the scores,
  `return` and `save` need, so with both lists empty only the scores come back.
- The seed is part of `SimulatorConfig` (`seed`, 0 by default), and `simr` uses each simulator
  entry's seed for every model. SPAM's draws don't depend on the model, so all models share one
  disk pattern and score differences come only from their parameters. Comparing seeds means
  entries that differ only in `seed`. Why: [docs/spam_physics.md](docs/spam_physics.md), `seed`.

Details and an example are in [docs/architecture.md](docs/architecture.md).

## Parallelization
Code takes a `map` function (default: built-in `map`). The application picks it once from `--mode`:
- solo: built-in `map`
- local: `concurrent.futures.ProcessPoolExecutor`
- MPI: `mpi4py.futures.MPIPoolExecutor`, launched with `python -m mpi4py.futures`

Both pools hand out tasks dynamically, one at a time. Start simple; add per-task error capture and
progress reporting later if needed.
- The application chooses how to split the work, sending each process a slice. Shrinking batches
  are one example, and suit SPAM, where a simulation can take from a second to over a minute.
- SIMR's own pool is simple and optional. Higher-level code can run its own and call lower-level
  code directly.
- Code that owns a pool creates it once and shuts it down at the end (`with`). It passes the pool's
  `map` down to the code that needs it.
- Parallelize at one level only: the outermost loop with many independent tasks, each taking at
  least seconds (e.g. batches of models in one optimizer generation).
- No nesting: a task running in a worker never starts its own pool.
- The lowest level stays plain: functions like a single SPAM run know nothing about pools or MPI, so
  they can be tested and called on their own.
- Parallelism above the pool comes from Slurm, not Python (e.g. a job array with one task per target,
  each running its own pool).

## Environment
One Docker image, run with Apptainer on the cluster.
- CUDA 12.x only (the cluster driver is 12.4)
- Ubuntu 24.04
- OpenMPI 4.1.6, matching the cluster
- Built locally, pushed to Docker Hub, pulled on the cluster with `apptainer pull`

### Packaging and install
- **Packaging:** `pyproject.toml` + `uv`. The lockfile is part of reproducibility.
- **Install:** JSPAM is a normal dependency installed from git at a pinned commit, not nested in
  this repo. Users run `uv sync`; clusters use the container. Local development keeps the repos
  side by side.

### SPAM
- **Own repo:** a fork of `jfwallin/JSPAM`, for now (it can move to a branch in that repo later).
  Start from a fresh copy of the original and reimplement only the changes needed. This repo depends
  on the fork at a pinned commit. Fortran only for now; JSPAM's other language versions are not used.
- **Build:** `meson-python`, with `f2py` generating the Python interface. The package is named
  `jspam`. Work happens on `simr-branch`; the fork's `master` stays as the original code.

## Repository
- **Name:** `galactic_tidal_constraints` for the GitHub repo, the local folder and the Python
  package (`import galactic_tidal_constraints as gtc`).
- **Layout:**
  ```
  galactic_tidal_constraints/
  ├── pyproject.toml, uv.lock, README.md, PLAN.md, CLAUDE.md
  ├── src/galactic_tidal_constraints/   ← the package
  ├── tests/
  ├── docs/
  ├── docker/
  └── examples/                         ← example configs, a generic Slurm template, curated notebooks
  ```
  - Only `src/galactic_tidal_constraints/` is installed; everything else stays in the repo.
  - Modules are added only when their code is ported: one file per component, split into a
    subpackage only when it grows.
  - Applications start as runnable modules (`python -m galactic_tidal_constraints.<module>`);
    a short command can come later.
  - Committed notebooks have their outputs stripped.
- **Data:** never in git; it lives outside the repo.
- **Design notes:** before a module is ported, a short note in `docs/design/<module>.md`:
  purpose, main functions with inputs and outputs, what it ports or changes from the old
  code, known issues. Mention future features only when they shape the current design;
  other ideas don't go in design notes.

## Documentation
| Doc | What goes in it | When |
|---|---|---|
| `docs/architecture.md` | The levels (experiment, run, workflow, stage), and what passes between them | Exists |
| `docs/configuration.md` | Config fields, stage lists, how they combine, the 4D score array | When the fields are real, or when that section outgrows the architecture doc |
| `docs/design/simr.md` | The `simr` module's functions, inputs and outputs | Exists; updated with each `simr` step |
| `docs/design/<stage>.md` | The same for each stage | Before each stage is written |
| `docs/running_experiments.md` | A how-to for setting up an experiment and launching runs on Babbage | Once there's something to run |

## Code standards
- **No import side effects:** importing a module only defines things (functions, classes,
  constants). It never runs code: no SPAM or MPI initialization, no logging setup, no file
  reads, no `sys.path` changes. That work happens in functions, called from an application.
  Avoid module-level mutable state; pass values in as arguments.
- **Type hints** on function signatures (arguments and return values), checked by the editor
  (VS Code / Pylance). No strict type checker for now. Array shapes go in the docstring.
- **`ruff`** for linting (`ruff check`) and formatting (`ruff format`), configured in
  `pyproject.toml`.

### Testing
`pytest`, with tests in `tests/`.
- Compare numbers with `numpy.testing.assert_allclose`, not `==`.
- Golden tests: save reference outputs once, compare later runs against them. Orbits match
  exactly; particle runs only statistically until the Fortran RNG takes a seed.
- Mark long tests `@pytest.mark.slow` so quick runs can skip them.
- Parallel code is tested in solo mode (built-in `map`); one small test checks the MPI pool
  under `mpiexec`.
- Later: coverage (`pytest-cov`) and CI (GitHub Actions), once the Fortran build is settled.

### Logging
Python's standard `logging` module, used throughout.
- DEBUG logs trace what the code did; checks that a result is correct belong in tests.
- DEBUG statements are kept for development and testing. They must cost almost nothing when
  DEBUG is off:
  - Lazy formatting: `logger.debug("x = %s", x)`, not f-strings or `%` inside log calls.
  - No logging inside tight loops (per particle, per time step); log summaries per run or
    per generation.
- One logger per module: `logger = logging.getLogger(__name__)`. Modules never configure logging.
- Configure once in the application: the level comes from `--log-level`, and the format
  includes time, level, module, function and MPI rank. Quiet noisy libraries individually
  (e.g. `logging.getLogger("matplotlib").setLevel(logging.WARNING)`).
- One log call per error: `logger.exception("...")` records the message and traceback,
  then raise. Don't log the same error again further up. The workflow is where a failed model stops:
  it logs once and returns `NaN` instead of raising.

## Branches
- `modern`: created from `working_research`. The first commit cleared the old tree, so the
  old history is still reachable.
- `working_research`: stays in use for research until `modern` reaches parity.
- Old branches (`Master`, `old_master`) are kept as archive tags. Deleting them is not required.
- **Small edits** can be committed directly to `modern`.
- **Side branches:** large changes in code or experiments get a short branch off `modern`, merged
  back when tested, then deleted. Name: `<what>-branch`, lowercase with `-` (e.g. `env-2-branch`,
  `packaging-branch`). Older branches (e.g. `exp/e1-base-image`) keep their names.
- **At parity:** `modern` becomes `main`, `working_research` becomes `legacy`, and the GitHub
  default branch becomes `main`.

## To port
Science changes committed to `working_research` after `modern` was created
(2026-10-27, from `9f87fea`). Each one must also be made on `modern`.

| Date | Commit | Change | Ported? |
|---|---|---|---|
