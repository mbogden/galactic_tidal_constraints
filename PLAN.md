# Modernization Plan

The `modern` branch is a rewrite of galaxyJSPAM as an installable, reproducible package,
renamed `galactic_tidal_constraints`.
This file records what has been decided. Items are added one at a time as they are decided.
How each decision is carried out lives in the docs it links to; when they differ from this file
about how something works, the docs describe the code as it is.

## Scope
- Start by building around TNG → SPAM: simulate and optimize, for the TNG vs SPAM paper.
- Port only what that paper needs. The scope grows with the code. Machine scoring comes later.
  A basic Image Creator comes first (MDRN-3, 2026-11-16), so `simr` and the metrics stage are designed with
  a working image in mind; it isn't a full port of the old one.
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
  simulation is done; everything after it (returned data, images, metrics, saved files, externals)
  expects `float32`. That precision is enough once a simulation is done.
- **Saving:** one folder per model, one uncompressed `.npy` per array, a `model.json` record per
  model, and a `catalog.json` per output recording every run. A file counts as saved only once
  `model.json` lists it, and saved data is reused only when its config matches exactly. Layout, write
  order, the catalog and reuse: [docs/modules/simr.md](docs/modules/simr.md#saving).
- **A run's size is known before it starts** (MDRN-4 step 10): `simr.plan` and `--dry-run` give the
  counts, the sizes of what would be returned and saved, and about how long the simulations would
  take (particles × steps × the machine's measured speed), after every check a run makes, without
  simulating. Two caps stop a run before anything is computed, each settable per run: returned data
  all lands in one process's memory, so `run.max_return_gb` defaults to half the memory the process may
  use; and `run.max_hours` (4 by default) limits the simulations' worst-case time over all the run's
  workers (decided 2026-11-23).
  [docs/modules/simr.md](docs/modules/simr.md#planning-a-run-simrplan).
- **What a run leaves on record** (MDRN-4 step 7): its metric values and composite, in its catalog
  record. Externals and derived arrays are saved only when `run.save` names them: once per output,
  named by hash, or in the model's folder when made from that model's data. Saved metric values and
  derived arrays are a record, not reused (reuse waits until a metric is expensive). A recorded run
  can be rebuilt and run again (`simr.load_run`).
- **Metric or derived** (decided 2026-11-23): the size of what a function returns decides its stage.
  One float or a small dict is a **metric**; an array about the size of its input (a mask, an edge
  map, a window, weights) is a **derived** method. A metric never takes another metric's value, so
  data moves one way: items and externals → derived arrays → metrics → composite.
  [docs/modules/derived.md](docs/modules/derived.md), [docs/modules/metrics.md](docs/modules/metrics.md).
- **Derived stage** (decided 2026-11-23): an optional `derived` config section of named entries, each
  a method on externals, `sim.*` or `img.*` items, or entries above it in the list (so no loops).
  Metrics name one as `drv.<name>`, as data or as weights. An entry is computed once for the run, per
  simulation or per image, depending on what it's made from, and adds no new combinations.
  [docs/configuration.md](docs/configuration.md#derived-entries).
- **Input names** (decided 2026-11-23): every input a config names has a prefix that says where it
  comes from: `sim.` (the simulator), `img.` (the Image Creator), `drv.` (the derived section) or
  `ext.` (an external). A name without one is an error. An external's key and a derived entry's name
  are Python identifiers in plain ASCII (letters, digits and `_`, not starting with a digit): a
  small enforcement of good naming, so a name never looks like a number.
- **Weights** are optional inputs: a metric that can use them takes them last, and without them every
  row or pixel counts equally. They match the array they weigh (row weights: the metric's first
  input, the target).
- **Applications** live in the repo. Each owns its run's tables and final fitness scores, and chooses how to
  parallelize.
- **Composite scores** (MDRN-4 step 5): `simr` returns one composite per model, a weighted mean of
  the scalar metrics, when the config has a `composite` section (the weights). The math is one
  function in the metrics module (`metrics.composite`), so applications can also compute composites
  themselves from the returned arrays. Without the section there is no composite, only the metric
  arrays. How the weights are written: [docs/configuration.md](docs/configuration.md#composite).
- **Failures** are recorded, not raised: a failed simulation or image holds `None`, a failed metric cell
  NaN (or `None` for a descriptive entry's dict), gets
  the status `"failed"`, and logs its error, and the run moves on; nothing is saved for it, so a later
  run retries it. Errors while writing stop the run. Statuses, per stage and cell:
  [docs/modules/simr.md](docs/modules/simr.md#failures-and-status).
- **One door to SPAM:** only one module imports `jspam`. Each run's saved config records the commit
  of both repos and whether either has uncommitted changes.
- **Physics fixes:** a fix that changes SPAM's physics is an option, and Wallin's behavior stays
  available. `jspam`'s defaults reproduce the original program; `sim` chooses ours (disk heat,
  mass normalization, start time, dynamical friction). Each default and why:
  [docs/spam_physics.md](docs/spam_physics.md#the-settings-we-expose-simulatorconfig).

## Configuration
One config tree for everything that can change a result, plus a few runtime flags. Sections, checks
and examples: [docs/configuration.md](docs/configuration.md).
- Config tree: one section per component (`run`, `simulator`, `image`, `derived`, `sim_metrics`, `img_metrics`, `composite`, `ga`, …); `run`
  holds what to return or save.
- Precedence: defaults in code (dataclasses) < JSON config file. No command-line overrides
  (dropped 2026-11-12): the config file is the whole run, and small changes are new config files.
- Unknown keys or wrong types are an error. Each run saves its full merged config with its
  output. No separate `config.py`: `simr` checks the request for the command line and notebooks alike.
- Parameter studies are generated config files, one run per file.
- Runtime flags (`argparse`) don't change results and stay outside the tree: `--config`,
  `--externals`, `--log-level`, `--mode`, output folder.
- **Models** are their own input, separate from the config: the config says how to simulate (and
  later image and measure), and the models say what. Either can then change while the other stays
  fixed. On the command line they come from a text file (`--models`).
- **Externals** are their own input too, like the models: arrays from outside the run (e.g. a
  target's orbit, image or particles) that metrics compare each model against. The config only names
  them, as `ext.<key>`; the arrays come with the run (`externals=`, or `--externals`). Each is checked,
  converted to `float32` and hashed before anything runs, and the run records its source, shape and
  hash. A pool shares them through one node's memory, so such a pool is single-node (see
  Parallelization, In `simr`).
- **The command line** requires an output, so every run leaves its models, configs and statuses on
  disk; `run.return` is ignored there, so one config works in a notebook and on the command line.
  Flags, the models file and exit codes: [docs/modules/simr.md](docs/modules/simr.md#the-command-line).

### Models and stage lists
- A model is SPAM's first 14 parameters (`x` to `theta2`). We only use the extended potential, which
  never reads the last 8 (`epsilon1/2`, the six `rscale`), so they aren't tracked anywhere. `sim`
  appends fixed values for them when it calls `jspam`. Why: [docs/spam_physics.md](docs/spam_physics.md).
- Each stage can have one entry or a list, directly under the stage, and lists combine as every
  combination.
- Metrics come back as full arrays on two levels, with no automatic reduction: `sim_metrics` for
  metrics on `sim.*` items and `img_metrics` for metrics with any `img.*` input.
- Other data is kept only when a run asks for it by name, in `run.return` (passed back in memory) or
  `run.save` (written to disk); `simr` computes only what the metrics, `return` and `save` need.
- The seed is part of `SimulatorConfig` (0 by default), and `simr` uses each simulator entry's seed
  for every model, so metric differences come only from the models' parameters. Why:
  [docs/spam_physics.md](docs/spam_physics.md), `seed`.

Details and an example: [docs/architecture.md](docs/architecture.md#models-and-stage-lists).

## Image stage (decided 2026-11-16)
A basic Image Creator, `simr/image_creator.py`, numpy only. How it works (the frame, the steps, the
checks): [docs/modules/image_creator.md](docs/modules/image_creator.md); in `simr` (the `image`
section, `img.*` items, saving and reuse): [docs/modules/simr.md](docs/modules/simr.md).
- **Target-free:** an `ImageConfig` sets the image size and the pixels the two galaxies' centers
  land on. Higher-level code decides whether those come from a target image.
- **The frame** is one matrix per model, built from the model's parameters, not from the particles,
  so every image of a model, and its orbit, share it exactly.
- Not yet: radial brightness, luminosity balance, separate galaxy images, other view planes, each image
  entry choosing its own items.

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

### In `simr` (decided 2026-11-12)
How it's done (the catalog while results arrive, crashes, MPI logging, sharing externals):
[docs/modules/simr.md](docs/modules/simr.md#spreading-the-work-map).
- The unit of work is **one model**, with all its simulator entries; one task per model, handed out as
  workers free up (no shrinking batches for now).
- Only the main process touches the catalog.
- A worker crash ends the run; a rerun resumes from what's saved.
- Externals shared through a node's memory (in-memory or non-`float32`) keep a run on one node: an
  MPI job spanning more nodes stops before anything is written (decided 2026-11-23). Large external
  data stays local to a node, and different data sets (e.g. different targets' particles) get their
  own jobs. MPI runs without such externals can still span nodes.
- `mpi4py` is an optional dependency group (decided 2026-11-16), and `simr` never imports it.

## Environment
One Docker image, run with Apptainer on the cluster.
- CUDA 12.x only (the cluster driver is 12.4)
- Ubuntu 24.04
- OpenMPI 4.1.6, matching the cluster
- Built locally, pushed to Docker Hub, pulled on the cluster with `apptainer pull`
- **Tags:** a tag names one exact image, and a pushed tag is never overwritten. Which changes need a
  new tag, and the steps: [docs/guides/docker_image.md](docs/guides/docker_image.md).

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

## TNG targets
How it works: [docs/modules/tng.md](docs/modules/tng.md). What the TNG values mean, with sources:
[docs/tng_physics.md](docs/tng_physics.md).
- The TNG code starts from the data already in hand (decided 2026-11-24): the target table and the
  target archives, both HDF5, in the data folder. It doesn't download from the TNG server.
- One module, `tng.py`, grown a function at a time. Readers return values as stored; unit and frame
  conversions are separate functions, so each can be tested on its own.
- `h5py` is its only added dependency: the `tng` group, which `dev` includes so the tests always run.
- IllustrisTNG is acknowledged, with the citations its project asks for, in the README and
  `docs/tng_physics.md` (2026-11-24). No TNG data goes in the repo.
- Tests on the real files run only when `GTC_DATA_DIR` names the data folder.
- What the stored values mean is pinned by tests against the TNG catalog (2026-11-24): a galaxy
  measured from the archive's particles must give back the catalog's mass, velocity, spin and
  half-mass radius. The measuring functions work in the units they are given.
- The box's wrap-around is undone explicitly, on stored positions, relative to a reference point
  (2026-11-24). Only positions relative to a galaxy's center are used after that.
- A galaxy's center is its nucleus (the catalog's most bound particle) and its velocity is the
  nucleus's: the mean of the stars within 1 kpc, cross-checked against its black hole (2026-11-24).
  Not the catalog's halo averages.
- The frame for SPAM (2026-11-24): the primary's nucleus at the origin and at rest; x toward the
  secondary and z along the orbit's angular momentum, both at the target snapshot. The Hubble flow is
  added to relative velocities unless switched off. Rotations are plain 3×3 matrices.
- Which galaxy is the primary is a named rule (2026-11-25): `"table"` (the table's own label) or
  `"stars"` (more mass in the stars it owned before the encounter). Not the catalog's total mass,
  which swings during an encounter. The table and archives are never relabelled.
- A target is measured by one function, `measure_target`, whose choices are the fields of
  `TngTargetConfig` (2026-11-25): the primary rule, the nucleus radius (1 kpc), the Hubble flow (on),
  which mass goes to SPAM (all particles the galaxy owned before the encounter; the catalog's and the
  mass of those particles within the disk radius are recorded too), the disk radius (5 stellar
  half-mass radii), and which stars give the spin (those within the disk radius; all is an option).
  It writes no files. These defaults are first choices, to be compared against the alternatives later.
- Units change in two explicit steps (2026-11-24): stored TNG values to physical (kpc, km/s, M☉),
  then physical to SPAM's units. Each step is one function with its factors in one table.

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
  - Modules are added only when their code is needed: one file per component, split into a
    subpackage only when it grows.
  - Applications start as runnable modules (`python -m galactic_tidal_constraints.<module>`);
    a short command can come later.
  - Committed notebooks have their outputs stripped.
- **Data:** never in git; it lives outside the repo.
- **Module docs:** every module has a short doc in `docs/modules/<module>.md`: purpose, main
  functions with inputs and outputs, known issues. Make sure one exists; it can be written with
  the module or after it. Mention future features only when they shape the current design;
  other ideas don't go in module docs.

## Documentation
`docs/` has three parts: the top level for the overview and references (architecture,
configuration, SPAM's physics), `docs/modules/` for one doc per code module, and `docs/guides/` for
how-to guides.

| Doc | What goes in it | When |
|---|---|---|
| `docs/architecture.md` | The levels (experiment, run, workflow, stage), and what passes between them | Exists |
| `docs/configuration.md` | The config's sections, externals, how stage lists combine, checks, what's recorded, examples (fields stay in each stage's doc) | Exists; a row per stage as stages are added |
| `docs/spam_physics.md` | What SPAM's parameters and options do, our defaults vs Wallin's, open questions | Exists |
| `docs/modules/simr.md` | The `simr` module's functions, inputs and outputs, saving, the command line | Exists; updated with each `simr` step |
| `docs/modules/simulation.md` | The Simulator stage's functions, inputs and outputs | Exists |
| `docs/modules/image_creator.md` | The Image Creator's config, pixel frame and steps | Exists |
| `docs/modules/metrics.md` | The Metrics stage: the method registry, input kinds, each method | Exists |
| `docs/modules/derived.md` | The Derived stage: derived or metric, the registry, each method | Exists (the methods; not yet in `simr`) |
| `docs/modules/tng.md` | The TNG readers: the target table and a target archive, and what their stored values are | Exists |
| `docs/modules/<module>.md` | The same for each later module | With the module or after it |
| `docs/guides/running_spam.md` | Compile and run SPAM itself, from Python or as the Fortran program | Exists |
| `docs/guides/running_on_babbage.md` | A how-to for setting up an experiment and launching runs on Babbage | Exists (CPU only) |
| `docs/guides/docker_image.md` | Build, check, tag and push the image; pull it as a `.sif`; run in it | Exists |

## Code standards
- **No import side effects:** importing a module only defines things (functions, classes,
  constants). It never runs code: no SPAM or MPI initialization, no logging setup, no file
  reads, no `sys.path` changes. That work happens in functions, called from an application.
  Avoid module-level mutable state; pass values in as arguments.
- **Type hints** on function signatures (arguments and return values), checked by the editor
  (VS Code / Pylance). No strict type checker for now. Array shapes go in the docstring.

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
  it logs once and records the failure instead of raising.

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
