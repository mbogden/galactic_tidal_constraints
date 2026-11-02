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

## Decided
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
  - Entry points start as runnable modules (`python -m galactic_tidal_constraints.<module>`);
    a short command can come later.
  - Committed notebooks have their outputs stripped.
- **No import side effects:** importing a module only defines things (functions, classes,
  constants). It never runs code: no SPAM or MPI initialization, no logging setup, no file
  reads, no `sys.path` changes. That work happens in functions, called from an entry point.
  Avoid module-level mutable state; pass values in as arguments.
- **SPAM lives in its own repo:** a fork of `jfwallin/JSPAM`, for now (it can move to a branch
  in that repo later). Start from a fresh copy of the original and reimplement only the changes
  needed. This repo depends on the fork at a pinned commit. Fortran only for now; JSPAM's other
  language versions are not used.
- **SPAM build:** `meson-python`, with `f2py` generating the Python interface. The package is
  named `jspam`. Work happens on `simr-branch`; the fork's `master` stays as the original code.
- **Install:** JSPAM is a normal dependency installed from git at a pinned commit, not nested in
  this repo. Users run `uv sync`; clusters use the container. Local development keeps the repos
  side by side.
- **Separation of storage:** stages and the pipeline take and return data, and never read or
  write run or target folders. Disk access lives only in application wrappers, which handle
  saving, resuming and folder layouts. `info_module` is not ported as a whole.
- **One door to SPAM:** only one module imports `jspam`. Each run's saved config records the commit
  of both repos and whether either has uncommitted changes.
- **Type hints** on function signatures (arguments and return values), checked by the editor
  (VS Code / Pylance). No strict type checker for now. Array shapes go in the docstring.
- **`ruff`** for linting (`ruff check`) and formatting (`ruff format`), configured in
  `pyproject.toml`.
- **Design notes:** before a module is ported, a short note in `docs/design/<module>.md`:
  purpose, main functions with inputs and outputs, what it ports or changes from the old
  code, known issues. Mention future features only when they shape the current design;
  other ideas don't go in design notes.
- **Packaging:** `pyproject.toml` + `uv`. The lockfile is part of reproducibility.
- **Environment:** one Docker image, run with Apptainer on the cluster.
  - CUDA 12.x only (the cluster driver is 12.4)
  - Ubuntu 24.04
  - OpenMPI 4.1.6, matching the cluster
  - Built locally, pushed to Docker Hub, pulled on the cluster with `apptainer pull`
- **Data:** never in git; it lives outside the repo.
- **Input:** one config tree for everything that can change a result, plus a few runtime flags.
  - Config tree: one section per component (`run`, `simulator`, `fitness`, `ga`, …); `run`
    holds per-run values like target and seed.
  - Precedence: defaults in code (dataclasses) < JSON config file < command-line `key=value`
    overrides (e.g. `ga.population=128`).
  - Unknown keys or wrong types are an error. Each run saves its full merged config with its
    output.
  - Parameter studies are lists of override sets, one run per set.
  - Runtime flags (`argparse`) don't change results and stay outside the tree: `--config`, the
    override list, `--log-level`, `--mode`, output folder.
- **Testing:** `pytest`, with tests in `tests/`.
  - Compare numbers with `numpy.testing.assert_allclose`, not `==`.
  - Golden tests: save reference outputs once, compare later runs against them. Orbits match
    exactly; particle runs only statistically until the Fortran RNG takes a seed.
  - Mark long tests `@pytest.mark.slow` so quick runs can skip them.
  - Parallel code is tested in solo mode (built-in `map`); one small test checks the MPI pool
    under `mpiexec`.
  - Later: coverage (`pytest-cov`) and CI (GitHub Actions), once the Fortran build is settled.
- **Parallel:** code takes a `map` function (default: built-in `map`). The entry point picks it
  once from `--mode`:
  - solo: built-in `map`
  - local: `concurrent.futures.ProcessPoolExecutor`
  - MPI: `mpi4py.futures.MPIPoolExecutor`, launched with `python -m mpi4py.futures`

  Both pools hand out tasks dynamically, one at a time. Start simple; add per-task error
  capture and progress reporting later if needed.
  - The entry point creates the pool once and shuts it down at the end (`with`). It passes
    the pool's `map` down to the code that needs it.
  - Parallelize at one level only: the outermost loop with many independent tasks, each
    taking at least seconds (e.g. the models in one optimizer generation).
  - No nesting: a task running in a worker never starts its own pool.
  - The lowest level stays plain: functions like a single SPAM run know nothing about pools
    or MPI, so they can be tested and called on their own.
  - Parallelism above the pool comes from Slurm, not Python (e.g. a job array with one task
    per target, each running its own pool).
- **Logging:** Python's standard `logging` module, used throughout.
  - DEBUG logs trace what the code did; checks that a result is correct belong in tests.
  - DEBUG statements are kept for development and testing. They must cost almost nothing when
    DEBUG is off:
    - Lazy formatting: `logger.debug("x = %s", x)`, not f-strings or `%` inside log calls.
    - No logging inside tight loops (per particle, per time step); log summaries per run or
      per generation.
  - One logger per module: `logger = logging.getLogger(__name__)`. Modules never configure logging.
  - Configure once at the entry point: the level comes from `--log-level`, and the format
    includes time, level, module, function and MPI rank. Quiet noisy libraries individually
    (e.g. `logging.getLogger("matplotlib").setLevel(logging.WARNING)`).
  - One log call per error: `logger.exception("...")` records the message and traceback,
    then raise. Don't log the same error again further up.

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
