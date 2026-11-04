# Design: `simr`

`src/galactic_tidal_constraints/simr/simr.py` holds the workflow, all disk reads and writes, and running
in parallel. For now it only runs the Simulator stage. Image and fitness stages come later, and with
them the 4D score array in [architecture.md](../architecture.md).

## Package layout
```
src/galactic_tidal_constraints/simr/
├── __main__.py   ← python -m galactic_tidal_constraints.simr
├── config.py     ← the config tree (see ../configuration.md)
├── sim.py        ← the Simulator stage
└── simr.py       ← this module
```
`img.py` and `fit.py` join them when their code is written.

## Main functions
| Function | What it does |
|---|---|
| `run(config, output, map_function, n_workers)` | Runs every model on every simulator; returns summaries |
| `run_batch(batch)` | The workflow for one process: every task in its slice |
| `run_task(task, run_seed, output)` | One simulation: reuse, or run and save |
| `make_tasks(config)`, `make_batches(...)` | Expand the config, and split it into shrinking batches |
| `save_simulation(folder, settings, simulation)` | Writes one simulation |
| `load_simulation(folder)` | Reads it back |
| `main(argv)` | The command line |

## On disk
```
<output>/
├── run_config.json      ← full merged config, plus the commit of both repos
├── run_summary.json     ← status and time of each simulation
└── model_000000/
    ├── sim_00/
    │   ├── settings.json    ← params, simulator settings and seed
    │   └── simulation.npz   ← arrays and timing
    └── sim_01/
```
- **Crash safety:** each file is written under a temporary name and then renamed. `simulation.npz` is
  written last, so if it exists, the folder is complete. A crash or Slurm time limit only costs
  unfinished simulations; rerunning the same command picks up where it stopped.
- **Reuse:** a saved simulation is reused only if its `settings.json` matches exactly. Any difference
  stops the run with `SettingsMismatch`, so an old result is never silently kept or replaced.
- **`run.save = "none"`:** writes nothing, and returns simulations inside the summaries.

## Failures
A failed model is logged with its traceback and reported as `"failed"`, and the run carries on. NaN or
infinite results count as failures and aren't saved. A worker crashing outright, such as a segfault, isn't
handled yet; `df_index` no longer crashes on a NaN, which removes the one cause known so far.

## Seeds
A model's seed comes from `run.seed` and the model's index, using `numpy.random.SeedSequence`. It never
depends on the MPI rank or the order workers finish in, so a run gives identical numbers with any number of
workers. All simulators on one model share a seed, so differences between them don't come from different
starting particles.

## Parallel runs
SIMR follows `PLAN.md`: lower code takes a `map` function and knows nothing about pools. `main` creates one
pool, by `--mode`:

| `--mode` | Runs with |
|---|---|
| `solo` | Built-in `map` |
| `local` | `ProcessPoolExecutor`, with `--workers` |
| `mpi` | `MPIPoolExecutor` |

Each batch takes half of what's left, shared between the workers. Early batches keep messaging low; late
ones are small, so no worker holds a long batch while others wait.

### Using MPI
```bash
uv sync --group mpi
mpirun -n 8 uv run --group mpi python -m mpi4py.futures -m galactic_tidal_constraints.simr \
    --config examples/simr/config.json --output out --mode mpi
```
Rank 0 hands out work, so 8 ranks means 7 workers. Run it in the container (see
[running_spam.md](../running_spam.md)); `--oversubscribe` is needed if you have fewer CPUs than ranks.

### On a batch queue
`examples/slurm/run_node.slurm` already runs any command on whole nodes:
```bash
sbatch examples/slurm/run_node.slurm python -m mpi4py.futures -m galactic_tidal_constraints.simr \
    --config run.json --output ~/galStuff/data/run-1 --mode mpi
```
> **Not tested on Babbage yet.** This has only run under `mpirun` in Docker on one machine. The first
> supervised run also needs the package installed where the image's Python can import it, `v2` pulled as a
> `.sif`, and a check of how Babbage's shared disk handles many workers writing small files.

## Changes from the old code
| Old | New |
|---|---|
| Hash of rounded values as the file name, silently reused | Standard folders; settings compared exactly |
| A failed read silently reruns (`except: pass`) | Errors are logged |
| Hard-coded `/home/mbo2d/...` paths | `--output` |
| Parameters set in the parent never reach workers | Each task carries its full settings |

## Tests
`tests/test_simr.py` and `tests/test_config.py`. Most parallel code is tested in solo mode; one test checks
a local pool and one (`slow`) runs under `mpirun`, and both confirm results are identical to solo.
