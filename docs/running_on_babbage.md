# Running on Babbage

How to set up an experiment and run `simr` on Babbage, MTSU's Slurm cluster, with this project's
image. Other Slurm clusters with Apptainer work the same way, with their own modules and partitions.

What `simr` does with its inputs, and what it writes, is in [design/simr.md](design/simr.md).

## How it fits together
- **The image** (`gtc-base-image:v2`, pulled as an Apptainer `.sif`) has the tools: Python 3.12,
  gfortran, OpenMPI 4.1.6 (the same version as Babbage's), `uv` and git.
- **The repo's `.venv`**, built inside the image, has this package, `jspam` (at the commit pinned in
  `pyproject.toml`) and `mpi4py`. The venv uses the image's Python, so it only works inside the image.
- **The job scripts** in [`examples/slurm/`](../examples/slurm/) load Babbage's modules, start the
  command inside the image with the host's `mpirun`, and begin each log with a header: node, image
  build, modules, branch and commit, and the command.

Later, the package may be built into the image itself; then the venv steps go away.

## One-time setup
Everything lives in one folder in your home directory, which every node shares:
```
~/galStuff/
├── galactic_tidal_constraints/   this repo
├── images/                       .sif files, named <image>-<tag>.sif
├── logs/                         every job's log, <job name>-<job id>.log
└── <experiment>/                 one folder per experiment: inputs and outputs
```

**1. Clone the repo and make the folders** (on the login node):
```bash
mkdir -p ~/galStuff/images ~/galStuff/logs && cd ~/galStuff
git clone https://github.com/mbogden/galactic_tidal_constraints.git
```

**2. Pull the image** as a small job, not on the login node:
```bash
cd ~/galStuff/images && sbatch -p interactive-cpu -c 4 -t 30 -J gtc-v2-pull -o pull-%j.log \
  --wrap 'module load apptainer; apptainer pull gtc-base-image-v2.sif docker://ogdenm12/gtc-base-image:v2'
```

**3. Build the venv**, from the repo (takes about 3 minutes):
```bash
cd ~/galStuff/galactic_tidal_constraints && sbatch examples/slurm/uv_sync.slurm
```
The log ends with `imports ok` and `=== exit 0`. Run it again after a pull that changes
dependencies (`pyproject.toml` or `uv.lock`, such as a new `jspam` commit), with a new image, or in a
new clone. Changes to this package's own code need only a `git pull`.

## Set up an experiment
Make a folder for it outside the repo, so its inputs and outputs stay out of git and don't depend
on the branch checked out:
```bash
mkdir -p ~/galStuff/my-exp && cd ~/galStuff/my-exp
```
It needs:
- **A models file:** one model per line, the 14 SPAM parameters comma-separated
  ([design/simr.md](design/simr.md#the-command-line)).
- **A config** (optional): only what differs from the defaults. For example, to save each
  simulation's particles with 2,000 particles per disk:
  ```json
  {"run": {"save": ["sim.particles"]}, "simulator": {"n1": 2000, "n2": 2000}}
  ```

## Launch
Run these from the experiment's folder, so relative paths (`models.txt`, `runs`) point there. Set
two variables first:
```bash
REPO=~/galStuff/galactic_tidal_constraints
export SIF=~/galStuff/images/gtc-base-image-v2.sif
```
`sbatch` passes your environment into the job, so `run_node.slurm` uses this image.

**On one whole node, with MPI** (32 ranks: rank 0 hands out models, 31 workers run them):
```bash
sbatch -J my-exp -t 02:00:00 $REPO/examples/slurm/run_node.slurm \
  $REPO/.venv/bin/python -m mpi4py.futures -m galactic_tidal_constraints.simr --mode mpi \
  --models models.txt --config config.json --output runs
```

**On one whole node, with a local pool** (32 workers):
```bash
OMPI_MCA_hwloc_base_binding_policy=none sbatch -J my-exp -t 02:00:00 --ntasks-per-node=1 \
  $REPO/examples/slurm/run_node.slurm \
  $REPO/.venv/bin/python -m galactic_tidal_constraints.simr --mode local \
  --models models.txt --config config.json --output runs
```
`--ntasks-per-node=1` starts one process instead of 32. `run_node.slurm` still starts it through
`mpirun`, which by default pins a single process to one core. `simr` would then see one CPU and
start one worker. The `OMPI_MCA_...=none` setting turns that pinning off; the log's
`local mode: 32 workers` line confirms it.

Both modes give byte-identical results (tested on Babbage, 2026-11-16). MPI can also span several
nodes (`--nodes=2` for 64 ranks), which hasn't been tested with `simr` yet. Which mode is faster,
and how best to write results to the shared `/home`, is still being measured.

**Many runs at once** (for example one per target): put one command per line in a file and submit it
as a job array, each line on its own whole node:
```bash
$REPO/examples/slurm/submit_array.sh commands.txt 5       # at most 5 at once
```
See [`example_commands.txt`](../examples/slurm/example_commands.txt). Each line runs under
`run_node.slurm`, so the same rules apply: a `--mode local` line needs
`--ntasks-per-node=1` (passed after the max-at-once number) and the binding setting in your
environment.

## Watch and check a job
```bash
squeue -u $USER                                        # waiting (PD) or running (R); empty when done
sacct -j <job id> -X -o JobID,JobName,State,ExitCode,Elapsed,NodeList
tail ~/galStuff/logs/<job name>-<job id>.log
```
A finished run's log ends with a summary and the job's exit code, for example:
```
simr: 64 simulations: 64 ran, 0 reused, 0 failed, 0 empty (nothing requested)
=== exit 0 after 21 s
```
The run's record (statuses, times, errors, code versions) is in `runs/catalog.json`
([design/simr.md](design/simr.md#the-output-catalog)). `simr`'s exit codes are 0 (ran), 1 (couldn't
run or finish) and 2 (every simulation failed).

## If a job stops early
Submit the same command again. Every simulation already saved is reused and only the rest run.
Each file is written in full before `model.json` lists it, so a killed job leaves no half-written
results. Tested on Babbage (2026-11-16): a run of 1,024 models cancelled after 95 s left 396 saved;
the rerun reported `628 ran, 396 reused, 0 failed`.

The killed run's record in `catalog.json` stays with `"finished": null`. Its statuses can be up to a
minute behind what was saved, because the catalog is written at most once a minute while a run is
going.

## Babbage specifics
- **CPU partitions** (nodes c1–c10, 32 CPUs each):

  | Partition | Time limit |
  |---|---|
  | `interactive-cpu` | 2 hours |
  | `research-cpu` | 3 days |

  `run_node.slurm` uses `research-cpu` with a 4-hour default; set `-t` to what the run needs.
- **Modules:** jobs start with none loaded. The scripts load `apptainer gnu12/12.4.0 openmpi4/4.1.6`
  (`openmpi4` stays hidden until `gnu12` is loaded).
- **MPI starts with the host's `mpirun`**, each rank inside the image. `srun` can't start the image's
  MPI.
- **Don't run heavy work on the login node:** builds, image pulls and runs go through `sbatch`.
- **Compute nodes have no git and run on UTC.** The log header reads the branch and commit from
  `.git` directly.
- **Node clocks can lag the shared `/home` by minutes.** Builds that check file times (meson, used to
  build `jspam`) then fail with `Clock skew detected`. `uv_sync.slurm` avoids it by keeping uv's
  cache on the node's own disk.
