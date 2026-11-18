# galactic_tidal_constraints

Simulates galaxy collisions with SPAM and compares them to observed and simulated galaxies, to study
what tidal features can tell us about how galaxies interact. Formerly `galaxyJSPAM`.

> **Status: being rewritten.** This `modern` branch is a from-scratch rewrite as an installable,
> reproducible package. The working research code is on the
> [`working_research`](https://github.com/mbogden/galactic_tidal_constraints/tree/working_research)
> branch. Decisions behind the rewrite are in [PLAN.md](PLAN.md).

**What works today:**
- **The Simulator stage** (`simr/simulation.py`): runs SPAM from Python, one simulation at a time,
  with SPAM's physics settings exposed (unit galaxy, dynamical friction, start time, seed).
- **The `simr` workflow** (`simr/simr.py`): runs many models through the simulator, hands back or
  saves the data you ask for, records every run in a catalog, reuses what's already saved, and
  carries on past failed simulations. From Python or the command line, in one process, a local pool
  or MPI.
- **The Image Creator stage** (`simr/image_creator.py`, a first draft): turns a particle set into an image,
  with the two galaxies' centers on chosen pixels, and maps orbits into the same pixels. `simr` returns,
  saves and reuses images (`image.model`, `image.unperturbed`).
- **Running on Babbage**, MTSU's Slurm cluster, with the project's image.

**Next:** the metrics stage, the optimizer and the TNG application.

## SPAM
Collisions are simulated with SPAM, by John Wallin and collaborators, through a modified fork of the
original code:

- Original: [jfwallin/JSPAM](https://github.com/jfwallin/JSPAM)
- Fork used here: [mbogden/JSPAM](https://github.com/mbogden/JSPAM), installed as the `jspam`
  package at the commit pinned in `pyproject.toml`

## Install
The project runs inside its image, [`ogdenm12/gtc-base-image:v2`](https://hub.docker.com/r/ogdenm12/gtc-base-image)
(recipe: [docker/Dockerfile](docker/Dockerfile)). It has Python, uv, gfortran and OpenMPI 4.1.6.
Use Docker on your own machine and Apptainer on clusters
([docs/guides/running_on_babbage.md](docs/guides/running_on_babbage.md)). From the repo's root:
```bash
docker run --rm -it -v "$PWD:/src:ro" ogdenm12/gtc-base-image:v2 bash -lc \
  "cp -r /src /work && cd /work && uv sync --locked --group mpi && bash"
```
The repo is mounted read-only and copied into the container, so the container's `.venv` stays out
of your checkout. In Git Bash on Windows, put `MSYS_NO_PATHCONV=1` before `docker run`, or Git Bash
rewrites the `/src` path. To work on the repo in place, run notebooks or save data, see
[docs/guides/docker_image.md](docs/guides/docker_image.md).

Without the image: Python 3.12 or newer, git, `gfortran` and a C compiler (Linux, macOS or WSL).
With [uv](https://docs.astral.sh/uv/), from the repo's root:
```bash
uv sync                       # the package, jspam (compiled on install) and pytest
uv sync --group notebook      # also JupyterLab and matplotlib, for the notebooks
uv sync --group mpi           # also mpi4py, for --mode mpi (needs an MPI library)
```

## Quick start
From Python:
```python
from galactic_tidal_constraints.simr import simr

models = simr.read_models("models.txt")                 # (N, 14): one model per row
result = simr.run(models, {"run": {"return": ["sim.orbit"]}})
result.data["sim.orbit"][0, 0]                          # model 0's orbit, simulator entry 0
```

From the command line, saving each simulation's particles to `runs/`:
```bash
echo '{"run": {"save": ["sim.particles"]}}' > config.json
uv run python -m galactic_tidal_constraints.simr --models models.txt --config config.json \
  --output runs --mode local
```
A models file has one model per line: SPAM's 14 parameters, comma-separated. Everything else (the
request, the output folder and catalog, failures, modes) is in
[docs/modules/simr.md](docs/modules/simr.md).

The notebooks in [`examples/notebooks/`](examples/notebooks/) are the guided way in:
`simulator_tour.ipynb` for the Simulator stage, `simr_tour.ipynb` for the workflow and
`image_tour.ipynb` for the Image Creator (on its own, through `simr`, and saving and loading images).

## Docs
| Doc | What it covers |
|---|---|
| [architecture.md](docs/architecture.md) | The planned structure of the whole project (a plan; partly built) |
| [configuration.md](docs/configuration.md) | How to write a config: its sections, how lists combine, the checks, what's recorded, and examples |
| [spam_physics.md](docs/spam_physics.md) | SPAM's physics and every setting the Simulator stage exposes, with code and open questions |
| [modules/simr.md](docs/modules/simr.md) | How `simr` works: inputs, results, saving and reuse, the catalog, the command line |
| [modules/simulation.md](docs/modules/simulation.md) | How the Simulator stage works: a model, `SimulatorConfig`, the functions and what they return, costs |
| [modules/image_creator.md](docs/modules/image_creator.md) | How the Image Creator works: the config, the pixel frame, the image steps |
| [guides/running_spam.md](docs/guides/running_spam.md) | Compile and run SPAM itself, from Python or as the Fortran program |
| [guides/running_on_babbage.md](docs/guides/running_on_babbage.md) | Set up an experiment and run `simr` on Babbage |
| [guides/docker_image.md](docs/guides/docker_image.md) | Run in the project's image (copies, working in place, notebooks, saving data, Apptainer), check it, and make a new tag |

## Repo layout
```
src/galactic_tidal_constraints/
└── simr/            simulation.py (Simulator stage), simr.py (workflow), image_creator.py (Image Creator), __main__.py (command line)
docs/                the docs above: overview and references at the top
├── modules/         one per code module
└── guides/          how-to guides
examples/
├── configs/         example configs and a models file
├── notebooks/       guided tours
├── slurm/           Slurm job scripts for Babbage
└── spam/            a check that the original SPAM runs correctly
docker/              the base image's Dockerfile and its checks
tests/               pytest tests
```

## Tests
In the image, from the repo's root:
```bash
docker run --rm -v "$PWD:/src:ro" ogdenm12/gtc-base-image:v2 bash -lc \
  "cp -r /src /work && cd /work && uv sync --locked --group mpi && uv run pytest"
```
Inside a running container (or without the image):
```bash
uv run pytest                  # everything
uv run pytest -m "not slow"    # skip the long tests
```
`test_original_jspam` needs Docker on the host, so it skips inside the container. To run that check,
run `bash examples/spam/check_original_jspam.sh` on the host (Git Bash on Windows).

## License
Licensed under the Academic Free License version 3.0. See [LICENSE](LICENSE). This project's use of
SPAM is modified from the original, as noted in the fork's README.
