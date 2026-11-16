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

**Next:** fitness functions, the optimizer and the TNG application.

## SPAM
Collisions are simulated with SPAM, by John Wallin and collaborators, through a modified fork of the
original code:

- Original: [jfwallin/JSPAM](https://github.com/jfwallin/JSPAM)
- Fork used here: [mbogden/JSPAM](https://github.com/mbogden/JSPAM), installed as the `jspam`
  package at the commit pinned in `pyproject.toml`

## Install
You need Python 3.12 or newer, git, `gfortran` and a C compiler (Linux, macOS or WSL). With
[uv](https://docs.astral.sh/uv/), from the repo's root:
```bash
uv sync                       # the package, jspam (compiled on install) and pytest
uv sync --group notebook      # also JupyterLab and matplotlib, for the notebooks
uv sync --group mpi           # also mpi4py, for --mode mpi (needs an MPI library)
```
No compilers? The project's image, `ogdenm12/gtc-base-image:v2`, has them all
([docs/running_spam.md](docs/running_spam.md) shows how to start it).

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
[docs/design/simr.md](docs/design/simr.md).

The notebooks in [`examples/notebooks/`](examples/notebooks/) are the guided way in:
`simulator_tour.ipynb` for the Simulator stage, `simr_tour.ipynb` for the workflow and
`image_tour.ipynb` for the Image Creator (on its own, through `simr`, and saving and loading images).

## Docs
| Doc | What it covers |
|---|---|
| [design/simr.md](docs/design/simr.md) | How `simr` works: inputs, results, saving and reuse, the catalog, the command line |
| [design/image_creator.md](docs/design/image_creator.md) | How the Image Creator works: the config, the pixel frame, the image steps |
| [spam_physics.md](docs/spam_physics.md) | SPAM's physics and every setting the Simulator stage exposes, with code and open questions |
| [running_spam.md](docs/running_spam.md) | Compile and run SPAM itself, from Python or as the Fortran program |
| [running_on_babbage.md](docs/running_on_babbage.md) | Set up an experiment and run `simr` on Babbage |
| [architecture.md](docs/architecture.md) | The planned structure of the whole project (a plan; partly built) |

## Repo layout
```
src/galactic_tidal_constraints/
└── simr/            simulation.py (Simulator stage), simr.py (workflow), image_creator.py (Image Creator), __main__.py (command line)
docs/                the docs above
examples/
├── notebooks/       guided tours
├── slurm/           Slurm job scripts for Babbage
└── spam/            a check that the original SPAM runs correctly
docker/              the base image's Dockerfile and its checks
tests/               pytest tests
```

## Tests
```bash
uv run pytest                  # everything
uv run pytest -m "not slow"    # skip the long tests
```

## License
Licensed under the Academic Free License version 3.0. See [LICENSE](LICENSE). This project's use of
SPAM is modified from the original, as noted in the fork's README.
