# The project's image

Everything in this project runs inside one image: tests, runs and notebooks, on your own machine and on
clusters (`PLAN.md`, Environment). The recipe is [docker/Dockerfile](../../docker/Dockerfile); the built
image is [`ogdenm12/gtc-base-image`](https://hub.docker.com/r/ogdenm12/gtc-base-image) on Docker Hub.
Use Docker on your own machine and Apptainer on clusters.

The image holds the tools and every library `uv.lock` pins (`jspam` included), but not this package:
it is read from your checkout, so a code change never needs a new image. The image's `python` runs
it with `PYTHONPATH=src`, with no install step:
```bash
docker run --rm -v "$PWD:/src:ro" ogdenm12/gtc-base-image:v3 bash -lc   "cp -r /src /work && cd /work && PYTHONPATH=src python -m pytest"
```
`uv sync` inside the image still works, and builds a separate environment from the checkout's own
`uv.lock`. Use it when your `uv.lock` has moved past the image's (below).

## What's in it
`v3`, the current tag. CPU only.

| | |
|---|---|
| OS | Ubuntu 24.04 |
| Python | 3.12, from Ubuntu (uv never downloads its own: `UV_PYTHON_DOWNLOADS=never`) |
| Compilers | `gfortran`, `gcc` (`build-essential`), `pkg-config`: enough to build `jspam` |
| MPI | OpenMPI 4.1.6, matching Babbage's `openmpi4/4.1.6`, so host and image MPI agree |
| Tools | `uv` 0.13.6, git (which uv needs to install `jspam` from GitHub), `hdf5-tools` (`h5ls`, `h5dump`) |
| `/opt/venv` | On `PATH`. Every dependency group in `uv.lock`: numpy, `jspam` (built from its pinned commit), mpi4py (built against the OpenMPI above), h5py, scipy, JupyterLab, matplotlib, pytest. Also meson and ninja, for building Fortran by hand with f2py |
| `/opt/gtc/` | What the image was built from: `packages.txt` (installed versions), `project/` (the `pyproject.toml` and `uv.lock` used) and `commit.txt` (the repo commit) |
| `/opt/gtc/checks/` | The image's own checks (below) |

## Run it with Docker
Commands are run from the repo's root. In Git Bash on Windows, put `MSYS_NO_PATHCONV=1` before
`docker run`, or Git Bash rewrites the container paths.

### Quick checks: a copy of the repo
The README's [Install](../../README.md#install) and [Tests](../../README.md#tests) commands mount the
repo read-only and copy it inside the container. Nothing is written to your checkout, and the
container's environment is gone when it exits. Best for running the tests as they are.

**Always mount the uv cache volume** (`-v gtc-uv-cache:/root/.cache/uv`, as those commands do). Each
`docker run --rm` starts with an empty cache, so without the volume every run re-downloads all wheels
and rebuilds `jspam` from its Fortran source: about 3 minutes of installs before anything runs, every
time. With the volume, the cache outlives the container and a repeat `uv sync` takes about a second
(measured on ProArt-26: "Prepared 9 packages in 3m 05s" cold, 634 ms warm). The volume is created on
first use; `docker volume rm gtc-uv-cache` resets it. uv's "Failed to hardlink files" warning is
expected (the cache volume and `/work` are different filesystems, so files are copied) and harmless.

### Working in the repo
To edit code or notebooks and keep the results, mount the repo itself and keep its environment in a
Docker volume:
```bash
docker run --rm -it -v "$PWD:/work" -w /work \
  -v gtc-venv:/venv -e UV_PROJECT_ENVIRONMENT=/venv -e VIRTUAL_ENV=/venv \
  ogdenm12/gtc-base-image:v3 bash
```
Then, inside:
```bash
uv sync --group mpi --group notebook     # the first time takes a few seconds; then it's instant
uv run pytest -m "not slow"
```
- `-v gtc-venv:/venv` with `UV_PROJECT_ENVIRONMENT=/venv`: the environment lives in a Docker volume
  named `gtc-venv`, not in the repo. Your checkout never gets a Linux `.venv`, and the next container
  reuses the volume (a second start with nothing to install takes under a second).
- `VIRTUAL_ENV=/venv`: replaces the image's `/opt/venv`, so `uv` doesn't warn that the two differ.
- Files the container writes in `/work` are your repo's files. Commit from your own machine as usual.
- A bind-mounted Windows folder is slower than a copy: the quick tests take about 30 s this way and
  15 s in a copy.
- To start the environment fresh: `docker volume rm gtc-venv`.
- A parallel run that compares models with large external arrays shares them through `/dev/shm`, which
  Docker limits to 64 MB. Add `--shm-size=4g` (or what the arrays need) to `docker run`; see Externals
  in [configuration.md](../configuration.md).

### Notebooks
Start JupyterLab in the same setup, publishing its port to your machine only:
```bash
docker run --rm -it -p 127.0.0.1:8888:8888 -v "$PWD:/work" -w /work \
  -v gtc-venv:/venv -e UV_PROJECT_ENVIRONMENT=/venv -e VIRTUAL_ENV=/venv \
  ogdenm12/gtc-base-image:v3 bash -lc \
  "uv sync --group notebook && uv run jupyter lab --ip 0.0.0.0 --no-browser --allow-root"
```
Open the `http://127.0.0.1:8888/lab?token=...` address it prints. `--ip 0.0.0.0` lets Docker reach
the server inside the container; `127.0.0.1:` keeps the port off your network, and the token stays
on. `--allow-root` is needed because Docker runs as root.

### Runs that save data
Mount a data folder too, and give `simr` an output inside it:
```bash
docker run --rm -v "$PWD:/work" -w /work -v ../data:/data \
  -v gtc-venv:/venv -e UV_PROJECT_ENVIRONMENT=/venv -e VIRTUAL_ENV=/venv \
  ogdenm12/gtc-base-image:v3 bash -lc \
  "uv sync && uv run python -m galactic_tidal_constraints.simr --models /data/my-exp/models.txt --output /data/my-exp/runs"
```
Use an absolute path for the data folder if your Docker doesn't accept relative ones.

### MPI as root
Docker runs as root, and OpenMPI refuses that unless told otherwise. Inside the container, before
`mpirun`:
```bash
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
```
Apptainer runs as your own user, so clusters don't need it.

## Run it with Apptainer (clusters)
Pull the image into a `.sif` file, named `<image>-<tag>.sif`:
```bash
apptainer pull gtc-base-image-v3.sif docker://ogdenm12/gtc-base-image:v3
```
Then build the repo's environment inside it and run through the job scripts: Babbage's steps,
modules and pitfalls are in [running_on_babbage.md](running_on_babbage.md).

## Check an image
```bash
docker run --rm ogdenm12/gtc-base-image:v3 bash /opt/gtc/checks/run_all.sh
```
It builds a small Fortran extension with f2py (standing in for `jspam`'s build) and runs an MPI
exchange on 2 ranks; it ends with `ALL PASS`. On Babbage, `docker/checks/babbage.slurm` runs the
same checks through Apptainer with the host's `mpirun`, on one node or two.

## Change the image
A tag names one exact image: **never overwrite a tag that's been pushed**, so every run can say which
image it used.

Since `v3` the image builds in `uv.lock`'s libraries, so a run's image records exactly what it used.
Two kinds of change get the next tag (`v4`, …): a change to the Dockerfile (a system package, uv,
Python, OpenMPI, the base OS), and a change to `uv.lock` (a new dependency, a new `jspam` commit).
This package's own code isn't in the image, so changing it needs no new tag. A later tag may build
the package in as well, for example for a paper's runs.

Between tags, a checkout whose `uv.lock` has moved past the image's still runs: `uv sync` builds an
environment from it inside the image (`examples/slurm/uv_sync.slurm` on a cluster), and
`run_node.slurm` warns in its log header when the two lockfiles differ.

Limits that come from the clusters (`PLAN.md`, Environment): Ubuntu 24.04, OpenMPI matching the
cluster's (4.1.6 on Babbage), and CUDA 12.x at most if a GPU image is ever made.

1. **Edit** `docker/Dockerfile`, and say what the new tag adds in its header comment (as `v2` adds git). A change to `uv.lock` alone needs no edit.
2. **Build** from the repo's root, with the changes committed (the image records the commit, and the
   build reads `pyproject.toml` and `uv.lock`; `.dockerignore` keeps everything else out):
   ```bash
   docker build -f docker/Dockerfile --build-arg GTC_COMMIT=$(git rev-parse HEAD) -t gtc-base-image:v4 .
   ```
3. **Check** it: the image checks above (`ALL PASS`), then the repo's tests in it (the command at
   the top of this page, with the new tag).
4. **Push** to Docker Hub (`docker login` once first):
   ```bash
   docker tag gtc-base-image:v4 ogdenm12/gtc-base-image:v4
   docker push ogdenm12/gtc-base-image:v4
   ```
5. **On each cluster:** pull the new `.sif` and run `babbage.slurm` against it.
6. **Point the repo at the new tag**: the README, the guides, the job scripts' default `SIF` and
   `examples/spam/check_original_jspam.sh`. To find them all:
   ```bash
   git grep -n -e "gtc-base-image:v" -e "gtc-base-image-v"
   ```
