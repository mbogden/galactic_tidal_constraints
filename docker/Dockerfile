# gtc-base-image: CPU-only image (Ubuntu 24.04, Python 3.12, gfortran, OpenMPI 4.1.6, git).
# v2 adds git, which uv needs to install jspam from GitHub.
# v3 builds in every library in uv.lock (all dependency groups, jspam included; not this package),
#    adds hdf5-tools (h5ls, h5dump), and records what it holds in /opt/gtc/.
# Build, from the repo's root (the context is the repo, for pyproject.toml and uv.lock):
#   docker build -f docker/Dockerfile --build-arg GTC_COMMIT=$(git rev-parse HEAD) -t gtc-base-image:v3 .
# Check:  docker run --rm gtc-base-image:v3 bash /opt/gtc/checks/run_all.sh
FROM ubuntu:24.04

# System packages. Ubuntu 24.04's OpenMPI is 4.1.6, matching Babbage (openmpi4/4.1.6).
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates git \
      build-essential gfortran pkg-config \
      python3 python3-dev \
      openmpi-bin libopenmpi-dev \
      hdf5-tools \
 && rm -rf /var/lib/apt/lists/*

# uv, copied from its official image (pinned version).
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /usr/local/bin/

# Python environment at a fixed path, using apt's Python 3.12 (uv never downloads its own).
# PYTHONNOUSERSITE stops ~/.local packages leaking in when Apptainer mounts the home folder.
ENV UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONNOUSERSITE=1

# The libraries uv.lock pins, in every dependency group, but not this package: it is read from a
# clone (PYTHONPATH=<clone>/src), so a code change needs no new image.
# - mpi4py is built from source (pyproject.toml's no-binary-package), so it links to the apt
#   OpenMPI above, not a bundled MPI. jspam is built from its pinned commit with gfortran.
# - UV_PROJECT_ENVIRONMENT is set for this command only. Set image-wide, a later `uv sync` in a clone
#   would try to write to /opt/venv, which is read-only under Apptainer.
# - meson and ninja aren't in uv.lock; f2py needs them to build Fortran by hand (the image's check).
COPY pyproject.toml uv.lock /opt/gtc/project/
RUN uv venv --python /usr/bin/python3.12 /opt/venv \
 && cd /opt/gtc/project \
 && UV_PROJECT_ENVIRONMENT=/opt/venv uv sync --locked --all-groups --no-install-project --no-cache \
 && uv pip install --no-cache meson==1.12.1 ninja==1.13.2 \
 && uv pip freeze > /opt/gtc/packages.txt

COPY docker/checks/ /opt/gtc/checks/

# What this image was built from: the repo commit (its pyproject.toml and uv.lock are in
# /opt/gtc/project/), beside the installed versions in /opt/gtc/packages.txt.
ARG GTC_COMMIT=unknown
RUN echo "$GTC_COMMIT" > /opt/gtc/commit.txt
LABEL org.opencontainers.image.revision=$GTC_COMMIT
