# gtc-base-image: CPU-only base image (Ubuntu 24.04, Python 3.12, gfortran, OpenMPI 4.1.6, git).
# v2 adds git, which uv needs to install jspam from GitHub.
# Build:  docker build -t gtc-base-image:v2 docker
# Check:  docker run --rm gtc-base-image:v2 bash /opt/gtc/checks/run_all.sh
FROM ubuntu:24.04

# System packages. Ubuntu 24.04's OpenMPI is 4.1.6, matching Babbage (openmpi4/4.1.6).
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates git \
      build-essential gfortran pkg-config \
      python3 python3-dev \
      openmpi-bin libopenmpi-dev \
 && rm -rf /var/lib/apt/lists/*

# uv, copied from its official image (pinned version).
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /usr/local/bin/

# Python environment at a fixed path, using apt's Python 3.12 (uv never downloads its own).
# PYTHONNOUSERSITE stops ~/.local packages leaking in when Apptainer mounts the home folder.
ENV UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONNOUSERSITE=1

# mpi4py is built from source so it links to the apt OpenMPI above, not a bundled MPI.
RUN uv venv --python /usr/bin/python3.12 /opt/venv \
 && uv pip install --no-cache \
      numpy==2.5.4 meson==1.12.1 ninja==1.13.2 \
 && uv pip install --no-cache --no-binary mpi4py \
      mpi4py==4.1.2

COPY checks/ /opt/gtc/checks/
