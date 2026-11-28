#!/bin/bash
# Build the repo's venv (.venv, with the mpi group) inside the Apptainer image, on a compute node.
# Only needed when the clone's uv.lock differs from the one the image was built from (image v3 on
# holds uv.lock's libraries; run_node.slurm's log header says when they differ), e.g. after a pull
# with a new jspam pin and before a new image tag. Code changes alone need only a pull.
# The venv's python points at the image's /usr/bin/python3.12, so it only works inside the image:
#   SIF=<this image> sbatch run_node.slurm <repo>/.venv/bin/python ...
#
# Usage: sbatch uv_sync.slurm
#        SIF=~/galStuff/images/<image>-<tag>.sif REPO=<clone> sbatch uv_sync.slurm
#SBATCH --job-name=gtc-uv-sync
#SBATCH --partition=interactive-cpu
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/home/%u/galStuff/logs/%x-%j.log

SIF=${SIF:-$HOME/galStuff/images/gtc-base-image-v3.sif}
REPO=${REPO:-$HOME/galStuff/galactic_tidal_constraints}

module load apptainer || exit 1
echo "=== $(hostname) · $(date '+%Y-%m-%d %H:%M:%S %Z') · apptainer $(apptainer --version | cut -d' ' -f3)"
echo "image: $SIF"
echo "repo:  $REPO ($(cat "$REPO/.git/HEAD"))"

# uv's cache goes on the node's local disk, not the shared /home: files unpacked there get the file
# server's clock, which can be minutes ahead of the node's, and meson then stops building jspam with
# "Clock skew detected" (Babbage, 2026-11-16). The job deletes its own cache folder at the end.
export UV_CACHE_DIR=$(mktemp -d "/tmp/uv-$SLURM_JOB_ID-XXXX") || exit 1
trap 'rm -rf "$UV_CACHE_DIR"' EXIT
echo "cache: $UV_CACHE_DIR (deleted at the end)"

# Copy instead of hardlinking from uv's cache (the cache and venv are on different filesystems).
# Unset the image's VIRTUAL_ENV (/opt/venv) so uv uses the project's .venv without a warning.
apptainer exec --env UV_LINK_MODE=copy --env UV_CACHE_DIR="$UV_CACHE_DIR" "$SIF" bash -c "
    unset VIRTUAL_ENV
    cd '$REPO' &&
    uv sync --group mpi --python /usr/bin/python3.12 &&
    .venv/bin/python -c 'import galactic_tidal_constraints, jspam, mpi4py, numpy; print(\"imports ok · numpy\", numpy.__version__, \"· mpi4py\", mpi4py.__version__)'
"
status=$?
echo "=== exit $status"
exit $status
