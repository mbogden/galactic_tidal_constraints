#!/bin/bash
# Run one command on whole Babbage node(s): host mpirun starts one rank per core, each rank
# inside the Apptainer image. A command that doesn't use MPI runs once per rank, so use
# --ntasks-per-node=1 for those.
#
# Usage: sbatch run_node.slurm <command> [args...]
#        sbatch --nodes=2 run_node.slurm <command> [args...]           (64 ranks)
#        SIF=~/galStuff/images/<image>-<tag>.sif sbatch run_node.slurm ...
# Example: sbatch run_node.slurm python /opt/gtc/checks/mpi_check.py
#
# Why host mpirun and not srun: host mpirun works on Babbage and srun can't start this image's
# MPI (ENV-1, 2026-10-28). The image's OpenMPI (4.1.6) must match the host's.
#SBATCH --job-name=gtc-run
#SBATCH --partition=research-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/galStuff/logs/%x-%j.log

SIF=${SIF:-$HOME/galStuff/images/gtc-base-image-v1.sif}
REPO=${REPO:-$HOME/galStuff/galactic_tidal_constraints}
MODULES="apptainer gnu12/12.4.0 openmpi4/4.1.6"

if [ $# -eq 0 ]; then
    echo "Usage: sbatch run_node.slurm <command> [args...]" >&2
    exit 2
fi

# Batch jobs start with no modules loaded; openmpi4 is hidden until gnu12 is loaded.
module load $MODULES || exit 1

# Each rank runs in its own Apptainer user namespace, so OpenMPI's CMA shared-memory copy
# isn't available; turn it off instead of warning about it on every run.
export OMPI_MCA_btl_vader_single_copy_mechanism=none

# Babbage's compute nodes have no git, so read the branch and commit from .git directly.
git_info() {
    local head ref sha
    head=$(cat "$REPO/.git/HEAD" 2>/dev/null) || { echo "no repo at $REPO"; return; }
    if [[ $head == ref:* ]]; then
        ref=${head#ref: }
        sha=$(cat "$REPO/.git/$ref" 2>/dev/null || grep " $ref\$" "$REPO/.git/packed-refs" | cut -d' ' -f1)
        echo "${ref#refs/heads/} @ ${sha:0:7}"
    else
        echo "detached @ ${head:0:7}"
    fi
}

# The command as you'd type it: arguments with spaces or special characters in single quotes.
quote_args() {
    local a out=()
    for a in "$@"; do
        if [[ $a =~ ^[A-Za-z0-9_./=:,+@%-]+$ ]]; then out+=("$a"); else out+=("'${a//\'/\'\\\'\'}'"); fi
    done
    echo "${out[*]}"
}

# The .sif header's ID is unique to each build of the image.
sif_id=$(apptainer sif header "$SIF" 2>/dev/null | awk '/^ID:/ {sub(/^ID: */, ""); id=$0} /^Created At:/ {sub(/^Created At: */, ""); at=$0} END {print id " (built " at ")"}')

# Array tasks are named <array id>_<task>, as in squeue and the log file name.
job=${SLURM_ARRAY_TASK_ID:+${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}
echo "=== gtc job ${job:-$SLURM_JOB_ID} · $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "nodes:     $SLURM_JOB_NODELIST ($SLURM_JOB_NUM_NODES × $SLURM_NTASKS_PER_NODE ranks = $SLURM_NTASKS)"
echo "partition: $SLURM_JOB_PARTITION"
echo "image:     $SIF"
echo "image id:  $sif_id"
echo "apptainer: $(apptainer --version | cut -d' ' -f3)    modules: $MODULES"
echo "git:       $(git_info)"
# run_array.slurm sets GTC_COMMANDS_FILE and GTC_COMMAND (the line as written in the file).
[ -n "$GTC_COMMANDS_FILE" ] && echo "commands:  $GTC_COMMANDS_FILE, line $SLURM_ARRAY_TASK_ID"
echo "command:   ${GTC_COMMAND:-$(quote_args "$@")}"
echo "==="

start=$(date +%s)
mpirun apptainer exec "$SIF" "$@"
status=$?

echo "=== exit $status after $(( $(date +%s) - start )) s"
exit $status
