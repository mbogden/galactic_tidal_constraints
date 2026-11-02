#!/bin/bash
# gtc-base-image checks on Babbage: Fortran, then MPI with 2 ranks via host mpirun.
# Host mpirun works on Babbage; srun can't start this image's MPI (tested 2026-10-28).
# The admin's example also uses host mpiexec.
# Usage: sbatch babbage.slurm               (SIF defaults to ~/galStuff/images/gtc-base-image-v1.sif)
#        SIF=/path/to/img.sif sbatch babbage.slurm
#        sbatch --nodes=2 --ntasks-per-node=1 babbage.slurm    (MPI across 2 nodes)
#SBATCH --job-name=gtc-base-image-checks
#SBATCH --partition=research-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --time=00:20:00
#SBATCH --output=/home/%u/galStuff/logs/%x-%j.log

SIF=${SIF:-$HOME/galStuff/images/gtc-base-image-v1.sif}
CHECKS=/opt/gtc/checks
status=0

# Batch jobs start with no modules loaded; OpenMPI is hidden until gnu12 is loaded.
module load apptainer gnu12/12.4.0 openmpi4/4.1.6

echo "=== host: $(hostname) | $(apptainer --version) | job $SLURM_JOB_ID | nodes: $SLURM_JOB_NODELIST"

echo; echo "=== Fortran (one process)"
apptainer exec "$SIF" bash $CHECKS/fortran_check.sh || status=1

echo; echo "=== MPI: host mpirun ($(mpirun --version | head -1)), $SLURM_NTASKS ranks"
mpirun -n "$SLURM_NTASKS" apptainer exec "$SIF" python $CHECKS/mpi_check.py || status=1

echo; [ $status -eq 0 ] && echo "ALL PASS" || echo "SOME CHECKS FAILED"
exit $status
