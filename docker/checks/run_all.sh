#!/bin/bash
# Run every single-node check inside the container: Fortran, then MPI with 2 ranks.
# Usage: bash /opt/gtc/checks/run_all.sh
# Multi-node MPI is launched from the host with mpirun (see the Babbage job script).
status=0
echo "=== container: $(grep PRETTY_NAME /etc/os-release | cut -d= -f2), host $(hostname)"

echo; echo "=== Fortran"
bash /opt/gtc/checks/fortran_check.sh || status=1

echo; echo "=== MPI, 2 ranks"
# Docker runs as root; OpenMPI refuses that unless told otherwise. No effect under Apptainer.
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
mpirun -n 2 --oversubscribe python /opt/gtc/checks/mpi_check.py || status=1

echo; [ $status -eq 0 ] && echo "ALL PASS" || echo "SOME CHECKS FAILED"
exit $status
