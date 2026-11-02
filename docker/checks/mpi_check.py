"""MPI check: ranks talk to each other, using the image's OpenMPI.

Run with 2+ ranks, e.g. `mpirun -n 2 python mpi_check.py`.
Rank 0 prints PASS/FAIL; exits non-zero on failure.
"""
import socket
import sys

from mpi4py import MPI

EXPECTED_MPI = "Open MPI v4.1.6"

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

total = comm.allreduce(rank + 1, op=MPI.SUM)
hosts = comm.gather(socket.gethostname(), root=0)

if rank == 0:
    # Open MPI's version string ends with a null character; strip it so logs stay plain text.
    library = MPI.Get_library_version().replace("\x00", "").splitlines()[0]
    print(f"MPI: {library}")
    for r, host in enumerate(hosts):
        print(f"rank {r}: {host}")
    expected = size * (size + 1) // 2
    print(f"allreduce sum = {total} (expected {expected}), ranks = {size}")
    ok = size >= 2 and total == expected and EXPECTED_MPI in library
    if EXPECTED_MPI not in library:
        print(f"mpi4py is not using {EXPECTED_MPI}")
    print(f"{'PASS' if ok else 'FAIL'} mpi")
    sys.exit(0 if ok else 1)
