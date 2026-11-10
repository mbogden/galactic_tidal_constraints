"""Simulator stage: runs one SPAM simulation, taking values and returning arrays.

This is the only module that imports `jspam`. It doesn't touch the disk, know about pools or MPI, or
keep state between calls.
"""

import jspam
import numpy as np


def simulate(params: np.ndarray | list[float], n1: int = 1000, n2: int = 1000) -> np.ndarray:
    """Run one SPAM simulation and return every particle at the end.

    Args:
        params: the 22 collision parameters, in SPAM's model-string order, shape (22,).
        n1, n2: number of particles in the primary's and secondary's disks.

    Returns:
        x, y, z, vx, vy, vz of each particle, shape (n1 + n2 + 1, 6). Rows 0 to n1 - 1 are the
        primary's disk, the next n2 the secondary's, and the last row is the secondary's center.
        The primary's center stays at the origin.
    """
    params = np.asarray(params, dtype=float)
    return jspam.run_collision(params, n1=n1, n2=n2)
