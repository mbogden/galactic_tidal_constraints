"""Simulator stage: runs one SPAM simulation, taking values and returning arrays.

This is the only module that imports `jspam`. It doesn't touch the disk, know about pools or MPI, or
keep state between calls.

A model is a plain array of SPAM's 22 collision parameters, shape (22,), or (n_models, 22) for many.
"""

import jspam
import numpy as np

# The 22 parameters in SPAM's order, named as in its Fortran parser (`parse_state_info_string`).
# Galaxy 1 is the primary, which sits at the origin; galaxy 2 is the secondary.
#   x, y, z, vx, vy, vz:    the secondary's position and velocity relative to the primary
#   mass1, mass2:           the two masses
#   rout1, rout2:           the two disk radii
#   phi1, phi2, theta1, theta2: the two disks' orientations, in degrees
#   epsilon1, epsilon2:     softening for gravity between the two centers
#   rscale1_*, rscale2_*:   three scale values per galaxy (usually all 0)
PARAM_NAMES = (
    "x", "y", "z", "vx", "vy", "vz",
    "mass1", "mass2", "rout1", "rout2",
    "phi1", "phi2", "theta1", "theta2",
    "epsilon1", "epsilon2",
    "rscale1_1", "rscale1_2", "rscale1_3", "rscale2_1", "rscale2_2", "rscale2_3",
)  # fmt: skip
PARAM_INDEX = {name: i for i, name in enumerate(PARAM_NAMES)}


def parse_model_string(model: str) -> np.ndarray:
    """Read a comma-separated SPAM model string into an array, shape (22,).

    Like SPAM, it reads the first 22 values and ignores the rest, so longer strings (such as the
    34-value lines in JSPAM's `archive/data.txt`) work too.
    """
    values = model.strip().split(",")
    if len(values) < len(PARAM_NAMES):
        raise ValueError(f"expected at least {len(PARAM_NAMES)} values, got {len(values)}")
    return np.array(values[: len(PARAM_NAMES)], dtype=float)


def simulate(
    params: np.ndarray | list[float], n1: int = 1000, n2: int = 1000, seed: int | None = None
) -> np.ndarray:
    """Run one SPAM simulation and return every particle at the end.

    Args:
        params: the 22 collision parameters, in SPAM's order (`PARAM_NAMES`), shape (22,).
        n1, n2: number of particles in the primary's and secondary's disks.
        seed: sets where the disk particles start; the same seed gives the same particles. None
            gives different particles every run, as in Wallin's original.

    Returns:
        x, y, z, vx, vy, vz of each particle, shape (n1 + n2 + 1, 6). Rows 0 to n1 - 1 are the
        primary's disk, the next n2 the secondary's, and the last row is the secondary's center.
        The primary's center stays at the origin.
    """
    return jspam.run_collision(params, n1=n1, n2=n2, seed=seed)  # checks there are 22 values
