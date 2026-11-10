"""Simulator stage: runs one SPAM simulation, taking values and returning arrays.

This is the only module that imports `jspam`. It doesn't touch the disk, know about pools or MPI, or
keep state between calls.

A model is a plain array of SPAM's 22 collision parameters, shape (22,), or (n_models, 22) for many.
How it's simulated is set by a `SimulatorConfig`. Runs always use SPAM's extended potential (halo, disk
and bulge).
"""

from dataclasses import dataclass

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


@dataclass(frozen=True)
class SimulatorConfig:
    """Everything that sets a simulation's result, apart from the 22 parameters and the seed.

    Attributes:
        n1, n2: particles in the primary's and secondary's disks.
        heat1, heat2: random motion added to each disk's particle velocities (0: none).
        heat_mode: "centered" (the default) adds a value in [-heat/2, heat/2) to each of vx, vy and vz.
            "wallin" (the original) adds a value in [0, heat): the same spread, but always positive, so a
            heated disk also drifts toward +x, +y, +z.
    """

    n1: int = 1000
    n2: int = 1000
    heat1: float = 0.0
    heat2: float = 0.0
    heat_mode: str = "centered"

    def __post_init__(self) -> None:
        """Check the values when the config is made, so mistakes show up before any run."""
        for name in ("n1", "n2"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a whole number of at least 1: {value!r}")
        for name in ("heat1", "heat2"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} can't be negative: {getattr(self, name)!r}")
        if self.heat_mode not in jspam.HEAT_MODES:
            raise ValueError(f"heat_mode must be one of {list(jspam.HEAT_MODES)}: {self.heat_mode!r}")


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
    params: np.ndarray | list[float], config: SimulatorConfig = SimulatorConfig(), seed: int | None = None
) -> np.ndarray:
    """Run one SPAM simulation and return every particle at the end.

    Args:
        params: the 22 collision parameters, in SPAM's order (`PARAM_NAMES`), shape (22,).
        config: how to simulate. The default is `SimulatorConfig()`.
        seed: sets where the disk particles start; the same seed gives the same particles. None
            gives different particles every run, as in Wallin's original.

    Returns:
        x, y, z, vx, vy, vz of each particle, shape (n1 + n2 + 1, 6). Rows 0 to n1 - 1 are the
        primary's disk, the next n2 the secondary's, and the last row is the secondary's center.
        The primary's center stays at the origin.
    """
    return jspam.run_collision(params, seed=seed, **_jspam_args(config))  # jspam checks there are 22 values


def initial_state(
    params: np.ndarray | list[float], config: SimulatorConfig = SimulatorConfig(), seed: int | None = None
) -> np.ndarray:
    """Every particle at the start of the run, before SPAM takes a step.

    Takes the same arguments as `simulate` and returns the same shape. With the same seed, it's exactly
    the state `simulate` starts from. Useful for checking how the disks are set up.
    """
    return jspam.initial_state(params, seed=seed, **_jspam_args(config))


def _jspam_args(config: SimulatorConfig) -> dict:
    """The config as jspam's keyword arguments."""
    return {
        "n1": config.n1,
        "n2": config.n2,
        "heat1": config.heat1,
        "heat2": config.heat2,
        "heat_mode": config.heat_mode,
    }
