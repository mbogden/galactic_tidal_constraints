"""Simulator stage: runs one SPAM simulation, taking values and returning arrays.

This is the only module that imports `jspam`. It doesn't touch the disk, know about pools or MPI, or
keep state between calls.

A model is a plain array of the 14 collision parameters that matter, shape (14,). SPAM takes 22; the
last 8 (`epsilon1/2` and six `rscale`) do nothing in the extended potential, so `sim` appends fixed
values for them (`UNUSED_PARAMS`) when it calls `jspam`. How it's simulated is set by a `SimulatorConfig`. Runs always use SPAM's extended potential (halo, disk
and bulge).
"""

from dataclasses import dataclass

import jspam
import numpy as np

# A model's 14 parameters: SPAM's first 14, in its order, named as in its Fortran parser
# (`parse_state_info_string`). Galaxy 1 is the primary, which sits at the origin; galaxy 2 is the secondary.
#   x, y, z, vx, vy, vz:    the secondary's position and velocity relative to the primary
#   mass1, mass2:           the two masses
#   rout1, rout2:           how far out each disk's particles are placed
#   phi1, phi2, theta1, theta2: the two disks' orientations, in degrees
PARAM_NAMES = (
    "x", "y", "z", "vx", "vy", "vz",
    "mass1", "mass2", "rout1", "rout2",
    "phi1", "phi2", "theta1", "theta2",
)  # fmt: skip
PARAM_INDEX = {name: i for i, name in enumerate(PARAM_NAMES)}

# SPAM's last 8 parameters, appended to every model before it goes to jspam. The extended potential never
# reads them, so any values give the same run (docs/spam_physics.md, "Parameters that do nothing").
#   epsilon1, epsilon2: softening, used only by the point-mass and MOND potentials
#   rscale1_1..3, rscale2_1..3: disk profile shapes, switched off by SPAM's fixed disk option
UNUSED_PARAMS = (0.3, 0.3, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0)


@dataclass(frozen=True)
class SimulatorConfig:
    """Everything that sets a simulation's result, apart from the 14 parameters and the seed.

    Attributes:
        n1, n2: particles in the primary's and secondary's disks.
        heat1, heat2: random motion added to each disk's particle velocities (0: none).
        heat_mode: "centered" (the default) adds a value in [-heat/2, heat/2) to each of vx, vy and vz.
            "wallin" (the original) adds a value in [0, heat): the same spread, but always positive, so a
            heated disk also drifts toward +x, +y, +z.
        lnl, r_scale, m_halo, m_bulge, h_bulge, h_disk: the unit galaxy both galaxies are built from,
            scaled by their masses in the parameters. Defaults are Wallin's.
            lnl: strength of dynamical friction (Coulomb logarithm).
            r_scale: the halo's core radius. It also sets the radius grid of the galaxy's tables, so it
                stretches every component, while SPAM still reads the tables at fixed distances.
            m_halo, m_bulge: halo and bulge masses. h_bulge: bulge scale length.
            h_disk: disk scale length, which also sets the disk's mass.
        normalization: "total" (the default) divides the galaxy by its actual total mass, so it's always 1
            and the six values only change how the mass is distributed. "wallin" (the original) divides by
            a fixed 7.13333, so changing a mass changes the galaxy's total (and the defaults give 0.9955).
    """

    n1: int = 1000
    n2: int = 1000
    heat1: float = 0.0
    heat2: float = 0.0
    heat_mode: str = "centered"
    lnl: float = 0.001
    r_scale: float = 10.0
    m_halo: float = 5.8
    m_bulge: float = 0.3333
    h_bulge: float = 0.2
    h_disk: float = 1.0
    normalization: str = "total"

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
        if self.lnl < 0:
            raise ValueError(f"lnl can't be negative: {self.lnl!r}")
        for name in ("r_scale", "m_halo", "m_bulge", "h_bulge", "h_disk"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive: {getattr(self, name)!r}")
        if self.normalization not in jspam.NORMALIZATIONS:
            raise ValueError(
                f"normalization must be one of {list(jspam.NORMALIZATIONS)}: {self.normalization!r}"
            )


def parse_model_string(model: str) -> np.ndarray:
    """Read a comma-separated SPAM model string into a model, shape (14,).

    Reads the first 14 values and ignores the rest, so SPAM's 22-value strings and the 34-value lines in
    JSPAM's `archive/data.txt` both work.
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
        params: the 14 collision parameters, in `PARAM_NAMES` order, shape (14,).
        config: how to simulate. The default is `SimulatorConfig()`.
        seed: sets where the disk particles start; the same seed gives the same particles. None
            gives different particles every run, as in Wallin's original.

    Returns:
        x, y, z, vx, vy, vz of each particle, shape (n1 + n2 + 1, 6). Rows 0 to n1 - 1 are the
        primary's disk, the next n2 the secondary's, and the last row is the secondary's center.
        The primary's center stays at the origin.
    """
    return jspam.run_collision(_spam_params(params), seed=seed, **_jspam_args(config))


def initial_state(
    params: np.ndarray | list[float], config: SimulatorConfig = SimulatorConfig(), seed: int | None = None
) -> np.ndarray:
    """Every particle at the start of the run, before SPAM takes a step.

    Takes the same arguments as `simulate` and returns the same shape. With the same seed, it's exactly
    the state `simulate` starts from. Useful for checking how the disks are set up.
    """
    return jspam.initial_state(_spam_params(params), seed=seed, **_jspam_args(config))


def _spam_params(params: np.ndarray | list[float]) -> np.ndarray:
    """A 14-value model as SPAM's 22 parameters: the model, then `UNUSED_PARAMS`."""
    params = np.asarray(params, dtype=float)
    if params.shape != (len(PARAM_NAMES),):
        raise ValueError(
            f"a model is {len(PARAM_NAMES)} values, shape ({len(PARAM_NAMES)},), got shape {params.shape}"
            " (for a SPAM model string, use parse_model_string)"
        )
    return np.concatenate([params, UNUSED_PARAMS])


def _jspam_args(config: SimulatorConfig) -> dict:
    """The config as jspam's keyword arguments."""
    return {
        "n1": config.n1,
        "n2": config.n2,
        "heat1": config.heat1,
        "heat2": config.heat2,
        "heat_mode": config.heat_mode,
        "mass_model": _mass_model(config),
        "normalization": config.normalization,
    }


def mass_profile(config: SimulatorConfig = SimulatorConfig()) -> dict[str, np.ndarray]:
    """The unit galaxy's tables for this config, as SPAM reads them.

    Returns:
        Arrays of shape (10000,): "radius" (SPAM distance units), the enclosed mass of "halo", "disk",
        "bulge" and "total" for a galaxy of mass 1 in the parameters, and "force", the acceleration
        a particle feels at that radius. Multiply masses and force by mass1 or mass2 for a real galaxy.
    """
    return jspam.mass_profile(_mass_model(config), config.normalization)


def _mass_model(config: SimulatorConfig) -> dict[str, float]:
    """The config's mass model, as jspam names it."""
    return {name: getattr(config, name) for name in ("lnl", "r_scale", "m_halo", "m_bulge", "h_bulge", "h_disk")}
