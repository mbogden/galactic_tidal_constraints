"""Simulator stage: runs one SPAM simulation, taking values and returning arrays.

This is the only module that imports `jspam`. It doesn't touch the disk, know about pools or MPI, or
keep state between calls. See docs/design/sim.md.
"""

import logging
from dataclasses import dataclass, field

import jspam
import numpy as np

logger = logging.getLogger(__name__)

# SPAM's fixed step, and its time unit in Gyr (time_s in JSPAM's parameters_module).
STEP_SIZE = 0.001
GYR_PER_TIME_UNIT = 2.733342473337471e15 / (365.25 * 24 * 3600 * 1e9)

POTENTIALS = {"extended": jspam.EXTENDED, "point_mass": jspam.POINT_MASS}
KINDS = ("particles", "orbit")
ENDING_UNITS = ("fraction", "gyr", "time", "step")


@dataclass(frozen=True)
class CollisionParams:
    """The 22 values of a SPAM model string, named as in the Fortran."""

    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    mass1: float
    mass2: float
    rout1: float
    rout2: float
    phi1: float
    phi2: float
    theta1: float
    theta2: float
    epsilon1: float
    epsilon2: float
    rscale1: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rscale2: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_array(self) -> np.ndarray:
        """SPAM's order, shape (22,)."""
        values = [getattr(self, name) for name in _SCALAR_FIELDS]
        return np.array([*values, *self.rscale1, *self.rscale2], dtype=float)

    @classmethod
    def from_array(cls, values: np.ndarray | list[float]) -> "CollisionParams":
        values = np.asarray(values, dtype=float)
        if values.shape != (jspam.N_PARAMS,):
            raise ValueError(f"expected {jspam.N_PARAMS} values, got shape {values.shape}")
        scalars = dict(zip(_SCALAR_FIELDS, map(float, values[:16])))
        return cls(
            **scalars, rscale1=tuple(map(float, values[16:19])), rscale2=tuple(map(float, values[19:22]))
        )


_SCALAR_FIELDS = (
    "x", "y", "z", "vx", "vy", "vz", "mass1", "mass2", "rout1", "rout2",
    "phi1", "phi2", "theta1", "theta2", "epsilon1", "epsilon2",
)  # fmt: skip


@dataclass(frozen=True)
class Endings:
    """When to save a particle run. See `resolve_end_steps` for what each unit means."""

    unit: str = "fraction"
    values: tuple[float, ...] = (1.0,)

    def __post_init__(self) -> None:
        if self.unit not in ENDING_UNITS:
            raise ValueError(f"endings.unit must be one of {ENDING_UNITS}: {self.unit!r}")
        if len(self.values) == 0:
            raise ValueError("endings.values can't be empty")


@dataclass(frozen=True)
class SimulatorConfig:
    """Everything that sets one simulation's result, apart from the 22 values and the seed.

    Attributes:
        kind: "particles" saves every particle at each ending. "orbit" saves the secondary's center at
            every step, through the latest ending.
        n1, n2: particles in the primary's and secondary's disks.
        heat1, heat2: random motion added to each disk's particles.
        heat_mode: "wallin" (adds a drift) or "centered" (same spread, no drift).
        potential: "extended" (halo, disk and bulge; has dynamical friction) or "point_mass".
        lnl, r_scale, m_halo, m_bulge, h_bulge, h_disk: the mass model. Defaults are SPAM's own.
        normalization: "wallin" (fixed 7.13333, so these values change the total mass) or "total"
            (the model's actual total, so they change only how the mass is distributed).
        endings: when to save.
    """

    kind: str = "particles"
    n1: int = 1000
    n2: int = 1000
    heat1: float = 0.0
    heat2: float = 0.0
    heat_mode: str = "wallin"
    potential: str = "extended"
    lnl: float = 0.001
    r_scale: float = 10.0
    m_halo: float = 5.8
    m_bulge: float = 0.3333
    h_bulge: float = 0.2
    h_disk: float = 1.0
    normalization: str = "wallin"
    endings: Endings = field(default_factory=Endings)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}: {self.kind!r}")
        if self.potential not in POTENTIALS:
            raise ValueError(f"potential must be one of {list(POTENTIALS)}: {self.potential!r}")
        if self.heat_mode not in jspam.HEAT_MODES:
            raise ValueError(f"heat_mode must be one of {list(jspam.HEAT_MODES)}: {self.heat_mode!r}")
        self.mass_model().fortran_args()  # raises if a value or the normalization is invalid

    def mass_model(self) -> jspam.MassModel:
        return jspam.MassModel(
            lnl=self.lnl,
            r_scale=self.r_scale,
            m_halo=self.m_halo,
            m_bulge=self.m_bulge,
            h_bulge=self.h_bulge,
            h_disk=self.h_disk,
            normalization=self.normalization,
        )


@dataclass
class Simulation:
    """One simulation's result. Times are in SPAM units; t = 0 is when the 22 values apply.

    Attributes:
        kind: "particles" or "orbit".
        t_start: when the run started.
        n_steps: steps to the run's normal end, which is 1 to 2 steps past t = 0.
        start_fallback: True if SPAM rejected its start-time estimate and started at t = -5.
        steps: the steps saved, shape (n_ends,) for particles or (n_rows,) for an orbit.
        times: the time of each, same shape as `steps`.
        snapshots: particle runs only. x, y, z, vx, vy, vz at each ending, shape
            (n_ends, n1 + n2 + 1, 6). The last row of each ending is the secondary's center.
        orbit: orbit runs only. x, y, z, vx, vy, vz of the secondary's center, shape (n_rows, 6).
    """

    kind: str
    t_start: float
    n_steps: int
    start_fallback: bool
    steps: np.ndarray
    times: np.ndarray
    snapshots: np.ndarray | None = None
    orbit: np.ndarray | None = None


def resolve_end_steps(endings: Endings, timing: jspam.Timing) -> list[int]:
    """Turn endings in any unit into SPAM's integer steps.

    - "fraction": share of the normal run. 0 is the start, 1 is the normal end, 1.2 goes 20% further.
    - "gyr": Gyr relative to t = 0, the moment the 22 values describe. Negative is earlier.
    - "time": SPAM time units relative to t = 0.
    - "step": integrator steps from the start.

    Each model has its own start time, so a fraction means a different physical time for each model.
    Use "gyr" when physical time matters.

    Raises:
        ValueError: values aren't ascending, an ending is before the start, or two land on one step.
    """
    values = np.asarray(endings.values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"endings must be finite: {endings.values}")
    if endings.unit == "step" and (values != np.round(values)).any():
        raise ValueError(f"step endings must be whole numbers: {endings.values}")

    if endings.unit == "fraction":
        steps = values * timing.n_steps
    elif endings.unit == "gyr":
        steps = (values / GYR_PER_TIME_UNIT - timing.t_start) / STEP_SIZE
    elif endings.unit == "time":
        steps = (values - timing.t_start) / STEP_SIZE
    else:
        steps = values
    steps = np.round(steps).astype(int)

    if (values[1:] <= values[:-1]).any():
        raise ValueError(f"endings must be strictly ascending: {endings.values}")
    if steps[0] < 0:
        raise ValueError(
            f"ending {endings.values[0]} ({endings.unit}) is before the start, t = {timing.t_start}"
        )
    if (steps[1:] == steps[:-1]).any():
        raise ValueError(
            f"two endings land on the same step: {endings.values} ({endings.unit}) gives {steps}"
        )
    return [int(step) for step in steps]


def simulate(
    params: CollisionParams | np.ndarray, config: SimulatorConfig, seed: int | None = None
) -> Simulation:
    """Run one simulation.

    Args:
        params: the 22 values, as `CollisionParams` or an array of shape (22,).
        config: the simulation's settings.
        seed: sets where the disk particles start. Required for particle runs; orbit runs have no
            random numbers and ignore it.

    Returns:
        The result. The same inputs and seed give identical numbers on the same build and machine.
    """
    values = params.to_array() if isinstance(params, CollisionParams) else np.asarray(params, dtype=float)
    potential = POTENTIALS[config.potential]
    mass_model = config.mass_model()

    timing = jspam.get_timing(values, mass_model, potential)
    if timing.start_fallback:
        logger.warning("SPAM rejected its start-time estimate, so this run starts at t = %s", timing.t_start)
    end_steps = resolve_end_steps(config.endings, timing)
    logger.debug("t_start = %s, n_steps = %s, end_steps = %s", timing.t_start, timing.n_steps, end_steps)

    base = {"t_start": timing.t_start, "n_steps": timing.n_steps, "start_fallback": timing.start_fallback}
    if config.kind == "orbit":
        path = jspam.run_orbit(values, mass_model, potential, n_steps=end_steps[-1])
        return Simulation(
            kind="orbit", steps=np.arange(len(path)), times=path[:, 6], orbit=path[:, :6], **base
        )

    if seed is None:
        raise ValueError("particle runs need a seed")
    snapshots, steps, times = jspam.run_particles(
        values,
        seed=seed,
        n1=config.n1,
        n2=config.n2,
        mass_model=mass_model,
        potential=potential,
        heat1=config.heat1,
        heat2=config.heat2,
        heat_mode=config.heat_mode,
        end_steps=end_steps,
    )
    return Simulation(kind="particles", steps=steps, times=times, snapshots=snapshots, **base)


def to_gyr(times: np.ndarray | float) -> np.ndarray | float:
    """Convert SPAM time units to Gyr."""
    return times * GYR_PER_TIME_UNIT
