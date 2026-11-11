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

# The start of a run, with start="pass": this many rout1 before the chosen pass, at its speed there
# (SPAM's own lead, from getTStart).
LEAD_FACTOR = 10.0


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
            scaled by their masses in the parameters. Defaults are Wallin's, except lnl.
            lnl: strength of dynamical friction (Coulomb logarithm). 0.1 by default; Wallin's 0.001 makes
                friction almost nothing. It only acts while the centers are within r = 2.
            r_scale: the halo's core radius. It also sets the radius grid of the galaxy's tables, so it
                stretches every component, while SPAM still reads the tables at fixed distances.
            m_halo, m_bulge: halo and bulge masses. h_bulge: bulge scale length.
            h_disk: disk scale length, which also sets the disk's mass.
        normalization: "total" (the default) divides the galaxy by its actual total mass, so it's always 1
            and the six values only change how the mass is distributed. "wallin" (the original) divides by
            a fixed 7.13333, so changing a mass changes the galaxy's total (and the defaults give 0.9955).
        start: when the run starts (SPAM time, now = 0; one unit is about 87 Myr).
            "pass" (the default): `LEAD_FACTOR` x rout1 / speed before a close pass, chosen by `start_pass`
                from the passes back to t = -30 (see `find_passes`).
            "wallin" (the original): SPAM's own search, which uses the closest pass back to -30 and replaces
                a start earlier than -12 (or within a step of 0) with -5.
            A negative number: start exactly then.
        start_pass: with start="pass", which pass: 1 is the most recent, 2 the one before it, and so on.
        no_pass: with start="pass", what to do when that pass doesn't exist back to t = -30 (a first
            approach, a pass older than the window, or fewer passes than start_pass): start at this time
            (-5 by default, close to what SPAM ends up doing), or "error" to raise a ValueError.
            `get_timing` reports it as start="fallback", with the reason.
    """

    n1: int = 1000
    n2: int = 1000
    heat1: float = 0.0
    heat2: float = 0.0
    heat_mode: str = "centered"
    lnl: float = 0.1
    r_scale: float = 10.0
    m_halo: float = 5.8
    m_bulge: float = 0.3333
    h_bulge: float = 0.2
    h_disk: float = 1.0
    normalization: str = "total"
    start: str | float = "pass"
    start_pass: int = 1
    no_pass: str | float = -5.0

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
        if isinstance(self.start, str):
            if self.start not in ("pass", "wallin"):
                raise ValueError(f'start must be "pass", "wallin" or a negative number: {self.start!r}')
        elif not self.start < 0:
            raise ValueError(f"a start time must be before now (negative): {self.start!r}")
        if not isinstance(self.start_pass, (int, np.integer)) or self.start_pass < 1:
            raise ValueError(f"start_pass must be a whole number of at least 1: {self.start_pass!r}")
        if isinstance(self.no_pass, str):
            if self.no_pass != "error":
                raise ValueError(f'no_pass must be "error" or a negative number: {self.no_pass!r}')
        elif not self.no_pass < 0:
            raise ValueError(f"a no_pass start time must be before now (negative): {self.no_pass!r}")


def parse_model_string(model: str) -> np.ndarray:
    """Read a comma-separated SPAM model string into a model, shape (14,).

    Reads the first 14 values and ignores the rest, so SPAM's 22-value strings and the 34-value lines in
    JSPAM's `archive/data.txt` both work.
    """
    values = model.strip().split(",")
    if len(values) < len(PARAM_NAMES):
        raise ValueError(f"expected at least {len(PARAM_NAMES)} values, got {len(values)}")
    return np.array(values[: len(PARAM_NAMES)], dtype=float)


def simulate_particles(
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
    return jspam.run_collision(_spam_params(params), seed=seed, **_jspam_args(params, config))


def initial_state(
    params: np.ndarray | list[float], config: SimulatorConfig = SimulatorConfig(), seed: int | None = None
) -> np.ndarray:
    """Every particle at the start of the run, before SPAM takes a step.

    Takes the same arguments as `simulate_particles` and returns the same shape. With the same seed, it's
    exactly the state `simulate_particles` starts from. Useful for checking how the disks are set up.
    """
    return jspam.initial_state(_spam_params(params), seed=seed, **_jspam_args(params, config))


def get_timing(params: np.ndarray | list[float], config: SimulatorConfig = SimulatorConfig()) -> dict:
    """When a run starts and how many steps it takes, without running it.

    How the start is chosen is set by `config.start` (see `SimulatorConfig`). Only the start settings and
    the mass model matter here; particles and heat don't. Time is in SPAM units (about 87 Myr).

    Returns:
        "t_start": the start time a run uses.
        "n_steps": steps from t_start to the end of a run, 1 to 2 steps past now.
        "start": how it was chosen: "pass", "fallback" (no such pass; see `no_pass`), "wallin" or "fixed".
        "start_pass": the pass used, or None.
        "t_closest", "closest_distance", "closest_speed": the pass the start is based on (with "wallin",
            the closest pass SPAM's search found; NaN for a fixed or fallback start).
        "start_fallback": True when a fallback replaced the start: `no_pass`, or with "wallin", SPAM's -5.
        "fallback_reason": why there was no pass, or None.
        "t_start_found": with "wallin", SPAM's start before its fallback. Otherwise t_start.
    """
    t_start, chosen, reason = _choose_start(params, config)
    timing = jspam.timing(_spam_params(params), t_start=t_start, **_orbit_args(config))
    timing["fallback_reason"] = reason
    if t_start is None:
        timing.update(start="wallin", start_pass=None)
        return timing
    timing["t_start_found"] = t_start
    if chosen is None:
        timing.update(
            start="fixed" if reason is None else "fallback",
            start_fallback=reason is not None,
            start_pass=None,
            t_closest=np.nan,
            closest_distance=np.nan,
            closest_speed=np.nan,
        )
    else:
        timing.update(
            start="pass",
            start_pass=chosen["pass"],
            t_closest=chosen["t"],
            closest_distance=chosen["distance"],
            closest_speed=chosen["speed"],
        )
    return timing


def search_orbit(params: np.ndarray | list[float], config: SimulatorConfig = SimulatorConfig()) -> np.ndarray:
    """The secondary galaxy's center from now back to t = -30, one row per step.

    This is the path SPAM integrates backward to find the start; a run is placed by stepping back along it,
    so every run's start lies on it. Only the mass model in `config` matters.

    Returns:
        x, y, z, vx, vy, vz, t, from t = 0 (row 0, the model as given) back to -30, shape (about 30001, 7).
    """
    return jspam.search_orbit(_spam_params(params), **_orbit_args(config))


def find_passes(params: np.ndarray | list[float], config: SimulatorConfig = SimulatorConfig()) -> list[dict]:
    """Every close pass between the two centers from now back to t = -30, most recent first.

    A pass is a point where the distance between the centers stops falling and starts rising: a local
    minimum along `search_orbit`. The two ends of the window aren't passes, so a pair that is still
    approaching now has no pass at t = 0.

    Returns:
        One dict per pass: "pass" (1 is the most recent), "t", "distance" and "speed" there.
    """
    return _passes(search_orbit(params, config))


def _passes(path: np.ndarray) -> list[dict]:
    """The local minima of the distance between the centers along a search path, most recent first."""
    distance = np.linalg.norm(path[:, :3], axis=1)
    rows = np.flatnonzero((distance[1:-1] < distance[:-2]) & (distance[1:-1] <= distance[2:])) + 1
    return [
        {
            "pass": number,
            "t": float(path[row, 6]),
            "distance": float(distance[row]),
            "speed": float(np.linalg.norm(path[row, 3:6])),
        }
        for number, row in enumerate(rows, start=1)
    ]


def _choose_start(
    params: np.ndarray | list[float], config: SimulatorConfig
) -> tuple[float | None, dict | None, str | None]:
    """The start time to give jspam (None: SPAM's own search), the pass it's based on (if any), and why there
    was no pass (if the `no_pass` fallback was used)."""
    if config.start == "wallin":
        return None, None, None
    if not isinstance(config.start, str):
        return float(config.start), None, None
    path = search_orbit(params, config)
    passes = _passes(path)
    if len(passes) < config.start_pass:
        reason = _missing_pass_message(path, len(passes), config.start_pass)
        if config.no_pass == "error":
            raise ValueError(reason)
        return float(config.no_pass), None, reason
    chosen = passes[config.start_pass - 1]
    rout1 = np.asarray(params, dtype=float)[PARAM_INDEX["rout1"]]
    return float(chosen["t"] - LEAD_FACTOR * rout1 / chosen["speed"]), chosen, None


def _missing_pass_message(path: np.ndarray, found: int, wanted: int) -> str:
    """Why the search path has fewer passes than asked for. Rows run backward in time from now."""
    distance = np.linalg.norm(path[:, :3], axis=1)
    edge = f"t = {path[-1, 6]:.0f}"
    message = f"start_pass={wanted}, but the centers make {found} close pass{'' if found == 1 else 'es'} back to {edge}"
    if found == 0 and distance[1] > distance[0]:
        return message + ": they are approaching now, with no earlier pass (a first approach)"
    if distance[-1] < distance[-2]:
        return message + f": their distance is still shrinking at {edge}, so an older pass lies beyond it"
    return message


def simulate_orbit(params: np.ndarray | list[float], config: SimulatorConfig = SimulatorConfig()) -> np.ndarray:
    """The secondary galaxy's center at every step of a run, with no particles (about 0.05 s).

    The primary's center stays at the origin, so this is the whole orbit. The centers' motion doesn't
    depend on the particles or the seed, so the last row is exactly where `simulate_particles` leaves the
    secondary's center.

    Returns:
        x, y, z, vx, vy, vz, t at the start and after each step, shape (n_steps + 1, 7), with n_steps
        from `get_timing`.
    """
    t_start = _choose_start(params, config)[0]
    return jspam.run_orbit(_spam_params(params), t_start=t_start, **_orbit_args(config))


def _spam_params(params: np.ndarray | list[float]) -> np.ndarray:
    """A 14-value model as SPAM's 22 parameters: the model, then `UNUSED_PARAMS`."""
    params = np.asarray(params, dtype=float)
    if params.shape != (len(PARAM_NAMES),):
        raise ValueError(
            f"a model is {len(PARAM_NAMES)} values, shape ({len(PARAM_NAMES)},), got shape {params.shape}"
            " (for a SPAM model string, use parse_model_string)"
        )
    return np.concatenate([params, UNUSED_PARAMS])


def _jspam_args(params: np.ndarray | list[float], config: SimulatorConfig) -> dict:
    """The config as jspam's keyword arguments, with the start time it gives for these parameters."""
    return {
        "t_start": _choose_start(params, config)[0],
        "n1": config.n1,
        "n2": config.n2,
        "heat1": config.heat1,
        "heat2": config.heat2,
        "heat_mode": config.heat_mode,
        "mass_model": _mass_model(config),
        "normalization": config.normalization,
    }


def _orbit_args(config: SimulatorConfig) -> dict:
    """The parts of the config that move the galaxy centers, as jspam's keyword arguments."""
    return {"mass_model": _mass_model(config), "normalization": config.normalization}


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
