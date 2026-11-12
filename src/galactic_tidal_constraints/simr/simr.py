"""SIMR workflow: run the simulations a request asks for, for many models, and return the requested data.

A request is a plain dict, shaped like the config file to come (Step 11):

    request = {
        "run": {"return": ["sim.orbit", "sim.particles"]},
        "simulator": [{"lnl": 0.1}, {"lnl": 0.01}],   # one dict or a list; each becomes a SimulatorConfig
    }
    result = run(models, request)
    result.data["sim.orbit"][12, 1]   # model 12's orbit from simulator entry 1

Every model runs on every simulator entry, with that entry's seed (`SimulatorConfig.seed`, 0 by default).
Only what `return` names is computed: asking only for "sim.orbit" never runs particles. Simulated arrays
come back as float32 (`PLAN.md`, Precision).

Not yet here: saving to disk (Step 8b), failure handling (Step 9) and pools (Step 12). An error in
any simulation stops the run.
"""

from dataclasses import dataclass, fields

import numpy as np

from galactic_tidal_constraints.simr import simulation as sim
from galactic_tidal_constraints.simr.simulation import SimulatorConfig


def _particles(params, config):
    return sim.simulate_particles(params, config).astype(np.float32)


def _unperturbed(params, config):
    return sim.unperturbed_particles(params, config).astype(np.float32)


def _orbit(params, config):
    return sim.simulate_orbit(params, config).astype(np.float32)


def _timing(params, config):
    return sim.get_timing(params, config)


# The data the Simulator stage can hand back, by name. Each is indexed [model, simulator].
#   sim.particles:   every particle at the end of the run, (n1 + n2 + 1, 6), float32
#   sim.unperturbed: the same disks placed at today's centers with no steps, (n1 + n2 + 1, 6), float32
#   sim.orbit:       the secondary's center at every step, (n_steps + 1, 7), float32
#   sim.timing:      the dict from `sim.get_timing` (start time, steps, which pass)
SIM_DATA = {
    "sim.particles": _particles,
    "sim.unperturbed": _unperturbed,
    "sim.orbit": _orbit,
    "sim.timing": _timing,
}

REQUEST_KEYS = ("run", "simulator")
RUN_KEYS = ("return",)


@dataclass
class SimrResult:
    """What `run` hands back.

    Attributes:
        models: the models that ran, shape (N, 14): row m is model m in `data`.
        simulators: the simulator entries, with every default filled in (seed included): entry s is
            simulator s in `data`.
        data: one entry per name in the request's `return` list, and nothing else. Each is a numpy
            object array indexed by the levels its stage depends on: [model, simulator] for "sim.*".
            Each cell holds that run's array (or dict, for "sim.timing").
    """

    models: np.ndarray
    simulators: list[SimulatorConfig]
    data: dict[str, np.ndarray]


def run(models: np.ndarray, request: dict) -> SimrResult:
    """Run every model on every simulator entry in `request` and return the requested data.

    Args:
        models: the models, shape (N, 14) (one model of shape (14,) also works), in `sim.PARAM_NAMES` order.
        request: a dict with optional sections "run" and "simulator" (see the module docstring).
            run.return (default []): names from `SIM_DATA` to hand back.
            simulator (default one `SimulatorConfig()`): a dict or `SimulatorConfig`, or a list of them.
                Each entry's seed is used for every model, so all models start from the same disk
                pattern (see `SimulatorConfig.seed`).

    Returns:
        A `SimrResult`; see its docstring.
    """
    models = _check_models(models)
    wanted, configs = _read_request(request)

    data = {name: np.empty((len(models), len(configs)), dtype=object) for name in wanted}
    for m, params in enumerate(models):
        for s, config in enumerate(configs):
            for name in wanted:
                data[name][m, s] = SIM_DATA[name](params, config)
    return SimrResult(models=models, simulators=configs, data=data)


def _check_models(models) -> np.ndarray:
    models = np.atleast_2d(np.asarray(models, dtype=float))
    if models.ndim != 2 or models.shape[1] != len(sim.PARAM_NAMES):
        raise ValueError(f"models must have shape (N, {len(sim.PARAM_NAMES)}): got {models.shape}")
    if len(models) == 0:
        raise ValueError("models is empty")
    if not np.isfinite(models).all():
        raise ValueError("models contain NaN or inf")
    return models.copy()  # the result keeps its own copy, so later edits to the caller's array don't reach it


def _read_request(request: dict) -> tuple[list[str], list[SimulatorConfig]]:
    _reject_unknown("request", request, REQUEST_KEYS)
    run_section = request.get("run", {})
    _reject_unknown("run", run_section, RUN_KEYS)

    wanted = run_section.get("return", [])
    if isinstance(wanted, str):
        raise ValueError(f'run.return must be a list of names, e.g. ["{wanted}"]')
    unknown = [name for name in wanted if name not in SIM_DATA]
    if unknown:
        raise ValueError(f"unknown names in run.return: {unknown}. Known: {list(SIM_DATA)}")
    wanted = list(dict.fromkeys(wanted))  # drop repeats, keep order

    entries = request.get("simulator", [{}])
    if isinstance(entries, (dict, SimulatorConfig)):
        entries = [entries]
    if len(entries) == 0:
        raise ValueError("simulator is an empty list")
    configs = [_simulator_config(i, entry) for i, entry in enumerate(entries)]
    return wanted, configs


def _simulator_config(index: int, entry) -> SimulatorConfig:
    if isinstance(entry, SimulatorConfig):
        return entry
    if not isinstance(entry, dict):
        raise ValueError(f"simulator[{index}] must be a dict or SimulatorConfig: {entry!r}")
    _reject_unknown(f"simulator[{index}]", entry, [f.name for f in fields(SimulatorConfig)])
    try:
        return SimulatorConfig(**entry)
    except ValueError as error:
        raise ValueError(f"simulator[{index}]: {error}") from error


def _reject_unknown(where: str, section, known) -> None:
    if not isinstance(section, dict):
        raise ValueError(f"{where} must be a dict: {section!r}")
    unknown = [key for key in section if key not in known]
    if unknown:
        raise ValueError(f"unknown keys in {where}: {unknown}. Known: {list(known)}")
