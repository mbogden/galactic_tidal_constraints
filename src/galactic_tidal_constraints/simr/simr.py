"""SIMR workflow: run the simulations a request asks for, for many models, and return the requested data.

A request is a plain dict, shaped like the config file to come (Step 11):

    request = {
        "run": {"return": ["sim.orbit", "sim.particles"]},
        "simulator": [{"lnl": 0.1}, {"lnl": 0.01}],   # one dict or a list; each becomes a SimulatorConfig
    }
    result = run(models, request)
    result.data["sim.orbit"][12, 1]   # model 12's orbit from simulator entry 1

Every model runs on every simulator entry, with that entry's seed (`SimulatorConfig.seed`, 0 by default).
Only what `return` and `save` name is computed: asking only for "sim.orbit" never runs particles.
Simulated arrays come back as float32 (`PLAN.md`, Precision).

Saving ("run": {"save": [...]}, with `run(..., output=folder)`) writes one folder per model:

    <output>/
        index.json                    "models": id -> 14 parameters, every model ever saved here;
                                      "runs": one record per run that saved here (see `run`)
        model_000000/                 one folder per model id
            model.json                the model's record: its 14 parameters, and each saved simulation by
                                      name ("sim_00" for simulator entry 0) with its config, timing, files, code
            simulations/
                sim_00_particles.npy  one file per saved array: <simulation>_<item>.npy
                sim_00_orbit.npy

A run's models are matched exactly against the index: a model already there uses its folder, and a new
one gets the next id at the bottom, so folders stay the same however runs order or overlap their models.
One run at a time per output folder.

Each .npy is written to a temporary name and renamed into place, and only then does model.json (written
the same way) list it, so an array counts as saved only if model.json lists it. `load_model` and
`load_simulation` read them back.

With an output folder, saved data is used again. A simulation counts as already saved when model.json
lists its name ("sim_00" for entry 0) with exactly this entry's config. Then anything requested that it
already has is loaded instead of computed, and nothing it has is written again; arrays it's missing are
computed and added. A saved simulation with a different config is neither used nor changed (a warning is
logged). The code versions aren't compared: model.json records them, and higher-level code can check.

Not yet here: failure handling (Step 9) and pools (Step 12). An error in any simulation stops the run.
"""

import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from galactic_tidal_constraints.simr import simulation as sim
from galactic_tidal_constraints.simr.simulation import SimulatorConfig

logger = logging.getLogger(__name__)


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
RUN_KEYS = ("return", "save")


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
        ids: with an output folder, each row's model id there (its folder is `model_<id>`); None for a
            row the folder doesn't have (a run that only reads adds nothing). All None without a folder.
    """

    models: np.ndarray
    simulators: list[SimulatorConfig]
    data: dict[str, np.ndarray]
    ids: list[str | None]


def run(models: np.ndarray, request: dict, output: str | Path | None = None) -> SimrResult:
    """Run every model on every simulator entry in `request`; return and save the requested data.

    Args:
        models: the models, shape (N, 14) (one model of shape (14,) also works), in `sim.PARAM_NAMES` order.
        request: a dict with optional sections "run" and "simulator" (see the module docstring).
            run.return (default []): names from `SIM_DATA` to hand back.
            run.save (default []): names from `SIM_DATA` to write to `output`. Independent of `return`.
            simulator (default one `SimulatorConfig()`): a dict or `SimulatorConfig`, or a list of them.
                Each entry's seed is used for every model, so all models start from the same disk
                pattern (see `SimulatorConfig.seed`).
        output: the folder to save in, and to load already saved data from. Needed when run.save names
            something; optional otherwise. It isn't part of the request because it doesn't change any result.

    Returns:
        A `SimrResult`; see its docstring.

    A run that saves adds a record to `index.json`'s "runs" when it starts and completes it when it ends:
    "started", "finished" (None if the run never got there), "code", "run" (its return and save lists),
    "simulator" (the full configs), "image" and "fitness" (empty until those stages exist) and "model_ids"
    (row m's model id).
    """
    models = _check_models(models)
    returned, saved, configs = _read_request(request)
    if saved and output is None:
        raise ValueError("run.save names data to save, but no output folder was given")
    versions = code_versions() if saved else None
    index = _open_index(Path(output)) if output is not None else None
    ids = _model_ids(index, models, add=bool(saved)) if index is not None else [None] * len(models)
    if saved:
        run_record = {
            "started": _now(),
            "finished": None,
            "code": versions,
            "run": {"return": returned, "save": saved},
            "simulator": [_json_ready(asdict(config)) for config in configs],
            "image": [],
            "fitness": [],
            "model_ids": ids,
        }
        index["runs"].append(run_record)
        _write_index(Path(output), index)

    data = {name: np.empty((len(models), len(configs)), dtype=object) for name in returned}
    for m, params in enumerate(models):
        folder = Path(output) / f"model_{ids[m]}" if ids[m] is not None else None
        record = _open_record(folder, params) if folder is not None else None
        for s, config in enumerate(configs):
            name = f"sim_{s:02d}"
            stored, new = _saved_data(folder, record, name, config) if record is not None else ({}, True)
            usable = stored or {}  # None: saved with a different config, so nothing is used or added
            values = {item: usable[item]() for item in returned if item in usable}  # load what's saved
            to_save = [] if stored is None else [item for item in saved if item not in usable]
            if new and saved:
                to_save.append("sim.timing")  # every saved simulation records its timing in model.json
            for item in dict.fromkeys(returned + to_save):
                if item not in values:
                    values[item] = SIM_DATA[item](params, config)
            for item in returned:
                data[item][m, s] = values[item]
            if to_save:
                arrays = {item: values[item] for item in to_save if item != "sim.timing"}
                _save_simulation(folder, record, name, config, arrays, values.get("sim.timing"), versions)
    if saved:
        run_record["finished"] = _now()
        _write_index(Path(output), index)
    return SimrResult(models=models, simulators=configs, data=data, ids=ids)


def load_index(output: str | Path) -> dict:
    """An output folder's `index.json`: "models" (id -> the 14 named parameters) and "runs"."""
    with open(Path(output) / "index.json", encoding="utf-8") as f:
        return json.load(f)


def load_model(folder: str | Path) -> dict:
    """A model folder's record (`model.json`): "model" (the 14 named parameters) and "simulations"
    (each saved simulation by name: "config", "timing", "files" (item: path in the folder) and "code")."""
    with open(Path(folder) / "model.json", encoding="utf-8") as f:
        return json.load(f)


def load_simulation(folder: str | Path, name: str) -> dict[str, np.ndarray]:
    """One saved simulation's arrays, by name ("sim.particles", ...), as float32, just as `run` returns them.

    Only what `model.json` lists counts as saved; anything else is an error, even if a file exists.
    """
    folder = Path(folder)
    entry = load_model(folder)["simulations"].get(name)
    if entry is None:
        raise KeyError(f"{folder.name} has no saved simulation {name!r}")
    return {item: np.load(folder / file) for item, file in entry["files"].items()}


def array_file(name: str, item: str) -> str:
    """Where one saved array goes in its model folder: "sim.particles" of "sim_00" is
    "simulations/sim_00_particles.npy"."""
    return f"simulations/{name}_{item.split('.', 1)[1]}.npy"


def model_parameters(params: np.ndarray) -> dict[str, float]:
    """A model as {name: value}, the form `model.json` stores."""
    return {name: float(value) for name, value in zip(sim.PARAM_NAMES, params)}


def code_versions() -> dict:
    """The code a simulation was made with: each repo's git commit, and whether it had uncommitted changes.

    For `galactic_tidal_constraints`, only changes to what can change a result count (`src/`,
    `pyproject.toml`, `uv.lock`), not notebooks or docs. `jspam` installed from GitHub has its commit in its
    install record and no local changes; a local (editable) `jspam` is read with git like this repo.
    """
    jspam = sim.jspam_version()
    if jspam["commit"] is None and (jspam["url"] or "").startswith("file://"):
        jspam = {**jspam, **_git_state(Path(jspam["url"].removeprefix("file://")), [])}
    else:
        jspam = {**jspam, "uncommitted_changes": False}
    return {
        "galactic_tidal_constraints": _git_state(Path(__file__).parent, ["src", "pyproject.toml", "uv.lock"]),
        "jspam": jspam,
    }


def _git_state(folder: Path, paths: list[str]) -> dict:
    """The commit at `folder` and whether `paths` (all, if empty) have uncommitted changes. None for
    either when git can't tell (no git, or not a repository)."""

    def git(*args):
        result = subprocess.run(["git", "-C", str(folder), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    try:
        commit = git("rev-parse", "HEAD")
        top = git("rev-parse", "--show-toplevel")
        status = git("-C", top, "status", "--porcelain", "--", *paths) if top else None
    except OSError:
        commit, status = None, None
    return {"commit": commit, "uncommitted_changes": None if status is None else status != ""}


def _open_index(output: Path) -> dict:
    return load_index(output) if (output / "index.json").exists() else {"models": {}, "runs": []}


def _model_ids(index: dict, models: np.ndarray, add: bool) -> list[str | None]:
    """Each row's id in the index, matching all 14 values exactly. With `add`, a model the index doesn't
    have gets the next id at the bottom (and the same model twice gets one id); otherwise it gets None."""
    known = {tuple(params.values()): model_id for model_id, params in index["models"].items()}
    next_id = max((int(model_id) for model_id in index["models"]), default=-1) + 1
    ids = []
    for params in models:
        model = model_parameters(params)
        key = tuple(model.values())
        if key not in known and add:
            known[key] = f"{next_id:06d}"
            index["models"][known[key]] = model
            next_id += 1
        ids.append(known.get(key))
    return ids


def _write_index(output: Path, index: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    text = json.dumps(index, indent=2).encode("utf-8")
    _write_atomically(output / "index.json", lambda f: f.write(text))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _open_record(folder: Path, params: np.ndarray) -> dict:
    """The model folder's record, or a new one (written only when something is saved). A folder that holds
    a different model than the index says is an error: its saved simulations would be mixed with this one's,
    or used for it."""
    model = model_parameters(params)
    if not (folder / "model.json").exists():
        return {"model": model, "simulations": {}}
    record = load_model(folder)
    if record["model"] != model:
        raise ValueError(
            f"{folder} holds a different model; save this run in another output folder.\n"
            f"  saved: {record['model']}\n  this:  {model}"
        )
    return record


def _saved_data(folder, record, name, config) -> tuple[dict | None, bool]:
    """What this simulation already has saved, and whether model.json lists it at all.

    Returns ({item: function that loads it}, new). The items are its arrays and "sim.timing", but only if
    the saved config is exactly this one; a different config gives None: nothing is used or added.
    """
    entry = record["simulations"].get(name)
    if entry is None:
        return {}, True
    if entry["config"] != _json_ready(asdict(config)):
        logger.warning("%s/%s was saved with a different config; not used or changed", folder.name, name)
        return None, False
    stored = {item: (lambda file=file: np.load(folder / file)) for item, file in entry["files"].items()}
    stored["sim.timing"] = lambda: dict(entry["timing"])
    return stored, False


def _save_simulation(folder, record, name, config, arrays, timing, versions) -> None:
    """Write each array, then list them in model.json (all atomically)."""
    entry = record["simulations"].get(name) or {
        "config": _json_ready(asdict(config)),
        "timing": _json_ready(timing),
        "files": {},
        "code": versions,
    }
    for item, array in arrays.items():
        file = array_file(name, item)
        (folder / file).parent.mkdir(parents=True, exist_ok=True)
        _write_atomically(folder / file, lambda f, array=array: np.save(f, array))
        entry["files"][item] = file
    record["simulations"][name] = entry
    folder.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, indent=2).encode("utf-8")
    _write_atomically(folder / "model.json", lambda f: f.write(text))


def _write_atomically(path: Path, write) -> None:
    """Write to a temporary name in the same folder, then rename it into place, so `path` is either
    complete or absent, never half-written."""
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as f:
        write(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def _json_ready(values: dict) -> dict:
    """Numpy numbers as plain Python ones, so json can write them (and a config compares equal to its
    copy read back from model.json)."""
    return {key: value.item() if isinstance(value, np.generic) else value for key, value in values.items()}


def _check_models(models) -> np.ndarray:
    models = np.atleast_2d(np.asarray(models, dtype=float))
    if models.ndim != 2 or models.shape[1] != len(sim.PARAM_NAMES):
        raise ValueError(f"models must have shape (N, {len(sim.PARAM_NAMES)}): got {models.shape}")
    if len(models) == 0:
        raise ValueError("models is empty")
    if not np.isfinite(models).all():
        raise ValueError("models contain NaN or inf")
    return models.copy()  # the result keeps its own copy, so later edits to the caller's array don't reach it


def _read_request(request: dict) -> tuple[list[str], list[str], list[SimulatorConfig]]:
    _reject_unknown("request", request, REQUEST_KEYS)
    run_section = request.get("run", {})
    _reject_unknown("run", run_section, RUN_KEYS)
    returned, saved = (_names(run_section, key) for key in ("return", "save"))

    entries = request.get("simulator", [{}])
    if isinstance(entries, (dict, SimulatorConfig)):
        entries = [entries]
    if len(entries) == 0:
        raise ValueError("simulator is an empty list")
    configs = [_simulator_config(i, entry) for i, entry in enumerate(entries)]
    return returned, saved, configs


def _names(run_section: dict, key: str) -> list[str]:
    names = run_section.get(key, [])
    if isinstance(names, str):
        raise ValueError(f'run.{key} must be a list of names, e.g. ["{names}"]')
    unknown = [name for name in names if name not in SIM_DATA]
    if unknown:
        raise ValueError(f"unknown names in run.{key}: {unknown}. Known: {list(SIM_DATA)}")
    return list(dict.fromkeys(names))  # drop repeats, keep order


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
