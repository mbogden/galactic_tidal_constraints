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

With an output (`run(..., output=...)`), every run is recorded in a catalog, whether or not it saves data.
The output is a folder or a `.json` file, told apart by name:

    runs/        a folder: runs/catalog.json, plus one folder per model for the data run.save names
    gen_07.json  a catalog file on its own: no model data, so run.save with it is an error
    runs/x.txt   any other extension is an error

In a folder, saving ("run": {"save": [...]}) writes one folder per model:

    <output>/
        catalog.json                  "models": id -> 14 parameters, every model a run here has used;
                                      "runs": one record per run here (see `run`)
        model_000000/                 one folder per model id
            model.json                the model's record: its 14 parameters, and each saved simulation by
                                      name ("sim_00" for simulator entry 0) with its config, timing, files, code
            simulations/
                sim_00_particles.npy  one file per saved array: <simulation>_<item>.npy
                sim_00_orbit.npy

A run's models are matched exactly against the catalog: a model already there uses its folder, and a new
one gets the next id at the bottom, so folders stay the same however runs order or overlap their models.
One run at a time per output.

Each .npy is written to a temporary name and renamed into place, and only then does model.json (written
the same way) list it, so an array counts as saved only if model.json lists it. `load_model` and
`load_simulation` read them back.

With an output folder, saved data is used again. A simulation counts as already saved when model.json
lists its name ("sim_00" for entry 0) with exactly this entry's config. Then anything requested that it
already has is loaded instead of computed, and nothing it has is written again; arrays it's missing are
computed and added. A saved simulation with a different config is neither used nor changed (a warning is
logged). The code versions aren't compared: model.json records them, and higher-level code can check.

A simulation (one model on one simulator entry) fails as a whole when computing any requested item
raises an error, or when a simulated array holds NaN or inf. The run moves on: its cells hold None, nothing
is saved for it, and nothing is recorded in model.json, so a later run retries it. The error is logged with
its traceback. Errors while writing files stop the run. Each simulation's status, time and error come back
in the result, and a run that saves adds them to its record in catalog.json when it ends.

Not yet here: pools (Step 12).
"""

import json
import logging
import os
import subprocess
import time
import types
import typing
from concurrent.futures import BrokenExecutor
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

# While a run collects results, its statuses are written to the catalog at most this often (and at the end).
CATALOG_WRITE_SECONDS = 60.0

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
        ids: with an output, each row's model id in its catalog (in a folder, its data goes in
            `model_<id>`). All None without an output.
        status, seconds, errors: per stage ("sim" so far), each indexed like that stage's data ([m, s]):
            status: "ran", "reused" (saved, and nothing had to be computed), "failed", or "empty" (nothing
                was requested, so there was nothing to do).
            seconds: wall time, loading included.
            errors: "Type: message" for a failed simulation, else None. The traceback is in the log.
    """

    models: np.ndarray
    simulators: list[SimulatorConfig]
    data: dict[str, np.ndarray]
    ids: list[str | None]
    status: dict[str, np.ndarray]
    seconds: dict[str, np.ndarray]
    errors: dict[str, np.ndarray]


def run(models: np.ndarray, request: dict, output: str | Path | None = None, map=map) -> SimrResult:
    """Run every model on every simulator entry in `request`; return and save the requested data.

    Args:
        models: the models, shape (N, 14) (one model of shape (14,) also works), in `sim.PARAM_NAMES` order.
        request: a dict with optional sections "run" and "simulator" (see the module docstring).
            run.return (default []): names from `SIM_DATA` to hand back.
            run.save (default []): names from `SIM_DATA` to write to `output`. Independent of `return`.
            simulator (default one `SimulatorConfig()`): a dict or `SimulatorConfig`, or a list of them.
                Each entry's seed is used for every model, so all models start from the same disk
                pattern (see `SimulatorConfig.seed`).
        output: where to record the run: a folder (its `catalog.json`, plus model folders for the data
            run.save names, and saved data is loaded from there instead of computed) or a `.json` catalog
            file (no model data). A folder is needed when run.save names something; optional otherwise.
            It isn't part of the request because it doesn't change any result.
        map: how the models' work is spread out: built-in `map` (the default) runs one model after
            another in this process; a pool's `map` (e.g. `ProcessPoolExecutor().map`) runs them in
            parallel. Each call handles one model with all its simulator entries, and the same model twice
            is run once. Only this process touches the catalog, writing the statuses so far at most once
            a minute while results arrive. If a worker dies (a broken pool, e.g. a crash in SPAM's
            Fortran), the unfinished models are logged, the catalog keeps the statuses so far with
            "finished" None, and `SimrRunError` is raised; a rerun reuses what was saved.

    Returns:
        A `SimrResult`; see its docstring.

    A run with an output adds a record to the catalog's "runs" when it starts and completes it when it ends:
    "started", "finished" (None if the run never got there), "code", "run" (its return and save lists),
    "simulator" (the full configs), "image" and "fitness" (empty until those stages exist), "model_ids"
    (row m's model id), and at the end "status", "seconds" and "errors" (as in the result, as lists).
    """
    models = _check_models(models)
    returned, saved, configs = _read_request(request)
    catalog_file, root = output_paths(output) if output is not None else (None, None)
    if saved and root is None:
        where = "no output was given" if output is None else f"{output} is a catalog file, not a folder"
        raise ValueError(f"run.save names data to save, which needs an output folder, but {where}")
    catalog = _open_catalog(catalog_file) if catalog_file is not None else None
    ids = _model_ids(catalog, models) if catalog is not None else [None] * len(models)
    versions = None
    if catalog is not None:
        versions = code_versions()
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
        catalog["runs"].append(run_record)
        _write_catalog(catalog_file, catalog)

    shape = (len(models), len(configs))
    data = {name: np.empty(shape, dtype=object) for name in returned}
    status = {"sim": np.empty(shape, dtype=object)}
    seconds = {"sim": np.zeros(shape)}
    errors = {"sim": np.full(shape, None, dtype=object)}

    # One task per model; rows holding the same model (one id) share a task, so no two tasks write one folder.
    groups: dict = {}
    for m in range(len(models)):
        groups.setdefault(ids[m] if ids[m] is not None else m, []).append(m)
    rows = list(groups.values())
    tasks = [
        _ModelTask(row=r[0], model_id=ids[r[0]], params=models[r[0]], configs=configs, returned=returned,
                   saved=saved, folder=root / f"model_{ids[r[0]]}" if root is not None else None, versions=versions)
        for r in rows
    ]
    last_write = time.monotonic()
    done = 0
    try:
        for task_rows, outcome in zip(rows, map(_run_model, tasks)):
            for m in task_rows:
                for item in returned:
                    data[item][m, :] = outcome["data"][item]
                status["sim"][m, :] = outcome["status"]
                seconds["sim"][m, :] = outcome["seconds"]
                errors["sim"][m, :] = outcome["errors"]
            done += 1
            if catalog is not None and time.monotonic() - last_write >= CATALOG_WRITE_SECONDS:
                _record_status(catalog_file, catalog, run_record, status, seconds, errors)
                last_write = time.monotonic()
    except BrokenExecutor as error:  # a worker died: nothing more can come back from the pool
        unfinished = [ids[r[0]] if ids[r[0]] is not None else r[0] for r in rows[done:]]
        logger.error("a worker died (%s); %d of %d models unfinished: %s", error, len(unfinished), len(rows),
                     unfinished)
        if catalog is not None:
            _record_status(catalog_file, catalog, run_record, status, seconds, errors)
        raise SimrRunError(f"a worker died with {len(unfinished)} models unfinished; rerun to resume") from error

    counts = {word: int((status["sim"] == word).sum()) for word in ("ran", "reused", "failed", "empty")}
    logger.info("simr: %d simulations: %d ran, %d reused, %d failed, %d empty (nothing requested)",
                status["sim"].size, *counts.values())
    if catalog is not None:
        run_record["finished"] = _now()
        _record_status(catalog_file, catalog, run_record, status, seconds, errors)
    return SimrResult(models=models, simulators=configs, data=data, ids=ids,
                      status=status, seconds=seconds, errors=errors)


LOG_FORMAT = "%(asctime)s %(levelname)s [%(processName)s] %(name)s: %(message)s"


def configure_logging(level: str | int = "INFO") -> None:
    """Set up logging for an application: once in the main process, and once in each worker a pool starts
    (as its initializer). The process name in every line tells interleaved workers apart. simr itself never
    calls this."""
    logging.basicConfig(level=level, format=LOG_FORMAT)


def available_cpus() -> int:
    """The CPUs this process may use: under Slurm, the ones the job was given, not the whole machine."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        return os.cpu_count() or 1


class SimrRunError(RuntimeError):
    """The run couldn't finish: a worker died. What finished is saved and recorded; rerun to resume."""


def _record_status(catalog_file, catalog, run_record, status, seconds, errors) -> None:
    """Write the statuses so far into the run's record (cells not yet run are None) and the catalog."""
    run_record.update(
        status={"sim": status["sim"].tolist()},
        seconds={"sim": seconds["sim"].tolist()},
        errors={"sim": errors["sim"].tolist()},
    )
    _write_catalog(catalog_file, catalog)


@dataclass
class _ModelTask:
    """One model's work, as plain data, so it can be sent to any worker process."""

    row: int  # the first row of `models` holding this model (for log messages)
    model_id: str | None
    params: np.ndarray
    configs: list[SimulatorConfig]
    returned: list[str]
    saved: list[str]
    folder: Path | None  # the model's folder, when the output is a folder
    versions: dict | None


def _run_model(task: _ModelTask) -> dict:
    """Every simulator entry of one model: load what's saved, compute the rest, save what's asked for, and
    catch each simulation's failure. Touches only this model's folder, never the catalog.

    Returns {"data": {item: one value per entry}, "status", "seconds", "errors": one per entry}.
    """
    n = len(task.configs)
    outcome = {"data": {item: [None] * n for item in task.returned},
               "status": [None] * n, "seconds": [0.0] * n, "errors": [None] * n}
    folder = task.folder
    record = _open_record(folder, task.params) if folder is not None else None
    for s, config in enumerate(task.configs):
        name = f"sim_{s:02d}"
        start = time.perf_counter()
        stored, new = _saved_data(folder, record, name, config) if record is not None else ({}, True)
        to_save = [] if stored is None else [item for item in task.saved if item not in (stored or {})]
        if new and task.saved:
            to_save.append("sim.timing")  # every saved simulation records its timing in model.json
        try:
            values, computed = _simulate(task.params, config, task.returned, to_save, stored or {})
        except Exception as error:  # a failed simulation: recorded, and the run moves on
            logger.exception("model %d (id %s) on simulator %d failed", task.row, task.model_id, s)
            outcome["status"][s] = "failed"
            outcome["errors"][s] = f"{type(error).__name__}: {error}"
            outcome["seconds"][s] = time.perf_counter() - start
            continue
        for item in task.returned:
            outcome["data"][item][s] = values[item]
        if to_save:  # errors while writing aren't caught: they stop the run
            arrays = {item: values[item] for item in to_save if item != "sim.timing"}
            _save_simulation(folder, record, name, config, arrays, values.get("sim.timing"), task.versions)
        if not task.returned and not task.saved:
            outcome["status"][s] = "empty"  # the request asked for nothing, so there was nothing to do
        else:
            outcome["status"][s] = "reused" if stored and not computed else "ran"  # reused: nothing computed
        outcome["seconds"][s] = time.perf_counter() - start
    return outcome


def _simulate(params, config, returned, to_save, stored) -> tuple[dict, bool]:
    """One simulation's requested items: loaded when saved (`stored`), computed otherwise. Returns them and
    whether anything was computed. Raises if any simulated array holds NaN or inf."""
    values = {item: stored[item]() for item in returned if item in stored}
    computed = False
    for item in dict.fromkeys(returned + to_save):
        if item not in values:
            values[item] = SIM_DATA[item](params, config)
            computed = True
    for item, value in values.items():
        if isinstance(value, np.ndarray) and not np.isfinite(value).all():
            raise ValueError(f"NaN or inf in {item}")
    return values, computed


def read_models(path: str | Path) -> np.ndarray:
    """Read a models file: one model per line, the first 14 comma-separated values on the line.

    Longer lines are accepted (SPAM's 22-value strings, JSPAM's `archive/data.txt`), and one log message
    says how many lines had more than 14 values. Blank lines and lines starting with `#` are skipped.
    Fewer than 14 values, or one that isn't a number, is an error naming the line.

    Returns:
        The models, shape (N, 14), in `sim.PARAM_NAMES` order.
    """
    n = len(sim.PARAM_NAMES)
    models, longer = [], 0
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values = line.split(",")
            if len(values) < n:
                raise ValueError(f"{path}, line {number}: expected at least {n} comma-separated values, got {len(values)}")
            try:
                models.append([float(value) for value in values[:n]])
            except ValueError as error:
                raise ValueError(f"{path}, line {number}: the first {n} values must be numbers ({error})") from None
            longer += len(values) > n
    if not models:
        raise ValueError(f"{path} has no models")
    if longer:
        logger.info("%s: %d of %d lines had more than %d values; the first %d were used",
                    path, longer, len(models), n, n)
    return np.array(models)


def load_catalog(output: str | Path) -> dict:
    """An output's catalog (a folder's `catalog.json`, or a `.json` catalog file): "models" (id -> the 14
    named parameters) and "runs"."""
    with open(output_paths(output)[0], encoding="utf-8") as f:
        return json.load(f)


def output_paths(output: str | Path) -> tuple[Path, Path | None]:
    """Where an output's catalog goes, and its folder for model data (None for a catalog file).

    Told apart by name, so it works before anything exists: no extension is a folder, `.json` is a
    catalog file, and any other extension (a dot in a folder's name included) is an error.
    """
    output = Path(output)
    if output.suffix == ".json":
        return output, None
    if output.suffix == "":
        return output / "catalog.json", output
    raise ValueError(f"the output must be a folder (no extension) or a .json catalog file: {output}")


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


def _open_catalog(catalog_file: Path) -> dict:
    if not catalog_file.exists():
        return {"models": {}, "runs": []}
    with open(catalog_file, encoding="utf-8") as f:
        return json.load(f)


def _model_ids(catalog: dict, models: np.ndarray) -> list[str]:
    """Each row's id in the catalog, matching all 14 values exactly. A model the catalog doesn't have gets
    the next id at the bottom, and the same model twice gets one id."""
    known = {tuple(params.values()): model_id for model_id, params in catalog["models"].items()}
    next_id = max((int(model_id) for model_id in catalog["models"]), default=-1) + 1
    ids = []
    for params in models:
        model = model_parameters(params)
        key = tuple(model.values())
        if key not in known:
            known[key] = f"{next_id:06d}"
            catalog["models"][known[key]] = model
            next_id += 1
        ids.append(known[key])
    return ids


def _write_catalog(catalog_file: Path, catalog: dict) -> None:
    catalog_file.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(catalog, indent=2).encode("utf-8")
    _write_atomically(catalog_file, lambda f: f.write(text))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _open_record(folder: Path, params: np.ndarray) -> dict:
    """The model folder's record, or a new one (written only when something is saved). A folder that holds
    a different model than the catalog says is an error: its saved simulations would be mixed with this one's,
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
    hints = typing.get_type_hints(SimulatorConfig)
    for key, value in entry.items():
        if not _has_type(value, hints[key]):
            raise ValueError(f"simulator[{index}].{key} must be {_type_name(hints[key])}: {value!r}")
    try:
        return SimulatorConfig(**entry)
    except ValueError as error:
        raise ValueError(f"simulator[{index}]: {error}") from error


def _has_type(value, hint) -> bool:
    """Whether a config value fits its field's type hint. Integers fit a float field (JSON writes 1.0 as 1);
    True and False fit neither numbers nor strings."""
    if isinstance(hint, types.UnionType) or typing.get_origin(hint) is typing.Union:
        return any(_has_type(value, option) for option in typing.get_args(hint))
    if isinstance(value, bool):
        return hint is bool
    if hint is int:
        return isinstance(value, (int, np.integer))
    if hint is float:
        return isinstance(value, (int, float, np.integer, np.floating))
    return isinstance(value, hint)


def _type_name(hint) -> str:
    options = typing.get_args(hint) if typing.get_args(hint) else (hint,)
    words = {int: "a whole number", float: "a number", str: "text", bool: "true or false"}
    return " or ".join(words.get(option, option.__name__) for option in options)


def _reject_unknown(where: str, section, known) -> None:
    if not isinstance(section, dict):
        raise ValueError(f"{where} must be a dict: {section!r}")
    unknown = [key for key in section if key not in known]
    if unknown:
        raise ValueError(f"unknown keys in {where}: {unknown}. Known: {list(known)}")
