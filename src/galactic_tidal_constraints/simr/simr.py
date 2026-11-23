"""SIMR workflow: run the simulations and images a request asks for, for many models, and return the
requested data.

A request is a plain dict, shaped like the config file (Step 11):

    request = {
        "run": {"return": ["sim.orbit", "sim.particles", "img.model"]},
        "simulator": [{"lnl": 0.1}, {"lnl": 0.01}],   # one dict or a list; each becomes a SimulatorConfig
        "image": [{}, {"blur_sigma": 1.0}],           # one dict or a list; each becomes an ImageConfig
        "sim_metrics": [{"method": "closest_approach", "inputs": ["sim.search_orbit"], "select": "distance"}],
        "img_metrics": [{"method": "absolute_difference", "inputs": ["img.model", "img.unperturbed"]}],
    }
    result = run(models, request)
    result.data["sim.orbit"][12, 1]       # model 12's orbit from simulator entry 1
    result.data["img.model"][12, 1, 0]    # its image from image entry 0
    result.sim_metrics[12, 1, 0]          # its closest-approach distance (entry 0 of sim_metrics)
    result.img_metrics[12, 1, 0, 0]       # its image difference (entry 0 of img_metrics)

Every model runs on every simulator entry, with that entry's seed (`SimulatorConfig.seed`, 0 by default),
and every simulation is imaged with every image entry. Only what `return` and `save` name is computed:
asking only for "sim.orbit" never runs particles, and an image computes (or loads) only the particles it
needs. Simulated arrays and images come back as float32 (`PLAN.md`, Precision).

Metrics (the Metrics stage, `simr/metrics.py`) are computed whenever their sections list entries, and
always come back in the result; they aren't part of run.return and aren't saved yet. Every `sim_metrics`
entry runs on every simulation, and every `img_metrics` entry on every (simulation, image) pair, with no
per-entry filters. A metric's inputs are simr items; simr computes (or loads) what they need even when
nothing else asks for those items, so an orbit-only request with `sim_metrics` never builds an image.
Entries are checked against the metrics table (`metrics.METHODS`) before anything runs.

An input that isn't a simr item is an external: a key in the `externals` dict given to `run`, holding
an array from outside the run (a target orbit, image or particles), or a path to its .npy file:

    request = {"sim_metrics": [{"method": "pca_difference", "inputs": ["tng_particles", "sim.particles"]}]}
    result = run(models, request, externals={"tng_particles": array})

The request holds only the key; the array takes the input kind of the place it's used. Every external a
metric uses is loaded once and checked before anything is simulated (see `run`).

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
            model.json                the model's record: its 14 parameters, each saved simulation by
                                      name ("sim_00" for simulator entry 0) with its config, timing, files,
                                      code, and each saved image entry ("img_0000") with its config and
                                      files (simulation -> item -> path)
            simulations/
                sim_00_particles.npy  one file per saved array: <simulation>_<item>.npy
                sim_00_orbit.npy
            images/
                sim_00_img_0000_model.npy   one file per saved image: <simulation>_<image>_<item>.npy

A run's models are matched exactly against the catalog: a model already there uses its folder, and a new
one gets the next id at the bottom, so folders stay the same however runs order or overlap their models.
One run at a time per output.

Each .npy is written to a temporary name and renamed into place, and only then does model.json (written
the same way) list it, so an array counts as saved only if model.json lists it. `load_model`,
`load_simulation` and `load_images` read them back.

With an output folder, saved data is used again. A simulation counts as already saved when model.json
lists its name ("sim_00" for entry 0) with exactly this entry's config. Then anything requested that it
already has is loaded instead of computed, and nothing it has is written again; arrays it's missing are
computed and added. A saved simulation with a different config is neither used nor changed (a warning is
logged). The code versions aren't compared: model.json records them, and higher-level code can check.
A saved image is used again when both its image entry's config and its simulation's config match; then
no simulation runs just to make it again.

A simulation (one model on one simulator entry) fails as a whole when computing any requested item
raises an error, or when a simulated array holds NaN or inf. The run moves on: its cells hold None, nothing
is saved for it, and nothing is recorded in model.json, so a later run retries it. The error is logged with
its traceback. Errors while writing files stop the run. An image (one simulation with one image entry)
fails on its own, the same way, and leaves its simulation alone; a failed simulation's images are
"skipped". Each simulation's and image's status, time and error come back in the result, and a run with an
output adds them to its record in catalog.json.
"""

import builtins
import hashlib
import json
import logging
import os
import shutil
import socket
import subprocess
import tempfile
import time
import types
import typing
from concurrent.futures import BrokenExecutor
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from galactic_tidal_constraints.simr import derived
from galactic_tidal_constraints.simr import image_creator as img
from galactic_tidal_constraints.simr import metrics
from galactic_tidal_constraints.simr import simulation as sim
from galactic_tidal_constraints.simr.image_creator import ImageConfig
from galactic_tidal_constraints.simr.simulation import SimulatorConfig

logger = logging.getLogger(__name__)


def _particles(params, config):
    return sim.simulate_particles(params, config).astype(np.float32)


def _unperturbed(params, config):
    return sim.unperturbed_particles(params, config).astype(np.float32)


def _orbit(params, config):
    return sim.simulate_orbit(params, config).astype(np.float32)


def _search_orbit(params, config):
    return sim.search_orbit(params, config).astype(np.float32)


def _timing(params, config):
    return sim.get_timing(params, config)


# The data the Simulator stage can hand back, by name. Each is indexed [model, simulator].
#   sim.particles:    every particle at the end of the run, (n1 + n2 + 1, 6), float32
#   sim.unperturbed:  the same disks placed at today's centers with no steps, (n1 + n2 + 1, 6), float32
#   sim.orbit:        the secondary's center at every step, (n_steps + 1, 7), float32
#   sim.search_orbit: the secondary's center from now back to t = -30, the path every run's start lies
#                     on, (about 30001, 7), float32
#   sim.timing:       the dict from `sim.get_timing` (start time, steps, which pass)
SIM_DATA = {
    "sim.particles": _particles,
    "sim.unperturbed": _unperturbed,
    "sim.orbit": _orbit,
    "sim.search_orbit": _search_orbit,
    "sim.timing": _timing,
}


def _model_image(values, params, sim_config, image_config):
    return img.create_image(values["sim.particles"], params, sim_config.n1, sim_config.n2, image_config)


def _unperturbed_image(values, params, sim_config, image_config):
    return img.create_image(values["sim.unperturbed"], params, sim_config.n1, sim_config.n2, image_config)


# The data the Image Creator stage can hand back, by name: (the `sim` items it's made from, how). Each is
# indexed [model, simulator, image]: float32 (height, width), values in [0, 1]. Asking for one makes simr
# compute (or load) the `sim` items it needs, whether or not those are returned or saved.
#   img.model:       the particles at the end of the run
#   img.unperturbed: the same disks placed at today's centers with no steps, in the same frame
IMAGE_DATA = {
    "img.model": (("sim.particles",), _model_image),
    "img.unperturbed": (("sim.unperturbed",), _unperturbed_image),
}

# While a run collects results, its statuses are written to the catalog at most this often (and at the end).
CATALOG_WRITE_SECONDS = 60.0

REQUEST_KEYS = ("run", "simulator", "image", "derived", "sim_metrics", "img_metrics")
DERIVED_ENTRY_KEYS = ("name", "method", "inputs", "params")
# How often a derived entry is computed, from the inputs it's made from: once for the run (externals
# only), per simulation (any sim.* item) or per image (any img.* item). Later ones include earlier ones.
DERIVED_LEVELS = ("run", "sim", "img")
# What a stage item's name starts with. A metric input without one is a key in the run's externals, so
# an externals key can't start with one.
STAGE_PREFIXES = ("sim.", "img.", "drv.")
RUN_KEYS = ("return", "save")
METRIC_ENTRY_KEYS = ("method", "inputs", "select", "params")

# The input kind (metrics.INPUT_KINDS) each simulator item satisfies as a metric input; img.* items are
# all "image", and sim.timing (a dict) is no metric input at all.
SIM_INPUT_KINDS = {
    "sim.orbit": "orbit",
    "sim.search_orbit": "orbit",
    "sim.particles": "particles",
    "sim.unperturbed": "particles",
}


@dataclass
class SimrResult:
    """What `run` hands back.

    Attributes:
        models: the models that ran, shape (N, 14): row m is model m in `data`.
        simulators: the simulator entries, with every default filled in (seed included): entry s is
            simulator s in `data`.
        images: the image entries (`ImageConfig`s), likewise: entry i is image i in `data`.
        data: one entry per name in the request's `return` list, and nothing else. Each is a numpy
            object array indexed by the levels its stage depends on: [model, simulator] for "sim.*",
            [model, simulator, image] for "img.*". Each cell holds that run's array (or dict, for
            "sim.timing"), or None if it failed or was skipped.
        sim_metrics: each `sim_metrics` entry's value for each simulation, an object array indexed
            [model, simulator, entry]: a float (a scalar method, or `select`), a dict (a descriptive
            method), or None (failed or skipped). Metrics always come back when the request lists them;
            they aren't part of run.return. Shape (N, S, 0) with no entries.
        img_metrics: likewise for each `img_metrics` entry per image, [model, simulator, image, entry].
        sim_metric_configs, img_metric_configs: the filled-in entries, each `{"method", "inputs",
            "select" (None when absent), "params"}` with every setting's default filled in, so
            `sim_metrics[..., k]` is `sim_metric_configs[k]` (and the same at the image level).
        derived_configs: the filled-in `derived` entries, in order: each `{"name", "method", "inputs",
            "params" (every default filled in), "level"}`, where "level" is how often it's computed
            ("run": once, from externals only). One computed for the run also records the "shape" and
            "hash" (`array_hash`) of its array. Metrics name an entry as "drv.<name>".
        externals: a record of each external the metrics used, by key: `{"source": the path it was
            loaded from, or "memory"; "shape"; "hash": `array_hash` of its data as the metrics saw it}`.
            Empty when no metric uses one. The same dict goes in the catalog's run record.
        ids: with an output, each row's model id in its catalog (in a folder, its data goes in
            `model_<id>`). All None without an output.
        status, seconds, errors: per stage ("sim", "img", "sim_metrics", "img_metrics"), each indexed
            like that stage's data or values:
            status: "ran", "reused" (saved, and nothing had to be computed), "failed",
                "skipped" (the simulation, or for an image metric the image, it needs failed), or
                "empty" (nothing was requested from the stage, so there was nothing to do; metric
                cells are never "empty": with no entries there are no cells).
                A metric cell is "failed" when its method raised, returned NaN or inf, or returned
                something that doesn't match its declaration; its simulation and image are unaffected.
            seconds: wall time, loading included. A simulation's time doesn't include its images or
                metrics; each metric cell is timed on its own.
            errors: "Type: message" for a failure, else None. The traceback is in the log.
    """

    models: np.ndarray
    simulators: list[SimulatorConfig]
    images: list[ImageConfig]
    data: dict[str, np.ndarray]
    sim_metrics: np.ndarray
    img_metrics: np.ndarray
    sim_metric_configs: list[dict]
    img_metric_configs: list[dict]
    derived_configs: list[dict]
    externals: dict[str, dict]
    ids: list[str | None]
    status: dict[str, np.ndarray]
    seconds: dict[str, np.ndarray]
    errors: dict[str, np.ndarray]


def run(models: np.ndarray, request: dict, output: str | Path | None = None, map=map,
        externals: dict | None = None) -> SimrResult:
    """Run every model on every simulator entry in `request`; return and save the requested data.

    Args:
        models: the models, shape (N, 14) (one model of shape (14,) also works), in `sim.PARAM_NAMES` order.
        request: a dict with optional sections "run", "simulator" and "image" (see the module docstring).
            run.return (default []): names from `SIM_DATA` and `IMAGE_DATA` to hand back.
            run.save (default []): names from `SIM_DATA` and `IMAGE_DATA` to write to `output`.
                Independent of `return`.
            simulator (default one `SimulatorConfig()`): a dict or `SimulatorConfig`, or a list of them.
                Each entry's seed is used for every model, so all models start from the same disk
                pattern (see `SimulatorConfig.seed`).
            image (default one `ImageConfig()`): a dict or `ImageConfig`, or a list of them. Every
                simulation gets every image entry. Centers can be given as lists ([column, row]).
            sim_metrics, img_metrics (default none): metric entries, each a dict `{"method", "inputs",
                "select" (optional), "params" (optional)}` (see the module docstring and
                `metrics.METHODS`). One entry or a list; an empty list is allowed and means none.
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
        externals: arrays from outside the run, by key, for metric inputs that aren't simr items (e.g.
            `{"tng_orbit": orbit, "tng_image": "/data/targets/tng_123/image.npy"}`). Each value is a
            numeric numpy array, an absolute path to a .npy file, or `{"path": ..., "hash": ...}` to
            also name the hash the file's data must have (`array_hash`; a mismatch is a ValueError).
            Not part of the request, like the models. Each used external's source, shape and hash are
            logged and recorded (`SimrResult.externals`, and the catalog's run record). Before anything is simulated, each external a metric uses is loaded (once), turned
            into float32 and checked: no NaN or inf, the shape of the input kind it's used as
            (`metrics.INPUT_SHAPES`; one external has one kind), and for an image the height and width
            of every image entry. A problem with any of them, or a metric input naming a key that isn't
            here, is a ValueError. A key can't start with "sim." or "img."; keys no metric uses are
            ignored, with one log message. Metrics get each external read-only.
            With a pool, the workers share one copy of each external in the node's memory instead of
            getting a copy each (see `_share_externals`), so a pool that uses in-memory or
            non-float32 externals must have all its workers on the node that started the run; under
            an MPI launcher spanning more nodes, that's a ValueError before anything is written.
            float32 .npy files are used in place, so on a disk every node reads they work on any
            number of nodes.

    Returns:
        A `SimrResult`; see its docstring.

    A run with an output adds a record to the catalog's "runs" when it starts and completes it when it ends:
    "started", "finished" (None if the run never got there), "code", "run" (its return and save lists),
    "simulator" and "image" (the full configs), "sim_metrics" and "img_metrics" (the filled-in entries),
    "externals" (each used external's source, shape and hash, as in the result),
    "model_ids" (row m's model id), and at the end "status", "seconds" and "errors" (as in the result, as
    lists). Metric values aren't saved yet; saving them is a later step.
    """
    models = _check_models(models)
    parts = _read_request(request)
    metric_sections, derived_entries = {"sim_metrics": parts[4], "img_metrics": parts[5]}, parts[6]
    arrays, files, records = _read_externals(
        externals, _external_uses(metric_sections, derived_entries), image_configs=parts[3]
    )
    # Derived entries made only from externals are computed here, once, and from then on travel with the
    # externals (as "drv.<name>"): checked, hashed, and shared with a pool's workers the same way.
    hashes = _derive_for_the_run(derived_entries, arrays, files, metric_sections, image_configs=parts[3])
    _check_external_weights(metric_sections, arrays)
    for_the_run = {key: arrays[key] for key in hashes}  # the arrays themselves, for run.return
    shared_folder = None
    if arrays and map is not builtins.map:  # a pool: its workers share one copy of each external
        arrays, shared_folder = _share_externals(arrays, files, {**records, **hashes})
    try:
        return _run(models, parts, arrays, records, for_the_run, output, map)
    finally:
        if shared_folder is not None:
            shutil.rmtree(shared_folder, ignore_errors=True)


def _run(models: np.ndarray, parts: tuple, external_arrays: dict, external_records: dict, for_the_run: dict,
         output, map) -> SimrResult:
    """`run`, once the models, the request and the externals are checked. `external_arrays` holds each
    external a metric uses: the array itself, or (for a pool) the path its workers map it from;
    `external_records` its `{"source", "shape", "hash"}`; `for_the_run` the derived arrays computed once
    for the run, by "drv.<name>".""" 
    returned, saved, configs, image_configs, sim_entries, img_entries, derived_entries = parts
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
            "image": [_json_ready(asdict(config)) for config in image_configs],
            "derived": derived_entries,
            "sim_metrics": sim_entries,
            "img_metrics": img_entries,
            "externals": external_records,
            "model_ids": ids,
        }
        catalog["runs"].append(run_record)
        _write_catalog(catalog_file, catalog)

    shapes = {
        "sim": (len(models), len(configs)),
        "img": (len(models), len(configs), len(image_configs)),
        "sim_metrics": (len(models), len(configs), len(sim_entries)),
        "img_metrics": (len(models), len(configs), len(image_configs), len(img_entries)),
    }
    levels = {f"drv.{entry['name']}": entry["level"] for entry in derived_entries}
    # A derived array computed once for the run comes back as itself; the rest are indexed like their stage's data.
    data = {
        name: for_the_run[name] if levels.get(name) == "run"
        else np.empty(shapes[levels.get(name) or _stage(name)], dtype=object)
        for name in returned
    }
    metric_values = {stage: np.full(shapes[stage], None, dtype=object) for stage in ("sim_metrics", "img_metrics")}
    status = {stage: np.empty(shape, dtype=object) for stage, shape in shapes.items()}
    seconds = {stage: np.zeros(shape) for stage, shape in shapes.items()}
    errors = {stage: np.full(shape, None, dtype=object) for stage, shape in shapes.items()}

    # One task per model; rows holding the same model (one id) share a task, so no two tasks write one folder.
    groups: dict = {}
    for m in range(len(models)):
        groups.setdefault(ids[m] if ids[m] is not None else m, []).append(m)
    rows = list(groups.values())
    tasks = [
        _ModelTask(row=r[0], model_id=ids[r[0]], params=models[r[0]], configs=configs,
                   image_configs=image_configs, returned=returned, saved=saved,
                   sim_metric_configs=sim_entries, img_metric_configs=img_entries,
                   derived_configs=derived_entries, externals=external_arrays,
                   folder=root / f"model_{ids[r[0]]}" if root is not None else None, versions=versions)
        for r in rows
    ]
    last_write = time.monotonic()
    done = 0
    try:
        for task_rows, outcome in zip(rows, map(_run_model, tasks)):
            for m in task_rows:
                for item in returned:
                    if levels.get(item) == "run":
                        continue  # already whole: the same array for every model
                    for index, value in np.ndenumerate(outcome["data"][item]):
                        data[item][(m, *index)] = value  # cell by cell: numpy would try to merge equal-shape arrays
                for stage in metric_values:
                    metric_values[stage][m] = outcome["metrics"][stage]
                for stage in shapes:
                    status[stage][m] = outcome["status"][stage]
                    seconds[stage][m] = outcome["seconds"][stage]
                    errors[stage][m] = outcome["errors"][stage]
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
    if any(_stage(name) == "img" for name in returned + saved) or img_entries:
        counts = {word: int((status["img"] == word).sum()) for word in ("ran", "reused", "failed", "skipped")}
        logger.info("simr: %d images: %d ran, %d reused, %d failed, %d skipped (simulation failed)",
                    status["img"].size, *counts.values())
    for stage, label in (("sim_metrics", "simulation-level"), ("img_metrics", "image-level")):
        if status[stage].size:
            counts = {word: int((status[stage] == word).sum()) for word in ("ran", "failed", "skipped")}
            logger.info("simr: %d %s metric cells: %d ran, %d failed, %d skipped",
                        status[stage].size, label, *counts.values())
    if catalog is not None:
        run_record["finished"] = _now()
        _record_status(catalog_file, catalog, run_record, status, seconds, errors)
    return SimrResult(models=models, simulators=configs, images=image_configs, data=data,
                      sim_metrics=metric_values["sim_metrics"], img_metrics=metric_values["img_metrics"],
                      sim_metric_configs=sim_entries, img_metric_configs=img_entries,
                      derived_configs=derived_entries, externals=external_records, ids=ids,
                      status=status, seconds=seconds, errors=errors)


LOG_FORMAT = "%(asctime)s %(levelname)s [%(processName)s] %(name)s: %(message)s"


# Where MPI launchers put each process's rank: OpenMPI's mpirun, then MPICH/Intel MPI.
MPI_RANK_VARIABLES = ("OMPI_COMM_WORLD_RANK", "PMI_RANK")


def configure_logging(level: str | int = "INFO", use_rank: bool = True) -> None:
    """Set up logging for an application: once in the main process, and once in each worker a pool starts
    (as its initializer). Every line names its process, to tell interleaved workers apart: the MPI rank
    ("rank 3") when an MPI launcher set one, else the process name. simr itself never calls this.

    use_rank=False always names the process. For a local pool started under mpirun (as a Slurm job script
    may do), where every process would otherwise read the same rank 0."""
    rank = None
    if use_rank:
        rank = next((os.environ[name] for name in MPI_RANK_VARIABLES if name in os.environ), None)
    log_format = LOG_FORMAT if rank is None else LOG_FORMAT.replace("%(processName)s", f"rank {rank}")
    logging.basicConfig(level=level, format=log_format)


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
        status={stage: values.tolist() for stage, values in status.items()},
        seconds={stage: values.tolist() for stage, values in seconds.items()},
        errors={stage: values.tolist() for stage, values in errors.items()},
    )
    _write_catalog(catalog_file, catalog)


def _stage(name: str) -> str:
    """The stage a data name belongs to: "sim.orbit" -> "sim"."""
    return name.split(".", 1)[0]


@dataclass
class _ModelTask:
    """One model's work, as plain data, so it can be sent to any worker process."""

    row: int  # the first row of `models` holding this model (for log messages)
    model_id: str | None
    params: np.ndarray
    configs: list[SimulatorConfig]
    image_configs: list[ImageConfig]
    returned: list[str]
    saved: list[str]
    sim_metric_configs: list[dict]
    img_metric_configs: list[dict]
    derived_configs: list[dict]  # every derived entry, in order; those with level "sim" or "img" are computed here
    externals: dict  # the externals the metrics use, by key: the array, or (in a pool) the path to map it from
    folder: Path | None  # the model's folder, when the output is a folder
    versions: dict | None


def _run_model(task: _ModelTask) -> dict:
    """Every simulator entry of one model, then every image entry of each simulation: load what's saved,
    compute the rest, save what's asked for, and catch each simulation's and image's failure. Touches only
    this model's folder, never the catalog.

    Returns {"data": {item: array of values}, "metrics", "status", "seconds", "errors": {stage: array}},
    each array shaped [simulator] for "sim", [simulator, image] for "img", and with an [entry] axis after
    those for "sim_metrics" and "img_metrics".
    """
    shapes = {
        "sim": (len(task.configs),),
        "img": (len(task.configs), len(task.image_configs)),
        "sim_metrics": (len(task.configs), len(task.sim_metric_configs)),
        "img_metrics": (len(task.configs), len(task.image_configs), len(task.img_metric_configs)),
    }
    levels = {f"drv.{entry['name']}": entry["level"] for entry in task.derived_configs}
    sim_derived = [entry for entry in task.derived_configs if entry["level"] == "sim"]
    img_derived = [entry for entry in task.derived_configs if entry["level"] == "img"]
    outcome = {
        "data": {item: np.empty(shapes[levels.get(item) or _stage(item)], dtype=object)
                 for item in task.returned if levels.get(item) != "run"},
        "metrics": {stage: np.full(shapes[stage], None, dtype=object) for stage in ("sim_metrics", "img_metrics")},
        "status": {stage: np.empty(shape, dtype=object) for stage, shape in shapes.items()},
        "seconds": {stage: np.zeros(shape) for stage, shape in shapes.items()},
        "errors": {stage: np.full(shape, None, dtype=object) for stage, shape in shapes.items()},
    }
    externals = _open_externals(task.externals)
    sim_returned = [item for item in task.returned if _stage(item) == "sim"]
    sim_saved = [item for item in task.saved if _stage(item) == "sim"]
    # What the metrics, and the derived entries computed here, need in memory
    metric_sim_needs = [
        name for entry in task.sim_metric_configs + task.img_metric_configs + sim_derived + img_derived
        for name in entry["inputs"] if name in SIM_DATA
    ]
    metric_image_needs = [
        name for entry in task.img_metric_configs + img_derived for name in entry["inputs"] if name in IMAGE_DATA
    ]
    image_items = list(dict.fromkeys(
        [item for item in task.returned + task.saved if _stage(item) == "img"] + metric_image_needs
    ))
    image_saved = [item for item in task.saved if _stage(item) == "img"]
    folder = task.folder
    record = _open_record(folder, task.params) if folder is not None else None
    # Each image entry's saved record, or None when it was saved with a different config (not used or changed)
    image_entries = [
        _saved_image_entry(folder, record, image_name(i), image_config) if record is not None else {}
        for i, image_config in enumerate(task.image_configs)
    ]
    for s, config in enumerate(task.configs):
        name = simulation_name(s)
        start = time.perf_counter()
        stored, new = _saved_data(folder, record, name, config) if record is not None else ({}, True)
        # Images count as saved only under a saved simulation with this config, and an image entry with its config
        usable = [stored is not None and not new and entry is not None for entry in image_entries]
        stored_images = [
            _saved_images(folder, entry, name) if ok else {} for entry, ok in zip(image_entries, usable)
        ]
        image_to_save = [
            [item for item in image_saved if item not in saved] if stored is not None and entry is not None else []
            for entry, saved in zip(image_entries, stored_images)
        ]
        # The sim items wanted in memory: those returned, those the metrics use, and those the images
        # still to be made need
        wanted = list(dict.fromkeys(
            sim_returned + metric_sim_needs
            + [need for saved in stored_images for item in image_items if item not in saved
               for need in IMAGE_DATA[item][0]]
        ))
        to_save = [] if stored is None else [item for item in sim_saved if item not in stored]
        if new and (sim_saved or any(image_to_save)):
            to_save.append("sim.timing")  # every saved simulation records its timing in model.json
        try:
            values, computed = _simulate(task.params, config, wanted, to_save, stored or {})
        except Exception as error:  # a failed simulation: recorded, and the run moves on
            logger.exception("model %d (id %s) on simulator %d failed", task.row, task.model_id, s)
            outcome["status"]["sim"][s] = "failed"
            outcome["errors"]["sim"][s] = f"{type(error).__name__}: {error}"
            outcome["seconds"]["sim"][s] = time.perf_counter() - start
            outcome["status"]["img"][s] = "skipped" if image_items else "empty"
            outcome["status"]["sim_metrics"][s] = "skipped"
            outcome["status"]["img_metrics"][s] = "skipped"
            continue
        for item in sim_returned:
            outcome["data"][item][s] = values[item]
        if to_save:  # errors while writing aren't caught: they stop the run
            arrays = {item: values[item] for item in to_save if item != "sim.timing"}
            _save_simulation(folder, record, name, config, arrays, values.get("sim.timing"), task.versions)
        if not sim_returned and not sim_saved and not image_items and not task.sim_metric_configs \
                and not sim_derived:
            outcome["status"]["sim"][s] = "empty"  # the request asked for nothing, so there was nothing to do
        else:
            outcome["status"]["sim"][s] = "reused" if stored and not computed else "ran"  # reused: nothing computed
        outcome["seconds"]["sim"][s] = time.perf_counter() - start
        # Derived entries made from this simulation's items, in order; a failure is that entry's alone,
        # and reaches only what uses it
        sim_made, sim_failed = {}, {}
        for entry in sim_derived:
            _derive(task, entry, (externals, sim_made, values), sim_made, sim_failed, f"simulator {s}")
        for item in task.returned:
            if levels.get(item) == "sim":
                outcome["data"][item][s] = sim_made.get(item)
        for k, entry in enumerate(task.sim_metric_configs):
            _run_metric(task, entry, (externals, sim_made, values), sim_failed,
                        outcome, "sim_metrics", (s, k), f"simulator {s}, sim_metrics[{k}]")
        for i, image_config in enumerate(task.image_configs):
            if not image_items:
                outcome["status"]["img"][s, i] = "empty"
                continue
            start = time.perf_counter()
            made, image_computed = {}, False
            try:
                for item in image_items:
                    if item in stored_images[i]:
                        if item in task.returned or item in metric_image_needs:
                            made[item] = stored_images[i][item]()
                    else:
                        made[item] = IMAGE_DATA[item][1](values, task.params, config, image_config)
                        image_computed = True
            except Exception as error:  # a failed image: recorded, nothing saved, and the run moves on
                logger.exception("model %d (id %s), simulator %d, image %d failed", task.row, task.model_id, s, i)
                outcome["status"]["img"][s, i] = "failed"
                outcome["errors"]["img"][s, i] = f"{type(error).__name__}: {error}"
                outcome["seconds"]["img"][s, i] = time.perf_counter() - start
                outcome["status"]["img_metrics"][s, i] = "skipped"
                continue
            for item in task.returned:
                if _stage(item) == "img":
                    outcome["data"][item][s, i] = made[item]
            if image_to_save[i]:  # errors while writing aren't caught: they stop the run
                arrays = {item: made[item] for item in image_to_save[i]}
                _save_images(folder, record, image_name(i), image_config, name, arrays)
            outcome["status"]["img"][s, i] = "ran" if image_computed else "reused"
            outcome["seconds"]["img"][s, i] = time.perf_counter() - start
            img_made, img_failed = {}, dict(sim_failed)
            sources = (externals, img_made, sim_made, made, values)
            for entry in img_derived:
                _derive(task, entry, sources, img_made, img_failed, f"simulator {s}, image {i}")
            for item in task.returned:
                if levels.get(item) == "img":
                    outcome["data"][item][s, i] = img_made.get(item)
            for k, entry in enumerate(task.img_metric_configs):
                _run_metric(task, entry, sources, img_failed, outcome, "img_metrics", (s, i, k),
                            f"simulator {s}, image {i}, img_metrics[{k}]")
    return outcome


def _derive(task: "_ModelTask", entry: dict, sources: tuple, made: dict, failed: dict, where: str) -> None:
    """One derived entry for one simulation or image: run its method on its inputs (each found in the
    first of `sources` that has it) and keep the array in `made` under "drv.<name>". A failure is
    logged and recorded in `failed` (name -> message); whatever uses the entry then fails in turn."""
    key = f"drv.{entry['name']}"
    try:
        for name in entry["inputs"]:
            if name in failed:
                raise ValueError(f"its input {name} failed ({failed[name]})")
        arrays = [next(source[name] for source in sources if name in source) for name in entry["inputs"]]
        made[key] = derived.compute(entry["method"], arrays, entry["params"])
    except Exception as error:
        logger.exception("model %d (id %s), %s, derived %r (%s) failed", task.row, task.model_id, where,
                         entry["name"], entry["method"])
        failed[key] = f"{type(error).__name__}: {error}"


def _run_metric(task: "_ModelTask", entry: dict, sources: tuple, failed: dict, outcome: dict, stage: str,
                index: tuple, where: str) -> None:
    """One metric cell: run the entry's method on its inputs (each found in the first of `sources` that
    has it) and record the value, status, time and error. A failure (the method raised, its value failed
    `metrics.compute`'s checks, or a derived entry it uses is in `failed`) is this cell's alone: it's
    logged, and the simulation, image and other metrics move on."""
    start = time.perf_counter()
    try:
        for name in entry["inputs"]:
            if name in failed:
                raise ValueError(f"its input {name} failed ({failed[name]})")
        arrays = [next(source[name] for source in sources if name in source) for name in entry["inputs"]]
        value = metrics.compute(entry["method"], arrays, select=entry["select"], params=entry["params"])
    except Exception as error:
        logger.exception("model %d (id %s), %s (%s) failed", task.row, task.model_id, where, entry["method"])
        outcome["status"][stage][index] = "failed"
        outcome["errors"][stage][index] = f"{type(error).__name__}: {error}"
    else:
        outcome["metrics"][stage][index] = value
        outcome["status"][stage][index] = "ran"
    outcome["seconds"][stage][index] = time.perf_counter() - start


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
    """A model folder's record (`model.json`): "model" (the 14 named parameters), "simulations" (each saved
    simulation by name: "config", "timing", "files" (item: path in the folder) and "code") and "images"
    (each image entry by name: "config", and "files": simulation name -> item -> path)."""
    with open(Path(folder) / "model.json", encoding="utf-8") as f:
        return json.load(f)


def load_images(
    folder: str | Path,
    simulation: str | None = None,
    image: str | None = None,
    items: str | list[str] | None = None,
) -> dict[tuple[str, str, str], np.ndarray]:
    """Saved images, as float32, just as `run` returns them, picked by any mix of simulation, image entry
    and item; each left as None means all.

    Args:
        simulation: e.g. "sim_00" (`simulation_name(0)`).
        image: e.g. "img_0001" (`image_name(1)`).
        items: an item or a list of them, e.g. "img.model".

    Returns:
        {(simulation, image, item): array}, sorted, e.g. {("sim_00", "img_0001", "img.model"): (128, 128)}.
        {} when nothing saved matches.

    Raises:
        KeyError: `model.json` has no such simulation or image entry.
        ValueError: an unknown item.

    Only what `model.json` lists counts as saved, even if other files exist.
    """
    folder = Path(folder)
    record = load_model(folder)
    entries = record.get("images", {})
    if simulation is not None and simulation not in record["simulations"]:
        raise KeyError(f"{folder.name} has no saved simulation {simulation!r}")
    if image is not None and image not in entries:
        raise KeyError(f"{folder.name} has no saved image entry {image!r}")
    if isinstance(items, str):
        items = [items]
    unknown = [item for item in items or [] if item not in IMAGE_DATA]
    if unknown:
        raise ValueError(f"unknown image items: {unknown}. Known: {list(IMAGE_DATA)}")
    found = {}
    for image_entry, entry in entries.items():
        for simulation_entry, files in entry["files"].items():
            for item, file in files.items():
                if (image in (None, image_entry) and simulation in (None, simulation_entry)
                        and (items is None or item in items)):
                    found[simulation_entry, image_entry, item] = np.load(folder / file)
    return dict(sorted(found.items()))


def simulation_name(s: int) -> str:
    """Simulator entry s's name in a model folder: "sim_00"."""
    return f"sim_{s:02d}"


def image_name(i: int) -> str:
    """Image entry i's name in a model folder: "img_0000"."""
    return f"img_{i:04d}"


def image_file(simulation: str, image: str, item: str) -> str:
    """Where one saved image goes in its model folder: "img.model" of "sim_00" with "img_0001" is
    "images/sim_00_img_0001_model.npy"."""
    return f"images/{simulation}_{image}_{item.split('.', 1)[1]}.npy"


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
        return {"model": model, "simulations": {}, "images": {}}
    record = load_model(folder)
    if record["model"] != model:
        raise ValueError(
            f"{folder} holds a different model; save this run in another output folder.\n"
            f"  saved: {record['model']}\n  this:  {model}"
        )
    record.setdefault("images", {})  # saved before images could be
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
    _write_record(folder, record)


def _saved_image_entry(folder, record, name, config) -> dict | None:
    """An image entry's record in model.json ({} if it has none yet), or None if it was saved with a
    different config: then nothing of it is used or added."""
    entry = record["images"].get(name)
    if entry is None:
        return {}
    if entry["config"] != _config_json(config):
        logger.warning("%s/%s was saved with a different image config; not used or changed", folder.name, name)
        return None
    return entry


def _saved_images(folder, entry, simulation) -> dict:
    """{item: function that loads it} for one simulation's saved images in one image entry."""
    files = entry.get("files", {}).get(simulation, {})
    return {item: (lambda file=file: np.load(folder / file)) for item, file in files.items()}


def _save_images(folder, record, name, config, simulation, arrays) -> None:
    """Write one simulation's images for one image entry, then list them in model.json (all atomically)."""
    entry = record["images"].get(name) or {"config": _config_json(config), "files": {}}
    for item, array in arrays.items():
        file = image_file(simulation, name, item)
        (folder / file).parent.mkdir(parents=True, exist_ok=True)
        _write_atomically(folder / file, lambda f, array=array: np.save(f, array))
        entry["files"].setdefault(simulation, {})[item] = file
    record["images"][name] = entry
    _write_record(folder, record)


def _write_record(folder: Path, record: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, indent=2).encode("utf-8")
    _write_atomically(folder / "model.json", lambda f: f.write(text))


def _config_json(config) -> dict:
    """A config as model.json stores it: plain numbers, and tuples as lists, so it compares equal to its
    copy read back."""
    return json.loads(json.dumps(_json_ready(asdict(config))))


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


def _read_request(
    request: dict,
) -> tuple[list[str], list[str], list[SimulatorConfig], list[ImageConfig], list[dict], list[dict]]:
    _reject_unknown("request", request, REQUEST_KEYS)
    run_section = request.get("run", {})
    _reject_unknown("run", run_section, RUN_KEYS)
    configs = _stage_configs(request, "simulator", SimulatorConfig)
    image_configs = _stage_configs(request, "image", ImageConfig)
    derived_entries = _derived_entries(request)
    made = {f"drv.{entry['name']}": entry for entry in derived_entries}
    returned, saved = _names(run_section, "return", list(made)), _names(run_section, "save", [])
    sim_entries = _metric_entries(request, "sim_metrics", made)
    img_entries = _metric_entries(request, "img_metrics", made)
    return returned, saved, configs, image_configs, sim_entries, img_entries, derived_entries


def _derived_entries(request: dict) -> list[dict]:
    """The `derived` section's entries, checked against the derived methods before anything runs, with
    every setting's default filled in and each entry's level worked out from its inputs.

    An entry's inputs may be externals, sim.* or img.* items, and entries above it in the list (as
    "drv.<name>"), so entries are computed in order and can never form a loop. One entry (a dict) or a
    list; a missing section or an empty list means none.
    """
    entries = request.get("derived", [])
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, (list, tuple)):
        raise ValueError(f"derived must be an entry (a dict) or a list of them: {entries!r}")
    filled: list[dict] = []
    made: dict[str, dict] = {}
    for i, entry in enumerate(entries):
        where = f"derived[{i}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{where} must be a dict with a name, a method and inputs: {entry!r}")
        _reject_unknown(where, entry, DERIVED_ENTRY_KEYS)
        name = entry.get("name")
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f'{where} needs a "name" of letters, digits and _ (metrics use it as "drv.<name>"): {name!r}')
        if f"drv.{name}" in made:
            raise ValueError(f"{where}: the name {name!r} is already used by an entry above it")
        method = derived.METHODS.get(entry.get("method"))
        if method is None:
            raise ValueError(f"{where}.method: unknown method {entry.get('method')!r}. Known: {list(derived.METHODS)}")
        inputs = entry.get("inputs", [])
        if isinstance(inputs, str) or not isinstance(inputs, (list, tuple)) or len(inputs) != len(method.inputs):
            raise ValueError(
                f"{where}.inputs: {method.name} takes {len(method.inputs)} input(s), one per kind: {list(method.inputs)}"
            )
        level = 0
        for j, source in enumerate(inputs):
            if source in SIM_INPUT_KINDS:
                kind, level = SIM_INPUT_KINDS[source], max(level, 1)
            elif source in IMAGE_DATA:
                kind, level = "image", 2
            elif isinstance(source, str) and source.startswith("drv."):
                if source not in made:
                    raise ValueError(
                        f"{where}.inputs[{j}]: {source!r} isn't a derived entry above this one. Above: {list(made)}"
                    )
                kind = derived.METHODS[made[source]["method"]].output
                level = max(level, DERIVED_LEVELS.index(made[source]["level"]))
            elif not isinstance(source, str) or source.startswith(STAGE_PREFIXES):
                raise ValueError(
                    f"{where}.inputs[{j}]: {source!r} isn't an input. Known: {[*SIM_INPUT_KINDS, *IMAGE_DATA]}, "
                    f"a derived entry above this one, or a key in the run's externals"
                )
            else:
                continue  # an external: `run` checks its key and its array, as the kind this place takes
            if kind != method.inputs[j]:
                raise ValueError(
                    f"{where}.inputs[{j}]: {method.name} takes a {method.inputs[j]} here, but {source} is a {kind}"
                )
        params = entry.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"{where}.params must be a dict: {params!r}")
        unknown = [key for key in params if key not in method.settings]
        if unknown:
            raise ValueError(f"{where}.params: unknown settings {unknown}. {method.name} has: {list(method.settings)}")
        made[f"drv.{name}"] = {"name": name, "method": method.name, "inputs": list(inputs),
                               "params": {**method.settings, **params}, "level": DERIVED_LEVELS[level]}
        filled.append(made[f"drv.{name}"])
    return filled


def _derive_for_the_run(entries: list[dict], arrays: dict, files: dict, metric_sections: dict,
                        image_configs: list) -> dict[str, dict]:
    """Compute the derived entries made only from externals, once, before anything is simulated, and add
    each to `arrays` as "drv.<name>" (and to `files` as needing a shared copy in a pool), so from here
    on it's handled like an external. Records each one's shape and hash in its entry.

    Returns "drv.<name>" -> {"hash"} for the shared copies' file names. Raises ValueError, naming the
    entry, if a method fails, or if a derived image a metric uses isn't the size of every image entry.
    """
    hashes = {}
    used_as_image = {
        name for section in ("img_metrics",) for entry in metric_sections[section] for name in entry["inputs"]
    }
    for i, entry in enumerate(entries):
        if entry["level"] != "run":
            continue
        key = f"drv.{entry['name']}"
        try:
            value = derived.compute(entry["method"], [arrays[name] for name in entry["inputs"]], entry["params"])
        except ValueError as error:
            raise ValueError(f"derived[{i}] ({entry['name']!r}): {error}") from None
        if value.ndim == 2 and key in used_as_image:
            for n, config in enumerate(image_configs):
                if value.shape != (config.height, config.width):
                    raise ValueError(
                        f"derived[{i}] ({entry['name']!r}) is {value.shape[0]} x {value.shape[1]} (height x "
                        f"width), but image[{n}] makes {config.height} x {config.width} images"
                    )
        value = value.view()
        value.flags.writeable = False
        arrays[key], files[key] = value, None
        entry["shape"], entry["hash"] = list(value.shape), array_hash(value)
        hashes[key] = {"hash": entry["hash"]}
        logger.info("simr: derived %r (%s): shape %s, hash %s", entry["name"], entry["method"], value.shape,
                    entry["hash"])
    return hashes


def _metric_entries(request: dict, section: str, made: dict[str, dict]) -> list[dict]:
    """A metric section's entries, filled in. One entry (a dict) or a list of them; a missing section or
    an empty list means no metrics (unlike the stage sections, [] is allowed: a placeholder while a
    config only computes simulations or images). `made` is the derived entries by "drv.<name>"."""
    entries = request.get(section, [])
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, (list, tuple)):
        raise ValueError(f"{section} must be a metric entry (a dict) or a list of them: {entries!r}")
    return [_metric_entry(f"{section}[{i}]", entry, section, made) for i, entry in enumerate(entries)]


def _metric_entry(where: str, entry, section: str, made: dict[str, dict]) -> dict:
    """One metric entry, checked against the metrics table before anything runs, with every setting's
    default filled in. The filled-in entry always records all four keys."""
    _reject_unknown(where, entry, METRIC_ENTRY_KEYS)
    if "method" not in entry:
        raise ValueError(f'{where} needs a "method": one of {list(metrics.METHODS)}')
    method = metrics.METHODS.get(entry["method"])
    if method is None:
        raise ValueError(f"{where}.method: unknown method {entry['method']!r}. Known: {list(metrics.METHODS)}")
    if method.setup is not None:
        raise NotImplementedError(
            f"{where}: {method.name} declares a setup step, which simr can't run yet (planned with DL metrics)"
        )
    inputs = entry.get("inputs", [])
    kinds = _input_kinds(method)
    if isinstance(inputs, str) or not isinstance(inputs, (list, tuple)) \
            or not len(method.inputs) <= len(inputs) <= len(kinds):
        raise ValueError(
            f"{where}.inputs: {method.name} takes {len(method.inputs)} input(s), one per kind: {list(method.inputs)}"
            + (f", and optionally its weights (a {method.weights}) after them" if method.weights else "")
        )
    for j, name in enumerate(inputs):
        if section == "sim_metrics" and name in IMAGE_DATA:
            raise ValueError(f"{where}.inputs[{j}]: an img.* input belongs in img_metrics, not sim_metrics")
        kind = SIM_INPUT_KINDS.get(name, "image" if name in IMAGE_DATA else None)
        if isinstance(name, str) and name.startswith("drv."):
            if name not in made:
                raise ValueError(
                    f"{where}.inputs[{j}]: {name!r} isn't an entry in the derived section, which has {list(made)}"
                )
            if section == "sim_metrics" and made[name]["level"] == "img":
                raise ValueError(
                    f"{where}.inputs[{j}]: {name} is made from img.* items, so it belongs in img_metrics, "
                    f"not sim_metrics"
                )
            kind = derived.METHODS[made[name]["method"]].output
        elif kind is None:
            if not isinstance(name, str) or name.startswith(STAGE_PREFIXES):
                raise ValueError(
                    f"{where}.inputs[{j}]: {name!r} isn't a metric input. Known: "
                    f"{[*SIM_INPUT_KINDS, *IMAGE_DATA]}, a derived entry (drv.<name>), or a key in the run's externals"
                )
            continue  # an external: `run` checks its key and its array, as the kind this place takes
        if kind != kinds[j]:
            raise ValueError(
                f"{where}.inputs[{j}]: {method.name} takes a {kinds[j]} here, but {name} is a {kind}"
            )
    from_model = [made[name]["level"] if name in made else "img" if name in IMAGE_DATA
                  else "sim" if name in SIM_INPUT_KINDS else "run" for name in inputs]  # an external: "run"
    if all(level == "run" for level in from_model):
        raise ValueError(
            f"{where}: no simr item among the inputs {list(inputs)}, so the value wouldn't depend on the model"
        )
    if section == "img_metrics" and "img" not in from_model:
        raise ValueError(f"{where}: no img.* input; a metric that uses only sim.* items belongs in sim_metrics")
    select = entry.get("select")
    if select is not None:
        if method.kind != "descriptive":
            raise ValueError(
                f"{where}.select: {method.name} is a scalar method; select picks one float out of a descriptive one"
            )
        if select not in method.scalar_outputs:
            raise ValueError(
                f"{where}.select: {select!r} isn't a single-float output of {method.name}. "
                f"Choices: {list(method.scalar_outputs)}"
            )
    params = entry.get("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"{where}.params must be a dict: {params!r}")
    unknown = [key for key in params if key not in method.settings]
    if unknown:
        raise ValueError(f"{where}.params: unknown settings {unknown}. {method.name} has: {list(method.settings)}")
    return {"method": method.name, "inputs": list(inputs), "select": select, "params": {**method.settings, **params}}


def _input_kinds(method: metrics.Metric) -> tuple[str, ...]:
    """The kind each place in a metric entry's `inputs` takes: the method's inputs, then, for a method
    that can use weights, its optional weights."""
    return method.inputs + ((method.weights,) if method.weights else ())


def _check_external_weights(sections: dict[str, list[dict]], arrays: dict[str, np.ndarray]) -> None:
    """Before anything runs: where an entry's row weights and the input they weigh (its first) are both
    externals, they must have the same number of rows. Otherwise every model's cell would fail the same
    way, after its simulation."""
    for section, entries in sections.items():
        for k, entry in enumerate(entries):
            method, inputs = metrics.METHODS[entry["method"]], entry["inputs"]
            if method.weights != "weights" or len(inputs) <= len(method.inputs):
                continue
            first, weights = inputs[0], inputs[-1]
            if first in arrays and weights in arrays and len(arrays[first]) != len(arrays[weights]):
                raise ValueError(
                    f"{section}[{k}]: the weights {weights!r} have {len(arrays[weights])} values, but "
                    f"{first!r}, which they weigh, has {len(arrays[first])} rows"
                )


def _external_uses(sections: dict[str, list[dict]], derived_entries: list[dict]) -> dict[str, tuple[str, str]]:
    """The externals the filled-in derived and metric entries use: key -> (the input kind it's used as,
    the first place that uses it). One external has one kind: using a key as two kinds is an error."""
    uses: dict[str, tuple[str, str]] = {}
    places = {"derived": [(entry, derived.METHODS[entry["method"]].inputs) for entry in derived_entries]}
    for section, entries in sections.items():
        places[section] = [(entry, _input_kinds(metrics.METHODS[entry["method"]])) for entry in entries]
    for section, entries in places.items():
        for k, (entry, kinds) in enumerate(entries):
            for j, name in enumerate(entry["inputs"]):
                if name in SIM_INPUT_KINDS or name in IMAGE_DATA or name.startswith("drv."):
                    continue
                where = f"{section}[{k}].inputs[{j}]"
                if name in uses and uses[name][0] != kinds[j]:
                    raise ValueError(
                        f"{where}: the external {name!r} is used as a {kinds[j]} here but as a "
                        f"{uses[name][0]} in {uses[name][1]}; one external has one kind"
                    )
                uses.setdefault(name, (kinds[j], where))
    return uses


def _read_externals(externals, uses: dict[str, tuple[str, str]], image_configs: list) -> tuple[dict, dict, dict]:
    """The external arrays the metrics use, loaded, checked and hashed before anything runs. Returns three
    dicts by key: the read-only float32 array; the file a pool's workers can map as it is (a float32 .npy
    the caller gave by path), else None; and its record `{"source", "shape", "hash"}` (see `SimrResult`).
    `uses` is `_external_uses`' dict. Raises ValueError for any problem (see `run`)."""
    if externals is None:
        externals = {}
    if not isinstance(externals, dict):
        raise ValueError(f"externals must be a dict, key -> array or path: got {type(externals).__name__}")
    for key in externals:
        if not isinstance(key, str) or key.startswith(STAGE_PREFIXES):
            raise ValueError(f"externals: a key must be a string that doesn't start with {STAGE_PREFIXES}: {key!r}")
    for key, (_, where) in uses.items():
        if key not in externals:
            given = f"externals has {sorted(externals)}" if externals else "no externals were given"
            raise ValueError(f"{where}: {key!r} isn't a simr item, so it must be a key in externals, but {given}")
    unused = sorted(set(externals) - set(uses))
    if unused:
        logger.info("simr: externals no metric uses, ignored: %s", unused)
    arrays, files, records = {}, {}, {}
    for key, (kind, where) in uses.items():
        array, files[key], source, expected = _external_array(key, externals[key])
        metrics.check_input(kind, array, f"externals[{key!r}] (used by {where})")
        if kind == "image":
            for i, config in enumerate(image_configs):
                if array.shape != (config.height, config.width):
                    raise ValueError(
                        f"externals[{key!r}] (used by {where}) is {array.shape[0]} x {array.shape[1]} "
                        f"(height x width), but image[{i}] makes {config.height} x {config.width} images"
                    )
        digest = array_hash(array)
        if expected is not None and expected != digest:
            raise ValueError(f"externals[{key!r}]: the data's hash is {digest}, not the expected {expected}")
        arrays[key] = array
        records[key] = {"source": source, "shape": list(array.shape), "hash": digest}
        logger.info("simr: external %r: %s, shape %s, hash %s", key, source, array.shape, digest)
    return arrays, files, records


def array_hash(array: np.ndarray) -> str:
    """The hash that identifies an external's data: SHA-256 of its values as little-endian float32,
    with its shape. The same data gives the same hash from memory or from a file, on any machine. To
    get the hash to expect for a file: `simr.array_hash(np.load(path))`."""
    data = np.ascontiguousarray(array, dtype="<f4")
    digest = hashlib.sha256(f"float32{np.shape(array)}".encode())
    digest.update(memoryview(data).cast("B") if data.size else b"")
    return digest.hexdigest()


def _external_array(key: str, value) -> tuple[np.ndarray, Path | None, str, str | None]:
    """One external as a read-only float32 array: `value` itself, or the .npy file it's the path of (given
    as the path, or as `{"path": ..., "hash": ...}` with the hash its data must have). Also returns that
    file when it already holds float32, so a pool's workers can map it as it is (else None); the source
    for the record (the path, or "memory"); and the expected hash (None when none was given)."""
    file, source, expected = None, "memory", None
    if isinstance(value, dict):
        if "path" not in value or set(value) - {"path", "hash"}:
            raise ValueError(
                f'externals[{key!r}]: a dict holds "path" and, optionally, "hash": got the keys {sorted(value)}'
            )
        expected = value.get("hash")
        if expected is not None and not isinstance(expected, str):
            raise ValueError(f"externals[{key!r}]: the hash must be a string: {expected!r}")
        value = value["path"]
        if not isinstance(value, (str, Path)):
            raise ValueError(f"externals[{key!r}]: the path must be a string or Path: {value!r}")
    if isinstance(value, (str, Path)):
        path = Path(value)
        source = str(path)
        if not path.is_absolute() or path.suffix != ".npy":
            raise ValueError(f"externals[{key!r}]: a path must be absolute and name a .npy file: {str(value)!r}")
        try:
            value = np.load(path, mmap_mode="r", allow_pickle=False)  # mapped, not read into this process
        except (OSError, ValueError) as error:
            raise ValueError(f"externals[{key!r}]: can't load {path}: {error}") from None
        if value.dtype == np.float32:
            file = path
    if not isinstance(value, np.ndarray) or value.dtype.kind not in "fiu":
        got = f"{value.dtype} array" if isinstance(value, np.ndarray) else type(value).__name__
        raise ValueError(f"externals[{key!r}] must be a numeric numpy array or the path of a .npy file: got {got}")
    with np.errstate(over="ignore"):  # a value too big for float32 becomes inf, caught just below
        array = np.ascontiguousarray(value, dtype=np.float32).view()
    if not np.isfinite(array).all():
        raise ValueError(f"externals[{key!r}] holds NaN or inf (as float32)")
    array.flags.writeable = False
    return array, file, source, expected


# Where a pool's workers share externals: a folder that lives in memory, not on disk (Linux). Each node
# has its own, so sharing through it works only among processes on one node.
SHARED_MEMORY = Path("/dev/shm")


def _share_externals(arrays: dict[str, np.ndarray], files: dict[str, Path | None],
                     records: dict[str, dict]) -> tuple[dict, Path | None]:
    """Put one copy of each external where every worker of a pool on this node can map it, so the tasks
    carry paths instead of the arrays (which would be copied to a worker once per model).

    An external the caller gave as a float32 .npy file is mapped from that file, with no copy. Every
    other one is written once, as `<hash>.npy` (the hash in `records`, so the same data under two keys
    is written once), to a new folder in `SHARED_MEMORY` (the system's temporary folder, on disk, where
    there's no /dev/shm). Workers map the files read-only (`_open_externals`), so the
    node holds each external in memory once, however many workers use it.

    Returns key -> path, and the folder made (None if nothing was written), which `run` removes when
    the run ends. Raises ValueError, before writing, if the folder's free space is too small.
    """
    paths = {key: str(files[key]) for key in arrays if files[key] is not None}
    to_write = {key: array for key, array in arrays.items() if key not in paths}
    if not to_write:
        return paths, None
    _check_one_node(sorted(to_write))
    root = SHARED_MEMORY if SHARED_MEMORY.is_dir() else Path(tempfile.gettempdir())
    needed, free = sum(array.nbytes for array in to_write.values()), shutil.disk_usage(root).free
    if needed >= free:
        raise ValueError(
            f"the externals {sorted(to_write)} need {needed / 2**20:.1f} MB in {root} to be shared with the "
            f"pool's workers, but {free / 2**20:.1f} MB is free there (in Docker, raise it with --shm-size)"
        )
    folder = Path(tempfile.mkdtemp(prefix="simr_externals_", dir=root))
    try:
        for key, array in to_write.items():
            path = folder / f"{records[key]['hash']}.npy"
            if not path.exists():
                np.save(path, array)
            paths[key] = str(path)
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    logger.info("simr: externals %s shared with the workers through %s (%.1f MB)",
                sorted(to_write), folder, needed / 2**20)
    return paths, folder


# What MPI launchers tell every rank: (ranks on this node, ranks in the job). OpenMPI's mpirun, then
# MPICH / Intel MPI (Hydra).
MPI_NODE_VARIABLES = (("OMPI_COMM_WORLD_LOCAL_SIZE", "OMPI_COMM_WORLD_SIZE"), ("MPI_LOCALNRANKS", "PMI_SIZE"))


def _check_one_node(keys: list[str]) -> None:
    """Stop a run, before anything is written, whose externals must go through this node's memory
    (`keys`) when it was launched by MPI on more than one node: workers on other nodes couldn't see
    them. Read from the launcher's environment, so `simr` never imports `mpi4py`. Without those
    variables (no MPI launcher) there is nothing to check; a worker that can't see an external still
    stops the run when it starts (`_open_externals`)."""
    for local_name, size_name in MPI_NODE_VARIABLES:
        if local_name in os.environ and size_name in os.environ:
            local, size = int(os.environ[local_name]), int(os.environ[size_name])
            if local < size:
                raise ValueError(
                    f"the externals {keys} are shared through this node's memory (/dev/shm), but this MPI "
                    f"job has {size} ranks and only {local} on this node ({socket.gethostname()}). Run "
                    f"it on one node (e.g. one job per target), or give the externals as float32 .npy "
                    f"files on a disk every node can read, which are used in place"
                )
            return


# The externals this process has mapped, by (path, modified time, size): a worker maps each once and
# uses it for every model of a run. Only the latest task's are kept, so a pool that outlives many runs
# doesn't hold on to the earlier runs' memory.
_OPEN_EXTERNALS: dict[tuple, np.ndarray] = {}


def _open_externals(externals: dict) -> dict[str, np.ndarray]:
    """A task's externals as arrays: an array is used as it is; a path (`_share_externals`) is mapped
    read-only, once per process. Raises `SimrRunError` if this process can't see a path, which means
    it isn't on the node that started the run."""
    opened, keep = {}, {}
    for key, value in externals.items():
        if isinstance(value, np.ndarray):
            opened[key] = value
            continue
        try:
            stat = os.stat(value)
        except OSError:
            raise SimrRunError(
                f"external {key!r}: this worker (on {socket.gethostname()}) can't see {value}. Externals are "
                f"shared through one node's memory, so every worker must run on the node that started the run"
            ) from None
        identity = (value, stat.st_mtime_ns, stat.st_size)
        array = keep.get(identity, _OPEN_EXTERNALS.get(identity))  # `keep`: two keys can share one file
        if array is None:
            array = np.load(value, mmap_mode="r").view(np.ndarray)
        opened[key] = keep[identity] = array
    _OPEN_EXTERNALS.clear()
    _OPEN_EXTERNALS.update(keep)
    return opened


def _names(run_section: dict, key: str, derived_names: list[str]) -> list[str]:
    """run.return's or run.save's names, checked. `derived_names` are the "drv.<name>" it may also hold
    (the derived entries for run.return; none for run.save: derived arrays aren't saved yet)."""
    names = run_section.get(key, [])
    if isinstance(names, str):
        raise ValueError(f'run.{key} must be a list of names, e.g. ["{names}"]')
    if key == "save" and any(isinstance(name, str) and name.startswith("drv.") for name in names):
        raise ValueError("run.save: derived arrays can't be saved yet (a later step); run.return hands them back")
    known = [*SIM_DATA, *IMAGE_DATA, *derived_names]
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(f"unknown names in run.{key}: {unknown}. Known: {known}")
    return list(dict.fromkeys(names))  # drop repeats, keep order


def _stage_configs(request: dict, section: str, kind: type) -> list:
    """A stage's entries (one dict or config, or a list of them) as configs; one default config if the
    section is missing."""
    entries = request.get(section, [{}])
    if isinstance(entries, (dict, kind)):
        entries = [entries]
    if not isinstance(entries, (list, tuple)):
        raise ValueError(f"{section} must be a dict or a list of them: {entries!r}")
    if len(entries) == 0:
        raise ValueError(f"{section} is an empty list")
    return [_stage_config(f"{section}[{i}]", entry, kind) for i, entry in enumerate(entries)]


def _stage_config(where: str, entry, kind: type):
    if isinstance(entry, kind):
        return entry
    if not isinstance(entry, dict):
        raise ValueError(f"{where} must be a dict or {kind.__name__}: {entry!r}")
    _reject_unknown(where, entry, [f.name for f in fields(kind)])
    hints = typing.get_type_hints(kind)
    for key, value in entry.items():
        if not _has_type(value, hints[key]):
            raise ValueError(f"{where}.{key} must be {_type_name(hints[key])}: {value!r}")
    try:
        return kind(**entry)
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from error


def _has_type(value, hint) -> bool:
    """Whether a config value fits its field's type hint. Integers fit a float field (JSON writes 1.0 as 1);
    True and False fit neither numbers nor strings. A tuple field takes a list or tuple of that length, as
    JSON has no tuples."""
    if isinstance(hint, types.UnionType) or typing.get_origin(hint) is typing.Union:
        return any(_has_type(value, option) for option in typing.get_args(hint))
    if typing.get_origin(hint) is tuple:
        options = typing.get_args(hint)
        return (isinstance(value, (list, tuple)) and len(value) == len(options)
                and all(_has_type(v, option) for v, option in zip(value, options)))
    if isinstance(value, bool):
        return hint is bool
    if hint is int:
        return isinstance(value, (int, np.integer))
    if hint is float:
        return isinstance(value, (int, float, np.integer, np.floating))
    return isinstance(value, hint)


def _type_name(hint) -> str:
    if typing.get_origin(hint) is tuple:
        return f"a list of {len(typing.get_args(hint))} ({', '.join(map(_type_name, typing.get_args(hint)))})"
    options = typing.get_args(hint) if typing.get_args(hint) else (hint,)
    words = {int: "a whole number", float: "a number", str: "text", bool: "true or false"}
    return " or ".join(words.get(option, option.__name__) for option in options)


def _reject_unknown(where: str, section, known) -> None:
    if not isinstance(section, dict):
        raise ValueError(f"{where} must be a dict: {section!r}")
    unknown = [key for key in section if key not in known]
    if unknown:
        raise ValueError(f"unknown keys in {where}: {unknown}. Known: {list(known)}")
