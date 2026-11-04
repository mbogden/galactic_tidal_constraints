"""SIMR module: the workflow, disk reads and writes, and parallel runs. See docs/design/simr.md.

For now SIMR only runs the Simulator stage. Image and fitness stages come later.
"""

import argparse
import json
import logging
import os
import subprocess
import tempfile
import time
import typing
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np

from galactic_tidal_constraints.simr import config as config_module
from galactic_tidal_constraints.simr import sim
from galactic_tidal_constraints.simr.config import Config

logger = logging.getLogger(__name__)

SETTINGS_FILE = "settings.json"
SIMULATION_FILE = "simulation.npz"
RUN_CONFIG_FILE = "run_config.json"
SUMMARY_FILE = "run_summary.json"


class SettingsMismatch(Exception):
    """A folder holds a simulation made with different settings. Never silently kept or replaced."""


@dataclass
class Task:
    """One model on one simulator. Plain data, so it can be sent to a worker."""

    model_index: int
    simulator_index: int
    params: tuple[float, ...]
    simulator: dict


@dataclass
class Batch:
    """A slice of a run, sent to one process."""

    tasks: list[Task]
    run_seed: int
    output: str | None


# ------------------------------------ Saving and loading ------------------------------------ #


def _write_atomic(path: Path, write: typing.Callable[[typing.IO[bytes]], None]) -> None:
    """Write to a temporary name, then rename, so a crash can't leave a half-written file that looks
    finished."""
    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "wb") as temp:
            write(temp)
        os.chmod(temp_name, 0o644)  # temporary files start private to their owner
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _write_json(path: Path, data: typing.Any) -> None:
    _write_atomic(path, lambda f: f.write((json.dumps(data, indent=2) + "\n").encode()))


def simulation_folder(output: str | Path, model_index: int, simulator_index: int) -> Path:
    """The standard folder for one simulation: `<output>/model_000012/sim_00`."""
    return Path(output) / f"model_{model_index:06d}" / f"sim_{simulator_index:02d}"


def save_simulation(folder: str | Path, settings: dict, simulation: sim.Simulation) -> None:
    """Save one simulation. `simulation.npz` is written last, so if it exists the folder is complete."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    _write_json(folder / SETTINGS_FILE, settings)

    arrays = {"steps": simulation.steps, "times": simulation.times}
    if simulation.snapshots is not None:
        arrays["snapshots"] = simulation.snapshots
    if simulation.orbit is not None:
        arrays["orbit"] = simulation.orbit
    info = {
        "kind": simulation.kind,
        "t_start": simulation.t_start,
        "n_steps": simulation.n_steps,
        "start_fallback": simulation.start_fallback,
    }
    _write_atomic(folder / SIMULATION_FILE, lambda f: np.savez(f, info=json.dumps(info), **arrays))
    logger.debug("Saved %s", folder)


def load_simulation(folder: str | Path) -> tuple[dict, sim.Simulation]:
    """Read back what `save_simulation` wrote.

    Returns:
        The settings it was made with, and the simulation.
    """
    folder = Path(folder)
    settings = json.loads((folder / SETTINGS_FILE).read_text())
    with np.load(folder / SIMULATION_FILE) as data:
        info = json.loads(str(data["info"]))
        simulation = sim.Simulation(
            steps=data["steps"],
            times=data["times"],
            snapshots=data["snapshots"] if "snapshots" in data else None,
            orbit=data["orbit"] if "orbit" in data else None,
            **info,
        )
    return settings, simulation


def _has_simulation(folder: Path, settings: dict) -> bool:
    """Whether this folder already holds this simulation.

    Raises:
        SettingsMismatch: it holds a finished simulation made with other settings.
    """
    if not (folder / SIMULATION_FILE).exists():
        return False
    saved = json.loads((folder / SETTINGS_FILE).read_text())
    if saved != settings:
        different = sorted(key for key in settings if saved.get(key) != settings[key])
        raise SettingsMismatch(
            f"{folder} holds a simulation made with different settings ({different}). "
            "Use a new output folder, or delete this one if you don't need it."
        )
    return True


# ------------------------------------ Workflow ------------------------------------ #


def model_seed(run_seed: int, model_index: int) -> int:
    """A model's particle seed. It depends only on the run's seed and the model's index, so a result
    doesn't change with the number of workers or the order they finish in. All simulators on one
    model share it, so differences between simulators don't come from different starting particles.
    """
    return int(np.random.SeedSequence([run_seed, model_index]).generate_state(1)[0])


def task_settings(task: Task, run_seed: int) -> dict:
    """Everything that sets this simulation's result, for checking a saved one can be reused."""
    return {
        "params": list(task.params),
        "simulator": task.simulator,
        "seed": model_seed(run_seed, task.model_index),
    }


def run_task(task: Task, run_seed: int, output: str | None = None) -> dict:
    """Run one simulation, or reuse a saved one. A failure is logged and reported, not raised, so one
    bad model doesn't stop a run.

    Returns:
        A summary: `model`, `simulator`, `status` ("ran", "reused" or "failed"), `seconds`, and for
        a finished run `t_start` and `start_fallback`. With no `output`, also `simulation`.

    Raises:
        SettingsMismatch: the output folder holds a different simulation. This is a mistake in how
            the run was set up, so it does stop the run.
    """
    start = time.perf_counter()
    summary: dict = {"model": task.model_index, "simulator": task.simulator_index}
    settings = task_settings(task, run_seed)
    folder = simulation_folder(output, task.model_index, task.simulator_index) if output else None

    if folder is not None and _has_simulation(folder, settings):
        _, simulation = load_simulation(folder)
        summary.update(status="reused", t_start=simulation.t_start, start_fallback=simulation.start_fallback)
        summary["seconds"] = time.perf_counter() - start
        return summary

    try:
        simulator = config_module.from_dict({"simulator": [task.simulator]}).simulator[0]
        simulation = sim.simulate(np.array(task.params), simulator, seed=settings["seed"])
        data = simulation.orbit if simulation.kind == "orbit" else simulation.snapshots
        if not np.isfinite(data).all():
            raise FloatingPointError("the simulation produced NaN or infinite values")
        if folder is not None:
            save_simulation(folder, settings, simulation)
    except Exception:
        logger.exception("Model %d on simulator %d failed", task.model_index, task.simulator_index)
        summary.update(status="failed", t_start=None, start_fallback=None)
        summary["seconds"] = time.perf_counter() - start
        return summary

    summary.update(status="ran", t_start=simulation.t_start, start_fallback=simulation.start_fallback)
    if folder is None:
        summary["simulation"] = simulation
    summary["seconds"] = time.perf_counter() - start
    return summary


def run_batch(batch: Batch) -> list[dict]:
    """The workflow for one process: every task in its slice, in order."""
    return [run_task(task, batch.run_seed, batch.output) for task in batch.tasks]


# ------------------------------------ Splitting and parallel runs ------------------------------------ #


def make_tasks(config: Config) -> list[Task]:
    """Every model on every simulator."""
    simulators = config.to_dict()["simulator"]  # plain JSON types, the same as a saved settings file
    return [
        Task(model_index=m, simulator_index=s, params=params, simulator=simulator)
        for m, params in enumerate(config.models)
        for s, simulator in enumerate(simulators)
    ]


def make_batches(tasks: list[Task], n_workers: int, run_seed: int, output: str | None) -> list[Batch]:
    """Split tasks into shrinking batches: each takes half of what's left, shared between the workers.

    Early batches keep workers busy with little messaging. Late ones are small, so that no worker sits
    on a long batch while the others wait, which suits SPAM, where a simulation can take from a second
    to over a minute.
    """
    batches = []
    position = 0
    while position < len(tasks):
        size = max(1, (len(tasks) - position) // (2 * max(1, n_workers)))
        batches.append(Batch(tasks[position : position + size], run_seed, output))
        position += size
    return batches


def _start_run_folder(folder: Path, config: Config) -> None:
    """Record the run's config and commits. Resuming with the same config keeps the first record.

    Raises:
        SettingsMismatch: the folder belongs to a run with a different config.
    """
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / RUN_CONFIG_FILE
    if not path.exists():
        _write_json(path, {"config": config.to_dict(), "provenance": provenance()})
    elif json.loads(path.read_text())["config"] != config.to_dict():
        raise SettingsMismatch(f"{folder} belongs to a run with a different config. Use a new output folder.")


def run(
    config: Config,
    output: str | Path | None = None,
    map_function: typing.Callable = map,
    n_workers: int = 1,
) -> list[dict]:
    """Run every model on every simulator.

    Args:
        config: the run.
        output: the folder to save in. Required when `config.run.save` is "all".
        map_function: how to run batches. Built-in `map` runs them here; a pool's `map` runs them in
            parallel. Nothing below this point knows which it is.
        n_workers: how many workers `map_function` has, for sizing batches.

    Returns:
        One summary per simulation (see `run_task`), sorted by model and then simulator.
    """
    if config.run.save == "all" and output is None:
        raise ValueError('run.save is "all", so an output folder is required')
    folder = str(output) if config.run.save == "all" else None

    if folder is not None:
        _start_run_folder(Path(folder), config)

    batches = make_batches(make_tasks(config), n_workers, config.run.seed, folder)
    logger.info(
        "%d simulations in %d batches for %d workers",
        len(config.models) * len(config.simulator),
        len(batches),
        n_workers,
    )

    summaries = [summary for result in map_function(run_batch, batches) for summary in result]
    summaries.sort(key=lambda s: (s["model"], s["simulator"]))

    if folder is not None:
        _write_json(Path(folder) / SUMMARY_FILE, summaries)
    failed = sum(s["status"] == "failed" for s in summaries)
    if failed:
        logger.error("%d of %d simulations failed; see the errors above", failed, len(summaries))
    logger.info("Finished: %d simulations", len(summaries))
    return summaries


# ------------------------------------ Records of what ran ------------------------------------ #


def _git_state(folder: Path) -> dict:
    try:
        commit = subprocess.run(
            ["git", "-C", str(folder), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(folder), "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "uncommitted_changes": "unknown"}
    return {"commit": commit, "uncommitted_changes": bool(status.strip())}


def _jspam_state() -> dict:
    """The commit `jspam` was installed from. It's a pinned git install, so it has no local changes."""
    try:
        direct_url = metadata.distribution("jspam").read_text("direct_url.json")
        commit = json.loads(direct_url or "{}").get("vcs_info", {}).get("commit_id")
    except metadata.PackageNotFoundError:
        commit = None
    if commit is None:
        return {"commit": "unknown", "uncommitted_changes": "unknown"}
    return {"commit": commit, "uncommitted_changes": False}


def provenance() -> dict:
    """The commit of both repos, and whether either has uncommitted changes."""
    return {
        "galactic_tidal_constraints": _git_state(Path(__file__).parent),
        "jspam": _jspam_state(),
    }


# ------------------------------------ Running from the command line ------------------------------------ #


def configure_logging(level: str) -> None:
    """Set up logging once per process. The format includes the MPI rank."""
    rank = os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", "-"))
    logging.basicConfig(
        level=level,
        format=f"%(asctime)s %(levelname)s rank={rank} %(module)s.%(funcName)s: %(message)s",
        force=True,
    )


def _run_with_mode(
    config: Config, output: str | None, mode: str, workers: int | None, log_level: str
) -> list[dict]:
    """Create the pool once, pass its `map` down, and shut it down at the end."""
    if mode == "solo":
        return run(config, output)
    if mode == "local":
        from concurrent.futures import ProcessPoolExecutor

        n_workers = workers or os.cpu_count() or 1
        with ProcessPoolExecutor(n_workers, initializer=configure_logging, initargs=(log_level,)) as pool:
            return run(config, output, pool.map, n_workers)

    from mpi4py.futures import MPIPoolExecutor

    with MPIPoolExecutor(initializer=configure_logging, initargs=(log_level,)) as pool:
        n_workers = pool.num_workers
        if n_workers < 1:
            raise RuntimeError("MPI mode needs at least 2 ranks: one to hand out work and one to do it")
        return run(config, output, pool.map, n_workers)


def main(argv: list[str] | None = None) -> None:
    """Run SIMR from the command line.

    One process:  python -m galactic_tidal_constraints.simr --config run.json --output out
    MPI:          mpirun -n 8 python -m mpi4py.futures -m galactic_tidal_constraints.simr \\
                      --config run.json --output out --mode mpi
    """
    parser = argparse.ArgumentParser(
        description="Run SPAM simulations.", prog="galactic_tidal_constraints.simr"
    )
    parser.add_argument("overrides", nargs="*", help="config changes, e.g. run.seed=3 simulator.n1=500")
    parser.add_argument("--config", help="JSON config file")
    parser.add_argument("--output", help="folder to save the run in")
    parser.add_argument("--mode", choices=["solo", "local", "mpi"], default="solo")
    parser.add_argument("--workers", type=int, help="workers for --mode local; default: every CPU")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    config = config_module.load_config(args.config, args.overrides)
    summaries = _run_with_mode(config, args.output, args.mode, args.workers, args.log_level)

    counts = {status: sum(s["status"] == status for s in summaries) for status in ("ran", "reused", "failed")}
    logger.info("Results: %s", counts)
