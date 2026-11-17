"""The command line for SIMR.

    python -m galactic_tidal_constraints.simr --models models.txt --output runs/ [--config config.json]
        [--mode solo|local|mpi] [--workers N] [--log-level INFO]

With MPI (`uv sync --group mpi`), rank 0 runs the command and every other rank is a worker:

    mpirun -n N python -m mpi4py.futures -m galactic_tidal_constraints.simr --mode mpi --models ... --output ...

Exit codes: 0 when the run ran (even if some simulations failed: they're in the catalog and the log), 1 when
it couldn't run or finish (a bad models file, config or output, or a worker that died), 2 when it ran but
every simulation failed. Under MPI a worker that crashes usually ends the whole job instead; a rerun resumes.
"""

import argparse
import json
import logging
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor

from galactic_tidal_constraints.simr import simr

logger = logging.getLogger("galactic_tidal_constraints.simr")

RAN, COULD_NOT_RUN, ALL_FAILED = 0, 1, 2


def main(argv: list[str] | None = None) -> int:
    """Run SIMR from command-line arguments (`sys.argv` by default). Returns the exit code."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.workers is not None and args.mode != "local":
        parser.error("--workers is for --mode local")
    simr.configure_logging(args.log_level, use_rank=args.mode == "mpi")
    try:
        models = simr.read_models(args.models)
        request = _read_config(args.config)
        if args.mode == "local":
            result = _run_local(models, request, args)
        elif args.mode == "mpi":
            result = _run_mpi(models, request, args)
        else:
            result = simr.run(models, request, output=args.output)
    except (OSError, ValueError, ImportError, simr.SimrRunError) as error:  # a short message, no traceback
        logger.error("%s", error)
        return COULD_NOT_RUN

    status = result.status["sim"]
    logger.info("the run is recorded in %s", simr.output_paths(args.output)[0])
    if status.size and (status == "failed").all():
        logger.error("every simulation failed; see the errors above or in the catalog")
        return ALL_FAILED
    return RAN


def _run_local(models, request, args) -> simr.SimrResult:
    """The run on a pool of processes on this machine, created once and closed at the end. Workers are
    started fresh ("spawn"), as MPI's are, and set up their logging as they start."""
    workers = args.workers or simr.available_cpus()
    logger.info("local mode: %d workers", workers)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=simr.configure_logging, initargs=(args.log_level, False)) as pool:
        return simr.run(models, request, output=args.output, map=pool.map)


def _run_mpi(models, request, args) -> simr.SimrResult:
    """The run on MPI ranks, launched with `mpirun -n N python -m mpi4py.futures -m ...`: rank 0 runs this
    and the other N-1 ranks wait for tasks. Workers set up their logging as they start."""
    try:
        from mpi4py import MPI
        from mpi4py.futures import MPIPoolExecutor
    except ImportError:
        raise ImportError("--mode mpi needs mpi4py: uv sync --group mpi") from None
    launch = "mpirun -n N python -m mpi4py.futures -m galactic_tidal_constraints.simr --mode mpi ..."
    ranks, rank = MPI.COMM_WORLD.Get_size(), MPI.COMM_WORLD.Get_rank()
    if rank != 0:  # with -m mpi4py.futures, only rank 0 runs this
        raise ValueError(f"rank {rank} is running the command: launch with {launch}")
    if ranks < 2:
        raise ValueError(f"--mode mpi needs 2 or more ranks (1 runs the command, the rest are workers): "
                         f"launch with {launch}")
    logger.info("mpi mode: %d workers", ranks - 1)
    with MPIPoolExecutor(initializer=simr.configure_logging, initargs=(args.log_level,)) as pool:
        return simr.run(models, request, output=args.output, map=pool.map)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m galactic_tidal_constraints.simr",
        description="Run SPAM simulations for many models and record them in a catalog.",
    )
    parser.add_argument("--models", required=True,
                        help="text file, one model per line: the first 14 comma-separated values")
    parser.add_argument("--output", required=True,
                        help="a folder (catalog.json, plus model folders for run.save) or a .json catalog file")
    parser.add_argument("--config", help="JSON request, listing only what differs from the defaults")
    parser.add_argument("--mode", default="solo", choices=["solo", "local", "mpi"],
                        help="solo: one process (the default); local: a pool of processes on this machine; "
                             "mpi: MPI ranks, launched with mpirun ... python -m mpi4py.futures -m ...")
    parser.add_argument("--workers", type=int,
                        help="with --mode local: how many processes (default: the CPUs this job may use)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def _read_config(path: str | None) -> dict:
    """The request from a JSON file, or the defaults with none. run.return is dropped: nothing comes back
    in memory here, and dropping it lets one config work in notebooks and here. Only a non-empty one is
    warned about; an empty list asks for nothing, so nothing is lost."""
    if path is None:
        return {}
    with open(path, encoding="utf-8") as f:
        try:
            request = json.load(f)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path} isn't valid JSON: {error}") from None
    if not isinstance(request, dict):
        raise ValueError(f"{path} must hold a JSON object (the request), not {type(request).__name__}")
    run_section = request.get("run")
    if isinstance(run_section, dict) and "return" in run_section:
        if run_section["return"]:
            logger.warning("run.return is ignored on the command line: nothing is returned in memory")
        request = {**request, "run": {key: value for key, value in run_section.items() if key != "return"}}
    return request


if __name__ == "__main__":
    sys.exit(main())
