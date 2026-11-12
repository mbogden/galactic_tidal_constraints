"""Tests for the SIMR command line (simr/__main__.py)."""

import json
import subprocess
import sys

import pytest

# Skipped where the dependencies aren't installed, such as native Windows.
np = pytest.importorskip("numpy")
jspam = pytest.importorskip("jspam")

from galactic_tidal_constraints.simr import simr  # noqa: E402
from galactic_tidal_constraints.simr.__main__ import main  # noqa: E402

# Model SDSS 587722984435351614 (as in test_simr.py), and the same approaching now and unbound: no pass
# before now, so with no_pass="error" it fails.
PARAMS = [-9.93853, -4.5805, 15.43348, -1.54025, -2.97899, 4.69234, 34.41671, 29.49607, 2.97913, 4.2501,
          301.0, 35.5, 310.80745, 321.98757]
MODEL = ",".join(str(value) for value in PARAMS)
FIRST_APPROACH = ",".join(str(value * -3 if 3 <= i < 6 else value) for i, value in enumerate(PARAMS))
SMALL = {"n1": 20, "n2": 10}


@pytest.fixture
def files(tmp_path):
    """Write a models file and a config; return a function that builds the arguments."""

    def make(models=(MODEL,), config=None, output="runs"):
        (tmp_path / "models.txt").write_text("\n".join(models) + "\n")
        args = ["--models", str(tmp_path / "models.txt"), "--output", str(tmp_path / output)]
        if config is not None:
            (tmp_path / "config.json").write_text(json.dumps(config))
            args += ["--config", str(tmp_path / "config.json")]
        return args

    return make


def test_a_run_records_its_catalog(tmp_path, files):
    assert main(files(config={"simulator": SMALL})) == 0
    catalog = simr.load_catalog(tmp_path / "runs")
    assert len(catalog["models"]) == 1
    assert catalog["runs"][0]["status"] == {"sim": [["empty"]]}  # nothing saved: nothing to do (until fitness)


def test_the_config_is_optional(tmp_path, files):
    assert main(files()) == 0  # no --config: one default simulator entry
    [record] = simr.load_catalog(tmp_path / "runs")["runs"]
    assert record["simulator"][0]["n1"] == 1000  # the full defaults are recorded


def test_save_writes_model_data(tmp_path, files):
    assert main(files(config={"run": {"save": ["sim.orbit"]}, "simulator": SMALL})) == 0
    assert simr.load_simulation(tmp_path / "runs" / "model_000000", "sim_00")["sim.orbit"].shape[1] == 7


def test_a_json_output(tmp_path, files):
    assert main(files(config={"simulator": SMALL}, output="gen_01.json")) == 0
    assert simr.load_catalog(tmp_path / "gen_01.json")["runs"][0]["model_ids"] == ["000000"]


def test_run_return_is_ignored_with_a_warning(tmp_path, files, caplog):
    config = {"run": {"return": ["sim.orbit"], "save": ["sim.timing"]}, "simulator": SMALL}
    assert main(files(config=config)) == 0
    assert "run.return is ignored on the command line" in caplog.text
    assert simr.load_catalog(tmp_path / "runs")["runs"][0]["run"] == {"return": [], "save": ["sim.timing"]}


# Until fitness exists, a run computes only what run.save asks for, so these save sim.timing (the cheapest).
FAILING = {"run": {"save": ["sim.timing"]}, "simulator": {**SMALL, "no_pass": "error"}}


def test_some_failures_still_exit_0(files):
    config = FAILING
    assert main(files(models=(MODEL, FIRST_APPROACH), config=config)) == 0


def test_every_simulation_failing_exits_2(files, caplog):
    assert main(files(models=(FIRST_APPROACH,), config=FAILING)) == 2
    assert "every simulation failed" in caplog.text


@pytest.mark.parametrize(
    "change, message",
    [
        ({"models": ("1,2,3",)}, "expected at least 14"),
        ({"config": {"simulator": {"lnl": "0.1"}}}, r"simulator\[0\]\.lnl must be a number"),
        ({"config": {"run": {"save": ["sim.orbit"]}}, "output": "gen.json"}, "needs an output folder"),
        ({"output": "runs.txt"}, "a folder .no extension. or a .json catalog file"),
        ({"config": ["not", "a", "request"]}, "must hold a JSON object"),
    ],
)
def test_bad_inputs_exit_1_with_a_short_message(files, caplog, change, message):
    assert main(files(**change)) == 1
    [error] = [record for record in caplog.records if record.levelname == "ERROR"]
    assert error.exc_info is None  # a message, not a traceback
    assert pytest.importorskip("re").search(message, error.getMessage())


def test_a_missing_file_exits_1(tmp_path, caplog):
    assert main(["--models", str(tmp_path / "nope.txt"), "--output", str(tmp_path / "runs")]) == 1
    assert "nope.txt" in caplog.text


def test_invalid_json_exits_1(tmp_path, files, caplog):
    args = files()
    (tmp_path / "config.json").write_text("{not json")
    assert main(args + ["--config", str(tmp_path / "config.json")]) == 1
    assert "isn't valid JSON" in caplog.text


def test_missing_flags_are_an_argparse_error(capsys):
    with pytest.raises(SystemExit) as stop:
        main(["--models", "models.txt"])
    assert stop.value.code == 2  # argparse's own code for a usage error
    assert "--output" in capsys.readouterr().err


def test_python_dash_m(tmp_path, files):
    result = subprocess.run([sys.executable, "-m", "galactic_tidal_constraints.simr", *files(config={"simulator": SMALL})],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "the run is recorded in" in result.stderr


def test_local_mode(tmp_path, files, caplog):
    config = {"run": {"save": ["sim.orbit"]}, "simulator": [SMALL, {**SMALL, "lnl": 0.01}]}
    with caplog.at_level("INFO"):
        assert main(files(models=(MODEL, FIRST_APPROACH), config=config) + ["--mode", "local", "--workers", "2"]) == 0
    assert "local mode: 2 workers" in caplog.text
    [record] = simr.load_catalog(tmp_path / "runs")["runs"]
    assert record["status"] == {"sim": [["ran", "ran"], ["ran", "ran"]]}
    solo = tmp_path / "solo"
    main(files(models=(MODEL, FIRST_APPROACH), config=config, output="solo"))
    np.testing.assert_array_equal(simr.load_simulation(tmp_path / "runs" / "model_000001", "sim_01")["sim.orbit"],
                                  simr.load_simulation(solo / "model_000001", "sim_01")["sim.orbit"])


def test_workers_needs_local_mode(files, capsys):
    with pytest.raises(SystemExit) as stop:
        main(files() + ["--workers", "2"])
    assert stop.value.code == 2
    assert "--workers is for --mode local" in capsys.readouterr().err
