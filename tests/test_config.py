"""Checks the config tree: defaults, JSON, overrides and errors."""

import json

import pytest

pytest.importorskip("jspam")

from galactic_tidal_constraints.simr import config as config_module  # noqa: E402
from galactic_tidal_constraints.simr.config import from_dict, load_config  # noqa: E402

MODEL = [1.5, 1.0, 0.5, -0.5, 0.5, 0.3, 1.0, 1.0, 1.0, 1.0, 30.0, 60.0, 20.0, 40.0, 0.3, 0.3] + [0.0] * 6


def test_defaults():
    config = from_dict({})
    assert config.run.seed == 0
    assert config.models == ()
    assert config.simulator[0].lnl == 0.001
    assert config.simulator[0].endings.values == (1.0,)


def test_one_simulator_or_a_list():
    one = from_dict({"simulator": {"n1": 5}})
    many = from_dict({"simulator": [{"n1": 5}, {"n1": 6}]})
    assert [s.n1 for s in one.simulator] == [5]
    assert [s.n1 for s in many.simulator] == [5, 6]


def test_nested_endings():
    config = from_dict({"simulator": {"endings": {"unit": "gyr", "values": [-0.1, 0, 0.1]}}})
    assert config.simulator[0].endings.unit == "gyr"
    assert config.simulator[0].endings.values == (-0.1, 0.0, 0.1)


@pytest.mark.parametrize(
    "raw",
    [
        {"typo": 1},
        {"run": {"sede": 1}},
        {"simulator": {"lnll": 0.1}},
        {"simulator": {"endings": {"unit": "fraction", "value": [1]}}},
        {"run": {"seed": "one"}},
        {"run": {"seed": True}},
        {"run": {"seed": 1.5}},
        {"run": {"seed": -1}},
        {"run": {"save": "some"}},
        {"simulator": {"n1": 1.5}},
        {"simulator": {"lnl": "big"}},
        {"simulator": {"kind": "image"}},
        {"simulator": []},
        {"models": [[1.0, 2.0]]},
        {"models": "file.txt"},
    ],
)
def test_unknown_keys_and_wrong_types_are_errors(raw):
    with pytest.raises(ValueError):
        from_dict(raw)


def test_integers_are_accepted_as_floats():
    assert from_dict({"simulator": {"m_halo": 5}}).simulator[0].m_halo == 5.0


def test_file_then_overrides(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(
        json.dumps({"run": {"seed": 1, "name": "a"}, "models": [MODEL], "simulator": {"n1": 5, "n2": 6}})
    )
    config = load_config(path, ["run.seed=3", "simulator.n1=500", "simulator.endings.values=[0.5,1.0]"])
    assert config.run.seed == 3
    assert config.run.name == "a"
    assert config.simulator[0].n1 == 500
    assert config.simulator[0].n2 == 6
    assert config.simulator[0].endings.values == (0.5, 1.0)


def test_override_with_no_file_or_simulator_section():
    assert load_config(None, ["simulator.n1=7"]).simulator[0].n1 == 7


def test_override_under_a_list(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"simulator": [{"n1": 1}, {"n1": 2}]}))
    every = load_config(path, ["simulator.lnl=0.5"])
    one = load_config(path, ["simulator.1.lnl=0.5"])
    assert [s.lnl for s in every.simulator] == [0.5, 0.5]
    assert [s.lnl for s in one.simulator] == [0.001, 0.5]
    with pytest.raises(ValueError):
        load_config(path, ["simulator.5.lnl=0.5"])


def test_string_override_and_bad_override():
    assert load_config(None, ["run.name=hello"]).run.name == "hello"
    assert load_config(None, ["simulator.normalization=total"]).simulator[0].normalization == "total"
    with pytest.raises(ValueError):
        load_config(None, ["run.seed"])
    with pytest.raises(ValueError):
        load_config(None, ["run.sede=1"])


def test_saved_config_reads_back_the_same():
    config = from_dict(
        {"models": [MODEL], "simulator": [{"n1": 5, "endings": {"unit": "step", "values": [0, 10]}}]}
    )
    assert config_module.from_dict(config.to_dict()) == config


def test_example_config_is_valid():
    from pathlib import Path

    example = Path(__file__).parent.parent / "examples" / "simr" / "config.json"
    config = load_config(example)
    assert len(config.models) >= 1
    assert config.simulator[0].endings.unit == "fraction"
