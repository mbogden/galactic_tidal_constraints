"""The config tree for a run. Precedence: defaults here < JSON file < `key=value` overrides.

Fields are described in docs/configuration.md.
"""

import copy
import json
import typing
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

from galactic_tidal_constraints.simr.sim import SimulatorConfig

SAVE_OPTIONS = ("all", "none")
N_MODEL_VALUES = 22


@dataclass(frozen=True)
class RunConfig:
    """Values for the whole run.

    Attributes:
        name: a label for the run.
        seed: each model's particle seed comes from this and the model's index, never its MPI rank.
        save: "all" saves each simulation to the output folder, and reuses it next time. "none"
            saves nothing and returns the simulations in memory.
    """

    name: str = "run"
    seed: int = 0
    save: str = "all"

    def __post_init__(self) -> None:
        if self.save not in SAVE_OPTIONS:
            raise ValueError(f"run.save must be one of {SAVE_OPTIONS}: {self.save!r}")
        if self.seed < 0:
            raise ValueError(f"run.seed can't be negative: {self.seed}")


@dataclass(frozen=True)
class Config:
    """A run: its models and the simulators to run on each. Every model runs on every simulator."""

    models: tuple[tuple[float, ...], ...] = ()
    simulator: tuple[SimulatorConfig, ...] = (SimulatorConfig(),)
    run: RunConfig = field(default_factory=RunConfig)

    def __post_init__(self) -> None:
        for i, model in enumerate(self.models):
            if len(model) != N_MODEL_VALUES:
                raise ValueError(f"models[{i}] has {len(model)} values, not {N_MODEL_VALUES}")
        if len(self.simulator) == 0:
            raise ValueError("simulator can't be an empty list")

    def to_dict(self) -> dict:
        """Plain JSON types, so it can be saved and read back."""
        return json.loads(json.dumps(asdict(self)))


def _build(kind: typing.Any, value: typing.Any, where: str) -> typing.Any:
    """Build a value of type `kind` from JSON, or raise a ValueError naming the field."""
    if is_dataclass(kind):
        if not isinstance(value, dict):
            raise ValueError(f"{where} must be an object, not {value!r}")
        hints = typing.get_type_hints(kind)
        names = {f.name for f in fields(kind)}
        unknown = set(value) - names
        if unknown:
            raise ValueError(
                f"unknown key(s) in {where or 'config'}: {sorted(unknown)}. Allowed: {sorted(names)}"
            )
        args = {key: _build(hints[key], item, f"{where}.{key}".lstrip(".")) for key, item in value.items()}
        try:
            return kind(**args)
        except ValueError as error:
            raise ValueError(f"{where or 'config'}: {error}") from error

    if typing.get_origin(kind) is tuple:
        if not isinstance(value, list):
            raise ValueError(f"{where} must be a list, not {value!r}")
        inner = typing.get_args(kind)[0]
        return tuple(_build(inner, item, f"{where}[{i}]") for i, item in enumerate(value))

    if isinstance(value, bool) or (kind is float and not isinstance(value, (int, float))):
        raise ValueError(f"{where} must be a {kind.__name__}, not {value!r}")
    if kind is float:
        return float(value)
    if not isinstance(value, kind):
        raise ValueError(f"{where} must be a {kind.__name__}, not {value!r}")
    return value


def from_dict(raw: dict) -> Config:
    """Build a checked `Config`. Unknown keys and wrong types are errors.

    `simulator` can be one object or a list of them; either way the result is a list.
    """
    raw = copy.deepcopy(raw)
    if isinstance(raw.get("simulator"), dict):
        raw["simulator"] = [raw["simulator"]]
    return _build(Config, raw, "")


def _apply_override(raw: dict, override: str) -> None:
    """Apply one `key=value`, e.g. `run.seed=3`, `simulator.n1=500` or `simulator.1.lnl=0.01`.

    Under a list, a plain key sets every entry and a number picks one entry.
    """
    if "=" not in override:
        raise ValueError(f"an override must look like key=value: {override!r}")
    path, text = override.split("=", 1)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = text

    keys = path.split(".")

    def set_in(node: typing.Any, keys: list[str]) -> None:
        key = keys[0]
        if isinstance(node, list):
            if key.isdigit():
                if int(key) >= len(node):
                    raise ValueError(f"override {override!r}: index {key} is past the end of the list")
                node = [node[int(key)]]
                keys = keys[1:]
                if not keys:
                    raise ValueError(f"override {override!r} must name a key inside the list entry")
            for item in node:
                set_in(item, keys)
            return
        if len(keys) == 1:
            node[key] = value
            return
        if key == "simulator" and isinstance(node.get(key), dict):
            node[key] = [node[key]]
        set_in(node.setdefault(key, {}), keys[1:])

    set_in(raw, keys)


def load_config(path: str | Path | None = None, overrides: typing.Iterable[str] = ()) -> Config:
    """Merge the JSON file (if any) and the overrides over the defaults, and check the result."""
    raw = json.loads(Path(path).read_text()) if path is not None else {}
    for override in overrides:
        _apply_override(raw, override)
    return from_dict(raw)
