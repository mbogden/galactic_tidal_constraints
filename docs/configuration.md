# Configuration

How to tell `simr` what to simulate and image: the config's sections, how its lists combine, what's
checked and what's recorded. The fields themselves are listed in each stage's doc, linked below.
Decisions behind it are in [PLAN.md](../PLAN.md) (Configuration).

## What a config is
A run has two inputs:
- **The models** say *what* to simulate: one model per line in a text file (`--models`), or an
  `(N, 14)` array in Python ([modules/simr.md](modules/simr.md#models)).
- **The config** says *how*: a JSON file (`--config`), or the same thing as a dict in Python (the
  "request"). It lists **only what differs from the defaults**; everything missing takes its default.

Either can change while the other stays fixed. There are no command-line overrides: the config file is
the whole run, and a small change is a new config file. Flags that don't change results stay outside
it: `--models`, `--output`, `--mode`, `--workers` and `--log-level`
([modules/simr.md](modules/simr.md#the-command-line)).

## The sections
| Section | Holds | Missing means | Fields |
|---|---|---|---|
| `run` | `return`: data to hand back in memory (Python only). `save`: data to write to the output folder. Both are lists of names (below) | Nothing returned or saved | [modules/simr.md](modules/simr.md#data-names) |
| `simulator` | One `SimulatorConfig` entry, or a list of them | One default entry | [modules/simulation.md](modules/simulation.md#simulatorconfig) |
| `image` | One `ImageConfig` entry, or a list of them | One default entry | [modules/image_creator.md](modules/image_creator.md#imageconfig) |

The `sim_metrics` and `img_metrics` sections come with the metrics stage. What each simulator setting does physically is in
[spam_physics.md](spam_physics.md#the-settings-we-expose-simulatorconfig).

### Data names
`run.return` and `run.save` take any of these. `simr` computes only what they name (and what that
needs), so asking only for `sim.orbit` never runs particles.

| Name | Holds | Notes |
|---|---|---|
| `sim.particles` | Every particle at the end of the run | |
| `sim.unperturbed` | The starting disks placed around today's centers, as if the galaxies never met | |
| `sim.orbit` | The secondary's center at every step | Fast: no particles |
| `sim.timing` | When the run starts, and why (a dict) | Always recorded in `model.json` when a simulation is saved; it has no `.npy` file |
| `img.model` | The model image | Needs `sim.particles`, which is made (or loaded) even if not asked for |
| `img.unperturbed` | The unperturbed image, in the same frame | Needs `sim.unperturbed` |

Shapes and details: [modules/simr.md](modules/simr.md#data-names).

On the command line `run.return` is ignored, as nothing comes back in memory there, so one config works
in a notebook and on the command line. A non-empty `run.return` gets a warning; an empty one doesn't.

## Lists and combinations
- **Every model runs on every simulator entry, and every simulation is imaged with every image entry.**
  So 2 models, 2 simulator entries and 2 image entries make 4 simulations and 8 images.
- A list can appear only directly under a section; anything deeper is an ordinary value, like an image
  center.
- **An entry's position is its name:** simulator entry 0 is `sim_00`, image entry 1 is `img_0001`. Data
  is indexed the same way: `[model, simulator]` for `sim.*`, `[model, simulator, image]` for `img.*`
  (later, metrics: `[model, simulator, image, metric]`, see [architecture.md](architecture.md#metric-array)).
- Combinations multiply quickly; keeping them to a sensible size is up to whoever writes the config.

## Checks
Everything is checked before anything runs, and an error says where the problem is:

| Mistake | Error |
|---|---|
| An unknown section or field | `unknown keys in simulator[1]: ['sed']. Known: ['seed', 'n1', …]` |
| An unknown data name | `unknown names in run.save: ['sim.orbits']. Known: ['sim.particles', …]` |
| A name instead of a list | `run.save must be a list of names, e.g. ["sim.orbit"]` |
| A wrong type | `simulator[0].lnl must be a number: '0.1'` |
| A bad value | `simulator[0]: lnl can't be negative: -1` |
| An empty list | `simulator is an empty list` |

Types follow JSON:
- A whole number is fine for a number field (JSON may write `10.0` as `10`), but a field that takes a
  whole number (`n1`, `seed`) rejects `1000.0`.
- `true` and `false` are never numbers.
- A pair, such as an image center, is a list of 2: `[48, 64]`.

## What's recorded
- With an output, the run's record in `catalog.json` holds every entry's **full** config, defaults
  filled in, so the output says exactly what ran even when the config file listed almost nothing.
- Each model's `model.json` records the full config of every simulation and image entry it saved.
- A saved simulation or image is reused only when its config matches **exactly**; a different config
  is neither used nor changed ([modules/simr.md](modules/simr.md#reuse)).

## Examples
The files are in [examples/configs/](../examples/configs/). A test checks that each one passes `simr`'s
checks, and that `defaults.json` still equals the code's defaults (`tests/test_example_configs.py`).

**Compare two dynamical-friction strengths**, saving orbits, particles and two kinds of images
([compare_lnl.json](../examples/configs/compare_lnl.json)):
```json
{
  "run": {"save": ["sim.orbit", "sim.particles", "img.model"]},
  "simulator": [
    {"lnl": 0.1},
    {"lnl": 0.001}
  ],
  "image": [
    {},
    {"blur_sigma": 1.0, "brightness": "log"}
  ]
}
```
```bash
python -m galactic_tidal_constraints.simr --models examples/configs/models.txt \
  --config examples/configs/compare_lnl.json --output runs
```
With the two models in [models.txt](../examples/configs/models.txt): 4 simulations (`sim_00` with
`lnl` 0.1, `sim_01` with Wallin's 0.001) and 8 images (`img_0000` plain, `img_0001` blurred, log
brightness).

**Every field at its default** ([defaults.json](../examples/configs/defaults.json)). Running with it
is the same as running with no config at all; it's here to show what can be set. The empty `run` lists
take the names [above](#data-names):
```json
{
  "run": {"return": [], "save": []},
  "simulator": {
    "seed": 0, "n1": 1000, "n2": 1000, "heat1": 0.0, "heat2": 0.0, "heat_mode": "centered",
    "lnl": 0.1, "r_scale": 10.0, "m_halo": 5.8, "m_bulge": 0.3333, "h_bulge": 0.2, "h_disk": 1.0,
    "normalization": "total", "start": "pass", "start_pass": 1, "no_pass": -5.0
  },
  "image": {
    "width": 128, "height": 128, "primary_center": [48.0, 64.0], "secondary_center": [80.0, 64.0],
    "blur_sigma": 0.0, "blur_truncate": 4.0, "brightness": "linear"
  }
}
```

## From Python
The same config, as a dict, is `simr.run`'s request. Entries can also be `SimulatorConfig` or
`ImageConfig` objects, and `run.return` works here:
```python
from galactic_tidal_constraints.simr import simr
from galactic_tidal_constraints.simr.simulation import SimulatorConfig

models = simr.read_models("examples/configs/models.txt")
result = simr.run(models, {
    "run": {"return": ["sim.orbit"]},
    "simulator": [SimulatorConfig(lnl=0.1), {"lnl": 0.001}],
})
result.data["sim.orbit"][1, 1]      # model 1's orbit with lnl 0.001
```
