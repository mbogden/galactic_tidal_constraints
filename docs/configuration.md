# Configuration

One config tree holds everything that can change a result. Precedence: defaults in code < JSON file <
command-line `key=value` overrides. Unknown keys and wrong types are errors. Each run saves its full merged
config in `run_config.json`.

So far only `run`, `models` and `simulator` exist; `image` and `fitness` come with their stages. A full
example is in [examples/simr/config.json](../examples/simr/config.json).

## `run`
| Field | Default | Meaning |
|---|---|---|
| `name` | `"run"` | A label |
| `seed` | `0` | Each model's particle seed comes from this and the model's index |
| `save` | `"all"` | `"all"` saves to the output folder and reuses what's there; `"none"` saves nothing and returns results in memory |

## `models`
A list of models, each the 22 SPAM values in model-string order:
`x, y, z, vx, vy, vz, mass1, mass2, rout1, rout2, phi1, phi2, theta1, theta2, epsilon1, epsilon2`, then
`rscale1` (3 values) and `rscale2` (3 values).

## `simulator`
One entry, or a list of them. Every model runs on every entry.

| Field | Default | Meaning |
|---|---|---|
| `kind` | `"particles"` | `"particles"` saves all particles at each ending; `"orbit"` saves the secondary's center at every step |
| `n1`, `n2` | `1000` | Particles in the primary's and secondary's disks |
| `heat1`, `heat2` | `0.0` | Random motion added to each disk's particles |
| `heat_mode` | `"wallin"` | `"wallin"` also adds a drift; `"centered"` has the same spread with no drift |
| `potential` | `"extended"` | `"extended"` (halo, disk, bulge, dynamical friction) or `"point_mass"` |
| `lnl` | `0.001` | Strength of dynamical friction |
| `r_scale` | `10.0` | Halo radial scale |
| `m_halo` | `5.8` | Halo mass |
| `m_bulge` | `0.3333` | Bulge mass |
| `h_bulge` | `0.2` | Bulge scale length |
| `h_disk` | `1.0` | Disk scale length; the disk's mass grows with it |
| `normalization` | `"wallin"` | `"wallin"` divides by a fixed 7.13333, so mass values change the total mass; `"total"` divides by the actual total |
| `endings.unit` | `"fraction"` | `"fraction"`, `"gyr"`, `"time"` or `"step"`; see [design/sim.md](design/sim.md#many-endings) |
| `endings.values` | `[1.0]` | When to save, ascending |

## Overrides
```bash
python -m galactic_tidal_constraints.simr --config run.json --output out \
    run.seed=3 simulator.n1=500 simulator.1.lnl=0.01 'simulator.endings.values=[0.5,1.0]'
```
Under a list, a plain key sets every entry, and a number picks one.

## Runtime flags
These don't change results, so they stay outside the tree: `--config`, the overrides, `--output`, `--mode`,
`--workers` and `--log-level`.
