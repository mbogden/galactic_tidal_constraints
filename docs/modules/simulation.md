# Simulator stage (`simr/simulation.py`)

The Simulator stage runs one SPAM simulation. It takes values and returns arrays: 
no saving/reading to disk space, no pools or MPI, no state kept between calls. 
It's the only module that imports `jspam` (`PLAN.md`, One door to SPAM), 
and every call passes the whole config through, so nothing carries over from one run to the
next. `simr` runs it for many models, as the `sim.*` items ([simr.md](simr.md)).

```python
from galactic_tidal_constraints.simr import simulation as sim
from galactic_tidal_constraints.simr.simulation import SimulatorConfig

params = sim.parse_model_string("-9.93853,-4.5805,15.43348,-1.54025,-2.97899,4.69234,34.41671,"
                                "29.49607,2.97913,4.2501,301.0,35.5,310.80745,321.98757")
config = SimulatorConfig(n1=2000, n2=2000)

particles = sim.simulate_particles(params, config)   # (4001, 6): every particle at the end
orbit = sim.simulate_orbit(params, config)           # (n_steps + 1, 7): the secondary's path
timing = sim.get_timing(params, config)              # when the run starts, and why
```

Only what exists today is described here. What each setting does physically, and why our defaults
differ from Wallin's, is in [spam_physics.md](../spam_physics.md). Decisions behind it are in
[PLAN.md](../../PLAN.md) (Physics fixes; Models and stage lists).

## A model
A model is a NumPy array of shape `(14,)`: SPAM's first 14 parameters, in SPAM's order.

| Name | |
|---|---|
| `x`, `y`, `z`, `vx`, `vy`, `vz` | The secondary's position and velocity relative to the primary, now |
| `mass1`, `mass2` | The two galaxies' masses |
| `rout1`, `rout2` | How far out each disk's particles are placed |
| `phi1`, `phi2`, `theta1`, `theta2` | The two disks' orientations, in degrees |

- `sim.PARAM_NAMES` lists them in order, and `sim.PARAM_INDEX` maps a name to its position.
- SPAM takes 22 values, but the last 8 (`epsilon1/2` and six `rscale`) do nothing in the extended
  potential, the only one `sim` runs. `sim` appends `sim.UNUSED_PARAMS` for them before calling `jspam`.
  Why: [spam_physics.md](../spam_physics.md#parameters-that-do-nothing).
- `parse_model_string(text)` reads the first 14 comma-separated values of a SPAM model string and
  ignores the rest, so 22-value strings and the 34-value lines of JSPAM's `archive/data.txt` both work.
- Every function takes the model as an array or a list. Any other length is a `ValueError` (a
  22-value array too; use `parse_model_string`).

## `SimulatorConfig`
Everything that sets a result, apart from the model. Frozen, and checked when made: a bad value is a
`ValueError`. What each field does physically: [spam_physics.md](../spam_physics.md#the-settings-we-expose-simulatorconfig).

| Field | Default | Meaning |
|---|---|---|
| `seed` | 0 | Where the disk particles start. The same seed gives exactly the same particles, and every model shares one draw |
| `n1`, `n2` | 1000, 1000 | Particles in the primary's and secondary's disks |
| `heat1`, `heat2` | 0, 0 | Random motion added to each disk's particle velocities |
| `heat_mode` | `"centered"` | `"centered"`: kicks in [−heat/2, heat/2). `"wallin"`: kicks in [0, heat), so a heated disk drifts |
| `lnl` | 0.1 | Strength of dynamical friction. Provisional (Wallin's is 0.001) |
| `r_scale`, `m_halo`, `m_bulge`, `h_bulge`, `h_disk` | 10, 5.8, 0.3333, 0.2, 1.0 | The unit galaxy both galaxies are built from (Wallin's values) |
| `normalization` | `"total"` | `"total"`: the unit galaxy's mass is always 1. `"wallin"`: divided by a fixed 7.13333 |
| `start` | `"pass"` | When the run starts: `"pass"`, `"wallin"` (SPAM's own search) or a negative time (below) |
| `start_pass` | 1 | With `"pass"`: which close pass, 1 being the most recent |
| `no_pass` | −5 | With `"pass"`, when that pass doesn't exist back to t = −30: start at this time, or `"error"` to raise a `ValueError` |

The defaults differ from Wallin's only in `heat_mode`, `lnl`, `normalization` and `start`
([spam_physics.md](../spam_physics.md#the-settings-we-expose-simulatorconfig) has both columns).

## Functions
| Function | Returns |
|---|---|
| `simulate_particles(params, config)` | Every particle at the end of the run, `(n1 + n2 + 1, 6)` |
| `initial_state(params, config)` | Every particle at the start of the run, before any step, `(n1 + n2 + 1, 6)` |
| `unperturbed_particles(params, config)` | The same starting disks placed around today's centers, as if the galaxies had never met, `(n1 + n2 + 1, 6)` |
| `simulate_orbit(params, config)` | The secondary's center at the start and after every step, `(n_steps + 1, 7)` |
| `get_timing(params, config)` | When the run starts and how many steps it takes, as a dict (below), without running it |
| `search_orbit(params, config)` | The secondary's center from now back to t = −30, one row per step, about `(30001, 7)` |
| `find_passes(params, config)` | Every close pass on that path, most recent first, as a list of dicts (below) |
| `mass_profile(config)` | The unit galaxy's tables, a dict of `(10000,)` arrays (below) |
| `jspam_version()` | Where the installed `jspam` came from: `{"url", "commit"}` (`commit` is `None` for a local folder) |
| `parse_model_string(text)` | A model, `(14,)` |

`config` is optional everywhere; it defaults to `SimulatorConfig()`.

### Arrays
- **Particles** (`simulate_particles`, `initial_state`, `unperturbed_particles`): columns `x, y, z,
  vx, vy, vz`. Rows 0 to n1 − 1 are the primary's disk, the next n2 the secondary's, and the last row
  is the secondary's center. The primary's center stays at the origin, so it has no row.
- **Orbits** (`simulate_orbit`, `search_orbit`): columns `x, y, z, vx, vy, vz, t`, for the secondary's
  center; the primary stays at the origin. `simulate_orbit` runs forward from the start;
  `search_orbit` runs backward from now (row 0 is the model as given).
- Times are in SPAM units, now = 0 (one unit is about 87 Myr). A run ends 1 to 2 steps past now.

### What lines up with what
- `simulate_orbit`'s last row is exactly where `simulate_particles` leaves the secondary's center: the
  centers' motion doesn't depend on the particles, the heat or the seed. It does depend on the mass model.
- `initial_state` is exactly the state `simulate_particles` starts from, with the same config.
- `unperturbed_particles` has the same disks as `initial_state` (same seed, heat and mass model),
  placed around the centers as they are now, so it lines up with `simulate_particles` with no
  translating. Its start settings don't matter, and no start-time search runs.
- The same model and config give the same arrays every time.

### The start
`get_timing`, `simulate_orbit` and the particle functions all choose the start the same way, from
`config.start`:

| `start` | The run starts | `get_timing`'s `"start"` |
|---|---|---|
| `"pass"` | `sim.LEAD_FACTOR` (10) × `rout1` / speed before the chosen close pass (`start_pass`), found by `find_passes` | `"pass"`, or `"fallback"` when there's no such pass and `no_pass` is a time |
| `"wallin"` | Where SPAM's own search puts it, including its −5 fallback | `"wallin"` |
| a negative number | Exactly then | `"fixed"` |

Why `"pass"` is the default: [spam_physics.md](../spam_physics.md#the-start-time).

`get_timing` returns:

| Key | |
|---|---|
| `"t_start"`, `"n_steps"` | The start time a run uses, and its steps to the end |
| `"start"`, `"start_pass"` | How the start was chosen (table above), and which pass (or `None`) |
| `"t_closest"`, `"closest_distance"`, `"closest_speed"` | The pass the start is based on (with `"wallin"`, the closest pass SPAM's search found; `NaN` for a fixed or fallback start) |
| `"start_fallback"`, `"fallback_reason"` | Whether a fallback replaced the start (`no_pass`, or SPAM's −5), and why there was no pass |
| `"t_start_found"` | With `"wallin"`, SPAM's start before its fallback; otherwise `t_start` |

`find_passes` returns one dict per pass: `"pass"` (1 is the most recent), `"t"`, `"distance"` and
`"speed"`. A pass is a local minimum of the distance between the centers along `search_orbit`; the
window's two ends aren't passes, so a pair still approaching now has no pass at t = 0.

### `mass_profile`
The unit galaxy as SPAM reads it, for a galaxy of mass 1: `"radius"` (SPAM units, 0.001 apart, to 10),
the enclosed mass of `"halo"`, `"disk"`, `"bulge"` and `"total"`, and `"force"`, the pull at that
radius. Multiply the masses and force by `mass1` or `mass2` for a real galaxy. Only the mass-model
fields and `normalization` matter.

## Costs
Measured in the project's image on a desktop CPU (i9-13900KF), one process, default config, for the
model above (6,916 steps; 2026-11-17). Best of five:

| Call | Time |
|---|---|
| `simulate_particles`, n1 = n2 = 1000 | 0.92 s |
| `simulate_particles`, n1 = n2 = 2000 | 1.83 s |
| `simulate_particles`, n1 = n2 = 4000 | 3.77 s |
| `simulate_orbit` | 0.03 s |
| `get_timing`, `initial_state`, `search_orbit` | 0.01–0.02 s |
| `unperturbed_particles`, `mass_profile` | 0.005 s |

Particle runs grow linearly with `n1 + n2`, and with the number of steps, which depends on the
model's start time.

## Known issues
SPAM's own limits, measured and explained in [spam_physics.md](../spam_physics.md):
- Results are reliable up to about the first close pass and the tails it makes (no self-gravity).
- Past r = 10 the force stops falling (Open questions).
- `r_scale` also scales the force outside the core (Open questions).
- The `"pass"` start keeps SPAM's short lead, measured at the pass's speed, its fastest point.
- Every galaxy has the same size, whatever its mass; `rout` only places the particles.

## Where to see it
- `examples/notebooks/simulator_tour.ipynb`: a hands-on tour.
- `tests/test_simulation.py`: the stage on its own, including the checks behind "What lines up with
  what" and each config value reaching its own part of the galaxy.
