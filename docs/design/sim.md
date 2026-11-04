# Design: `sim`

`src/galactic_tidal_constraints/simr/sim.py` runs one SPAM simulation from Python: it takes values and
returns arrays. It's the only module that imports `jspam` (`PLAN.md`, One door to SPAM). It doesn't
touch the disk, know about pools or MPI, or keep state between calls.

## Main types
| Type | What it holds |
|---|---|
| `CollisionParams` | The 22 values of a SPAM model string, named as in the Fortran. `to_array()`, `from_array()` |
| `Endings` | When to save a particle run: `unit` and `values` |
| `SimulatorConfig` | Everything else that sets the result: kind, particle counts, heat, potential, mass model, endings |
| `Simulation` | A result: the arrays, plus `t_start`, `n_steps` and `start_fallback` |

## Main functions
```python
simulate(params, config, seed=None) -> Simulation
resolve_end_steps(endings, timing) -> list[int]
to_gyr(times) -> times in Gyr
```
- `seed` is required for particle runs, so a run can't be unseeded by accident. Orbit runs have no
  random numbers.
- The same inputs and seed give identical numbers on the same build and machine. Different builds or
  CPUs can differ in the last digits, so compare those with a tolerance.

## Many endings
One integration can save particles at any list of times. Fortran only sees integer steps;
`resolve_end_steps` converts from whichever unit you choose:

| `unit` | Means | Example | Best for |
|---|---|---|---|
| `fraction` | Share of the normal run: 0 is the start, 1 the normal end | `[0.8, 1.0, 1.2]` | Comparing across models |
| `gyr` | Gyr relative to t = 0, when the 22 values apply | `[-0.1, 0, 0.1]` | Matching TNG snapshots |
| `time` | SPAM time units relative to t = 0 | `[-1, 0, 0.5]` | Debugging |
| `step` | Integrator steps from the start | `[0, 5000]` | Tests |

- The old `many_endings_run` is `fraction` with `[0.8, 0.84, ..., 1.2]`.
- `[0, 1.0]` as fractions gives the initial and final particles (the old `basic_run`); `[0]` gives the
  initial disks only (the old `basic_disk`).
- Values must be ascending. An ending before the start, or two landing on one step, raises an error.
- Each model has its own start time, so a fraction is a different physical time for each model.
- A SPAM time unit is about 86.6 Myr and a step is 0.001 of one, about 87 kyr.
- A run's normal end is 1 to 2 steps past t = 0, as in Wallin's `basic_run`.

## Changes from the old code
Old code: `Simulator/new_simulator.py` and `custom_runs.f90` on `working_research`, for reference only.

| Old | New |
|---|---|
| 22-float list | `CollisionParams`, or an array |
| `set_spam_parameters()` changes global Fortran state | The full mass model passes on every call |
| `m_bulge`, `h_bulge`, `h_disk` set, then overwritten with fixed values | All six values reach the tables |
| Two arguments swapped in the call | Passed by keyword; a test checks each reaches its own variable |
| `lnl` default 0.01 in Fortran, 0.1 in Python | SPAM's 0.001 |
| Unseeded particle runs | Required `seed` |
| Hash-based cache and `save_dir` in `basic_run` | Moved to `simr` |
| `basic_run`, `run_spam_bundle_1`, `basic_disk`, `many_endings_run` | One `simulate` with endings |
| Orbit times start at 0, then get shifted | True SPAM time |
| Orbit runs use 10 + 5 throwaway particles | No particles |
| Import runs `simr_init()`, sets up logging, changes `sys.path` | No import side effects |
| Plotting and `__main__` tests in the module | `pytest`; plotting left out |

## Options that change the physics
Both default to Wallin's behavior, so nothing changes unless you ask.

| Option | `wallin` (default) | Other |
|---|---|---|
| `normalization` | Divides by a fixed 7.13333, so changing `m_halo` or `m_bulge` changes the galaxy's total mass | `total`: divides by the model's actual total, so they change only how mass is distributed |
| `heat_mode` | Adds a value in [0, heat) to each velocity component, so disks also drift by about heat/2 | `centered`: a value in [-heat/2, heat/2), the same spread with no drift |

## Known behavior to be aware of
All of this is in Wallin's original code, and is unchanged.
- **Start time:** if SPAM's estimate is earlier than -12, or within one step of 0, the run starts at -5
  instead. `Simulation.start_fallback` reports it, and it's logged as a warning.
- **`rscale` values are ignored.** The default `opt = 1` gives a `1/r` particle distribution that doesn't
  use them.
- **`epsilon` values are ignored** by the extended potential.
- **`rout` doesn't scale the potential.** `df_index` receives it but doesn't use it, so it only sets how
  far a disk's particles extend.
- **Past r = 10 the force stops falling,** because the force table ends there. See the open questions.

## Open questions
These change results, so they're left for a decision. See the experiment notes for measurements.
1. Which defaults should `normalization` and `heat_mode` have?
2. Should the force keep falling past r = 10 (about 150 kpc)? Measured: it's constant from there out, so
   at r = 20 it's about 4 times what it would be if it kept falling.
3. Should `r_scale` only change the halo's shape? Measured: outside the core it also multiplies the force
   by (10 / `r_scale`)^2, so `r_scale` = 5 gives about 4 times the force.
4. `new_rho`, used by dynamical friction, divides enclosed mass (not shell mass) by the shell's volume.
   Is that intended?

## Tests
`tests/test_sim.py`, plus `tests/test_run_collision.py` in the JSPAM fork, run in the container.
