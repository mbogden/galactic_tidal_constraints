# SPAM physics notes

What SPAM's parameters and options actually do, worked out by reading the Fortran in
[JSPAM](https://github.com/mbogden/JSPAM) (`fortran/`, branch `sim-v2-branch`) and checked by running it.
Routine names are given so they can be found again; line numbers drift.

## What SPAM is
A restricted three-body code. The two galaxy centers are the only bodies with mass. Disk particles are
massless tracers that feel both centers but nothing else. It works in a frame centered on galaxy 1
(the primary), with G = 1, and integrates with RK4 at a fixed step of 0.001.

A run starts from the 22 parameters, which describe the pair **now**, as observed. SPAM integrates the
two centers backward to a start time before the encounter, places the disks there, then runs
everything forward to about now.

### Units
| Quantity | SPAM unit | Source |
|---|---|---|
| Length | 15 kpc | `parameters_module.f90` |
| Mass | 10¹¹ M☉ | `parameters_module.f90` |
| Time | ≈ 87 Myr | `parameters_module.f90` |

The Newtonian potentials are scale-free, so these only matter when comparing with real galaxies (and
for MOND, below). The mass-model tables use an **internal** length unit of 1/10 SPAM unit (1.5 kpc):
`df_index` reads a distance r from row `int(r × rs_internal / 100 × nnn) + 1`, with `rs_internal` = 10,
so internal radius = 10 × r. `r_scale`, `h_bulge` and `h_disk` are in internal units.

## Key facts at a glance
- **We run the `EXTENDED` potential.** Each galaxy is the same tabulated **unit galaxy** (halo, disk,
  bulge), scaled by its mass. [More](#the-extended-potential-what-we-run)
- **Only 14 of the 22 parameters do anything** in it: `epsilon1/2` and the six `rscale` values are
  never read. So a model in `sim` is 14 values, and `sim` appends fixed ones for SPAM.
  [More](#parameters-that-do-nothing)
- **Every galaxy has the same size**, whatever its mass. `rout` only sets how far out particles are
  placed. A per-galaxy unit galaxy is planned (sim-v2 step 5b).
- **Wallin's mass normalization makes the mass-model values change a galaxy's total mass.** `sim` uses
  `"total"` instead, so `mass1` and `mass2` are true totals. [More](#normalization-what-the-unit-galaxy-is-divided-by)
- **`r_scale` resizes the whole galaxy** and, outside the core, scales the force by
  (10 / `r_scale`)². [More](#r_scale-the-halos-core-radius-and-the-size-of-everything)
- **Dynamical friction only acts inside r = 2.** Wallin's `lnl` = 0.001 makes it nearly nothing;
  `sim` uses 0.1.
  [More](#lnl-dynamical-friction)
- **The force stops falling at r = 10**, where the tables end.
- **Wallin's disk heat drifts** every heated disk toward +x, +y, +z. `sim` centers it.
  [More](#heat1-heat2-heat_mode-random-motion-in-each-disk)
- **Results are reliable up to about the first close pass** and the tails it makes.
  [More](#limits-shared-by-all-potentials)
- **SPAM's start-time search can miss the encounter.** It keeps the closest pass anywhere back to
  t = −30, and silently starts at −5 when its answer is earlier than −12, which can be after the pass.
  `sim` starts before the most recent pass instead (`start="pass"`). [More](#the-start-time)

## The extended potential (what we run)
`potential_type` 1, "NBI" (`diffeq_nbi`). Each galaxy has the same tabulated mass model, scaled by its
mass: a halo (Hernquist's 1993 form, truncated by a Gaussian), a disk (exponential, but spherically
averaged) and a Gaussian bulge, with masses 5.8 : 1 : 0.33 (81 : 14 : 5 %). So **the disk is about 14% of
`mass`**, and `mass` is the galaxy's total including its halo. `init_distribution` (df_module.f90)
builds the tables once per run, and forces are looked up from them.

Rotation speed of a mass-1 galaxy compared with a point mass with ε = 0.3, from the default tables:

| r | mass inside r | v, extended | v, point mass |
|---|---|---|---|
| 0.1 | 0.08 | 0.90 | 1.00 |
| 0.3 | 0.31 | 1.02 | 1.29 |
| 0.5 | 0.53 | 1.03 | 1.21 |
| 1.0 | 0.85 | 0.92 | 0.96 |
| 2.0 | 1.00 | 0.71 | 0.70 |
| 5.0 | 1.00 | 0.45 | 0.45 |

Inside the disk (r ≲ 1) the rotation curve is close to flat. Beyond r ≈ 2 all the mass is inside, and
the extended and point-mass potentials agree.

**Why it's the right default:**
- **A realistic disk:** a flat rotation curve, and mass spread through the galaxy, so a companion that
  passes inside it feels only the mass inside its orbit. Both shape how tails and bridges form.
- **Dynamical friction** (Chandrasekhar's formula): galaxy 1 slows galaxy 2, and galaxy 2's disk moves
  with it. The backward integration flips the friction's sign, so going back to the start time and
  forward again are consistent.
- **It's what Wallin's model strings were run with:** `custom_collision` sets type 1 whenever it's
  given a model string, and the Python version does the same. Galaxy Zoo: Mergers models are meant for
  this potential; in another one, the same parameters give a different collision.
- **About as fast as the point mass:** a table lookup per particle.

**Quirks:**
- **Every galaxy has the same size.** `df_index` takes `rout` but doesn't use it, so a small companion
  has the same mass profile, in length, as the primary.
- **The force stops falling at r = 10.** `df_index` caps at the last table row, so beyond 10 SPAM units
  the pull is constant (M/100) instead of falling as 1/r². This only matters for very wide orbits at
  the start of a run. Inside r = 10, the force matches plain gravity from the enclosed mass to within
  0.2%, except inside r = 0.05, where it's deliberately smoothed.
- **The halo, disk and bulge end at r = 2** (2 halo core radii): past it, the enclosed mass is constant
  and the density is 0.
- **The disk is spherical in the potential,** and forces come from the nearest table row with no
  interpolation. Rows are 0.001 apart, so this is fine.

## The start time
A run starts from the pair **now** (t = 0) and has to begin before their encounter, so the disks are
undisturbed when it happens. `sim.get_timing` reports the start, `sim.simulate_orbit` returns the
secondary's path from it, and `sim.search_orbit` and `sim.find_passes` show the path back to −30 and
every close pass on it. One time unit is about 87 Myr. How `sim` chooses the start is under
[`start`, `start_pass`](#start-start_pass-when-the-run-starts); this section is SPAM's own way.

**Code:** `getTStart` (setup_module.f90), called from `create_collision` (init_module.f90):
1. Runs only the two centers **backward** from now in steps of 0.001, out to t = −30 (the window), and
   keeps the point where they were closest, over the whole window.
2. Picks a start: for a bound orbit, near the last apocenter it found (if any); otherwise the closest
   approach's time minus a **lead** of `10 × rout1 / speed at closest approach`.
3. `create_collision` overrides it: a start earlier than **−12**, or within one step of 0, becomes **−5**
   (reported as `start_fallback`).

A run then takes `int(−t_start / 0.001) + 2` steps, ending 1 to 2 steps past now. The orbit uses the
same potential, mass model and dynamical friction as the particles, and doesn't depend on them, so
`simulate_orbit`'s last row is exactly where `simulate_particles` leaves the secondary.

**Unperturbed galaxies (`sim.unperturbed_particles`):** the setup only, with `t_start` = 0. Placement steps
the secondary back while `t_start < t`, so at 0 it takes no steps: the disks are built around today's
centers (the secondary exactly at the model's position) and returned before any step. With the same seed
they're the same disks a run starts with. Positions depend only on the 14 parameters and the seed; heat
and the mass model change only the starting velocities.

**Physics:** the lead is meant to give the secondary about 10 disk radii to approach. Measured on the
SDSS model 587722984435351614 (default config): closest approach at t = −2.91, distance 5.4, speed 7.4,
so the lead is 4.0 and the run starts at −6.91.

**Quirks:**
- **The lead is short.** It uses the speed at closest approach, the fastest point of the orbit, so the
  secondary starts nearer than 10 × `rout1` (a candidate for why TNG runs didn't start early enough).
- **The search can pick an older pass, almost at random.** It keeps the closest point in the whole
  window, not the most recent one. With dynamical friction this weak (and none outside r = 2), a bound
  orbit repeats: every pass has the same distance, to rounding. Which one wins comes down to rounding.
  For the SDSS model, the default finds the pass near −3 (another, equally close, is at −27.0), but
  `r_scale` = 9 makes it pick one at −23.4, and `r_scale` = 5 (four equal passes) one at −20.5.
- **The −5 fallback can start after the encounter.** Both of those starts are before −12, so the run
  starts at −5: after the pass the search chose, and too late to show it.
- **The window, the −12 cap, the −5 fallback and the 10 are fixed in the Fortran.** `sim` keeps the
  window and the 10 (`sim.LEAD_FACTOR`) and has no cap or fallback.

## The settings we expose (`SimulatorConfig`)
Everything `sim.SimulatorConfig` can change, apart from the parameters. For each, what
it does in the code and what it means physically. `sim` passes each value through `jspam`, which sets it
in the Fortran before every run, so nothing carries over between runs.

`jspam`'s defaults reproduce Wallin's program. `sim`'s defaults are ours: they differ only where a fix
was chosen (`normalization`, `heat_mode`, `start`) or a value changed (`lnl`).

| Setting | `sim` default | Wallin's |
|---|---|---|
| `normalization` | `"total"` | `"wallin"` |
| `r_scale` | 10 | 10 |
| `m_halo`, `m_bulge` | 5.8, 0.3333 | 5.8, 0.3333 |
| `h_disk`, `h_bulge` | 1.0, 0.2 | 1.0, 0.2 |
| `lnl` | 0.1 | 0.001 |
| `heat1`, `heat2` | 0, 0 | 0, 0 |
| `heat_mode` | `"centered"` | `"wallin"` |
| `n1`, `n2` | 1000, 1000 | 1000, 500 (`basic_run`) |
| `start`, `start_pass`, `no_pass` | `"pass"`, 1, −5 | `"wallin"` (SPAM's search) |
| `seed` | 0 | none: random every run |

### The mass model
These shape the unit galaxy, so they change every force in the run.

#### `normalization`: what the unit galaxy is divided by
- **Code:** at the end of `init_distribution`, every table (masses, densities, dispersions, forces) is
  divided by `norm`. `"wallin"`: `norm` = 7.13333, Wallin's sum of the default masses
  (5.8 + 1.0 + 0.3333). `"total"`: `norm` = the table's own total, `mass_total(nmax)`.
- **Physics:** whether `mass1` and `mass2` are the galaxies' real total masses.
  - `"total"`: always. The unit galaxy has mass 1, and the five shape values only change how the mass
    is distributed.
  - `"wallin"`: only approximately. The true total at the defaults is 7.101 (the halo's tail past r = 2
    is cut off), so the unit galaxy is 0.9955, and changing a mass changes the total. For example,
    `m_halo` 11.6 gives 1.81, so the galaxy pulls 1.8× harder than `mass1` says.

#### `r_scale`: the halo's core radius, and the size of everything
- **Code:** `rchalo` in `init_distribution`. It sets the halo's profile (core radius `rchalo`, truncated
  by a Gaussian) **and the radius grid of all the tables**: `rad(j) = x × rchalo`, with x from 0.001 to
  2. But `df_index` reads the tables as if the grid were fixed at the default, 10. So row j, which the
  integrator uses at distance r, holds the galaxy at radius r × r_scale / 10. Not to be confused with
  the 22 parameters' `rscale1/2`, which do nothing.
- **Physics:** in effect, a size for the whole galaxy, halo, disk and bulge together: `r_scale` = 20
  makes the galaxy twice as large. Outside the core, where the enclosed mass is the whole galaxy, the
  force at a given distance scales as (10 / `r_scale`)², measured as 4.000 at `r_scale` 5 and 0.250 at
  20. So it's also, partly, a strength knob. The halo is cut off at 2 core radii (30 kpc by default).

#### `m_halo`: halo mass
- **Code:** `mhalo` multiplies the halo's profile (normalized to 1 over all space) in
  `init_distribution`. Only the part inside the table (r < 2) counts: 5.768 of the default 5.8.
- **Physics:** how much of the galaxy is dark matter, half of it inside about 8 kpc by default. With
  `"total"` it changes the halo's share (81% by default); with `"wallin"` it also adds to the galaxy's
  total mass.

#### `h_disk`: the disk in the potential
- **Code:** a density `exp(−r / h_disk) / (8π h_disk²)` in internal units, spherically averaged (added to
  the other components as a sphere, not a flat disk). It isn't rescaled, and its mass works out to
  exactly `h_disk`. So `h_disk` sets the disk's mass as well as its size.
- **Physics:** the disk's contribution to the galaxy's gravity: 14% by default, half of it inside about
  4 kpc (2.67 × `h_disk`). **This isn't the particle disk.** Particles are placed by `profile` (1/r, out
  to `rout`) whatever `h_disk` is; `h_disk` only changes the force they feel.

#### `m_bulge`, `h_bulge`: the bulge
- **Code:** a Gaussian density `exp(−r² / h_bulge²)` in internal units, integrated and then rescaled so
  its mass is exactly `m_bulge`.
- **Physics:** a compact central mass: 4.7% of the galaxy by default, half of it inside about 0.3 kpc
  (0.021 SPAM units). It mostly matters for particles and encounters close to the center.

#### `lnl`: dynamical friction
- **Code:** the Coulomb logarithm ln Λ in Chandrasekhar's formula, in `diffeq_nbi` (integrator.f90) and
  `diffq_nbi` (setup_module.f90):
  `a_df = 4π · m2 · lnl · ρ(r) / v² · [erf(X) − 2X/√π · e^(−X²)]`, with `X = v / σ(r)`, opposite the
  secondary's velocity v. m2 is the secondary's mass: the heavier the moving galaxy, the stronger its
  wake. ρ and σ² come from the primary's tables (`new_rho`, `new_vr2`) at the distance r between the
  centers. It acts on the secondary's center, and its disk particles get the same deceleration, so the
  disk moves with it. The primary isn't slowed.
- **Physics:** a massive body moving through a halo pulls a wake behind it, which drags on it, so orbits
  shrink and galaxies eventually merge. Real values of ln Λ are around 2 to 10. Wallin's 0.001 makes
  friction about 0.1% of gravity; over a run it changes the orbit speed by roughly 1%. `sim` uses 0.1,
  100 times stronger (and the value the old Python code set in `set_spam_parameters`).
- **Quirk:** the density table is 0 past r = 2, so friction only acts while the secondary is inside
  r = 2. For wide encounters `lnl` has no effect at all.

### The start
#### `start`, `start_pass`, `no_pass`: when the run starts
- **Code:** with `"pass"`, `sim.find_passes` takes `search_orbit` (the path SPAM's search integrates,
  now back to −30, one row per step) and lists every local minimum of the distance between the centers,
  most recent first. `start_pass` picks one (1 = most recent), and the run starts
  `LEAD_FACTOR × rout1 / speed` before it (`LEAD_FACTOR` = 10, SPAM's own lead), passed to `jspam` as
  `t_start`. The Fortran then skips `getTStart`: `setup_run` sets the start and `perturber_position_vec`
  steps the secondary back to it along the same path. `"wallin"` runs SPAM's
  search ([The start time](#the-start-time)). A negative number starts exactly then.
- **Physics:** the disks should be built undisturbed, before the encounter that made the tails, so the
  run starts a little before the pass that matters, normally the most recent one. Older passes (2, 3)
  are there to try: the restricted three-body model is only reliable to about the first close pass, so
  a run starting before pass 2 shows two passes with that caveat. For pass 1 the start matches SPAM's
  when SPAM finds the same pass (within a step, −6.915 against −6.914 for the SDSS model).
- **Quirks:** the lead keeps SPAM's short-lead issue (it uses the pass's speed, the fastest point); a
  distance-based lead is a planned option. A start can be before −30 (pass 2 at −27 plus a lead of 4):
  that works, since placement doesn't need the search path, but `search_orbit` won't reach it. A pair
  still approaching now has no recent pass, so pass 1 is an older one, or there is none.
- **No pass (`no_pass`):** when the chosen pass doesn't exist back to −30 (a first approach, a pass older
  than −30, or fewer passes than `start_pass`), the run starts at `no_pass`, −5 by default, close to what
  SPAM ends up doing. `get_timing` reports `start` = `"fallback"` and the reason. So such a model still
  gets a run and a score instead of failing, which gives an optimizer something to rank. For a first
  approach that run is physically right (the last few hundred Myr of the approach); for a pass older than
  −30 it misses the encounter, and should score badly. `no_pass="error"` raises a `ValueError` with the
  reason instead, before any particles are made.

### The disk particles
These change how the disks are sampled, not the forces.

#### `heat1`, `heat2`, `heat_mode`: random motion in each disk
- **Code:** at the end of `profile` (setup_module.f90), after the disk is rotated into the simulation's
  frame, each velocity component gets `(randm() - heat_offset) * heat`. `heat_offset`
  (parameters_module.f90) is 0 for `"wallin"` and 0.5 for `"centered"`.
- **Physics:** a velocity dispersion, as in a real disk's stars: a heated disk is thicker and its tidal
  features are fuzzier. The kicks are uniform, not Gaussian, and their spread (width `heat`) is the
  same in both modes.
  - `"wallin"`: kicks in [0, heat), always positive, so every heated disk also gains a net velocity of
    heat/2 toward +x, +y and +z (the simulation's axes, whatever the disk's orientation) and drifts off
    its galaxy's center. That's a bug, not physics.
  - `"centered"`: kicks in [−heat/2, heat/2), with no drift.

#### `seed`: which random particles
- **Code:** all of SPAM's random numbers come from `randm()` (setup_module.f90), which calls Fortran's
  `random_number`. `profile` uses it to pick each particle's radius bin, radius and angle, and (with
  heat) its velocity kicks. `jspam` expands the seed into gfortran's seed array with numpy's
  `SeedSequence` and calls `set_seed`; with no seed it calls `seed_from_system`. `sim` always passes
  `config.seed` (0 by default), so runs repeat exactly.
- **The draws don't depend on the model.** Each particle takes a fixed number of draws in a fixed order
  (ring, radius, angle, three heat kicks, drawn even when heat is 0), galaxy 1 first. With the fixed disk
  option (`opt = 1`) radii are spread evenly between `rin` = 0.05 and `rout`, so with one seed every
  model gets the same pattern: each particle keeps its angle and its fractional radius, and `rout`,
  `theta`, `phi` and the masses only stretch, tilt and spin it up. Galaxy 2's draws depend on `n1` (how
  many galaxy 1 took first), not on any parameter.
- **Physics:** none. Different seeds give different, equally valid samples of the same disk. The orbit
  has no random numbers, so it's identical for every seed. Sharing one draw across models means score
  differences come only from the parameters ("common random numbers"); the cost is that any quirk of
  that draw is in every model. Comparing a model across seeds measures it.

#### `n1`, `n2`: particles per disk
- **Code:** the number of rows `profile` fills for each disk. Particles are placed with surface density
  ∝ 1/r from `rin` = 0.05 to `rout` (`opt` = 1, see [rscale](#rscale-the-disks-starting-profile-switched-off)),
  on circular orbits at the speed `circular_velocity` gives, then tilted by `phi` and `theta`.
- **Physics:** resolution only. Particles are massless, so they don't change the orbit or each other;
  more particles trace the same tails and bridges more finely. Run time grows about linearly with
  n1 + n2.

## The 22 parameters
In SPAM's model-string order. Galaxy 1 is the primary, at the origin; galaxy 2 is the secondary.

| Values | What they set |
|---|---|
| `x, y, z, vx, vy, vz` | The secondary's position and velocity relative to the primary, now |
| `mass1`, `mass2` | Each galaxy's mass: the unit galaxy is scaled by it |
| `rout1`, `rout2` | How far out each disk's particles are placed. **Not** the galaxy's size in the potential |
| `phi1`, `phi2`, `theta1`, `theta2` | Each disk's orientation, in degrees |
| `epsilon1`, `epsilon2` | Unused with `EXTENDED` (below) |
| `rscale1_*`, `rscale2_*` | Unused (below) |

### Parameters that do nothing
With `EXTENDED`, **`epsilon1`, `epsilon2` and all six `rscale` values have no effect**: changing them
leaves the output bit-identical (tested for each, `tests/test_sim.py`). SPAM still needs all 22, so a
model in `sim` is the first 14 (`sim.PARAM_NAMES`), and `sim` appends `sim.UNUSED_PARAMS`
(ε = 0.3, 0.3; rscale = 3.0 × 6) before calling `jspam`. `sim.parse_model_string` reads the first 14 of
any longer string.

#### epsilon: softening, used only by the point-mass and MOND potentials
In those potentials, each galaxy pulls with `a = M / (r² + ε²)`. ε sets the size of the galaxy's core:
the disk's rotation speed peaks at r = ε and falls off beyond it.

`EXTENDED` never reads it. Each place that does branches on `potential_type`, and the type 1 branch uses
the mass-model tables instead:
- particle forces: `diffeq_nbi` (integrator.f90)
- the two centers' orbit, used for the start time and the backward integration: `diffq_nbi`
  (setup_module.f90). It copies eps into unused variables only to stop a compiler warning.
- disk rotation speeds: `circular_velocity` (setup_module.f90)

`EXTENDED` gets its core from a fixed smoothing of the innermost table rows in `init_distribution`, which
isn't a parameter. `epsilon_halo` in that routine is set but never used.

#### rscale: the disk's starting profile, switched off
rscale only sets where the disk particles are placed in radius at the start (`distrb`,
setup_module.f90). It never changes a force. The disk option `opt` picks the profile:
- `opt = 1`: surface density ∝ 1/r. rscale isn't used.
- `opt = 2`: exponential disk with scale length `rscale(1)`.
- `opt = 3`: `exp(-rscale(1) r² - rscale(2) r - rscale(3))`, which is why there are three values.

`opt1` and `opt2` are fixed at 1 by `standard_galaxy1/2` (parameters_module.f90), and nothing in the
22-value path changes them. So every disk is the 1/r profile, from r = 0.05 (`rin`, also fixed) to
`rout`. Using rscale would mean changing `opt` in the Fortran.

## Limits shared by all potentials
Particles are massless, halos are rigid and don't respond, and the galaxies can't merge: there is no
self-gravity, so nothing but dynamical friction takes energy out of the orbit. Simulated features are
only reliable up to about the first close pass and the tails it makes.

## The other potentials
`potential_type` in the Fortran; the `jspam` package exposes `POINT_MASS` (0) and `EXTENDED` (1), and
`sim` always uses `EXTENDED`. In all three, the frame, the integrator and the massless particles are the
same. Only the force law, the disk's starting speeds and dynamical friction change.

### 0: point mass (`diffeq_spm`)
Each galaxy is a softened point mass, `a = M / (r² + ε²)`. This is the classic Toomre & Toomre setup.
- **Pros:** the simplest and fastest option. Easy to reason about. ε is the only shape parameter.
- **Cons:** no dark halo, so the rotation curve rises to a peak at r = ε and then falls off like
  Kepler's, with nothing flat. When the companion passes through the primary's disk, it feels the full
  point mass instead of only the mass inside its orbit, which overstates close encounters. There is no
  dynamical friction, so orbits never decay.
- **Bug, this path only:** `create_collision` (init_module.f90) passes `sqrt(eps)` (that is, ε) to
  `getTStart`, which expects ε², so the start-time search uses the wrong softening.

### 2: MOND (`diffeq_mond`)
The point mass (with ε), with each galaxy's pull boosted by MOND's standard interpolating function,
`a = a_N · sqrt((1 + sqrt(1 + (2 a₀ / a_N)²)) / 2)`. Not exposed by the `jspam` package.
- **Pros:** flat rotation curves without dark matter, so it can test MOND against a halo.
- **Cons:**
  - Each galaxy's boost is worked out on its own and then added. MOND isn't linear, so this isn't real
    MOND (it also leaves out the external field effect).
  - **It isn't scale-free.** a₀ has to be converted to SPAM units through the fixed physical units
    above, which gives a₀ ≈ 1.9. That's larger than typical accelerations in a run, so the boost is
    strong, and the result depends on the physical size assumed.
  - No halo and no dynamical friction.
