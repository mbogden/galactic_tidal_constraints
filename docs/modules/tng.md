# TNG targets (`tng.py`)

The IllustrisTNG (TNG50-1) data this project already has: two readers, two functions that measure a
galaxy from its particles, one that takes an archive's particles out of the old frame, one that undoes
the simulation box's wrap-around, two that convert units, four that build the frame SPAM uses, one that turns a spin into SPAM's
disk angles, and one that says which galaxy is the primary. Both files are HDF5 and live in the data folder, outside the repo. The readers return the values **exactly as stored**, and the measuring
functions work in whatever units they are given; units change only where `physical` or `spam_units`
is called, and the frame only where `archive_to_box`, `relative_to` or a rotation is applied.

```python
from galactic_tidal_constraints import tng

table = tng.read_table("../data/tng-legacy-data/moi-4-targets.h5")
archive = tng.read_archive("../data/tng-legacy-data/target-archive-62000000190717.h5")           # stars
archive = tng.read_archive("../data/tng-legacy-data/target-archive-62000000190717.h5", "all")    # every type

primary = archive["subfind_galaxy"] == 0
measured = tng.galaxy_properties(archive["positions"][primary], archive["velocities"][primary],
                                 archive["masses"][primary])        # mass, center_of_mass, velocity, spin
```

Only what exists today is described here. The readers need `h5py`, which is in the `tng` dependency
group (`uv sync --group tng`; a plain `uv sync` installs it too, for the tests). Decisions behind it
are in [PLAN.md](../../PLAN.md); what the TNG values mean, and how that was checked, is in
[tng_physics.md](../tng_physics.md). A worked tour is in
[`examples/notebooks/tng_tour.ipynb`](../../examples/notebooks/tng_tour.ipynb).

## The target table: `read_table(path)`
One row per encounter per snapshot (1268 rows, 59 encounters). Returns a dict of column name to
array, in the table's column order. Every array has one entry per row:

| Kind of column | Shape | Examples |
|---|---|---|
| Number | (rows,) | `snap`, `p_SubhaloMass`, `time_gyr`, `scale_factor` |
| Vector | (rows, 3) | `p_SubhaloPos`, `s_SubhaloVel`, `p_SubhaloSpin` |
| Matrix | (rows, 4, 4) | `orbital_frame_pos_transform`, `orbital_frame_vel_transform` |
| Text | (rows,), `str` | `img_name_1` |

- `moi_2` identifies an encounter and `snap` the snapshot. `p_` is the primary galaxy, `s_` the
  secondary. IDs and snapshot numbers are `int64`, so a full `SubhaloIDRaw` matches exactly:
  `table["p_SubhaloIDRaw"] == 62000000190717`.
- An encounter's rows: `rows = table["moi_2"] == table["moi_2"][i]`.

What the stored values are:

| Columns | Values |
|---|---|
| `p_`, `s_` + a catalog field (`SubhaloPos`, `SubhaloCM`, `SubhaloHalfmassRad`, `SubhaloVel`, `SubhaloMass`, `SubhaloSpin`) | Raw TNG catalog values: lengths in ckpc/h, velocities in km/s, masses in 10¹⁰ M☉/h. `SubhaloSpin` is the angular momentum **per unit mass**, in (kpc/h)(km/s) with a physical (not comoving) length |
| `redshift`, `scale_factor`, `time_gyr` | The snapshot's redshift, a = 1 / (1 + z), and the age of the universe in TNG's cosmology (Gyr) |
| `pa_`, `sa_`, `ps_`, `ss_` columns; `_orbital_frame` columns; `_transform` matrices; `inclination`, `azimuthal` | The earlier pipeline's own values. Units not verified |

Each dataset in the file also carries a `units` attribute, and some a `note`.

## A target archive: `read_archive(path, particle_types=("stars",))`
One target's particles, both galaxies, at the target snapshot. The file is named after the primary's
`SubhaloIDRaw`. `particle_types` is any of `"gas"`, `"dm"`, `"stars"` and `"bh"` (one name or
several), or `"all"`. Returns a dict with one row per returned particle, in the file's order:

| Key | Shape, type | Values |
|---|---|---|
| `positions` | (n, 3) `float32` | ckpc/h |
| `velocities` | (n, 3) `float32` | km/s ÷ √a: multiply by √a for km/s |
| `masses` | (n,) `float32` | 10¹⁰ M☉/h |
| `particle_type` | (n,) `int8` | 0 gas, 1 dark matter, 4 stars, 5 black hole (`tng.PARTICLE_TYPES`) |
| `subfind_galaxy` | (n,) `int8` | The galaxy SUBFIND gives the particle to: 0 primary, 1 secondary |
| `historical_galaxy` | (n,) `int8` | The galaxy it belonged to before the encounter: 0 or 1 |
| `p_id`, `s_id` | `int` | The primary's and secondary's `SubhaloIDRaw` |
| `t_snap` | `int` | The target snapshot |

- **The file's own labels are wrong.** It labels its columns "kpc", "km_per_s" and "1e10_Msun", but
  the values are raw TNG snapshot units, as in the table above.
- **The frame is the earlier pipeline's orbital frame**, not the simulation box: the particles were
  moved by the table's `orbital_frame_pos_transform` and `orbital_frame_vel_transform` for that row.
- When only some types are asked for, the file is read a few million rows at a time, so a
  stars-only read doesn't load the whole file (the largest holds 28.8 million particles).

## Measuring a galaxy
Both functions take plain arrays and return results in the units of those arrays: nothing is
converted. Sums are done in `float64`.

**`galaxy_properties(positions, velocities, masses)`** returns a dict:

| Key | Type | Value |
|---|---|---|
| `mass` | `float` | The total mass |
| `center_of_mass` | (3,) | The mass-weighted mean position |
| `velocity` | (3,) | The mass-weighted mean velocity |
| `spin` | (3,) | The angular momentum per unit mass, about the center of mass in the galaxy's rest frame: Σ m (r − center) × (v − velocity) / Σ m |

The spin doesn't depend on where the origin is or on a velocity added to every particle, and it
turns with the frame. So a spin is moved to another frame by the rotation alone, never the shift.

**`half_mass_radius(positions, masses, center)`** returns the distance from `center` of the first
particle, going outward, at which the mass inside reaches half the total.

## Out of the old frame: `archive_to_box`
The archives store particles in the earlier pipeline's frame: the box frame turned by a rotation and
moved by a shift, with one 4×4 matrix for positions and one for velocities, both in the target table.
**`archive_to_box(positions, velocities, pos_matrix, vel_matrix)`** undoes both and returns
`(positions, velocities)` in the box frame, as `float64` and still in stored units:

```python
row = int(np.flatnonzero(table["p_SubhaloIDRaw"] == archive["p_id"])[0])
positions, velocities = tng.archive_to_box(archive["positions"], archive["velocities"],
                                           table["orbital_frame_pos_transform"][row],
                                           table["orbital_frame_vel_transform"][row])
```

- It is exact: the matrices hold a pure rotation and a shift, so x_box = Rᵀ (x_old − shift). A matrix
  that isn't a rotation and a shift is an error.
- In the box frame the particles can be compared with the catalog directly: a particle sits on each
  galaxy's `SubhaloPos` (the most bound one), and the center of mass is `SubhaloCM`.
- A galaxy that sits across a face of the box comes back in two parts, as TNG stored it. Follow with
  `nearest_image(positions[galaxy], nucleus)`. Two archives need it: `49000000112167` (47,690 of the
  primary's particles are across a face; unmended, its center of mass is 73 ckpc/h off and its spin
  28% off) and `87000000451863` (10 particles).

## The box edge
TNG50-1's box is 35,000 ckpc/h on a side (`tng.BOX`) and wraps around, so two points close together
can have coordinates almost a whole box apart
([why](../tng_physics.md#the-box)).

**`nearest_image(positions, reference, box=BOX)`** shifts each position by whole box lengths until
every coordinate is within half a box of `reference`, and returns them as `float64`:

```python
tng.nearest_image([50.0, 100.0, 200.0], [34950.0, 120.0, 180.0])       # [35050., 100., 200.]
secondary = tng.nearest_image(table["s_SubhaloPos"], table["p_SubhaloPos"])   # a whole track at once
apart = secondary - table["p_SubhaloPos"]
```

- `reference` is one point (a galaxy's center, for its particles) or one per position (a track).
- Call it on stored values, before `physical`: the box is 35,000 only in comoving ckpc/h.
- The results may lie outside 0 to 35,000. Only positions relative to the reference are meaningful.
- In the target table, 20 rows in 6 encounters have the pair split by a face; stored, they look
  about 34,800 ckpc/h apart. Two target archives (`49000000112167`, `87000000451863`) have galaxies
  whose own particles are split. An archive's positions can't be mended as read, because they are in
  the old rotated frame: use `archive_to_box` first.

## Units
Two steps, each one function with the factors in one table.

**`physical(values, quantity, scale_factor=None)`** converts stored TNG values to kpc, km/s and M☉.
h is TNG's 0.6774 (`tng.H`) and a is the snapshot's scale factor (the table's `scale_factor`):

| `quantity` | Stored as | Factor | Result |
|---|---|---|---|
| `"length"` | ckpc/h | × a ÷ h | kpc |
| `"velocity"` (catalog velocities) | km/s | none | km/s |
| `"particle_velocity"` (snapshot particles) | km/s ÷ √a | × √a | km/s |
| `"mass"` | 10¹⁰ M☉/h | × 10¹⁰ ÷ h | M☉ |
| `"spin"` (catalog `SubhaloSpin`) | (kpc/h)(km/s) | ÷ h | kpc km/s |

- `scale_factor` is needed for `"length"` and `"particle_velocity"`: one number, or one per row to
  convert a table column in one call (`tng.physical(table["p_SubhaloPos"], "length", table["scale_factor"])`).
- A spin measured from particles has no row: convert the positions and velocities first, and
  `galaxy_properties` then gives the spin in kpc km/s.

**`spam_units(values, quantity)`** converts physical values to SPAM's units:

| `quantity` | From | SPAM's unit | Constant |
|---|---|---|---|
| `"length"` | kpc | 15 kpc | `tng.SPAM_LENGTH` |
| `"mass"` | M☉ | 10¹¹ M☉ | `tng.SPAM_MASS` |
| `"velocity"` | km/s | 169.33 km/s | `tng.SPAM_VELOCITY` |
| `"time"` | Gyr | 0.08662 Gyr | `tng.SPAM_TIME` |

SPAM works with G = 1. Its length and mass units are Wallin's (`parameters_module.f90` in the JSPAM
fork); the velocity and time units follow from them, √(G × mass ÷ length) and length ÷ velocity, with
G = 4.30091 × 10⁻⁶ kpc (km/s)² / M☉ (`tng.G`). They agree with the constants in SPAM's source to
3 × 10⁻⁵.

## The frame SPAM uses
SPAM works with the primary's nucleus at the origin and at rest. Four functions build that frame from
box-frame values in physical units; why each step is what it is, is in
[tng_physics.md](../tng_physics.md#a-galaxys-velocity).

| Function | Returns |
|---|---|
| `core_velocity(positions, velocities, masses, center, radius)` | (3,), the mass-weighted mean velocity of the particles within `radius` of `center`. Units as given. No particle inside is an error |
| `hubble_rate(scale_factor)` | H at that snapshot, in km/s per kpc (0.06774 today) |
| `relative_to(positions, velocities, center, center_velocity, scale_factor, hubble_flow=True)` | `(positions, velocities)` relative to the center, in kpc and km/s. With `hubble_flow` on, H × position is added to the velocities |
| `orbit_rotation(relative_position, relative_velocity)` | (3, 3) rotation `R`: x toward the secondary, z along position × velocity. A head-on pair is an error |

One target, start to finish (`a` is the row's `scale_factor`; `p_pos`, `s_pos` its two `SubhaloPos`):

```python
positions, velocities = tng.archive_to_box(archive["positions"], archive["velocities"], pos_matrix, vel_matrix)
positions = tng.physical(tng.nearest_image(positions, p_pos), "length", a)           # kpc, whole galaxies
velocities = tng.physical(velocities, "particle_velocity", a)                        # km/s
nucleus = {"p": tng.physical(p_pos, "length", a),
           "s": tng.physical(tng.nearest_image(s_pos, p_pos), "length", a)}

stars = archive["particle_type"] == tng.PARTICLE_TYPES["stars"]
moving = {side: tng.core_velocity(positions[stars], velocities[stars], archive["masses"][stars], nucleus[side], 1.0)
          for side in "ps"}                                                          # each nucleus's velocity

secondary = tng.relative_to(nucleus["s"], moving["s"], nucleus["p"], moving["p"], a)
R = tng.orbit_rotation(*secondary)
positions, velocities = tng.relative_to(positions, velocities, nucleus["p"], moving["p"], a)
positions, velocities = positions @ R.T, velocities @ R.T                            # the same for a spin
```

- Afterwards the primary's nucleus is at the origin and at rest, the secondary's is at
  (distance, 0, 0), and the pair turns counter-clockwise in the x-y plane.
- A rotation is applied with `vectors @ R.T`. There is no shift inside it, so a spin can't pick one up.
- A galaxy's black hole gives a second measure of its nucleus's velocity: it is one row of the
  archive, `velocities[archive["particle_type"] == tng.PARTICLE_TYPES["bh"]]`. The tests compare the two.
- An earlier snapshot's nucleus positions (from the table) can be turned by the same `R` to draw the
  past orbit in this frame. Only the target snapshot has a measured nucleus velocity.

## Disk angles: `disk_angles(spin)`
Turns a disk's spin direction, in the frame SPAM will run in, into SPAM's two disk angles. Returns
`(phi, theta)` in degrees, for `phi1`/`theta1` or `phi2`/`theta2`:

```python
spin = tng.galaxy_properties(positions[disk], velocities[disk], masses[disk])["spin"]   # in SPAM's frame
phi, theta = tng.disk_angles(spin)
```

- `theta` is the angle from +z, 0 to 180: 0 is a flat disk turning counter-clockwise, 180 the same
  disk turning clockwise, 90 edge-on.
- `phi` is the direction of the spin's x-y part, 0 up to 360. A spin exactly along z has none, and
  `phi` is 0.
- Only the direction matters; a spin of zero is an error.
- The convention is SPAM's own, and a test builds disks with SPAM and measures their spin to confirm
  it ([tng_physics.md](../tng_physics.md#disk-angles)).

## Which galaxy is the primary: `primary_side(archive, rule)`
Returns `"p"` or `"s"`: the letter that starts the primary's columns in the target table.
`tng.GALAXY_LABELS` (`{"p": 0, "s": 1}`) gives each letter's number in an archive's galaxy labels.
The rule must be named:

| `rule` | The primary is |
|---|---|
| `"table"` | The galaxy the target table calls the primary. Always `"p"` |
| `"stars"` | The galaxy with more mass in stars, each star counted for the galaxy it belonged to before the encounter (`historical_galaxy`). A tie gives `"p"` |

```python
archive = tng.read_archive(path)                    # stars
primary = tng.primary_side(archive, "stars")        # "p" or "s"
secondary = "s" if primary == "p" else "p"
origin = table[f"{primary}_SubhaloPos"][row]        # the primary's nucleus
```

- Nothing is renamed or rewritten: the table and the archive keep their own `p` and `s`. Code that
  builds a target uses the answer to choose which galaxy goes at the origin.
- `"stars"` needs star particles in both galaxies, or it is an error.
- Why not the catalog's total mass: [tng_physics.md](../tng_physics.md#which-galaxy-is-the-primary).

## Checked against the TNG catalog
The archives' particles, split by SUBFIND's galaxy label, give back the catalog's values for the same
galaxies: mass, separation, relative velocity (with √a), spin (× a^1.5) and half-mass radius. The
comparison, its numbers and what it settles are in
[tng_physics.md](../tng_physics.md#checked-against-particles). Two results concern the archives' frame:

- **The frame matrices are a pure rotation plus a shift**, the same rotation for positions and
  velocities. The velocity matrix works on particle velocities (km/s ÷ √a), not on catalog km/s.
- Not settled: what the old frame's origin and rest velocity are. Neither galaxy, nor the pair's
  center of mass, sits at the origin or at rest in it.

## Known issues
- `"stars"` includes wind particles: the archives have no formation time to tell them apart.
- A black hole's mass is its `BH_Mass`, not its dynamical mass.

## Tests
`tests/test_tng.py`. The tests on small made-up files and particles always run. The tests on the
real files, including the check against the catalog, run only when `GTC_DATA_DIR` names the data
folder (the one holding `tng-legacy-data/`); they take about 40 seconds, or 25 with `-m "not slow"`:

```bash
GTC_DATA_DIR=../data uv run pytest tests/test_tng.py
```
