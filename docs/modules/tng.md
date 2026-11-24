# TNG targets (`tng.py`)

The IllustrisTNG (TNG50-1) data this project already has: two readers, two functions that measure a
galaxy from its particles, one that undoes the simulation box's wrap-around, and two that convert
units. Both files are HDF5 and live in the data folder, outside the repo. The readers return the values **exactly as stored**, and the measuring
functions work in whatever units they are given; units change only where `physical` or `spam_units`
is called. Nothing here changes the frame yet.

```python
from galactic_tidal_constraints import tng

table = tng.read_table("../data/tng-data/moi-4-targets.h5")
archive = tng.read_archive("../data/tng-data/target-archive-62000000190717.h5")           # stars
archive = tng.read_archive("../data/tng-data/target-archive-62000000190717.h5", "all")    # every type

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
  the old rotated frame; they have to be taken back to the box frame first.

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
folder (the one holding `tng-data/`); they take about 15 seconds:

```bash
GTC_DATA_DIR=../data uv run pytest tests/test_tng.py
```
