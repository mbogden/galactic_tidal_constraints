# TNG targets (`tng.py`)

The IllustrisTNG (TNG50-1) data this project already has: two readers, and two functions that
measure a galaxy from its particles. Both files are HDF5 and live in the data folder, outside the
repo. The readers return the values **exactly as stored**, and nothing here changes units or frame.
Converting comes later, in its own functions.

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
are in [PLAN.md](../../PLAN.md). A worked tour is in
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

## Checked against the TNG catalog
The archives' particles, split by SUBFIND's galaxy label, give back the catalog's values for the same
galaxies. The tests check this for both galaxies of `62000000190717`, `75000000121160` and
`75000000317209`; a is the snapshot's scale factor.

| From the particles | Catalog column | Largest difference seen |
|---|---|---|
| `mass` | `SubhaloMass` | 7 × 10⁻⁸ |
| Distance between the two `center_of_mass` | The same from `SubhaloCM` | 4 × 10⁻⁵ |
| Difference of the two `velocity`, × √a | The same from `SubhaloVel` | 2 × 10⁻⁷ |
| Size of `spin`, × a^1.5 | `SubhaloSpin` | under 10⁻⁵ |
| Direction of `spin` | `SubhaloSpin`, turned by the frame matrix's rotation | 0.0001° |
| `center_of_mass` | `SubhaloCM`, moved by `orbital_frame_pos_transform` | 0.001 ckpc/h |
| `velocity` | `SubhaloVel` ÷ √a, moved by `orbital_frame_vel_transform` | under 10⁻⁴ |
| `half_mass_radius` around `SubhaloPos` (moved the same way) | `SubhaloHalfmassRad` | 6 × 10⁻⁵ |

What this settles:
- **The archives hold raw TNG snapshot units**: ckpc/h, km/s ÷ √a and 10¹⁰ M☉/h. Without √a the
  galaxies' relative velocity is 16% (a = 0.74) to 27% (a = 0.62) too fast.
- **`SubhaloSpin` is per unit mass, with a physical length and a physical velocity**:
  Σ m (a r) × (√a v) / Σ m. Divide by h for kpc km/s.
- **The frame matrices are a pure rotation plus a shift**, the same rotation for positions and
  velocities. The velocity matrix works on particle velocities (km/s ÷ √a), not on catalog km/s.
- Not settled here: what the old frame's origin and rest velocity are. Neither galaxy, nor the pair's
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
