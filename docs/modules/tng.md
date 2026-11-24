# TNG targets (`tng.py`)

Readers for the IllustrisTNG (TNG50-1) data this project already has. Both files are HDF5 and live
in the data folder, outside the repo. The readers return the values **exactly as stored**: no unit
changes and no change of frame. Converting them comes later, in its own functions.

```python
from galactic_tidal_constraints import tng

table = tng.read_table("../data/tng-data/moi-4-targets.h5")
archive = tng.read_archive("../data/tng-data/target-archive-62000000190717.h5")           # stars
archive = tng.read_archive("../data/tng-data/target-archive-62000000190717.h5", "all")    # every type
```

Only what exists today is described here. `tng.py` needs `h5py`, which is in the `tng` dependency
group (`uv sync --group tng`; a plain `uv sync` installs it too, for the tests). Decisions behind it
are in [PLAN.md](../../PLAN.md).

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
| `p_`, `s_` + a catalog field (`SubhaloPos`, `SubhaloCM`, `SubhaloHalfmassRad`, `SubhaloVel`, `SubhaloMass`, `SubhaloSpin`) | Raw TNG catalog values: lengths in ckpc/h, velocities in km/s, masses in 10¹⁰ M☉/h, spin in (kpc/h)(km/s) |
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
| `velocities` | (n, 3) `float32` | km/s · √a: multiply by √a for km/s |
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

## Known issues
- `"stars"` includes wind particles: the archives have no formation time to tell them apart.
- A black hole's mass is its `BH_Mass`, not its dynamical mass.

## Tests
`tests/test_tng.py`. The tests on small made-up files always run. The tests on the real files run
only when `GTC_DATA_DIR` names the data folder (the one holding `tng-data/`):

```bash
GTC_DATA_DIR=../data uv run pytest tests/test_tng.py
```
