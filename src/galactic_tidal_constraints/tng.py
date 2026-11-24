"""TNG targets: readers for the IllustrisTNG data this project already has.

Two kinds of file, both HDF5, both made by the project's earlier TNG pipeline:

    the target table      one row per encounter per snapshot: IDs, catalog values, the old
                          pipeline's own values and its frame matrices
    a target archive      one target's particles at its target snapshot

The readers return the values exactly as stored. Nothing is converted or corrected here: no unit
changes, no change of frame. What the stored values mean is in each function's docstring.

Needs `h5py` (the `tng` dependency group).
"""

from pathlib import Path

import numpy as np

# Particle types, by the TNG snapshot's PartType number.
PARTICLE_TYPES = {"gas": 0, "dm": 1, "stars": 4, "bh": 5}

# Columns every target table must have.
TABLE_COLUMNS = ("moi_2", "snap", "p_SubhaloIDRaw", "s_SubhaloIDRaw")

# Rows read at a time when only some particle types are wanted (bounds the memory used).
_CHUNK_ROWS = 4_000_000


def _open(path, what: str):
    """Open an HDF5 file for reading, with a clear error when it isn't there."""
    import h5py

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{what} not found: {path}")
    return h5py.File(path, "r")


def read_table(path) -> dict[str, np.ndarray]:
    """Read the target table: one array per column, one row per encounter per snapshot.

    Args:
        path: the table file (`moi-4-targets.h5`).

    Returns:
        A dict of column name to array, in the table's column order. Every array has one entry per
        row: a number column is (rows,), a vector column (rows, 3), a matrix column (rows, 4, 4),
        and a text column is an array of `str`.

    The values are as stored:
        - `moi_2` identifies an encounter, `snap` the snapshot; `p_` is the primary galaxy and `s_`
          the secondary. IDs and snapshot numbers are `int64`.
        - `p_` and `s_` catalog columns (`SubhaloPos`, `SubhaloVel`, `SubhaloMass`, ...) are raw TNG
          catalog values: lengths in ckpc/h, masses in 1e10 Msun/h, velocities in km/s.
        - `pa_`, `sa_`, `ps_`, `ss_`, `_orbital_frame` and `_transform` columns are the earlier
          pipeline's own values; their units aren't verified.
        - Each dataset in the file has a `units` attribute, and some a `note`.
    """
    with _open(path, "Target table") as file:
        if "columns" in file.attrs:
            columns = [name.decode() if isinstance(name, bytes) else str(name) for name in file.attrs["columns"]]
        else:
            columns = list(file.keys())
        missing = [name for name in (*TABLE_COLUMNS, *columns) if name not in file]
        if missing:
            raise ValueError(f"Target table {path} has no column {missing[0]!r}")

        table = {}
        for name in columns:
            dataset = file[name]
            table[name] = dataset.asstr()[...].astype(str) if dataset.dtype.kind == "O" else dataset[...]

    rows = {len(array) for array in table.values()}
    if len(rows) != 1:
        raise ValueError(f"Target table {path} has columns of different lengths: {sorted(rows)}")
    return table


def read_archive(path, particle_types=("stars",)) -> dict:
    """Read one target archive: the particles of both galaxies at the target snapshot.

    Args:
        path: the archive file (`target-archive-<primary SubhaloIDRaw>.h5`).
        particle_types: which particles to return: any of "gas", "dm", "stars" and "bh" (one name
            or several), or "all". Stars by default.

    Returns:
        A dict, with one row per returned particle in each array (the file's order is kept):

        positions           (n, 3) float32   ckpc/h
        velocities          (n, 3) float32   km/s * sqrt(a): multiply by sqrt(a) for km/s
        masses              (n,)   float32   1e10 Msun/h
        particle_type       (n,)   int8      0 gas, 1 dark matter, 4 stars, 5 black hole
        subfind_galaxy      (n,)   int8      the galaxy SUBFIND gives the particle to: 0 primary, 1 secondary
        historical_galaxy   (n,)   int8      the galaxy it belonged to before the encounter: 0 or 1
        p_id, s_id          int              the primary's and secondary's SubhaloIDRaw
        t_snap              int              the target snapshot

    The values are as stored, which is not what the file's own labels say ("kpc", "km_per_s",
    "1e10_Msun"): they are raw TNG snapshot units. Positions and velocities are in the earlier
    pipeline's orbital frame (the table's `orbital_frame_pos_transform` and
    `orbital_frame_vel_transform` for that row), not the simulation box.

    Known limits of these files:
        - "stars" includes wind particles: the archive has no formation time to tell them apart.
        - A black hole's mass is its `BH_Mass`, not its dynamical mass.
    """
    if isinstance(particle_types, str):
        particle_types = (particle_types,)
    if "all" in particle_types:
        wanted = None
    else:
        unknown = [name for name in particle_types if name not in PARTICLE_TYPES]
        if unknown or not particle_types:
            raise ValueError(f"Unknown particle type {unknown[0] if unknown else particle_types!r}. "
                             f"Known: {list(PARTICLE_TYPES)} or 'all'")
        wanted = [PARTICLE_TYPES[name] for name in particle_types]

    with _open(path, "Target archive") as file:
        for name, width in (("particles_orf", 7), ("particles_meta", 3)):
            if name not in file:
                raise ValueError(f"Target archive {path} has no dataset {name!r}")
            if file[name].ndim != 2 or file[name].shape[1] != width:
                raise ValueError(f"Target archive {path}: {name!r} has shape {file[name].shape}, "
                                 f"expected (n, {width})")
        particles, meta = file["particles_orf"], file["particles_meta"][...]
        if len(meta) != len(particles):
            raise ValueError(f"Target archive {path}: {len(particles)} particles but {len(meta)} labels")

        if wanted is None:
            data = particles[...]
        else:
            keep = np.isin(meta[:, 0], wanted)
            data = np.empty((int(keep.sum()), 7), dtype=particles.dtype)
            filled = 0
            for start in range(0, len(keep), _CHUNK_ROWS):
                rows = keep[start:start + _CHUNK_ROWS]
                if rows.any():
                    chunk = particles[start:start + _CHUNK_ROWS][rows]
                    data[filled:filled + len(chunk)] = chunk
                    filled += len(chunk)
            meta = meta[keep]
        ids = {name: int(file.attrs[name]) for name in ("p_id", "s_id", "t_snap")}

    return {
        "positions": data[:, 0:3],
        "velocities": data[:, 3:6],
        "masses": data[:, 6],
        "particle_type": meta[:, 0],
        "subfind_galaxy": meta[:, 1],
        "historical_galaxy": meta[:, 2],
        **ids,
    }
