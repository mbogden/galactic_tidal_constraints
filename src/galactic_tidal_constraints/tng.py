"""TNG targets: the IllustrisTNG data this project already has.

Two kinds of file, both HDF5, both made by the project's earlier TNG pipeline:

    the target table      one row per encounter per snapshot: IDs, catalog values, the old
                          pipeline's own values and its frame matrices
    a target archive      one target's particles at its target snapshot

    read_table, read_archive                return the values exactly as stored
    galaxy_properties, half_mass_radius     measure a galaxy from its particles, in the units given
    archive_to_box                          take an archive's particles out of the old frame
    nearest_image                           undo the simulation box's wrap-around
    physical, spam_units                    stored values -> kpc, km/s, Msun -> SPAM's units

The readers and the measuring functions convert nothing; units change only where `physical` or
`spam_units` is called. What the stored values mean is in each function's docstring.

The readers need `h5py` (the `tng` dependency group).
"""

from pathlib import Path

import numpy as np

# Particle types, by the TNG snapshot's PartType number.
PARTICLE_TYPES = {"gas": 0, "dm": 1, "stars": 4, "bh": 5}

# Columns every target table must have.
TABLE_COLUMNS = ("moi_2", "snap", "p_SubhaloIDRaw", "s_SubhaloIDRaw")

# Rows read at a time when only some particle types are wanted (bounds the memory used).
_CHUNK_ROWS = 4_000_000

# The side of TNG50-1's simulation box, in ckpc/h. The box wraps around: what leaves through one face
# comes back in through the opposite one, so stored coordinates run from 0 to BOX.
BOX = 35000.0

# TNG's Hubble parameter h (Planck 2015): its lengths and masses are stored "per h".
H = 0.6774

# The gravitational constant, in kpc (km/s)^2 / Msun.
G = 4.30091e-6

# SPAM's units. It works with G = 1, so a length and a mass fix the velocity and the time.
SPAM_LENGTH = 15.0                                  # kpc
SPAM_MASS = 1.0e11                                  # Msun
SPAM_VELOCITY = (G * SPAM_MASS / SPAM_LENGTH) ** 0.5  # km/s, about 169.3
SPAM_TIME = SPAM_LENGTH / SPAM_VELOCITY * 0.9777922  # Gyr, about 0.0866 (1 kpc / (km/s) = 0.9777922 Gyr)

# What a stored TNG value is multiplied by to become physical: (constant, power of the scale factor).
_TO_PHYSICAL = {
    "length": (1 / H, 1.0),             # ckpc/h          -> kpc
    "velocity": (1.0, 0.0),             # km/s            -> km/s   (catalog velocities)
    "particle_velocity": (1.0, 0.5),    # km/s / sqrt(a)  -> km/s   (snapshot particles)
    "mass": (1e10 / H, 0.0),            # 1e10 Msun/h     -> Msun
    "spin": (1 / H, 0.0),               # (kpc/h)(km/s)   -> kpc km/s   (catalog SubhaloSpin)
}

# What a physical value is divided by to be in SPAM's units.
_SPAM_UNIT = {"length": SPAM_LENGTH, "velocity": SPAM_VELOCITY, "mass": SPAM_MASS, "time": SPAM_TIME}


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
          `SubhaloSpin` is the angular momentum per unit mass, in (kpc/h)(km/s) with a physical
          (not comoving) length.
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
        velocities          (n, 3) float32   km/s / sqrt(a): multiply by sqrt(a) for km/s
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


def _particles(positions, velocities, masses):
    """The three arrays as float64, checked: (n, 3), (n, 3) or None, (n,), with some mass."""
    positions, masses = np.asarray(positions, dtype=np.float64), np.asarray(masses, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions has shape {positions.shape}, expected (n, 3)")
    if masses.shape != (len(positions),):
        raise ValueError(f"masses has shape {masses.shape}, expected ({len(positions)},)")
    if velocities is not None:
        velocities = np.asarray(velocities, dtype=np.float64)
        if velocities.shape != positions.shape:
            raise ValueError(f"velocities has shape {velocities.shape}, expected {positions.shape}")
    if len(masses) == 0 or not masses.sum() > 0:
        raise ValueError("No mass: a galaxy needs at least one particle with mass")
    return positions, velocities, masses


def galaxy_properties(positions, velocities, masses) -> dict:
    """Measure one galaxy from its particles: mass, center of mass, velocity and spin.

    Args:
        positions: (n, 3), the particles' positions.
        velocities: (n, 3), their velocities.
        masses: (n,), their masses.

    Returns:
        A dict (sums are done in float64):

        mass             float   the total mass
        center_of_mass   (3,)    the mass-weighted mean position
        velocity         (3,)    the mass-weighted mean velocity
        spin             (3,)    the angular momentum per unit mass, about the center of mass in the
                                 galaxy's rest frame: sum(m (r - center) x (v - velocity)) / mass

    Nothing is converted: the results are in the units of the arrays given (a spin is a length times
    a velocity). The spin doesn't depend on where the origin is or on any velocity added to every
    particle, and it turns with the frame.

    For a target archive's particles (ckpc/h, km/s / sqrt(a), 1e10 Msun/h), with a the snapshot's
    scale factor, these match the TNG catalog's `SubhaloMass`, `SubhaloCM` (in the archive's frame),
    `SubhaloVel` (= velocity * sqrt(a)) and `SubhaloSpin` (= spin * a**1.5).
    """
    positions, velocities, masses = _particles(positions, velocities, masses)
    mass = masses.sum()
    center = masses @ positions / mass
    velocity = masses @ velocities / mass
    spin = masses @ np.cross(positions - center, velocities - velocity) / mass
    return {"mass": float(mass), "center_of_mass": center, "velocity": velocity, "spin": spin}


def half_mass_radius(positions, masses, center) -> float:
    """The radius around a center that holds half of the particles' mass.

    Args:
        positions: (n, 3), the particles' positions.
        masses: (n,), their masses.
        center: (3,), the point to measure from. The TNG catalog's `SubhaloHalfmassRad` is measured
            from `SubhaloPos` (the most bound particle), not from the center of mass.

    Returns:
        The distance of the first particle, going outward, at which the mass inside reaches half the
        total. In the units of `positions`.
    """
    positions, _, masses = _particles(positions, None, masses)
    center = np.asarray(center, dtype=np.float64)
    if center.shape != (3,):
        raise ValueError(f"center has shape {center.shape}, expected (3,)")
    radii = np.linalg.norm(positions - center, axis=1)
    order = np.argsort(radii)
    inside = np.cumsum(masses[order])
    return float(radii[order][np.searchsorted(inside, inside[-1] / 2)])


def physical(values, quantity: str, scale_factor=None) -> np.ndarray:
    """Convert stored TNG values to physical units: kpc, km/s, Msun, kpc km/s.

    Args:
        values: a number or an array of stored values (a table column, or an archive's positions,
            velocities or masses).
        quantity: what the values are:

            "length"              ckpc/h           x a / h       -> kpc
            "velocity"            km/s             unchanged     -> km/s      catalog velocities
            "particle_velocity"   km/s / sqrt(a)   x sqrt(a)     -> km/s      snapshot particles
            "mass"                1e10 Msun/h      x 1e10 / h    -> Msun
            "spin"                (kpc/h)(km/s)    / h           -> kpc km/s  catalog `SubhaloSpin`

        scale_factor: the snapshot's scale factor a = 1 / (1 + z). Needed for "length" and
            "particle_velocity". One number, or one per row of `values` (a table column each).

    Returns:
        The converted values, as float64, in the shape of `values`.

    h is TNG's 0.6774. A spin measured from particles has no row here: convert the positions and
    velocities first, and `galaxy_properties` then gives the spin in kpc km/s.
    """
    if quantity not in _TO_PHYSICAL:
        raise ValueError(f"Unknown quantity {quantity!r}. Known: {list(_TO_PHYSICAL)}")
    constant, power = _TO_PHYSICAL[quantity]
    values = np.asarray(values, dtype=np.float64)
    if power == 0:
        return values * constant
    if scale_factor is None:
        raise ValueError(f"A {quantity!r} needs the snapshot's scale factor")
    scale_factor = np.asarray(scale_factor, dtype=np.float64)
    if np.any(scale_factor <= 0) or np.any(scale_factor > 1):
        raise ValueError("scale_factor must be above 0 and at most 1")
    if scale_factor.ndim == 1:  # one per row: line it up with the rows of `values`
        if values.ndim == 0 or len(scale_factor) != len(values):
            raise ValueError(f"{len(scale_factor)} scale factors for values of shape {values.shape}")
        scale_factor = scale_factor.reshape(-1, *[1] * (values.ndim - 1))
    elif scale_factor.ndim > 1:
        raise ValueError("scale_factor must be one number, or one per row")
    return values * (constant * scale_factor ** power)


def spam_units(values, quantity: str) -> np.ndarray:
    """Convert physical values to SPAM's units.

    Args:
        values: a number or an array, in physical units.
        quantity: what the values are:

            "length"     kpc    / 15           (`SPAM_LENGTH`)
            "velocity"   km/s   / about 169.3  (`SPAM_VELOCITY`)
            "mass"       Msun   / 1e11         (`SPAM_MASS`)
            "time"       Gyr    / about 0.0866 (`SPAM_TIME`)

    Returns:
        The values in SPAM's units, as float64.

    SPAM works with G = 1: its length and mass units are Wallin's, and the velocity and time units
    follow from them, velocity = sqrt(G mass / length) and time = length / velocity.
    """
    if quantity not in _SPAM_UNIT:
        raise ValueError(f"Unknown quantity {quantity!r}. Known: {list(_SPAM_UNIT)}")
    return np.asarray(values, dtype=np.float64) / _SPAM_UNIT[quantity]


def nearest_image(positions, reference, box: float = BOX) -> np.ndarray:
    """Move positions to the copy nearest a reference point, undoing the box's wrap-around.

    The simulation box wraps around, so two points close together can sit on either side of a face
    with coordinates almost a whole box apart (34950 and 50, say). This shifts each position by whole
    box lengths until every coordinate is within half a box of the reference (50 becomes 35050), so
    that differences and averages come out right.

    Args:
        positions: (3,) or (n, 3), stored positions in the box frame (ckpc/h).
        reference: (3,), the point to stay near: for a galaxy's particles, its center. Or (n, 3),
            one reference per position: `nearest_image(secondary_track, primary_track)`.
        box: the box's side, in the units of `positions`. TNG50-1's by default.

    Returns:
        The positions as float64, in the shape of `positions`. They may lie outside 0 to `box`.

    Call it on stored values, before `physical`: the box is 35000 only in comoving ckpc/h. It can't
    be used on an archive's positions as read, which are in the old rotated frame.
    """
    if not box > 0:
        raise ValueError(f"box must be above 0, got {box}")
    positions = np.asarray(positions, dtype=np.float64)
    return positions - box * np.round((positions - np.asarray(reference, dtype=np.float64)) / box)


def archive_to_box(positions, velocities, pos_matrix, vel_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Take an archive's particles out of the old frame, back to the simulation box.

    The archives store particles in the earlier pipeline's frame: the box frame turned by a rotation
    and moved by a shift, x_old = R x_box + t, with one 4x4 matrix for positions and one for
    velocities. This undoes both, exactly: x_box = R^T (x_old - t).

    Args:
        positions: (n, 3), an archive's positions as read.
        velocities: (n, 3), its velocities as read.
        pos_matrix: (4, 4), the target's `orbital_frame_pos_transform` from the target table.
        vel_matrix: (4, 4), its `orbital_frame_vel_transform`.

    Returns:
        (positions, velocities) in the box frame, as float64 and still in stored units (ckpc/h and
        km/s / sqrt(a)).

    A galaxy that sits across a face of the box comes back in two parts, as TNG stored it: follow
    with `nearest_image` around the galaxy's center.
    """
    results = []
    for values, matrix, name in ((positions, pos_matrix, "pos_matrix"), (velocities, vel_matrix, "vel_matrix")):
        values, matrix = np.asarray(values, dtype=np.float64), np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError(f"Expected (n, 3) particles, got shape {values.shape}")
        if matrix.shape != (4, 4) or not np.array_equal(matrix[3], [0, 0, 0, 1]):
            raise ValueError(f"{name} must be a (4, 4) matrix with last row 0, 0, 0, 1")
        rotation, shift = matrix[:3, :3], matrix[:3, 3]
        if np.abs(rotation @ rotation.T - np.eye(3)).max() > 1e-6 or np.linalg.det(rotation) < 0:
            raise ValueError(f"{name} doesn't hold a rotation")
        results.append((values - shift) @ rotation)  # row form of R^T (x - t)
    return results[0], results[1]
