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
    core_velocity, hubble_rate, relative_to, orbit_rotation
                                            the frame SPAM uses: primary at the origin and at rest,
                                            secondary on +x, the orbit in the x-y plane
    disk_angles                             a spin direction -> SPAM's two disk angles
    primary_side                            which of the pair is the primary: "p" or "s"
    TngTargetConfig, measure_target         everything measured for one target at its snapshot
    spam_model                              a measured target -> SPAM's 14 parameters

The readers and the measuring functions convert nothing; units change only where `physical` or
`spam_units` is called. What the stored values mean is in each function's docstring.

The readers need `h5py` (the `tng` dependency group).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Particle types, by the TNG snapshot's PartType number.
PARTICLE_TYPES = {"gas": 0, "dm": 1, "stars": 4, "bh": 5}

# The two galaxies of a pair: the letter that starts their columns in the target table ("p_", "s_")
# and their number in an archive's galaxy labels.
GALAXY_LABELS = {"p": 0, "s": 1}

# The rules `primary_side` knows, and the masses `measure_target` can give to SPAM.
PRIMARY_RULES = ("table", "stars")
MASS_CHOICES = ("historical", "catalog", "within_radius")

# Columns every target table must have.
TABLE_COLUMNS = ("moi_2", "snap", "p_SubhaloIDRaw", "s_SubhaloIDRaw")

# Rows read at a time when only some particle types are wanted (bounds the memory used).
_CHUNK_ROWS = 4_000_000

# The side of TNG50-1's simulation box, in ckpc/h. The box wraps around: what leaves through one face
# comes back in through the opposite one, so stored coordinates run from 0 to BOX.
BOX = 35000.0

# TNG's Hubble parameter h (Planck 2015): its lengths and masses are stored "per h".
H = 0.6774

# TNG's matter density today, as a fraction of the critical density (Planck 2015). The universe is
# flat, so dark energy is 1 - OMEGA_M.
OMEGA_M = 0.3089

# The gravitational constant, in kpc (km/s)^2 / Msun.
G = 4.30091e-6

# SPAM's units. It works with G = 1, so a length and a mass fix the velocity and the time.
SPAM_LENGTH = 15.0                                  # kpc
SPAM_MASS = 1.0e11                                  # Msun
SPAM_VELOCITY = (G * SPAM_MASS / SPAM_LENGTH) ** 0.5  # km/s, about 169.3
SPAM_TIME = SPAM_LENGTH / SPAM_VELOCITY * 0.9777922  # Gyr, about 0.0866 (1 kpc / (km/s) = 0.9777922 Gyr)

# SPAM's 14 parameters, in SPAM's order (the same as `simr.simulation.PARAM_NAMES`).
SPAM_PARAMETERS = ("x", "y", "z", "vx", "vy", "vz", "mass1", "mass2", "rout1", "rout2",
                   "phi1", "phi2", "theta1", "theta2")

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


def hubble_rate(scale_factor):
    """The universe's expansion rate at a snapshot, in TNG's cosmology.

    Args:
        scale_factor: the snapshot's scale factor a = 1 / (1 + z); a number or an array.

    Returns:
        H(a) in km/s per kpc: two points a distance d kpc apart are carried apart at H * d km/s.
        Today (a = 1) it is 0.06774, which is 67.74 km/s per Mpc.

    H(a) = H0 * sqrt(OMEGA_M / a**3 + 1 - OMEGA_M): a flat universe of matter and dark energy.
    """
    scale_factor = np.asarray(scale_factor, dtype=np.float64)
    if np.any(scale_factor <= 0) or np.any(scale_factor > 1):
        raise ValueError("scale_factor must be above 0 and at most 1")
    return 0.1 * H * np.sqrt(OMEGA_M / scale_factor ** 3 + 1 - OMEGA_M)  # 100 h km/s/Mpc = 0.1 h km/s/kpc


def core_velocity(positions, velocities, masses, center, radius: float) -> np.ndarray:
    """The mean velocity of the particles within a radius of a center.

    A galaxy's nucleus moves differently from the average of its whole halo, most of all during an
    encounter. This measures the nucleus: give it the star particles, the nucleus's position and a
    small radius. Averaging over many stars removes their own motion inside the core.

    Args:
        positions: (n, 3), the particles' positions.
        velocities: (n, 3), their velocities.
        masses: (n,), their masses.
        center: (3,), the nucleus.
        radius: how far from the center a particle may be, in the units of `positions`.

    Returns:
        (3,), the mass-weighted mean velocity of the particles inside, in the units of `velocities`.
    """
    positions, velocities, masses = _particles(positions, velocities, masses)
    center = np.asarray(center, dtype=np.float64)
    if center.shape != (3,):
        raise ValueError(f"center has shape {center.shape}, expected (3,)")
    inside = np.linalg.norm(positions - center, axis=1) < radius
    if not masses[inside].sum() > 0:
        raise ValueError(f"No particles within {radius} of the center")
    return masses[inside] @ velocities[inside] / masses[inside].sum()


def relative_to(positions, velocities, center, center_velocity, scale_factor,
                hubble_flow: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Positions and velocities relative to a center: the primary at the origin and at rest.

    Args:
        positions: (3,) or (n, 3), in kpc (after `nearest_image` and `physical`).
        velocities: the same shape, in km/s.
        center: (3,), the primary's nucleus, in kpc.
        center_velocity: (3,), its velocity, in km/s.
        scale_factor: the snapshot's scale factor (used for the Hubble flow).
        hubble_flow: add the universe's expansion to the velocities. On by default.

    Returns:
        (positions, velocities) relative to the center, in kpc and km/s, as float64.

    TNG's velocities are peculiar: the expansion of the universe is taken out. The rate at which a
    physical distance really changes includes it, v = v_peculiar + H * r, and that is the velocity a
    code working in physical coordinates (SPAM) needs. The added part points along r, so it changes
    speeds a little (H is about 0.1 km/s per kpc) and never the plane of an orbit.
    """
    positions = np.asarray(positions, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64) - np.asarray(center_velocity, dtype=np.float64)
    if positions.shape != velocities.shape or positions.shape[-1] != 3:
        raise ValueError(f"positions {positions.shape} and velocities {velocities.shape} must both be (3,) or (n, 3)")
    if hubble_flow:
        velocities = velocities + hubble_rate(scale_factor) * positions
    return positions, velocities


def orbit_rotation(relative_position, relative_velocity) -> np.ndarray:
    """The rotation into the orbit's own axes: secondary on +x, the orbit in the x-y plane.

    Args:
        relative_position: (3,), the secondary's nucleus relative to the primary's.
        relative_velocity: (3,), its velocity relative to the primary's.

    Returns:
        (3, 3), a rotation matrix R. Its rows are the new axes: x toward the secondary, z along
        position x velocity (the orbit's angular momentum), y completing them. Apply it with
        `vectors @ R.T`, the same for positions, velocities and spins. Afterwards the secondary is at
        (distance, 0, 0), its velocity has no z part, and it moves counter-clockwise seen from +z.

    The plane is the one the pair moves in at this moment. A head-on pair (velocity along the line
    between the nuclei) has no such plane, which is an error.
    """
    position = np.asarray(relative_position, dtype=np.float64)
    velocity = np.asarray(relative_velocity, dtype=np.float64)
    if position.shape != (3,) or velocity.shape != (3,):
        raise ValueError(f"Expected two (3,) vectors, got {position.shape} and {velocity.shape}")
    momentum = np.cross(position, velocity)
    size = np.linalg.norm(position) * np.linalg.norm(velocity)
    if not size > 0 or np.linalg.norm(momentum) < 1e-9 * size:
        raise ValueError("No orbit plane: the relative velocity is along the line between the two nuclei")
    x_axis = position / np.linalg.norm(position)
    z_axis = momentum / np.linalg.norm(momentum)
    return np.stack([x_axis, np.cross(z_axis, x_axis), z_axis])


def disk_angles(spin) -> tuple[float, float]:
    """Turn a disk's spin direction into SPAM's two disk angles.

    SPAM builds each disk flat in the x-y plane, turning counter-clockwise seen from +z, then tilts
    it by theta about the y axis and turns it by phi about the z axis. Its spin axis ends up along
    (sin(theta) cos(phi), sin(theta) sin(phi), cos(theta)). This is the inverse.

    Args:
        spin: (3,), the disk's spin (angular momentum) in the frame SPAM will run in. Only its
            direction matters.

    Returns:
        (phi, theta) in degrees, as SPAM's `phi1`/`theta1` or `phi2`/`theta2`. theta is the angle
        from +z, 0 to 180; phi is the direction of the spin's x-y part, 0 up to 360. A spin exactly
        along z has no such direction, and phi is 0.
    """
    spin = np.asarray(spin, dtype=np.float64)
    if spin.shape != (3,):
        raise ValueError(f"spin has shape {spin.shape}, expected (3,)")
    size = np.linalg.norm(spin)
    if not size > 0:
        raise ValueError("A spin of zero has no direction")
    theta = np.degrees(np.arccos(np.clip(spin[2] / size, -1.0, 1.0)))
    in_plane = np.hypot(spin[0], spin[1])
    phi = np.degrees(np.arctan2(spin[1], spin[0])) % 360.0 if in_plane > 1e-12 * size else 0.0
    return float(phi), float(theta)


def primary_side(archive: dict, rule: str) -> str:
    """Say which galaxy of a pair is the primary: the table's "p" or its "s".

    Args:
        archive: one target's particles, as `read_archive` returns them. The "stars" rule uses
            "masses", "particle_type" and "historical_galaxy", and needs the star particles.
        rule: how to decide:

            "table"   the galaxy the target table calls the primary. Always "p".
            "stars"   the galaxy with more mass in stars, each star counted for the galaxy it
                      belonged to before the encounter ("historical_galaxy"). A tie gives "p".

    Returns:
        "p" or "s": the letter that starts the primary's columns in the target table.
        `GALAXY_LABELS` gives its number in the archive's labels.

    Why stars: during an encounter the catalog's total mass (`SubhaloMass`) swings, because the two
    galaxies share one dark halo and SUBFIND gives it to whichever it calls the central one. Stars
    sit deep in each galaxy, and their mass ratio stays steady.
    """
    if rule not in PRIMARY_RULES:
        raise ValueError(f"Unknown rule {rule!r}. Known: 'table', 'stars'")
    if rule == "table":
        return "p"
    stars = np.asarray(archive["particle_type"]) == PARTICLE_TYPES["stars"]
    masses = np.asarray(archive["masses"], dtype=np.float64)[stars]
    galaxy = np.asarray(archive["historical_galaxy"])[stars]
    mass = {side: masses[galaxy == label].sum() for side, label in GALAXY_LABELS.items()}
    if not (mass["p"] > 0 and mass["s"] > 0):
        raise ValueError("The 'stars' rule needs star particles in both galaxies")
    return "s" if mass["s"] > mass["p"] else "p"


@dataclass(frozen=True)
class TngTargetConfig:
    """The choices made when a target is measured (`measure_target`).

    Attributes:
        primary_rule: which galaxy is the primary: "stars" (the default) or "table". See `primary_side`.
        core_radius: a nucleus's velocity is the mean of the stars within this many kpc of it. 1 by
            default.
        hubble_flow: add the universe's expansion to the pair's relative velocity. On by default.
        mass: which of a galaxy's three measured masses is the one for SPAM (all three are always
            returned):

            "historical"      every particle (all types) that belonged to the galaxy before the
                              encounter. The default.
            "catalog"         the TNG catalog's `SubhaloMass`: every particle SUBFIND gives it now.
            "within_radius"   those of the same particles that are within the galaxy's disk
                              radius of its nucleus.

        disk_radius_factor: a galaxy's disk radius is this many stellar half-mass radii. 5 by default.
        spin_radius_factor: which stars give a disk's spin: those the galaxy owned before the
            encounter that are within this many stellar half-mass radii of its nucleus. 5 by default,
            the disk radius. None: all the stars it owned, however far out.
    """

    primary_rule: str = "stars"
    core_radius: float = 1.0
    hubble_flow: bool = True
    mass: str = "historical"
    disk_radius_factor: float = 5.0
    spin_radius_factor: float | None = 5.0

    def __post_init__(self):
        if self.primary_rule not in PRIMARY_RULES:
            raise ValueError(f"Unknown primary_rule {self.primary_rule!r}. Known: {list(PRIMARY_RULES)}")
        if self.mass not in MASS_CHOICES:
            raise ValueError(f"Unknown mass {self.mass!r}. Known: {list(MASS_CHOICES)}")
        for name in ("core_radius", "disk_radius_factor"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be above 0, got {getattr(self, name)}")
        if self.spin_radius_factor is not None and not self.spin_radius_factor > 0:
            raise ValueError(f"spin_radius_factor must be None or above 0, got {self.spin_radius_factor}")


def measure_target(table: dict, row: int, archive: dict, config: TngTargetConfig | None = None) -> dict:
    """Measure one target at its snapshot: the pair's orbit and each galaxy's mass, size and spin.

    Args:
        table: the target table, as `read_table` returns it.
        row: the target's row in the table.
        archive: the target's particles, as `read_archive(path, "all")` returns them. Every particle
            type is needed.
        config: the choices to make. `TngTargetConfig()` when None.

    Returns:
        A dict in physical units (kpc, km/s, Msun; spins in kpc km/s). Every vector is in the frame
        SPAM uses: the primary's nucleus at the origin and at rest, the secondary on +x, the orbit
        in the x-y plane.

        target           int      the table's `p_SubhaloIDRaw` for the row (the archive's name)
        snap             int      the snapshot
        scale_factor     float
        primary_side     str      "p" or "s": the table's letter for the galaxy used as the primary
        separation       float    the distance between the two nuclei
        position         (3,)     the secondary's nucleus: (separation, 0, 0)
        velocity         (3,)     the secondary's velocity, from the stars around each nucleus
        velocities       dict     the same by three measures: "stars" (the one used), "black_hole"
                                  (each nucleus's black hole; None when a galaxy has none) and
                                  "catalog" (`SubhaloVel`, the average over the whole halo)
        rotation         (3, 3)   from the box's axes to this frame: `vectors @ rotation.T`
        primary, secondary        one dict per galaxy:

            side               str     its letter in the table
            subhalo_id         int     its `SubhaloIDRaw`
            mass               float   the mass for SPAM: `masses[config.mass]`
            masses             dict    "historical", "catalog", "within_radius" (see `TngTargetConfig`)
            stellar_mass       float   the stars that belonged to it before the encounter
            half_mass_radius   float   the radius around its nucleus holding half of that stellar mass
            disk_radius        float   `config.disk_radius_factor` x the half-mass radius
            spin               (3,)    the spin of its stars, per unit mass
            spin_stars         int     how many stars the spin was measured from
            phi, theta         float   SPAM's disk angles for that spin, in degrees
            black_hole_offset  float   how far its black hole is from its nucleus (None without one)

    How it is measured:
        - A nucleus is the catalog's `SubhaloPos` (the most bound particle). Everything else is
          measured from the archive's particles, out of the old frame and whole across the box edge.
        - A galaxy's stars are those it owned before the encounter (`historical_galaxy`).
        - A nucleus's velocity is the mean of all stars within `config.core_radius` of it.
        - The frame comes from the nuclei's relative position and the "stars" relative velocity.
          The other two velocities are in the same frame, so they can have a z part.
    """
    config = TngTargetConfig() if config is None else config
    row = int(row)
    ids = {side: int(table[f"{side}_SubhaloIDRaw"][row]) for side in GALAXY_LABELS}
    snap = int(table["snap"][row])
    if (ids["p"], ids["s"], snap) != (archive["p_id"], archive["s_id"], archive["t_snap"]):
        raise ValueError(f"Row {row} is target {ids['p']} at snapshot {snap}, but the archive holds "
                         f"target {archive['p_id']} at snapshot {archive['t_snap']}")
    types = np.asarray(archive["particle_type"])
    stars = types == PARTICLE_TYPES["stars"]
    if stars.all():
        raise ValueError("The archive holds only stars: read it with read_archive(path, 'all')")

    scale_factor = float(table["scale_factor"][row])
    sides = {"primary": primary_side(archive, config.primary_rule)}
    sides["secondary"] = "s" if sides["primary"] == "p" else "p"
    origin = np.asarray(table[f"{sides['primary']}_SubhaloPos"][row], dtype=np.float64)

    # The particles: out of the old frame, whole across the box edge, relative to the primary's nucleus, physical.
    positions, velocities = archive_to_box(archive["positions"], archive["velocities"],
                                           table["orbital_frame_pos_transform"][row], table["orbital_frame_vel_transform"][row])
    positions = physical(nearest_image(positions, origin) - origin, "length", scale_factor)
    velocities = physical(velocities, "particle_velocity", scale_factor)
    masses = physical(archive["masses"], "mass")
    holes = np.flatnonzero(types == PARTICLE_TYPES["bh"])

    galaxies, nuclei, nucleus_velocity = {}, {}, {"stars": {}, "black_hole": {}, "catalog": {}}
    for role, side in sides.items():
        stored = np.asarray(table[f"{side}_SubhaloPos"][row], dtype=np.float64)
        nucleus = physical(nearest_image(stored, origin) - origin, "length", scale_factor)
        distance = np.linalg.norm(positions - nucleus, axis=1)
        own = np.asarray(archive["historical_galaxy"]) == GALAXY_LABELS[side]
        own_stars = own & stars
        half = half_mass_radius(positions[own_stars], masses[own_stars], nucleus)
        disk_radius = config.disk_radius_factor * half

        for_spin = own_stars if config.spin_radius_factor is None else own_stars & (distance < config.spin_radius_factor * half)
        if not for_spin.any():
            raise ValueError(f"No stars of galaxy {side!r} within {config.spin_radius_factor} half-mass radii of its nucleus")
        spin = galaxy_properties(positions[for_spin], velocities[for_spin], masses[for_spin])["spin"]

        nucleus_velocity["stars"][role] = core_velocity(positions[stars], velocities[stars], masses[stars],
                                                        nucleus, config.core_radius)
        nucleus_velocity["catalog"][role] = physical(table[f"{side}_SubhaloVel"][row], "velocity")
        hole = holes[np.argmin(distance[holes])] if len(holes) else None
        nucleus_velocity["black_hole"][role] = None if hole is None else velocities[hole]

        galaxy_masses = {"historical": float(masses[own].sum()),
                         "catalog": float(physical(table[f"{side}_SubhaloMass"][row], "mass")),
                         "within_radius": float(masses[own & (distance < disk_radius)].sum())}
        nuclei[role] = nucleus
        galaxies[role] = {"side": side, "subhalo_id": ids[side], "mass": galaxy_masses[config.mass],
                          "masses": galaxy_masses, "stellar_mass": float(masses[own_stars].sum()),
                          "half_mass_radius": half, "disk_radius": disk_radius, "spin": spin,
                          "spin_stars": int(for_spin.sum()),
                          "black_hole_offset": None if hole is None else float(distance[hole])}

    # The secondary relative to the primary, by each measure of velocity; the frame from the stars' measure.
    relative = {}
    for measure, velocity in nucleus_velocity.items():
        if velocity["primary"] is None or velocity["secondary"] is None:
            relative[measure] = None
            continue
        position, relative[measure] = relative_to(nuclei["secondary"], velocity["secondary"], nuclei["primary"],
                                                  velocity["primary"], scale_factor, config.hubble_flow)
    rotation = orbit_rotation(position, relative["stars"])

    for galaxy in galaxies.values():
        galaxy["spin"] = galaxy["spin"] @ rotation.T
        galaxy["phi"], galaxy["theta"] = disk_angles(galaxy["spin"])
    velocities_by_measure = {measure: None if velocity is None else velocity @ rotation.T
                             for measure, velocity in relative.items()}
    return {"target": ids["p"], "snap": snap, "scale_factor": scale_factor, "primary_side": sides["primary"],
            "separation": float(np.linalg.norm(position)), "position": position @ rotation.T,
            "velocity": velocities_by_measure["stars"], "velocities": velocities_by_measure,
            "rotation": rotation, "primary": galaxies["primary"], "secondary": galaxies["secondary"]}


def spam_model(target: dict) -> np.ndarray:
    """Turn a measured target into SPAM's 14 parameters: the model SPAM runs.

    Args:
        target: one target, as `measure_target` returns it.

    Returns:
        (14,), in the order of `SPAM_PARAMETERS` and in SPAM's units:

        x, y, z              the secondary's nucleus          `position` / 15 kpc: (separation, 0, 0)
        vx, vy, vz           its velocity                     `velocity` / about 169.3 km/s: no z part
        mass1, mass2         the primary's and secondary's    each galaxy's `mass` / 1e11 Msun
        rout1, rout2         their disk radii                 each `disk_radius` / 15 kpc
        phi1, phi2           their disk angles, in degrees    each `phi`
        theta1, theta2                                        each `theta`

    Nothing is measured or chosen here: every choice was made by the `TngTargetConfig` the target was
    measured with.
    """
    primary, secondary = target["primary"], target["secondary"]
    return np.concatenate([
        spam_units(target["position"], "length"),
        spam_units(target["velocity"], "velocity"),
        spam_units([primary["mass"], secondary["mass"]], "mass"),
        spam_units([primary["disk_radius"], secondary["disk_radius"]], "length"),
        [primary["phi"], secondary["phi"], primary["theta"], secondary["theta"]],
    ])
