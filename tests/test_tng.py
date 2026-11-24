"""Tests for tng.py: reading the target table and a target archive, measuring a galaxy, and units.

The first sections use small made-up files and particles and always run. The last section reads the real files and
runs only when the environment variable GTC_DATA_DIR names the workspace's data folder (the one
holding `tng-data/`).
"""

import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")

from galactic_tidal_constraints import tng  # noqa: E402

TEXT = h5py.string_dtype("utf-8")


# --- Made-up files ---------------------------------------------------------------------------------------


@pytest.fixture
def table_file(tmp_path):
    """A 3-row table with one column of each kind: integers, numbers, vectors, matrices, text."""
    path = tmp_path / "targets.h5"
    columns = ["moi_2", "snap", "p_SubhaloIDRaw", "s_SubhaloIDRaw", "p_SubhaloMass", "p_SubhaloPos",
               "orbital_frame_pos_transform", "img_name_1"]
    with h5py.File(path, "w") as file:
        file.attrs["columns"] = np.array(columns, dtype=TEXT)
        file["moi_2"] = np.array([7, 7, 9], dtype=np.int64)
        file["snap"] = np.array([61, 62, 75], dtype=np.int64)
        file["p_SubhaloIDRaw"] = np.array([61000000186579, 62000000190717, 75000000121160], dtype=np.int64)
        file["s_SubhaloIDRaw"] = np.array([61000000186580, 62000000190718, 75000000121161], dtype=np.int64)
        file["p_SubhaloMass"] = np.array([1.5, 2.5, 3.5])
        file["p_SubhaloPos"] = np.arange(9, dtype=np.float32).reshape(3, 3)
        file["orbital_frame_pos_transform"] = np.tile(np.eye(4), (3, 1, 1))
        file.create_dataset("img_name_1", data=np.array(["a.png", "b.png", "c.png"], dtype=object), dtype=TEXT)
    return path


# Particle types of the made-up archive's 8 particles, in the file's order.
TYPES = np.array([0, 1, 4, 4, 5, 1, 4, 0], dtype=np.int8)


@pytest.fixture
def archive_file(tmp_path):
    """An 8-particle archive: both galaxies, all four particle types."""
    path = tmp_path / "target-archive-62000000190717.h5"
    particles = np.arange(56, dtype=np.float32).reshape(8, 7)
    meta = np.stack([TYPES, [0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 0, 1, 1, 1]], axis=1).astype(np.int8)
    with h5py.File(path, "w") as file:
        file.attrs["p_id"], file.attrs["s_id"], file.attrs["t_snap"] = 62000000190717, 62000000190718, 62
        file["particles_orf"] = particles
        file["particles_meta"] = meta
    return path


# --- The target table ------------------------------------------------------------------------------------


def test_the_table_comes_back_as_one_array_per_column_in_the_files_order(table_file):
    table = tng.read_table(table_file)
    assert list(table) == ["moi_2", "snap", "p_SubhaloIDRaw", "s_SubhaloIDRaw", "p_SubhaloMass",
                           "p_SubhaloPos", "orbital_frame_pos_transform", "img_name_1"]
    assert table["p_SubhaloMass"].shape == (3,)
    assert table["p_SubhaloPos"].shape == (3, 3) and table["p_SubhaloPos"].dtype == np.float32
    assert table["orbital_frame_pos_transform"].shape == (3, 4, 4)
    assert table["p_SubhaloPos"][2].tolist() == [6.0, 7.0, 8.0]


def test_ids_are_integers_and_match_a_full_id_exactly(table_file):
    table = tng.read_table(table_file)
    assert table["p_SubhaloIDRaw"].dtype == np.int64 and table["snap"].dtype == np.int64
    assert np.flatnonzero(table["p_SubhaloIDRaw"] == 62000000190717).tolist() == [1]


def test_text_columns_are_str(table_file):
    names = tng.read_table(table_file)["img_name_1"]
    assert names.tolist() == ["a.png", "b.png", "c.png"] and isinstance(names[0], str)


def test_a_table_without_a_columns_list_returns_every_dataset(table_file):
    with h5py.File(table_file, "r+") as file:
        del file.attrs["columns"]
    assert set(tng.read_table(table_file)) == {"moi_2", "snap", "p_SubhaloIDRaw", "s_SubhaloIDRaw",
                                               "p_SubhaloMass", "p_SubhaloPos",
                                               "orbital_frame_pos_transform", "img_name_1"}


@pytest.mark.parametrize("column", ["snap", "p_SubhaloMass"])
def test_a_missing_column_is_named_in_the_error(table_file, column):
    with h5py.File(table_file, "r+") as file:
        del file[column]
    with pytest.raises(ValueError, match=f"no column '{column}'"):
        tng.read_table(table_file)


def test_columns_of_different_lengths_are_an_error(table_file):
    with h5py.File(table_file, "r+") as file:
        del file["p_SubhaloMass"]
        file["p_SubhaloMass"] = np.array([1.5, 2.5])
    with pytest.raises(ValueError, match="different lengths"):
        tng.read_table(table_file)


def test_a_missing_table_file_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Target table not found"):
        tng.read_table(tmp_path / "nothing.h5")


# --- A target archive ------------------------------------------------------------------------------------


def test_stars_are_returned_by_default_in_the_files_order(archive_file):
    archive = tng.read_archive(archive_file)
    assert archive["particle_type"].tolist() == [4, 4, 4]
    assert archive["positions"].tolist() == [[14.0, 15.0, 16.0], [21.0, 22.0, 23.0], [42.0, 43.0, 44.0]]
    assert archive["velocities"][0].tolist() == [17.0, 18.0, 19.0]
    assert archive["masses"].tolist() == [20.0, 27.0, 48.0]
    assert archive["subfind_galaxy"].tolist() == [0, 0, 1]
    assert archive["historical_galaxy"].tolist() == [0, 1, 1]


def test_shapes_types_and_ids(archive_file):
    archive = tng.read_archive(archive_file, "all")
    assert archive["positions"].shape == (8, 3) and archive["velocities"].shape == (8, 3)
    assert archive["masses"].shape == (8,)
    assert all(archive[name].dtype == np.float32 for name in ("positions", "velocities", "masses"))
    assert all(archive[name].dtype == np.int8
               for name in ("particle_type", "subfind_galaxy", "historical_galaxy"))
    assert (archive["p_id"], archive["s_id"], archive["t_snap"]) == (62000000190717, 62000000190718, 62)
    assert all(type(archive[name]) is int for name in ("p_id", "s_id", "t_snap"))


@pytest.mark.parametrize("asked, kept", [
    ("all", TYPES.tolist()),
    ("dm", [1, 1]),
    (("gas", "bh"), [0, 5, 0]),
    (("stars", "dm"), [1, 4, 4, 1, 4]),
    (("stars", "all"), TYPES.tolist()),
])
def test_only_the_asked_particle_types_are_kept(archive_file, asked, kept):
    assert tng.read_archive(archive_file, asked)["particle_type"].tolist() == kept


def test_reading_in_chunks_gives_the_same_particles(archive_file, monkeypatch):
    whole = tng.read_archive(archive_file, ("stars", "dm"))
    monkeypatch.setattr(tng, "_CHUNK_ROWS", 3)
    chunked = tng.read_archive(archive_file, ("stars", "dm"))
    assert all(np.array_equal(whole[name], chunked[name]) for name in whole)


def test_a_type_with_no_particles_gives_empty_arrays(archive_file):
    with h5py.File(archive_file, "r+") as file:
        file["particles_meta"][:, 0] = 1
    archive = tng.read_archive(archive_file, "stars")
    assert archive["positions"].shape == (0, 3) and archive["masses"].shape == (0,)


@pytest.mark.parametrize("asked", ["star", ("stars", "wind"), ()])
def test_an_unknown_particle_type_is_an_error(archive_file, asked):
    with pytest.raises(ValueError, match="Unknown particle type"):
        tng.read_archive(archive_file, asked)


def test_a_missing_dataset_is_named_in_the_error(archive_file):
    with h5py.File(archive_file, "r+") as file:
        del file["particles_meta"]
    with pytest.raises(ValueError, match="no dataset 'particles_meta'"):
        tng.read_archive(archive_file)


def test_a_wrong_shape_is_an_error(archive_file):
    with h5py.File(archive_file, "r+") as file:
        del file["particles_orf"]
        file["particles_orf"] = np.zeros((8, 6), dtype=np.float32)
    with pytest.raises(ValueError, match=r"expected \(n, 7\)"):
        tng.read_archive(archive_file)


def test_a_missing_archive_file_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Target archive not found"):
        tng.read_archive(tmp_path / "target-archive-1.h5")


# --- Measuring a galaxy ----------------------------------------------------------------------------------


def ring(radius=2.0, speed=3.0, n=360):
    """Equal-mass particles on a circle in the x-y plane, all moving counter-clockwise at one speed."""
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
    positions = radius * np.stack([np.cos(angle), np.sin(angle), np.zeros(n)], axis=1)
    velocities = speed * np.stack([-np.sin(angle), np.cos(angle), np.zeros(n)], axis=1)
    return positions, velocities, np.full(n, 0.5)


def test_a_ring_has_spin_radius_times_speed_along_its_axis():
    properties = tng.galaxy_properties(*ring(radius=2.0, speed=3.0, n=360))
    assert properties["mass"] == pytest.approx(180.0)
    assert properties["spin"] == pytest.approx([0, 0, 6.0], abs=1e-12)
    assert properties["center_of_mass"] == pytest.approx([0, 0, 0], abs=1e-12)
    assert properties["velocity"] == pytest.approx([0, 0, 0], abs=1e-12)


def test_moving_the_whole_galaxy_changes_its_center_and_velocity_but_not_its_spin():
    positions, velocities, masses = ring()
    properties = tng.galaxy_properties(positions + [30000.0, -5.0, 7.0], velocities + [100.0, 250.0, -40.0], masses)
    assert properties["center_of_mass"] == pytest.approx([30000.0, -5.0, 7.0])
    assert properties["velocity"] == pytest.approx([100.0, 250.0, -40.0])
    assert properties["spin"] == pytest.approx([0, 0, 6.0], abs=1e-9)


def test_the_spin_turns_with_the_galaxy():
    positions, velocities, masses = ring()
    turn = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])  # the z axis goes to -y
    properties = tng.galaxy_properties(positions @ turn.T, velocities @ turn.T, masses)
    assert properties["spin"] == pytest.approx([0, -6.0, 0], abs=1e-12)


def test_the_center_velocity_and_spin_are_weighted_by_mass():
    properties = tng.galaxy_properties([[0.0, 0, 0], [4.0, 0, 0]], [[0.0, 0, 0], [0, 8.0, 0]], [3.0, 1.0])
    assert properties["mass"] == 4.0
    assert properties["center_of_mass"].tolist() == [1.0, 0, 0]
    assert properties["velocity"].tolist() == [0, 2.0, 0]
    # About the center, in the rest frame: (3 * (-1 * -2) + 1 * (3 * 6)) / 4 = 6.
    assert properties["spin"].tolist() == [0, 0, 6.0]


def test_float32_particles_are_summed_in_float64():
    positions, velocities, masses = (array.astype(np.float32) for array in ring())
    properties = tng.galaxy_properties(positions, velocities, masses)
    assert type(properties["mass"]) is float
    assert all(properties[name].dtype == np.float64 for name in ("center_of_mass", "velocity", "spin"))


def test_the_half_mass_radius_is_where_the_mass_inside_reaches_half():
    positions = np.array([[1.0, 0, 0], [0, 2.0, 0], [0, 0, -3.0], [4.0, 0, 0]])
    assert tng.half_mass_radius(positions, [1.0, 1.0, 1.0, 1.0], [0, 0, 0]) == 2.0
    assert tng.half_mass_radius(positions, [3.0, 1.0, 1.0, 1.0], [0, 0, 0]) == 1.0
    assert tng.half_mass_radius(positions, [1.0, 1.0, 1.0, 9.0], [0, 0, 0]) == 4.0


def test_the_half_mass_radius_is_measured_from_the_center_given():
    positions = np.array([[1.0, 0, 0], [0, 2.0, 0], [0, 0, -3.0], [4.0, 0, 0]])
    assert tng.half_mass_radius(positions + [10.0, 20.0, 30.0], np.ones(4), [10.0, 20.0, 30.0]) == 2.0
    assert tng.half_mass_radius(positions, np.ones(4), [4.0, 0, 0]) == 3.0  # distances 3, sqrt(20), 5, 0


@pytest.mark.parametrize("positions, velocities, masses, message", [
    (np.zeros((4, 2)), np.zeros((4, 2)), np.ones(4), r"positions has shape \(4, 2\)"),
    (np.zeros((4, 3)), np.zeros((3, 3)), np.ones(4), r"velocities has shape \(3, 3\)"),
    (np.zeros((4, 3)), np.zeros((4, 3)), np.ones(5), r"masses has shape \(5,\)"),
    (np.zeros((0, 3)), np.zeros((0, 3)), np.ones(0), "No mass"),
    (np.zeros((4, 3)), np.zeros((4, 3)), np.zeros(4), "No mass"),
])
def test_wrong_shapes_and_no_mass_are_errors(positions, velocities, masses, message):
    with pytest.raises(ValueError, match=message):
        tng.galaxy_properties(positions, velocities, masses)


def test_half_mass_radius_errors():
    with pytest.raises(ValueError, match="No mass"):
        tng.half_mass_radius(np.zeros((0, 3)), np.ones(0), [0, 0, 0])
    with pytest.raises(ValueError, match=r"center has shape \(2,\)"):
        tng.half_mass_radius(np.zeros((4, 3)), np.ones(4), [0, 0])


# --- The box edge ----------------------------------------------------------------------------------------


def test_a_position_across_the_face_comes_back_beside_the_reference():
    assert tng.nearest_image([50.0, 100.0, 34990.0], [34950.0, 120.0, 10.0]).tolist() == [35050.0, 100.0, -10.0]


def test_positions_within_half_a_box_do_not_move():
    positions = np.array([[10.0, 20.0, 30.0], [17000.0, 0.0, 34999.0], [100.0, 17400.0, 18000.0]])
    moved = tng.nearest_image(positions, [0.0, 0.0, 34000.0])
    assert np.array_equal(moved[0], [10.0, 20.0, 35030.0])   # only z is more than half a box away
    assert np.array_equal(moved[1], [17000.0, 0.0, 34999.0])
    assert np.array_equal(moved[2], [100.0, 17400.0, 18000.0])


def test_one_reference_per_row_fixes_a_whole_track():
    primary = np.array([[34900.0, 500.0, 500.0], [34990.0, 500.0, 500.0], [80.0, 500.0, 500.0]])
    secondary = np.array([[34950.0, 500.0, 500.0], [40.0, 500.0, 500.0], [130.0, 500.0, 500.0]])
    apart = tng.nearest_image(secondary, primary) - primary
    assert apart[:, 0].tolist() == [50.0, 50.0, 50.0]


def test_the_box_size_can_be_given_and_results_are_float64():
    moved = tng.nearest_image(np.array([9.0, 1.0, 5.0], dtype=np.float32), [1.0, 9.0, 5.0], box=10.0)
    assert moved.tolist() == [-1.0, 11.0, 5.0] and moved.dtype == np.float64
    assert tng.BOX == 35000.0


def test_a_box_of_zero_is_an_error():
    with pytest.raises(ValueError, match="box must be above 0"):
        tng.nearest_image([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], box=0.0)


# --- Units -----------------------------------------------------------------------------------------------


def test_known_tng_values_in_physical_units():
    # TNG50-1's dark matter particle is 4.5e5 Msun, and its box is 35 Mpc/h = 51.7 Mpc across today.
    assert tng.physical(3.0737e-5, "mass") == pytest.approx(4.5375e5, rel=1e-4)
    assert tng.physical(35000.0, "length", 1.0) == pytest.approx(51668.0, rel=1e-5)


@pytest.mark.parametrize("quantity, stored, expected", [
    ("length", 6.774, 2.5),                # x a / h
    ("velocity", 100.0, 100.0),            # the catalog's km/s, unchanged
    ("particle_velocity", 100.0, 50.0),    # x sqrt(a)
    ("mass", 6.774, 1.0e11),               # x 1e10 / h
    ("spin", 677.4, 1000.0),               # / h, and no a
])
def test_each_quantitys_factor_at_a_quarter(quantity, stored, expected):
    assert tng.physical(stored, quantity, 0.25) == pytest.approx(expected)


def test_arrays_keep_their_shape_and_come_back_float64():
    positions = np.arange(12, dtype=np.float32).reshape(4, 3)
    converted = tng.physical(positions, "length", 0.5)
    assert converted.shape == (4, 3) and converted.dtype == np.float64
    assert converted == pytest.approx(positions * 0.5 / 0.6774)


def test_one_scale_factor_per_row_converts_a_table_column():
    positions = np.ones((3, 3))
    converted = tng.physical(positions, "length", [0.25, 0.5, 1.0])
    assert converted[:, 0] == pytest.approx(np.array([0.25, 0.5, 1.0]) / 0.6774)
    assert tng.physical([4.0, 4.0, 4.0], "particle_velocity", [0.25, 0.25, 1.0]) == pytest.approx([2.0, 2.0, 4.0])


@pytest.mark.parametrize("values, quantity, scale_factor, message", [
    (1.0, "distance", 0.5, "Unknown quantity 'distance'"),
    (1.0, "length", None, "needs the snapshot's scale factor"),
    (1.0, "particle_velocity", None, "needs the snapshot's scale factor"),
    (1.0, "length", 0.0, "above 0 and at most 1"),
    (1.0, "length", 1.5, "above 0 and at most 1"),
    (np.ones((3, 3)), "length", [0.5, 0.5], "2 scale factors"),
    (np.ones((3, 3)), "length", np.full((3, 3), 0.5), "one number, or one per row"),
])
def test_physical_errors(values, quantity, scale_factor, message):
    with pytest.raises(ValueError, match=message):
        tng.physical(values, quantity, scale_factor)


def test_mass_velocity_and_spin_need_no_scale_factor():
    assert tng.physical(1.0, "mass") == tng.physical(1.0, "mass", 0.3)
    assert tng.physical(1.0, "velocity") == 1.0
    assert tng.physical(1.0, "spin") == pytest.approx(1 / 0.6774)


def test_spams_units_are_15_kpc_and_1e11_msun_with_g_equal_to_one():
    assert tng.spam_units(30.0, "length") == 2.0
    assert tng.spam_units(5.0e10, "mass") == 0.5
    # Velocity and time follow from G = 1; SPAM's own constants (parameters_module.f90) are
    # 4.6285203749999994e22 cm / 2.733342473337471e15 s = 169.3355 km/s, and that time is 86.614 Myr.
    assert tng.SPAM_VELOCITY == pytest.approx(169.3355, rel=1e-4)
    assert tng.SPAM_TIME == pytest.approx(2.733342473337471e15 / 3.15576e16, rel=1e-4)
    assert tng.G * tng.SPAM_MASS / (tng.SPAM_LENGTH * tng.SPAM_VELOCITY ** 2) == pytest.approx(1.0)
    assert tng.spam_units([169.3355, 338.671], "velocity") == pytest.approx([1.0, 2.0], rel=1e-4)
    assert tng.spam_units(0.866, "time") == pytest.approx(10.0, rel=1e-3)


def test_an_unknown_spam_quantity_is_an_error():
    with pytest.raises(ValueError, match="Unknown quantity 'spin'"):
        tng.spam_units(1.0, "spin")


# --- The real files (only when GTC_DATA_DIR is set) ------------------------------------------------------

DATA = Path(os.environ["GTC_DATA_DIR"]) / "tng-data" if os.environ.get("GTC_DATA_DIR") else None
needs_data = pytest.mark.skipif(DATA is None, reason="GTC_DATA_DIR is not set")
TARGETS = (62000000190717, 75000000121160, 75000000317209)


@needs_data
def test_the_real_table():
    table = tng.read_table(DATA / "moi-4-targets.h5")
    assert len(table) == 113 and all(len(column) == 1268 for column in table.values())
    assert len(np.unique(table["moi_2"])) == 59
    assert table["p_SubhaloPos"].shape == (1268, 3)
    assert table["orbital_frame_vel_transform"].shape == (1268, 4, 4)
    for target in TARGETS:
        assert (table["p_SubhaloIDRaw"] == target).sum() == 1
    assert np.allclose(table["scale_factor"], 1 / (1 + table["redshift"]))


@needs_data
def test_real_pairs_split_by_the_box_edge_are_mended_by_nearest_image():
    table = tng.read_table(DATA / "moi-4-targets.h5")
    primary, secondary = table["p_SubhaloPos"].astype(np.float64), table["s_SubhaloPos"].astype(np.float64)
    stored = np.linalg.norm(secondary - primary, axis=1)
    mended = np.linalg.norm(tng.nearest_image(secondary, primary) - primary, axis=1)
    split = np.abs(stored - mended) > 1
    # 20 rows in 6 encounters have the pair on either side of a face: stored, they look a whole box apart.
    assert split.sum() == 20 and stored[split].min() > 34000
    assert sorted(set(table["moi_2"][split].tolist())) == [40000000066106, 55000000131486, 55000000306245,
                                                            67000000260243, 99000000229933, 99000000493433]
    # Mended, every pair in the table is within 1400 ckpc/h, and the other rows haven't moved.
    assert mended.max() < 1400
    assert np.array_equal(stored[~split], mended[~split])


@needs_data
def test_a_real_archives_particle_counts():
    archive = tng.read_archive(DATA / "target-archive-62000000190717.h5", "all")
    counts = {name: int((archive["particle_type"] == number).sum()) for name, number in tng.PARTICLE_TYPES.items()}
    assert counts == {"gas": 2263173, "dm": 3576822, "stars": 1519823, "bh": 2}
    assert (archive["p_id"], archive["t_snap"]) == (62000000190717, 62)
    stars = tng.read_archive(DATA / "target-archive-62000000190717.h5")
    assert len(stars["masses"]) == 1519823
    assert np.array_equal(stars["positions"], archive["positions"][archive["particle_type"] == 4])


@needs_data
@pytest.mark.parametrize("target", TARGETS)
def test_real_archives_hold_raw_tng_masses(target):
    """Every dark matter particle has TNG50-1's particle mass in 1e10 Msun/h (4.54e5 Msun for h = 0.6774)."""
    masses = tng.read_archive(DATA / f"target-archive-{target}.h5", "dm")["masses"]
    assert len(masses) > 0 and np.all(masses == masses[0])
    assert masses[0] * 1e10 / 0.6774 == pytest.approx(4.54e5, rel=2e-3)


# --- The real particles against the TNG catalog (only when GTC_DATA_DIR is set) --------------------------
#
# Each target's particles, split by SUBFIND's galaxy label, must give back what the TNG catalog says
# about the same two galaxies. That pins down the archive's units and what `SubhaloSpin` means. The
# tolerances are about 5 times the largest difference seen; the archives are float32.


def in_frame(matrix, vector):
    """A box-frame position or velocity moved into the archive's frame by a 4x4 matrix of the table."""
    return matrix[:3, :3] @ vector + matrix[:3, 3]


def angle_between(a, b):
    """The angle between two vectors, in degrees."""
    return np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))


@pytest.fixture(scope="module", params=TARGETS)
def measured(request):
    """One target: its table row and both galaxies measured from the archive's particles."""
    if DATA is None:
        pytest.skip("GTC_DATA_DIR is not set")
    table = tng.read_table(DATA / "moi-4-targets.h5")
    row = int(np.flatnonzero(table["p_SubhaloIDRaw"] == request.param)[0])
    row = {name: column[row] for name, column in table.items()}
    archive = tng.read_archive(DATA / f"target-archive-{request.param}.h5", "all")
    galaxies = {}
    for galaxy, side in ((0, "p"), (1, "s")):
        kept = archive["subfind_galaxy"] == galaxy
        positions, masses = archive["positions"][kept], archive["masses"][kept]
        galaxies[side] = tng.galaxy_properties(positions, archive["velocities"][kept], masses)
        a = row["scale_factor"]
        galaxies[side]["physical"] = tng.galaxy_properties(
            tng.physical(positions, "length", a), tng.physical(archive["velocities"][kept], "particle_velocity", a),
            tng.physical(masses, "mass"))
        most_bound = in_frame(row["orbital_frame_pos_transform"], row[f"{side}_SubhaloPos"])
        galaxies[side]["half_mass_radius"] = tng.half_mass_radius(positions, masses, most_bound)
    return row, galaxies


@pytest.mark.parametrize("side", ["p", "s"])
def test_particle_masses_add_up_to_the_catalogs_mass(measured, side):
    row, galaxies = measured
    assert galaxies[side]["mass"] == pytest.approx(row[f"{side}_SubhaloMass"], rel=5e-7)


def test_positions_are_comoving_the_separation_matches_the_catalog_as_stored(measured):
    row, galaxies = measured
    separation = np.linalg.norm(galaxies["p"]["center_of_mass"] - galaxies["s"]["center_of_mass"])
    assert separation == pytest.approx(np.linalg.norm(row["p_SubhaloCM"] - row["s_SubhaloCM"]), rel=2e-4)


def test_velocities_need_sqrt_a_to_match_the_catalogs_relative_velocity(measured):
    row, galaxies = measured
    relative = np.linalg.norm(galaxies["p"]["velocity"] - galaxies["s"]["velocity"])
    catalog = np.linalg.norm(row["p_SubhaloVel"] - row["s_SubhaloVel"])
    assert relative * np.sqrt(row["scale_factor"]) == pytest.approx(catalog, rel=2e-6)
    assert relative > 1.1 * catalog  # without sqrt(a) it is 16 to 27% too fast


@pytest.mark.parametrize("side", ["p", "s"])
def test_the_catalogs_spin_is_per_unit_mass_with_a_physical_length(measured, side):
    """`SubhaloSpin` = sum(m (a r) x (sqrt(a) v)) / sum(m): the particles' spin times a**1.5."""
    row, galaxies = measured
    spin, catalog = galaxies[side]["spin"], row[f"{side}_SubhaloSpin"]
    size = np.linalg.norm(spin) * row["scale_factor"] ** 1.5
    assert size == pytest.approx(np.linalg.norm(catalog), rel=5e-5)
    # The other readings are far off: comoving length (16 to 27%), not per unit mass (a factor of the mass).
    assert np.linalg.norm(spin) * np.sqrt(row["scale_factor"]) > 1.1 * np.linalg.norm(catalog)
    assert abs(size * galaxies[side]["mass"] / np.linalg.norm(catalog) - 1) > 1
    # A spin only turns with the frame: the rotation part of the matrix, no shift.
    assert angle_between(spin, row["orbital_frame_vel_transform"][:3, :3] @ catalog) < 1e-3


@pytest.mark.parametrize("side", ["p", "s"])
def test_the_archive_is_the_box_moved_by_the_tables_matrices(measured, side):
    row, galaxies = measured
    center = in_frame(row["orbital_frame_pos_transform"], row[f"{side}_SubhaloCM"])
    assert np.linalg.norm(galaxies[side]["center_of_mass"] - center) < 5e-3  # ckpc/h
    # The velocity matrix works on particle velocities (km/s / sqrt(a)), so the catalog's km/s is divided first.
    velocity = in_frame(row["orbital_frame_vel_transform"], row[f"{side}_SubhaloVel"] / np.sqrt(row["scale_factor"]))
    assert np.linalg.norm(galaxies[side]["velocity"] - velocity) < 2e-4


@pytest.mark.parametrize("side", ["p", "s"])
def test_the_half_mass_radius_around_the_most_bound_particle_matches_the_catalog(measured, side):
    row, galaxies = measured
    assert galaxies[side]["half_mass_radius"] == pytest.approx(row[f"{side}_SubhaloHalfmassRad"], rel=3e-4)


@pytest.mark.parametrize("side", ["p", "s"])
def test_particles_converted_first_give_the_catalogs_values_in_physical_units(measured, side):
    """Converting the particles and then measuring agrees with converting the catalog's own values."""
    row, galaxies = measured
    galaxy = galaxies[side]["physical"]
    assert galaxy["mass"] == pytest.approx(tng.physical(row[f"{side}_SubhaloMass"], "mass"), rel=5e-7)
    catalog_spin = tng.physical(row[f"{side}_SubhaloSpin"], "spin")
    assert np.linalg.norm(galaxy["spin"]) == pytest.approx(np.linalg.norm(catalog_spin), rel=5e-5)


def test_the_pairs_relative_velocity_in_km_per_s_matches_the_catalog(measured):
    row, galaxies = measured
    relative = np.linalg.norm(galaxies["p"]["physical"]["velocity"] - galaxies["s"]["physical"]["velocity"])
    catalog = tng.physical(row["p_SubhaloVel"] - row["s_SubhaloVel"], "velocity")
    assert relative == pytest.approx(np.linalg.norm(catalog), rel=2e-6)
