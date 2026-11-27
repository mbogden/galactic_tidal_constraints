"""Tests for tng.py: reading the target table and a target archive, measuring a galaxy, and units.

The first sections use small made-up files and particles and always run. The last section reads the real files and
runs only when the environment variable GTC_DATA_DIR names the workspace's data folder (the one
holding `tng-legacy-data/`).
"""

import json
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


# --- Out of the old frame --------------------------------------------------------------------------------


def old_frame(rotation, shift):
    """A 4x4 matrix as the target table stores them: x_old = rotation @ x_box + shift."""
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, shift
    return matrix


# A rotation that mixes all three axes: 30 degrees about z, then 50 degrees about x.
_c, _s, _C, _S = np.cos(np.radians(30)), np.sin(np.radians(30)), np.cos(np.radians(50)), np.sin(np.radians(50))
ROTATION = np.array([[1, 0, 0], [0, _C, -_S], [0, _S, _C]]) @ np.array([[_c, -_s, 0], [_s, _c, 0], [0, 0, 1]])


def test_archive_to_box_undoes_the_old_frames_rotation_and_shift():
    rng = np.random.default_rng(0)
    positions, velocities = rng.uniform(0, 35000, (50, 3)), rng.normal(0, 200, (50, 3))
    pos_shift, vel_shift = np.array([-15000.0, 31000.0, -12000.0]), np.array([58.0, -317.0, 51.0])
    stored = positions @ ROTATION.T + pos_shift, velocities @ ROTATION.T + vel_shift
    back = tng.archive_to_box(*stored, old_frame(ROTATION, pos_shift), old_frame(ROTATION, vel_shift))
    assert back[0] == pytest.approx(positions, abs=1e-8) and back[1] == pytest.approx(velocities, abs=1e-10)


def test_archive_to_box_returns_float64_and_changes_no_units():
    positions = np.arange(12, dtype=np.float32).reshape(4, 3)
    back = tng.archive_to_box(positions, positions * 2, np.eye(4), np.eye(4))
    assert back[0].dtype == np.float64 and back[1].dtype == np.float64
    assert np.array_equal(back[0], positions) and np.array_equal(back[1], positions * 2)


@pytest.mark.parametrize("matrix, message", [
    (np.eye(3), "pos_matrix must be a"),
    (old_frame(ROTATION, [1.0, 2.0, 3.0])[[0, 1, 2, 2]], "pos_matrix must be a"),      # last row isn't 0, 0, 0, 1
    (old_frame(ROTATION * 2, [0.0, 0.0, 0.0]), "pos_matrix doesn't hold a rotation"),  # it also stretches
    (old_frame(np.diag([1.0, 1.0, -1.0]), [0.0, 0.0, 0.0]), "pos_matrix doesn't hold a rotation"),  # a mirror
])
def test_archive_to_box_refuses_a_matrix_that_is_not_a_rotation_and_shift(matrix, message):
    with pytest.raises(ValueError, match=message):
        tng.archive_to_box(np.zeros((4, 3)), np.zeros((4, 3)), matrix, np.eye(4))


def test_archive_to_box_names_the_bad_matrix_and_checks_the_particles():
    with pytest.raises(ValueError, match="vel_matrix must be a"):
        tng.archive_to_box(np.zeros((4, 3)), np.zeros((4, 3)), np.eye(4), np.eye(3))
    with pytest.raises(ValueError, match=r"Expected \(n, 3\) particles"):
        tng.archive_to_box(np.zeros((4, 2)), np.zeros((4, 3)), np.eye(4), np.eye(4))


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


# --- The frame SPAM uses ---------------------------------------------------------------------------------


def test_the_hubble_rate_today_and_at_half_the_universes_size():
    assert tng.hubble_rate(1.0) == pytest.approx(0.06774)  # 67.74 km/s per Mpc
    # At a = 0.5 matter is 8 times denser: H = H0 * sqrt(0.3089 * 8 + 0.6911).
    assert tng.hubble_rate(0.5) == pytest.approx(0.06774 * np.sqrt(3.1623))
    assert tng.hubble_rate([1.0, 0.5]).shape == (2,)
    with pytest.raises(ValueError, match="above 0 and at most 1"):
        tng.hubble_rate(0.0)


def test_core_velocity_averages_only_the_particles_inside_the_radius_by_mass():
    positions = np.array([[10.0, 0, 0], [10.5, 0, 0], [10.0, 0.9, 0], [12.0, 0, 0]])
    velocities = np.array([[100.0, 0, 0], [200.0, 0, 0], [0, 400.0, 0], [9999.0, 9999.0, 9999.0]])
    velocity = tng.core_velocity(positions, velocities, [1.0, 1.0, 2.0, 5.0], [10.0, 0, 0], 1.0)
    assert velocity.tolist() == [75.0, 200.0, 0.0]  # (100 + 200) / 4 and 2 * 400 / 4; the far particle is left out


def test_core_velocity_errors():
    with pytest.raises(ValueError, match="No particles within 0.5 of the center"):
        tng.core_velocity(np.ones((3, 3)), np.ones((3, 3)), np.ones(3), [5.0, 5.0, 5.0], 0.5)
    with pytest.raises(ValueError, match=r"center has shape \(2,\)"):
        tng.core_velocity(np.ones((3, 3)), np.ones((3, 3)), np.ones(3), [5.0, 5.0], 0.5)


def test_relative_to_puts_the_center_at_the_origin_and_at_rest():
    positions = np.array([[110.0, 20.0, 30.0], [100.0, 20.0, 30.0]])
    velocities = np.array([[50.0, 60.0, 70.0], [40.0, 60.0, 70.0]])
    moved, speeds = tng.relative_to(positions, velocities, [100.0, 20.0, 30.0], [40.0, 60.0, 70.0], 1.0, hubble_flow=False)
    assert moved.tolist() == [[10.0, 0, 0], [0, 0, 0]] and speeds.tolist() == [[10.0, 0, 0], [0, 0, 0]]


def test_the_hubble_flow_adds_h_times_the_distance_along_the_separation():
    without = tng.relative_to([100.0, 0, 0], [0, 80.0, 0], [0, 0, 0], [0, 0, 0], 0.5, hubble_flow=False)
    with_flow = tng.relative_to([100.0, 0, 0], [0, 80.0, 0], [0, 0, 0], [0, 0, 0], 0.5)  # on by default
    assert np.array_equal(without[0], with_flow[0])
    assert with_flow[1] - without[1] == pytest.approx([100 * tng.hubble_rate(0.5), 0, 0])
    # It points along the separation, so the orbit's plane is the same either way.
    assert np.cross(with_flow[0], with_flow[1]) == pytest.approx(np.cross(without[0], without[1]))


def test_relative_to_needs_matching_shapes():
    with pytest.raises(ValueError, match="must both be"):
        tng.relative_to(np.zeros((4, 3)), np.zeros((3, 3)), [0, 0, 0], [0, 0, 0], 1.0)


def test_orbit_rotation_puts_a_tilted_orbit_in_the_x_y_plane():
    # A circular orbit in a tilted plane: the secondary at 30 from the primary, moving at 200.
    position, velocity = ROTATION @ [0.0, 30.0, 0.0], ROTATION @ [-200.0, 0.0, 0.0]
    turn = tng.orbit_rotation(position, velocity)
    assert turn @ turn.T == pytest.approx(np.eye(3), abs=1e-12) and np.linalg.det(turn) == pytest.approx(1.0)
    assert position @ turn.T == pytest.approx([30.0, 0, 0], abs=1e-12)     # the secondary on +x
    assert velocity @ turn.T == pytest.approx([0, 200.0, 0], abs=1e-12)    # counter-clockwise, no z part
    assert np.cross(position, velocity) @ turn.T == pytest.approx([0, 0, 6000.0], abs=1e-9)  # momentum on +z


def test_orbit_rotation_keeps_lengths_and_turns_any_vector_the_same_way():
    turn = tng.orbit_rotation([3.0, 4.0, 0.0], [0.0, 1.0, 2.0])
    vectors = np.random.default_rng(1).normal(size=(20, 3))
    assert np.linalg.norm(vectors @ turn.T, axis=1) == pytest.approx(np.linalg.norm(vectors, axis=1))


@pytest.mark.parametrize("position, velocity", [
    ([10.0, 0, 0], [-50.0, 0, 0]),     # falling straight in
    ([10.0, 0, 0], [0, 0, 0]),         # not moving
    ([0, 0, 0], [0, 50.0, 0]),         # on top of each other
])
def test_a_head_on_pair_has_no_orbit_plane(position, velocity):
    with pytest.raises(ValueError, match="No orbit plane"):
        tng.orbit_rotation(position, velocity)


# --- Disk angles -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("spin, phi, theta", [
    ([0, 0, 5.0], 0.0, 0.0),            # along +z: flat, counter-clockwise
    ([0, 0, -5.0], 0.0, 180.0),         # along -z: flat, clockwise
    ([2.0, 0, 0], 0.0, 90.0),           # edge-on, spin along +x
    ([0, 3.0, 0], 90.0, 90.0),
    ([-1.0, 0, 0], 180.0, 90.0),
    ([0, -1.0, 0], 270.0, 90.0),        # phi runs 0 up to 360
    ([1.0, 1.0, np.sqrt(2)], 45.0, 45.0),
    ([1.0, -1.0, -np.sqrt(2)], 315.0, 135.0),
])
def test_disk_angles_of_known_spin_directions(spin, phi, theta):
    assert tng.disk_angles(spin) == pytest.approx((phi, theta), abs=1e-9)


def test_disk_angles_depend_only_on_the_spins_direction():
    assert tng.disk_angles([3.0, 4.0, 12.0]) == pytest.approx(tng.disk_angles([0.3, 0.4, 1.2]))
    assert all(type(angle) is float for angle in tng.disk_angles([3.0, 4.0, 12.0]))


def test_disk_angles_errors():
    with pytest.raises(ValueError, match="no direction"):
        tng.disk_angles([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match=r"spin has shape \(2,\)"):
        tng.disk_angles([1.0, 0.0])


@pytest.mark.parametrize("axis_1, axis_2", [
    ([0.0, 0.0, 1.0], [0.0, 0.0, -1.0]),          # flat, and flat the other way round
    ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),           # both edge-on
    ([0.3, -0.5, 0.8], [-0.7, 0.2, -0.4]),        # tilted, one in each half
    ([-0.6, -0.6, 0.1], [0.2, 0.9, 0.5]),
])
def test_spam_builds_a_disk_whose_spin_points_where_disk_angles_was_told(axis_1, axis_2):
    """Through SPAM itself: angles from a spin axis give a disk that spins about that axis."""
    sim = pytest.importorskip("galactic_tidal_constraints.simr.simulation")
    model = np.zeros(14)
    model[[sim.PARAM_INDEX[name] for name in ("x", "vy", "mass1", "mass2", "rout1", "rout2")]] = [8.0, 0.5, 1.0, 1.0, 1.0, 1.0]
    for number, axis in ((1, axis_1), (2, axis_2)):
        model[sim.PARAM_INDEX[f"phi{number}"]], model[sim.PARAM_INDEX[f"theta{number}"]] = tng.disk_angles(axis)
    particles = sim.unperturbed_particles(model, sim.SimulatorConfig(n1=500, n2=500))
    for disk, axis in ((particles[:500], axis_1), (particles[500:1000], axis_2)):
        spin = tng.galaxy_properties(disk[:, :3], disk[:, 3:], np.ones(500))["spin"]
        assert angle_between(spin, np.array(axis)) < 0.5  # degrees


# --- Which galaxy is the primary -------------------------------------------------------------------------


def pair(star_masses_p, star_masses_s, other_p=0.0, other_s=0.0):
    """A made-up archive: stars of each galaxy by its historical label, and one dark matter particle each."""
    masses = [*star_masses_p, *star_masses_s, other_p, other_s]
    return {"masses": np.array(masses, dtype=np.float32),
            "particle_type": np.array([4] * (len(masses) - 2) + [1, 1], dtype=np.int8),
            "historical_galaxy": np.array([0] * len(star_masses_p) + [1] * len(star_masses_s) + [0, 1], dtype=np.int8),
            "subfind_galaxy": np.zeros(len(masses), dtype=np.int8)}  # SUBFIND gives everything to the primary


def test_the_table_rule_keeps_the_tables_primary():
    assert tng.primary_side(pair([1.0], [5.0]), "table") == "p"
    assert tng.GALAXY_LABELS == {"p": 0, "s": 1}


def test_the_stars_rule_picks_the_galaxy_with_more_mass_in_its_own_stars():
    assert tng.primary_side(pair([1.0, 2.0], [5.0]), "stars") == "s"
    assert tng.primary_side(pair([4.0, 2.0], [5.0]), "stars") == "p"
    assert tng.primary_side(pair([5.0], [2.0, 3.0]), "stars") == "p"  # a tie keeps the table's primary


def test_the_stars_rule_ignores_other_particles_and_subfinds_labels():
    # The table's primary holds a huge halo and SUBFIND's label gives it everything; the stars still decide.
    assert tng.primary_side(pair([1.0], [5.0], other_p=1000.0, other_s=0.1), "stars") == "s"


def test_primary_side_errors():
    with pytest.raises(ValueError, match="Unknown rule 'mass'"):
        tng.primary_side(pair([1.0], [5.0]), "mass")
    with pytest.raises(ValueError, match="needs star particles in both galaxies"):
        tng.primary_side(pair([1.0], []), "stars")


# --- Measuring a target ----------------------------------------------------------------------------------
#
# A made-up pair with known answers, built in physical units and then stored the way TNG stores it.
# Scale factor 0.5. The secondary's nucleus is 100 kpc from the primary's along +x and moves at 150 km/s
# along +y, so with no rotation needed the answers can be read off. Each galaxy has:
#   - 24 equal stars on three rings (0.5, 2 and 4 kpc), each turning at 100 km/s. The inner two rings
#     spin about +z, the outer one about (0.6, 0.8, 0), in the orbit's plane.
#   - a black hole on the nucleus, moving a little differently from the stars
#   - two dark matter particles, 5 and 50 kpc from the nucleus

A = 0.5
SEPARATION, ORBIT_SPEED = 100.0, 150.0                                  # kpc, km/s
STAR = {"p": 1.0e8, "s": 5.0e7}                                         # Msun, each star
HOLE, NEAR_HALO, FAR_HALO = 1.0e6, {"p": 1.0e9, "s": 5.0e8}, {"p": 1.0e10, "s": 5.0e9}
HOLE_KICK = {"p": [3.0, 0.0, 0.0], "s": [0.0, 0.0, 4.0]}                # km/s, black hole relative to its galaxy
CATALOG_MASS = {"p": 2.0, "s": 1.0}                                     # stored: 1e10 Msun/h


def tilted_ring(radius, speed, normal, count=8):
    """Evenly spaced particles on a circle about the origin, turning counter-clockwise about `normal`."""
    normal = np.asarray(normal, dtype=np.float64)
    u = np.cross(normal, [0.0, 1.0, 0.0]) if abs(normal[1]) < 0.5 else np.cross(normal, [1.0, 0.0, 0.0])
    u = u / np.linalg.norm(u)
    w = np.cross(normal, u)
    angle = 2 * np.pi * np.arange(count) / count
    cos, sin = np.cos(angle)[:, None], np.sin(angle)[:, None]
    return radius * (cos * u + sin * w), speed * (-sin * u + cos * w)


def made_up_target(star=STAR, primary_at=(100.0, 200.0, 300.0), old_frame=None):
    """A one-row table and its archive (every particle type), for the pair described above.

    Args:
        star: each galaxy's mass per star.
        primary_at: the primary's nucleus in the box, stored units (ckpc/h).
        old_frame: (rotation, shift) the archive's particles are stored in; the box frame when None.
    """
    nucleus = {"p": np.zeros(3), "s": np.array([SEPARATION, 0.0, 0.0])}               # kpc
    bulk = {"p": np.array([10.0, 20.0, 30.0]), "s": np.array([10.0, 20.0 + ORBIT_SPEED, 30.0])}  # km/s
    positions, velocities, masses, types, owner = [], [], [], [], []
    for side, label in tng.GALAXY_LABELS.items():
        for radius, normal in ((0.5, [0, 0, 1]), (2.0, [0, 0, 1]), (4.0, [0.6, 0.8, 0])):
            where, moving = tilted_ring(radius, 100.0, normal)
            positions += [nucleus[side] + where]; velocities += [bulk[side] + moving]
            masses += [np.full(8, star[side])]; types += [np.full(8, 4)]
        positions += [nucleus[side][None], nucleus[side] + [[0, 0, 5.0], [0, 0, 50.0]]]
        velocities += [(bulk[side] + HOLE_KICK[side])[None], np.tile(bulk[side], (2, 1))]
        masses += [[HOLE], [NEAR_HALO[side], FAR_HALO[side]]]; types += [[5], [1, 1]]
        owner += [np.full(27, label)]
    positions, velocities = np.concatenate(positions), np.concatenate(velocities)

    primary_at = np.asarray(primary_at, dtype=np.float64)
    in_box = lambda kpc: (primary_at + np.asarray(kpc) * tng.H / A) % tng.BOX      # stored, wrapped into the box
    stored_positions, stored_velocities = in_box(positions), velocities / np.sqrt(A)
    rotation, shift = (np.eye(3), np.zeros(3)) if old_frame is None else old_frame
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    pos_matrix, vel_matrix = matrix.copy(), matrix.copy()
    pos_matrix[:3, 3] = shift
    archive = {"positions": stored_positions @ rotation.T + shift, "velocities": stored_velocities @ rotation.T,
               "masses": np.concatenate(masses) * tng.H / 1e10, "particle_type": np.concatenate(types).astype(np.int8),
               "historical_galaxy": np.concatenate(owner).astype(np.int8), "subfind_galaxy": np.zeros(54, dtype=np.int8),
               "p_id": 62000000000001, "s_id": 62000000000002, "t_snap": 62}
    table = {"p_SubhaloIDRaw": np.array([62000000000001]), "s_SubhaloIDRaw": np.array([62000000000002]),
             "snap": np.array([62]), "scale_factor": np.array([A]),
             "orbital_frame_pos_transform": pos_matrix[None], "orbital_frame_vel_transform": vel_matrix[None]}
    for side in "ps":
        table[f"{side}_SubhaloPos"] = in_box(nucleus[side])[None]
        table[f"{side}_SubhaloVel"] = np.array([[0.0, 90.0 if side == "s" else 0.0, 0.0]])  # the halo's, slower
        table[f"{side}_SubhaloMass"] = np.array([CATALOG_MASS[side]])
    return table, archive


EXPANSION = float(tng.hubble_rate(A)) * SEPARATION      # km/s: what the Hubble flow adds along the separation


def test_measure_target_gives_the_pairs_orbit_in_spams_frame():
    target = tng.measure_target(*made_up_target()[:1], 0, made_up_target()[1])
    assert (target["target"], target["snap"], target["scale_factor"]) == (62000000000001, 62, A)
    assert target["primary_side"] == "p"
    assert target["separation"] == pytest.approx(SEPARATION)
    assert target["position"] == pytest.approx([SEPARATION, 0, 0], abs=1e-9)
    assert target["velocity"] == pytest.approx([EXPANSION, ORBIT_SPEED, 0], abs=1e-9)
    assert target["rotation"] == pytest.approx(np.eye(3), abs=1e-12)


def test_measure_target_gives_three_measures_of_the_velocity():
    table, archive = made_up_target()
    velocities = tng.measure_target(table, 0, archive)["velocities"]
    assert velocities["stars"] == pytest.approx([EXPANSION, ORBIT_SPEED, 0], abs=1e-9)             # the one used
    assert velocities["black_hole"] == pytest.approx([EXPANSION - 3, ORBIT_SPEED, 4], abs=1e-9)    # its own kicks
    assert velocities["catalog"] == pytest.approx([EXPANSION, 90, 0], abs=1e-9)                    # the halo's


def test_measure_target_without_the_hubble_flow():
    table, archive = made_up_target()
    target = tng.measure_target(table, 0, archive, tng.TngTargetConfig(hubble_flow=False))
    assert target["velocity"] == pytest.approx([0, ORBIT_SPEED, 0], abs=1e-9)


@pytest.mark.parametrize("role, side", [("primary", "p"), ("secondary", "s")])
def test_measure_target_gives_each_galaxys_masses_and_sizes(role, side):
    table, archive = made_up_target()
    galaxy = tng.measure_target(table, 0, archive)[role]
    assert (galaxy["side"], galaxy["subhalo_id"]) == (side, int(table[f"{side}_SubhaloIDRaw"][0]))
    assert galaxy["stellar_mass"] == pytest.approx(24 * STAR[side])
    assert galaxy["half_mass_radius"] == pytest.approx(2.0)        # the middle ring holds the 12th of 24 stars
    assert galaxy["disk_radius"] == pytest.approx(10.0)            # 5 x 2
    everything = 24 * STAR[side] + HOLE + NEAR_HALO[side] + FAR_HALO[side]
    assert galaxy["masses"] == pytest.approx({"historical": everything,
                                              "catalog": CATALOG_MASS[side] * 1e10 / tng.H,
                                              "within_radius": everything - FAR_HALO[side]})   # 50 kpc is outside 10
    assert galaxy["mass"] == galaxy["masses"]["historical"]
    assert galaxy["black_hole_offset"] == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("mass", tng.MASS_CHOICES)
def test_measure_target_hands_spam_the_mass_the_config_names(mass):
    table, archive = made_up_target()
    target = tng.measure_target(table, 0, archive, tng.TngTargetConfig(mass=mass))
    for role in ("primary", "secondary"):
        assert target[role]["mass"] == target[role]["masses"][mass]


def test_measure_target_gives_each_disks_spin_and_angles():
    table, archive = made_up_target()
    for galaxy in (tng.measure_target(table, 0, archive)[role] for role in ("primary", "secondary")):
        # Per unit mass: rings of radius 0.5 and 2 about +z, 4 about (0.6, 0.8, 0), at 100 km/s, a third of the stars each.
        assert galaxy["spin"] == pytest.approx([80, 320 / 3, 250 / 3], abs=1e-9)
        assert galaxy["spin_stars"] == 24                      # within 5 x 2 kpc: every ring
        angles = (np.degrees(np.arctan2(0.8, 0.6)), np.degrees(np.arctan2(400, 250)))     # 53.13, 57.99
        assert (galaxy["phi"], galaxy["theta"]) == pytest.approx(angles, abs=1e-9)


def test_measure_target_can_take_the_spin_from_the_inner_stars_only():
    table, archive = made_up_target()
    target = tng.measure_target(table, 0, archive, tng.TngTargetConfig(spin_radius_factor=1.5))   # within 3 kpc
    for galaxy in (target["primary"], target["secondary"]):
        assert galaxy["spin"] == pytest.approx([0, 0, 125], abs=1e-9)      # the two inner rings: flat in the orbit's plane
        assert galaxy["spin_stars"] == 16 and galaxy["theta"] == pytest.approx(0, abs=1e-6)
    with pytest.raises(ValueError, match="No stars of galaxy 'p' within 0.1 half-mass radii"):
        tng.measure_target(table, 0, archive, tng.TngTargetConfig(spin_radius_factor=0.1))


def test_measure_target_puts_the_galaxy_the_rule_names_at_the_origin():
    table, archive = made_up_target(star={"p": 1.0e8, "s": 2.0e8})         # the table's secondary has more stars
    by_table = tng.measure_target(table, 0, archive, tng.TngTargetConfig(primary_rule="table"))
    by_stars = tng.measure_target(table, 0, archive)                       # "stars" is the default
    assert (by_table["primary_side"], by_table["primary"]["side"]) == ("p", "p")
    assert (by_stars["primary_side"], by_stars["primary"]["side"], by_stars["secondary"]["side"]) == ("s", "s", "p")
    # Seen from the other galaxy the orbit is the same, and the frame is turned half way round about z.
    assert by_stars["position"] == pytest.approx([SEPARATION, 0, 0], abs=1e-9)
    assert by_stars["velocity"] == pytest.approx([EXPANSION, ORBIT_SPEED, 0], abs=1e-9)
    assert by_stars["rotation"] == pytest.approx(np.diag([-1.0, -1.0, 1.0]), abs=1e-12)
    assert by_stars["primary"]["spin"] == pytest.approx([-80, -320 / 3, 250 / 3], abs=1e-9)
    assert by_stars["primary"]["phi"] == pytest.approx(180 + np.degrees(np.arctan2(0.8, 0.6)))
    assert by_stars["primary"]["stellar_mass"] == pytest.approx(24 * 2.0e8)


def same(a, b):
    """Two results of `measure_target` agree, entry by entry."""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[key], b[key]) for key in a)
    if isinstance(a, (str, int)) or a is None:
        return a == b
    return np.allclose(a, b, rtol=1e-7, atol=1e-7)


def test_measure_target_does_not_depend_on_the_old_frame_or_the_box_edge():
    """The same pair stored in a turned and shifted frame, and sitting across a face of the box."""
    plain = tng.measure_target(*made_up_target()[:1], 0, made_up_target()[1])
    turn = tng.orbit_rotation([1.0, 2.0, 3.0], [-2.0, 0.5, 1.0])           # any rotation
    table, archive = made_up_target(old_frame=(turn, np.array([40.0, -700.0, 3.0])))
    assert same(tng.measure_target(table, 0, archive), plain)
    table, archive = made_up_target(primary_at=(34990.0, 200.0, 300.0), old_frame=(turn, np.zeros(3)))
    assert np.ptp(table["s_SubhaloPos"] - table["p_SubhaloPos"]) > 34000    # stored, the pair looks a box apart
    assert same(tng.measure_target(table, 0, archive), plain)


def test_measure_target_errors():
    table, archive = made_up_target()
    with pytest.raises(ValueError, match="the archive holds target 75000000121160"):
        tng.measure_target(table, 0, {**archive, "p_id": 75000000121160})
    stars = archive["particle_type"] == 4
    only_stars = {name: value[stars] if isinstance(value, np.ndarray) else value for name, value in archive.items()}
    with pytest.raises(ValueError, match="holds only stars"):
        tng.measure_target(table, 0, only_stars)
    no_holes = {name: value[archive["particle_type"] != 5] if isinstance(value, np.ndarray) else value
                for name, value in archive.items()}
    target = tng.measure_target(table, 0, no_holes)                        # no black holes: that measure is left out
    assert target["velocities"]["black_hole"] is None and target["primary"]["black_hole_offset"] is None


@pytest.mark.parametrize("setting, message", [
    ({"primary_rule": "mass"}, "Unknown primary_rule 'mass'"),
    ({"mass": "peak"}, "Unknown mass 'peak'"),
    ({"core_radius": 0.0}, "core_radius must be above 0"),
    ({"disk_radius_factor": -5.0}, "disk_radius_factor must be above 0"),
    ({"spin_radius_factor": 0.0}, "spin_radius_factor must be None or above 0"),
])
def test_a_target_config_refuses_values_it_does_not_know(setting, message):
    with pytest.raises(ValueError, match=message):
        tng.TngTargetConfig(**setting)


def test_a_target_configs_defaults():
    config = tng.TngTargetConfig()
    assert (config.primary_rule, config.core_radius, config.hubble_flow) == ("stars", 1.0, True)
    assert (config.mass, config.disk_radius_factor, config.spin_radius_factor) == ("historical", 5.0, 5.0)


# --- A target's stars as SPAM's particles ----------------------------------------------------------------


def made_up_stars(side):
    """One galaxy's 24 stars about its own nucleus, as `made_up_target` places them: kpc and km/s."""
    rings = [tilted_ring(radius, 100.0, normal) for radius, normal in ((0.5, [0, 0, 1]), (2.0, [0, 0, 1]), (4.0, [0.6, 0.8, 0]))]
    return np.concatenate([where for where, _ in rings]), np.concatenate([moving for _, moving in rings])


def test_target_particles_gives_the_stars_in_spams_frame_units_and_order():
    table, archive = made_up_target()
    target = tng.measure_target(table, 0, archive)
    result = tng.target_particles(table, 0, archive, target)
    particles = result["particles"]
    assert (result["n1"], result["n2"]) == (24, 24) and particles.shape == (49, 6) and particles.dtype == np.float64
    rate = float(tng.hubble_rate(A))
    for first, side, center, orbit in ((0, "p", [0, 0, 0], [0, 0, 0]), (24, "s", [SEPARATION, 0, 0], [0, ORBIT_SPEED, 0])):
        where, moving = made_up_stars(side)
        where = where + center
        assert particles[first:first + 24, :3] == pytest.approx(where / tng.SPAM_LENGTH, abs=1e-9)
        # relative to the primary's nucleus, plus the Hubble flow: H x the distance from that nucleus
        assert particles[first:first + 24, 3:] == pytest.approx((moving + orbit + rate * where) / tng.SPAM_VELOCITY, abs=1e-9)
        assert result["masses"][first:first + 24] == pytest.approx(STAR[side] / tng.SPAM_MASS)
    assert particles[-1] == pytest.approx(tng.spam_model(target)[:6], abs=1e-12)        # the secondary's center
    assert result["masses"].shape == (48,)


def test_target_particles_needs_only_the_stars_and_follows_the_primary_rule():
    table, archive = made_up_target(star={"p": 1.0e8, "s": 2.0e8})         # the table's secondary has more stars
    target = tng.measure_target(table, 0, archive)
    stars = archive["particle_type"] == 4
    only_stars = {name: value[stars] if isinstance(value, np.ndarray) else value for name, value in archive.items()}
    result = tng.target_particles(table, 0, only_stars, target)
    assert same(result, tng.target_particles(table, 0, archive, target))
    assert result["masses"][:24] == pytest.approx(2.0e8 / tng.SPAM_MASS)   # the heavier galaxy's stars come first
    assert result["particles"][:24, :3].mean(axis=0) == pytest.approx([0, 0, 0], abs=1e-9)
    assert result["particles"][24:48, :3].mean(axis=0) == pytest.approx([SEPARATION / tng.SPAM_LENGTH, 0, 0], abs=1e-9)


def test_target_particles_does_not_depend_on_the_old_frame_or_the_box_edge():
    table, archive = made_up_target()
    plain = tng.target_particles(table, 0, archive, tng.measure_target(table, 0, archive))
    turn = tng.orbit_rotation([1.0, 2.0, 3.0], [-2.0, 0.5, 1.0])
    table, archive = made_up_target(primary_at=(34990.0, 200.0, 300.0), old_frame=(turn, np.array([40.0, -700.0, 3.0])))
    assert same(tng.target_particles(table, 0, archive, tng.measure_target(table, 0, archive)), plain)


def test_target_particles_follows_the_config_the_target_was_measured_with():
    table, archive = made_up_target()
    still = tng.TngTargetConfig(hubble_flow=False)
    result = tng.target_particles(table, 0, archive, tng.measure_target(table, 0, archive, still), still)
    where, moving = made_up_stars("p")
    assert result["particles"][:24, 3:] == pytest.approx(moving / tng.SPAM_VELOCITY, abs=1e-9)       # no Hubble flow
    with pytest.raises(ValueError, match="give the config the target was measured with"):
        tng.target_particles(table, 0, archive, tng.measure_target(table, 0, archive, still))


def test_target_particles_errors():
    table, archive = made_up_target()
    target = tng.measure_target(table, 0, archive)
    with pytest.raises(ValueError, match="the archive holds target 75000000121160"):
        tng.target_particles(table, 0, {**archive, "p_id": 75000000121160}, target)
    with pytest.raises(ValueError, match="the measurements are of 5"):
        tng.target_particles(table, 0, archive, {**target, "target": 5})
    one_galaxy = {**archive, "historical_galaxy": np.zeros(54, dtype=np.int8)}
    with pytest.raises(ValueError, match="needs star particles in both galaxies"):
        tng.target_particles(table, 0, one_galaxy, target)


# --- SPAM's 14 parameters --------------------------------------------------------------------------------


def test_spam_model_gives_the_14_parameters_in_spams_units():
    table, archive = made_up_target()
    model = tng.spam_model(tng.measure_target(table, 0, archive))
    assert model.shape == (14,) and model.dtype == np.float64
    named = dict(zip(tng.SPAM_PARAMETERS, model))
    velocity_unit = np.sqrt(4.30091e-6 * 1e11 / 15)                              # km/s, about 169.33
    assert (named["x"], named["y"], named["z"]) == pytest.approx((100 / 15, 0, 0), abs=1e-12)
    assert (named["vx"], named["vy"], named["vz"]) == pytest.approx((EXPANSION / velocity_unit, 150 / velocity_unit, 0), abs=1e-12)
    everything = {side: 24 * STAR[side] + HOLE + NEAR_HALO[side] + FAR_HALO[side] for side in "ps"}
    assert (named["mass1"], named["mass2"]) == pytest.approx((everything["p"] / 1e11, everything["s"] / 1e11))
    assert (named["rout1"], named["rout2"]) == pytest.approx((10 / 15, 10 / 15))       # 5 x 2 kpc
    angles = (np.degrees(np.arctan2(0.8, 0.6)), np.degrees(np.arctan2(400, 250)))
    assert (named["phi1"], named["phi2"]) == pytest.approx((angles[0], angles[0]))
    assert (named["theta1"], named["theta2"]) == pytest.approx((angles[1], angles[1]))


def test_spam_model_follows_the_targets_config():
    table, archive = made_up_target(star={"p": 1.0e8, "s": 2.0e8})                 # the table's secondary has more stars
    config = tng.TngTargetConfig(mass="catalog", disk_radius_factor=3.0)
    named = dict(zip(tng.SPAM_PARAMETERS, tng.spam_model(tng.measure_target(table, 0, archive, config))))
    # The "stars" rule makes the table's secondary SPAM's primary, so its values come first.
    assert (named["mass1"], named["mass2"]) == pytest.approx((CATALOG_MASS["s"] / tng.H / 10, CATALOG_MASS["p"] / tng.H / 10))
    assert named["rout1"] == pytest.approx(3 * 2.0 / 15)
    assert named["phi1"] == pytest.approx(180 + np.degrees(np.arctan2(0.8, 0.6)))


def test_spam_parameters_are_in_the_simulators_order():
    sim = pytest.importorskip("galactic_tidal_constraints.simr.simulation")
    assert tng.SPAM_PARAMETERS == sim.PARAM_NAMES


# --- An encounter's rows ---------------------------------------------------------------------------------


def test_encounter_rows_are_the_rows_of_one_encounter_in_snapshot_order():
    #                     row:  0   1   2   3   4   5   6
    table = {"moi_2": np.array([7,  9,  7,  7,  9,  7,  8]),
             "snap": np.array([62, 40, 60, 63, 41, 58, 62])}        # encounter 7 has no snapshots 59 and 61
    rows = tng.encounter_rows(table, 0)
    assert rows.tolist() == [5, 2, 0, 3] and rows.dtype == np.int64
    assert table["snap"][rows].tolist() == [58, 60, 62, 63]          # gaps are kept as they are
    assert int(np.flatnonzero(rows == 0)[0]) == 2                    # the target: two rows of past, one of future
    assert tng.encounter_rows(table, 3).tolist() == [5, 2, 0, 3]     # any row of the encounter gives the same track
    assert tng.encounter_rows(table, 6).tolist() == [6]              # an encounter of one row


def test_encounter_rows_errors():
    table = {"moi_2": np.array([7, 7, 7]), "snap": np.array([60, 61, 61])}
    with pytest.raises(ValueError, match="Encounter 7 has two rows for snapshot 61"):
        tng.encounter_rows(table, 0)
    with pytest.raises(ValueError, match="row 3 is outside the table's 3 rows"):
        tng.encounter_rows(table, 3)


# --- The separation along a track ------------------------------------------------------------------------


def made_up_track():
    """Three rows of one pair, stored as TNG stores it. In the last row the pair sits across the x face of the box."""
    return {"scale_factor": np.array([0.5, 0.8, 1.0]),
            "p_SubhaloPos": np.array([[100.0, 200.0, 300.0], [100.0, 200.0, 300.0], [34990.0, 200.0, 300.0]]),
            "s_SubhaloPos": np.array([[100.0 + 2 * tng.H, 200.0, 300.0],                 # 2 ckpc x 0.5 = 1 kpc along +x
                                      [100.0, 200.0 - 10 * tng.H, 300.0 + 5 * tng.H],    # (0, -8, 4) kpc
                                      [20.0, 200.0, 300.0]])}                            # 30 ckpc/h through the face


def test_track_separation_is_secondary_minus_primary_in_physical_kpc():
    separation = tng.track_separation(made_up_track(), np.array([0, 1, 2]), "p")
    assert separation.shape == (3, 3) and separation.dtype == np.float64
    assert separation[0] == pytest.approx([1.0, 0, 0])                   # each row scaled by its own scale factor
    assert separation[1] == pytest.approx([0, -8.0, 4.0])
    assert separation[2] == pytest.approx([30 / tng.H, 0, 0])            # across the box edge: 44 kpc, not a box


def test_track_separation_with_the_other_galaxy_as_primary_points_the_other_way():
    table = made_up_track()
    assert tng.track_separation(table, np.array([0, 1, 2]), "s") == pytest.approx(-tng.track_separation(table, np.array([0, 1, 2]), "p"))


def test_track_separation_follows_the_rows_given():
    table = made_up_track()
    assert tng.track_separation(table, np.array([1]), "p")[0] == pytest.approx([0, -8.0, 4.0])
    assert tng.track_separation(table, np.array([2, 0]), "p")[1] == pytest.approx([1.0, 0, 0])


def test_track_separation_errors():
    with pytest.raises(ValueError, match="primary_side must be 'p' or 's', got 'primary'"):
        tng.track_separation(made_up_track(), np.array([0]), "primary")
    with pytest.raises(ValueError, match="1-D array of row numbers"):
        tng.track_separation(made_up_track(), np.array([0.0, 1.0]), "p")


# --- The time axis ---------------------------------------------------------------------------------------


def test_the_age_of_the_universe_today_and_early_on():
    assert tng.age_of_universe(1.0) == pytest.approx(13.80, abs=0.01)             # Gyr, TNG's cosmology
    assert tng.age_of_universe(0.5) == pytest.approx(5.87, abs=0.01)              # redshift 1
    # Early on only matter counts, and the age is 2/3 of the Hubble time 1 / H(a).
    early = 0.01
    assert tng.age_of_universe(early) * tng.hubble_rate(early) / tng.GYR_PER_KPC_PER_KMS == pytest.approx(2 / 3, rel=1e-5)
    ages = tng.age_of_universe(np.array([0.2, 0.5, 1.0]))
    assert ages.shape == (3,) and np.all(np.diff(ages) > 0)


@pytest.mark.parametrize("a", [0.1, 0.3, 0.6167, 0.7419, 1.0])
def test_the_age_is_the_time_integral_of_the_hubble_rate(a):
    """dt = da / (a H): the age's slope against the scale factor matches `hubble_rate`."""
    step = 1e-5
    slope = (tng.age_of_universe(a) - tng.age_of_universe(a - step)) / step       # Gyr per unit of a
    middle = a - step / 2
    assert slope == pytest.approx(tng.GYR_PER_KPC_PER_KMS / (middle * tng.hubble_rate(middle)), rel=1e-7)


def test_age_of_universe_errors():
    for bad in (0.0, 1.5, [0.5, -0.1]):
        with pytest.raises(ValueError, match="above 0 and at most 1"):
            tng.age_of_universe(bad)


def test_track_time_is_zero_at_the_target_and_negative_before_it():
    table = {"scale_factor": np.array([0.9, 0.5, 0.6, 0.7, 0.8])}                 # rows 1 to 3 are the track
    rows = np.array([1, 2, 3])
    time = tng.track_time(table, rows, 2)
    assert time.shape == (3,) and time.dtype == np.float64
    assert time[1] == 0 and time[0] < 0 < time[2]
    assert time == pytest.approx(tng.age_of_universe([0.5, 0.6, 0.7]) - tng.age_of_universe(0.6))
    assert time[2] - time[0] == pytest.approx(tng.track_time(table, rows, 1)[2])  # moving the target only shifts the zero
    with pytest.raises(ValueError, match="row 4 is not one of the track's rows"):
        tng.track_time(table, rows, 4)


# --- A curve through a track's rows ----------------------------------------------------------------------

UNEVEN = np.array([-3.0, -2.6, -1.5, -1.3, -0.4, 0.0, 0.9])            # times with uneven spacing, like a track with gaps


def test_a_track_spline_passes_through_every_row():
    values = np.column_stack([np.sin(UNEVEN), UNEVEN ** 2, np.cos(2 * UNEVEN)])
    curve = tng.track_spline(UNEVEN, values)
    assert curve(UNEVEN) == pytest.approx(values, abs=1e-12)
    assert curve(-1.5).shape == (3,) and curve(np.array([-2.0, -1.0])).shape == (2, 3)   # one time, or several
    assert curve.slope(-1.5).shape == (3,)


def test_a_track_spline_with_known_end_slopes_is_exact_for_a_cubic():
    cubic = lambda t: np.column_stack([2 * t ** 3 - t ** 2 + 3 * t - 1, -t ** 3 + 4 * t])
    slope = lambda t: np.column_stack([6 * t ** 2 - 2 * t + 3, -3 * t ** 2 + 4])
    curve = tng.track_spline(UNEVEN, cubic(UNEVEN), start_slope=slope(UNEVEN[:1])[0], end_slope=slope(UNEVEN[-1:])[0])
    between = np.linspace(UNEVEN[0], UNEVEN[-1], 101)
    assert curve(between) == pytest.approx(cubic(between), abs=1e-10)
    assert curve.slope(between) == pytest.approx(slope(between), abs=1e-9)


def test_a_track_spline_with_free_ends_keeps_a_straight_line_straight():
    line = np.column_stack([3 * UNEVEN + 1, -2 * UNEVEN])
    curve = tng.track_spline(UNEVEN, line)
    between = np.linspace(UNEVEN[0], UNEVEN[-1], 50)
    assert curve(between) == pytest.approx(np.column_stack([3 * between + 1, -2 * between]), abs=1e-12)
    assert curve.slope(between) == pytest.approx(np.tile([3.0, -2.0], (50, 1)), abs=1e-12)
    assert tng.track_spline(UNEVEN[:2], line[:2])(-2.8) == pytest.approx([3 * -2.8 + 1, 5.6])   # two rows: the line between them


def test_a_track_splines_slope_is_its_own_rate_of_change_with_no_kinks():
    values = np.column_stack([np.sin(UNEVEN), np.exp(UNEVEN / 2)])
    curve = tng.track_spline(UNEVEN, values)
    when, step = np.linspace(-2.9, 0.8, 40), 1e-6
    assert curve.slope(when) == pytest.approx((curve(when + step) - curve(when - step)) / (2 * step), abs=1e-6)
    for row in UNEVEN[1:-1]:                                             # the slope is the same on both sides of a row
        assert curve.slope(row - 1e-9) == pytest.approx(curve.slope(row + 1e-9), abs=1e-6)
    close = tng.track_spline(np.linspace(-3, 1, 81), np.sin(np.linspace(-3, 1, 81))[:, None])
    assert close(when)[:, 0] == pytest.approx(np.sin(when), abs=1e-5)    # with close rows it follows a smooth curve


def test_track_spline_errors():
    values = np.zeros((7, 3))
    with pytest.raises(ValueError, match="at least two increasing values"):
        tng.track_spline(UNEVEN[::-1], values)
    with pytest.raises(ValueError, match=r"values has shape \(7,\), expected \(7, k\)"):
        tng.track_spline(UNEVEN, np.zeros(7))
    with pytest.raises(ValueError, match=r"A slope has shape \(2,\), expected \(3,\)"):
        tng.track_spline(UNEVEN, values, end_slope=[1.0, 2.0])
    with pytest.raises(ValueError, match="not extended beyond its rows"):
        tng.track_spline(UNEVEN, values)(1.0)


# --- A target's sampled orbit ----------------------------------------------------------------------------
#
# A made-up track of five rows. The secondary moves in a straight line at a steady velocity in physical
# kpc, so the curve through the rows is that line and every sample is known. The target is row 3.

TRACK_A = np.array([0.50, 0.55, 0.60, 0.65, 0.70])                     # scale factors of the five rows
TRACK_NOW = 3
TRACK_START, TRACK_VELOCITY = np.array([30.0, -6.0, 9.0]), np.array([-40.0, 25.0, 10.0])   # kpc at now; kpc per Gyr


def made_up_track_table():
    time = tng.age_of_universe(TRACK_A) - tng.age_of_universe(TRACK_A[TRACK_NOW])           # Gyr from the target
    separation = TRACK_START + TRACK_VELOCITY * time[:, None]                                # kpc
    primary = np.tile([100.0, 200.0, 300.0], (5, 1))
    return {"moi_2": np.full(5, 7), "snap": np.arange(40, 45), "scale_factor": TRACK_A,
            "p_SubhaloIDRaw": 43000000000001 + np.arange(5) * 1000000000000 - 3000000000000,
            "p_SubhaloPos": primary, "s_SubhaloPos": primary + separation * (tng.H / TRACK_A)[:, None]}, time


def test_target_orbit_samples_the_track_in_spams_units_up_to_now():
    table, time = made_up_track_table()
    turn = tng.orbit_rotation([1.0, 2.0, 3.0], [-2.0, 0.5, 1.0])
    target = {"target": int(table["p_SubhaloIDRaw"][TRACK_NOW]), "primary_side": "p", "rotation": turn}
    orbit = tng.target_orbit(table, TRACK_NOW, target, spacing=0.5)
    first = time[0] / tng.SPAM_TIME                                                           # about -28: later than -30
    assert orbit.shape == (int(-first // 0.5) + 1, 7) and orbit.dtype == np.float64
    assert orbit[:, 6] == pytest.approx(-0.5 * np.arange(len(orbit) - 1, -1, -1))             # ..., -1, -0.5, 0
    assert orbit[-1, 6] == 0 and not np.signbit(orbit[-1, 6]) and orbit[0, 6] >= first        # ends at now; starts inside the track
    assert np.all(orbit[:, 3:6] == 0)                                                         # velocities are not measured
    known = (TRACK_START + TRACK_VELOCITY * (orbit[:, 6] * tng.SPAM_TIME)[:, None]) @ turn.T / tng.SPAM_LENGTH
    assert orbit[:, :3] == pytest.approx(known, abs=1e-9)
    assert orbit[-1, :3] == pytest.approx(TRACK_START @ turn.T / 15)                          # the target's own row


def test_target_orbit_stops_at_the_earliest_time_asked_for():
    table, _ = made_up_track_table()
    target = {"target": int(table["p_SubhaloIDRaw"][TRACK_NOW]), "primary_side": "p", "rotation": np.eye(3)}
    orbit = tng.target_orbit(table, TRACK_NOW, target, spacing=0.25, earliest=-10.0)
    assert (len(orbit), orbit[0, 6], orbit[-1, 6]) == (41, -10.0, 0.0)
    assert len(tng.target_orbit(table, TRACK_NOW, target)) == int(-tng.track_time(table, np.arange(5), TRACK_NOW)[0] / tng.SPAM_TIME / 0.01) + 1
    other = {**target, "primary_side": "s"}
    assert tng.target_orbit(table, TRACK_NOW, other, spacing=0.25, earliest=-10.0)[:, :3] == pytest.approx(-orbit[:, :3])


def test_target_orbit_of_a_target_with_no_earlier_row_is_one_sample():
    table, _ = made_up_track_table()
    target = {"target": int(table["p_SubhaloIDRaw"][0]), "primary_side": "p", "rotation": np.eye(3)}
    orbit = tng.target_orbit(table, 0, target)
    assert orbit.shape == (1, 7) and orbit[0, 6] == 0
    assert orbit[0, :3] == pytest.approx(tng.track_separation(table, np.array([0]), "p")[0] / 15)


def test_target_orbit_errors():
    table, _ = made_up_track_table()
    target = {"target": int(table["p_SubhaloIDRaw"][TRACK_NOW]), "primary_side": "p", "rotation": np.eye(3)}
    with pytest.raises(ValueError, match="but the measurements are of"):
        tng.target_orbit(table, 2, target)
    with pytest.raises(ValueError, match="spacing must be above 0"):
        tng.target_orbit(table, TRACK_NOW, target, spacing=0.0)
    with pytest.raises(ValueError, match="earliest must be below 0"):
        tng.target_orbit(table, TRACK_NOW, target, earliest=0.0)


# --- Notes on an orbit, and a target's files -------------------------------------------------------------
#
# A made-up track of five rows on a straight line that passes the primary 20 kpc off: x = 100 kpc/Gyr x the
# time since the pass, y = 20 (or -20). The closest row is `pass_row`; the target is `now`.


def made_up_pass(pass_row, now, y=20.0, primary_at=(10000.0, 200.0, 300.0)):
    age = tng.age_of_universe(TRACK_A)
    separation = np.stack([100.0 * (age - age[pass_row]), np.full(5, y), np.zeros(5)], axis=1)       # kpc
    primary = np.tile(np.asarray(primary_at, dtype=np.float64), (5, 1))
    table = {"moi_2": np.full(5, 7), "snap": np.arange(40, 45), "scale_factor": TRACK_A,
             "p_SubhaloIDRaw": 40000000000001 + np.arange(5) * 1000000000000,
             "p_SubhaloPos": primary, "s_SubhaloPos": (primary + separation * (tng.H / TRACK_A)[:, None]) % tng.BOX}
    target = {"target": int(table["p_SubhaloIDRaw"][now]), "primary_side": "p", "rotation": np.eye(3)}
    return table, target, (age - age[now]) / tng.SPAM_TIME


def test_orbit_notes_find_the_last_pass_and_whether_a_window_fits():
    table, target, time = made_up_pass(pass_row=2, now=3)
    orbit = tng.target_orbit(table, 3, target, earliest=-100.0)
    notes = tng.orbit_notes(table, 3, target, orbit)
    assert (notes["rows_before"], notes["rows_after"]) == (3, 1)
    assert notes["samples"] == len(orbit) and notes["start"] == orbit[0, 6] == pytest.approx(time[0], abs=0.01)
    assert notes["velocity_columns"] == "zeros"
    assert notes["last_pass"]["time"] == pytest.approx(time[2], abs=0.011)          # the pass is at row 2
    assert notes["last_pass"]["distance"] == pytest.approx(20 / 15, abs=1e-4)
    assert notes["recent_approach_window"] is True                                  # twice as far back is still inside
    assert notes["crosses_box_edge"] is False
    assert set(type(value) for value in notes.values()) <= {int, float, str, bool, dict}     # plain values, for JSON
    assert notes["track_plane_angle"] == pytest.approx(180)                         # see the test of which way round

    table, target, time = made_up_pass(pass_row=1, now=4)                           # the pass is 3 rows back of 4
    notes = tng.orbit_notes(table, 4, target, tng.target_orbit(table, 4, target, earliest=-100.0))
    assert notes["last_pass"]["time"] == pytest.approx(time[1], abs=0.011) and 2 * time[1] < time[0]
    assert notes["recent_approach_window"] is False                                 # the window would start before the orbit
    assert (notes["rows_before"], notes["rows_after"]) == (4, 0)


def test_orbit_notes_without_a_pass_or_without_a_past():
    table, target, _ = made_up_pass(pass_row=4, now=3)                              # still approaching at the target
    notes = tng.orbit_notes(table, 3, target, tng.target_orbit(table, 3, target, earliest=-100.0))
    assert notes["last_pass"] is None and notes["recent_approach_window"] is False
    table, target, _ = made_up_pass(pass_row=2, now=0)                              # no earlier row: one sample
    notes = tng.orbit_notes(table, 0, target, tng.target_orbit(table, 0, target))
    assert (notes["rows_before"], notes["rows_after"], notes["samples"], notes["start"]) == (0, 4, 1, 0.0)
    assert notes["last_pass"] is None and notes["track_plane_angle"] is None


def test_orbit_notes_say_which_way_round_and_whether_the_box_edge_is_crossed():
    # Moving along +x at y = +20 the secondary goes clockwise seen from +z: against the frame's z. At y = -20, with it.
    for y, angle in ((20.0, 180.0), (-20.0, 0.0)):
        table, target, _ = made_up_pass(pass_row=2, now=3, y=y)
        assert tng.orbit_notes(table, 3, target, tng.target_orbit(table, 3, target))["track_plane_angle"] == pytest.approx(angle)
    tilted = {**target, "rotation": np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])}     # the track's plane is now x-z
    assert tng.orbit_notes(table, 3, tilted, tng.target_orbit(table, 3, tilted))["track_plane_angle"] == pytest.approx(90)
    table, target, _ = made_up_pass(pass_row=2, now=3, primary_at=(34995.0, 200.0, 300.0))
    notes = tng.orbit_notes(table, 3, target, tng.target_orbit(table, 3, target, earliest=-100.0))
    assert notes["crosses_box_edge"] is True and notes["last_pass"]["distance"] == pytest.approx(20 / 15, abs=1e-4)


def made_up_files(tmp_path):
    table, archive = made_up_target()
    table["moi_2"] = np.array([7])
    return table, archive, tng.write_target(tmp_path, table, 0, archive), tmp_path / "62000000000001"


def test_write_target_writes_the_files_a_simr_run_needs(tmp_path):
    table, archive, record, folder = made_up_files(tmp_path)
    assert sorted(path.name for path in folder.iterdir()) == sorted(tng.TARGET_FILES.values())
    target = tng.measure_target(table, 0, archive)
    model = tng.spam_model(target)
    text = (folder / "tng_spam_model.txt").read_text()
    assert text.endswith("\n") and np.array(text.split(","), dtype=float) == pytest.approx(model, abs=0)     # exact
    orbit = np.load(folder / "tng_orbit.npy")
    assert orbit.dtype == np.float64 and np.array_equal(orbit, tng.target_orbit(table, 0, target))
    stars = tng.target_particles(table, 0, archive, target)
    particles, masses = np.load(folder / "tng_star_particles.npy"), np.load(folder / "tng_star_masses.npy")
    assert particles.dtype == masses.dtype == np.float32
    assert np.array_equal(particles, stars["particles"].astype(np.float32)) and np.array_equal(masses, stars["masses"].astype(np.float32))
    assert json.loads((folder / "simr_externals.json").read_text()) == {"tng_orbit": "tng_orbit.npy",
                                                                        "tng_star_particles": "tng_star_particles.npy"}


def test_write_target_records_what_was_measured_and_how(tmp_path):
    table, archive, record, folder = made_up_files(tmp_path)
    assert json.loads((folder / "tng_target.json").read_text()) == record               # the record is what was written
    assert list(record) == ["target", "config", "model", "stars", "orbit", "files"]
    target = tng.measure_target(table, 0, archive)
    assert record["target"]["target"] == 62000000000001 and record["target"]["primary_side"] == "p"
    assert record["target"]["rotation"] == pytest.approx(target["rotation"]) and isinstance(record["target"]["rotation"], list)
    assert record["target"]["primary"]["masses"] == pytest.approx(target["primary"]["masses"])
    assert record["config"] == {"primary_rule": "stars", "core_radius": 1.0, "hubble_flow": True, "mass": "historical",
                                "disk_radius_factor": 5.0, "spin_radius_factor": 5.0}
    assert list(record["model"]) == list(tng.SPAM_PARAMETERS)
    assert [record["model"][name] for name in tng.SPAM_PARAMETERS] == pytest.approx(tng.spam_model(target), abs=0)
    assert record["stars"] == {"n1": 24, "n2": 24}
    assert record["orbit"]["samples"] == 1 and record["orbit"]["last_pass"] is None      # this made-up target has one row
    assert record["files"] == tng.TARGET_FILES


def test_write_target_follows_its_config_and_replaces_what_was_there(tmp_path):
    table, archive, record, folder = made_up_files(tmp_path)
    again = tng.write_target(tmp_path, table, 0, archive, tng.TngTargetConfig(hubble_flow=False, mass="catalog"))
    assert again["config"]["hubble_flow"] is False and again["model"]["vx"] == pytest.approx(0, abs=1e-12)
    assert again["model"]["mass1"] == pytest.approx(CATALOG_MASS["p"] * 1e10 / tng.H / 1e11)
    assert json.loads((folder / "tng_target.json").read_text()) == again
    assert len(list(folder.iterdir())) == len(tng.TARGET_FILES)


# --- SPAM's units, checked against SPAM's own dynamics ---------------------------------------------------
#
# `spam_units` assumes SPAM works with G = 1. These run SPAM's orbit with friction off and check that it does.


def test_spams_orbit_obeys_g_equal_one():
    """Along SPAM's orbit the secondary's inward acceleration is (mass1 + mass2) x the unit galaxy's force at r."""
    sim = pytest.importorskip("galactic_tidal_constraints.simr.simulation")
    free = sim.SimulatorConfig(lnl=0.0)
    model = np.array([3.0, 0, 0, 0.2, 0.6, 0, 2.0, 1.0, 1.0, 1.0, 0, 0, 0, 0])
    orbit = sim.search_orbit(model, free)[5:-5]
    profile = sim.mass_profile(free)
    distance = np.linalg.norm(orbit[:, :3], axis=1)
    assert distance.max() < profile["radius"][-1] and distance.min() > 0.1          # inside the force table throughout
    change = np.gradient(orbit[:, 3:6], orbit[:, 6], axis=0)
    inward = -(change * orbit[:, :3]).sum(axis=1) / distance
    expected = (model[6] + model[7]) * np.interp(distance, profile["radius"], profile["force"])
    assert inward[2:-2] == pytest.approx(expected[2:-2], rel=0.01)
    assert np.median(inward / expected) == pytest.approx(1.0, abs=0.002)


def test_spams_circular_orbit_keeps_keplers_time():
    """Two unit-mass galaxies 5 lengths apart on a circle: outside both galaxies' mass, so they act as points."""
    sim = pytest.importorskip("galactic_tidal_constraints.simr.simulation")
    model = np.array([5.0, 0, 0, 0, np.sqrt(2.0 / 5.0), 0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0])
    orbit = sim.search_orbit(model, sim.SimulatorConfig(lnl=0.0))
    distance = np.linalg.norm(orbit[:, :3], axis=1)
    swept = abs(np.ptp(np.unwrap(np.arctan2(orbit[:, 1], orbit[:, 0]))))
    assert distance == pytest.approx(5.0, rel=0.001)
    assert swept == pytest.approx(30.0 * np.sqrt(2.0 / 5.0 ** 3), rel=0.002)        # angular speed sqrt(G M / r^3), for 30 time units


def test_spams_unit_galaxy_is_the_same_size_whatever_its_mass():
    """Why a galaxy's mass for SPAM matters: SPAM puts all of it within 3 lengths (45 kpc), 85% within 1 (15 kpc)."""
    sim = pytest.importorskip("galactic_tidal_constraints.simr.simulation")
    profile = sim.mass_profile()
    inside = lambda r: float(np.interp(r, profile["radius"], profile["total"]))
    assert inside(1.0) == pytest.approx(0.853, abs=0.005) and inside(3.0) == pytest.approx(1.0, abs=0.001)
    assert profile["radius"][-1] == pytest.approx(10.0, abs=0.01)                   # the force table ends at 10 lengths (150 kpc)


# --- The real files (only when GTC_DATA_DIR is set) ------------------------------------------------------

DATA = Path(os.environ["GTC_DATA_DIR"]) / "tng-legacy-data" if os.environ.get("GTC_DATA_DIR") else None
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
    archive = tng.read_archive(DATA / f"target-archive-{request.param}.h5", "all")
    target = {"default": tng.measure_target(table, row, archive),
              "by_table": tng.measure_target(table, row, archive, tng.TngTargetConfig(primary_rule="table")),
              "inner_spin": tng.measure_target(table, row, archive, tng.TngTargetConfig(spin_radius_factor=2.0)),
              "all_spin": tng.measure_target(table, row, archive, tng.TngTargetConfig(spin_radius_factor=None))}
    as_particles = tng.target_particles(table, row, archive, target["default"])
    row = {name: column[row] for name, column in table.items()}
    box_positions, box_velocities = tng.archive_to_box(
        archive["positions"], archive["velocities"], row["orbital_frame_pos_transform"], row["orbital_frame_vel_transform"])
    galaxies = {}
    for galaxy, side in ((0, "p"), (1, "s")):
        kept = archive["subfind_galaxy"] == galaxy
        positions, masses = archive["positions"][kept], archive["masses"][kept]
        galaxies[side] = tng.galaxy_properties(positions, archive["velocities"][kept], masses)
        a = row["scale_factor"]
        in_box = tng.nearest_image(box_positions[kept], row[f"{side}_SubhaloPos"])
        galaxies[side]["box"] = tng.galaxy_properties(in_box, box_velocities[kept], masses)
        galaxies[side]["box"]["nearest"] = np.linalg.norm(in_box - row[f"{side}_SubhaloPos"], axis=1).min()
        galaxies[side]["physical"] = tng.galaxy_properties(
            tng.physical(positions, "length", a), tng.physical(archive["velocities"][kept], "particle_velocity", a),
            tng.physical(masses, "mass"))
        most_bound = in_frame(row["orbital_frame_pos_transform"], row[f"{side}_SubhaloPos"])
        galaxies[side]["half_mass_radius"] = tng.half_mass_radius(positions, masses, most_bound)

    # The frame SPAM uses, built from the nuclei: box frame -> nearest image -> physical -> relative -> rotated.
    a, origin = row["scale_factor"], row["p_SubhaloPos"]
    positions = tng.physical(tng.nearest_image(box_positions, origin), "length", a)
    velocities = tng.physical(box_velocities, "particle_velocity", a)
    stars = archive["particle_type"] == tng.PARTICLE_TYPES["stars"]
    frame = {"nucleus": {}, "stars": {}, "black_hole": {}}
    for galaxy, side in ((0, "p"), (1, "s")):
        frame["nucleus"][side] = tng.physical(tng.nearest_image(row[f"{side}_SubhaloPos"], origin), "length", a)
        frame["stars"][side] = tng.core_velocity(positions[stars], velocities[stars], archive["masses"][stars],
                                                 frame["nucleus"][side], 1.0)
        hole = (archive["particle_type"] == tng.PARTICLE_TYPES["bh"]) & (archive["subfind_galaxy"] == galaxy)
        frame["black_hole"][side] = velocities[hole][0]
        frame["black_hole_offset_" + side] = np.linalg.norm(positions[hole][0] - frame["nucleus"][side])
    for measure in ("stars", "black_hole"):
        frame["secondary_" + measure] = tng.relative_to(frame["nucleus"]["s"], frame[measure]["s"],
                                                        frame["nucleus"]["p"], frame[measure]["p"], a)
    frame["rotation"] = tng.orbit_rotation(*frame["secondary_stars"])
    sample = positions[::5000]
    frame["sample"] = (sample - frame["nucleus"]["p"], (sample - frame["nucleus"]["p"]) @ frame["rotation"].T)
    galaxies["frame"] = frame
    galaxies["target"] = target
    galaxies["particles"] = as_particles
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


# --- Real archives back in the box frame (only when GTC_DATA_DIR is set) ---------------------------------


@pytest.mark.parametrize("side", ["p", "s"])
def test_out_of_the_old_frame_particles_match_the_catalog_with_no_matrices(measured, side):
    """In the box frame the particles sit where the catalog says, compared directly."""
    row, galaxies = measured
    galaxy, nucleus = galaxies[side]["box"], row[f"{side}_SubhaloPos"]
    assert galaxy["nearest"] < 5e-3  # a particle sits on `SubhaloPos`: the most bound one (ckpc/h)
    assert np.linalg.norm(galaxy["center_of_mass"] - tng.nearest_image(row[f"{side}_SubhaloCM"], nucleus)) < 5e-3
    assert np.linalg.norm(galaxy["velocity"] * np.sqrt(row["scale_factor"]) - row[f"{side}_SubhaloVel"]) < 2e-4
    assert angle_between(galaxy["spin"], row[f"{side}_SubhaloSpin"]) < 0.05


def split_archive(target):
    """A target whose primary sits across a face of the box: its row, and the primary's particles in the box frame."""
    table = tng.read_table(DATA / "moi-4-targets.h5")
    row = int(np.flatnonzero(table["p_SubhaloIDRaw"] == target)[0])
    row = {name: column[row] for name, column in table.items()}
    archive = tng.read_archive(DATA / f"target-archive-{target}.h5", "all")
    primary = archive["subfind_galaxy"] == 0
    positions, velocities = tng.archive_to_box(archive["positions"][primary], archive["velocities"][primary],
                                               row["orbital_frame_pos_transform"], row["orbital_frame_vel_transform"])
    return row, positions, velocities, archive["masses"][primary]


@needs_data
@pytest.mark.parametrize("target, across, off_before", [
    (87000000451863, 10, 0.02),                                          # 10 particles across the z face
    pytest.param(49000000112167, 47690, 70.0, marks=pytest.mark.slow),   # 47,690 across the y face; 28.8 M particles
])
def test_a_galaxy_split_by_the_box_edge_is_whole_again_after_nearest_image(target, across, off_before):
    row, positions, velocities, masses = split_archive(target)
    nucleus = row["p_SubhaloPos"]
    center = tng.nearest_image(row["p_SubhaloCM"], nucleus)
    # As stored, the galaxy spans the whole box along one axis, and its center of mass is off.
    assert np.ptp(positions, axis=0).max() > 34900
    assert np.linalg.norm(tng.galaxy_properties(positions, velocities, masses)["center_of_mass"] - center) > off_before

    mended = tng.nearest_image(positions, nucleus)
    assert int((np.abs(mended - positions).max(axis=1) > 1).sum()) == across
    assert np.ptp(mended, axis=0).max() < 1500
    galaxy = tng.galaxy_properties(mended, velocities, masses)
    assert np.linalg.norm(galaxy["center_of_mass"] - center) < 5e-3
    assert np.linalg.norm(galaxy["spin"]) * row["scale_factor"] ** 1.5 == pytest.approx(np.linalg.norm(row["p_SubhaloSpin"]), rel=5e-5)
    assert tng.half_mass_radius(mended, masses, nucleus) == pytest.approx(row["p_SubhaloHalfmassRad"], rel=3e-4)


# --- Real targets in the frame SPAM uses (only when GTC_DATA_DIR is set) ---------------------------------


def test_in_the_new_frame_the_secondary_is_on_x_and_the_orbit_is_in_the_x_y_plane(measured):
    row, galaxies = measured
    frame = galaxies["frame"]
    position, velocity = (vector @ frame["rotation"].T for vector in frame["secondary_stars"])
    apart = tng.physical(np.linalg.norm(tng.nearest_image(row["s_SubhaloPos"], row["p_SubhaloPos"]) - row["p_SubhaloPos"]),
                         "length", row["scale_factor"])
    assert position == pytest.approx([apart, 0, 0], abs=1e-9)         # kpc: the distance between the nuclei
    assert abs(velocity[2]) < 1e-9 and velocity[1] > 0                # km/s: in the plane, counter-clockwise
    before, after = frame["sample"]                                   # every 5000th particle, relative to the primary
    assert np.linalg.norm(after, axis=1) == pytest.approx(np.linalg.norm(before, axis=1))


@pytest.mark.parametrize("side", ["p", "s"])
def test_each_galaxys_black_hole_sits_on_its_nucleus_and_moves_with_its_core_stars(measured, side):
    """The two measures of a nucleus's velocity agree: its black hole, and the stars within 1 kpc."""
    frame = measured[1]["frame"]
    assert frame["black_hole_offset_" + side] < 0.1                                     # kpc (seen: 0.07 at most)
    assert np.linalg.norm(frame["black_hole"][side] - frame["stars"][side]) < 15        # km/s (seen: 11 at most)


def test_the_two_velocity_measures_give_nearly_the_same_orbit(measured):
    frame = measured[1]["frame"]
    stars, hole = frame["secondary_stars"][1], frame["secondary_black_hole"][1]
    assert np.linalg.norm(stars - hole) < 15                                            # km/s (seen: 11 at most)
    normal = tng.orbit_rotation(frame["secondary_black_hole"][0], hole)[2]
    assert angle_between(frame["rotation"][2], normal) < 10                             # degrees (seen: 7.5 at most)


def test_the_nuclei_move_faster_than_the_catalogs_halo_velocities_say(measured):
    """`SubhaloVel` averages the whole halo; the nuclei's relative speed is 40 to 65% higher on these targets."""
    row, galaxies = measured
    nuclei = np.linalg.norm(galaxies["frame"]["stars"]["s"] - galaxies["frame"]["stars"]["p"])
    catalog = np.linalg.norm(row["s_SubhaloVel"] - row["p_SubhaloVel"])
    assert 1.3 < nuclei / catalog < 1.8


@needs_data
@pytest.mark.parametrize("target, by_stars", [
    (62000000190717, "p"),    # stars 3.66 and 2.73 (1e10 Msun/h): the table's primary is the heavier
    (75000000121160, "p"),    # 6.69 and 3.18
    (75000000317209, "s"),    # 2.81 and 7.07: the table's secondary is the heavier
])
def test_the_primary_of_real_targets_by_both_rules(target, by_stars):
    archive = tng.read_archive(DATA / f"target-archive-{target}.h5")  # stars
    assert tng.primary_side(archive, "table") == "p"
    assert tng.primary_side(archive, "stars") == by_stars


# --- Real targets measured (only when GTC_DATA_DIR is set) -----------------------------------------------

# For each test target: the primary by the "stars" rule, the nuclei's separation (kpc), their relative
# speed (km/s), and how far each disk's spin turns (degrees, primary then secondary) when it is taken
# from the stars within 2 half-mass radii and not from all of the galaxy's stars.
SEEN = {62000000190717: ("p", 35.43, 145.1, (6.7, 22.6)),
        75000000121160: ("p", 19.31, 106.6, (19.3, 30.0)),
        75000000317209: ("s", 110.78, 86.9, (44.5, 7.2))}


def test_a_real_target_measured_with_the_default_config(measured):
    row, galaxies = measured
    target = galaxies["target"]["default"]
    primary, separation, speed, _ = SEEN[int(row["p_SubhaloIDRaw"])]
    assert (target["target"], target["snap"]) == (row["p_SubhaloIDRaw"], row["snap"])
    assert target["primary_side"] == target["primary"]["side"] == primary
    assert target["separation"] == pytest.approx(separation, abs=0.01)
    assert target["position"] == pytest.approx([target["separation"], 0, 0], abs=1e-9)
    assert abs(target["velocity"][2]) < 1e-9 and target["velocity"][1] > 0       # in the plane, counter-clockwise
    assert np.linalg.norm(target["velocity"]) == pytest.approx(speed, abs=0.1)
    for role in ("primary", "secondary"):
        assert target[role]["black_hole_offset"] < 0.1                             # kpc: its own black hole


def test_measure_target_agrees_with_the_same_steps_done_by_hand(measured):
    """With the table's primary at the origin, it gives what the frame tests above build step by step."""
    row, galaxies = measured
    target, frame = galaxies["target"]["by_table"], galaxies["frame"]
    assert target["primary_side"] == "p"
    assert target["rotation"] == pytest.approx(frame["rotation"], abs=1e-9)
    assert target["velocities"]["stars"] == pytest.approx(frame["secondary_stars"][1] @ frame["rotation"].T, abs=1e-6)
    assert target["velocities"]["black_hole"] == pytest.approx(frame["secondary_black_hole"][1] @ frame["rotation"].T, abs=1e-6)
    by_stars = galaxies["target"]["default"]
    assert by_stars["separation"] == pytest.approx(target["separation"])           # either galaxy at the origin
    assert np.linalg.norm(by_stars["velocity"]) == pytest.approx(np.linalg.norm(target["velocity"]))


@pytest.mark.parametrize("side", ["p", "s"])
def test_a_real_galaxys_three_masses(measured, side):
    row, galaxies = measured
    target = galaxies["target"]["by_table"]
    galaxy = target["primary" if side == "p" else "secondary"]
    assert galaxy["masses"]["catalog"] == pytest.approx(tng.physical(row[f"{side}_SubhaloMass"], "mass"))
    # The earlier pipeline's own sums by historical ownership: all types ("pa_", "sa_") and stars ("ps_", "ss_").
    assert galaxy["masses"]["historical"] == pytest.approx(tng.physical(row[f"{side}a_SubhaloMass"], "mass"), rel=1e-5)
    assert galaxy["stellar_mass"] == pytest.approx(tng.physical(row[f"{side}s_SubhaloMass"], "mass"), rel=1e-5)
    assert galaxy["mass"] == galaxy["masses"]["historical"]
    assert galaxy["masses"]["within_radius"] < galaxy["masses"]["historical"]     # the same particles, fewer
    assert galaxy["disk_radius"] == pytest.approx(5 * galaxy["half_mass_radius"])


def test_a_real_disks_spin_depends_on_which_stars_are_used(measured):
    row, galaxies = measured
    everything, inner = galaxies["target"]["all_spin"], galaxies["target"]["inner_spin"]
    turned = SEEN[int(row["p_SubhaloIDRaw"])][3]
    for role, seen in zip(("primary", "secondary"), turned):
        assert inner[role]["spin_stars"] < everything[role]["spin_stars"]
        assert angle_between(inner[role]["spin"], everything[role]["spin"]) == pytest.approx(seen, abs=0.2)


def test_spam_runs_a_real_targets_model_and_ends_where_the_target_is(measured):
    """SPAM integrates back from the model and forward again: its secondary ends on the measured values."""
    sim = pytest.importorskip("galactic_tidal_constraints.simr.simulation")
    target = measured[1]["target"]["default"]
    model = tng.spam_model(target)
    assert model[[1, 2, 5]] == pytest.approx(0, abs=1e-9)                          # y, z, vz: the frame's own zeros
    assert model[0] == pytest.approx(target["separation"] / 15)
    particles = sim.simulate_particles(model, sim.SimulatorConfig(n1=50, n2=50))
    assert np.isfinite(particles).all()
    assert particles[-1] == pytest.approx(model[:6], abs=6e-3)                     # the secondary at time 0 (seen: 0.0035 off at most)


# --- The real table's tracks (only when GTC_DATA_DIR is set) ---------------------------------------------


@needs_data
def test_the_real_tracks():
    table = tng.read_table(DATA / "moi-4-targets.h5")
    row = int(np.flatnonzero(table["p_SubhaloIDRaw"] == 62000000190717)[0])
    rows = tng.encounter_rows(table, row)
    assert len(rows) == 26 and int(np.flatnonzero(rows == row)[0]) == 23          # 23 snapshots of past, 2 of future
    assert table["snap"][rows].tolist() == list(range(39, 65))
    # Four of the 61 targets are this one encounter, seen at snapshots 61 to 64.
    assert table["p_SubhaloIDRaw"][rows][-4:].tolist() == [61000000186579, 62000000190717, 63000000192216, 64000000153811]

    targets = [int(path.stem.split("-")[-1]) for path in DATA.glob("target-archive-*.h5")]
    tracks = {}
    for target in targets:
        row = int(np.flatnonzero(table["p_SubhaloIDRaw"] == target)[0])
        tracks.setdefault(int(table["moi_2"][row]), []).append(target)
    assert (len(targets), len(tracks)) == (61, 29)                                 # 61 targets on 29 encounters

    gaps = {}
    for encounter in np.unique(table["moi_2"]):
        rows = tng.encounter_rows(table, int(np.flatnonzero(table["moi_2"] == encounter)[0]))
        missing = sorted(set(range(table["snap"][rows[0]], table["snap"][rows[-1]])) - set(table["snap"][rows].tolist()))
        if missing:
            gaps[int(encounter)] = missing
    assert set(gaps) & set(tracks) == {55000000046171, 67000000260243, 99000000438148}
    assert gaps[67000000260243] == [45, 47, 53] and gaps[99000000438148] == [60]


@needs_data
def test_p_and_s_are_the_same_two_galaxies_along_every_real_track():
    """TNG's merger tree: each galaxy's descendant is the same letter's galaxy in the next snapshot's row."""
    table = tng.read_table(DATA / "moi-4-targets.h5")
    links = 0
    for encounter in np.unique(table["moi_2"]):
        rows = tng.encounter_rows(table, int(np.flatnonzero(table["moi_2"] == encounter)[0]))
        for here, after in zip(rows[:-1], rows[1:]):
            if table["snap"][after] != table["snap"][here] + 1:
                continue                                                           # a gap: the tree skips a snapshot
            links += 1
            for side in "ps":
                assert table[f"{side}_DescendantID"][here] == table[f"{side}_SubhaloID"][after]
    assert links == 1196


@needs_data
@pytest.mark.parametrize("target, first, now, closest", [
    (62000000190717, 319.8, 35.43, 8.83),      # kpc: at the track's first row, at the target, and the smallest
    (75000000121160, 381.0, 19.31, 8.33),
    (75000000317209, 438.7, 110.78, 11.01),
])
def test_the_separation_along_real_tracks(target, first, now, closest):
    table = tng.read_table(DATA / "moi-4-targets.h5")
    row = int(np.flatnonzero(table["p_SubhaloIDRaw"] == target)[0])
    rows = tng.encounter_rows(table, row)
    distance = np.linalg.norm(tng.track_separation(table, rows, "p"), axis=1)
    assert distance[0] == pytest.approx(first, abs=0.1)
    assert distance[rows == row][0] == pytest.approx(now, abs=0.01)       # the same as measure_target's "separation"
    assert distance.min() == pytest.approx(closest, abs=0.01)


@needs_data
def test_no_real_track_jumps():
    """Along every track with a target the pair starts far apart and never moves impossibly fast."""
    table = tng.read_table(DATA / "moi-4-targets.h5")
    encounters = {int(table["moi_2"][np.flatnonzero(table["p_SubhaloIDRaw"] == int(path.stem.split("-")[-1]))[0]])
                  for path in DATA.glob("target-archive-*.h5")}
    fastest, across = 0.0, set()
    for encounter in encounters:
        rows = tng.encounter_rows(table, int(np.flatnonzero(table["moi_2"] == encounter)[0]))
        separation = tng.track_separation(table, rows, "p")
        distance = np.linalg.norm(separation, axis=1)
        assert 250 < distance[0] < 620 and distance.max() < 620             # kpc: far apart at the start, never a box apart
        speed = np.linalg.norm(np.diff(separation, axis=0), axis=1) / np.diff(table["time_gyr"][rows]) * 0.9778   # km/s
        fastest = max(fastest, float(speed.max()))
        stored = np.linalg.norm(table["s_SubhaloPos"][rows].astype(np.float64) - table["p_SubhaloPos"][rows], axis=1)
        if stored.max() > 30000:
            across.add(encounter)
    assert 600 < fastest < 700                                              # seen: 676 km/s, during a close pass
    assert across == {55000000131486, 67000000260243, 99000000493433}       # stored a whole box apart in some rows


def test_a_real_track_turned_into_spams_frame(measured):
    """The track's rows by the target's one rotation: the target's own row lands on (separation, 0, 0)."""
    row, galaxies = measured
    target = galaxies["target"]["default"]
    table = tng.read_table(DATA / "moi-4-targets.h5")
    index = int(np.flatnonzero(table["p_SubhaloIDRaw"] == row["p_SubhaloIDRaw"])[0])
    rows = tng.encounter_rows(table, index)
    along_box = tng.track_separation(table, rows, target["primary_side"])
    path = along_box @ target["rotation"].T
    now = int(np.flatnonzero(rows == index)[0])
    assert path[now] == pytest.approx(target["position"], abs=1e-6)                 # where measure_target put the secondary
    assert path[now] == pytest.approx([target["separation"], 0, 0], abs=1e-6)
    assert np.linalg.norm(path, axis=1) == pytest.approx(np.linalg.norm(along_box, axis=1))   # a rotation keeps distances
    # The rest of the track is not in that plane: earlier rows have a z part (seen: 12 to 250 kpc at the first row).
    assert abs(path[0, 2]) > 10


@needs_data
def test_the_tables_times_agree_with_the_age_formula():
    table = tng.read_table(DATA / "moi-4-targets.h5")
    difference = np.abs(tng.age_of_universe(table["scale_factor"]) - table["time_gyr"])
    assert difference.max() < 1e-6                                                 # Gyr, over all 1268 rows
    row = int(np.flatnonzero(table["p_SubhaloIDRaw"] == 62000000190717)[0])
    rows = tng.encounter_rows(table, row)
    time = tng.track_time(table, rows, row)
    assert time[rows == row][0] == 0 and np.all(np.diff(time) > 0)
    assert (time[0], time[-1]) == pytest.approx((-3.58, 0.29), abs=0.01)           # Gyr: 23 snapshots back, 2 on
    assert tng.spam_units(time[0], "time") == pytest.approx(-41.3, abs=0.1)        # SPAM's own search stops at -30


@needs_data
def test_a_curve_through_real_tracks_finds_a_left_out_snapshot():
    """Leave each snapshot out in turn (a gap of two snapshots): how far the curve lands from it."""
    table = tng.read_table(DATA / "moi-4-targets.h5")
    encounters = {int(table["moi_2"][np.flatnonzero(table["p_SubhaloIDRaw"] == int(path.stem.split("-")[-1]))[0]])
                  for path in DATA.glob("target-archive-*.h5")}
    off, near_pass = [], []
    for encounter in sorted(encounters):
        rows = tng.encounter_rows(table, int(np.flatnonzero(table["moi_2"] == encounter)[0]))
        time, path = tng.track_time(table, rows, rows[0]), tng.track_separation(table, rows, "p")
        assert tng.track_spline(time, path)(time) == pytest.approx(path, abs=1e-9)
        distance = np.linalg.norm(path, axis=1)
        for left_out in range(1, len(rows) - 1):
            kept = np.arange(len(rows)) != left_out
            off.append(np.linalg.norm(tng.track_spline(time[kept], path[kept])(time[left_out]) - path[left_out]))
            around = range(max(left_out - 1, 1), min(left_out + 1, len(rows) - 2) + 1)
            near_pass.append(any(distance[i] < distance[i - 1] and distance[i] < distance[i + 1] for i in around))
    off, near_pass = np.array(off), np.array(near_pass)
    assert (len(off), int(near_pass.sum())) == (614, 127)
    assert np.median(off[~near_pass]) == pytest.approx(1.3, abs=0.1)      # kpc, away from a close pass
    assert np.median(off[near_pass]) == pytest.approx(10.4, abs=0.5)      # kpc, at or next to one


def test_a_real_targets_orbit_is_what_the_orbit_metrics_take(measured):
    sim = pytest.importorskip("galactic_tidal_constraints.simr.simulation")
    from galactic_tidal_constraints.simr import derived, metrics

    row, galaxies = measured
    target = galaxies["target"]["default"]
    table = tng.read_table(DATA / "moi-4-targets.h5")
    index = int(np.flatnonzero(table["p_SubhaloIDRaw"] == row["p_SubhaloIDRaw"])[0])
    orbit = tng.target_orbit(table, index, target)
    model = tng.spam_model(target)
    assert orbit.shape == (3001, 7) and (orbit[0, 6], orbit[-1, 6]) == (-30.0, 0.0)          # these tracks reach past -30
    assert np.all(np.diff(orbit[:, 6]) > 0) and np.all(orbit[:, 3:6] == 0)
    assert orbit[-1, :3] == pytest.approx(model[:3], abs=1e-9)                               # it ends on the model's x, y, z
    # The curve goes through the track's rows: at each snapshot's time the sample beside it is close to the row.
    rows = tng.encounter_rows(table, index)
    time = tng.spam_units(tng.track_time(table, rows, index), "time")
    path = tng.spam_units(tng.track_separation(table, rows, target["primary_side"]) @ target["rotation"].T, "length")
    inside = (time >= -30) & (time <= 0)
    nearest = np.abs(orbit[:, 6][None, :] - time[inside][:, None]).argmin(axis=1)
    assert np.abs(orbit[nearest, :3] - path[inside]).max() < 0.02                            # within half a sample's travel

    spam = sim.search_orbit(model)
    weights = derived.recent_approach_cosine_window(orbit)                                    # each has a pass in its past
    assert weights.shape == (3001,) and weights.max() == 1 and weights[-1] == 0
    for error in (metrics.orbit_position_error, metrics.orbit_time_error):
        assert np.isfinite(error(orbit, spam)) and error(orbit, spam) > 0
        assert error(orbit, spam, weights) < error(orbit, spam)                               # closer near the last pass than overall


@needs_data
def test_every_real_targets_orbit():
    """Shapes for all 61. Distances don't depend on the frame, so no particles are read."""
    from galactic_tidal_constraints.simr import derived

    table = tng.read_table(DATA / "moi-4-targets.h5")
    lengths, no_window = [], []
    for path in DATA.glob("target-archive-*.h5"):
        target = int(path.stem.split("-")[-1])
        row = int(np.flatnonzero(table["p_SubhaloIDRaw"] == target)[0])
        orbit = tng.target_orbit(table, row, {"target": target, "primary_side": "p", "rotation": np.eye(3)})
        assert orbit[-1, 6] == 0 and orbit[0, 6] >= -30 and np.isfinite(orbit).all()
        lengths.append(len(orbit))
        try:
            derived.recent_approach_cosine_window(orbit)
        except ValueError:
            no_window.append(target)
    assert sorted(lengths)[0] == 1 and lengths.count(3001) == 46          # 46 reach back to -30; one target has no past
    assert len(no_window) == 13                                           # no close pass in their past to center a window on


SLOPE_AT_NOW = {62000000190717: (2, 194.9, 6.6),         # later rows, the curve's speed (km/s), its angle from the stars' velocity
                75000000121160: (0, 263.7, 130.4),       # now is the curve's free end, just after a close pass: the slope is no guide
                75000000317209: (5, 70.3, 23.5)}


def test_the_curves_slope_at_now_is_not_the_measured_velocity(measured):
    """The check at now: what the curve through the snapshots implies, against the stars' own velocity (pins)."""
    row, galaxies = measured
    target = int(row["p_SubhaloIDRaw"])
    later_rows, curve_speed, angle = SLOPE_AT_NOW[target]
    seen = galaxies["target"]["default"]
    table = tng.read_table(DATA / "moi-4-targets.h5")
    index = int(np.flatnonzero(table["p_SubhaloIDRaw"] == target)[0])
    rows = tng.encounter_rows(table, index)
    time = tng.track_time(table, rows, index)
    path = tng.track_separation(table, rows, seen["primary_side"]) @ seen["rotation"].T
    slope = tng.track_spline(time, path).slope(0.0) * tng.GYR_PER_KPC_PER_KMS            # kpc/Gyr -> km/s
    stars = np.asarray(seen["velocities"]["stars"])
    assert int((time > 1e-9).sum()) == later_rows
    assert np.linalg.norm(slope) == pytest.approx(curve_speed, abs=0.5)
    cosine = slope @ stars / (np.linalg.norm(slope) * np.linalg.norm(stars))
    assert np.degrees(np.arccos(cosine)) == pytest.approx(angle, abs=0.5)


@needs_data
def test_every_real_targets_last_pass_falls_between_snapshots():
    """Closest approach for all 61: 49 orbits have a close pass, and the curve, not a snapshot, sets how close."""
    table = tng.read_table(DATA / "moi-4-targets.h5")
    free_end, off_snapshot, deeper = 0, [], []
    for path in DATA.glob("target-archive-*.h5"):
        target = int(path.stem.split("-")[-1])
        row = int(np.flatnonzero(table["p_SubhaloIDRaw"] == target)[0])
        rows = tng.encounter_rows(table, row)
        time = tng.spam_units(tng.track_time(table, rows, row), "time")
        free_end += int(time[-1] < 1e-9)
        orbit = tng.target_orbit(table, row, {"target": target, "primary_side": "p", "rotation": np.eye(3)})
        if len(orbit) < 3:
            continue
        distance = np.linalg.norm(orbit[:, :3], axis=1)
        dips = np.flatnonzero((distance[1:-1] < distance[:-2]) & (distance[1:-1] <= distance[2:])) + 1
        if len(dips):
            nearest = int(np.abs(time - orbit[dips[-1], 6]).argmin())
            off_snapshot.append(abs(time[nearest] - orbit[dips[-1], 6]))
            at_snapshot = np.linalg.norm(tng.spam_units(tng.track_separation(table, rows[nearest:nearest + 1], "p"), "length"))
            deeper.append(distance[dips[-1]] / at_snapshot)
    assert free_end == 11                                         # targets with no later row: now is the curve's free end
    assert len(off_snapshot) == 49                                # the other 12 have no close pass in their orbit
    assert int((np.array(off_snapshot) < 0.2).sum()) == 2         # only 2 passes sit on a snapshot (they are 1.73 apart)
    assert np.median(off_snapshot) == pytest.approx(0.58, abs=0.03)
    assert np.median(deeper) == pytest.approx(0.67, abs=0.03)     # the curve's pass, as a share of the nearest snapshot's distance
    assert min(deeper) < 0.1                                      # and for some the curve dips far inside anything a snapshot saw


STARS = {62000000190717: (871840, 647983), 75000000121160: (1619317, 770091), 75000000317209: (1689610, 685381)}   # n1, n2 (pins)


def test_a_real_targets_stars_as_spams_particles(measured):
    """The stars come back in SPAM's frame: each galaxy's stars sit on its center and turn about the model's spin."""
    pytest.importorskip("galactic_tidal_constraints.simr.simulation")
    from galactic_tidal_constraints.simr import image_creator, metrics

    row, galaxies = measured
    target, result = galaxies["target"]["default"], galaxies["particles"]
    particles, masses, n1, n2 = result["particles"], result["masses"], result["n1"], result["n2"]
    model = tng.spam_model(target)
    assert (n1, n2) == STARS[int(row["p_SubhaloIDRaw"])]
    assert particles.shape == (n1 + n2 + 1, 6) and np.isfinite(particles).all() and masses.shape == (n1 + n2,)
    assert particles[-1] == pytest.approx(model[:6], abs=1e-12)
    for role, rows, center in (("primary", slice(0, n1), np.zeros(3)), ("secondary", slice(n1, n1 + n2), model[:3])):
        galaxy = target[role]
        assert masses[rows].sum() == pytest.approx(galaxy["stellar_mass"] / tng.SPAM_MASS, rel=1e-6)
        distance = np.linalg.norm(particles[rows, :3] - center, axis=1)
        # Half of the galaxy's stellar mass is within its half-mass radius of its center.
        inside = masses[rows][distance < galaxy["half_mass_radius"] / tng.SPAM_LENGTH].sum() / masses[rows].sum()
        assert inside == pytest.approx(0.5, abs=0.001)
        # Its stars within the spin radius turn about the model's disk angles.
        near = distance < 5.0 * galaxy["half_mass_radius"] / tng.SPAM_LENGTH
        spin = tng.galaxy_properties(particles[rows, :3][near], particles[rows, 3:][near], masses[rows][near])["spin"]
        assert tng.disk_angles(spin) == pytest.approx((galaxy["phi"], galaxy["theta"]), abs=0.01)
    # The particle metrics and the image maker take it as they take SPAM's own particles.
    shape = metrics.pca_shape(particles)
    assert np.isfinite(shape["axes"]).all() and 0 < shape["c_a"] <= shape["b_a"] <= 1
    image = image_creator.create_image(particles, model, n1, n2)
    assert image.ndim == 2 and image.max() == pytest.approx(1.0)


@needs_data
def test_a_real_targets_files_are_what_simr_takes(tmp_path):
    """Two real targets written, then given to SIMR as its command line would: the plan accepts them."""
    simr = pytest.importorskip("galactic_tidal_constraints.simr.simr")
    from galactic_tidal_constraints.simr import __main__ as cli

    lines = []
    records = tng.write_targets(DATA, tmp_path, targets=[75000000121160, 62000000190717, 5], report=lines.append)
    assert [record["target"] if "error" in record else record["target"]["target"] for record in records] == [
        5, 62000000190717, 75000000121160]                                           # in order of ID
    assert records[0]["error"] == "the table has 0 rows for it" and "NOT WRITTEN" in lines[0]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["62000000190717", "75000000121160", "tng_spam_models.txt", "tng_targets.h5"]

    record, folder = records[1], tmp_path / "62000000190717"
    assert record["stars"] == {"n1": 871840, "n2": 647983}
    assert record["orbit"]["samples"] == 3001 and record["orbit"]["recent_approach_window"] is True
    assert record["orbit"]["last_pass"]["time"] == pytest.approx(-2.12, abs=0.01)
    assert (record["orbit"]["rows_before"], record["orbit"]["rows_after"]) == (23, 2)
    assert np.load(folder / "tng_star_particles.npy", mmap_mode="r").shape == (871840 + 647983 + 1, 6)

    models = simr.read_models(tmp_path / "tng_spam_models.txt")
    assert models.shape == (2, 14) and models[0] == pytest.approx(simr.read_models(folder / "tng_spam_model.txt")[0])
    request = {"derived": [{"name": "window", "method": "recent_approach_cosine_window", "inputs": ["ext.tng_orbit"]}],
               "sim_metrics": [{"method": "orbit_position_error", "inputs": ["ext.tng_orbit", "sim.search_orbit", "drv.window"]},
                               {"method": "pca_difference", "inputs": ["ext.tng_star_particles", "sim.particles"]}]}
    plan = simr.plan(models[:1], request, externals=cli._read_externals(str(folder / "simr_externals.json")))
    assert {key: tuple(value["shape"]) for key, value in plan["externals"].items()} == {
        "tng_orbit": (3001, 7), "tng_star_particles": (871840 + 647983 + 1, 6)}

    import h5py
    with h5py.File(tmp_path / "tng_targets.h5") as file:
        assert list(file["target"][...]) == [62000000190717, 75000000121160]
        assert file["model"][...] == pytest.approx(models) and list(file["model"].attrs["parameters"]) == list(tng.SPAM_PARAMETERS)
        assert list(file["recent_approach_window"][...]) == [True, True] and list(file["rows_after"][...]) == [2, 0]
        assert file["primary_mass_historical"][0] == pytest.approx(record["target"]["primary"]["masses"]["historical"])


@needs_data
def test_every_real_targets_orbit_notes():
    """The notes for all 61, against what the orbit surveys found (no particles are read: distances need no frame)."""
    from galactic_tidal_constraints.simr import derived

    table = tng.read_table(DATA / "moi-4-targets.h5")
    counts = {"pass": 0, "window": 0, "edge": 0, "free_end": 0}
    for path in DATA.glob("target-archive-*.h5"):
        target = {"target": int(path.stem.split("-")[-1]), "primary_side": "p", "rotation": np.eye(3)}
        row = int(np.flatnonzero(table["p_SubhaloIDRaw"] == target["target"])[0])
        orbit = tng.target_orbit(table, row, target)
        notes = tng.orbit_notes(table, row, target, orbit)
        try:
            derived.recent_approach_cosine_window(orbit)
            window = True
        except ValueError:
            window = False
        assert notes["recent_approach_window"] is window, target["target"]     # the note says what the metric's window does
        counts["pass"] += notes["last_pass"] is not None
        counts["window"] += window
        counts["edge"] += notes["crosses_box_edge"]
        counts["free_end"] += notes["rows_after"] == 0
    assert counts == {"pass": 49, "window": 48, "edge": counts["edge"], "free_end": 11}
    assert 3 <= counts["edge"] <= 12                    # three encounters cross the box edge; each holds one or more targets
