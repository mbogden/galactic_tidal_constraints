"""Tests for the TNG readers (tng.py): the target table and a target archive.

The first sections use small made-up files and always run. The last section reads the real files and
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
