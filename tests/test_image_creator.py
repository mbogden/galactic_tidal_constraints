"""Tests for the Image Creator stage (simr/image_creator.py). Each section covers one step of MDRN-3."""

import struct
import zlib

import pytest

np = pytest.importorskip("numpy")

from galactic_tidal_constraints.simr import image_creator as img  # noqa: E402
from galactic_tidal_constraints.simr.image_creator import ImageConfig  # noqa: E402

# Model SDSS 587722984435351614 (as in test_simulation.py): its secondary is up and to the left in the sky.
PARAMS = np.array(
    [-9.93853, -4.5805, 15.43348, -1.54025, -2.97899, 4.69234, 34.41671, 29.49607, 2.97913, 4.2501,
     301.0, 35.5, 310.80745, 321.98757]
)  # fmt: skip
EAST = np.array([10.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1, 1, 1, 1, 0, 0, 0, 0])  # secondary 10 units along +x


def fake_particles(params, primary_xyz, secondary_xyz):
    """A particle array with the given disk positions and the model's secondary as the last row."""
    rows = [list(p) + [0, 0, 0] for p in primary_xyz] + [list(p) + [0, 0, 0] for p in secondary_xyz]
    rows.append(list(params[:6]))
    return np.array(rows, dtype=float)


# --- Step 1: the config ----------------------------------------------------------------------------------


def test_config_defaults_and_lists_become_tuples():
    config = ImageConfig(primary_center=[10, 20], secondary_center=[30, 20])
    assert config.primary_center == (10.0, 20.0)
    assert config.secondary_center == (30.0, 20.0)
    assert ImageConfig().width == 128 and ImageConfig().brightness == "linear"


@pytest.mark.parametrize(
    "fields",
    [
        {"width": 0},
        {"height": 1.5},
        {"primary_center": (1, 2, 3)},
        {"secondary_center": "middle"},
        {"primary_center": (5, 5), "secondary_center": (5, 5)},
        {"blur_sigma": -1},
        {"blur_truncate": 0},
        {"brightness": "sqrt"},
    ],
)
def test_config_rejects_bad_values(fields):
    with pytest.raises(ValueError):
        ImageConfig(**fields)


def test_centers_may_lie_outside_the_image():
    ImageConfig(primary_center=(64, 64), secondary_center=(400, -50))  # zooming in on the primary


# --- Step 1: the frame -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "centers", [((48, 64), (80, 64)), ((10.5, 100), (90, 3)), ((64, 64), (400, -50)), ((80, 64), (48, 64))]
)
def test_centers_land_on_their_pixels(centers):
    config = ImageConfig(primary_center=centers[0], secondary_center=centers[1])
    pixels = img.to_pixels(np.array([[0.0, 0.0, 0.0], PARAMS[:3]]), PARAMS, config)
    np.testing.assert_allclose(pixels, np.array(centers, dtype=float), atol=1e-9)


def test_sky_up_is_image_up_and_not_mirrored():
    # Secondary along +x lands to the right (default centers): +y must point up (smaller row), not down
    up, right = img.to_pixels(np.array([[0.0, 5.0, 0.0], [5.0, 0.0, 0.0]]), EAST, ImageConfig())
    assert up[1] < 64 and np.isclose(up[0], 48)
    assert right[0] > 48 and np.isclose(right[1], 64)


def test_frame_keeps_shapes():
    # A rotation and a scale, nothing else: distances shrink or grow by one factor, angles are kept
    rng = np.random.default_rng(0)
    points = rng.normal(size=(20, 3))
    pixels = img.to_pixels(points, PARAMS, ImageConfig(secondary_center=(20, 5)))
    sky = np.linalg.norm(points[:, None, :2] - points[None, :, :2], axis=-1)
    pix = np.linalg.norm(pixels[:, None] - pixels[None, :], axis=-1)
    ratio = pix[sky > 0] / sky[sky > 0]
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-9)


def test_line_of_sight_is_ignored():
    points = np.array([[1.0, 2.0, 0.0], [1.0, 2.0, 50.0]])
    moved = PARAMS.copy()
    moved[2] += 7.0
    first, second = img.to_pixels(points, PARAMS, ImageConfig())
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(img.frame_matrix(PARAMS), img.frame_matrix(moved))


def test_secondary_behind_the_primary_cant_be_framed():
    behind = EAST.copy()
    behind[:3] = [0.0, 0.0, 5.0]
    with pytest.raises(ValueError, match="on top of the primary"):
        img.frame_matrix(behind)


def test_to_pixels_reads_only_the_first_three_columns():
    orbit_like = np.array([[1.0, 2.0, 3.0, 9.0, 9.0, 9.0, -1.0]])
    np.testing.assert_allclose(
        img.to_pixels(orbit_like, PARAMS), img.to_pixels(orbit_like[:, :3], PARAMS)
    )


# --- Step 1: the image -----------------------------------------------------------------------------------


def test_particles_land_in_their_pixels():
    config = ImageConfig(width=40, height=30, primary_center=(10, 15), secondary_center=(30, 15))
    # One unit of sky is 2 pixels here (10 units apart, 20 pixels)
    particles = fake_particles(EAST, [[0, 0, 0], [0, 0, 0], [1, 0, 0]], [[10, 0, 0], [10, 2, 0]])
    result = img.create_image(particles, EAST, 3, 2, config)
    assert result.dtype == np.float32 and result.shape == (30, 40)
    assert result[15, 10] == 1.0  # two primary particles at its center: the brightest pixel
    assert result[15, 12] == 0.5 and result[15, 30] == 0.5 and result[11, 30] == 0.5
    assert np.count_nonzero(result) == 4


def test_log_brightness():
    config = ImageConfig(width=40, height=30, primary_center=(10, 15), secondary_center=(30, 15), brightness="log")
    particles = fake_particles(EAST, [[0, 0, 0]] * 3, [[10, 0, 0]])
    result = img.create_image(particles, EAST, 3, 1, config)
    assert result[15, 10] == 1.0
    np.testing.assert_allclose(result[15, 30], np.log(2) / np.log(4), rtol=1e-6)


def test_particles_outside_the_image_are_dropped_and_empty_is_zero():
    config = ImageConfig(primary_center=(500, 500), secondary_center=(600, 500))
    particles = fake_particles(EAST, [[0, 0, 0]], [[10, 0, 0]])
    result = img.create_image(particles, EAST, 1, 1, config)
    assert np.isfinite(result).all() and not result.any()


def test_wrong_rows_fail():
    particles = fake_particles(EAST, [[0, 0, 0]] * 3, [[10, 0, 0]] * 2)
    with pytest.raises(ValueError, match="rows"):
        img.create_image(particles, EAST, 2, 2)
    flipped = np.roll(particles, 1, axis=0)  # the center first, not last
    with pytest.raises(ValueError, match="last row"):
        img.create_image(flipped, EAST, 3, 2)


def test_blur_keeps_light_away_from_edges_and_loses_it_at_them():
    middle = np.zeros((31, 31))
    middle[15, 15] = 1.0
    blurred = img._blur(middle, 2.0, 4.0)
    np.testing.assert_allclose(blurred.sum(), 1.0, rtol=1e-12)
    np.testing.assert_allclose(blurred, blurred.T, atol=1e-15)  # round
    np.testing.assert_allclose(blurred, blurred[::-1, ::-1], atol=1e-15)
    assert blurred[15, 15 + 9] == 0 and blurred[15, 15 + 8] > 0  # radius round(4 * 2) = 8
    corner = np.zeros((31, 31))
    corner[0, 0] = 1.0
    assert img._blur(corner, 2.0, 4.0).sum() < 0.5
    np.testing.assert_array_equal(img._blur(middle, 0.0, 4.0), middle)


def test_blur_matches_a_direct_gaussian():
    rng = np.random.default_rng(1)
    picture = rng.random((12, 17))
    sigma, radius = 1.3, 5  # round(4 * 1.3)
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(picture, radius)
    direct = np.zeros_like(picture)
    for dr in offsets:
        for dc in offsets:
            shifted = padded[radius + dr : radius + dr + 12, radius + dc : radius + dc + 17]
            direct += kernel[dr + radius] * kernel[dc + radius] * shifted
    np.testing.assert_allclose(img._blur(picture, sigma, 4.0), direct, rtol=1e-12)


# --- Step 1: viewing -------------------------------------------------------------------------------------


def test_to_png_writes_the_pixels(tmp_path):
    picture = np.linspace(0, 1, 12, dtype=np.float32).reshape(3, 4)
    path = tmp_path / "picture.png"
    img.to_png(picture, path)
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height, depth, color = struct.unpack(">IIBB", data[16:26])
    assert (width, height, depth, color) == (4, 3, 8, 0)
    length = struct.unpack(">I", data[33:37])[0]
    assert data[37:41] == b"IDAT"
    rows = zlib.decompress(data[41 : 41 + length])
    stored = np.frombuffer(rows, dtype=np.uint8).reshape(3, 5)
    assert (stored[:, 0] == 0).all()  # filter bytes
    np.testing.assert_array_equal(stored[:, 1:], np.round(picture * 255).astype(np.uint8))


# --- Step 1: with real simulations -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def runs():
    pytest.importorskip("jspam")
    from galactic_tidal_constraints.simr import simulation as sim

    config = sim.SimulatorConfig(n1=400, n2=150)
    return {
        "config": config,
        "particles": sim.simulate_particles(PARAMS, config).astype(np.float32),
        "unperturbed": sim.unperturbed_particles(PARAMS, config).astype(np.float32),
        "orbit": sim.simulate_orbit(PARAMS, config).astype(np.float32),
    }


@pytest.mark.parametrize("which", ["particles", "unperturbed"])
@pytest.mark.parametrize("settings", [{}, {"blur_sigma": 1.5, "brightness": "log"}])
def test_real_particles_make_an_image(runs, which, settings):
    config = ImageConfig(**settings)
    result = img.create_image(runs[which], PARAMS, runs["config"].n1, runs["config"].n2, config)
    assert result.dtype == np.float32 and result.shape == (128, 128)
    assert np.isfinite(result).all() and result.min() >= 0 and result.max() == 1.0
    assert np.count_nonzero(result) > 50


def test_orbit_ends_at_the_secondary_pixel(runs):
    config = ImageConfig()
    pixels = img.to_pixels(runs["orbit"], PARAMS, config)
    assert pixels.shape == (len(runs["orbit"]), 2)
    assert np.linalg.norm(pixels[-1] - np.array(config.secondary_center)) < 1.0  # a run ends 1-2 steps past now


def test_real_particles_need_the_simulators_counts(runs):
    n1, n2 = runs["config"].n1, runs["config"].n2
    with pytest.raises(ValueError, match="rows"):
        img.create_image(runs["particles"], PARAMS, n1, n2 + 1)


def test_real_run_centers_are_within_tolerance(runs):
    # The check that makes create_image fail must pass for a real run and its unperturbed particles
    for which in ("particles", "unperturbed"):
        img._check_particles(runs[which], PARAMS, runs["config"].n1, runs["config"].n2)
