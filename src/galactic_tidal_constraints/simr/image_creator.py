"""Image Creator stage: turns one particle set into one image, taking values and returning arrays.

Like `sim`, it doesn't touch the disk (apart from `to_png`, which is for looking at an image), know about
pools or MPI, or keep state between calls. It imports neither `jspam` nor `sim`: a model is a plain array
of its 14 parameters (`sim.PARAM_NAMES`), and only the first six (the secondary's position and velocity
relative to the primary, which sits at the origin) are read here.

Pixels: an image is an array of shape (height, width); `image[row, column]`. Positions in pixels are
(column, row) pairs, so they plot directly over `plt.imshow(image)`: integer positions are pixel centers,
and pixel (c, r) covers columns c - 0.5 to c + 0.5 and rows r - 0.5 to r + 0.5. Rows count down, so the
sky's +y points up in the image.

How an image is made (`create_image`):
    1. Frame: one matrix for the model (`frame_matrix`) maps every particle to its pixel position.
    2. Bin: count each galaxy's particles per pixel on its own, with a weight per particle (all 1 for now).
    3. Combine: add the two galaxies.
    4. Blur, then brightness, then divide by the max, giving float32 values in [0, 1].
"""

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# How the counts become brightness, before dividing by the max.
#   linear: the counts as they are
#   log:    log(1 + counts), which brightens faint regions. It depends on the counts themselves, so the same
#           model with more particles or at another resolution looks different.
BRIGHTNESS = ("linear", "log")

# The secondary's center (the particles' last row) must be this close to the model's position and velocity:
# a fraction of max(|x, y, z|, 1) and of max(|vx, vy, vz|, 1). A run ends 1 to 2 steps past now, so its
# center is close but not exact: up to 3.8% across JSPAM's 62 archive models (2026-11-16).
CENTER_TOLERANCE = 0.10


@dataclass(frozen=True)
class ImageConfig:
    """Everything that sets an image, apart from the particles and the model.

    Attributes:
        width, height: image size in pixels.
        primary_center, secondary_center: the (column, row) pixel positions the two galaxies' centers land on.
            They may lie outside the image, e.g. to zoom in on one galaxy, but can't be equal. Higher-level
            code decides whether they come from a target image.
        blur_sigma: Gaussian blur's standard deviation, in pixels. 0 (the default) means no blur.
        blur_truncate: the blur kernel reaches this many sigmas each side (scipy's name and default, 4).
            Light blurred past the image's edges is lost.
        brightness: one of `BRIGHTNESS`, applied after the blur.
    """

    width: int = 128
    height: int = 128
    primary_center: tuple[float, float] = (32.0, 64.0)
    secondary_center: tuple[float, float] = (96.0, 64.0)
    blur_sigma: float = 0.0
    blur_truncate: float = 4.0
    brightness: str = "log"

    def __post_init__(self) -> None:
        """Check the values when the config is made, so mistakes show up before any run. Centers given as
        lists (as JSON writes them) become tuples of floats."""
        for name in ("width", "height"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a whole number of at least 1: {value!r}")
        for name in ("primary_center", "secondary_center"):
            value = getattr(self, name)
            try:
                center = tuple(float(v) for v in value)
            except (TypeError, ValueError):
                center = ()
            if len(center) != 2 or not np.isfinite(center).all():
                raise ValueError(f"{name} must be two numbers (column, row): {value!r}")
            object.__setattr__(self, name, center)
        if self.primary_center == self.secondary_center:
            raise ValueError(f"primary_center and secondary_center can't be equal: {self.primary_center}")
        if not self.blur_sigma >= 0:
            raise ValueError(f"blur_sigma can't be negative: {self.blur_sigma!r}")
        if not self.blur_truncate > 0:
            raise ValueError(f"blur_truncate must be positive: {self.blur_truncate!r}")
        if self.brightness not in BRIGHTNESS:
            raise ValueError(f"brightness must be one of {list(BRIGHTNESS)}: {self.brightness!r}")


# --- The frame -------------------------------------------------------------------------------------------


def frame_matrix(params: np.ndarray | list[float], config: ImageConfig = ImageConfig()) -> np.ndarray:
    """The matrix that takes positions in SPAM units to pixel positions, for one model.

    Built from the model alone (primary at the origin, secondary at x, y, z), so a model's particles,
    unperturbed particles and orbit all share it exactly. It chains, applied right to left:
        shift (primary to its pixel) · flip y (rows count down) · rotate and scale in the plane (secondary
        to its pixel) · drop the line of sight · view rotation

    Returns:
        shape (2, 4): `matrix @ [x, y, z, 1]` is the (column, row) pixel position. `to_pixels` applies it.

    Raises:
        ValueError: the secondary is on top of the primary as seen in the view, so no scale fits.
    """
    params = np.asarray(params, dtype=float)
    to_plane = _projection() @ _view_rotation()  # (2, 3): SPAM coordinates to sky-plane coordinates
    secondary = to_plane @ params[:3]
    if np.hypot(*secondary) <= 1e-9 * max(1.0, float(np.linalg.norm(params[:3]))):
        raise ValueError(f"the secondary is on top of the primary in the image's view: x, y, z = {params[:3]}")
    flip = np.diag([1.0, -1.0])
    primary_px = np.array(config.primary_center)
    wanted = flip @ (np.array(config.secondary_center) - primary_px)  # where the secondary goes, sky-plane axes
    # The rotation and scale that turn `secondary` into `wanted`, as one complex factor: wanted / secondary
    factor = complex(*wanted) / complex(*secondary)
    rotate_scale = np.array([[factor.real, -factor.imag], [factor.imag, factor.real]])
    matrix = np.empty((2, 4))
    matrix[:, :3] = flip @ rotate_scale @ to_plane
    matrix[:, 3] = primary_px
    return matrix


def to_pixels(
    positions: np.ndarray, params: np.ndarray | list[float], config: ImageConfig = ImageConfig()
) -> np.ndarray:
    """Pixel positions of any positions in a model's frame: particles, unperturbed particles or an orbit.

    Args:
        positions: shape (N, 3) or more columns; only the first three (x, y, z) are read, so
            `sim.simulate_particles`'s (N, 6) and `sim.simulate_orbit`'s (N, 7) arrays work as they are.
        params: the model's 14 parameters; only x, y, z are read.

    Returns:
        (column, row) of each position, shape (N, 2), float64. Plot an orbit over an image with
        `plt.imshow(image); plt.plot(*to_pixels(orbit, params, config).T)`.
    """
    matrix = frame_matrix(params, config)
    positions = np.asarray(positions, dtype=float)
    return positions[:, :3] @ matrix[:, :3].T + matrix[:, 3]


def _view_rotation() -> np.ndarray:
    """Which way the image looks at the encounter, as a 3D rotation applied first. The identity: looking down
    z onto the x–y plane (the sky plane, z along the line of sight), as the old code did. The hook for other
    views later."""
    return np.eye(3)


def _projection() -> np.ndarray:
    """Drops the line of sight (z, after the view rotation): (2, 3)."""
    return np.eye(2, 3)


# --- The image -------------------------------------------------------------------------------------------


def create_image(
    particles: np.ndarray, params: np.ndarray | list[float], n1: int, n2: int, config: ImageConfig = ImageConfig()
) -> np.ndarray:
    """One image of a particle set, such as `sim.simulate_particles` or `sim.unperturbed_particles`.

    Args:
        particles: shape (n1 + n2 + 1, 6 or more): the primary's n1 particles, the secondary's n2, then the
            secondary's center.
        params: the model the particles came from (14 parameters); it sets the frame.
        n1, n2: particles in each galaxy (the simulator config's), which split the rows.

    Returns:
        float32, shape (height, width), values in [0, 1]. All zeros if no particle lands in the image.

    Raises:
        ValueError: the rows don't number n1 + n2 + 1, the last row isn't at the model's secondary (see
            `CENTER_TOLERANCE`), or the model can't be framed (`frame_matrix`).
    """
    particles = np.asarray(particles)
    _check_particles(particles, params, n1, n2)
    pixels = to_pixels(particles[:-1], params, config)  # the disks; the center row isn't drawn
    weights = _particle_weights(n1, n2)
    counts = np.stack(
        [
            _bin(pixels[:n1], weights[:n1], config.width, config.height),
            _bin(pixels[n1:], weights[n1:], config.width, config.height),
        ]
    )  # (2, height, width): one per galaxy, kept apart so each can be weighted or saved on its own later
    image = counts.sum(axis=0)
    image = _blur(image, config.blur_sigma, config.blur_truncate)
    if config.brightness == "log":
        image = np.log1p(image)
    peak = image.max()
    if peak > 0:
        image = image / peak
    return image.astype(np.float32)


def _check_particles(particles: np.ndarray, params, n1: int, n2: int) -> None:
    if particles.ndim != 2 or particles.shape[0] != n1 + n2 + 1 or particles.shape[1] < 6:
        raise ValueError(
            f"particles must have n1 + n2 + 1 = {n1 + n2 + 1} rows and at least 6 columns: {particles.shape}"
        )
    params = np.asarray(params, dtype=float)
    center = particles[-1, :6].astype(float)
    off_position = np.linalg.norm(center[:3] - params[:3]) / max(1.0, float(np.linalg.norm(params[:3])))
    off_velocity = np.linalg.norm(center[3:6] - params[3:6]) / max(1.0, float(np.linalg.norm(params[3:6])))
    if off_position > CENTER_TOLERANCE or off_velocity > CENTER_TOLERANCE:
        raise ValueError(
            f"the particles' last row isn't the model's secondary: {center} against {params[:6]} "
            f"(off by {off_position:.1%} in position and {off_velocity:.1%} in velocity)"
        )


def _particle_weights(n1: int, n2: int) -> np.ndarray:
    """How much light each disk particle gives, shape (n1 + n2,). All 1 for now; the hook for radial
    brightness and luminosity balance later."""
    return np.ones(n1 + n2)


def _bin(pixels: np.ndarray, weights: np.ndarray, width: int, height: int) -> np.ndarray:
    """Adds each position's weight to the pixel it falls in; positions outside the image are dropped.
    Returns float64, shape (height, width)."""
    columns = np.floor(pixels[:, 0] + 0.5).astype(np.int64)
    rows = np.floor(pixels[:, 1] + 0.5).astype(np.int64)
    inside = (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
    flat = rows[inside] * width + columns[inside]
    return np.bincount(flat, weights=weights[inside], minlength=width * height).reshape(height, width)


def _blur(image: np.ndarray, sigma: float, truncate: float) -> np.ndarray:
    """Gaussian blur, with zeros past the edges. The kernel reaches round(truncate * sigma) pixels each side
    (as in scipy.ndimage.gaussian_filter) and sums to 1."""
    if sigma == 0:
        return image
    radius = int(truncate * sigma + 0.5)
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()

    def along(n: int) -> np.ndarray:  # (n, n) matrix that blurs a length-n line: entry [i, j] = kernel[j - i]
        gap = np.arange(n)[None, :] - np.arange(n)[:, None]
        return np.where(np.abs(gap) <= radius, kernel[np.clip(gap + radius, 0, 2 * radius)], 0.0)

    height, width = image.shape
    return along(height) @ image @ along(width).T


# --- Viewing ---------------------------------------------------------------------------------------------


def to_png(image: np.ndarray, path: str | Path) -> None:
    """Writes an image as an 8-bit grayscale PNG, for looking at it: values in [0, 1] become 0 to 255
    (values outside are clipped). Row 0 is the top of the picture, as in `plt.imshow`."""
    pixels = np.round(np.clip(np.asarray(image, dtype=float), 0.0, 1.0) * 255).astype(np.uint8)
    if pixels.ndim != 2:
        raise ValueError(f"an image must be 2D (height, width): {pixels.shape}")
    height, width = pixels.shape

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    rows = b"".join(b"\x00" + row.tobytes() for row in pixels)  # each row starts with filter type 0 (none)
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)  # 8-bit grayscale, no interlace
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")
    Path(path).write_bytes(png)
