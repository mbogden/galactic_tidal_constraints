# Image Creator stage (`simr/image_creator.py`)

The Image Creator turns one particle set into one image. Like the Simulator stage it takes values and
returns arrays: no disk (apart from `to_png`, for looking at an image), no pools or MPI, no state.
It imports neither `jspam` nor `sim`. `simr` runs it for many models: the `image` request section and
the `image.model` / `image.unperturbed` items ([simr.md](simr.md)).

```python
from galactic_tidal_constraints.simr import image_creator as img
from galactic_tidal_constraints.simr import simulation as sim
from galactic_tidal_constraints.simr.image_creator import ImageConfig

config = ImageConfig(blur_sigma=1.0, brightness="log")
particles = sim.simulate_particles(params, sim_config)
picture = img.create_image(particles, params, sim_config.n1, sim_config.n2, config)
orbit_px = img.to_pixels(sim.simulate_orbit(params, sim_config), params, config)
```

Only what exists today is described here. Decisions behind it are in [PLAN.md](../../PLAN.md)
("Image stage").

## `ImageConfig`

| Field | Default | Meaning |
|---|---|---|
| `width`, `height` | 128, 128 | Image size in pixels |
| `primary_center` | (48, 64) | The (column, row) pixel the primary's center lands on |
| `secondary_center` | (80, 64) | The same for the secondary. Either may lie outside the image (to zoom in on one galaxy); the two can't be equal |
| `blur_sigma` | 0 | Gaussian blur's standard deviation in pixels; 0 = none |
| `blur_truncate` | 4 | The kernel reaches this many sigmas each side (scipy's name and default) |
| `brightness` | `"linear"` | `"linear"` (the counts) or `"log"` (log(1 + counts)), applied after the blur |

Frozen, checked when made (a bad value is a `ValueError`). Centers given as lists, as JSON writes them,
become tuples of floats.

## Pixels
An image is an array of shape `(height, width)`, indexed `image[row, column]`. Pixel positions are
`(column, row)` pairs, and integer positions are pixel centers, so they plot straight over
`plt.imshow(image)`. Rows count down; the sky's +y points up in the picture.

## The frame
`frame_matrix(params, config)` → shape `(2, 4)`; `matrix @ [x, y, z, 1]` is a position's pixel.

It's built from the model alone, with the primary at the origin and the secondary at the model's `x, y, z`,
so a model's particles, unperturbed particles and orbit share it exactly (a run's last row sits 1–2 steps
past now, so it's close to, not on, the model's position). It chains, applied right to left:

1. **View rotation:** a 3D rotation. The identity for now: look down z onto the x–y (sky) plane, as the
   old code did. The hook for other views.
2. **Projection:** drop the line of sight.
3. **Rotate and scale** in the plane so the secondary lands on its pixel relative to the primary.
4. **Flip y** (rows count down).
5. **Shift** the primary onto its pixel.

A secondary on top of the primary in the view (x = y = 0) can't be framed: `ValueError`.

`to_pixels(positions, params, config)` applies the frame to any `(N, ≥3)` array (only `x, y, z` are
read), so `sim.simulate_particles`' `(N, 6)` and `sim.simulate_orbit`'s `(N, 7)` arrays work as they are.
Returns `(N, 2)` (column, row), float64.

## `create_image(particles, params, n1, n2, config)`
Takes a particle set of shape `(n1 + n2 + 1, ≥6)` (primary's disk, secondary's disk, secondary's center),
the model it came from, and the simulator's `n1` and `n2`. Returns `float32`, `(height, width)`, in
[0, 1]; all zeros if no particle lands in the image.

1. **Check:** the rows number `n1 + n2 + 1`, and the last row is within `CENTER_TOLERANCE` (10%) of the
   model's position and velocity, as a fraction of max(|x, y, z|, 1) and max(|vx, vy, vz|, 1). Catches
   the wrong particles, or the center row in the wrong place. A run's offset is at most 3.8% across
   JSPAM's 62 archive models.
2. **Frame** the disk particles (the center row isn't drawn).
3. **Bin** each galaxy on its own into `(2, height, width)` counts, with a weight per particle (all 1 for
   now). Positions outside the image are dropped.
4. **Combine:** add the two galaxies.
5. **Blur:** Gaussian, separable, with zeros past the edges, so light blurred off the image is lost.
6. **Brightness**, then **divide by the max**.

Any failed check is a `ValueError`.

## `to_png(image, path)`
Writes an 8-bit grayscale PNG: values in [0, 1] become 0–255 (others clipped), row 0 at the top. For
looking at an image only; standard library (`zlib`, `struct`), no dependency.

## Hooks for later
Kept as their own small functions so later options slot in:
- `_particle_weights`: radial brightness, luminosity balance.
- The `(2, height, width)` counts before combining: separate galaxy images, per-galaxy factors.
- `_view_rotation`: other view planes.
- `_blur`: could become `scipy.ndimage.gaussian_filter(mode="constant")` if scipy is ever added; same kernel.
