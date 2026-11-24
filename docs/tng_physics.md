# TNG physics notes

What the IllustrisTNG values this project uses actually mean: their units, the simulation box, how
galaxies are defined, and how each fact was checked. The functions that use these facts are described
in [modules/tng.md](modules/tng.md).

Every fact here names its source: the TNG documentation
([background](https://www.tng-project.org/data/docs/background/),
[data specifications](https://www.tng-project.org/data/docs/specifications/), read 2026-11-24), or a
test in `tests/test_tng.py` that checks it on this project's files. Quotes are the documentation's
own wording.

## Acknowledgement
This project uses data from the [IllustrisTNG](https://www.tng-project.org) project's TNG50
simulation. No TNG data is in this repository; the code reads files kept outside it. As the TNG
project asks of work that uses TNG50, we cite its public data release and the two papers that
introduce TNG50:

- Nelson et al. 2019a, the IllustrisTNG public data release ([arXiv:1812.05609](https://arxiv.org/abs/1812.05609))
- Nelson et al. 2019b, TNG50: outflows ([arXiv:1902.05554](https://arxiv.org/abs/1902.05554))
- Pillepich et al. 2019, TNG50: structure and kinematics ([arXiv:1902.05553](https://arxiv.org/abs/1902.05553))

The simulations were run by the TNG collaboration on the Hazel Hen supercomputer at HLRS, with
compute time from the Gauss Centre for Supercomputing, and on machines of the MPCDF. Their full
citation request, including the acknowledgement text they suggest for papers, is in the
[background page](https://www.tng-project.org/data/docs/background/).

## What TNG50-1 is
A cosmological simulation: a cube of the universe followed from early times to today, with gravity,
gas, star formation and black holes. Galaxies form in it on their own, and some of them collide.
TNG50-1 is the highest-resolution run of TNG50 (its run name is `L35n2160TNG`: a box 35 Mpc/h on a
side, 2 × 2160³ resolution elements).

| Fact | Value | Source |
|---|---|---|
| Cosmology | "consistent with the Planck 2015 results": Ωm = 0.3089, ΩΛ = 0.6911, Ωb = 0.0486, σ8 = 0.8159, ns = 0.9667, h = 0.6774 | Background page |
| Box side | 35,000 ckpc/h = 35 Mpc/h, about 51.7 Mpc today | The run's name; the target table's coordinates run from 20 to 34,997 |
| Dark matter particle mass | 4.54 × 10⁵ M☉ (stored as 3.0737 × 10⁻⁵) | The target archives: `test_real_archives_hold_raw_tng_masses` |
| Snapshots | 100, numbered 0 to 99; 99 is today | Background page |

Each snapshot has a redshift z and a scale factor a = 1 / (1 + z): the size of the universe then,
relative to today (a = 1). The target table carries both for every row, and `time_gyr`, the age of
the universe at that snapshot in this cosmology.

## The box
The cube has no walls. The documentation calls it "the periodic simulation domain": what leaves
through one face comes back in through the opposite one, and gravity acts across the faces too. The
real universe has no edge, and this is how a finite simulation avoids having one.

Stored coordinates therefore run from 0 to 35,000 along each axis, and **two points close together
can have coordinates almost a whole box apart**: one at x = 34,950 and one at x = 50 are 100 apart,
not 34,900. Longitude works the same way: 179° East and 179° West are 2° apart, not 358°. Positions
have to be compared the short way round. The same applies inside one galaxy that sits across a face:
some of its particles have coordinates near 35,000 and the rest near 0.

`tng.nearest_image` does this. In this project's target table, 20 rows in 6 encounters have the two
galaxies on either side of a face (`test_real_pairs_split_by_the_box_edge_are_mended_by_nearest_image`).

## Units
TNG stores values in units that follow the expanding universe, so most need a factor before they are
kpc, km/s or M☉. `tng.physical` applies these; a is the snapshot's scale factor and h = 0.6774.

| Quantity | Stored unit (documentation) | To physical | Why |
|---|---|---|---|
| Lengths: particle `Coordinates`, `SubhaloPos`, `SubhaloCM`, `SubhaloHalfmassRad` | ckpc/h, "Comoving coordinate" | × a ÷ h → kpc | Comoving lengths grow with the universe: a distance stored as 100 was a × 100 at that time. "Per h" is divided out |
| Masses: particle `Masses`, `SubhaloMass`, the header's `MassTable` | 10¹⁰ M☉/h | × 10¹⁰ ÷ h → M☉ | "Per h" again |
| Particle `Velocities` | km √a / s | × √a → km/s | "Multiply this value by sqrt(a) to obtain peculiar velocity" |
| Catalog `SubhaloVel` | km/s | none | "Peculiar velocity of the group ... No unit conversion is needed" |
| Catalog `SubhaloSpin` | (kpc/h)(km/s) | ÷ h → kpc km/s | Its length is already physical, not comoving (checked below) |

- **A peculiar velocity** is the motion left after the universe's expansion is removed: how fast
  something moves through space, not how fast space carries it away.
- **Particle velocities and catalog velocities are stored differently.** The √a belongs only to the
  particles. Leaving it out makes two galaxies' relative velocity 16% too fast at a = 0.74 and 27%
  at a = 0.62 (`test_velocities_need_sqrt_a_to_match_the_catalogs_relative_velocity`).
- Dark matter particles have no `Masses` entry; they all have the one mass in the header's
  `MassTable` ("Masses of particle types which have a constant mass (only DM)").

## Galaxies in the catalog
TNG finds galaxies with the SUBFIND algorithm: a galaxy (a "subhalo") is a set of particles of all
types (gas, dark matter, stars, black holes) that are gravitationally bound together. The catalog
then lists properties computed from each subhalo's member particles.

| Field | The documentation's description | Notes |
|---|---|---|
| `SubhaloMass` | "Total mass of all member particle/cells which are bound to this Subhalo, of all types" | Mostly dark matter: the whole bound halo, not the visible galaxy |
| `SubhaloPos` | "Spatial position within the periodic box (of the particle with the minium [sic] gravitational potential energy)" | The most bound particle: the galaxy's nucleus |
| `SubhaloCM` | "Comoving center of mass of the Subhalo ... of all particles/cells in the Subhalo, of all types" | Follows the dark matter halo, so it can sit far from the nucleus |
| `SubhaloVel` | "Peculiar velocity of the group, computed as the sum of the mass weighted velocities of all particles/cells in this group, of all types" | The mass-weighted mean velocity |
| `SubhaloSpin` | "Total spin per axis, computed for each as the mass weighted sum of the relative coordinate times relative velocity of all member particles/cells" | Per unit mass, physical length (checked below) |
| `SubhaloHalfmassRad` | "Comoving radius containing half of the total mass (SubhaloMass) of this Subhalo" | Measured from `SubhaloPos` (checked below) |

- **`SubhaloPos` and `SubhaloCM` can differ a lot during an encounter.** For target
  `62000000190717` the two nuclei are 35 kpc apart, while the two centers of mass are under 6 kpc
  apart: the larger galaxy's halo has absorbed most of the pair's dark matter, which drags its center
  of mass away from its nucleus. This project uses the nucleus as a galaxy's center.
- **SUBFIND's membership shifts during an encounter.** As two galaxies overlap, it gives more and
  more of the smaller one's particles to the larger. The archives also carry a "historical" label:
  the galaxy each particle belonged to before the encounter.

## Checked against particles
The documentation's wording leaves some things open: whether `SubhaloSpin` is a total or per unit
mass, and whether its length is comoving. So each catalog value was recomputed from the particles of
the same galaxy and compared, for both galaxies of targets `62000000190717`, `75000000121160` and
`75000000317209`. The tests in `tests/test_tng.py` repeat this whenever the data folder is available.

| Recomputed from the particles | Catalog field | Largest difference | Test |
|---|---|---|---|
| Sum of masses | `SubhaloMass` | 7 × 10⁻⁸ | `test_particle_masses_add_up_to_the_catalogs_mass` |
| Distance between the two centers of mass | The same from `SubhaloCM` | 4 × 10⁻⁵ | `test_positions_are_comoving_the_separation_matches_the_catalog_as_stored` |
| Difference of the two mean velocities, × √a | The same from `SubhaloVel` | 2 × 10⁻⁷ | `test_velocities_need_sqrt_a_to_match_the_catalogs_relative_velocity` |
| Size of Σ m (r × v) / Σ m, × a^1.5 | `SubhaloSpin` | under 10⁻⁵ | `test_the_catalogs_spin_is_per_unit_mass_with_a_physical_length` |
| Direction of the same | `SubhaloSpin` | 0.0001° | the same test |
| Radius around `SubhaloPos` holding half the mass | `SubhaloHalfmassRad` | 6 × 10⁻⁵ | `test_the_half_mass_radius_around_the_most_bound_particle_matches_the_catalog` |

What this settles:
- **`SubhaloSpin` is the angular momentum per unit mass, with a physical length and a physical
  velocity:** Σ m (a r) × (√a v) / Σ m, where r and v are stored particle positions and velocities
  relative to the galaxy's center of mass and mean velocity. The a^1.5 is a (for the length) times
  √a (for the velocity).
- **The other readings fail clearly.** With a comoving length the size is off by 16 to 27%; as a
  total (not divided by the mass) it is off by a factor of the galaxy's mass, which is different for
  every galaxy.
- **The spin doesn't depend on the reference point.** Once the mean velocity is subtracted, the
  angular momentum about the nucleus and about the center of mass are the same.
- **A spin only turns with a change of frame.** Moving the origin or adding a velocity to everything
  leaves it unchanged; a rotation turns it like any vector.

## A galaxy's velocity
The catalog's `SubhaloVel` is the mass-weighted mean over every particle bound to the galaxy, which is
mostly its dark matter halo. During an encounter the halo and the nucleus stop moving together: the
halos overlap, exchange energy and are stripped, while the dense nuclei keep orbiting each other. SPAM's
two bodies stand for the nuclei, so the velocity it needs is the nucleus's. Two measures of it:

- **The stars within 1 kpc of the nucleus** (`tng.core_velocity`). A single star won't do: the most
  bound particle is a star moving at a few hundred km/s inside the core (on the six test galaxies its
  velocity is 110 to 330 km/s away from the core's mean). Averaging 100,000 or more stars removes that.
- **The galaxy's central black hole.** Each galaxy in the 61 target archives has exactly one, within
  0.1 kpc of `SubhaloPos`.

On the three test targets (`test_the_nuclei_move_faster_than_the_catalogs_halo_velocities_say`,
`test_each_galaxys_black_hole_sits_on_its_nucleus_and_moves_with_its_core_stars`):

| Relative speed of the pair (km/s, Hubble flow included) | `62000000190717` | `75000000121160` | `75000000317209` |
|---|---|---|---|
| Catalog `SubhaloVel` | 101 | 65 | 53 |
| Stars within 1 kpc of each nucleus | 145 | 107 | 87 |
| The two black holes | 142 | 111 | 90 |

- The nuclei move 40 to 65% faster relative to each other than the halo averages say, and the plane
  of the orbit differs by 9°, 27° and 33°.
- The two nucleus measures agree to 5 to 11 km/s, and their orbit planes to 1° to 7.5°.
- The star measure is steady: between 0.5 and 2 kpc the result changes by 3 to 7 km/s.

**Why the black hole follows the core: TNG moves it there.** A real supermassive black hole is held at
a galaxy's center by the drag of the stars around it. In the simulation a star or dark matter particle
is itself 10⁵ M☉ or more, so that drag is "not captured accurately", and close passes with single
particles would knock a black hole out of its galaxy. TNG therefore uses "an ad hoc centring
prescription designed to keep black holes very close to the potential minimum": at every global
timestep the black hole is moved to the potential minimum of its surroundings (a region holding the
equivalent of 1000 resolution elements), "and its velocity is set to the mean mass-weighted velocity
of the region" (Weinberger et al. 2017, section 2.2, [arXiv:1607.03486](https://arxiv.org/abs/1607.03486)).
So a black hole's velocity is itself a local average, and its agreement with the stars is expected,
not an independent confirmation.

## The Hubble flow
TNG's velocities are peculiar: the expansion of the universe is taken out. But the rate at which a
physical distance between two galaxies really changes includes it:

    v = v_peculiar + H(a) × r

where r is the physical separation and H(a) = H0 √(Ωm / a³ + 1 − Ωm) is the expansion rate at that
snapshot (`tng.hubble_rate`; 0.0677 km/s per kpc today, 0.096 at a = 0.62). SPAM works in physical
coordinates, so this is the velocity it needs, and `tng.relative_to` adds the H × r part unless told
not to.

- It is small for close pairs: 3 km/s for nuclei 35 kpc apart, 9 km/s at 111 kpc.
- It points along the separation, so it changes the speed and never the plane of the orbit.

## The frame SPAM needs
SPAM puts the primary at the origin and at rest ([spam_physics.md](spam_physics.md#what-spam-is)), and
any rotation of the whole system is allowed: the physics doesn't change. This project fixes the
rotation by the orbit itself (`tng.orbit_rotation`), at the target snapshot:

- **x** points from the primary's nucleus to the secondary's.
- **z** is along the orbit's angular momentum, relative position × relative velocity, so the pair
  turns counter-clockwise seen from +z.
- **y** completes them.

The secondary is then at (distance, 0, 0) and its velocity has no z part.

- **This is the plane the pair moves in now, and SPAM's own orbit lies in it**, since the forces
  on SPAM's two centers lie in that plane.
- **TNG's earlier orbit need not lie in it.** Real halos are extended and lumpy, so the plane turns
  over time. Drawn in this frame, the track of `62000000190717` over the last 1.5 Gyr stays within
  20% of its distance from the plane; those of `75000000121160` and `75000000317209` leave it by half
  their distance.
- **A spin only turns:** positions, velocities and spins are all rotated by the same matrix, and
  nothing else is done to a spin.
- A pair moving exactly along the line between the nuclei has no orbit plane.

## SPAM's side
SPAM works with G = 1 in its own units ([spam_physics.md](spam_physics.md#units)). `tng.spam_units`
converts physical values into them:

| Quantity | SPAM's unit | Source |
|---|---|---|
| Length | 15 kpc | `parameters_module.f90` in the JSPAM fork |
| Mass | 10¹¹ M☉ | `parameters_module.f90` |
| Velocity | √(G × mass ÷ length) = 169.33 km/s | Follows from G = 1 |
| Time | length ÷ velocity = 86.62 Myr | Follows from G = 1 |

The velocity and time are computed with G = 4.30091 × 10⁻⁶ kpc (km/s)² / M☉ and agree with the
constants in SPAM's source to 3 × 10⁻⁵ (`test_spams_units_are_15_kpc_and_1e11_msun_with_g_equal_to_one`).

## Known limits of this project's TNG files
- **Star particles include wind.** TNG stores wind-phase gas cells with the stars; the documentation:
  "The only differentiation between a real star (>0) and a wind phase gas cell (<=0) is the sign" of
  `GFM_StellarFormationTime`. The target archives don't carry that field, so wind can't be removed.
- **A black hole's mass is its `BH_Mass`**, the "Actual mass of the BH; does not include gas
  reservoir", not the particle's full mass. It is a tiny part of a galaxy's total.
- **One snapshot of particles per target.** The archives hold each target's particles at its own
  snapshot only. Earlier snapshots are known only through the catalog values in the target table.
