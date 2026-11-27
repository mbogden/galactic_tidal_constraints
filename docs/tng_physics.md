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

**The age of the universe** follows from the same H(a), since dt = da / (a H). For matter and dark
energy it has a closed form (`tng.age_of_universe`), with Λ = 1 − Ωm:

    t(a) = 2 / (3 H0 √Λ) · asinh( √(Λ / Ωm) · a^1.5 )

It gives 13.80 Gyr today. A track's time axis is the difference of two such ages (`tng.track_time`).

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
  over time: [The track leaves the plane](#the-track-leaves-the-plane).
- **A spin only turns:** positions, velocities and spins are all rotated by the same matrix, and
  nothing else is done to a spin.
- A pair moving exactly along the line between the nuclei has no orbit plane.

### The track leaves the plane
Each target's track (the catalog's `SubhaloPos` separations at every snapshot of its encounter) was
turned into the target's own frame, for all 61 targets. How far a row sits out of the x-y plane is
|z| ÷ its distance: 0 in the plane, 1 straight above it.

| Over the rows up to the target | Median | Middle half | Largest |
|---|---|---|---|
| Largest \|z\| ÷ distance, whole past (about 3 Gyr) | 0.25 | 0.12 to 0.37 | 0.85 |
| Largest \|z\| ÷ distance, since the most recent close pass | 0.07 | 0.03 to 0.14 | 0.71 |
| The track's own plane over the last Gyr, degrees from the frame's z | 9° | 5° to 29° | 175° |

- **Since the last close pass the track stays close to the plane** for most targets. Further back it
  does not: one plane does not describe a whole encounter.
- **For 8 of the 61 targets the last Gyr of track goes round the other way** (more than 90° from the
  frame's z). Most of these pairs are moving almost straight toward or away from each other at the
  target snapshot, so the sideways part of the velocity, which alone decides the plane and the sense
  of the orbit, is small. For `73000000145217` it is 8 km/s out of 240, while the two measures of
  the nucleus velocity (stars within 1 kpc, and the black holes) differ by about 10 km/s. For such a
  target the plane, and with it whether a disk is prograde or retrograde, is not well determined
  from one snapshot's velocity.
- An alternative for those targets is to take the plane from the track's positions (the target
  snapshot and the one before). It is not used: the frame is always built from the target snapshot's
  position and velocity.

## A curve through the snapshots
TNG50-1 saves a snapshot about every 0.15 Gyr, and a close pass takes less than that. To have the
pair's path between snapshots, a cubic spline is fitted through the catalog's separations, each axis
against time (`tng.track_spline`). Only positions are used: the catalog's velocities are halo
averages ([A galaxy's velocity](#a-galaxys-velocity)). Two tests say how far to trust the curve.

**Leaving a snapshot out.** On the 29 tracks that hold targets, each snapshot in turn was left out
and the curve fitted through the rest, a gap of two snapshots. How far the curve lands from the
left-out position (`test_a_curve_through_real_tracks_finds_a_left_out_snapshot`):

| Left-out snapshot | Count | Cubic spline | Straight line | Hermite, catalog velocities |
|---|---|---|---|---|
| Away from a close pass | 487 | 1.3 kpc | 3.3 kpc | 1.8 kpc |
| At or next to a close pass | 127 | 10.4 kpc | 18.2 kpc | 9.2 kpc |

(Medians. The gap here is twice the real one, so the real error is smaller.) Forcing the curve
through the catalog's velocities as well (a Hermite spline, the earlier pipeline's choice) is no
better overall than positions alone.

**Against a known answer.** SPAM's own orbit for three targets' models was kept only at TNG's
spacing, fitted, and compared with the full orbit. These models have fast, tight passes (0.3 to 1.3
SPAM lengths, 5 to 20 kpc):

- **The time of a close pass is recovered** to about 0.1 to 0.3 SPAM time units (10 to 25 Myr).
- **The distance of a tight pass is not.** For a true pass at 0.32 SPAM lengths the curve gave 0.09
  to 1.6, depending only on where the snapshots happened to fall. Near the pass the curve is off by
  up to 2 SPAM lengths (30 kpc).
- **Knowing the true velocity at every snapshot does not rescue it**: the same test with a Hermite
  spline through exact velocities gave 0.06 to 1.1.
- Where now is the curve's last point, setting its slope there to the true velocity reduces the error
  in the last interval by about a third.

So the curve is good between passes and for when a pass happened, and not for how close a tight pass
came. Comparisons with SPAM's orbit should not lean on the closest distance.

**On the 61 targets.** Two checks of the sampled orbit (`tng.target_orbit`) against what is known:

- **Its slope at now is not the measured velocity.** Where the track has a later snapshot (50
  targets) the slope and the stars' velocity differ by a median 42 km/s as vectors (11° in direction;
  no lean to fast or slow). Where now is the curve's last point (11 targets) the median is 171 km/s
  and 22°, and three targets just after a pass are over 110° off. The measured velocity is the one
  to use at now; the curve's slope is not a velocity measurement
  (`test_the_curves_slope_at_now_is_not_the_measured_velocity`).
- **The last close pass falls between snapshots.** 49 orbits have a close pass. Only 2 of them sit
  on a snapshot; the median is 0.58 SPAM time units from the nearest one. The pass distance is then
  the curve's own: a median 0.67 of the nearest snapshot's distance, and for some under 0.1, a dip
  no snapshot saw (`test_every_real_targets_last_pass_falls_between_snapshots`).

## Disk angles
SPAM describes each disk's tilt with two angles, φ and θ. It builds the disk flat in the x-y plane,
turning counter-clockwise seen from +z, tilts it by θ about the y axis, then turns it by φ about the
z axis (`setup_module.f90` in the JSPAM fork). The disk's spin axis ends up along

    (sin θ cos φ, sin θ sin φ, cos θ)

So, from a galaxy's spin L measured in the frame SPAM runs in (`tng.disk_angles`):

    θ = arccos(L_z / |L|)        φ = atan2(L_y, L_x)

- θ = 0 is a disk in the orbit's plane turning the same way as the orbit (prograde); θ = 180 is the
  same disk turning against it (retrograde); θ = 90 is edge-on to the orbit.
- **Checked through SPAM itself**, sense of rotation included: for several spin directions, SPAM
  builds a disk from the angles, and the spin measured from its particles points along the direction
  asked for, to within 0.5° (`test_spam_builds_a_disk_whose_spin_points_where_disk_angles_was_told`).
- The spin must already be in SPAM's frame, and it gets there by rotation alone
  ([The frame SPAM needs](#the-frame-spam-needs)).

### Which stars give a disk's spin
A galaxy in an encounter has stars far from its disk: tidal tails, a stellar halo, satellites it
swallowed. They are few, but each one's share of the spin grows with its distance, so they can turn
the total. `tng.measure_target` therefore takes the choice as a setting (`spin_radius_factor`): the
stars the galaxy owned before the encounter that are within a number of stellar half-mass radii of
its nucleus (5 by default, the disk radius), or all of them.

How far the spin's direction moves, in degrees, from the all-stars spin to the spin of the stars
within 2 and within 10 half-mass radii (`test_a_real_disks_spin_depends_on_which_stars_are_used`):

| Target | Galaxy | Within 2 | Within 10 | Stars inside 10 |
|---|---|---|---|---|
| `62000000190717` | primary | 6.7 | 3.7 | 90% |
| `62000000190717` | secondary | 22.5 | 7.3 | 93% |
| `75000000121160` | primary | 19.3 | 5.4 | 99% |
| `75000000121160` | secondary | 30.0 | 0.0 | 100% |
| `75000000317209` | primary | 44.5 | 40.6 | 84% |
| `75000000317209` | secondary | 7.2 | 3.3 | 100% |

- The choice moves a disk's angles by 7° to 45° on these three targets. It is not a detail.
- For the primary of `75000000317209`, the 16% of its stars beyond 10 half-mass radii (56 kpc) turn
  the spin by 41°. The inner disk points one way and the all-stars spin another.
- Neither choice is "the" answer. SPAM's disk is a thin, flat disk out to its radius, so the default
  takes the spin from the stars inside that radius: the ones SPAM's disk stands for.
- **No radius recovers the spin of earlier snapshots.** Over all 61 targets, the spin of today's stars
  is a median 20° from the galaxy's all-stars spin 0.6 Gyr earlier and 28° from it 1.6 Gyr earlier,
  whichever radius is used (the earlier pipeline's `ps_` and `ss_` spins; radii from 0.5 to 10
  half-mass radii differ by 2° to 6°, and all stars is the furthest off by a few degrees).

## Which galaxy is the primary
SPAM's primary is the galaxy at the origin; by convention it is the more massive one. In the target
table, "primary" is only the earlier pipeline's label, and for some targets it is the lighter galaxy.
Deciding by mass needs care, because **the catalog's total mass is not steady during an encounter**.

The two galaxies come to share one dark halo, and SUBFIND gives that halo to whichever galaxy it
calls the central one. The `SubhaloMass` of the other drops, and the label can change from snapshot to
snapshot. Stars sit deep in each galaxy and are not part of the shared halo, so their mass ratio
holds. Mass ratio (table's secondary ÷ primary) along two encounters, every third snapshot up to the
target:

| Target | Catalog `SubhaloMass` | Stars, by the galaxy they belonged to before the encounter |
|---|---|---|
| `62000000190717` | 0.78, 0.76, 0.54, 0.40, 0.09, 0.13, 0.13, 0.07 | 0.80, 0.75, 0.74, 0.73, 0.78, 0.74, 0.75, 0.78 |
| `75000000317209` | 43.4, 1.81, 1.85, 0.72, 0.50, 7.88, 22.1 | 6.47, 1.38, 1.33, 1.27, 1.24, 1.31, 2.35 |

(The star columns are the earlier pipeline's `ps_SubhaloMass` and `ss_SubhaloMass`; at the target
snapshot they equal the sums over the archives' particles.) The first galaxy pair did not go from
0.78 to 0.07 of each other's mass: its halo was relabelled.

Rules based on the catalog's total mass therefore disagree with each other. Five were tried on the 61
targets (mass at the target snapshot, at the first snapshot, each galaxy's own peak, both at the
lighter one's peak, and total mass by historical ownership): they name a different primary for 17 of
the 61.

`tng.primary_side` offers two rules: `"table"` keeps the table's label, and `"stars"` picks the
galaxy with more mass in the stars it owned before the encounter, at the target snapshot. By
`"stars"` the table's secondary is the heavier galaxy for 8 of the 61 targets.

- **Many pairs are close:** for 21 of the 61 the two stellar masses are within a factor of 1.5. For
  those the choice is nearly arbitrary, and it only decides which nucleus is the origin.
- **It is decided per target, at its own snapshot:** for 13 of the 61 the order changes somewhere
  along the earlier track.

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
