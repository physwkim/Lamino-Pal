# Aligning a mount group with Lamino-Pal

A laminography reconstruction needs two numbers the data does not carry: the
rotation center (the detector column of the axis, `center_px`, full-resolution
columns) and the tilt (`tilt_deg`). A group's scans share both, and a sample
band `z_band` (the slices that hold the sample). This is tomoxide's
`docs/LAMINOGRAPHY_ALIGNMENT.md` as Lamino-Pal ops, in the order to run them.
Its numbers come from one instrument and one sample class: the procedure
transfers, the values do not.

Every decision goes through `group_decide` with evidence: a file under the
work directory the decision was read from. Each image a tool returns is also
saved under `artifacts/figures/`, and its caption names the file; cite that.
Pass `method` for how the value was read: rings, sweep, eye or focus.
Decisions made here are recorded as the agent's, for a person to confirm.

## 0. The scan

- `raw_range_check(tag)`: a full turn, a half turn or neither. A half turn's
  rings are open arcs and give no center (the 261003 half turns read 462–467
  against 504 at bin 2).
- `raw_drift(tag)`: how far the sample moved over one turn.

## 1. The center from the rings, before reconstructing anything

`align_rings(tag)`: the mean −log projection over a full turn. Every point
traces a ring centred on the axis, so the mean is a bullseye whose centre
column is the center.

- Look at the image first. Closed concentric rings: the scan is aligned and
  the bullseye column is the center. Arcs that never close: the sample was
  misaligned at acquisition, and no center or tilt recovers it. Say so and
  stop; do not search.
- The number is the mirror correlation of the mean projection with its flip,
  trustworthy at prominence 8 or more. It drifts toward incomplete arcs (281
  at prominence 2.4 against a true 138 on a misaligned scan, where the eye
  read 138; 397 at prominence 21 against 396 on an aligned one). Eye first,
  correlation second: a disagreement between them is the misalignment flag,
  not two numbers to average.
- `bands` gives the center each row band reads on its own; a large
  `band_spread_px` means the rows do not agree.

## 2. Coarse sweep

`align_center_sweep(tag, center_px, half_width_px, step_px, tilt_deg, slice)`
reconstructs one slice at each center from the full projection set and scores
its focus (mean |∇|² inside the field of view), at the slice inside the
group's z band.

- `resolved`: one peak. `railed`: the peak sits at the edge of the grid: the
  search is broken, widen the range or move it; never report the edge.
  `ambiguous`: rivals close behind: compare them by eye (step 3). `flat`: the
  slice holds no sample: fix the z band.
- A monotone curve with its maximum at a grid corner is the symptom of a
  broken search, not an optimum.
- Without a z band yet, reconstruct once at the prior (`recon_preview`) and
  read the band off it (`volume_focus_by_z` on a volume, or the slices that
  show round particles).

## 3. Confirm by eye, without the metric

`align_center_compare(tag, centers_px)` reconstructs each candidate fully and
keeps its best slice in the band, each on its own p0.5–p99.5 window (a window
from one volume applied to another saturates it into a fake-crisp binary
look). At the right center particles are round; a few tens of pixels off they
smear into elongated dashes in a swirl. Decide `center_px` on this, citing
the figure.

## 4. Tilt, then a small refine only

- `align_tilt_scan(tag, tilts_deg)` at the decided center keeps each tilt's
  best slice in the band carried to that tilt and its xz plane. Decide the
  tilt by eye on both: on the 261003 sample the focus value grows with tilt
  and does not pick it, and the xz planes agreed with the slices on 20°.
- Refine (center, tilt) only on a small grid around the prior. A sweep refines
  a prior; it does not find one: widening to ±40 px grew three lobes whose
  highest was the wrong one, while ±8 px around the rings' center returned
  395.75 against a known 396. Keep each grid tight enough to hold one lobe.

## 5. Band and checks

- `volume_focus_by_z(volume)` on a reconstruction at the decided geometry
  suggests the band: the run of slices whose focus is at least half its peak.
  Decide `z_band` with the bin, tilt and depth (slices) of the volume it was
  read from; an op at another tilt carries it there.
- `recon_split(tag)`: what both halves of the projections show is the sample;
  what one shows is noise or artifact.
- An angular-range check is meaningful only at an established center; a
  range curve at a wrong center means nothing.

## 6. Methods that failed validation: do not retry

Each was tried on a scan with a known center (396) and did not reproduce it.

| Method | Result | Why it fails |
|---|---|---|
| 0°/180° mirror registration | scattered 395…607 | The tilted axis has a component along the beam, so 180° is not a mirror of the object. |
| 1-D column-profile symmetry of the mean projection | 847 | The sample is wider than the field of view; truncation breaks the 1-D profile's symmetry while the 2-D rings survive it. |
| Sub-sampled (1/3) projections with a fixed z band | monotone surface, argmax at a grid corner | Streaks dominate the focus on a subsample, and a band fixed in z misses the sample when the tilt moves it. |
| A wide (center, tilt) grid to find the answer | three lobes, the wrong one highest | Both responses are broad; a wide grid finds a wrong answer, not a faster one. |
