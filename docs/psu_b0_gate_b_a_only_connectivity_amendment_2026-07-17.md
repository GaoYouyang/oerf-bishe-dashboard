# PSU-B0 Gate B A-only connectivity amendment

## v3 invalid result

The v3 formal runner passed the same-run singleton graph control and stopped
during the first production A-only setup because the frozen runtime contract
incorrectly required all 2744 support voxels to be data-coupled.

- Parent v3 commit: `81d9513eadb1f8ee6d3c0a33f9a913b9f58bd863`
- Parent v3 config SHA-256: `aedec9db6c8a81229cbe4c51f9c42538485f853cd24003564ba0e130d57239dd`
- Exception: `ValueError: Gate-B data-coupled voxel count changed`.
- Factor trajectories completed: `0`.
- Factor solver calls: `0`.
- Factor truth metrics observed: `0`.
- Gate B result directory created: `false`.

v3 is `INVALID_NO_FACTOR_RANKING`.

## Setup-only evidence

The tracked 16-sample diagnostic performs only the two declared absolute-factor
setup calls per sample. It finds one identical A-only coordinate mask for every
replicate and morphology:

- support-active coordinates: `2744`;
- directly A-coupled coordinates: `2322`;
- support coordinates in the exact A-null set: `422`;
- active detector-component rows: `4608`;
- signed factor solver calls: `0`;
- truth scoring: `false`.

The diagnostic report SHA-256 is
`d3418148c4855419fdba15c2e0515b1dbbe4d81223352227fb9bee1b01582f19`.

The earlier 2744 value came from the A+D setup, where TV connectivity retains
all support coordinates. It cannot be reused for a true A-only experiment.

## v4 rule before factor trajectories

v4 changes only the frozen `data_coupled_voxel_count` from 2744 to 2322 and
requires the shared active-index digest recorded by the setup-only diagnostic.
Scalar, view-block, and voxel-factor PDHG all use that same reduced gauge. The
422 A-null support values are embedded as zero before physical scoring.

The graph-PCGLS comparator still uses its frozen full-support Sobolev search
direction and may smooth into the A-null support. This is disclosed as a
different classical prior and makes the graph-gap requirement conservative; it
does not authorize an algorithm-superiority claim.

No sample, checkpoint, factor recurrence, operator-call budget, timing order,
factor decision threshold, aggregation formula, or claim boundary changes.
No factor reconstruction metric has been seen when v4 is written.
