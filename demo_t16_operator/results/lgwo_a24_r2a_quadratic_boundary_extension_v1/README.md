# LGWO-A24 R2-A quadratic regularization scale pilot

Status: **VALID_POSTOPEN_VALIDATION_ONLY_QUADRATIC_SCALE_SIGNAL**

This checksum-bound package compares one unregularized CGLS path with
`26` fixed quadratic L2, H1, and mixed
Sobolev configurations on the already-opened validation split. Every path uses
exactly `24F/24A^T`; six checkpoints are scored.

- validation cases: `24`
- configurations: `27`
- scored rows: `3888`
- pilot-signal points: `55`
- neural model authorized: `false`
- real BOST evidence: `false`

A pilot signal only narrows a later frozen formal screen. It is not algorithm
superiority, a new method, unseen-rig generalization, or a paper result.

Files:

- `rows.csv`: every case/configuration/checkpoint metric and call ledger
- `aggregate.csv`: validation means, CGLS-relative gains, Pareto flags, safeguards
- `summary.json`: status, strongest descriptive points, cost, provenance, boundaries
- `diagnostic.png`: aggregate four-panel pilot visualization
- `checksums.sha256`: integrity manifest
