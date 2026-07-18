# N2-CVCR-N0 post-open reference sensitivity

Status: `POSTOPEN_REFERENCE_SENSITIVITY_UNRESOLVED_AT_4096`

The original preregistered decision remains
`HOLD_REFERENCE_QUADRATURE_NOT_CONVERGED`. This package only evaluates the
fixed deterministic pupil-reference ladder at 1024, 1600, 2304, and 4096
points. It does not re-score the candidate or authorize a learner, a 3D
reconstruction claim, an experimental/OERF claim, or a generalization claim.

Read `summary.json` for the machine status, `reference_ladder.csv` for all
rig/order rows, and `checksums.sha256` for integrity verification.

```bash
shasum -a 256 -c checksums.sha256
```

The figure is a visualization of those frozen files and is not an additional
source of numerical evidence.
