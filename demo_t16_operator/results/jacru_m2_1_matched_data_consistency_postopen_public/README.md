# JACRU-M2.1 post-open data-consistency diagnostic

Status: `M2_1_POSTOPEN_DATA_CONSISTENCY_NO_GO`

This export reuses the opened synthetic M2-T0 train/development/exploratory-OOD
splits. It is diagnostic evidence, not confirmatory OOD/fresh evidence. Two
truth-free paths use one logical forward and one adjoint call per added step:
measured pullback minimizes `||Ax-y||^2`; base-nullspace filtering minimizes
`||A(x-x0)||^2`. A finite second path is only a near-null spectral filter, not
an exact null-space projection. Every learned point is compared with
`CGLS-(13+k)`, `Huber-PDHG-(13+k)`, and CGLS-12 plus `(k+1)` base-only
Landweber pairs at the same total logical F/A count. Dense numerical SVD setup
is reported separately and is not counted as reconstruction runtime or calls.

Zero-step reproduction passed: `True`. No checkpoint,
restricted paper, experimental array, or sealed split is exported. Even if a
post-open point meets the declared headroom gate, it only authorizes drafting a
new preregistered test with new data.
