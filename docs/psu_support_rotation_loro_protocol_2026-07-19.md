# PSU support rotation LORO protocol

## Question

For the same PSU high-speed-flow run, does a zero-start B0 CGLS field fitted on
two complete rotation blocks retain observable consistency on the third block?
How do the answer and the CGLS stopping trajectory change between 16 cubed and
32 cubed discretizations?

This is a same-flow observable-transfer diagnostic. It is not volumetric field
validation and it is not cross-flow generalization.

## Frozen split

| Fold | Reconstruction rotations | Entire held-out rotation | Held-out views |
|---|---|---|---|
| A | 50 and 90 degrees | 0 degrees | 0, 1, 2 |
| B | 0 and 90 degrees | 50 degrees | 3, 4, 5 |
| C | 0 and 50 degrees | 90 degrees | 6, 7, 8 |

The three cameras inside a rotation are correlated measurements from one
rotation block, not three independent experimental replicates. A complete
rotation is excluded from the CGLS operator, observation tensor, and chunk
traversal. Its observation tensor and subset operator are not constructed until
all train-only CGLS snapshots have been fixed.

## Frozen trajectory

- Grids: 16 cubed and 32 cubed.
- Start: exactly zero field with the same one-node zero outer boundary gauge.
- Solver: one CGLS pass to iteration 12 for each fold and grid.
- Snapshots: k = 0, 1, 2, 3, 4, 6, 8, 12.
- k = 0 means the zero field before any CGLS update. The initial adjoint needed
  to start CGLS is still counted.
- k = 4 is the historical reference checkpoint. Held-out scores are not used
  to select a deployable stopping rule in this run.

## Frozen measurements

Every snapshot reports train and held-out vector relative L2, camera-macro
relative L2, worst-camera relative L2, and the p95 per-ray vector error divided
by the corresponding group signal RMS. Train and held-out normal residuals are
recomputed with their own fresh subset adjoints rather than copied from the CGLS
recurrence. Every call ledger records the physical view IDs traversed. Forward
and adjoint calls, wall time, and maximum resident memory are part of the result.

The p95 is a tail diagnostic. It is not a confidence interval because rays are
spatially dependent and cameras inside a rotation are not independent repeats.

## Multiresolution decomposition

Let U map the 14 cubed active coarse degrees of freedom to the 30 cubed active
fine degrees of freedom by zero-boundary, align-corners trilinear prolongation.
For every matched checkpoint, the fine field is projected onto range(U):

```text
P x32 = U (U^T U)^(-1) U^T x32
x32 - U x16 = (P x32 - U x16) + (I - P) x32
```

The solve uses separable Cholesky factors and never materializes the full 3-D U.
The metric is Euclidean on active nodes; uniform voxel-volume weighting is a
constant scalar and therefore produces the same projector. A consistent Q1
mass matrix is not silently substituted.

The second component is only the orthogonal complement of the frozen 32 cubed
reconstruction relative to range(U). It must not be called high-frequency
truth, null-space truth, field error, or recovered physical detail.

Image-space squared-error change from Ux16 to x32 is attributed exactly between
the coarse-range disagreement and orthogonal-complement components with the
two-player Shapley identity. Negative contribution means reduced squared
reprojection error. This attribution is descriptive, not causal.

## Claim firewall

Completion authorizes only a statement about observable transfer to an unseen
rotation block of the same flow run. The three folds still describe one physical
field, not three statistically independent field replicates. The fixed k=4
resolution screen uses equal-camera macro error as its one primary anchor and
requires no harm in pooled, every-camera, worst-camera, and p95 diagnostics;
all other checkpoint cells are descriptive. It does not authorize field relative L2,
unique three-dimensional truth, cross-flow or cross-session generalization,
neural-operator superiority, practical significance, or a paper-level success
claim. Direction thresholds are numerical tie breakers, not repeatability or
physical-significance floors.
