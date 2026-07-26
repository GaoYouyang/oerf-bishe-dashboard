# PoolFire C geometry-equalized BP v7 implementation failure

## Result

The first and only sealed v7 execution failed closed before the 606-frame
observation audit began. It produced no final result directory and no scientific
T0 decision.

This is an implementation failure, not evidence that geometry-equalized BP
works or fails.

## Root cause

PoolFire stores all three coordinate axes in descending order. The accepted v6
data bridge reverses each axis and reconstructs a uniformly spaced ascending
coordinate vector before cropping and block coarsening.

The v7 coordinate-only geometry reconstruction checked the raw coordinate mesh
but passed the descending vectors directly to the straight-ray operator. The
operator correctly rejected the first axis because its cell centres were not
strictly ascending.

## Evidence boundary

- No v7 final result directory exists.
- No v7 T0 metrics were released.
- No aligned-containment T1 or SVD stage ran.
- No stopping-validation or untouched-test trajectory was opened.
- Neural training remains unauthorized.
- `algorithm_breakthrough=false`.

## Next valid gate

Version 7.1 may change only the coordinate canonicalization step. It must match
the accepted v6 transformation exactly:

1. Prove each stored axis is finite, uniform, and strictly descending.
2. Reverse the axis.
3. Reconstruct the canonical ascending vector from its endpoints.
4. Require the maximum correction to remain within the accepted coordinate
   tolerance.
5. Crop and block-coarsen only after canonicalization.
6. Reproduce the sealed geometry identity for all six open trajectories.

The equalizer formula, relative floor, trajectory membership, T0 thresholds,
raw control, cost ledger, and failure actions remain unchanged.
