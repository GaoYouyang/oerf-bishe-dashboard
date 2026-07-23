# v5c post-open no-go diagnosis

Status: `V5C_POSTOPEN_NO_GO_DIAGNOSIS_NOT_A_NEW_LOCK`

This directory is a read-only diagnosis of the committed first-open result. It
does not alter the frozen protocol or create a new lock.

## What failed

- joint radius-kappa CV recovered the nearest bank radius in
  `3 / 6` blocks;
- fixed-ridge profiling recovered `5 / 6`;
- operator, clean-truth and noisy-truth oracles remained
  `6 / 6`,
  `6 / 6`, and
  `6 / 6`;
- all `6 / 6`
  blocks selected the largest frozen kappa, and all best-CV curves decreased
  monotonically over the complete grid;
- mean true camera-deletion radius stability was
  `0.375`; no block was
  stable under all four deletions;
- final coverage was `0 / 48`.

The forward model still ranks radius correctly when truth is available. The
failure is therefore localized to the current joint predictive-selection rule,
not to the finite-aperture bank semantics.

## Why zero coverage is not success

Only `2` of six blocks changed the
metadata operator. The other `32` rows were
explicit fallback. Among the 16 rows where the candidate actually changed:

- mean raw field gain was `1.325%`;
- `7 / 16` lost more than 1% field accuracy;
- mean raw audit change was `-0.950%` (positive is worse);
- `8 / 16` worsened the audit camera.

The strict gate correctly rejected these candidates, but a method that always
falls back has no demonstrated utility.

## Next algorithm test

Do not enlarge the network yet. On the opened development data, compare GCV,
UPRE/effective degrees of freedom, Morozov discrepancy, equal-complexity radius
scores, and genuinely nested inner-camera CV. Freeze the winning rule before
opening new `helical_plume` and `stratified_ignition` families, a new seed,
and a new rig/session draw.

None of these numbers supports real BOS, OERF, arbitrary-view safety, or
DeepONet/FNO/FFNO superiority.
