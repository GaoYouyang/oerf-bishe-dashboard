# PSU-B0 Gate B pre-performance infrastructure amendment

## Frozen incident record

- Parent preregistration commit: `7a65cce82cea8799cacf3f6bbffb8e6050ce8440`
- Parent config SHA-256: `7e4564d57e37fc2928534926e80c5df0bc51389db86514fcc3b15a35c59baca9`
- First formal-launch time: `2026-07-17T13:52:14+08:00`
- Failure point: construction of the first single-sample graph whitener, before the first graph-PCGLS or factor-PDHG solver call.
- Exception: `ValueError: source calibration mean shape is inconsistent`.
- Emitted Gate B metric rows: `0`.
- Emitted Gate B timing pairs: `0`.
- Result directory created: `false`.

The production `DetectorCovarianceWhitening` stores `calibration_mean` as
`[view, ray, component] = [9,256,2]`; the preregistered adapter accepted only
the algebraically identical flattened shape `[view,2*ray] = [9,512]`.

## Only allowed change

The adapter may reshape `[view,ray,2]` to `[view,2*ray]` before cloning the
calibration mean. This is a view-preserving reshape with no arithmetic,
permutation, parameter, threshold, sample, solver, metric, timing, or decision
change. A focused regression test must cover both storage layouts.

## Scientific protocol held fixed

The two opened replicates, eight reaction morphologies, exact checkpoints,
three true A-only metrics, exact-K graph-PCGLS control, zero initialization,
operator-call budgets, thresholds, aggregation formulas, timing orders,
objective-unit conversion, claim boundary, and NO-GO/GO labels remain byte-for-
byte identical in the amendment config except for source/test hashes and this
incident provenance.

No performance result may be run until the amendment config, runner,
validator, test manifest, and source hashes are committed in a clean worktree.
