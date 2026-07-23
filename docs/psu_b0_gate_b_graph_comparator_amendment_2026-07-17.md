# PSU-B0 Gate B graph-comparator amendment

## v2 invalid result

The v2 formal runner completed all 64 graph-PCGLS replay rows in memory and
then failed its binding parent-replay check before constructing any factor
setup or running any scalar, view-block, or voxel-factor trajectory.

- Parent v2 commit: `eea30a7dc231e1e3050983f7ccacd30b716d420f`
- Parent v2 config SHA-256: `3b0623f9644cb8ea2bd4a35e6abfbc4b2f646757f706ee6d5247b20b94875719`
- Exception: `ValueError: single-sample graph replay differs from the frozen batch-eight release`.
- Gate B result directory created: `false`.
- Factor setup count: `0`.
- Factor solver calls: `0`.
- Factor metric rows observed: `0`.

v2 is therefore `INVALID_NO_FACTOR_RANKING`. Its failed compatibility gate is
not silently relaxed inside the same protocol.

## Graph-only diagnosis

The tracked three-repeat diagnostic is strictly graph-only. It shows:

- old batch and new singleton K=32 mean field error remain close;
- per-row MPS differences are larger than the v2 `5e-5` replay ceiling;
- CPU float64 tests confirm that PCGLS reductions and updates are sample-wise,
  so there is no intended cross-sample mathematical coupling;
- MPS batch-shape finite-precision sensitivity is the supported explanation,
  but remains an inference rather than a proof.

The graph diagnostic report SHA-256 is
`5499333734aca7e8327546a3db354ec84b930f057862e89b5563df89be02fda4`.

## v3 rule before factor results

v3 starts a new post-diagnostic development protocol. The old batch rows are
retained only as a traceability diagnostic. They are not a binding numerical
oracle for a singleton MPS execution. The scored comparator is one exact-K,
single-sample graph-PCGLS trajectory executed in the same formal run and on the
same device as the three A-only methods.

The factor decision thresholds, sample set, checkpoints, solver parameters,
timing order, exact operator-call budgets, objective conversion, aggregation
formulas, and claim boundary remain unchanged. No factor result has been seen
when this rule is written. Passing v3 can still establish only an opened,
truth-scaled E2 conditioning signal; it cannot establish fresh, real-flow, or
algorithm-superiority evidence.
