# Temporal q_cal fresh audit: dormant contract

Status: `FROZEN_BUT_NOT_AUTHORIZED_TO_RUN`

This contract prevents post-result method switching. It does not authorise a
fresh run. Activation requires both:

1. v4 shows that the registered exact-score Godambe arm is informative without
   relying on unbounded confidence regions; and
2. He Yuanzhe confirms the laboratory target parameter, callable/JVP/VJP,
   acquisition independence, covariance unit, geometry manifest, and primary
   physical endpoint.

If either condition is absent, the next action is interface/data clarification,
not another same-family synthetic sweep.

## Frozen primary method

The primary remains `iterative_full_profile + exact-score Godambe`. It cannot be
replaced by the best post-open Wald, contrast, cross-fit, neural, or oracle arm.
Secondary methods are the frozen v3 plug-in, Gauss--Newton sandwich, penalised
profile contrast, and corrected two-fold cross-fit ablation.

## New synthetic rig manifest

Use 12 rigs, four per existing phantom family, with base seeds:

```text
1201, 1303, 1427, 1549,
2203, 2339, 2473, 2591,
3209, 3331, 3463, 3581
```

The seed namespace must be new and committed before any output is read. Camera,
field, transport, anchor-noise, evaluation-noise, and secondary-noise seeds must
be derived from disjoint named branches. Seeds `709`, `811`, `907`, `1709`, and
all v2--v4 noise namespaces are forbidden.

These 12 rigs remain the same synthetic generator family. Passing them would not
establish experimental or out-of-family generalisation.

## Primary cells

For every rig:

- null `q=0`;
- signed coordinate axes `+e1...+e5` and `-e1...-e5` at `||q||=q_ref=0.04`;
- signed anchor-weak directions `+v_min` and `-v_min`.

`v_min` is built only from an independent `q=0` anchor observation:

1. fit the anchor field with the frozen ridge;
2. build its projected geometry design;
3. take the smallest generalized direction under frozen parameter scaling;
4. normalise to unit Euclidean norm and fix sign by making the largest-magnitude
   component positive;
5. hash the direction before evaluation observations are generated.

Truth-field weak directions may be reported as oracle stress controls, but they
cannot enter the primary gate or choose any method setting.

Primary noise is `1/128` of the frozen base sigma. Use 1,000 independent
evaluation replicates per cell. Null calibration uses a disjoint 1,000-replicate
flow-off/anchor namespace per rig. No observation is shared across cells.

## Primary gates

- Global null-test size: pooled one-sided 95% CP upper bound at most 0.06; each
  rig empirical size at most 0.08.
- Each of the 12 direction classes: pooled one-sided 95% coverage lower bound at
  least 0.925.
- Every rig-direction cell: empirical coverage at least 0.90.
- Positive/negative coverage difference per axis/direction at most 0.04.
- Pooled q relative-L2 median at most 0.25 and p90 at most 0.75; every direction
  class p90 at most 1.0.
- Calibrated maximum ellipsoid-axis radius: median at most `0.5 q_ref`, p90 at
  most `q_ref`.
- Numerical failures, boundary hits, non-PSD covariance, all-space intervals,
  and abstentions remain in denominators and cannot pass by blanket rejection.
- Relative to full-data variable projection, field and six-frame sequence
  median error may not worsen by more than 2%, and p90 may not worsen by more
  than 5%.

## Secondary stress cells

Only after the primary report is immutable:

- amplitude `0.5 q_ref`;
- noise `1/64`;
- truth-field weakest direction as oracle-only stress;
- correlated camera/frame noise generated from a covariance model frozen from
  independent repeats, never from the evaluation residuals.

Secondary results cannot rescue a failed primary gate.

## Cost ledger

Record per method and per observation:

- `A`, `A^T`, JVP, VJP calls;
- nuisance RHS solves and factorizations;
- profile evaluations and backtracks;
- calibration/anchor/cross-fit extra fits;
- wall time, CPU time, and peak RAM;
- estimator-only versus evaluator-only work.

Point-estimator superiority requires equal physical call budgets. A more
expensive interval method may instead be shown on a cost--coverage Pareto plot.
The current dense `512 x 512` Cholesky reference cannot support a matrix-free
speed claim.

## Highest possible label

Even a complete pass authorises only:

```text
FRESH_SAME_FAMILY_SYNTHETIC_CALIBRATION_SIGNAL
```

It does not authorise `new algorithm`, `real BOST`, `3-D/4-D reconstruction`,
`cross-rig experimental generalisation`, `better than NeRIF/TDBOST`,
`publication ready`, or `breakthrough`.
