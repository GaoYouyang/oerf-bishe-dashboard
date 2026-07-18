# N2-ADRC-N1 development protocol

## Status

This is a development-only mechanism screen. It cannot authorize a paper,
experimental, reconstruction, or generalization claim. Its hard conclusion is
`DEVELOPMENT_ONLY_NO_AUDIT_AUTHORIZATION` regardless of the measured gain.

## Question

Can a two-level residual estimator reduce work-normalized variance for a narrow
differentiable-BOST cost conflict?

- Low fidelity: the coordinate gradient of one smoothstep grid primitive on a
  prescribed straight path.
- High fidelity: a six-query central-difference gradient on a prescribed path
  that may contain a fixed bend.
- Correction: `mean(low_B) + mean(high_D - low_D)`.

The mechanism is classical two-level estimation specialized to this surrogate.
It is not yet a novel algorithm. The fixed bend does not depend on the field,
so this pilot does not test physical curved-ray trajectory sensitivity.

## Declared target

For each development case, the target is exactly the mean of the high renderer
over a frozen finite scrambled-Sobol reference population. It is not a
continuous pupil/path integral and is not a real BOS observation.

The analytic reaction-flow morphology is sampled once onto a `17 x 17 x 17`
grid. The renderer then sees only that grid. The continuous analytic field is
not used as an oracle for the reported target.

## Randomness roles

- `T`: an independent calibration Sobol population estimates low, high, and
  residual variance and times the three evaluation routes.
- `B`: low-only indices sampled with replacement from the frozen reference
  population.
- `D`: paired high/low residual indices sampled with replacement independently
  from `B`. Independent draws may coincidentally select the same population
  item; this is not pool reuse.
- `H`: a separate high-only comparator index stream.
- Two complete independent two-level replicas are used for the double-sample
  nonlinear-loss gradient check.

Common random numbers are used only inside each paired `high - low` residual.
The resampling replicate, not a pixel or path point, is the uncertainty unit.
Because the development estimators sample with replacement, no finite-population
correction is applied.

## Cost reporting

The exact primitive ledger is reported separately:

| Route | Field queries | Coordinate VJP | Geometry |
| --- | ---: | ---: | --- |
| low automatic | 1 | 1 | prescribed straight |
| high discrete | 6 | 0 | prescribed high path |
| paired residual | 7 | 1 | both paths |

The scalar allocation uses warm CPU `float64` median wall time per sample at the
frozen calibration batch size. This is a machine-specific development proxy,
not a hardware-independent complexity result. Median and p90 timings remain in
the report. A conservative timing screen additionally compares the high route's
p10 cost against the low and paired routes' p90 costs; this deliberately favors
the high-only comparator.

## Derivative checks

One frozen grid-space direction is reused for forward, finite-difference JVP,
autograd JVP, and VJP-dot checks. The same `B` and `D` sample states are reused
through each fixed-state check. Their counts, seed role, and index digest are
independent of the timing-derived matched-cost allocation.

An unbiased forward estimator does not make a squared loss or its gradient
unbiased. The report therefore contrasts:

1. the plug-in gradient from one stochastic forward estimate; and
2. the symmetric double-sample gradient from two independent complete
   estimator replicas.

## Development screens

The configuration contains promotion screens for measured-cost efficiency,
the nested half-to-full finite-reference sensitivity, and fixed-state
derivatives. Passing them only permits design of an unseen audit.
The reserved analytic families remain unopened in this run. Any later audit
must additionally freeze reference convergence, interface validity, per-rig
tail gates, simultaneous uncertainty intervals, and full forward/JVP/VJP
costs before opening held-out cases.

## Fail-closed boundaries

This pilot must not be cited as evidence for:

- NeRIF, NeDF, or the 2026 neural refractive-index-primitive implementation;
- field-dependent ray bending or its trajectory derivative;
- masks, frusta, occlusions, flame-front support changes, or camera calibration;
- a full reconstruction, operator-learning model, real experiment, or
  out-of-distribution generalization;
- novelty over MLMC, multi-fidelity Monte Carlo, regression control variates,
  or neural control variates.
