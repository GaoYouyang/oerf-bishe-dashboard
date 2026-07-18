# N2 next hypothesis: topology-certified randomized fidelity routing

## Status

This is a research hypothesis, not an implemented or validated algorithm.

## Prior-art boundary

The following ingredients are already established and cannot be the novelty:

- two-level residual estimation and optimal cost allocation:
  [MLMC](https://people.maths.ox.ac.uk/gilesm/files/OPRE_2008.pdf),
  [MFMC](https://doi.org/10.1137/15M1046472), and
  [approximate control variates](https://arxiv.org/abs/1811.04988);
- neural control variates:
  [Neural Control Variates](https://arxiv.org/abs/2006.01524);
- automatic/discrete/hybrid refractive-index gradients, 3D mask and occupancy
  sampling:
  [2026 Neural Refractive Index Primitives](https://arxiv.org/html/2605.11454);
- field-dependent nonlinear ray tracing and its adjoint:
  [Adjoint Nonlinear Ray Tracing](https://imaging.cs.cmu.edu/adjoint_nonlinear_tracing/);
- finite-aperture cone-ray BOS:
  [Depth-of-Field Effects in BOS](https://arxiv.org/html/2402.15954).

No claim of first use is justified by the current literature audit.

## Narrow testable contribution

Let `L(z,theta)` be a straight/sparse/automatic-gradient route and
`H(z,theta)` be the field-dependent curved cone-ray route for the same optical
state `z`. A frozen safety certificate produces a high-fidelity probability
`pi(z)` with `pi_min > 0`. Draw `I ~ Bernoulli(pi)` and use

\[
\widehat H
=L+\frac{I}{\pi}(H-L).
\]

Conditioned on the routing features and a fixed nonzero `pi`, the correction is
unbiased for `H`. Topologically unsafe rays use `pi=1`, so the high route is
always evaluated. Safe rays can use smaller `pi` according to a preregistered
residual-risk model.

The potentially useful unit is not the formula alone. It is a BOST-specific
certificate that jointly tests:

- support/mask/occupancy active-set stability;
- AABB entry/exit face and event-order stability;
- flame-front crossing count and grazing events;
- background-plane and calibrated-frustum validity;
- curved-ray local truncation error and momentum invariant;
- low/high forward residual and full/frozen JVP discrepancy;
- forward, JVP and VJP cost under the same random state.

## Mathematical traps

1. If `pi` depends on trainable parameters, blindly differentiating through the
   discrete draw is biased. Freeze the routing decision during a gradient step,
   or derive and include the required score-function/pathwise terms.
2. `pi` may never be zero on a supposedly safe region unless equality `H=L` is
   proved there. Otherwise the residual correction loses support.
3. A topology change invalidates a smooth finite-difference/VJP comparison; it
   must be reported as an event, not averaged away.
4. Forward unbiasedness does not imply an unbiased squared loss. Training still
   needs independent estimator replicas or an explicit covariance correction.
5. A learned certificate must be calibrated on disjoint data and evaluated with
   per-rig tail bounds; mean compute savings alone are insufficient.

## Minimum comparison set

- high-only curved cone-ray;
- low-only straight ray;
- static N2 two-level allocation;
- optimized MFMC/ACV coefficient and allocation;
- randomized topology-certified routing;
- an oracle route using the true residual, reported only as an upper bound;
- the host reconstruction's NeRIF/NeDF/2026-primitive baseline.

Metrics must include field relative-L2/H1, front or support geometry, held-out
reprojection, per-rig Q95 error, simultaneous intervals, field queries,
coordinate VJPs, full forward/JVP/VJP wall time and calibration overhead.

## Immediate falsification question

Does the laboratory contain a regime where the curved-path contribution is
larger than optical-flow and calibration uncertainty but smaller than the point
where rays leave the calibrated frustum or change event topology?

If the answer is no, this line should be stopped even if the synthetic
estimator has excellent variance reduction.

