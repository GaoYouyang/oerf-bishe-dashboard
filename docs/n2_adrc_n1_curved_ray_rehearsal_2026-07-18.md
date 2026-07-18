# N2-ADRC-N1 field-dependent curved-ray rehearsal

## Machine decision

`CURVED_RAY_REHEARSAL_ONLY_RESERVED_FAMILIES_UNOPENED`

This is an E0 synthetic development rehearsal. Passing its numerical checks
does not authorize a paper, reconstruction, experiment, or generalization
claim. The two reserved morphology names are not evaluated by this runner.

## Why this increment was necessary

The earlier N2 pilot used a prescribed bend. Its sampling coordinates did not
depend on the refractive-index field, so its VJP could not contain trajectory
sensitivity. The new clean-room kernel integrates the geometric-optics system

\[
\frac{d}{ds}(n\mathbf d)=\nabla n,
\qquad
\frac{d\mathbf r}{ds}=\mathbf d,
\qquad
\frac{d\mathbf d}{ds}
=\frac{(I-\mathbf d\mathbf d^\mathsf T)\nabla n}{n}
\]

with explicit RK4. NeDF reports the same ray equation and uses cubic spatial
interpolation with fourth-order Runge-Kutta for nonlinear synthetic BOS image
generation. This source supports the physical equation, not the novelty of our
implementation: [NeDF primary paper](https://arxiv.org/html/2409.19971).

The implementation separately evaluates:

1. a straight-path automatic-gradient low route;
2. a field-dependent curved-path central-difference high route;
3. the high-route VJP with the trajectory graph intact; and
4. a frozen-path VJP that retains only the direct integrand derivative.

## Frozen scope

- Grid: `17 x 17 x 17`, `float64`, CPU.
- Field: an existing normalized analytic morphology sampled once onto the grid.
- Index map: `n = 1 + s_n q`, with base `s_n = 3e-4`.
- This is not a physical density map. No wavelength, gas composition,
  Gladstone-Dale coefficient, metre-scale ROI, camera calibration, background
  plane, or image formation model is available.
- Development cases: smooth/narrow aperture, wrinkled/wide aperture, and
  smooth/wide aperture.
- Reserved family names remain unevaluated by this runner.
- Because those family definitions and some seeds already exist elsewhere in
  the repository, a later audit can only be fresh-instance/fresh-rig blind; it
  cannot honestly be called family-blind.

## Numerical result

After an initial `16 -> 32` step rehearsal failed reference convergence on all
three cases, the frozen runner was raised to `64 -> 128` steps without relaxing
the `1%` threshold. The final run passed all six numerical checks on `3/3`
development cases:

| Case | 64->128 high relative L2 | Exit vs curvature integral | Momentum balance | Full trajectory JVP FD error | Trajectory JVP fraction at 1x |
| --- | ---: | ---: | ---: | ---: | ---: |
| smooth narrow | 0.318% | 0.0565% | 0.215% | `8.14e-9` | 0.0830% |
| wrinkled wide | 0.261% | 0.0275% | 0.0349% | `4.10e-9` | 0.0225% |
| smooth wide | 0.323% | 0.0520% | 0.196% | `1.06e-8` | 0.0215% |

The VJP dot-product relative errors are below `8.6e-15`; finite-difference
perturbations preserve the support/frustum signature at the base scale.

These checks show that the discrete curved-ray derivative is internally
consistent. They do not show that curved-ray physics matters in the laboratory.

## Dimensionless stress envelope

The same three development cases were evaluated at scale multipliers
`1, 3, 10, 30, 100`. These multipliers are numerical stress levels, not gas or
temperature conditions.

- At `1x`, trajectory sensitivity is only `0.021%-0.083%` of the full
  directional JVP. A real experiment may easily be dominated by optical-flow
  noise and calibration uncertainty instead.
- At `30x`, smooth-narrow and wrinkled-wide first cross a diagnostic boundary:
  trajectory share exceeds `1%` or straight/curved output mismatch exceeds
  `1%`.
- At `100x`, all three cases break; straight/curved mismatch reaches
  `1.87%-4.42%`, trajectory share reaches `3.50%-6.49%`, and all derivative
  subsets leave the frozen synthetic frustum.

The very large calculated multi-fidelity efficiency ceiling at weak scales is
therefore not an algorithm success. It occurs in a regime where low and high
are almost identical. The scientifically useful question is whether a
measurable real BOST regime exists between “trajectory term below noise” and
“topology/frustum invalid”.

## Independent invariants

The test suite checks:

- uniform index gives a straight ray and zero deflection;
- RK4 `h/h2/h4` refinement;
- exit-direction change against integrated transverse curvature;
- endpoint momentum balance
  `n_out d_out - n_in d_in = integral grad(n) ds`;
- full trajectory JVP against fixed-state central finite differences;
- VJP/JVP dot identity;
- frozen-path derivative is numerically distinct from the full derivative;
- support/frustum signatures are stable under the derivative perturbation;
- domain or stencil escape raises a fail-closed error rather than clipping.

## What must come from He Yuanzhe

1. ROI dimensions in metres and the coordinate normalization map.
2. Laser/background wavelength, gas composition assumptions, pressure and
   temperature range, and the chosen Gladstone-Dale model.
3. Camera intrinsics/extrinsics, pixel pitch, focus, f-number, background plane,
   aperture model and sign convention.
4. A case with the largest measured deflection and a same-geometry weak case.
5. Raw flow-off/flow-on images or at least optical-flow uncertainty and repeated
   calibration data.
6. Whether mask/occupancy/support is fixed or updated during reconstruction.
7. An independent physical endpoint such as thermocouple, PLIF, density or
   held-out-view reprojection.

Without these fields, no numerical scale multiplier can be translated into a
real flame condition and no paper should claim that trajectory-aware routing
solves an experimental problem.

