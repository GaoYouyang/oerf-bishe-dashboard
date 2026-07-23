# JACRU N1.8 camera/ray hybrid design freeze

Date: 2026-07-18

Status: `POSTOPEN_DESIGN_SCREEN_NO_NEW_GEOMETRY`

Evidence: `E1_SYNTHETIC_OPENED_DEVELOPMENT_HYPOTHESIS_DESIGN_ONLY`

## 1. What this run may establish

This run reuses the already-opened synthetic development cases. It may compare
five frozen representation families and nominate at most one hypothesis for a
later, separately committed new-split preregistration. It cannot establish an
algorithm gain, learned-model gain, fresh confirmation, OOD transfer, real-BOST
validity, or superiority over DeepONet, FNO, NeRIF, or TDBOST.

No learner and no finite-K truth-conditioned field search are permitted in this
screen. The evaluated-case truth is forbidden during basis construction.

## 2. Physical and numerical target

The current fixture isolates one narrow model-error problem: observations are
rendered from a continuous analytic refractive-index gradient, while inversion
uses a voxel finite-difference and trilinear ray operator. It does not yet model
finite aperture, ray bending, optical-flow bias, or calibration drift.

Let `d` be the visible component-damping correction, `r` the normalized warm
data residual, `P` the reconstruction support, and `K = A P A^T`. Every candidate
constructs an orthonormal measurement-space basis and applies one total
correction `delta = Bc`.

The entire correction, including the damping direction, obeys

```text
||c||_2 = ||delta||_2 <= min(2||d||, max(||d||, 16||r||)).
```

## 3. Frozen candidate set and fair cost

| Candidate | Raw frame | Setup | Refine | Total low-operator budget |
|---|---|---:|---:|---:|
| `krylov4_total` | `d, r, Kd, Kr` | 2F / 2AT | 10 | 25F / 24AT |
| `fit_pca2_krylov6_total` | `d, r, fit-PCA1, fit-PCA2, Kd, Kr` | 2F / 2AT | 10 | 25F / 24AT |
| `camera_block6_total` | per-camera masked `d` and `r` | 0F / 0AT | 12 | 25F / 24AT |
| `pose_fourier_krylov6_total` | `d, r, sin(theta)r, cos(theta)r, Ksin(theta)r, Kcos(theta)r` | 2F / 2AT | 10 | 25F / 24AT |
| `detector_moment_krylov6_total` | `d, r, ur, vr, Kur, Kvr` | 2F / 2AT | 10 | 25F / 24AT |

The fit-PCA modes come only from the fit partition and exclude the fit mean.
Detector moments fail closed unless the current synthetic camera-major ray order
is exact; a real-data adapter must provide authoritative detector coordinates.

## 4. Design-screen decision rule

Reconstruction eligibility requires all frozen checks, including mean field gain
at least 5% over matched CGLS-24, mean H1 gain at least 3%, mean field gain at
least 1% over component damping, nonnegative opened-set tails, no case harmed by
more than 1%, exact-oracle gain retention at least 70%, matched cost, valid rank,
and orthonormality defect at most `1e-10`.

Incremental extra-headroom is not the old gain ratio. It is frozen as

```text
R_extra = sum_i(E_damping_i - E_candidate_i)
          / sum_i(E_damping_i - E_exact_i).
```

The threshold is 60%, above all already-opened N1.7 variants. Per-case ratios are
forbidden because cases with nearly zero exact headroom would dominate.

`P A^T` residual gain is a separate mechanism gate, not a substitute for field
or H1 performance. Reconstruction pass plus at least 50% adjoint gain may nominate
a forward-correction representation. Reconstruction pass with lower nonnegative
adjoint gain may only nominate a solver-aware representation. Otherwise the
result is no-go.

Protocol audit amendment: every candidate must attain its designed rank exactly
(4 for Krylov-4 and 6 for the four rank-6 candidates), and comparison with the
frozen N1.7 aggregate is legal only after the development case IDs and geometry
digests are verified identical. A negative `P A^T` gain always fails closed even
if all reconstruction checks pass.

The independent experimental unit is one base-seed geometry cluster. Its paired
phantom families are repeated observations, not additional independent samples.

## 5. Reproduction before any new split

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q \
  demo_t16_operator/test_jacru_n1_8_camera_ray_hybrid.py \
  site_tools/test_run_jacru_n1_8_hybrid_design_screen.py

PYTHONPATH=. .venv/bin/python \
  site_tools/run_jacru_n1_8_hybrid_design_screen.py \
  --output-dir demo_t16_operator/results/jacru_n1_8_hybrid_design_screen_postopen_full1
```

The code, config, tests, and this document must be committed before the full
screen. A pass only authorizes drafting a new-split preregistration. It does not
authorize generating or opening that split.
