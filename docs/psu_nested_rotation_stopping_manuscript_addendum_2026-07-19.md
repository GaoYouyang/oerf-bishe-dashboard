# Geometry-Dependent Camera-Tail Failure of Nested Early Stopping in Experimental BOST

**Methods and results addendum, 2026-07-19**

**Status:** reproducible negative-result draft; not submission-ready

**Evidence level:** one experimental PSU flow, three support rotations, post-hoc programmatically withheld cross-fit reanalysis

**Primary decision:** `NO_GO` on both `16^3` and `32^3` grids

**Closed claims:** field accuracy, statistical significance, independent replication, cross-flow generalization, algorithm superiority

## Plain-language note

This experiment asked whether two camera rotations could choose a safe CGLS stopping step for a third rotation. The inner check selected earlier stopping only when the held-out rotation was `50 deg`. Those choices looked safe on the two inner transfer directions, but the third rotation exposed worse errors for one camera and for the residual tail. The result is useful because it prevents a weak average-only method from becoming the thesis algorithm. It also identifies the next research target: a geometry-aware camera-tail risk certificate with a conservative classical fallback.

## Abstract

Background-oriented schlieren tomography (BOST) reconstructs a volumetric refractive-index or density field from sparse, gradient-sensitive optical observations. Conjugate-gradient least-squares (CGLS) iterates can exhibit semi-convergence: continued reduction of the training residual need not improve a withheld projection geometry. We tested whether an early-stopping checkpoint selected only from transfer between two support rotations could safely generalize to a third rotation.

The study used three rotation blocks (`0 deg`, `50 deg`, and `90 deg`) from one experimental PSU flow and two discretizations (`16^3` and `32^3`). Six unique single-rotation CGLS trajectories were run from the zero field. Each trajectory retained checkpoints `k in {0,1,2,3,4,6,8,12}`, yielding 48 private float64 fields. For each outer rotation, the other two rotations formed two directed inner transfers. A non-reference checkpoint was eligible only if it was no worse than fixed `k=4` for pooled, camera-macro, worst-camera, group-p95, and every target-camera relative-L2 and p95 metric in both directions. Selections were sealed before programmatic access to the outer scores.

The sealed choices, ordered by outer rotation and then grid, were `[4,4,3,2,4,4]`. Only outer `50 deg` selected an earlier checkpoint. On `16^3`, `k=3` yielded a positive equal-rotation mean camera-macro improvement of `+4.17966e-4`, but outer-`50 deg` group p95, camera-5 relative-L2, and camera-5 p95 worsened by `0.0398%`, `1.6893%`, and `2.1896%`, respectively. On `32^3`, `k=2` produced a negative mean camera-macro improvement of `-2.10334e-3`; the corresponding camera-5 relative-L2 and p95 worsened by `3.0282%` and `3.8310%`. Both grids therefore failed the predeclared outer no-harm screen.

These results do not compare reconstructed fields with ground truth and do not establish a universal stopping rule. They provide a bounded experimental counterexample: inner-safe projection transfer across two rotations did not protect the camera-tail behavior of a third rotation. Future methods should condition stopping or correction on camera geometry and tail risk, retain strong H1/TV/pyramid baselines, and abstain when a deployable risk bound cannot certify no harm.

## 1. Motivation

### 1.1 Physical inverse problem

In BOST, the observation operator is sensitive to integrated transverse refractive-index gradients along camera rays. Sparse view coverage, finite aperture, masking, displacement extraction, and the spatial support of steep gradients can make the inverse problem strongly geometry dependent. A low pooled reprojection residual is therefore not sufficient evidence that every camera, every high-error ray, or the three-dimensional field has improved.

Neural refractive-index fields replace a fixed voxel representation with a continuous neural representation and directly connect representation learning to BOST reconstruction. That result motivates learned volumetric models, but it does not make projection-space model selection geometry invariant. Likewise, a true pyramid BOST method changes field and background-image resolution while correcting projection data and projection matrices across levels; merely changing the voxel grid or upsampling a coarse field is not an equivalent baseline.

### 1.2 Numerical inverse problem

Let `A_r,g` denote the BOST operator for rotation `r` and grid `g`, and let `y_r` denote the corresponding observations. For a training rotation `r`, CGLS produces a trajectory

```text
x[r,g,0], x[r,g,1], ..., x[r,g,12].
```

Early stopping acts as an iterative regularizer in many inverse problems. However, theory distinguishes prediction risk from reconstruction risk and requires assumptions on the noise and operator. In this experiment, withheld reprojection is only an observable prediction-space proxy. It cannot be relabeled as field error.

### 1.3 Research question

For outer rotation `o`, let `r` and `s` be the two remaining rotations. The narrow question is:

> Can directed transfers `r -> s` and `s -> r`, without programmatic access to outer-`o` scores, select a checkpoint that improves the outer camera-macro score relative to fixed `k=4` while preserving every declared outer group, camera, and tail metric?

The experiment does not ask whether CGLS is globally optimal, whether one grid is physically more accurate, or whether a neural operator outperforms classical reconstruction.

## 2. Evidence design

### 2.1 Data roles

| outer rotation | inner direction 1 | inner direction 2 | outer reconstruction source |
|---:|---|---|---|
| `0 deg` | train `50 deg`, validate `90 deg` | train `90 deg`, validate `50 deg` | E71 train `50+90 deg`, score `0 deg` |
| `50 deg` | train `0 deg`, validate `90 deg` | train `90 deg`, validate `0 deg` | E71 train `0+90 deg`, score `50 deg` |
| `90 deg` | train `0 deg`, validate `50 deg` | train `50 deg`, validate `0 deg` | E71 train `0+50 deg`, score `90 deg` |

The three cameras inside a rotation are correlated observations of the same physical flow and are not treated as independent experimental replicates. The two grids are selected and judged separately; outer results cannot be used to select a grid.

### 2.2 Unique-fit contract

The unique fit key was

```text
(training rotation, grid, cache-manifest hash, solver-config hash).
```

Outer-fold identity was excluded from this key. The full experiment therefore contained exactly

```text
3 rotations x 2 grids = 6 unique fits.
```

Each fit ran 12 CGLS iterations from the exact zero field and cloned eight declared checkpoints. The final evidence budget was:

| evidence unit | measured value |
|---|---:|
| unique CGLS fits | 6 |
| private float64 fields | 48 |
| complete-artifact forward calls `A` | 122 |
| complete-artifact adjoint calls `A^T` | 86 |
| outer-join calls `A / A^T` | `0 / 0` |
| wall time | 1,104.606 s |
| maximum resident memory | 7.110 GB |
| numerical/isolation/call/privacy gates | 15/15 passed |

### 2.3 Transparent preflight correction

The initial protocol commit contained a transcription error in the three per-view ray counts for rotation `50 deg`. Their incorrect sum happened to equal the correct sum, so a total-ray check did not detect the problem. Per-view comparison against two frozen cache manifests stopped the run before the shared adjoint check, any CGLS trajectory, or any selection. The corrected values were `988006`, `1363792`, and `1311324` rays for views 3, 4, and 5. Only these three metadata integers changed; arrays, hashes, checkpoints, metrics, tolerances, fallback, and decision rules did not.

The corrected formal protocol commit was:

```text
78eb2d5f5a31b6bca68652828ca194900b4326ff
```

This incident is retained because it demonstrates why per-view identity checks are required even when aggregate dimensions match.

## 3. Nested checkpoint selection

### 3.1 Inner ratios

For inner direction `d`, metric `m`, and checkpoint `k`, define the ratio relative to fixed `k=4`:

```text
q[d,m](k) = E[d,m](k) / max(E[d,m](4), 1e-15).
```

The primary risk and average macro ratio were

```text
R(k) = max q[d,m](k)
       over both directions and
       {camera-macro L2, worst-camera L2, group p95},

A(k) = mean q[d,camera-macro L2](k)
       over both directions.
```

### 3.2 Feasibility and fail-closed fallback

A candidate had to satisfy, in both inner directions:

- pooled, camera-macro, and worst-camera relative-L2 no harm;
- group p95 no harm;
- relative-L2 no harm for every target camera;
- p95 no harm for every target camera.

The numerical tolerance was `1e-8`. This tolerance is not a physical significance threshold or a repeatability scale.

A non-reference candidate was selectable only if it was feasible and `R(k) < 1 - 1e-8`. If none existed, the selector returned fixed `k=4`. Otherwise it minimized `R(k)`, then `A(k)` within tolerance, then selected the earlier checkpoint. This ordering was frozen before the six formal trajectories.

### 3.3 Two-stage sealing

The inner stage wrote aggregate scores and `sealed_selection.json` without reading the E71 outer summary. The selection was committed before the outer join. The outer stage verified the protocol commit, selection commit, byte hashes of the inner files, and frozen hashes of the E71 source evidence before joining the selections to the existing outer scores. It did not call `A` or `A^T`.

The selection commit and aggregate selection hash were:

```text
08cb57ff282ac5ed81ad35fa1d8b3ca9de2cd583
bbce7570ab7c7be5946b2c25102985c03833bec4442d2f003e94c898a12e12ea
```

This software separation prevents the selector from loading outer scores. It does not turn the experiment into a genuinely human-blind final test because the E71 outer trajectories already existed and had been inspected before E72 was designed.

## 4. Results

### 4.1 Inner selections

| outer | grid | selected | worst inner primary-risk ratio | mean inner macro ratio | decision |
|---:|---:|---:|---:|---:|---|
| `0 deg` | `16^3` | 4 | 1.000000 | 1.000000 | fallback |
| `0 deg` | `32^3` | 4 | 1.000000 | 1.000000 | fallback |
| `50 deg` | `16^3` | 3 | 0.998016749 | 0.996808334 | strict inner improvement |
| `50 deg` | `32^3` | 2 | 0.998748518 | 0.998239369 | strict inner improvement |
| `90 deg` | `16^3` | 4 | 1.000000 | 1.000000 | fallback |
| `90 deg` | `32^3` | 4 | 1.000000 | 1.000000 | fallback |

The selection vector, ordered by outer rotation and then grid, was:

```text
[4, 4, 3, 2, 4, 4].
```

### 4.2 Outer decision

Outer `0 deg` and `90 deg` used the reference checkpoint and therefore produced zero difference. All nontrivial outer evidence came from rotation `50 deg`.

| grid | selected | three-rotation mean macro `k4-selected` | outer-50 group p95 change | camera-5 L2 change | camera-5 p95 change | final |
|---:|---:|---:|---:|---:|---:|---|
| `16^3` | 3 | `+0.000417966` | `+0.0398%` | `+1.6893%` | `+2.1896%` | `NO_GO` |
| `32^3` | 2 | `-0.002103342` | `+1.1220%` | `+3.0282%` | `+3.8310%` | `NO_GO` |

For `16^3`, the outer-`50 deg` camera-macro relative-L2 improved from `0.961193995` at `k=4` to `0.959940096` at `k=3`. However, group p95 increased from `2.088186845` to `2.089018184`; camera-5 relative-L2 increased from `0.893601874` to `0.908697823`; and camera-5 p95 increased from `1.949612386` to `1.992301877`. The positive mean therefore failed the camera-tail no-harm requirement.

For `32^3`, outer-`50 deg` camera-macro relative-L2 worsened from `0.978850638` at `k=4` to `0.985160664` at `k=2`. Pooled relative-L2, camera macro, group p95, camera-5 relative-L2, and camera-5 p95 all failed their declared gates.

### 4.3 Classical diagnostics

The minimum training residual selected the latest checkpoint `k=12` in all six fit cells, illustrating why training consistency alone does not define a safe stopping rule. A sparse-checkpoint L-curve selected checkpoints in `{3,4,6}`, but it was retained only as a diagnostic. Generalized cross-validation was disabled because no validated influence trace or grouped-noise model was available. The discrepancy principle was disabled because no independent repeatability-noise scale or model-bias bound was available. An outer oracle checkpoint was reported only as an undeployable diagnostic upper bound.

## 5. Interpretation

### 5.1 Supported inference

The data support the following narrow statement:

> In a post-hoc, programmatically withheld cross-fit reanalysis of one experimental PSU flow, a checkpoint that was safe for bidirectional transfer between two rotations did not preserve all predeclared projection-space camera and tail metrics on the third rotation, on either tested grid.

This is a counterexample to the current selector, not a theorem that rotation-based stopping can never work.

### 5.2 Physical hypotheses, not conclusions

The camera-5 failure could be associated with one or more of:

- rotation- and camera-dependent ray coverage or support/null-space structure;
- finite-aperture or depth-of-field mismatch relative to the current forward model;
- concentrated steep-gradient regions that dominate residual tails;
- camera-specific displacement extraction, registration, masking, or noise;
- grid-dependent interaction between CGLS spectral filtering and geometry.

The current evidence cannot distinguish these mechanisms. No mechanism should be named as the cause until camera pose/aperture metadata, flow-off repeats, and targeted forward-model ablations are available.

### 5.3 Why this matters for operator learning

A generic FNO, DeepONet, CNO, or MgNO trained on pooled loss could reproduce the same failure while improving the mean. The appropriate operator-learning target is therefore not an unconstrained replacement of the inverse solver. A more defensible architecture would:

1. begin from a strong classical reconstruction;
2. expose camera geometry, ray coverage, and early residual-path features;
3. predict a bounded correction or a risk envelope rather than an unrestricted field;
4. enforce forward/data consistency;
5. evaluate every rotation-camera-tail group;
6. abstain and return the classical baseline whenever the certificate is insufficient.

## 6. Next falsifiable methods

### 6.1 E73-0: strong classical baseline ladder

Before another learned selector, compare fixed CGLS with:

- H1/Tikhonov regularization paths;
- TV or Huber primal-dual reconstruction;
- a faithful pyramid BOST implementation that updates field scale, background-image scale, projection data, and projection matrices together;
- discrepancy stopping after an independent noise-scale contract is available.

This ladder tests whether the observed tail can be controlled without learning.

### 6.2 E73-A: GroupTail-CGLS

Define groups over

```text
(flow instance, source rotation, target rotation, camera, residual-tail bin)
```

and select a checkpoint by a worst-group or CVaR objective, with earlier-step tie-breaking and fixed-`k4` fallback. This is not novel merely because it uses a maximum or CVaR loss. A contribution would require BOST-specific geometry features, a calibrated uncertainty contract, and evidence on newly held-out flows.

### 6.3 E73-B: geometry-calibrated tail envelope

Use only deployment-observable inputs:

- camera pose and rotation;
- ray-coverage and support summaries;
- finite-aperture metadata;
- slopes, curvature, quantiles, and cross-camera disagreement from early residual paths;
- low-cost spectral summaries related to the observable/null-space split;
- grid and signal scales.

The model predicts an upper bound for target-camera tail risk relative to a strong classical baseline. If the upper bound cannot certify the declared no-harm condition, the method abstains. This route is more promising than checkpoint classification because its failure mode is explicit and auditable.

### 6.4 E73-C: certificate-constrained neural correction

Only after E73-0 and E73-B are established should a neural operator learn a correction to the classical field. The correction must pass a physical forward projection, data-consistency check, and geometry-tail certificate. The primary comparison remains fixed classical reconstruction under equal operator calls, not a weak unconverged baseline.

## 7. Publication-conversion gate

This addendum can become part of a defensible paper only after the following evidence exists:

| missing evidence | minimum acceptable addition | claim enabled if passed |
|---|---|---|
| field truth | CFD/synthetic refractive-index fields rendered through the target geometry | field relative-L2 and front metrics |
| cross-flow diversity | multiple independent flow conditions or time blocks | bounded cross-flow transfer claim |
| noise scale | flow-off repeats with camera-specific covariance/quality metadata | discrepancy rule and uncertainty calibration |
| strong baselines | tuned H1/TV/pyramid and equal-budget neural baselines | comparative method claim |
| final holdout | protocol frozen before any scores from new flow/session are viewed | genuine prospective test |
| statistical unit | flow/session-level replicates rather than camera rows | uncertainty intervals and statistical comparison |
| mechanism audit | camera-5 pose/aperture/coverage/extraction ablations | physical attribution |

Until these conditions are met, the correct manuscript role is a negative result and method-design constraint, not a headline performance contribution.

## 8. Reproducibility

Independent public-package validation:

```bash
.venv/bin/python -m site_tools.validate_psu_nested_rotation_stopping_public
```

Figure reproduction:

```bash
.venv/bin/python -m site_tools.plot_psu_nested_rotation_stopping
```

Focused regression:

```bash
.venv/bin/python -m pytest -q \
  site_tools/test_validate_psu_nested_rotation_stopping_public.py \
  site_tools/test_plot_psu_nested_rotation_stopping.py \
  site_tools/test_psu_nested_rotation_selector.py \
  site_tools/test_nested_rotation_stopping_runner.py
```

Public artifacts contain aggregate metrics, hashes, and decision metadata only. Private ray arrays, cache paths, measurement vectors, and 48 reconstructed checkpoint fields remain outside Git.

## 9. Verified primary sources

1. He, Y. et al. [Neural refractive index field: Unlocking the potential of background-oriented schlieren tomography in volumetric flow visualization](https://doi.org/10.1063/5.0250899). *Physics of Fluids* 37, 017143 (2025). Directly relevant to the OERF/He-Yuanzhe continuous-field BOST route; it does not establish the E72 stopping rule.
2. Hu, W. et al. [A pyramid approach for background-oriented schlieren tomography](https://doi.org/10.1007/s00348-025-04153-3). *Experiments in Fluids* 67, 4 (2026). A required coarse-to-fine baseline because it synchronizes field/background scaling and projection correction.
3. Hucker, L. and Reiss, M. [Early stopping for conjugate gradients in statistical inverse problems](https://doi.org/10.1007/s00211-025-01469-4). *Numerische Mathematik* 157, 1739-1791 (2025). Establishes early stopping as regularization under explicit statistical assumptions and distinguishes prediction from reconstruction risk.
4. Cai, W. et al. [Direct background-oriented schlieren tomography with radial basis functions](https://opg.optica.org/oe/fulltext.cfm?uri=oe-30-11-19100). *Optics Express* 30, 19100 (2022). Supports independent-camera ray-traced reprojection as an observable validation surface; reprojection remains distinct from field truth.

## 10. Reviewer-facing claim table

| proposed statement | status | reason |
|---|---|---|
| six formal trajectories and selections are reproducible | open | independent validator and hashes pass |
| both grids fail the declared outer screen | open | directly recomputed from public aggregate evidence |
| inner-safe transfer does not guarantee third-rotation camera-tail safety in this case | open | observed counterexample |
| camera 5 fails because of null space or aperture | closed | mechanism not isolated |
| selected fields are less accurate in 3D | closed | no field truth |
| the result is statistically significant | closed | one flow and no independent replication |
| the selector generalizes across flows | closed | no cross-flow test |
| a new neural operator outperforms FNO/DeepONet | closed | no neural operator trained and no fair benchmark |
