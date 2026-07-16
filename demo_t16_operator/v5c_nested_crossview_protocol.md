# v5c M0 nested cross-view first-open protocol

Status: `V5C_M0_NEW_GENERATOR_FAMILY_PRECOMMITTED_FIRST_OPEN_NOT_CONFIRMATORY`

This document freezes a small-grid development experiment before its formal
data seed is opened. It is a falsification test for a BOST-specific selection
and fallback protocol. It is not a neural operator result, real-BOS result,
safety guarantee, or paper-level superiority claim.

## 1. Why this experiment exists

The v5b M1 shared-aperture profile experiment recovered the nearest operator
bank radius in five of six blocks, but its selective gate accepted 11 of 48
nominal rows and five of those 11 worsened the audit camera. The worst accepted
audit increase was 32.11%. v5c therefore asks a narrower question:

> Can inner-camera cross-validation select a shared effective-aperture proxy
> and a dimensionless ridge ratio, while sample-excluded per-camera outer
> gates reject the tail failures that passed the v5b pooled gate?

The answer may be no. A no-go result is a valid outcome and must be retained.

## 2. Frozen synthetic scope

- prescribed linear weak-deflection finite-aperture surrogate;
- grid `8 x 8 x 5`, with an explicit `96 / 320` boolean support;
- candidate effective radii: `0, 0.04, 0.08, 0.12, 0.16`;
- off-grid truth radii: `0.055, 0.095, 0.135`;
- two previously Stage-0-audited rigs;
- four inner cameras, two outer cameras, and one final audit camera per rig;
- new generator families relative to v5b M1: `jet_shear` and `shock_cell`;
- four fields per family and two nominal noise levels;
- formal data seed `2026071701`;
- dimensionless ridge ratios `3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3`.

The phrase "new generator families" does not mean family-OOD validation for
the current selector. These fields participate in the v5c inner-camera
selection. Only cameras are held out against the current selection.

## 3. Decision algorithm

For each rig-radius acquisition block:

1. On the four declared inner cameras, score every `(radius, kappa)` pair by
   leave-one-camera-out prediction error.
2. For each field and candidate, use
   `lambda = kappa * mean(diag(A_w^T A_w))`. The effective `lambda` varies
   by field, fold, and radius; it is not a single block-level lambda.
3. Select the pair with minimum mean inner validation MSE.
4. Re-run the complete selection four times after physically deleting one
   inner camera. This is true camera deletion, not deletion of one stored fold
   score.
5. Refit each field on all four inner cameras at the selected pair.
6. Independently tune `kappa` for the metadata-radius fallback on the same
   inner-camera data.
7. Score candidate and fallback separately on each outer camera.

The minimum candidate CV score is selected from 30 pairs whereas fallback CV
is selected from six `kappa` values. Their percentage difference is reported
only as an asymmetric search diagnostic and is never a gate.

## 4. Frozen routing gate

For each sample, first remove that sample from the block-level outer summary.
For each outer camera separately, the remaining samples must satisfy:

- median candidate improvement at least `2%`;
- positive fraction at least `0.75`;
- no improvement below `-1%`.

The inner block must also satisfy:

- true camera-deletion radius stability `4 / 4`;
- profiled radius margin at least `0.001`;
- metadata discrepancy at most `2 sigma`;
- selected radius is not a bank boundary;
- selected radius differs from the metadata fallback.

The current sample must improve by at least `2%` on each outer camera. A
radius equal to the fallback is coded `NO_ACTION_FALLBACK`, not as evidence
of unsafe failure.

Two directional records are constructed:

- outer camera 0 forms the leave-one-sample block/sample gate; outer camera 1
  is recorded as its held-out transfer evaluation;
- outer camera 1 forms the gate; outer camera 0 is its transfer evaluation.

Both directional gate decisions must pass for final acceptance. Because the
same two fixed cameras exchange roles, these statistics are descriptive and
do not constitute an independent risk bound.

## 5. Opening order and evidence boundary

All sample routes are constructed first and hashed in memory. The routing
commitment excludes report-only family labels, true-aperture labels, and block
IDs that encode true radius. Only then are truth-field errors, truth-radius
oracle diagnostics, and the final audit camera evaluated. The frozen protocol
is itself a tracked pre-open dependency and is included in the report source
hashes. The current runner still opens all stages in the same process and does
not provide cryptographic escrow or an independent evaluator.

Consequently:

- audit is not used for sample routing;
- audit is used for the post-open scientific verdict;
- `overall_pilot_pass` is a post-open development verdict;
- two outer views cannot support arbitrary-view safety, conformal coverage, a
  95% harm guarantee, or a view-tail claim;
- held-out projection consistency does not prove the 3-D field is correct.

## 6. Statistical units

The output contains 48 nominal sample-radius rows, but they are not IID.

- field-rig bundles: 16;
- paired rig-radius blocks: 6;
- independent rigs: 2;
- views, detector pixels, and voxels are repeated measurements;
- radius blocks within a rig reuse the same fields and paired noise seeds.

No binomial interval or safety rate may use 48 rows, 96 sample-view values,
pixels, or voxels as independent trials.

## 7. Known oracle and model-assisted elements

- noise sigma is scaled from clean inner-view truth, so whitening is an
  oracle/nominal simulation device, not an experimental noise calibration;
- metadata radius is a truth-derived synthetic proxy with a frozen rig bias;
- the support mask is generator matched;
- the effective radius is a simulation coordinate, not a recovered mechanical
  f-number or entrance-pupil diameter;
- the surrogate omits nonlinear ray bending, real background extraction,
  camera calibration error, illumination drift, and real reacting-flow data.

These limitations remain in the report even if every pilot gate passes.

## 8. Predefined post-open development checks

The pilot reports, but does not promote to confirmation, whether:

- nearest-bank truth recovery is `6 / 6`;
- final nominal-row coverage is at least `20%`;
- accepted mean field gain is at least `2%`;
- accepted field-gain p10 is nonnegative;
- accepted field harm below `-1%` is at most `5%` as a point estimate;
- accepted audit-increase rate is at most `5%` as a point estimate;
- mean selected audit change is nonpositive;
- both directional outer-transfer summaries satisfy the same utility/tail
  thresholds and every accepted block has nonnegative mean held-out gain.

Zero accepted samples is `INCONCLUSIVE_UTILITY_FAIL`, not safety success.
Failure of any post-open check is `NO_GO` for promotion of this gate.

## 9. Baselines required before a paper claim

v5c does not yet compare against all explanations an external reviewer would
expect. Before any method claim, the same split and forward operator must test:

1. pinhole reconstruction;
2. metadata finite-aperture reconstruction;
3. known-truth-radius oracle, labelled as headroom only;
4. fixed ridge;
5. generalized cross-validation (GCV);
6. Morozov discrepancy selection;
7. per-scene aperture rather than shared aperture;
8. shared-aperture nested CV without selective fallback;
9. the full nested CV plus fallback gate.

NeRIF, TDBOST, FNO, FFNO, and DeepONet are later matched-budget baselines.
They are not part of this small-matrix mechanism check.

## 10. Closest prior-art boundary

- [Golub and Pereyra, variable projection](https://epubs.siam.org/doi/10.1137/0710036):
  profiling linear nuisance variables is established.
- [Golub, Heath and Wahba, GCV](https://www.tandfonline.com/doi/abs/10.1080/00401706.1979.10489751):
  cross-validated regularization selection is established.
- [Morozov discrepancy principle](https://www.mathnet.ru/eng/zvmmf/v8/i2/p295):
  noise-level-based regularization selection is established.
- [Reeves and Mersereau 1992](https://pubmed.ncbi.nlm.nih.gov/18296164/):
  joint blur/model/regularization identification already exists in imaging.
- [Pathuri-Bhuvana et al. 2020](https://eurasip.org/Proceedings/Eusipco/Eusipco2020/pdfs/0001931.pdf):
  constrained VarPro for tomography and sensor-position calibration exists.
- [Molnar et al., finite-aperture BOS](https://arxiv.org/abs/2402.15954):
  cone-ray forward/inverse modelling and real f-stop experiments exist.
- [NeRIF](https://arxiv.org/abs/2409.14722):
  neural refractive-index fields and held-out-view validation exist.
- [TDBOST](https://dl.acm.org/doi/10.1145/3809488):
  four-dimensional BOST and a held-out test perspective exist.
- [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html):
  rejection and risk-coverage analysis are established.

The narrow contribution still available for falsification is a BOST-specific,
leakage-audited combination of shared effective-aperture selection and
candidate-versus-metadata fallback. None of its individual components may be
claimed as a mathematical first.

## 11. Real f-stop promotion gate

Promotion to a real experiment requires, at minimum:

- one physical rig with at least three fixed f-stops;
- paired raw flow-off/reference and flow-on captures;
- randomized f-stop order and before/after references to expose drift;
- lens/camera IDs, focal length, nominal f-number, entrance pupil or effective
  f-number, pixel pitch, exposure, gain, distances, illumination and timestamp;
- controlled photon count/SNR across apertures;
- intrinsic/extrinsic/distortion calibration and multi-depth dot-PSF or MTF;
- a frozen inner/outer/final camera split and independently calibrated
  thresholds;
- an external density/temperature/pressure standard, CFD comparison, or other
  truth proxy beyond held-out reprojection.

Until that gate is met, v5c can guide algorithm design but cannot establish an
OERF experimental conclusion.
