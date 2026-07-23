# JACRU N1.2: primary-source literature draft

- Draft date: 2026-07-18
- Scope: finite-sample split-conformal calibration, session/group shift, discrepancy stopping, covariance-aware tomography, BOS background/displacement uncertainty, and a fail-closed learned-reconstruction boundary.
- Source policy: every entry below links to a publisher/proceedings/official archive/DOI record. No blog, survey, or secondary database is used to carry a technical claim.
- Claim boundary: this is an experiment-design bibliography. It does **not** establish that N1.2 has calibrated coverage, that a BOST covariance model is correct, or that JACRU is safe or superior.

## Reading rule

`[GENERAL MATH/ML]` means the source is not OERF/BOS/BOST-specific and supplies only a mathematical or methodological constraint. `[TOMOGRAPHY TRANSFER]` means the imaging mechanism differs from BOS/BOST. Only entries marked `[BOS]` directly study BOS measurements. None of the cited sources directly validates the current synthetic JACRU fixture or an OERF instrument.

## A. Finite-sample conformal quantiles and session/group shift

### 1. Romano, Patterson, and Candès (2019), *Conformalized Quantile Regression*

- **Type:** `[GENERAL MATH/ML]`
- **Official URL:** [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html)
- **Exact claim supported:** the paper combines quantile regression with conformal calibration and proves finite-sample marginal coverage while allowing the base interval width to adapt to heteroscedasticity. It does not promise conditional, per-session, or OOD coverage.
- **N1.2 constraint:** a learned/heteroscedastic score may set the *shape* of a candidate-specific interval or discrepancy envelope, but the final inflation quantile must be computed on a disjoint threshold-calibration split. Do not tune its score, its quantile, or its camera grouping on the audit rows; report marginal coverage as such, not per-camera conditional coverage.

### 2. Tibshirani, Barber, Candès, and Ramdas (2019), *Conformal Prediction Under Covariate Shift*

- **Type:** `[GENERAL MATH/ML]`
- **Official URL:** [NeurIPS paper PDF](https://proceedings.neurips.cc/paper/2019/file/8fb21ee7a2207526da55a679f0332de2-Paper.pdf)
- **Exact claim supported:** for exchangeable scores, the augmented empirical quantile that includes `+infinity` gives finite-sample coverage. The paper also gives a weighted construction for covariate shift when the test/train density ratio is known or accurately estimated; it is not a free guarantee for arbitrary drift.
- **N1.2 constraint:** with `n_cal=64` and target `1-alpha=0.95`, use the `ceil((64+1)*0.95)=62`nd ordered calibration score (equivalently the augmented conformal quantile), not the ordinary empirical 95th percentile. A new session must not inherit a strict 95% statement from a pooled session unless N1.2 pre-specifies and validates a legitimate weighting/shift model; otherwise call its result a transfer audit only.

### 3. Barber, Candès, Ramdas, and Tibshirani (2023), *Conformal Prediction Beyond Exchangeability*

- **Type:** `[GENERAL MATH/ML]`
- **Official URL / DOI:** [Annals of Statistics official article](https://projecteuclid.org/journals/annals-of-statistics/volume-51/issue-2/Conformal-prediction-beyond-exchangeability/10.1214/23-AOS2276.full), [DOI](https://doi.org/10.1214/23-AOS2276)
- **Exact claim supported:** ordinary split/full conformal validity relies on data exchangeability and a symmetric fitting algorithm. The paper quantifies coverage loss under nonexchangeability and develops weighted/randomized variants with robustness bounds; it does not make arbitrary temporal or correlated sessions exchangeable.
- **N1.2 constraint:** define the unit before splitting: a calibration packet/session, not an individual repeated frame that shares optics, temperature, or background state with its siblings. Keep a complete session/rig block out for audit. If time weighting is explored, freeze the weights before audit and publish the resulting nonexchangeability-bound assumptions; never relabel a within-session frame shuffle as an independent-session coverage proof.

### 4. Ma, Pitt, Azizzadenesheli, and Anandkumar (2024), *Calibrated Uncertainty Quantification for Operator Learning via Conformal Prediction*

- **Type:** `[GENERAL OPERATOR LEARNING]`
- **Official URL:** [TMLR OpenReview record](https://openreview.net/forum?id=cGpegxy12T)
- **Exact claim supported:** the paper proposes a distribution-free, finite-sample conformal calibration method for operator outputs and defines its functional coverage as the expected percentage of domain points whose true value lies in the predicted uncertainty ball. This is a functional/point-fraction target, not simultaneous whole-function coverage.
- **N1.2 constraint:** if N1.2 conformalizes a voxel uncertainty map or a correction envelope, pre-register whether the target is (a) expected voxel/point fraction covered, (b) per-volume all-voxel coverage, or (c) an event such as “safe to correct.” Report them separately. A 95% point-fraction result cannot be described as a 95% simultaneous volumetric BOST confidence set.

### 5. Moya, Mollaali, Zhang, Lu, and Lin (2025), *Conformalized-DeepONet: A distribution-free framework for uncertainty quantification in deep operator networks*

- **Type:** `[GENERAL OPERATOR LEARNING]`
- **Official URL / DOI:** [DOI](https://doi.org/10.1016/j.physd.2024.134418), [publisher record](https://www.sciencedirect.com/science/article/pii/S0167278924003683)
- **Exact claim supported:** the paper applies conformal prediction to DeepONet regression to construct distribution-free prediction intervals with coverage guarantees. Its claim is about the stated prediction-interval construction, not BOST data consistency, correlated detector noise, or a fail-safe reconstruction gate.
- **N1.2 constraint:** retain a conformalized DeepONet/JACRU output only as an uncertainty baseline. It cannot replace a whitened measurement residual, a held-out-camera check, or the raw-center no-harm comparison; interval coverage and reconstruction safety are distinct endpoints.

## B. Discrepancy principles and early stopping

### 6. Morozov (1966), *Regularization of incorrectly posed problems and the choice of regularization parameter*

- **Type:** `[GENERAL INVERSE-PROBLEM MATH]`
- **Official URL / DOI:** [MathNet record](https://www.mathnet.ru/eng/zvmmf7494), [DOI](https://doi.org/10.1016/0041-5553(66)90046-2)
- **Exact claim supported:** this is the original error/discrepancy-principle line: choose a regularization level in relation to the data-error level rather than drive an ill-posed inverse residual arbitrarily to zero. It is a deterministic inverse-problem principle, not a proof that the N1.2 residual is Gaussian or that its covariance is known.
- **N1.2 constraint:** estimate the stopping/noise scale from the independent flow-off/background packet, freeze `tau` and the first-crossing rule, and prohibit target-truth or clean-target residual from selecting lambda/K. Failure to identify a deployment-visible error scale means N1.2 may report a ceiling only, not a Morozov deployment rule.

### 7. Blanchard and Mathé (2012), *Discrepancy Principle for Statistical Inverse Problems with Application to Conjugate Gradient Iteration*

- **Type:** `[GENERAL INVERSE-PROBLEM MATH]`
- **Official URL / DOI:** [IOP DOI](https://doi.org/10.1088/0266-5611/28/11/115011), [authors' official publication list](https://www.imo.universite-paris-saclay.fr/~gilles.blanchard/publi/index.html)
- **Exact claim supported:** under its statistical inverse-problem assumptions, a plain discrepancy rule can be ill-defined or rate-suboptimal; the paper introduces a modified discrepancy and analyzes it for linear regularization and conjugate-gradient iteration. It does not prescribe a BOS camera-block statistic.
- **N1.2 constraint:** do not crown one global whitened norm as the only stopping certificate. Freeze and compare at least global upper, per-camera/block upper, and an anti-over-regularization lower rule, then show which statistic fails under each predeclared synthetic stress. The selected rule must be chosen without audit labels.

### 8. Hucker and Reiß (2025), *Early stopping for conjugate gradients in statistical inverse problems*

- **Type:** `[GENERAL INVERSE-PROBLEM MATH]`
- **Official URL / DOI:** [publisher DOI](https://doi.org/10.1007/s00211-025-01469-4), [Humboldt University repository record](https://edoc.hu-berlin.de/items/0d5a63e8-0dee-4df3-adfd-80ccdabd6032)
- **Exact claim supported:** early-stopped CG acts as regularization; their data-driven stopping result requires that noise-level estimation error is not dominant, and distinguishes prediction and reconstruction error analyses. The stated theory is for its statistical inverse model, not correlated BOS bias or learned nonlinear correction.
- **N1.2 constraint:** publish a noise-scale sensitivity sweep around the flow-off estimate and report field/H1, raw and whitened residual, and held-out-camera/renderer endpoints separately. If modest noise-scale perturbations reverse tail safety, fail closed; a favorable reprojection curve alone cannot authorize learned stopping.

## C. Covariance whitening and robust tomography

### 9. Moorkamp and Avdeeva (2020), *Using non-diagonal data covariances in geophysical inversion*

- **Type:** `[TOMOGRAPHY TRANSFER]` (magnetotellurics, not BOS/BOST)
- **Official URL / DOI:** [Oxford Academic article](https://academic.oup.com/gji/article/222/2/1023/5836719), [DOI](https://doi.org/10.1093/gji/ggaa235)
- **Exact claim supported:** a generalized least-squares data term can use a non-diagonal data covariance, including correlations induced when observed data are transformed; the paper demonstrates the method for geophysical inversion. It does not identify an OERF camera covariance rank, block structure, or shrinkage value.
- **N1.2 constraint:** compare identity, diagonal, camera/component block, and low-rank-plus-diagonal covariance models against the same fixed reconstructor and cost accounting. Estimate each covariance only from its allowed flow-off fit split, publish condition number/shrinkage/whitened correlation diagnostics, and treat a failed whiteness diagnostic as a covariance-model failure rather than as field evidence.

### 10. Stayman, Zbijewski, Tilley, and Siewerdsen (2014), *Generalized Least-Squares CT Reconstruction with Detector Blur and Correlated Noise Models*

- **Type:** `[TOMOGRAPHY TRANSFER]` (cone-beam CT, not BOS/BOST)
- **Official URL / DOI:** [NIH/PMC author manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC4201055/), [DOI](https://doi.org/10.1117/12.2043067)
- **Exact claim supported:** in their CT study, detector blur and correlated projection noise are jointly modeled in regularized GLS; ignoring correlation can impair the noise-resolution trade-off. It does not show that a BOS blur/error can always be represented by covariance alone.
- **N1.2 constraint:** predefine mutually interpretable branches: `C`-only whitening, forward-operator blur/model-mismatch correction, and a joint branch. Do not absorb an optical/calibration bias into `C` and then claim statistical noise calibration; compare against a forward-mismatch stress and record whether the same residual structure remains after whitening.

## D. BOS displacement, background, and uncertainty propagation

### 11. Rajendran, Zhang, Bhattacharya, Bane, and Vlachos (2020), *Uncertainty quantification in density estimation from background-oriented Schlieren measurements*

- **Type:** `[BOS]`
- **Official URL / DOI:** [OSTI official record](https://www.osti.gov/pages/biblio/1598804), [DOI](https://doi.org/10.1088/1361-6501/ab60c8)
- **Exact claim supported:** the paper estimates local, instantaneous, a-posteriori BOS density uncertainty from displacement uncertainty and propagates it through density integration; sharp density changes can increase reconstructed uncertainty outside the sharp-gradient region, and boundary/integration choices matter. It is a 2D BOS density-integration study, not a proof for 3D BOST covariance.
- **N1.2 constraint:** retain spatial/camera displacement confidence, masks, and integration/geometry metadata in the flow-off/flow-on contract. Do not replace them with one global sigma; report whether a single high-gradient or low-confidence region drives a camera/block gate, and separately test propagated sensor uncertainty versus 3D forward-model mismatch.

### 12. Rajendran, Zhang, Bane, and Vlachos (2020), *Uncertainty-based weighted least squares density integration for background-oriented schlieren*

- **Type:** `[BOS]`
- **Official URL / DOI:** [OSTI official record](https://www.osti.gov/pages/biblio/1756077-uncertainty-based-weighted-least-squares-density-integration-background-oriented-schlieren), [DOI](https://doi.org/10.1007/s00348-020-03071-w)
- **Exact claim supported:** their BOS WLS integration assigns lower influence to higher-uncertainty gradient measurements and reports lower propagated density uncertainty than the Poisson baseline in its synthetic and spark-discharge experiments. This is not a result about learned priors, conformal coverage, or multiview BOST.
- **N1.2 constraint:** use only an independently estimated displacement/flow-off uncertainty to construct weights; add an IID/Poisson or unweighted classical control under matched calls. Treat weighted reconstruction as a deterministic baseline whose benefit must still pass raw-center no-harm, matched-classical, per-camera, and session audit gates.

### 13. Reichenzer, Schneider, and Herkommer (2021), *Improvement in systematic error in background-oriented schlieren results by using dynamic backgrounds*

- **Type:** `[BOS]`
- **Official URL / DOI:** [Springer open article](https://link.springer.com/article/10.1007/s00348-021-03285-6), [DOI](https://doi.org/10.1007/s00348-021-03285-6)
- **Exact claim supported:** changing reference backgrounds and aggregating a set of displacement estimates with a median improved the evaluated displacement field under the paper's artificial and experimental distortions. It does not say that arbitrary flow-off frames are IID, that their mean is unbiased, or that a median yields a covariance.
- **N1.2 constraint:** store raw, time-ordered flow-off/reference repeats and separately estimate a robust location/background correction and a covariance/scale. Freeze mean-versus-median/trim rule on fit data, keep the threshold-cal and audit repeats untouched, and add a repeated-background/dynamic-background control so that static pattern artefact is not misidentified as sample noise.

## E. Learned reconstruction safety boundary

### 14. Prémont-Schwarz, Vítků, and Feyereisl (2022), *A Simple Guard for Learned Optimizers*

- **Type:** `[GENERAL ML/OPTIMIZATION]`
- **Official URL:** [ICML/PMLR proceedings](https://proceedings.mlr.press/v162/premont-schwarz22a.html)
- **Exact claim supported:** learned optimizers can fail out of distribution; the paper studies a loss-guarded switch to a generic optimizer and proves convergence for its guarded optimization setting. It is not a BOST reconstruction theorem, and its loss guard is not automatically an observable field-safety certificate.
- **N1.2 constraint:** do not open a learned lambda/proximal/robust-weight controller until a deterministic covariance-aware baseline and an observable fail-closed fallback exist. The guard must read only deployment-visible quantities and must be evaluated against both the raw learned center and the strongest matched classical baseline; truth error, clean target, or audit labels cannot trigger the switch.

## N1.2 minimum literature-derived contract

1. **Data split and unit:** split by session/rig packet. Within any session, preserve ordered raw repeats; do not manufacture independent calibration examples by frame shuffle.
2. **Finite-sample quantile:** for 64 threshold-cal scores at 95%, freeze the 62nd order statistic. Candidate-specific score, mean, covariance, and threshold use disjoint fit/calibration/audit payloads.
3. **Coverage naming:** publish the exact event: score/discrepancy envelope, voxel point fraction, whole-volume set, or correction-safe decision. Include binomial uncertainty for empirical audit coverage and never call marginal coverage per-camera conditional coverage.
4. **Shift discipline:** a new session/geometry/background is a transfer audit unless exchangeability or a predeclared, validated weighting model is defensible. Coverage degradation is a result, not a threshold-tuning opportunity.
5. **Noise versus mismatch:** estimate sensor covariance from permitted flow-off/background data, but keep calibration drift, blur/ray error, and discretization/model mismatch as explicit separate stresses/branches.
6. **Stopping:** freeze global and per-camera whitened upper rules plus a lower/anti-over-regularization control; test sensitivity to noise-scale error and score field/H1 separately from residual fit.
7. **Classical floor:** compare unweighted/IID, diagonal, structured GLS/WLS, whitened CGLS/LSQR, and robust fixed data terms under matched accounting before learning a controller.
8. **Fail closed:** any learned component needs an observable guard and deterministic fallback, and must pass no-harm versus both raw learned center and strongest matched classical method on the locked audit.

## Explicit non-claims

- These papers do not prove that flow-off covariance is the right model for the OERF instrument.
- They do not identify the N1.1 `single_interface` failure mechanism.
- They do not authorize a real-BOST claim from the opened synthetic fixture.
- A conformal coverage result alone does not certify a reconstruction as data-consistent, physically correct, or safe to deploy.
