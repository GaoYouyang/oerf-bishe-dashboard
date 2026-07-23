# Metric-A v2 public analysis slice

**V2 NO AUTH.** This is a checksum-frozen analysis of a tiny synthetic signed-matrix interface smoke. It is not a new algorithm, a real BOST result, a generalization result, or a superiority claim.

## What was recomputed

- Four complete fresh geometry-OOD rigs and six fixed methods were read from the frozen source CSV files.
- Residual means, field-relative-L2 means, per-rig ranks, unsafe-rig counts, Schur violation counts, mass coverage, mass conservatism, and call/cost ledgers were recomputed.
- Mean summaries and per-rig outcomes are separate. The four rigs are not treated as IID samples; no confidence interval, p-value, or significance claim is produced.
- Provenance is `COMMITTED_CLEAN_REPRODUCIBLE_FROM_COMMIT`; the runner captured Git state before writing tracked outputs.

## Central result

- Calibrated envelope mean field gain vs factor: **-1.534e+07%** (negative means harm).
- Calibrated envelope mean field gain vs train-selected scalar-factor control: **-1.759e+07%** (negative means harm).
- Per-rig stable field wins against both controls: **2/4**.
- Calibrated envelope unsafe rigs: **4/4**, with **39** total Schur violations.
- Calibrated wins versus factor and scalar on **ood-00** and **ood-02**, but harms the scalar control on **ood-01 by 5.705e+07%** and **ood-03 by 732.17%** field error.
- Both source authorization decisions remain false.
- The selected exact-factor interpolation has `alpha=1` and is exactly identical to the exact oracle; it is a disclosed duplicate control, not an independent evidence unit.
- Learned and calibrated inference each consume the factor row/column features: 8 factor-vector accesses and 4 feature-construction calls over four fresh rigs. The shared synthetic bundle records 15 factor-majorizer materializations; no end-to-end cost reduction is claimed.

## Method means and safety

| Method | Mean residual | Mean field L2 | Unsafe rigs | Violations | Mass coverage | Median mass/exact |
|---|---:|---:|---:|---:|---:|---:|
| Factor | 0.181168 | 0.988963 | 0/4 | 0 | 100.0% | 2.71 |
| Exact oracle | 0.094946 | 0.703056 | 0/4 | 0 | 100.0% | 1 |
| Scalar x factor | 0.127546 | 0.86256 | 4/4 | 28 | 73.0% | 1.355 |
| Exact duplicate (alpha=1) | 0.094946 | 0.703056 | 0/4 | 0 | 100.0% | 1 |
| Raw learned | 2.28966e+26 | 2.18047e+26 | 4/4 | 68 | 35.0% | 0.7537 |
| Calibrated | 149389 | 151737 | 4/4 | 39 | 63.0% | 1.4 |

## Files

- `summary.json`: machine-readable evidence boundary and recomputed aggregate.
- `method_summary.csv`: method means, safety, mass, timing, and call accounting.
- `fresh_rig_comparison.csv`: long-form per-rig values and rankings.
- `decision_gates.csv`: passed and failed scientific gates.
- `diagnostic.png` / `diagnostic.pdf`: four-panel visual audit.
- `checksums.sha256`: hashes of every generated public artifact except the manifest itself.

## Interpretation boundary

The raw learned estimator diverges strongly and the calibrated envelope has one extreme fresh-rig failure, so both are isolated in a log-scale inset. The calibrated envelope is unsafe on every fresh rig, with per-rig violations [11, 18, 1, 9], and beats both deployable controls on only two of four rigs. It is a strict negative result, not authorization to substitute the metric.
