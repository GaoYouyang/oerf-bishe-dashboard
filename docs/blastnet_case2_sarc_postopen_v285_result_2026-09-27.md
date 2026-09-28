# v285: Same-Family H2-Air Transfer Fails the Joint Accuracy Gate

**2026-09-27 · Frozen SARC-K3-M4 · 54-frame post-open replication within the H2-air family**

## Verdict

`FAIL_CASE2_SAME_FAMILY_POSTOPEN_TRANSFER_ACCURACY_V285`. The frozen SARC-K3-M4 candidate passes the Direct-K3 observation guard and is not Pareto-dominated by Direct-K3, but the joint accuracy gate passes **0/54 frames**. The candidate does not satisfy the 1.01x Direct-K4 component bound and is no worse than Zero-K4 on all required metrics on only 1/54 frames. No resource test was run.

This is a **same-family, post-open configuration replication**. The configuration had not been opened before this frozen protocol, but it belongs to the already studied H2-air dataset family. It is not a new independent dataset family, an untouched confirmatory test, cross-family generalization, or real experimental BOST.

## Independent recomputation

A separate implementation replayed 216 curved-forward outputs across the 54 frames and four methods. It independently reproduced the formal metric table with a maximum absolute metric difference of `2.50e-16`. The sealed field outputs and prediction barrier were unchanged. The validation chain is numerically valid; the scientific accuracy gate still fails.

## Where the tradeoff appears

A descriptive, post-open breakdown of the sealed scores finds:

| Comparison | Frames with lower candidate error |
|---|---:|
| Field vs Direct-K4 | 54/54 |
| Gradient vs Direct-K4 | 54/54 |
| Observation vs Direct-K4 | 0/54 |
| Field vs Zero-K4 | 54/54 |
| Gradient vs Zero-K4 | 51/54 |
| Observation vs Zero-K4 | 1/54 |
| Field / gradient / observation all lower vs Direct-K3 | 54/54 |

Against Direct-K4, the candidate's frame-wise observation-error ratio has median `1.18921`, linear p90 `1.33596`, and maximum `1.48144`. These descriptive quantiles were independently recomputed from the sealed 54-row report on 2026-09-28. The candidate therefore improves truth-scored field and gradient errors while consistently worsening measurement fit relative to the stronger K4 controls. The comparison to Direct-K3 is favorable on these three metrics, but that does not repair the frozen joint gate against Direct-K4 and Zero-K4.

This breakdown is post hoc: it localizes the observed tradeoff in this opened configuration but does not establish causality or authorize a blend, threshold, or safeguard. The frozen candidate remains closed; do not tune it against these scores.

## Interpretation and limits

- Same-family replication: 0/54 frames pass the joint accuracy gate.
- The principal observed conflict is improved field/gradient error alongside worse observation consistency against K4.
- No wall-time, peak-RSS, operator-call reduction, or deployment advantage was measured.
- The result does not establish failure on every external reacting-flow family; it narrows the evidence for this fixed method on the tested H2-air conditions.
- It does not change the separate PoolFire v284 cost verdict or merge v284 operator-call totals with SARC resource measurements.
- `algorithm_breakthrough=false`, `paper_success=false`, `independent_external_generalization=false`, and `real_bost=false`.

## Reproducibility boundary

This public report contains only the scientific summary and independent numerical agreement. It excludes source fields, raw arrays, checkpoints, model parameters, private execution paths, and private hashes. The dataset identity and preprocessing contract remain in the private sealed record.
