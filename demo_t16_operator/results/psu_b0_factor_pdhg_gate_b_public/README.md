# Deterministic factor-PDHG Gate B public audit

Status: `GATE_B_E2_MECHANISM_NO_GO`

This package is a sanitized, aggregated view of the preregistered deterministic
Gate B. It contains no private filesystem path, raw detector geometry, ray
indices, reconstruction volume, restricted paper, credential, or experimental
OERF material.

## What was compared

Sixteen opened development samples cover two replicates and eight analytic
reaction-field families. All methods use the same real PSU detector geometry
and synthetic correlated noise. At each checkpoint `K = 4, 8, 16, 32`, the
three PDHG variants use exactly `K` signed forward and `K` signed adjoint calls:

- scalar A-only PDHG;
- view-block A-only PDHG;
- voxel-factor A-only PDHG;
- same-run, single-sample, exact-`K` graph PCGLS control.

The factor methods share the same 2,322-dimensional A-coupled gauge. Another
422 support voxels are A-null and are embedded as zero. The graph control uses
its declared full-support Sobolev representation; this asymmetry is disclosed
and no superiority claim is made.

## Result

At `K=32`, mean field relative-L2 is:

| Method | Mean field relative-L2 |
|---|---:|
| graph PCGLS | 0.421042 |
| scalar A-only PDHG | 0.996031 |
| view-block A-only PDHG | 0.995241 |
| voxel-factor A-only PDHG | 0.982876 |

Voxel-factor PDHG improves scalar PDHG on 15/16 samples, but the mean reduction
is only 1.321%, far below the frozen 25% mechanism threshold. Its reduction
against view-block PDHG is 1.242%, below the 3% attribution threshold, and its
mean error gap against graph PCGLS is 133.439%, above the 20% limit. Five of
eight gates pass, so the final decision remains NO-GO. Neural training and any
algorithm-superiority claim remain unauthorized.

## Files

- `summary.json`: sanitized scope, decision, connectivity, and headline metrics;
- `method_frontier.csv`: four methods at four exact-call checkpoints;
- `paired_k32_gains.csv`: 16 paired K=32 field-error comparisons;
- `decision_gates.csv`: all eight preregistered gates;
- `factor_gate_b_no_go.png/.pdf`: publication-style four-panel audit;
- `checksums.sha256`: hashes of every public artifact above.

Rebuild from the independently validated formal bundle:

```bash
.venv/bin/python site_tools/analyze_psu_b0_factor_gate_b_no_go.py
.venv/bin/python -m pytest site_tools/test_analyze_psu_b0_factor_gate_b_no_go.py -q
```

This is an opened synthetic mechanism result. It is not a fresh-seed result,
not a real-flow reconstruction, and not evidence that PDHG or diagonal
preconditioning is ineffective in general.
