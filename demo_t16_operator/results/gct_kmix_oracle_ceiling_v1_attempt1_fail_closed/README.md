# GCT-KMix O1 result packet

- Status: `GCT_KMIX_FAIL_CLOSED_SOLVER_OR_CONTRACT_FAILURE`
- Evidence: 16^3 JACRU synthetic truth-oracle representation ceiling only.
- Breakthrough: no.
- Learner trained/authorized: no/no.
- Real or confirmatory BOST evidence: no/no.
- Cases/clusters: `18` cases in `6` geometry clusters; three families sharing a geometry are not independent rigs.
- Source commit: `88e4364cd941c3c066342ba1f6aac067e945453d`

`summary.json` contains the fail-closed gate. `case_metrics.csv` keeps all scalar
baselines, `solver_diagnostics.csv` exposes conic primal/dual evidence, and
`oracle_weights.csv` contains evaluator-only truth-conditioned weights. No raw
truth, observation, geometry, field, residual, checkpoint or private material
is included.
