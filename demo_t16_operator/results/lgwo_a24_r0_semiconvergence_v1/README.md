# LGWO-A24 R0 semiconvergence trajectory package

Status: **VALID_NO_GO_STOPPING_HEADROOM_OR_DIVERSITY_ABSENT_POSTOPEN**

This checksum-bound package records every `k=1..24` checkpoint of zero-correction,
fully reorthogonalized CGLS on public PSU ray geometry with deterministic analytic
reaction fields and synthetic noise. It asks whether instance-specific stopping has
enough headroom and checkpoint diversity to justify a later observable-policy screen.

- validation-selected global checkpoint: `k=24`
- R1 observable screen authorized: `false`
- neural model authorized by R0: `false`
- experimental PSU deflections used for scoring: `false`
- experimental volumetric truth available: `false`

`heldout_b_clean_oracle_k`, `truth_gradient_oracle_k`, and
`truth_field_oracle_k` leak evaluator truth and are ceilings, not deployable methods.
The discrepancy selector knows synthetic sensor sigma but no model-bias bound, so it
is not a validated OERF rule. All source splits and PSU-C1 results were opened before
this protocol; this package is descriptive rather than fresh or confirmatory.

Files:

- `trajectory_rows.csv`: all case/checkpoint metrics and exact call ledgers
- `selection_rows.csv`: eight preregistered selectors for each evaluation case
- `aggregate.csv`: split/method summaries and paired bootstrap intervals
- `summary.json`: gate decisions, provenance, environment, and claim boundary
- `diagnostic.png`: aggregate curves, oracle-k distribution, method gains, metric conflict
- `checksums.sha256`: integrity manifest over every other file in this directory
