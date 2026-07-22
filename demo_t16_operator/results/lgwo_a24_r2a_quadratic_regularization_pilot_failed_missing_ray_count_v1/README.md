# R2-A pilot fail-closed weighted-operator interface record

Status: **INVALID_R2A_PILOT_FAIL_CLOSED**

The second frozen invocation completed geometry and synthetic split construction,
then stopped before the first CDLS trajectory because `SingleCaseWeightedOperator`
did not expose the base operator's already-known `ray_count`. It produced zero
trajectory or scientific rows. The source commit was
`898fd160e8b9af47a1199e0d977c54a5360223b3`.

The interface amendment only publishes that validated integer and adds a regression
assertion. No method, lambda, checkpoint, split, metric, or gate changed.
