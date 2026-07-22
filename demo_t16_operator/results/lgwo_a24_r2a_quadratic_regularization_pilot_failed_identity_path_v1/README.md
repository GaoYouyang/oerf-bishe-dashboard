# R2-A pilot fail-closed identity-path record

Status: **INVALID_R2A_PILOT_FAIL_CLOSED**

The first frozen invocation stopped before geometry loading because the configured
identity-manifest path did not exist. It produced zero trajectory or scientific rows.
The recorded source commit is `ebe5415acf5a53a8507be12b74e2c55e5aed4be8`.

The only subsequent amendment corrected the path to the public identity manifest
already used by R0. No method, lambda, checkpoint, split, metric, or gate changed.
