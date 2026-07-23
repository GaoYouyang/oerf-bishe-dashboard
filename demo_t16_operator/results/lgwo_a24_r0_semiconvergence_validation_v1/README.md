# Independent LGWO-A24 R0 validation

Status: **VALID**. Recomputed experiment status: `VALID_NO_GO_STOPPING_HEADROOM_OR_DIVERSITY_ABSENT_POSTOPEN`.

This validator does not import the R0 runner. It verifies the sealed file set and hashes, independently checks the frozen config and reconstructs deterministic truth, masks, sigma, noise, observations, clean projections, and operator mismatch, then binds metric code to the source commit, reselects every case checkpoint, recomputes paired gains and 20,000-draw bootstrap intervals, then reconstructs the gate decision and claim boundary. Full checkpoint volumes are not duplicated into this package, so low-level field metrics are code/hash-bound rather than recomputed from saved volumes.
