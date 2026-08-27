# v276 Case 19 TV basis-pursuit reference: execution note

Date: 2026-08-28

## Chinese

### Why this run was opened

v275 left Case 19 without an adequate and reproducible reference. Before reading any new result, v276 froze one physically distinct classical reference: equality-constrained isotropic three-dimensional total variation with fixed zero-boundary and zero-mean conventions, a fixed iteration budget, no regularization-parameter search, and no truth-guided early stopping. A full-size synthetic audit made the two implementations agree to numerical precision, so one formal execution was authorized.

### What actually happened

The single formal execution used ten CPU worker threads. At least one frozen camera rig did not satisfy the preregistered row-scaled equality-replay limit at the fixed stopping point. The process therefore failed closed **before** `PREDICTION_READY`, formal metric arrays, pass counts, or `READY` were generated.

No field, gradient, observation, matched-accuracy, or exact-call result exists for v276. The independent scientific validator was not started because there was no valid formal result to validate.

### Decision boundary

The authoritative status is `INCONCLUSIVE_INVALID_CASE19_TV_PDHG_REFERENCE_V276`. This closes only this fixed TV-PDHG execution. It will not be rerun with more iterations, a looser equality limit, different steps, a different stencil, or changed boundary handling. This is a numerical execution failure, not a scientific negative result for total variation and not an algorithmic breakthrough.

v275 remains the latest independently evaluated scientific evidence. There is still no established matched-accuracy improvement, effective exact-call reduction, wall/RSS advantage, external generalization, curved-ray validation, or real-BOST result. `algorithm_breakthrough=false`.

## English

### Why this run was opened

v275 left Case 19 without an adequate and reproducible reference. Before reading any new result, v276 froze one physically distinct classical reference: equality-constrained isotropic three-dimensional total variation with fixed zero-boundary and zero-mean conventions, a fixed iteration budget, no regularization-parameter search, and no truth-guided early stopping. A full-size synthetic audit made the two implementations agree to numerical precision, so one formal execution was authorized.

### What actually happened

The single formal execution used ten CPU worker threads. At least one frozen camera rig did not satisfy the preregistered row-scaled equality-replay limit at the fixed stopping point. The process therefore failed closed **before** `PREDICTION_READY`, formal metric arrays, pass counts, or `READY` were generated.

No field, gradient, observation, matched-accuracy, or exact-call result exists for v276. The independent scientific validator was not started because there was no valid formal result to validate.

### Decision boundary

The authoritative status is `INCONCLUSIVE_INVALID_CASE19_TV_PDHG_REFERENCE_V276`. This closes only this fixed TV-PDHG execution. It will not be rerun with more iterations, a looser equality limit, different steps, a different stencil, or changed boundary handling. This is a numerical execution failure, not a scientific negative result for total variation and not an algorithmic breakthrough.

v275 remains the latest independently evaluated scientific evidence. There is still no established matched-accuracy improvement, effective exact-call reduction, wall/RSS advantage, external generalization, curved-ray validation, or real-BOST result. `algorithm_breakthrough=false`.
