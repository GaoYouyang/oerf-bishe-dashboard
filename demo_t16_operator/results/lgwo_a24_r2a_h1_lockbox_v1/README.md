# LGWO-A24 R2-A H1 clustered synthetic lockbox

Status: **VALID_R2A_SYNTHETIC_LOCKBOX_PARETO_SIGNAL_AUTHORIZE_HYBRID_COMPARISON_ONLY**

The candidate was frozen before these synthetic field, mask, and noise seeds were
generated: `lambda_h1=1e-3`, `lambda_l2=0`, `k=20`. It is compared with
same-budget fully reorthogonalized CGLS. Every phantom cluster retains four
mask/noise replicates and inference resamples clusters.

- clusters: `84`
- cases: `336`
- solver paths: `672`
- physical calls per path: `20F/20A^T`
- neural model authorized: `false`
- real BOST evidence: `false`

Even a passing status only establishes a bounded classical H1 synthetic baseline
signal and authorizes a later hybrid comparison. It is not a new algorithm, real-rig
generalization, NeRIF/TDBOST superiority, or a paper breakthrough.
