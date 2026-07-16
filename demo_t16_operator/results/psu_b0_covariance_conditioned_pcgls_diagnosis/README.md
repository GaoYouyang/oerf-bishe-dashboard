# Covariance-conditioned PCGLS opened-seed diagnosis

Status: `POSTOPEN_COVARIANCE_CONDITIONING_DIAGNOSIS_COMPLETE`

This is a causal diagnosis on 16 already opened synthetic replicates using the
real PSU detector geometry. It is not confirmatory evidence and does not use
experimental flow-field truth or real temporal flow-off repeats.

The frozen screen compared 120 combinations of detector-spatial covariance
tempering, static Sobolev strength, and PCGLS depth. The original declared
selection rule chose `full_graph_s3_k4`. It passed the selection and opened
diagnostic halves against the legacy `component_s5_k4` reference.

That total comparison is confounded by solver retuning: `component_s3_k4`
already supplies most of the approximately 25% gain. Relative to that stronger
same-depth baseline, `full_graph_s3_k4` improves the pooled mean by about 1.41%
but has a worst paired field regression of about 7.92%. The result therefore
authorizes a fresh preregistration only under the original gates; it does not
yet authorize a robust covariance-conditioned reconstruction claim.

The next step is an opened-data-only audit of observable PCGLS trajectory
features and tail-safe stopping or fallback. A fresh seed set must remain
unopened until that rule and its stronger fair-baseline gates are committed.
