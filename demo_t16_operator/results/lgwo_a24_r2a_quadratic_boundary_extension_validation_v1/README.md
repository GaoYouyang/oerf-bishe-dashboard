# R2-A quadratic pilot independent validation

Status: **VALID**

The validator independently checks the checksum manifest, source bindings, complete
24 x 28 x 6 rectangle, finite values, call ledgers, residual/objective identities,
all aggregate arithmetic, CGLS Pareto relations, pilot-signal rules, paired case
gains, bootstrap intervals, and tail harm.

It does not rerun the solver and cannot recover checkpoint volumes because the pilot
package saves metrics rather than fields. A valid result remains a post-open
validation-only scale signal, not an algorithm or real-BOST result.
