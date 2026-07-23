# GCT-KMix zero-start dominating upper-bound certificate

- Status: `GCT_KMIX_ZERO_START_FIELD_HEADROOM_UPPER_BOUND_NO_GO_LEARNER_NOT_AUTHORIZED`
- Development optimistic mean field-gain upper bound: `2.333430%`
- Public exploratory OOD optimistic mean field-gain upper bound: `1.964900%`
- Frozen field gate: `5.000000%` on both splits.
- Learner authorized: no.
- Breakthrough / real BOST / generalization evidence: no / no / no.

This certificate does not pretend the tail-safe `AlmostSolved` cases succeeded.
It uses only the fully `Solved` unconstrained simplex oracle. Because removing
all ray constraints enlarges the feasible set, its optimum is an optimistic
upper bound for every tail-safe mixture. The upper bound itself misses the
conjunctive field gate, so zero-start GCT-KMix cannot pass and no weight learner
should be trained on this representation.
