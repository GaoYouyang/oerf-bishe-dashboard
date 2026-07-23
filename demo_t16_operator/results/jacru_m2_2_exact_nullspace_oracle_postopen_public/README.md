# M2.2 exact dense null-space headroom oracle

Status: `M2_2_EXACT_NULLSPACE_HEADROOM_FOUND_ORACLE_ONLY`

This opened synthetic T0 experiment uses a full CPU float64 SVD on each frozen
12-cubed voxel operator. It is an oracle and has no runtime, efficiency,
deployment, real-BOST, or fresh-data claim. The dense setup is excluded from
the reconstruction call budget and reported separately. Numerical `ker(A)`
belongs only to the approximate inverse operator; it is not the true optical
null space. Zero-step source reproduction passed: `True`.
