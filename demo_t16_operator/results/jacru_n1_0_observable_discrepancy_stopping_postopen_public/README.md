# JACRU N1.0 observable discrepancy stopping

Status: `N1_0_OBSERVABLE_DISCREPANCY_STOPPING_NO_GO`

This post-open ceiling reuses the complete M2.7 K=0..10 trajectory.  Every
selector reads only measured residuals, the prepared CGLS-12 residual, or the
PCG system-residual fraction.  Field truth and the independent clean renderer
are evaluation-only.  A missed threshold returns the already prepared CGLS-12
base after charging the full attempted trajectory budget.

The simulator noise floor (`0.02236068`) is derived from frozen nuisance
parameters, not from real flow-off repeats.  The exact camera-block setup is a
1001-forward-equivalent oracle excluded from the call budget.  Therefore this
packet cannot establish deployment, runtime superiority, fresh performance, or
real BOST generalization even if an opened candidate passes.
