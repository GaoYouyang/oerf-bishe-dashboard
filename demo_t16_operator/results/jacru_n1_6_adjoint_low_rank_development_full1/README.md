# N1.6 adjoint-weighted low-rank post-open screen

Status: **POSTOPEN_NO_GO_NO_CONFIRMATION_ROUTE**.

This package is an opened synthetic development diagnostic.  It does not prove
an algorithm gain, real-BOST performance, OOD transfer, or novelty.

- Deployment candidate inputs: geometry, measured observation, low-CGLS warm
  field, and its low projection.
- Deployment candidate high-order calls: 0 forward / 0 adjoint.
- Matched production budget: 25 low forward / 24 low adjoint calls.
- PCA basis and coefficient targets use fit only; model selection uses
  calibration only.  Development was already open during N1.5 and can only
  generate a future hypothesis.
- Exact mismatch, measurement-L2 coefficient, and adjoint coefficient rows are
  evaluator-only ceilings.
- Selected route: `adjlr_camera_r4_tl21em02_a1e00_s1p0`.
- Opened development field gain versus matched low CGLS-24:
  0.035391.
- Opened development field gain versus component damping:
  -0.001841.
- Future confirmation gate passed: `False`.
