# JACRU-M2.9 fixed learned warm-start CGLS

Status: `M2_9_FIXED_WARM_CGLS_DEVELOPMENT_NO_GO`

This is an opened synthetic mechanism screen. The learned field is a fixed
warm start; every subsequent update uses the declared linear forward/adjoint
pair. Truth is used only by the evaluator. No fresh, real BOST, deployment,
method-superiority, or breakthrough claim is authorized by this packet.

## Frozen development selections

| Method | Development selection | OOD pass |
|---|---:|---:|
| jacru_m2 | none | false |
| pooled_cnn | none | false |
| grid_deeponet | none | false |
| pooled_fno | none | false |

The retrospective dense SVD only audits preservation of the discrete
support-restricted kernel. It is not part of the reconstruction algorithm
or efficiency accounting and is not the true optical null space.
