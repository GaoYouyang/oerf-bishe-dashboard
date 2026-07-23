# LGWO-A24 R0 post-open regularization-conflict audit

Status: **VALID_POSTOPEN_SYNTHETIC_FIELD_GRADIENT_PATH_CONFLICT**

This package re-reads the checksum-bound R0 trajectory and independently aggregates
the field, gradient, and front metrics. It does not run a solver or train a model.

- IID: mean field best `k=24`; mean gradient best `k=1`
- family OOD: mean field best `k=24`; mean gradient best `k=1`
- all 24 mean checkpoints are field/gradient Pareto-efficient on both primary splits
- learned early stopping is not authorized: the field oracle supplies no useful early-stop labels

The finding is narrower than algorithm success. It says that choosing among the
existing checkpoints cannot remove the observed field/gradient trade-off. A later
candidate must change the reconstruction path or regularizer and must be compared
under the same forward/adjoint-call budget.

Files:

- `curve.csv`: recomputed split-by-checkpoint means, standard deviations, and changes from `k=1`
- `discrepancy_selector.csv`: copied-and-typed R0 discrepancy summaries used for the observable warning
- `summary.json`: split mechanisms, primary conflict gate, provenance, and claim boundary
- `diagnostic.png` / `diagnostic.pdf`: publication-style field/gradient/front trajectory view
- `checksums.sha256`: integrity manifest over every other file in this directory
