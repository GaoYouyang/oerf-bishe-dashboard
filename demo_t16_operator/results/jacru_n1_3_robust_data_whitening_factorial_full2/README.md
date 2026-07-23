# JACRU N1.3 robust-data whitening development screen

- Status: `N1_3_ROBUST_DATA_WHITENING_DEVELOPMENT_NO_GO`
- The solver uses Huber measurement fidelity; the older Huber-PDHG reference uses quadratic measurement fidelity and Huber spatial regularization.
- A complete zero/estimated mean x unwhitened/diagonal/isotropic/structured whitening x quadratic/Huber data-loss factorial is reported.
- Sparse flow-on outliers are deterministic synthetic stress, not measured BOST.
- OOD was neither constructed nor evaluated by this runner.
- Dense whitened-norm setup is outside the matched solve budget and is not deployable.
- No algorithm, real-data, generalization, efficiency, or publication claim is authorized.
