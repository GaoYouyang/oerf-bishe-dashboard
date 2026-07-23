# JACRU N1.4 adaptive-edge warm robust development screen

- Status: `N1_4_ADAPTIVE_EDGE_WARM_ROBUST_DEVELOPMENT_NO_GO`
- Every deployed candidate is costed as CGLS-12 plus robust PDHG-12, or a zero-start 24-pair control.
- Edge weights read only the warm-start gradient; they never read truth.
- Truth and clean projections are absent from reconstruction but are used by the opened-development diagnostic gates.
- OOD was neither constructed nor evaluated by this runner.
- Development-grid warm starts are cached for runtime only; their 12F/12AT cost remains in every candidate ledger.
- No algorithm, real-data, generalization, efficiency, or publication claim is authorized.
