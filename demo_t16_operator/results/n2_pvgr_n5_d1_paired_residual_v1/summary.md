# N2-PVGR N5-D1 paired residual accumulation audit

Machine decision: `D1_ACCUMULATION_ORDER_TOO_SMALL_TO_EXPLAIN_N4_FLOOR`

This is a post-N4 selected numerical-mechanism audit. It does not authorize a reference, field adjoint, reconstruction, neural training, real-data or paper claim.

- Selected cells: 4 (two N4.1 failures plus matched wide controls)
- Numerical levels: 8 (H1024/H2048)
- Maximum frozen-route equivalence relative-L2: 1.998829e-12
- Maximum parent-N4 adjacent-metric replay difference: 2.277911e-13
- Failed-cell accumulation/refinement fractions: [1.2703965352324395e-09, 5.190933176769265e-10]
- Paired logical queries: 132,120,576
- Extra equivalence-audit queries: 44,040,192

The next allowed action is only the separately preregistered D2 H4096/H8192 truncation probe when the D1 contract passes.
