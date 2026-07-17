# JACRU-M2.7 target/no-harm Pareto ceiling

Status: `M2_7_TARGET_NO_HARM_PARETO_ORACLE_NO_GO`

This packet reuses the opened M2-T0 synthetic splits. The candidate runs
fixed-step CG only through the supplied forward/adjoint pair; it does not read
truth. Development selects one fixed variant/iteration per learned backbone,
and exploratory OOD is not used for that selection. Dense SVD is retrospective
toy-oracle evaluation only and is excluded from the algorithm and efficiency
claims. Finite CG is not an exact optical null-space projector.
