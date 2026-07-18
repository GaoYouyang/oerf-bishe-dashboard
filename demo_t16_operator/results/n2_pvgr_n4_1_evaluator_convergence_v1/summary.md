# N2-PVGR N4.1 evaluator convergence audit

## Machine decision

`FAIL_CLOSED_EVALUATOR_REMAINS_UNAUTHORIZED`

## Execution amendment

N4.1 inherits the complete frozen N4 v1 scientific protocol. It changes only the control-flow order used to request H2048 after the complete H1024 gate bundle has been computed. No v1 checkpoint was reused.

## Counts

- H1024 pass: 23 / 32
- H2048 escalations: 9 / 32
- Final cellwise references authorized: 30 / 32
- Uniform H1024 reference authorized: false
- Tiny field-JVP/VJP gate authorized: false

## Claim boundary

This is a post-N3 selected synthetic evaluator audit with a disclosed execution amendment. It cannot establish algorithm superiority, real-BOST validity, three-dimensional reconstruction quality, novelty, or generalization.

- Artifact recovery: attested opaque-checkpoint reuse; only the Matplotlib bar x input changed from a dict to its list of keys.
