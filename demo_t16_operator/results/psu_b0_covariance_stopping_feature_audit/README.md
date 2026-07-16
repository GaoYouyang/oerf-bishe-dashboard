# Observable PCGLS trajectory feature audit

Status: `POSTOPEN_OBSERVABLE_PCGLS_TRAJECTORY_EXTRACTED`

This directory contains 512 stage rows from 16 already opened replicates,
eight reaction morphologies, and graph-whitened PCGLS checkpoints 2 through 5.
The fair reference is component-IID PCGLS with Sobolev strength 3 and four
stages.

Candidate stopping rules may use only the declared observable columns:
whitened residual objective, marginal residual reduction, PCGLS alpha, previous
beta, relative volume update, and gradient-to-field norm. Ground-truth field,
gradient, and front gains are diagnostic labels only.

No fresh seeds, experimental truth, or real temporal flow-off repeats are used.
Any rule selected from these rows remains post-open development and must pass a
separate fresh protocol before supporting a scientific claim.
