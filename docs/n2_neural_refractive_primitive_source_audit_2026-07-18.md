# N2 neural-refractive-primitive source audit

## Why the candidate changed

The 2026 paper *Neural Refractive Index Primitives for Flame Field
Reconstruction Using Background-Oriented Schlieren* closes several broad
novelty claims that would otherwise look attractive. It already uses a single
neural refractive-index primitive, smoothstep hash interpolation, automatic,
central-discrete, and combined gradient formulations, masks, and hierarchical
path sampling. The paper reports both synthetic combustion phantoms and flame
experiments.

Primary sources:

- [2026 paper: arXiv HTML](https://arxiv.org/html/2605.11454)
- [2026 paper: publisher DOI](https://doi.org/10.1016/j.combustflame.2026.115082)
- [2026 author repository](https://github.com/Weihu22/Neural-Implicit-Reconstruction-for-BOS)
- [NeRIF: arXiv](https://arxiv.org/abs/2409.14722)
- [NeRIF: publisher DOI](https://doi.org/10.1063/5.0250899)
- [NeDF: arXiv](https://arxiv.org/abs/2409.19971)
- [NeDF: publisher DOI](https://doi.org/10.1063/5.0241191)

## What is already covered

1. **NeRIF (2024/2025)** represents refractive index and a gradient head, uses
   ray-path sampling that grows from 60 to 200 points, and combines losses from
   a direct gradient prediction and a central-difference gradient of the index.
2. **NeDF (2024)** directly represents the density-gradient/deflection field and
   uses coarse-to-fine hierarchical ray sampling to focus queries near thin
   flame fronts. It evaluates RMSE, PSNR, and SSIM under sparse and limited
   views.
3. **The 2026 primitive paper** makes refractive index the sole primitive. It
   explicitly compares automatic, central-discrete, and combined gradient
   formulations. It uses smoothstep interpolation because ordinary linear hash
   interpolation has zero second spatial derivative inside a cell, which is
   unsuitable for the automatic-gradient training graph.
4. **Classical multi-level and control-variate work** already covers residual
   telescoping, cost allocation, learned regression controls, and neural
   controls. Therefore `automatic + discrete`, `multi-fidelity`, or `residual
   correction` alone is not a publishable novelty claim.

Required method-family anchors:

- [MLMC overview and primary papers](https://people.maths.ox.ac.uk/gilesm/mlmc.html)
- [Multi-index Monte Carlo](https://arxiv.org/abs/1405.3757)
- [Multifidelity Monte Carlo](https://doi.org/10.1137/15M1046472)
- [Regression-based Monte Carlo](https://arxiv.org/abs/2211.07422)
- [Primary-space adaptive control variates](https://arxiv.org/abs/2008.06722)
- [Neural control variates](https://arxiv.org/abs/2006.01524)
- [Automatic-integration neural control variates](https://arxiv.org/abs/2409.15394)
- [StackMC](https://arxiv.org/abs/1606.02261)

## Narrow contribution that remains testable

The remaining candidate is not “a new gradient formula.” It is a BOST-specific
**coupled aperture-path residual estimator and derivative contract**:

1. define the target measure jointly over pupil position, pixel
   footprint/point-spread response, and path location;
2. pair low and high renderers with the same random optical state inside each
   residual, while keeping low-only and residual pools independent;
3. allocate queries jointly across pupil and path fidelity;
4. reuse the exact state for forward, JVP, and VJP, including trajectory
   sensitivity when rays depend on the reconstructed index field;
5. fail closed when a perturbation changes mask, frustum, support, or front
   topology; and
6. report matched-cost bias, variance, field error, held-out reprojection,
   per-rig tails, and physical endpoints rather than a synthetic mean alone.

`N2-ADRC-N1` tests only the first statistical mechanism on a prescribed-path
surrogate. It does not yet implement the joint optical target, field-dependent
ray ODE, neural primitive, or reconstruction.

## Prohibited claims

Do not write any of the following until separate evidence exists:

- first use of automatic and central-discrete gradients;
- first multi-fidelity or residual estimator for BOST;
- unbiased squared loss merely because the forward estimator is unbiased;
- physical curved-ray differentiation from the current fixed `bend` parameter;
- superiority over NeRIF, NeDF, or the 2026 primitive method;
- real-flame, out-of-distribution, or generalization success from analytic
  morphology proxies;
- temperature accuracy without a declared composition/Gladstone-Dale model and
  an independent physical endpoint.

## Clean-room and licensing boundary

The author repository was inspected locally at commit
`a385cce83d88df24ed05dccfd6fde20e124f5604`. No repository-level license file
was found during the audit. Public access is not permission to redistribute or
derive code. We may cite the paper, link the repository, inspect behavior, and
write an independent clean-room implementation. We must not copy its Python,
MATLAB, CUDA, data, or glue code into this public repository unless the authors
provide an applicable license or written permission.

The current smoothstep-grid mechanism implementation was written independently
and intentionally does not reproduce the external network, renderer, data, or
training pipeline.
