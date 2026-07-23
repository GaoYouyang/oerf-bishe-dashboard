# T16: physics-lift residual neural operator for 3D BOST

This is a runnable undergraduate-scale closure test for the proposed T16
direction. It asks whether a shared neural model can map a coarse BOST physics
lift to a 3D refractive-index-like field across unseen samples and conditions.

It is not a reproduction of NeRIF or TDBOST, and it does not claim performance
on OERF experimental data.

## Implemented pipeline

```text
synthetic 3D field
  -> one shared linear BOST forward operator
  -> sparse noisy view selection
  -> filtered-backprojection stack
  -> one train-only global affine calibration
  -> residual 3D U-Net or official NeuralOperator 2.0 FNO
  -> field metrics + observed/held-out reprojection + mass/centroid audit
```

Both neural models receive the same seven channels:

1. calibrated physics lift;
2. support window;
3. view-count fraction;
4. normalized noise metadata;
5. z coordinate;
6. y coordinate;
7. x coordinate.

Both predict a residual that is added to the calibrated lift. The FNO uses the
official `neuralop.models.FNO` implementation; the comparison model is a small
3D U-Net.

## Run

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r demo_t16_operator/requirements.txt
.venv/bin/python demo_t16_operator/run_benchmark.py
```

Run the smoke tests:

```bash
.venv/bin/python -m pytest -q demo_t16_operator/test_smoke.py
```

Run the three-seed causal ablation:

```bash
.venv/bin/python demo_t16_operator/run_ablations.py --device cpu
```

Run the three-seed reliability-gate screen:

```bash
.venv/bin/python demo_t16_operator/run_reliability_gates.py --device cpu
```

Run the support/query dual-branch prototype:

```bash
.venv/bin/python demo_t16_operator/run_dual_branch_query.py --device cpu
```

Run the independent-expert, nullspace, and query-calibration audit chain:

```bash
.venv/bin/python demo_t16_operator/run_dual_branch_query.py \
  --config demo_t16_operator/configs/independent_dual_support.json
.venv/bin/python demo_t16_operator/run_nullspace_identifiability_audit.py
.venv/bin/python demo_t16_operator/run_independent_nullspace_headroom.py
.venv/bin/python demo_t16_operator/run_support_nullspace_corrector.py
.venv/bin/python demo_t16_operator/run_query_calibrated_nullspace.py

.venv/bin/python demo_t16_operator/validate_independent_dual_results.py
.venv/bin/python demo_t16_operator/validate_nullspace_results.py
.venv/bin/python demo_t16_operator/validate_support_nullspace_results.py
.venv/bin/python demo_t16_operator/validate_query_calibrated_results.py
```

Run the training-mask-matched camera-budget direct-operator pilot:

```bash
.venv/bin/python -m demo_t16_operator.run_direct_operator_pilot --device mps
.venv/bin/python -m pytest demo_t16_operator/test_direct_operator.py -q
.venv/bin/python -m demo_t16_operator.validate_direct_operator_results
```

This v3a experiment rebuilds observations and lifts for fixed K=4/6/8 camera
masks, locks camera index 3 as Q_audit, trains one model per budget, and compares
six equal-budget paths: train-calibrated FBP-style lift, validation-tuned ridge,
lift-residual U-Net/FNO, and ridge-initialized residual U-Net/FNO.

The 15-channel neural input contains the reconstruction field, support, view
fraction, nine camera-identity channels, and z/y/x coordinates. Synthetic noise
is regenerated from the current K-view RMS; the simulation noise label is not a
network input. Test truth and Q_audit cannot change the reconstruction input.

The current development result uses 256 source fields, including 96 independent
test fields across IID, noise OOD, thin-front family OOD, and joint OOD. Three
model seeds are collapsed inside each field before 20,000 stratified bootstrap
replicates. Ridge-initialized residual FNO is the only candidate that passes all
three development gates:

| K | mean gain vs locked ridge | 95% cluster CI | p10 | harm >1% | Q_audit residual gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 21.54% | [19.11%, 23.97%] | 4.27% | 0.00% | 14.63% |
| 6 | 19.68% | [17.22%, 22.09%] | 3.19% | 0.00% | 11.26% |
| 8 | 16.91% | [14.61%, 19.21%] | 0.69% | 4.17% | 10.01% |

This is still a linear `8 x 16 x 16` development pilot. It authorizes a
ridge-FNO-to-NeRIF warm-start experiment; it does not establish real BOST
performance or a publishable method. The next confirmation must use new locked
fields, an independent forward model, CGLS/TV/RBF baselines, and end-to-end NeRIF
time-to-quality.

Run the v3b provisional own-algorithm benchmark:

```bash
.venv/bin/python -m demo_t16_operator.run_own_algorithm_benchmark --device mps
.venv/bin/python -m pytest demo_t16_operator/test_own_algorithm.py -q
.venv/bin/python -m demo_t16_operator.validate_own_algorithm_results
```

v3b gives the same 42-channel information contract to every neural method:
ridge anchor, support, view fraction, nine camera masks, z/y/x, nine independent
ray backprojections, and active `sin(theta)/cos(theta)` channels. It compares a
94,193-parameter U-Net, 44,203-parameter FNO, 49,313-parameter grid DeepONet,
and the 45,973-parameter provisional ray-set augmented residual operator under
the same splits, 24 epochs, optimizer, loss, and three model seeds.

The validation-locked neural comparator is FNO for every budget. Relative to
that comparator, the three-seed field-level result is:

| K | mean superiority | 95% cluster CI | p10 | harm >1% | clean Q_audit superiority | gate |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4 | 1.77% | [1.14%, 2.43%] | -0.74% | 7.29% | 0.13% | fail: effect below 2% CI gate |
| 6 | 5.36% | [4.49%, 6.25%] | -0.72% | 5.21% | 1.61% | development pass |
| 8 | 1.09% | [0.57%, 1.64%] | -1.02% | 12.50% | -0.76% | fail: tail harm |

All three K=6 seed means are positive, while K=4 and K=8 each have one negative
seed. The model mainly helps IID/noise OOD and does not yet improve thin-front
family/joint OOD consistently. The name and architecture are provisional; these
inspected development fields cannot support a novelty or superiority claim.
The next v3c experiment must use new dev2 fields and checksum-sealed blind
fields, with a zero-initialized set adapter that begins exactly at the FNO
prediction and learns only a budget-aware correction.

Run the v3c architecture and data-protocol gate:

```bash
.venv/bin/python -m demo_t16_operator.run_v3c_protocol_gate
.venv/bin/python -m demo_t16_operator.validate_v3c_protocol_gate
```

The gate creates a new 328-field dev2 dataset with zero sample-seed overlap
against v3b. A frozen 44,203-parameter FNO is wrapped by a 4,988-trainable-
parameter ray-set adapter, for 49,191 total parameters. The adapter head starts
at exact zero: the maximum initial prediction difference from the base FNO and
the maximum frozen-base parameter drift after three optimizer steps are both
`0.0`; the adapter head receives a nonzero first-step gradient. This proves an
engineering fallback property, not reconstruction superiority.

`configs/v3c_blind_preregistration.json` locks field counts, primary endpoint,
baselines, gates and unseal conditions, but deliberately contains no final
seed. Its status remains `NOT_SEALED_NO_CONFIRMATORY_CLAIM_ALLOWED` until the
mentor approves geometry and the dev2 analysis is frozen.

Run the three-seed K=6 dev2 continuation control:

```bash
.venv/bin/python -m demo_t16_operator.run_v3c_k6_dev2_pilot --device mps
.venv/bin/python -m demo_t16_operator.validate_v3c_k6_dev2_results
```

Each seed first trains the same ridge-FNO for 24 epochs. The selected checkpoint
then branches into a full continued FNO and a frozen-FNO zero-init adapter. Both
receive the same additional 12 epochs, data order, optimizer settings, losses
and validation rule. The comparison matches additional epochs, not FLOPs.

The current adapter fails its dev2 gate:

| paired comparison | mean field superiority | 95% field-cluster CI | p10 | harm >1% |
| --- | ---: | ---: | ---: | ---: |
| adapter vs continued FNO | -4.965% | [-5.569%, -4.397%] | -12.143% | 74.219% |
| adapter vs base FNO | +1.425% | [1.281%, 1.570%] | +0.351% | 0% |
| continued FNO vs base FNO | +5.846% | [5.342%, 6.353%] | +1.497% | 0% |

All three adapter-vs-continued seed means and all four domain means are
negative. The adapter also costs about 7.18 times the additional training time
and 6.46 times the inference time because it applies a 3D encoder per view.
This stops the current frozen per-view adapter and exposes that the 24-epoch FNO
baseline was not saturated. A subsequent prior-art audit identified F-Adapter as
the closest PEFT baseline, while R2-FFNO and MG-TFNO already cover low-rank or
tensorized spectral kernels. The next candidate, if pursued, must compare LoRA,
vanilla/F-Adapter, matched last-block fine-tuning and acquisition-geometry-
conditioned adaptation, including shuffled/constant/static geometry controls.
Blind final remains closed.

## v3d FNO validation-plateau audit

Before any new PEFT or geometry-conditioned branch is trained, the v3d audit
checks whether the K=6 ridge-FNO baseline has reached a validation plateau:

```bash
python -m demo_t16_operator.run_v3d_fno_saturation_audit --device mps
python -m demo_t16_operator.validate_v3d_fno_saturation_results
```

The protocol preserves the v3c 24-epoch base schedule, then continues in locked
12-epoch blocks. A block counts as plateau-like only when mean seed validation
improvement is at most 0.5% and every seed improves by at most 1.0%; the final
two blocks must both pass. Validation is sample-weighted so each 3D field has
equal weight. Reused dev2 fields and clean Q_audit are evaluated only after this
run's validation-only decision; they are not a fresh project-level audit.

The complete three-seed audit did **not** reach plateau by 96 epochs. Mean
validation L2 changed from 0.16646 to 0.11746, while the final block still
improved by 2.81% on average and 2.96% for the strongest-improving seed. Reused dev2
field L2 improved by 16.53% relative to epoch 24, but this is diagnostic evidence,
not a checkpoint-selection signal. LoRA, F-Adapter and the proposed acquisition-
geometry branch remain blocked until the baseline gate is passed.

## v3d FNO optimizer-protocol audit

The follow-up audit separates Adam-moment carry from cosine-schedule horizon
while keeping the same K=6 data, ridge anchor, 24-epoch base checkpoint and
per-block batch seeds:

```bash
python -m demo_t16_operator.run_v3d_fno_optimizer_audit --device mps
python -m demo_t16_operator.validate_v3d_fno_optimizer_results
```

Three strategies receive 216 continuation epochs after the shared base:
restart Adam/restart cosine, carry-continuation-Adam/restart-cosine, and carry-
continuation-Adam/one-long-cosine. The continuation optimizers are newly created
after the base checkpoint, so no base-training Adam moments are restored. Training
continues from each raw block endpoint, while the reported
checkpoint is the validation prefix-best. Dev2 fields and Q_audit are evaluated
only after all strategy and checkpoint decisions are frozen.

At 240 attempted epochs, carry-continuation-Adam/restart-cosine is the sample-weighted validation champion
(mean L2 0.094139) but remains above the 0.5%/1.0% plateau thresholds. The long-
cosine branch plateaus from epoch 192 but has a 1.70% worse final validation L2.
This locks a fixed-epoch FNO development baseline without pretending that the
strongest FNO is saturated or that different architectures have equal FLOPs. The result package contains 2,016 history rows, 171
strategy-seed checkpoints, 1,536 post-selection sample rows, paired diagnostics,
checksums and an independent validator; no NPZ or model weights are public.

## v3e cross-architecture compute accounting

The v3e ledger profiles 3D U-Net, FNO, DeepONet, the historical ray-set
operator and the frozen-FNO zero-init adapter on the same K=6, 8x16x16,
42-channel input. Each model uses three fresh worker processes. The report keeps
total/trainable parameters, versioned FLOPs-v1, synchronized MPS inference,
full physics-loss training-step latency, sampled memory and FNO time-to-target
separate from accuracy.

```bash
python -m demo_t16_operator.run_v3e_compute_accounting --device mps
python -m demo_t16_operator.validate_v3e_compute_results
pytest demo_t16_operator/test_v3e_compute_accounting.py -q
```

The public result directory contains only CSV, JSON, PNG and checksums. Source
NPZ files and checkpoints remain local. FLOPs-v1 excludes normalization,
activation, pooling, softmax, indexing, most elementwise work and optimizer
arithmetic, so it must be interpreted beside measured wall time. At the v3e
checkpoint only FNO had a matched 24-to-240 validation trajectory; v3f below
adds the first DeepONet/FNO comparison while confirmatory superiority stays locked.

## v3f DeepONet/FNO matched development frontier

The v3f run gives the 49,313-parameter grid DeepONet a validation-only learning-
rate screen over four rates and then applies the same three 24-to-240 optimizer
protocols, seeds, losses and block batch-order contracts used by FNO. The selected
DeepONet uses learning rate 0.002 and carry-continuation-Adam/restart-cosine, but
its sample-weighted final mean validation L2 is 0.175725 versus FNO's 0.094139. All four reported
DeepONet checkpoints are Pareto-dominated by an observed FNO checkpoint in
validation error and measured training time.

```bash
python -m demo_t16_operator.run_v3f_deeponet_frontier --device mps
python -m demo_t16_operator.run_v3f_deeponet_frontier --analysis-only
python -m demo_t16_operator.validate_v3f_deeponet_frontier
pytest demo_t16_operator/test_v3f_deeponet_frontier.py -q
```

The architecture winner is frozen by final validation before dev2 is read. On
the 128 reused post-selection development fields, FNO has 25.50% domain-equal mean field
superiority with a field-cluster-bootstrap 95% interval of [24.17%, 26.84%], positive
mean direction in every domain and seed, and zero weighted harm beyond 1%. This
passes only the FNO-control development gate; real geometry, blind final and
confirmatory claims remain closed. The interval resamples fields after model-seed
collapse; seed means are a separate directional diagnostic. The next engineering hypothesis keeps FNO as
the spatial mixer and uses a DeepONet/set branch only for acquisition-geometry
conditioning.

## v3g bounded DeepONet capacity audit

The v3g audit tests whether v3f's pooled DeepONet was an intentionally weak
baseline. Eight pre-registered variants cover rank 32/48/64/96 and alternative
3D pooling allocations. Every eligible variant is trained for 24 fixed epochs
at three learning rates and three model seeds, producing 72 validation-only
screen runs. Two higher-resolution pooling variants are excluded before
training because they exceed a 1.5x parameter cap.

```bash
python -m demo_t16_operator.run_v3g_deeponet_capacity_audit --device mps
python -m demo_t16_operator.run_v3g_deeponet_capacity_audit --analysis-only
python -m demo_t16_operator.validate_v3g_deeponet_capacity_results
pytest demo_t16_operator/test_v3g_deeponet_capacity_audit.py -q
```

The 24-epoch screen selects rank 64, pool 4x4x4 and learning rate 0.002.
After the selected cell alone receives all three 24-to-240 optimizer protocols,
its best final validation L2 is 0.176094, slightly worse than the v3f rank-48
reference at 0.175725 and far above FNO at 0.094139. Its reused-dev2 mean field
superiority versus the reference is -0.160%, with a field-cluster 95% interval
of [-0.339%, 0.006%]. The selected variant therefore does not replace the
reference. The result is a bounded baseline audit, not evidence that every
possible DeepONet is weak: only the short-horizon champion receives a long
curve, and that survivor-bias limitation is retained in the report.

The public result package contains the parameter manifest, all 72 screen rows,
three long-horizon protocols, a validation-only selection commit, post-selection
diagnostics, plots, checksums and a validator. Local checkpoints, work histories
and the NPZ dataset remain ignored.

Useful overrides:

```bash
.venv/bin/python demo_t16_operator/run_benchmark.py --epochs 3
.venv/bin/python demo_t16_operator/run_benchmark.py --device mps
.venv/bin/python demo_t16_operator/run_benchmark.py \
  --config demo_t16_operator/configs/full.json
```

The committed checkpoint is the CPU smoke configuration. `full.json` is a
budget proposal and has not been run or validated yet.

## Data split contract

| Split | Samples | Role |
| --- | ---: | --- |
| train | 64 | Gaussian/flame families, 5/7 views, noise 0.02/0.05 |
| val | 16 | Same domain, new seeds |
| test_iid | 20 | Same declared domain, new samples |
| test_view_ood | 16 | Three views only |
| test_noise_ood | 16 | Noise 0.10 only |
| test_joint_ood | 16 | Three views and noise 0.10 |
| test_family_ood | 20 | Entire thin-front family held out |

The noise values are dimensionless synthetic multipliers of the RMS
deflection. They are not pixels, angles, or measured OERF noise levels.

## Verified smoke checkpoint

Environment: Python 3.11.5, PyTorch 2.13.0, NeuralOperator 2.0.0, CPU training,
30 epochs, volume `8 x 16 x 16`. The exact sample-level metric CSV reproduced
byte-for-byte in two independent runs with the same seed.

| Method | Parameters | IID field rel. L2 | IID held-out rel. L2 |
| --- | ---: | ---: | ---: |
| physics lift | 0 | 0.4573 | 0.5203 |
| residual 3D U-Net | 86,633 | 0.2706 | 0.3191 |
| residual 3D FNO | 43,363 | 0.2321 | 0.2766 |

The FNO has lower field error than the U-Net in all five test splits in this
small closure test. The result is not uniformly positive: under three-view and
joint OOD, the mean held-out improvement over U-Net is small and its paired 95%
interval crosses zero. Family OOD remains poor for every method.

These failure signatures are more important than the leaderboard:

- three-view FNO field L2: 0.5081;
- thin-front family OOD FNO field L2: 0.6789;
- three-view held-out reprojection: FNO 0.5420 vs U-Net 0.5466, with a paired
  mean delta of -0.0046 +/- 0.0218 (approximate 95% interval);
- the small benchmark does not test resolution transfer or nonlinear rays.

## Three-seed causal ablation

The second checkpoint keeps the same 168 volumes fixed and varies only three
optimization seeds. Error bars use a Student-t 95% interval across those three
seeds, so they are a stability check rather than a population-level confidence
interval. The complete 12-training run was repeated independently; the paired
metric-delta CSV reproduced byte-for-byte. Wall-clock fields are intentionally
excluded from that determinism claim.

| Method | Parameters | IID field rel. L2 | IID held-out rel. L2 |
| --- | ---: | ---: | ---: |
| physics lift | 0 | 0.4573 | 0.5203 |
| residual FNO | 43,363 | **0.2187** | **0.2507** |
| absolute-output FNO | 43,363 | 0.2367 | 0.2677 |
| FNO without reprojection loss | 43,363 | 0.2375 | 0.2698 |
| parameter-matched U-Net | 49,111 | 0.3000 | 0.3488 |

The matched U-Net has more parameters than the FNO but higher field and
held-out error in every one of the 15 seed-domain cells. The residual skip is
not uniformly best: it wins all three seeds on IID and noise OOD, while the
absolute-output FNO wins all three seeds on three-view and joint OOD. Removing
the reprojection loss raises mean field error in all five test domains, but the
three-seed intervals remain too wide for a strong significance claim.

The absolute-output model still receives the calibrated physics lift as an
input. It ablates only the residual skip and is not a direct raw-projection
operator. The condition flip motivates a reliability-gated residual/absolute
hybrid, not a claim that one output parameterization dominates everywhere.

## Reliability-gate screen

The first reliability experiment tests six FNO variants across the same three
seeds and five domains. A fixed rule that reduces the lift weight to 0.6 for
three-view inputs makes view and joint OOD worse. A learned metadata gate only
changes mean field regret to the retrospective Residual/Absolute oracle from
3.10% to 3.08%, while held-out regret increases from 1.65% to 2.00%.

The observed-view lift reprojection residual is diagnostic (about 0.40 in IID
and 0.85 in view/joint OOD), but broadcasting this scalar as a full 3D input
channel destabilizes the small model. These negative controls motivate a true
dual-branch model trained with view dropout and held-out-camera consistency;
they do not validate that proposed model yet.

The complete reliability run was repeated independently. The oracle-regret and
paired-delta CSV files reproduced byte-for-byte. Wall-clock fields are excluded
from that determinism statement.

## Dual-branch support/query prototype

The third checkpoint uses view dropout to create 128 fixed support/query
variants from the 64 base training samples. A 43,485-parameter FNO emits a
physics-lift residual field and an absolute field. Five inference paths are
then compared across three optimization seeds:

| Inference path | IID field rel. L2 | IID held-out rel. L2 | Five-domain field regret to best endpoint |
| --- | ---: | ---: | ---: |
| residual head | 0.2629 | 0.3050 | not applicable |
| absolute head | 0.2612 | 0.3040 | not applicable |
| uniform dual | 0.2577 | 0.3015 | 0.197% |
| **closed-form support fit** | **0.2571** | **0.3001** | **0.014%** |
| query-trained router | 0.2578 | 0.3015 | 0.233% |

The closed-form support fit minimizes the observed support-view reprojection
error along the line segment joining the two expert fields. It uses no field
truth or held-out camera at inference. Its residual-head weight changes by
domain (about 0.31 to 0.68), and it reaches or improves on the better endpoint
field error in 75.8% of the 264 sample-seed cells.

The learned router is a retained negative result. Its mean weight remains near
0.50 in every domain, its field-wise endpoint selection accuracy is 40.5%, and
its weight-to-field-advantage Pearson correlation is -0.228. Training targets
also remain close to 0.5 with standard deviation about 0.08--0.09 because the
almost fully shared heads are too similar. By contrast, the observable support
reprojection advantage has Spearman 0.695 with the field-wise head advantage.

The next model therefore should not add a larger free-standing gate. It should
use genuinely diverse or low-sharing experts, retain the analytic support fit
as the mandatory baseline, and learn only a bounded correction or a
support-nullspace field residual. The support projection of the branch
difference supplies an identifiability score for uncertainty or abstention.

The dual experiment was rerun independently. Oracle regret, selection audit,
and feature-alignment CSVs reproduced byte-for-byte; training-time fields are
excluded from the determinism claim. Because three-view support variants are
seen in training here, the legacy `test_view_ood` name no longer denotes a
strict view-count OOD test. The model still uses a small linear synthetic
forward operator and is not evidence of performance on OERF data.

## Independent experts and support-nullspace audit

The fourth checkpoint removes the almost fully shared dual head. Two independent
FNO experts increase the total parameter count from 43,485 to 86,823 and raise
mean branch disagreement from about 0.119 to 0.190. The analytic support-fit
mixture reaches or improves on the better endpoint in 79.17% of the 264
sample-seed cells; its mean field regret to the better endpoint is -1.364%, where
negative regret means that interpolation beats the endpoint average.

This is not yet a fair capacity claim: the independent model roughly doubles the
operator parameters. An equally wide single model and an independent ensemble
remain mandatory controls.

An exact SVD audit then projects the remaining support-fit field error into the
declared support-camera nullspace. This truth-oracle correction improves every
sample-seed cell, gives 38.626% mean field improvement, and finds that 59.459%
of the remaining error energy lies in the support nullspace. The maximum change
to support projections is 5.51e-16. This proves structural headroom only; the
correction uses field truth and is unavailable at inference.

## Matched free and nullspace correctors

Two 13,105-parameter FNO correctors use identical initialization within each
seed. One emits a free correction; the other is projected exactly into the
declared support nullspace before a 0.5 relative-norm cap.

- The free corrector is unstable and has -32.06% aggregate field improvement.
- The nullspace corrector limits maximum support leakage to 2.59e-8.
- Hard projection prevents the catastrophic free-corrector seed but does not
  recover the oracle direction: aggregate field improvement is -0.126% and
  held-out improvement is -0.412%.
- Noise and family OOD improve slightly; view and joint OOD become worse.

The exact constraint is therefore a useful stability and consistency mechanism,
not sufficient evidence that the network has learned the missing field.

## Query-calibrated nullspace correction

The fifth checkpoint treats the learned null correction as a direction rather
than accepting its full amplitude. A withheld query view is selected by maximum
predicted correction-projection energy. Its noisy observation then solves a
clipped scalar least-squares problem:

```text
alpha_Q = clip(<A_Q d_N, y_Q - A_Q x_S> / ||A_Q d_N||^2, 0, 1)
x_hat = x_S + alpha_Q d_N
```

Because `A_S d_N` is numerically zero, this calibration does not alter support
projections. Across three seeds and five test domains:

| Method | Mean field improvement | Better than support fit | Extra views |
| --- | ---: | ---: | ---: |
| Full learned null correction | -0.126% | 60.98% | 0 |
| One-query accept/reject | +0.677% | 44.32% | 1 |
| **One-query line search** | **+0.746%** | **52.65%** | **1** |
| All-query line search | +0.956% | 65.53% | all withheld |
| Field-oracle line search | +1.212% | 81.82% | 3D truth |

The one-query and all-query mean improvements are positive in all 15
seed-domain cells. Maximum support leakage for the all-query result is 2.26e-9.
The gain remains small and consumes additional inference measurements; query
camera placement, nonlinear rays, independent resolution transfer, and OERF
data are untested.

## Equal-reconstruction-budget controlled inference audit

The v2d audit responds to the main v2c red-team objection: the same `Q_fit`
view must also be allowed to enter reconstruction directly. For each
reconstruction budget `K in {4, 6, 8}` it now fixes:

- `K-1` support views;
- one `Q_fit` view shared by direct reconstruction, learned alpha calibration,
  and the numerical query-null update;
- a pre-locked 60-degree `Q_audit` camera that never selects a method or fits a
  coefficient.

The installed/evaluation camera count is therefore `K+1`; `K` is not claimed as
the total hardware budget. Fixed `Q_fit` is 80 degrees. Random, max-gap, and
adaptive-energy policies are secondary analyses.

Across 88 independent fields and three model seeds, seeds are collapsed within
field before five-source-split-equal, 20,000-replicate stratified bootstrap.
For the fixed query:

| Reconstruction budget | S-union-Q direct vs S | learned vs S | learned vs direct | >1% harm vs direct |
| ---: | ---: | ---: | ---: | ---: |
| 4 | +15.25% | +1.12% | -20.17% | 88.0% |
| 6 | +10.51% | +1.37% | -13.07% | 72.0% |
| 8 | +12.69% | +1.18% | -15.11% | 86.2% |

The current checkpoint path therefore passes 0/3 same-budget gates. A K=6
max-gap setting has a +0.56% mean over direct, but its cluster interval crosses
zero, p10 is about -5.9%, and 28.7% of fields are harmed by more than 1%.

This is a controlled inference audit, not a final algorithm verdict. The base
and corrector checkpoints were trained under the earlier view-dropout mask
distribution, the 88 fields have informed earlier v1-v2c development, and the
forward remains the 8x16x16 linear slice stack. The committed scientific status
is `PILOT_ONLY_CURRENT_CHECKPOINT_PATH_FAILS`.

Run and validate:

```bash
.venv/bin/python -m demo_t16_operator.run_fair_camera_budget
.venv/bin/python -m demo_t16_operator.validate_fair_camera_budget_results
```

Primary assets:

- `configs/fair_camera_budget.json`;
- `run_fair_camera_budget.py`;
- `results/fair_camera_budget/fair_camera_budget_samples.csv`;
- `results/fair_camera_budget/fair_camera_budget_clusters.csv`;
- `results/fair_camera_budget/fair_camera_budget_report.json`;
- `validate_fair_camera_budget_results.py`.

## Leakage controls

- The affine lift calibration is fitted once on all training samples, then
  frozen for validation and every test split.
- No prediction is aligned to its own ground truth.
- Held-out reprojection uses canonical views absent from that sample's input.
- The thin-front generator family is absent from training and validation.
- Every split uses disjoint seeds.

## Files

- `bost_physics.py`: phantom families, forward matrix, and physics lift.
- `data.py`: paired dataset, condition splits, calibration, and seven channels.
- `models.py`: residual 3D U-Net and official NeuralOperator FNO wrapper.
- `train_eval.py`: losses, paired metrics, deterministic evaluation, and plots.
- `run_benchmark.py`: end-to-end entry point.
- `run_ablations.py`: three-seed residual, loss, and capacity controls.
- `run_reliability_gates.py`: fixed/learned gate and observable-quality
  controls across three seeds and five domains.
- `run_dual_branch_query.py`: support/query view-dropout dual-head prototype,
  analytic support-fit baseline, router-collapse and feature-alignment audits.
- `run_nullspace_identifiability_audit.py`: exact support-nullspace upper-bound
  and rank/nullity audit.
- `run_independent_nullspace_headroom.py`: oracle headroom above the independent
  expert support-fit baseline.
- `run_support_nullspace_corrector.py`: matched free/null FNO correctors.
- `run_query_calibrated_nullspace.py`: one/all-query closed-form amplitude
  calibration of a learned null direction.
- `test_smoke.py`: forward-matrix, split-isolation, and 3D model-shape tests.
- `configs/smoke.json`: verified checkpoint.
- `configs/full.json`: proposed 24 x 24 x 16 scale-up.
- `results/run_report.json`: machine-readable environment, metrics, and paired
  comparisons.
- `results/sample_metrics.csv`: one row per method, split, and sample.
- `results/ablations/ablation_report.json`: machine-readable multi-seed report.
- `results/reliability_gates/gate_report.json`: machine-readable reliability
  report, including oracle regret and claims boundary.
- `results/dual_branch_query/dual_report.json`: machine-readable dual-branch
  report, including endpoint regret, router selection and observable-feature
  alignment.
- `results/independent_dual_support/dual_report.json`: independent-expert
  support-fit results and capacity boundary.
- `results/nullspace_identifiability/nullspace_report.json`: structural
  range/null upper bound.
- `results/support_nullspace_corrector/support_nullspace_report.json`: matched
  learned-corrector positive and negative results.
- `results/query_calibrated_nullspace/query_calibrated_report.json`: reserved
  query-camera amplitude calibration and claims boundary.

## Next falsifiable experiments

### v3k-A equal-exposure counterfactual supervision

The v3k-A audit separates layout diversity from training exposure. `M1-repeat`
repeats one geometry four times per source field, while `M4-counterfactual`
uses four distinct geometries. Both arms contain 160 independent fields, 640
training rows, identical optimizer steps, three seeds, and the same frozen FNO
plus 1,023 trainable adapter parameters. Validation averages all four held-out
layouts within each source field before field-cluster bootstrap.

Run and validate:

```bash
.venv/bin/python -m demo_t16_operator.run_v3k_a_counterfactual_supervision --device cpu
.venv/bin/python -m pytest -q demo_t16_operator/test_v3k_a_counterfactual_supervision.py
.venv/bin/python -m demo_t16_operator.validate_v3k_a_counterfactual_results
```

The committed result contains 24 training runs, 576 history rows, 20,160
sample-metric rows, and 960 same-model descriptor swaps. M4 correct geometry
beats shuffled geometry by 0.0675% on validation with a source-field cluster CI
of [0.0410%, 0.0952%], but the entire interval remains below the preregistered
0.25% minimum meaningful effect. Descriptor swapping changes the M4 correction
by 8.1399% while final field gain is only 0.0130% with an interval crossing
zero. The global geometry-to-channel modulation is therefore stopped; this is
not a superiority result. Private checkpoints stay under the ignored
`results/v3k_a_counterfactual_supervision_work/` directory.

### v3k-B voxel-local ray-set mechanism pilot

v3k-B replaces the global geometry vector with one local token per camera and
voxel. The shared encoder receives normalized backprojection, camera mask, and
angular coordinates; a frozen-FNO query performs permutation-invariant masked
attention before a zero-initialized support-limited residual. The formal
control keeps the active ray set exactly fixed and cyclically reassigns only
the ray-angle correspondence. An earlier whole-layout derangement was stopped
after one seed because it leaked additional camera observations and is not part
of the committed result.

Run, refresh analysis, and validate:

```bash
.venv/bin/python -m demo_t16_operator.run_v3k_b_voxel_ray_set_pilot --device cpu
.venv/bin/python -m demo_t16_operator.run_v3k_b_voxel_ray_set_pilot --analysis-only
.venv/bin/python -m pytest -q demo_t16_operator/test_v3k_b_voxel_ray_set.py
.venv/bin/python -m demo_t16_operator.validate_v3k_b_voxel_ray_set_results
```

The committed result contains 12 selected runs, 288 history rows, 10,080
sample rows, and 480 same-model swaps. Correct local rays beat geometry-only
tokens by 1.4445% with a field-cluster interval of [0.5149%, 2.3801%], but
correct ray-angle pairing loses 0.1967% to the information-matched pairing
shuffle. A same-model swap changes the correction by 19.0683% while worsening
field error by 0.3586%. The scalar angle-pairing mechanism is therefore
stopped. The next bounded direction is a numerical Landweber baseline followed
by per-view adjoint-residual tokens, not wider attention. Private checkpoints
remain in the ignored `results/v3k_b_voxel_ray_set_work/` directory.

### v3k-C zero-new-learning adjoint/Landweber gate

v3k-C freezes the same FNO and first separates the free feasibility operation
from numerical data-consistency refinement. The feasibility map is an
idempotent nonnegative hard-support projection; it deliberately thresholds the
synthetic taper instead of multiplying the soft window at every iteration.
For each of 28 camera masks, the runner checks the exact discrete adjoint,
finite-difference data-fidelity gradient, spectral constant, Jacobi diagonal,
and audit-camera exclusion.

Validation selects only source-field relative L2 after collapsing four layouts
within each of 40 fields. Its 112-row screen contains no audit or reprojection
columns. The selection commit is written before Q_audit and four reused
development test domains are evaluated.

Run and validate:

```bash
PYTHONPATH=. .venv/bin/pytest -q demo_t16_operator/test_v3k_c_adjoint_landweber.py
PYTHONPATH=. .venv/bin/python demo_t16_operator/run_v3k_c_adjoint_landweber_gate.py
PYTHONPATH=. .venv/bin/python demo_t16_operator/validate_v3k_c_adjoint_landweber_results.py
```

The selected standard method uses geometry-normalized `beta=1.9` for 64
iterations. It improves validation source-field error by 44.6024% over the
feasible FNO, with a field-cluster interval of [40.1023%, 49.1099%]. The four
development test domains retain positive mean gains of 22.4001% to 44.5911%
and positive held-out audit gains. The worst split-layout cell still contains
a -17.5799% field outlier and a 9.375% harm rate, so the result is not described
as universal improvement.

The independent validator recomputes 28 geometry checks, 112 validation-grid
rows, 672 field-layout pairs, 2,688 sample-method rows, 20 pairwise summaries,
20 layout summaries, and 11 public checksums. It also verifies checkpoint drift
is zero and public results contain no weights, arrays, or absolute private
paths. The dataset NPZ and frozen checkpoint remain in ignored result folders.

This result makes tuned projected Landweber a required baseline. It authorizes
only a bounded conditional-scalar development prototype after global-step,
closed-form line-search, lookup, and ridge-start controls are added. It does
not authorize confirmatory training, a learned voxel preconditioner, per-view
set aggregation, superiority, or blind-final claims. The current forward is a
depth-separable linear slice stack; real BOST requires a matched nonlinear
forward/Jacobian-adjoint contract.

### v3k-E projected-BB and noise-stopping gate

v3k-E inherits the frozen FNO/ridge starts and the validation-selected v3k-D
fixed-step and quadratic controls. It adds BB1, BB2, and deterministic BB1/BB2
alternation using secants between projected feasible iterates. Invalid
curvature falls back to the inherited `1.9/L(mask)` step before all candidates
are clipped in normalized spectral coordinates. There is no line search, so
this is a projected-BB mechanism control rather than the convergent
nonmonotone SPG algorithm.

Run and independently validate:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q demo_t16_operator/test_v3k_e_projected_bb.py
PYTHONPATH=. .venv/bin/python demo_t16_operator/run_v3k_e_projected_bb_gate.py --device cpu
PYTHONPATH=. .venv/bin/python demo_t16_operator/validate_v3k_e_projected_bb_results.py
```

The validation screen has 180 cells: two starts, three BB variants, six
iteration counts, one strict cap and four wide caps. Strict and wide winners
are selected independently using source-field error after collapsing four
layouts. The selection commit is written before the four reused development
audit domains are constructed. Every PBB iteration is charged one `A` and one
`A^T`; the final objective and evaluation projections are explicitly excluded
from the solver budget.

The FNO-start wide winner alternates BB1/BB2, uses a normalized upper cap of
10, and runs 64 iterations. Its validation mean field relative L2 is 0.138640,
versus 0.147083 for the 128-call fixed Landweber control and 0.143855 for the
193-call quadratic control. The source-field paired mean gain over fixed
Landweber is 1.4011%, but 37.5% of validation fields are harmed by more than
1%. Noise OOD regresses by 11.5611%, while family and joint OOD improve by
14.8321% and 12.2851%. The data objective remains nearly monotone, so objective
descent alone is not a noise-regularizing stopping rule.

The independent validator recomputes 180 selection rows, 6,720 sample rows,
50 split summaries, 45 pairwise comparisons and bootstrap intervals, 20 step
audits, 50 call-ledger rows, and 12 public checksums. It verifies support,
nonnegativity, selection timing, call counts, claim boundaries, and that no
private dataset or checkpoint was published.

The next deterministic experiment is a noise-aware discrepancy or
residual-whiteness stop on the same PBB trajectory, followed by a fully charged
nonmonotone SPG control. Learned stopping remains blocked until those controls
leave reproducible headroom and a fresh lock is available.

### Base-correction and selective-ensemble independent gates

The current base-correction track asks a narrower question: can one learned
field correction improve a strong four-call projected-BB fallback without
repeating the physical trajectory?  It uses a separate fully 3D prescribed
curved/cone weak-deflection generator with correlated, signal-dependent noise.
This remains a synthetic audit, not nonlinear refraction, CFD, OpenBOS/OERF,
or a comparison against FNO, FFNO, DeepONet, NeRIF, NeDF, or learned
primal-dual reconstruction.

Run the focused tests, the fresh family-held-out ensemble gate, and its
independent artifact validator:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q \
  demo_t16_operator/test_base_correction_cg_pdno.py \
  demo_t16_operator/test_cg_pdno.py
PYTHONPATH=. .venv/bin/python demo_t16_operator/run_pbb_ensemble_selective_gate.py
PYTHONPATH=. .venv/bin/python demo_t16_operator/validate_pbb_ensemble_selective_gate.py
```

The evidence ladder is deliberately negative. Fixed-PG base correction loses
17.9029% to PBB on its independent lock. A PBB-base correction reduces the
mean deficit to 0.1023% but has 19.44% tail harm. A saturation guard reaches
+0.2809% mean gain, but p10 remains -0.8312% and harm is 7.5%. The five-head
selective ensemble shares one four-call PBB trajectory and uses exact tiny-grid
spectral norms; its selection split contains no threshold satisfying 20%
coverage, nonnegative p10, and at most 5% harm. It therefore commits to zero
coverage before constructing a new family-held-out lock, where the result is
exactly the deterministic PBB fallback and the superiority gate fails.

The PBB quadratic bound certifies data descent relative to the shared base; it
does not certify lower reconstruction error or improvement over the final PBB
fallback. The next admissible signal must come from held-out optical views,
operator mismatch, independently measured noise, or real data rather than a
wider or larger ensemble.

### v5a finite-aperture first-open mismatch gate

v5a is the first track here with explicitly different truth and reconstruction
operators. Observations use a denser deterministic finite-aperture sub-ray and
path quadrature, while reconstruction sees only a five-radius approximate bank.
One audit camera is excluded from reconstruction, cross-view probes, selection,
and synthetic noise scaling. The model remains a prescribed linear
weak-deflection surrogate, not nonlinear ray tracing or a reproduction of the
full Molnar cone-ray model.

The source, config, selection rule, threshold, call budget, and five lock gates
were frozen in commit `90ff961` and pushed before the first-open lock was
constructed. Reproduce and validate with:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q demo_t16_operator
PYTHONPATH=. .venv/bin/python \
  demo_t16_operator/run_v5a_blind_aperture_calibration.py
PYTHONPATH=. .venv/bin/python \
  demo_t16_operator/validate_v5a_blind_aperture_calibration.py
PYTHONPATH=. .venv/bin/python \
  demo_t16_operator/analyze_v5a_blind_aperture_failure.py
```

The first-open claim gate fails. Pinhole mismatch is nontrivial at 3.1435%, but
the frozen selective calibration reaches only +1.7527% mean field gain, p10 is
-17.7844%, and the overall harm rate is 25%. The originally preregistered harm
rate includes abstained samples as zero-gain fallback cases; the more relevant
accepted-conditional harm is 9/24 = 37.5%, with accepted-only p10 -24.7790%.
Both must be reported. The independent validator recomputes 16,103 atomic
checks over 66 sample rows and 267 calibration rows; validation success means
the failed result is internally truthful, not that the method passed.

Post-lock diagnosis is descriptive only. It shows 34/36 aperture choices at
the bank boundaries: all helical-plume fields select radius zero, while 16/18
stratified-ignition fields select 0.16. Confidence has essentially zero rank
correlation with gain among accepted samples. v5a is therefore sealed as
negative evidence of morphology/operator confounding; its lock cannot be used
to tune a replacement threshold.

The v5b design protocol is
`v5b_rig_shared_profile_calibration_protocol.md`. It replaces per-scene blind
radius selection with metadata-anchored, rig-shared low-dimensional cone
calibration; a profile-Fisher/Schur-complement score tests whether optical
parameters remain identifiable after the field nuisance is removed; outer
held-out cameras control accepted-conditional risk. These ingredients are not
claimed as individually new. A publishable contribution would require a BOST-
specific implementation, a paired factorial mismatch benchmark, a new
confirmatory lock, matched strong baselines, and a real f-stop sweep.

1. Match the independent dual model against an equally wide single operator and
   independent ensemble.
2. Replace the linear stack operator with the lab geometry/ray operator.
3. Sweep query-camera count, angle, noise, calibration error, and synchronization
   error as a value-of-information experiment.
4. Train at one grid and test a separately generated higher-resolution field;
   interpolation of the same array is not a valid resolution-transfer test.
5. Compare global FNO with localized kernels/CNO only after thin-front failure is
   reproduced at independent resolution.
6. Use the query-calibrated output as a NeRIF initialization and compare total runtime,
   final reprojection, and failure rate against random initialization.
