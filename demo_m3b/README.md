# M3B 4D BOST Low-Rank Temporal Toy

This demo is a lightweight bridge to the tensor-decomposition 4D BOST direction.

It does not reproduce the full ACM TOG paper. Instead, it isolates one bachelor-thesis-scale question:

```text
moving 3D refractive-index volume
  -> sparse-view BOS-like deflection stacks over time
  -> frame-by-frame 3D-stack reconstruction
  -> low-rank temporal denoising
  -> error / smoothness / rank trade-off
```

## Run

From the dashboard root:

```bash
python3 demo_m3b/run_m3b_4d_lowrank_bost.py
```

Outputs:

- `demo_m3b/results/m3b_4d_summary.png`
- `demo_m3b/results/m3b_rank_scan.png`
- `demo_m3b/results/m3b_temporal_trace.png`
- `demo_m3b/results/metrics.csv`
- `demo_m3b/results/rank_metrics.csv`

## Six-Axis Multi-Seed Sweep

The second script turns the single run into an OFAT stress test with paired
random seeds:

```bash
python3 demo_m3b/run_m3b_six_axis_sweep.py
```

It varies rank, deflection noise, frame count, view count, systematic-bias
signature, and temporal dynamics. The default run uses eight paired noise
seeds and writes:

- `demo_m3b/results/m3b_six_axis_overview.png`
- `demo_m3b/results/m3b_rank_seed_stability.png`
- `demo_m3b/results/m3b_bias_dynamics_diagnostic.png`
- `demo_m3b/results/six_axis_raw.csv`
- `demo_m3b/results/six_axis_summary.csv`
- `demo_m3b/results/six_axis_paired_improvements.csv`
- `demo_m3b/results/six_axis_report.json`

Use `--quick` for a two-seed smoke test. The sweep is deliberately small
(24 x 24 x 10) and one-factor-at-a-time; it is not a full factorial design.

## Crossed Rank-Noise-View-Dynamics Sweep

The third script turns the OFAT hypotheses into a balanced crossed experiment:

```bash
python3 demo_m3b/run_m3b_interaction_sweep.py
```

The full design uses five ranks (1/2/3/5/8), five noise levels
(0/0.07/0.14/0.28/0.42), four view counts (3/5/7/9), four dynamics families
(smooth/fast/chirp/transient), and eight paired observation-noise seeds. It
contains 80 environment cells, 640 observation cells, 3,840 method rows, and
3,200 paired low-rank comparisons. Confidence intervals use Student-t 95%
intervals with `t(7)=2.365`.

Outputs:

- `demo_m3b/results/m3b_interaction_heatmaps.png`
- `demo_m3b/results/m3b_rank_selection_stability.png`
- `demo_m3b/results/m3b_interaction_tradeoffs.png`
- `demo_m3b/results/m3b_rank3_operating_map.png`
- `demo_m3b/results/interaction_raw.csv`
- `demo_m3b/results/interaction_paired.csv`
- `demo_m3b/results/interaction_cell_summary.csv`
- `demo_m3b/results/interaction_rank_selection.csv`
- `demo_m3b/results/interaction_report.json`

Use `--quick --output-dir /tmp/m3b_interaction_quick` for a reduced smoke test.
Two independent quick runs matched numerically to floating-point precision after
excluding runtime columns.

## Family-Generalized Observable Rank Selector

The fourth script replaces the single-phantom oracle question with a deployment
question:

```bash
python3 demo_m3b/run_m3b_family_selector.py
```

The formal design crosses four morphology families, three dynamics modes, four
noise levels (including noise-free), three support-view counts, six paired
noise seeds, and seven rank candidates including full rank. It contains 144
environment cells, 864 observation cells, and 6,048 candidate reconstructions.

Outer evaluation leaves one complete phantom family out. Ridge regularization
is tuned only inside the remaining families. Test-time features exclude family,
noise labels, field truth, and oracle rank; they use only capacity/spectrum,
support reprojection, temporal statistics, view count, and optionally one
held-out measured view.

Run a reduced smoke test:

```bash
python3 demo_m3b/run_m3b_family_selector.py --quick --output-dir /tmp/m3b_family_selector_quick
```

Resume an expanded design without recomputing completed cells:

```bash
python3 demo_m3b/run_m3b_family_selector.py --resume
```

Rebuild all selectors, tables, and plots from the committed raw table:

```bash
python3 demo_m3b/run_m3b_family_selector.py --reuse-raw
python3 demo_m3b/validate_family_selector_results.py
```

Outputs live in `demo_m3b/results/family_selector/`:

- five publication-style figures for oracle morphology maps, selector regret,
  family transfer, rank choices, and observable alignment;
- one feature-group ablation figure;
- 6,048 candidate rows and 6,048 main selector decisions;
- 4,320 feature-ablation decisions;
- a JSON report containing every outer-fold training-family list, tuned ridge
  strength, standardized coefficient vector, claims boundary, and key finding.
- `family_selector_checksums.sha256`, covering all six core CSVs and the
  deterministic JSON report; the independent validator verifies every digest.

## Leave-One-Geometry-Out Capacity and UQ Audit

The fifth script crosses six angular-layout signatures with four morphology
families, three dynamics, three noise levels, two support-view counts, four
paired seeds, and seven rank candidates:

```bash
python3 demo_m3b/run_m3b_geometry_uq.py
python3 demo_m3b/run_m3b_geometry_uq.py --reuse-raw
python3 demo_m3b/validate_geometry_uq_results.py
```

The formal matrix contains 1,728 observation cells, 12,096 candidates, and
12,096 selector rows. Outer evaluation leaves one complete geometry out;
ensemble members exclude both the outer geometry and one additional inner
geometry. The validator checks row counts, unique keys, rank coverage,
outer-geometry exclusion, deterministic reports, and checksums.

Key findings:

- fixed rank 3 mean/p95 regret: 1.918% / 5.519%;
- no-heldout nested-LOGO selector: 0.273% / 1.196%;
- 93.81% of cells are within 1% of the retrospective oracle;
- adding held-out-camera features does not improve the aggregate result;
- morphology, dynamics, and noise dominate the remaining failures more than
  the six declared synthetic geometry signatures;
- the combined uncertainty score fails (Spearman 0.015, high-risk AUC 0.532);
- prediction standard deviation has high-risk AUC about 0.697, but replacing
  rejected predictions with full rank makes end-to-end system risk worse.

The geometry families are synthetic and include one controlled two-degree
calibration offset. They are not OERF camera calibrations or curved-ray tests.

## Current Interpretation

This toy should be read as a sanity check for 4D BOST, not as a final method.

Current default result:

- Framewise baseline mean relative L2: about 0.366.
- Low-rank rank 3 mean relative L2: about 0.347.
- Temporal smoothness drops from about 0.279 to about 0.177.
- Centroid trajectory RMSE drops from about 0.0401 to about 0.0381.

The eight-seed sweep adds a stricter interpretation at its smaller sweep
resolution:

- Rank 3 remains the lowest global relative-L2 setting.
- Global norm-ratio L2 improves by about 3.90%; paper-style squared L2 improves
  by about 7.65%.
- Temporal-gradient and temporal-curvature errors improve by about 44.68% and
  73.19%.
- Held-out deflection error improves by about 9.23%.
- Mass-trace RMSE worsens by about 0.94%, so smoother output is not uniformly
  more physical.
- In the noise-free case, rank 3 worsens field L2 by about 0.56%; its benefit
  grows with noise.

The crossed experiment adds the operating-domain result that OFAT cannot
establish:

- Cell-wise best rank is not constant: rank 2/3/5/8 wins 27/20/24/9 of the 80
  environment cells; rank 1 wins none.
- Rank 3 is the best fixed default by mean field L2 and by regret to the
  cell-wise oracle: 0.79% mean regret and 2.21% p95 regret.
- All 20 three-view environments select rank 2. Noise-free, adequately viewed
  environments usually need a higher rank, while high-noise environments move
  toward rank 2-3.
- Rank 3 improves field L2 in 61/80 cells and worsens it in 19/80 cells under
  paired Student-t intervals. The negative cells concentrate at noise-free or
  low-noise 5-9-view settings.
- Field accuracy and held-out reprojection mostly agree, but physical traces do
  not: 116/400 environment-rank cells improve field L2 while worsening the mass
  trace.
- These counts use one synthetic phantom family and observation-noise seeds;
  they are not evidence of generalization to eight independent flow cases.

The family-generalized experiment then tests the missing transfer and selector
questions:

- Oracle rank is morphology-dependent: ranks 2/3/5/8/12/full win
  77/116/230/245/124/72 of the 864 observation cells.
- Fixed rank 3 has 1.561% mean oracle regret and 5.635% p95 regret.
- A nested-LOFO ridge selector using no held-out camera reduces mean regret to
  0.252%, reaches within 1% of oracle in 92.4% of cells, and beats rank 3 in
  77.8% of cells.
- Adding a held-out camera lowers mean regret to 0.210% and raises 1%-oracle
  coverage to 94.4%; the incremental mean-regret gain is only 0.042 percentage
  point.
- Directly minimizing held-out residual fails badly: 2.423% mean regret and
  9.467% p95 regret. Held-out consistency is a feature, not a standalone field
  quality metric.
- Capacity/spectrum, support, and temporal feature groups alone reach
  0.789%/0.614%/0.668% mean regret; their combination reaches 0.252%. The
  useful result is feature complementarity, not ridge regression itself.
- The result validator proves row counts, unique keys, rank coverage, oracle
  consistency, outer-fold family exclusion, forbidden-feature absence, and
  report/table agreement.

The useful thesis question is now:

- Does a low-rank temporal prior reduce frame-to-frame reconstruction jitter?
- At what rank does it stop smoothing useful motion and start preserving noise?
- Which metric should be shown to He Yuanzhe: per-frame L2, temporal smoothness, centroid trajectory, runtime, or memory?
- Can a deployment-time capacity selector combine support reprojection,
  singular spectrum, event retention, and physical traces without field truth?
- Does a held-out camera add enough value to justify removing it from the
  reconstruction set, or should it be retained only for audit and uncertainty?
- Can the selector transfer across flow morphology and camera geometry, then
  abstain or trigger NeRIF refinement when its confidence is low?

This is a good M3B branch if He Yuanzhe says the group needs a 4D BOST submodule,
visualization helper, parameter scan, or temporal capacity-control module rather
than a PIV-BOST compensation project. The rank/noise/view/dynamics and
cross-morphology selector experiments are complete. The next experiment should
be chosen with He Yuanzhe from leave-one-geometry-out transfer, bias/sync/
exposure-blur thresholds, uncertainty/abstention, or a small real-data failure-
signature audit.
