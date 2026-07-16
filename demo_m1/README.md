# M1 2.5D / 3D Stack BOST Demo

This demo upgrades M0 from a single 2D phantom to a lightweight 3D refractive-index volume.

It is still a simplified synthetic demo:

- It uses a straight-ray, small-angle proxy model.
- It reconstructs the volume as a stack of 2D BOST slices.
- It adds a compact 3D coordinate regularizer on top of the stack reconstruction.
- It is not a full endoscopic 9-view OERF geometry.
- It does not claim to reproduce the full NeRIF paper.

What it does prove:

```text
3D refractive-index phantom
  -> multi-view BOS-like deflection stacks
  -> baseline stack reconstruction
  -> 3D coordinate-regularized stack reconstruction
  -> volume-level metrics and figures
```

Current result:

- Main stress test: 5 sparse views.
- Baseline relative L2: about 0.257.
- 3D coordinate-regularized relative L2: about 0.234.
- View-count scan: the coordinate regularizer helps at 3-5 views, while the clean classical stack baseline becomes stronger from about 7 views onward.

This is the useful thesis-level conclusion: implicit or coordinate-based priors are not magic. They are most worth studying under sparse-view, noisy, masked, or poorly conditioned measurement settings.

## Run

From the dashboard root:

```bash
python3 demo_m1/run_m1_3d_stack_bost.py
```

Outputs:

- `demo_m1/results/m1_volume_summary.png`
- `demo_m1/results/m1_metrics_card.png`
- `demo_m1/results/m1_view_count_curve.png`
- `demo_m1/results/metrics.csv`
- `demo_m1/results/view_count_metrics.csv`

## Thesis Use

This demo can support the thesis chapter on synthetic data and minimal BOST/NeRIF-style reconstruction:

- It gives a 3D phantom.
- It gives volume-level L2 / CC / SSIM-proxy / PSNR metrics.
- It shows how view count affects 3D-stack reconstruction and coordinate regularization.
- It gives a clean bridge from M0 2D toy to future real 9-view BOST data.
- It gives a concrete question for He Yuanzhe: on real OERF data, is the limiting factor view count, noise, geometry, mask, or representation capacity?

## Next Upgrade

1. Replace per-slice reconstruction with true 3D ray geometry.
2. Replace the post-FBP coordinate regularizer with a PyTorch NeRIF-style forward-model loss.
3. Add `data_templates/` loader for a real OERF sample.
4. Add leave-one-view-out reprojection validation.
