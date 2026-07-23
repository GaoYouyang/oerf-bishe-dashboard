# M0 2D BOST / Coordinate-Field Toy Demo

This is the first runnable demo for the OERF / BOST-NeRIF graduation-design workbench.

It deliberately avoids PyTorch because the current local Python environment does not provide `torch`. The goal is not to reproduce NeRIF exactly, but to create the smallest closed loop:

```text
synthetic refractive-index field
  -> simplified BOS deflection sinograms
  -> voxel/tomography baseline
  -> coordinate-field inverse with random Fourier features
  -> metrics and figures
```

## Run

From the dashboard root:

```bash
python3 demo_m0/run_m0_2d_bost.py
```

Outputs:

- `demo_m0/results/m0_summary.png`
- `demo_m0/results/view_count_curve.png`
- `demo_m0/results/metrics.csv`

## What This Demo Proves

- You can generate a known refractive-index phantom.
- You can simulate multi-view BOS-like deflection measurements.
- You can reconstruct the field with a conventional baseline.
- You can compare that baseline with a coordinate-field representation.
- You can automatically save figures and metrics for weekly reports.

## What This Demo Does Not Claim

- It is not a full 3D BOST implementation.
- It is not the full NeRIF architecture from the Physics of Fluids paper.
- It does not model a real camera/endoscope geometry.
- It uses a simplified small-angle, straight-ray model.

## Next Upgrade

1. Replace the 2D phantom with a 3D phantom.
2. Replace random Fourier feature ridge fitting with a PyTorch MLP after `torch` is available.
3. Add real BOST displacement data if He Yuanzhe can provide a sample.
4. Add PIV-BOST or 4D low-rank toy as the second-stage branch.
