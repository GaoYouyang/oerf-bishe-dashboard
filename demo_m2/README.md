# M2 Robustness Scan Demo

This demo turns M1 into a small systematic experiment.

It asks three thesis-useful questions:

1. Does a compact 3D coordinate regularizer help under sparse views?
2. Does it remain useful as deflection noise increases?
3. Does increasing coordinate-field capacity always improve reconstruction?

This is still a synthetic proxy, not a full NeRIF reproduction. It is designed to create clean figures for an opening report and to reveal which real-data question should be asked next.

## Run

From the dashboard root:

```bash
python3 demo_m2/run_m2_robustness_scan.py
```

Outputs:

- `demo_m2/results/m2_noise_view_scan.png`
- `demo_m2/results/m2_improvement_heatmap.png`
- `demo_m2/results/m2_capacity_scan.png`
- `demo_m2/results/m2_metrics_grid.csv`
- `demo_m2/results/m2_capacity_metrics.csv`

## Current Interpretation

The expected pattern is not that the coordinate regularizer wins everywhere.

The useful pattern is:

- It tends to help when views are sparse and artifacts dominate.
- It can lose when enough clean views make the classical stack baseline accurate.
- Higher capacity is not automatically better; too little capacity oversmooths and too much capacity may fit artifacts.

That is exactly the kind of result a bachelor thesis can honestly defend: identify where implicit priors are useful, where they fail, and what real OERF data is needed to decide the next model.

