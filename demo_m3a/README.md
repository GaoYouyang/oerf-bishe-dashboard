# M3A PIV-BOST Compensation Toy

This demo is a lightweight bridge to He Yuanzhe's simultaneous PIV-BOST direction.

It does not perform real particle-image cross-correlation. Instead, it models the velocity-field error that PIV can see when particle images are refractively displaced:

```text
true velocity
  + change of refractive image displacement along the particle motion
  = observed PIV displacement
```

If BOST provides a refractive-displacement estimate, the bias term can be subtracted.

## Run

From the dashboard root:

```bash
python3 demo_m3a/run_m3a_piv_bost_compensation.py
```

Outputs:

- `demo_m3a/results/m3a_compensation_summary.png`
- `demo_m3a/results/m3a_error_profile.png`
- `demo_m3a/results/metrics.csv`

## Current Result

The default synthetic case produces a clear compensation effect:

- Observed PIV velocity contains a refractive bias.
- BOST-style compensation reduces vector RMSE from about 0.0101 to about 0.0067.
- The 95th percentile error drops from about 0.0240 to about 0.0072.
- Residual local spikes remain because the refractive-displacement estimate is intentionally noisy and the endpoint correction is approximate.

This gives a concrete question for He Yuanzhe: on real simultaneous PIV-BOST data, should the bachelor project correct raw particle images, vector fields, or only quantify the error propagation?

## Next Upgrade

The current toy works at the velocity-field level. The next useful step is an image-layer ladder:

1. Simulate particle-position error caused by a refractive-index gradient.
2. Generate a pair of synthetic particle images.
3. Estimate velocity with OpenPIV or a minimal cross-correlation routine.
4. Compare uncorrected, image-corrected, and vector-field-corrected results.

The most relevant support paper is:

- Vanselow et al., "Particle image velocimetry in refractive index fields of combustion flows", Experiments in Fluids 2019, DOI `10.1007/s00348-019-2795-1`.

This upgrade would make M3A closer to He Yuanzhe's PIV-BOST line without requiring a full stereo-PIV reconstruction.
