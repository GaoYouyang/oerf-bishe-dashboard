# N2-ADRC-N1 development pilot

**Machine decision:** `DEVELOPMENT_ONLY_NO_AUDIT_AUTHORIZATION`

Promotion screen met: **True** (this permits design of an unseen audit only).

| Case | residual/high variance | predicted gain | conservative timing | empirical MSE gain (95% bootstrap) | JVP rel. err. | screen |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| smooth_gradient_only | 0.004219 | 1.809x | 1.397x | 1.784x [1.626, 1.960] | 1.81e-09 | True |
| wrinkled_gradient_only | 0.004418 | 1.798x | 1.512x | 1.744x [1.559, 1.950] | 1.58e-09 | False |
| smooth_coupled_bend | 0.02663 | 1.316x | 1.102x | 1.359x [1.238, 1.492] | 5.02e-09 | True |
| wrinkled_coupled_bend | 0.01921 | 1.461x | 1.195x | 1.396x [1.234, 1.565] | 7.26e-10 | False |

## Interpretation boundary

The target is a frozen finite-population high-renderer mean. The fields are gridded analytic morphology proxies. The bend is prescribed and has no field-dependent trajectory derivative. Timings are CPU batch proxies. No neural field, reconstruction, experimental data, or held-out audit is present.

The plug-in and double-sample gradient columns are Monte Carlo diagnostics. The double-sample construction is unbiased in expectation; a smaller observed bias in one finite run is not itself a proof.

Reserved audit families were not opened: oblique_compression_sheet, shock_expansion_pair.
