# N1.5-A representation-mismatch headroom screen

Status: **NO_GO_VISIBLE_FORWARD_PREDICTOR**.

This is an opened synthetic development diagnostic. It measures only the mismatch between the continuous analytic-gradient renderer and the voxel finite-difference/trilinear operator. It is not a complete optical camera model, real BOST evidence, or a confirmatory result.

- Independent units: 12 fit, 4 calibration, and 6 development geometry clusters. Two phantom families share each geometry and are not counted as independent rigs.
- Sensor noise and camera bias are disabled to keep representation mismatch separate from sensor covariance.
- Frozen simple comparator selected on calibration only: `component_damping`.
- Deployable ridge inputs: geometry, measured observation, and/or a 12-pair CGLS warm projection. Exact mismatch, truth, clean labels, family labels, and development targets are absent from inference.
- PCA rows use each fresh exact mismatch coefficient and are evaluator-only representational ceilings. They never enter the route gate.
- N1.5-B field/H1 reconstruction gates have not been run.
- OOD, fresh, final, experimental, finite-aperture, and ray-bending claims remain closed.
