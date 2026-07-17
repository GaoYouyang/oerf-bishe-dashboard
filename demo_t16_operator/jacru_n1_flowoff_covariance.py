"""Truth-free flow-off calibration and dense toy covariance proximal tools.

The public payload contains only zero-flow detector repeats, camera grouping,
and immutable identifiers.  Hidden field truth, clean flow observations, and
the nuisance realization used to generate a reconstruction target never enter
the estimator or the discrepancy selector.

The dense proximal solver is intentionally limited to a toy evidence ceiling.
It assembles ``A A^T`` and performs dense solves, so it cannot support a
deployment or runtime claim for volumetric BOST.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Literal

import torch


Tensor = torch.Tensor
FLOWOFF_SCHEMA = "jacru-n1-flowoff-calibration-payload-1.0"
COVARIANCE_SCHEMA = "jacru-n1-camera-random-effect-covariance-1.0"
PROXIMAL_SCHEMA = "jacru-n1-dense-covariance-proximal-ceiling-1.0"
FlowOffMode = Literal["paired_static", "unpaired_distribution"]


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_seed(*parts: object) -> int:
    encoded = _canonical_json(parts).encode("ascii")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") % (2**63 - 1)


def _tensor_digest(*tensors: Tensor, metadata: object) -> str:
    digest = hashlib.sha256(_canonical_json(metadata).encode("ascii"))
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(_canonical_json(tuple(int(v) for v in value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _finite_float(value: float, label: str, *, positive: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return result


def _validated_camera_index(camera_index: Tensor, ray_count: int) -> Tensor:
    value = torch.as_tensor(camera_index).detach().cpu()
    if value.ndim != 1 or value.numel() != int(ray_count):
        raise ValueError("camera_index must contain one entry per ray")
    if value.dtype == torch.bool or value.dtype.is_floating_point or value.is_complex():
        raise TypeError("camera_index must use an integer dtype")
    value = value.to(torch.int64)
    if bool(torch.any(value < 0)):
        raise ValueError("camera_index must be nonnegative")
    unique = torch.unique(value, sorted=True)
    if not torch.equal(unique, torch.arange(unique.numel(), dtype=torch.int64)):
        raise ValueError("camera_index labels must be contiguous from zero")
    return value


def _validated_samples(samples: Tensor, label: str) -> Tensor:
    value = torch.as_tensor(samples).detach().cpu().to(torch.float64)
    if value.ndim != 3 or value.shape[0] < 2 or value.shape[1] < 1 or value.shape[2] != 2:
        raise ValueError(f"{label} must have shape [repeat, ray, 2]")
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError(f"{label} must contain finite values")
    return value.contiguous()


@dataclass(frozen=True)
class JACRUFlowOffCalibrationPayload:
    """Zero-flow repeats exposed to covariance fitting and calibration only."""

    case_id: str
    geometry_digest: str
    mode: FlowOffMode
    camera_index: Tensor
    fit_samples_uv: Tensor
    selection_samples_uv: Tensor
    lock_samples_uv: Tensor
    payload_digest: str
    schema: str = FLOWOFF_SCHEMA

    @property
    def repeat_counts(self) -> dict[str, int]:
        return {
            "fit": int(self.fit_samples_uv.shape[0]),
            "selection": int(self.selection_samples_uv.shape[0]),
            "lock": int(self.lock_samples_uv.shape[0]),
        }

    def estimator_kwargs(self) -> dict[str, Tensor]:
        return {
            "fit_samples_uv": self.fit_samples_uv,
            "camera_index": self.camera_index,
        }


@dataclass(frozen=True)
class CameraRandomEffectCovariance:
    """Low-parameter ``diagonal + camera/component random effect`` model."""

    mean_uv: Tensor
    covariance: Tensor
    camera_component_index: Tensor
    iid_variance_by_group: Tensor
    shared_variance_by_group: Tensor
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    condition_number: float
    fit_repeat_count: int
    shrinkage: float
    ridge_fraction: float
    schema: str = COVARIANCE_SCHEMA


@dataclass(frozen=True)
class DiscrepancyCalibration:
    threshold: float
    quantile: float
    scores: Tensor
    score_mean: float
    score_maximum: float


@dataclass(frozen=True)
class DenseCovarianceProximalResult:
    field: Tensor
    alpha: float
    target_crossed: bool
    raw_discrepancy: float
    selected_discrepancy: float
    correction_norm: float
    measurement_residual_norm: float
    proximal_covariance_scale: float
    residual_closure_relative_error: float
    bisection_iterations: int
    schema: str = PROXIMAL_SCHEMA


def build_flowoff_calibration_payload(
    *,
    case_id: str,
    geometry_digest: str,
    camera_index: Tensor,
    persistent_camera_bias_uv: Tensor,
    iid_noise_std: float,
    camera_bias_std: float,
    mode: FlowOffMode,
    fit_repeats: int,
    selection_repeats: int,
    lock_repeats: int,
    seed: int,
) -> JACRUFlowOffCalibrationPayload:
    """Generate separated flow-off repeats for one frozen synthetic rig.

    ``paired_static`` repeats share the reconstruction target's persistent
    camera/component offset and therefore model a same-session flow-off stack.
    ``unpaired_distribution`` repeats draw an independent offset per frame and
    expose only the nuisance distribution, not the target's realized bias.
    """

    if mode not in {"paired_static", "unpaired_distribution"}:
        raise ValueError("unsupported flow-off mode")
    if not isinstance(case_id, str) or not case_id or not isinstance(geometry_digest, str) or not geometry_digest:
        raise ValueError("case_id and geometry_digest must be nonempty strings")
    counts = {
        "fit": int(fit_repeats),
        "selection": int(selection_repeats),
        "lock": int(lock_repeats),
    }
    if any(value < 2 for value in counts.values()):
        raise ValueError("every flow-off split must contain at least two repeats")
    iid_std = _finite_float(iid_noise_std, "iid_noise_std", positive=True)
    bias_std = _finite_float(camera_bias_std, "camera_bias_std")
    if bias_std < 0.0:
        raise ValueError("camera_bias_std must be nonnegative")
    persistent = torch.as_tensor(persistent_camera_bias_uv).detach().cpu().to(torch.float64)
    if persistent.ndim != 2 or persistent.shape[0] < 1 or persistent.shape[1] != 2:
        raise ValueError("persistent_camera_bias_uv must have shape [camera, 2]")
    if not bool(torch.all(torch.isfinite(persistent))):
        raise ValueError("persistent_camera_bias_uv must contain finite values")
    ray_count = int(torch.as_tensor(camera_index).numel())
    cameras = _validated_camera_index(camera_index, ray_count)
    if int(torch.max(cameras)) + 1 != int(persistent.shape[0]):
        raise ValueError("camera_index and persistent bias disagree on camera count")

    def draw(split: str, repeat_count: int) -> Tensor:
        generator = torch.Generator().manual_seed(
            _stable_seed(FLOWOFF_SCHEMA, case_id, geometry_digest, mode, split, int(seed))
        )
        iid = torch.randn(
            (repeat_count, ray_count, 2), generator=generator, dtype=torch.float64
        ) * iid_std
        if mode == "paired_static":
            frame_bias = persistent[cameras][None, :, :].expand(repeat_count, -1, -1)
        else:
            random_bias = torch.randn(
                (repeat_count, persistent.shape[0], 2),
                generator=generator,
                dtype=torch.float64,
            ) * bias_std
            frame_bias = random_bias[:, cameras, :]
        return (frame_bias + iid).contiguous()

    fit = draw("fit", counts["fit"])
    selection = draw("selection", counts["selection"])
    lock = draw("lock", counts["lock"])
    metadata = {
        "schema": FLOWOFF_SCHEMA,
        "case_id": case_id,
        "geometry_digest": geometry_digest,
        "mode": mode,
        "repeat_counts": counts,
        "seed": int(seed),
    }
    digest = _tensor_digest(
        cameras, fit, selection, lock, metadata=metadata
    )
    return JACRUFlowOffCalibrationPayload(
        case_id=case_id,
        geometry_digest=geometry_digest,
        mode=mode,
        camera_index=cameras,
        fit_samples_uv=fit,
        selection_samples_uv=selection,
        lock_samples_uv=lock,
        payload_digest=digest,
    )


def _camera_component_groups(camera_index: Tensor) -> Tensor:
    camera = _validated_camera_index(camera_index, int(torch.as_tensor(camera_index).numel()))
    component = torch.arange(2, dtype=torch.int64)[None, :].expand(camera.numel(), 2)
    return (2 * camera[:, None] + component).reshape(-1)


def estimate_camera_random_effect_covariance(
    *,
    fit_samples_uv: Tensor,
    camera_index: Tensor,
    shrinkage: float = 0.25,
    ridge_fraction: float = 1e-6,
) -> CameraRandomEffectCovariance:
    """Fit a structured SPD covariance without using field observations."""

    samples = _validated_samples(fit_samples_uv, "fit_samples_uv")
    cameras = _validated_camera_index(camera_index, int(samples.shape[1]))
    shrink = _finite_float(shrinkage, "shrinkage")
    ridge = _finite_float(ridge_fraction, "ridge_fraction", positive=True)
    if not 0.0 <= shrink <= 1.0:
        raise ValueError("shrinkage must lie in [0, 1]")
    groups = _camera_component_groups(cameras)
    flat = samples.reshape(samples.shape[0], -1)
    group_count = int(torch.max(groups)) + 1
    means = torch.zeros_like(flat[0])
    iid_raw = torch.empty(group_count, dtype=torch.float64)
    shared_raw = torch.empty(group_count, dtype=torch.float64)
    for group in range(group_count):
        mask = groups == group
        values = flat[:, mask]
        if values.shape[1] < 2:
            raise ValueError("every camera/component group needs at least two rays")
        scalar_mean = torch.mean(values)
        means[mask] = scalar_mean
        repeat_means = torch.mean(values, dim=1)
        within = values - repeat_means[:, None]
        iid_variance = torch.sum(within.square()) / (
            values.shape[0] * (values.shape[1] - 1)
        )
        mean_variance = torch.var(repeat_means, correction=1)
        shared_variance = torch.clamp(
            mean_variance - iid_variance / values.shape[1], min=0.0
        )
        iid_raw[group] = iid_variance
        shared_raw[group] = shared_variance
    global_iid = torch.mean(iid_raw)
    global_shared = torch.mean(shared_raw)
    iid = (1.0 - shrink) * iid_raw + shrink * global_iid
    shared = (1.0 - shrink) * shared_raw + shrink * global_shared
    floor = max(float(global_iid) * ridge, torch.finfo(torch.float64).eps)
    iid = torch.clamp(iid, min=floor)
    shared = torch.clamp(shared, min=0.0)
    measurement_count = flat.shape[1]
    covariance = torch.zeros(
        (measurement_count, measurement_count), dtype=torch.float64
    )
    for group in range(group_count):
        indices = torch.nonzero(groups == group, as_tuple=False).reshape(-1)
        covariance[indices, indices] += iid[group]
        covariance[indices[:, None], indices[None, :]] += shared[group]
    eigenvalues = torch.linalg.eigvalsh(covariance)
    minimum = float(eigenvalues[0])
    maximum = float(eigenvalues[-1])
    if minimum <= 0.0 or not math.isfinite(maximum):
        raise RuntimeError("structured covariance must remain finite SPD")
    return CameraRandomEffectCovariance(
        mean_uv=means.reshape(samples.shape[1], 2),
        covariance=covariance,
        camera_component_index=groups,
        iid_variance_by_group=iid,
        shared_variance_by_group=shared,
        minimum_eigenvalue=minimum,
        maximum_eigenvalue=maximum,
        condition_number=maximum / minimum,
        fit_repeat_count=int(samples.shape[0]),
        shrinkage=shrink,
        ridge_fraction=ridge,
    )


def exact_camera_random_effect_covariance(
    *,
    camera_index: Tensor,
    iid_noise_std: float,
    camera_bias_std: float,
) -> Tensor:
    """Evaluator-only covariance implied by the synthetic nuisance generator."""

    cameras = _validated_camera_index(camera_index, int(torch.as_tensor(camera_index).numel()))
    iid = _finite_float(iid_noise_std, "iid_noise_std", positive=True) ** 2
    bias_std = _finite_float(camera_bias_std, "camera_bias_std")
    if bias_std < 0.0:
        raise ValueError("camera_bias_std must be nonnegative")
    shared = bias_std**2
    groups = _camera_component_groups(cameras)
    covariance = torch.eye(groups.numel(), dtype=torch.float64) * iid
    for group in range(int(torch.max(groups)) + 1):
        indices = torch.nonzero(groups == group, as_tuple=False).reshape(-1)
        covariance[indices[:, None], indices[None, :]] += shared
    return covariance


def whitened_quadratic(residual: Tensor, covariance: Tensor) -> float:
    vector = torch.as_tensor(residual).detach().cpu().to(torch.float64).reshape(-1)
    matrix = torch.as_tensor(covariance).detach().cpu().to(torch.float64)
    if matrix.shape != (vector.numel(), vector.numel()):
        raise ValueError("covariance must match residual size")
    if not bool(torch.all(torch.isfinite(vector))) or not bool(torch.all(torch.isfinite(matrix))):
        raise ValueError("residual and covariance must be finite")
    try:
        solution = torch.linalg.solve(matrix, vector)
    except RuntimeError as error:
        raise ValueError("covariance must be nonsingular") from error
    value = float(torch.dot(vector, solution))
    if value < -1e-9 or not math.isfinite(value):
        raise ValueError("whitened quadratic must be finite and nonnegative")
    return max(value, 0.0)


def calibrate_discrepancy_threshold(
    *,
    samples_uv: Tensor,
    estimate: CameraRandomEffectCovariance,
    quantile: float = 0.95,
    mean_uv: Tensor | None = None,
) -> DiscrepancyCalibration:
    samples = _validated_samples(samples_uv, "samples_uv")
    center = (
        estimate.mean_uv
        if mean_uv is None
        else torch.as_tensor(mean_uv).detach().cpu().to(torch.float64)
    )
    if tuple(samples.shape[1:]) != tuple(center.shape):
        raise ValueError("samples and covariance mean must share detector shape")
    if not bool(torch.all(torch.isfinite(center))):
        raise ValueError("covariance mean must contain finite values")
    q = _finite_float(quantile, "quantile")
    if not 0.5 <= q < 1.0:
        raise ValueError("quantile must lie in [0.5, 1)")
    scores = torch.tensor(
        [
            whitened_quadratic(sample - center, estimate.covariance)
            for sample in samples
        ],
        dtype=torch.float64,
    )
    threshold = float(torch.quantile(scores, q, interpolation="higher"))
    return DiscrepancyCalibration(
        threshold=threshold,
        quantile=q,
        scores=scores,
        score_mean=float(torch.mean(scores)),
        score_maximum=float(torch.max(scores)),
    )


def lock_coverage(
    *,
    samples_uv: Tensor,
    estimate: CameraRandomEffectCovariance,
    threshold: float,
    mean_uv: Tensor | None = None,
) -> tuple[float, Tensor]:
    samples = _validated_samples(samples_uv, "samples_uv")
    boundary = _finite_float(threshold, "threshold", positive=True)
    center = (
        estimate.mean_uv
        if mean_uv is None
        else torch.as_tensor(mean_uv).detach().cpu().to(torch.float64)
    )
    if tuple(samples.shape[1:]) != tuple(center.shape):
        raise ValueError("samples and covariance mean must share detector shape")
    scores = torch.tensor(
        [
            whitened_quadratic(sample - center, estimate.covariance)
            for sample in samples
        ],
        dtype=torch.float64,
    )
    return float(torch.mean((scores <= boundary).to(torch.float64))), scores


def isotropic_covariance_like(covariance: Tensor) -> Tensor:
    matrix = torch.as_tensor(covariance).detach().cpu().to(torch.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError("covariance must be one nonempty square matrix")
    scale = float(torch.mean(torch.diag(matrix)))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("covariance diagonal mean must be positive")
    return torch.eye(matrix.shape[0], dtype=torch.float64) * scale


def dense_covariance_proximal_discrepancy(
    *,
    initial_field: Tensor,
    target_observation_uv: Tensor,
    dense_active_matrix: Tensor,
    support_mask: Tensor,
    proximal_covariance: Tensor,
    selector_covariance: Tensor,
    discrepancy_threshold: float,
    log10_alpha_bounds: tuple[float, float] = (-12.0, 12.0),
    bisection_iterations: int = 72,
) -> DenseCovarianceProximalResult:
    """Choose the least correction whose calibrated discrepancy crosses."""

    field = torch.as_tensor(initial_field).detach().cpu().to(torch.float64)
    support = torch.as_tensor(support_mask).detach().cpu().to(torch.bool)
    if field.ndim != 3 or support.shape != field.shape or not bool(torch.any(support)):
        raise ValueError("initial_field and nonempty support_mask must share a 3D shape")
    if bool(torch.any(field.masked_select(~support) != 0.0)):
        raise ValueError("initial_field must be zero outside support")
    matrix = torch.as_tensor(dense_active_matrix).detach().cpu().to(torch.float64)
    if matrix.ndim != 2 or matrix.shape[1] != int(torch.count_nonzero(support)):
        raise ValueError("dense_active_matrix must have one column per active voxel")
    target = torch.as_tensor(target_observation_uv).detach().cpu().to(torch.float64).reshape(-1)
    if target.numel() != matrix.shape[0]:
        raise ValueError("target observation must match matrix rows")
    c_prox = torch.as_tensor(proximal_covariance).detach().cpu().to(torch.float64)
    c_select = torch.as_tensor(selector_covariance).detach().cpu().to(torch.float64)
    if c_prox.shape != (matrix.shape[0], matrix.shape[0]) or c_select.shape != c_prox.shape:
        raise ValueError("proximal and selector covariance must match measurement space")
    threshold = _finite_float(discrepancy_threshold, "discrepancy_threshold", positive=True)
    lower_log, upper_log = (float(log10_alpha_bounds[0]), float(log10_alpha_bounds[1]))
    if not math.isfinite(lower_log) or not math.isfinite(upper_log) or lower_log >= upper_log:
        raise ValueError("log10_alpha_bounds must be finite and increasing")
    steps = int(bisection_iterations)
    if steps < 8:
        raise ValueError("bisection_iterations must be at least eight")
    active = field.masked_select(support)
    residual0 = matrix @ active - target
    raw_score = whitened_quadratic(residual0, c_select)
    if raw_score <= threshold:
        return DenseCovarianceProximalResult(
            field=field.clone(),
            alpha=math.inf,
            target_crossed=True,
            raw_discrepancy=raw_score,
            selected_discrepancy=raw_score,
            correction_norm=0.0,
            measurement_residual_norm=float(torch.linalg.vector_norm(residual0)),
            proximal_covariance_scale=0.0,
            residual_closure_relative_error=0.0,
            bisection_iterations=0,
        )
    gram = matrix @ matrix.mT
    mean_gram_diagonal = float(torch.mean(torch.diag(gram)))
    mean_covariance_diagonal = float(torch.mean(torch.diag(c_prox)))
    if mean_gram_diagonal <= 0.0 or mean_covariance_diagonal <= 0.0:
        raise ValueError("Gram and covariance diagonal means must be positive")
    covariance_scale = mean_gram_diagonal / mean_covariance_diagonal
    scaled_covariance = covariance_scale * c_prox

    def solve_at(log10_alpha: float) -> tuple[Tensor, Tensor, Tensor, float]:
        alpha = 10.0 ** log10_alpha
        dual = torch.linalg.solve(gram + alpha * scaled_covariance, residual0)
        candidate = active - matrix.mT @ dual
        residual = matrix @ candidate - target
        score = whitened_quadratic(residual, c_select)
        return candidate, residual, dual, score

    low_candidate, low_residual, low_dual, low_score = solve_at(lower_log)
    if low_score > threshold * (1.0 + 1e-8):
        alpha = 10.0**lower_log
        closure = low_residual - alpha * (scaled_covariance @ low_dual)
        closure_error = float(torch.linalg.vector_norm(closure)) / max(
            float(torch.linalg.vector_norm(residual0)), 1e-30
        )
        full = torch.zeros_like(field)
        full.masked_scatter_(support, low_candidate)
        return DenseCovarianceProximalResult(
            field=full,
            alpha=alpha,
            target_crossed=False,
            raw_discrepancy=raw_score,
            selected_discrepancy=low_score,
            correction_norm=float(torch.linalg.vector_norm(active - low_candidate)),
            measurement_residual_norm=float(torch.linalg.vector_norm(low_residual)),
            proximal_covariance_scale=covariance_scale,
            residual_closure_relative_error=closure_error,
            bisection_iterations=0,
        )
    _, _, _, upper_score = solve_at(upper_log)
    if upper_score <= threshold:
        raise RuntimeError("upper alpha bound failed to bracket raw discrepancy")
    low = lower_log
    high = upper_log
    selected_candidate = low_candidate
    selected_residual = low_residual
    selected_dual = low_dual
    selected_score = low_score
    for _ in range(steps):
        midpoint = 0.5 * (low + high)
        candidate, residual, dual, score = solve_at(midpoint)
        if score <= threshold:
            low = midpoint
            selected_candidate = candidate
            selected_residual = residual
            selected_dual = dual
            selected_score = score
        else:
            high = midpoint
    full = torch.zeros_like(field)
    full.masked_scatter_(support, selected_candidate)
    selected_alpha = 10.0**low
    closure = selected_residual - selected_alpha * (
        scaled_covariance @ selected_dual
    )
    closure_error = float(torch.linalg.vector_norm(closure)) / max(
        float(torch.linalg.vector_norm(residual0)), 1e-30
    )
    return DenseCovarianceProximalResult(
        field=full,
        alpha=selected_alpha,
        target_crossed=True,
        raw_discrepancy=raw_score,
        selected_discrepancy=selected_score,
        correction_norm=float(torch.linalg.vector_norm(active - selected_candidate)),
        measurement_residual_norm=float(torch.linalg.vector_norm(selected_residual)),
        proximal_covariance_scale=covariance_scale,
        residual_closure_relative_error=closure_error,
        bisection_iterations=steps,
    )
