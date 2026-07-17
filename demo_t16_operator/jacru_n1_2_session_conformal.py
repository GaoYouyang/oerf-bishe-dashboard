"""Isolated session-level conformal core for the JACRU N1.2 protocol.

The module deliberately stops at calibration.  It creates one synthetic
session, fits only flow-off repeats, freezes candidate-specific discrepancy
bands, and audits those bands on an untouched split.  No field labels, clean
observations, or realized nuisance parameters are carried by a calibration or
selector payload.

All numerical work is CPU ``torch.float64``.  The structured covariance fit is
the frozen N1.1 estimator; N1.2 adds session construction, exact finite-sample
order statistics, camera-local bands, and independent audit accounting.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Literal

import torch

from .jacru_n1_flowoff_covariance import (
    estimate_camera_random_effect_covariance,
    isotropic_covariance_like,
)


Tensor = torch.Tensor
CovariancePolicy = Literal["structured", "isotropic"]
MeanPolicy = Literal["estimated", "zero"]

SESSION_SCHEMA = "jacru-n1-2-session-packet-1.0"
FIT_SCHEMA = "jacru-n1-2-session-flowoff-fit-1.0"
CALIBRATION_SCHEMA = "jacru-n1-2-threshold-calibration-1.0"
AUDIT_SCHEMA = "jacru-n1-2-independent-audit-1.0"
SELECTOR_SCHEMA = "jacru-n1-2-capability-limited-selector-1.0"
DEFAULT_TWO_SIDED_COVERAGE = 0.95
WILSON_FALLBACK_NOTE = (
    "When scipy is unavailable, binomial intervals use the two-sided Wilson "
    "score interval with a normal critical value from statistics.NormalDist."
)


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
    except (TypeError, ValueError) as error:
        raise TypeError("stable metadata must be JSON serializable") from error


def stable_seed(*parts: object) -> int:
    """Return a process-independent 63-bit seed for JSON-compatible parts."""

    encoded = _canonical_json(parts).encode("ascii")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") % (
        2**63 - 1
    )


def stable_digest(*tensors: Tensor, metadata: object = None) -> str:
    """Hash tensor values, dtypes, shapes, and canonical JSON metadata."""

    digest = hashlib.sha256(_canonical_json(metadata).encode("ascii"))
    for tensor in tensors:
        value = torch.as_tensor(tensor).detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(
            _canonical_json(tuple(int(item) for item in value.shape)).encode("ascii")
        )
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _owned_tensor(value: Tensor, *, dtype: torch.dtype) -> Tensor:
    return torch.as_tensor(value).detach().cpu().to(dtype).clone().contiguous()


def _finite_float(value: float, label: str, *, positive: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return result


def _validated_identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _validated_seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("seed must be an integer")
    if value < 0:
        raise ValueError("seed must be nonnegative")
    return value


def _validated_camera_index(camera_index: Tensor, ray_count: int) -> Tensor:
    value = torch.as_tensor(camera_index).detach().cpu()
    if value.ndim != 1 or value.numel() != int(ray_count):
        raise ValueError("camera_index must contain one entry per ray")
    if value.dtype == torch.bool or value.dtype.is_floating_point or value.is_complex():
        raise TypeError("camera_index must use an integer dtype")
    value = value.to(torch.int64).clone().contiguous()
    if bool(torch.any(value < 0)):
        raise ValueError("camera_index must be nonnegative")
    unique = torch.unique(value, sorted=True)
    if not torch.equal(unique, torch.arange(unique.numel(), dtype=torch.int64)):
        raise ValueError("camera_index labels must be contiguous from zero")
    counts = torch.bincount(value)
    if bool(torch.any(counts < 2)):
        raise ValueError("every camera must contain at least two rays")
    return value


def _validated_samples(samples_uv: Tensor, label: str, *, minimum: int = 1) -> Tensor:
    value = _owned_tensor(samples_uv, dtype=torch.float64)
    if (
        value.ndim != 3
        or value.shape[0] < int(minimum)
        or value.shape[1] < 1
        or value.shape[2] != 2
    ):
        raise ValueError(f"{label} must have shape [sample, ray, 2]")
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError(f"{label} must contain finite values")
    return value


def _payload_digest(
    schema: str,
    manifest_digest: str,
    session_id: str,
    geometry_digest: str,
    camera_index: Tensor,
    samples_uv: Tensor,
) -> str:
    return stable_digest(
        camera_index,
        samples_uv,
        metadata={
            "schema": schema,
            "manifest_digest": manifest_digest,
            "session_id": session_id,
            "geometry_digest": geometry_digest,
        },
    )


@dataclass(frozen=True)
class SessionFlowOffFitPayload:
    """The only session data available to covariance fitting."""

    manifest_digest: str
    session_id: str
    geometry_digest: str
    camera_index: Tensor
    flow_off_samples_uv: Tensor
    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "camera_index", _owned_tensor(self.camera_index, dtype=torch.int64))
        object.__setattr__(
            self,
            "flow_off_samples_uv",
            _owned_tensor(self.flow_off_samples_uv, dtype=torch.float64),
        )

    def estimator_kwargs(self) -> dict[str, Tensor]:
        _verify_flowoff_payload(self, FIT_SCHEMA)
        return {
            "fit_samples_uv": self.flow_off_samples_uv.clone(),
            "camera_index": self.camera_index.clone(),
        }


@dataclass(frozen=True)
class SessionThresholdCalibrationPayload:
    """Flow-off repeats reserved solely for threshold calibration."""

    manifest_digest: str
    session_id: str
    geometry_digest: str
    camera_index: Tensor
    flow_off_samples_uv: Tensor
    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "camera_index", _owned_tensor(self.camera_index, dtype=torch.int64))
        object.__setattr__(
            self,
            "flow_off_samples_uv",
            _owned_tensor(self.flow_off_samples_uv, dtype=torch.float64),
        )


@dataclass(frozen=True)
class SessionAuditPayload:
    """Untouched flow-off repeats available only to coverage auditing."""

    manifest_digest: str
    session_id: str
    geometry_digest: str
    camera_index: Tensor
    flow_off_samples_uv: Tensor
    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "camera_index", _owned_tensor(self.camera_index, dtype=torch.int64))
        object.__setattr__(
            self,
            "flow_off_samples_uv",
            _owned_tensor(self.flow_off_samples_uv, dtype=torch.float64),
        )


@dataclass(frozen=True)
class SessionPacket:
    """One session's deployable observations and three isolated flow-off splits."""

    manifest_id: str
    manifest_digest: str
    session_id: str
    geometry_digest: str
    base_seed: int
    field_ids: tuple[str, ...]
    flow_on_observations_uv: Tensor
    fit: SessionFlowOffFitPayload
    calibration: SessionThresholdCalibrationPayload
    audit: SessionAuditPayload
    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "flow_on_observations_uv",
            _owned_tensor(self.flow_on_observations_uv, dtype=torch.float64),
        )


def verify_session_packet(packet: SessionPacket) -> None:
    """Fail closed if any mutable tensor or child payload changed after build."""

    _validated_identifier(packet.manifest_id, "manifest_id")
    _validated_identifier(packet.manifest_digest, "manifest_digest")
    _validated_identifier(packet.session_id, "session_id")
    _validated_identifier(packet.geometry_digest, "geometry_digest")
    _validated_seed(packet.base_seed)
    if len(packet.field_ids) not in {2, 3} or len(set(packet.field_ids)) != len(
        packet.field_ids
    ):
        raise ValueError("session packet must retain two or three unique field_ids")
    flow_on = _validated_samples(packet.flow_on_observations_uv, "flow_on_observations_uv")
    if flow_on.shape[0] != len(packet.field_ids):
        raise ValueError("session packet field count drifted")
    for payload, schema in (
        (packet.fit, FIT_SCHEMA),
        (packet.calibration, CALIBRATION_SCHEMA),
        (packet.audit, AUDIT_SCHEMA),
    ):
        _verify_flowoff_payload(payload, schema)
        if (
            payload.manifest_digest != packet.manifest_digest
            or payload.session_id != packet.session_id
            or payload.geometry_digest != packet.geometry_digest
        ):
            raise ValueError("session child payload provenance drifted")
    expected = stable_digest(
        flow_on,
        metadata={
            "schema": SESSION_SCHEMA,
            "manifest_digest": packet.manifest_digest,
            "session_id": packet.session_id,
            "geometry_digest": packet.geometry_digest,
            "fit_digest": packet.fit.digest,
            "calibration_digest": packet.calibration.digest,
            "audit_digest": packet.audit.digest,
        },
    )
    if packet.digest != expected:
        raise ValueError("session packet digest mismatch; packet was mutated")


@dataclass(frozen=True)
class SelectorPayload:
    """Exact capability boundary passed to a discrepancy selector."""

    candidate_id: str
    manifest_digest: str
    session_id: str
    geometry_digest: str
    source_fit_digest: str
    source_calibration_digest: str
    mean_uv: Tensor
    proximal_covariance: Tensor
    selector_covariance: Tensor
    global_bands: Tensor
    per_camera_bands: Tensor
    score_anchors: Tensor
    calibration_sample_count: int
    joint_tail_order: int
    joint_tail_quantile: float
    joint_upper_limit: float
    joint_lower_limit: float
    camera_index: Tensor
    digest: str

    def __post_init__(self) -> None:
        for name in (
            "mean_uv",
            "proximal_covariance",
            "selector_covariance",
            "global_bands",
            "per_camera_bands",
            "score_anchors",
        ):
            object.__setattr__(self, name, _owned_tensor(getattr(self, name), dtype=torch.float64))
        object.__setattr__(self, "camera_index", _owned_tensor(self.camera_index, dtype=torch.int64))


@dataclass(frozen=True)
class BinomialCoverage:
    successes: int
    sample_count: int
    coverage: float
    confidence_interval: tuple[float, float]
    interval_method: str


@dataclass(frozen=True)
class CandidateAuditCoverage:
    global_band: BinomialCoverage
    global_upper: BinomialCoverage
    global_lower: BinomialCoverage
    per_camera_bands: tuple[BinomialCoverage, ...]
    joint_upper: BinomialCoverage
    joint_lower: BinomialCoverage
    joint_band: BinomialCoverage
    nominal_two_sided_coverage_lower_bound: float
    digest: str


@dataclass(frozen=True)
class SelectorResidualScores:
    global_score: float
    per_camera_scores: Tensor

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "per_camera_scores",
            _owned_tensor(self.per_camera_scores, dtype=torch.float64),
        )


def _verify_flowoff_payload(
    payload: SessionFlowOffFitPayload | SessionThresholdCalibrationPayload | SessionAuditPayload,
    schema: str,
) -> None:
    _validated_identifier(payload.manifest_digest, "manifest_digest")
    _validated_identifier(payload.session_id, "session_id")
    _validated_identifier(payload.geometry_digest, "geometry_digest")
    samples = _validated_samples(payload.flow_off_samples_uv, "flow_off_samples_uv")
    cameras = _validated_camera_index(payload.camera_index, int(samples.shape[1]))
    expected = _payload_digest(
        schema,
        payload.manifest_digest,
        payload.session_id,
        payload.geometry_digest,
        cameras,
        samples,
    )
    if payload.digest != expected:
        raise ValueError("flow-off payload digest mismatch; payload was mutated")


def _make_payload(
    payload_type: type[SessionFlowOffFitPayload]
    | type[SessionThresholdCalibrationPayload]
    | type[SessionAuditPayload],
    schema: str,
    manifest_digest: str,
    session_id: str,
    geometry_digest: str,
    camera_index: Tensor,
    samples_uv: Tensor,
) -> SessionFlowOffFitPayload | SessionThresholdCalibrationPayload | SessionAuditPayload:
    digest = _payload_digest(
        schema,
        manifest_digest,
        session_id,
        geometry_digest,
        camera_index,
        samples_uv,
    )
    return payload_type(
        manifest_digest=manifest_digest,
        session_id=session_id,
        geometry_digest=geometry_digest,
        camera_index=camera_index,
        flow_off_samples_uv=samples_uv,
        digest=digest,
    )


def build_session_packet(
    *,
    manifest_id: str,
    session_id: str,
    geometry_digest: str,
    field_ids: tuple[str, ...],
    camera_index: Tensor,
    flow_on_fields_uv: Tensor,
    session_scale_uv: Tensor,
    camera_bias_relative_std: float = 1.0,
    frame_camera_jitter_relative_std: float = 0.5,
    fit_repeats: int = 64,
    calibration_repeats: int = 64,
    audit_repeats: int = 64,
    seed: int = 0,
) -> SessionPacket:
    """Build one session without deriving noise scale from any field.

    ``session_scale_uv`` is an explicit, deployment-visible heteroscedastic
    scale.  It alone controls innovation variance and the scale of the one
    persistent camera/component offset.  Consequently changing a supplied
    flow-on field changes only the signal term, never calibration samples or
    nuisance draws.
    """

    manifest = _validated_identifier(manifest_id, "manifest_id")
    session = _validated_identifier(session_id, "session_id")
    geometry = _validated_identifier(geometry_digest, "geometry_digest")
    base_seed = _validated_seed(seed)
    fields = _validated_samples(flow_on_fields_uv, "flow_on_fields_uv")
    identifiers = tuple(
        _validated_identifier(value, f"field_ids[{index}]")
        for index, value in enumerate(field_ids)
    )
    if len(identifiers) not in {2, 3}:
        raise ValueError("one session must contain exactly two or three fields")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("field_ids must be unique within a session")
    if fields.shape[0] != len(identifiers):
        raise ValueError("flow_on_fields_uv must contain one field per field_id")
    cameras = _validated_camera_index(camera_index, int(fields.shape[1]))
    scale = _owned_tensor(session_scale_uv, dtype=torch.float64)
    if scale.shape != fields.shape[1:]:
        raise ValueError("session_scale_uv must have shape [ray, 2]")
    if not bool(torch.all(torch.isfinite(scale))) or bool(torch.any(scale <= 0.0)):
        raise ValueError("session_scale_uv must contain positive finite values")
    bias_relative = _finite_float(
        camera_bias_relative_std, "camera_bias_relative_std"
    )
    if bias_relative < 0.0:
        raise ValueError("camera_bias_relative_std must be nonnegative")
    jitter_relative = _finite_float(
        frame_camera_jitter_relative_std,
        "frame_camera_jitter_relative_std",
    )
    if jitter_relative < 0.0:
        raise ValueError("frame_camera_jitter_relative_std must be nonnegative")
    counts = {
        "fit": int(fit_repeats),
        "calibration": int(calibration_repeats),
        "audit": int(audit_repeats),
    }
    if any(count < 2 for count in counts.values()):
        raise ValueError("every flow-off split must contain at least two repeats")

    camera_count = int(torch.max(cameras)) + 1
    camera_scale = torch.empty((camera_count, 2), dtype=torch.float64)
    for camera in range(camera_count):
        camera_scale[camera] = torch.sqrt(
            torch.mean(scale[cameras == camera].square(), dim=0)
        )
    manifest_digest = stable_digest(
        cameras,
        scale,
        metadata={
            "schema": SESSION_SCHEMA,
            "manifest_id": manifest,
            "session_id": session,
            "geometry_digest": geometry,
            "base_seed": base_seed,
            "field_ids": identifiers,
            "camera_bias_relative_std": bias_relative,
            "frame_camera_jitter_relative_std": jitter_relative,
            "fit_repeats": counts["fit"],
            "calibration_repeats": counts["calibration"],
            "audit_repeats": counts["audit"],
        },
    )
    bias_generator = torch.Generator().manual_seed(
        stable_seed(SESSION_SCHEMA, session, geometry, "persistent-bias", base_seed)
    )
    persistent_bias = (
        torch.randn((camera_count, 2), generator=bias_generator, dtype=torch.float64)
        * camera_scale
        * bias_relative
    )
    ray_bias = persistent_bias[cameras]

    def innovations(stream: str, sample_count: int) -> Tensor:
        iid_generator = torch.Generator().manual_seed(
            stable_seed(SESSION_SCHEMA, session, geometry, stream, "iid", base_seed)
        )
        iid = torch.randn(
            (sample_count, fields.shape[1], 2),
            generator=iid_generator,
            dtype=torch.float64,
        ) * scale[None, :, :]
        drift_generator = torch.Generator().manual_seed(
            stable_seed(SESSION_SCHEMA, session, geometry, stream, "camera-drift", base_seed)
        )
        drift = (
            torch.randn(
                (sample_count, camera_count, 2),
                generator=drift_generator,
                dtype=torch.float64,
            )
            * camera_scale[None, :, :]
            * jitter_relative
        )
        return iid + drift[:, cameras, :]

    flow_on = fields + ray_bias[None, :, :] + innovations("flow-on", fields.shape[0])
    fit_samples = ray_bias[None, :, :] + innovations("fit", counts["fit"])
    calibration_samples = ray_bias[None, :, :] + innovations(
        "threshold-calibration", counts["calibration"]
    )
    audit_samples = ray_bias[None, :, :] + innovations("audit", counts["audit"])

    fit = _make_payload(
        SessionFlowOffFitPayload,
        FIT_SCHEMA,
        manifest_digest,
        session,
        geometry,
        cameras,
        fit_samples,
    )
    calibration = _make_payload(
        SessionThresholdCalibrationPayload,
        CALIBRATION_SCHEMA,
        manifest_digest,
        session,
        geometry,
        cameras,
        calibration_samples,
    )
    audit = _make_payload(
        SessionAuditPayload,
        AUDIT_SCHEMA,
        manifest_digest,
        session,
        geometry,
        cameras,
        audit_samples,
    )
    packet_digest = stable_digest(
        flow_on,
        metadata={
            "schema": SESSION_SCHEMA,
            "manifest_digest": manifest_digest,
            "session_id": session,
            "geometry_digest": geometry,
            "fit_digest": fit.digest,
            "calibration_digest": calibration.digest,
            "audit_digest": audit.digest,
        },
    )
    return SessionPacket(
        manifest_id=manifest,
        manifest_digest=manifest_digest,
        session_id=session,
        geometry_digest=geometry,
        base_seed=base_seed,
        field_ids=identifiers,
        flow_on_observations_uv=flow_on,
        fit=fit,
        calibration=calibration,
        audit=audit,
        digest=packet_digest,
    )


def finite_sample_conformal_order(n: int, q: float) -> int:
    """Return the one-based order ``ceil((n + 1) q)`` or fail closed."""

    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    quantile = _finite_float(q, "q")
    if not 0.0 < quantile < 1.0:
        raise ValueError("q must lie in (0, 1)")
    order = int(math.ceil((n + 1) * quantile))
    if order > n:
        raise ValueError("requested conformal order exceeds sample count")
    return order


def finite_sample_conformal_threshold(scores: Tensor, q: float) -> float:
    """Return the exact kth order statistic, with no interpolation."""

    values = _owned_tensor(scores, dtype=torch.float64)
    if values.ndim != 1 or values.numel() < 1:
        raise ValueError("scores must be one nonempty vector")
    if not bool(torch.all(torch.isfinite(values))):
        raise ValueError("scores must contain finite values")
    order = finite_sample_conformal_order(int(values.numel()), q)
    return float(torch.kthvalue(values, order).values)


def _validated_covariance(covariance: Tensor, measurement_count: int, label: str) -> Tensor:
    value = _owned_tensor(covariance, dtype=torch.float64)
    if value.shape != (measurement_count, measurement_count):
        raise ValueError(f"{label} must match flattened measurement size")
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError(f"{label} must contain finite values")
    if not torch.allclose(value, value.mT, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{label} must be symmetric")
    try:
        torch.linalg.cholesky(value)
    except RuntimeError as error:
        raise ValueError(f"{label} must be positive definite") from error
    return value


def _score_samples(
    samples_uv: Tensor,
    mean_uv: Tensor,
    covariance: Tensor,
    camera_index: Tensor,
) -> tuple[Tensor, Tensor]:
    samples = _validated_samples(samples_uv, "samples_uv")
    center = _owned_tensor(mean_uv, dtype=torch.float64)
    if center.shape != samples.shape[1:]:
        raise ValueError("mean_uv must match detector shape")
    cameras = _validated_camera_index(camera_index, int(samples.shape[1]))
    measurement_count = int(samples.shape[1] * 2)
    matrix = _validated_covariance(covariance, measurement_count, "selector_covariance")
    residual = (samples - center[None, :, :]).reshape(samples.shape[0], -1)
    global_solution = torch.linalg.solve(matrix, residual.mT).mT
    global_scores = torch.sum(residual * global_solution, dim=1)
    camera_count = int(torch.max(cameras)) + 1
    camera_scores = torch.empty(
        (samples.shape[0], camera_count), dtype=torch.float64
    )
    flat_camera = cameras[:, None].expand(-1, 2).reshape(-1)
    for camera in range(camera_count):
        indices = torch.nonzero(flat_camera == camera, as_tuple=False).reshape(-1)
        local_residual = residual[:, indices]
        local_covariance = matrix[indices[:, None], indices[None, :]]
        local_solution = torch.linalg.solve(local_covariance, local_residual.mT).mT
        camera_scores[:, camera] = torch.sum(
            local_residual * local_solution, dim=1
        )
    return global_scores, camera_scores


def _selector_digest(selector: SelectorPayload) -> str:
    return stable_digest(
        selector.mean_uv,
        selector.proximal_covariance,
        selector.selector_covariance,
        selector.global_bands,
        selector.per_camera_bands,
        selector.score_anchors,
        selector.camera_index,
        metadata={
            "schema": SELECTOR_SCHEMA,
            "candidate_id": selector.candidate_id,
            "manifest_digest": selector.manifest_digest,
            "session_id": selector.session_id,
            "geometry_digest": selector.geometry_digest,
            "source_fit_digest": selector.source_fit_digest,
            "source_calibration_digest": selector.source_calibration_digest,
            "joint_tail_quantile": selector.joint_tail_quantile,
            "calibration_sample_count": selector.calibration_sample_count,
            "joint_tail_order": selector.joint_tail_order,
            "joint_upper_limit": selector.joint_upper_limit,
            "joint_lower_limit": selector.joint_lower_limit,
        },
    )


def _verify_selector(selector: SelectorPayload) -> None:
    _validated_identifier(selector.candidate_id, "selector candidate_id")
    _validated_identifier(selector.manifest_digest, "selector manifest_digest")
    _validated_identifier(selector.session_id, "selector session_id")
    _validated_identifier(selector.geometry_digest, "selector geometry_digest")
    _validated_identifier(selector.source_fit_digest, "selector source_fit_digest")
    _validated_identifier(
        selector.source_calibration_digest,
        "selector source_calibration_digest",
    )
    ray_count = int(selector.mean_uv.shape[0]) if selector.mean_uv.ndim == 2 else -1
    cameras = _validated_camera_index(selector.camera_index, ray_count)
    if selector.mean_uv.shape != (ray_count, 2):
        raise ValueError("selector mean_uv must have shape [ray, 2]")
    measurement_count = 2 * ray_count
    _validated_covariance(
        selector.proximal_covariance, measurement_count, "proximal_covariance"
    )
    _validated_covariance(
        selector.selector_covariance, measurement_count, "selector_covariance"
    )
    camera_count = int(torch.max(cameras)) + 1
    if selector.global_bands.shape != (2,):
        raise ValueError("global_bands must have shape [2]")
    if selector.per_camera_bands.shape != (camera_count, 2):
        raise ValueError("per_camera_bands must have shape [camera, 2]")
    if selector.score_anchors.shape != (camera_count + 1,):
        raise ValueError("score_anchors must contain global then per-camera anchors")
    if not bool(torch.all(torch.isfinite(selector.score_anchors))) or bool(
        torch.any(selector.score_anchors <= 0.0)
    ):
        raise ValueError("score_anchors must be positive finite")
    tail_quantile = _finite_float(
        selector.joint_tail_quantile, "joint_tail_quantile"
    )
    if not 0.5 < tail_quantile < 1.0:
        raise ValueError("joint_tail_quantile must lie in (0.5, 1)")
    expected_order = finite_sample_conformal_order(
        int(selector.calibration_sample_count), tail_quantile
    )
    if int(selector.joint_tail_order) != expected_order:
        raise ValueError("joint_tail_order does not match finite-sample contract")
    _finite_float(selector.joint_upper_limit, "joint_upper_limit", positive=True)
    _finite_float(selector.joint_lower_limit, "joint_lower_limit", positive=True)
    if not bool(selector.global_bands[0] <= selector.global_bands[1]) or not bool(
        torch.all(selector.per_camera_bands[:, 0] <= selector.per_camera_bands[:, 1])
    ):
        raise ValueError("selector score bands must be increasing")
    if selector.digest != _selector_digest(selector):
        raise ValueError("selector payload digest mismatch; payload was mutated")


def calibrate_candidate_selector(
    *,
    candidate_id: str = "anonymous_candidate",
    fit_payload: SessionFlowOffFitPayload,
    calibration_payload: SessionThresholdCalibrationPayload,
    proximal_covariance_policy: CovariancePolicy = "structured",
    selector_covariance_policy: CovariancePolicy = "structured",
    mean_policy: MeanPolicy = "estimated",
    shrinkage: float = 0.25,
    ridge_fraction: float = 1e-6,
    target_two_sided_coverage: float = DEFAULT_TWO_SIDED_COVERAGE,
) -> SelectorPayload:
    """Fit and calibrate one candidate without touching its audit split."""

    _verify_flowoff_payload(fit_payload, FIT_SCHEMA)
    _verify_flowoff_payload(calibration_payload, CALIBRATION_SCHEMA)
    if (
        fit_payload.session_id != calibration_payload.session_id
        or fit_payload.geometry_digest != calibration_payload.geometry_digest
        or fit_payload.manifest_digest != calibration_payload.manifest_digest
        or not torch.equal(fit_payload.camera_index, calibration_payload.camera_index)
    ):
        raise ValueError("fit and calibration payloads must describe one session")
    policies = {"structured", "isotropic"}
    if proximal_covariance_policy not in policies:
        raise ValueError("unsupported proximal covariance policy")
    if selector_covariance_policy not in policies:
        raise ValueError("unsupported selector covariance policy")
    if mean_policy not in {"estimated", "zero"}:
        raise ValueError("unsupported mean policy")
    candidate_name = _validated_identifier(candidate_id, "candidate_id")
    two_sided = _finite_float(
        target_two_sided_coverage, "target_two_sided_coverage"
    )
    if not 0.5 < two_sided < 1.0:
        raise ValueError("target_two_sided_coverage must lie in (0.5, 1)")
    tail_quantile = 1.0 - 0.5 * (1.0 - two_sided)

    estimate = estimate_camera_random_effect_covariance(
        **fit_payload.estimator_kwargs(),
        shrinkage=shrinkage,
        ridge_fraction=ridge_fraction,
    )
    structured = estimate.covariance.detach().cpu().to(torch.float64)
    isotropic = isotropic_covariance_like(structured)
    proximal = structured if proximal_covariance_policy == "structured" else isotropic
    selector_covariance = (
        structured if selector_covariance_policy == "structured" else isotropic
    )
    mean = estimate.mean_uv if mean_policy == "estimated" else torch.zeros_like(estimate.mean_uv)
    fit_global_scores, fit_camera_scores = _score_samples(
        fit_payload.flow_off_samples_uv,
        mean,
        selector_covariance,
        fit_payload.camera_index,
    )
    global_scores, camera_scores = _score_samples(
        calibration_payload.flow_off_samples_uv,
        mean,
        selector_covariance,
        fit_payload.camera_index,
    )
    score_anchors = torch.cat(
        (
            torch.median(fit_global_scores).reshape(1),
            torch.median(fit_camera_scores, dim=0).values,
        )
    )
    score_anchors = torch.clamp(
        score_anchors,
        min=torch.finfo(torch.float64).tiny,
    )
    combined_scores = torch.cat((global_scores[:, None], camera_scores), dim=1)
    upper_nonconformity = torch.max(
        combined_scores / score_anchors[None, :], dim=1
    ).values
    lower_nonconformity = torch.max(
        score_anchors[None, :]
        / combined_scores.clamp_min(torch.finfo(torch.float64).tiny),
        dim=1,
    ).values
    upper_limit = finite_sample_conformal_threshold(
        upper_nonconformity, tail_quantile
    )
    lower_limit = finite_sample_conformal_threshold(
        lower_nonconformity, tail_quantile
    )
    calibration_sample_count = int(global_scores.numel())
    joint_tail_order = finite_sample_conformal_order(
        calibration_sample_count, tail_quantile
    )
    lower_bands = score_anchors / lower_limit
    upper_bands = score_anchors * upper_limit
    global_bands = torch.stack((lower_bands[0], upper_bands[0]))
    per_camera_bands = torch.stack(
        (lower_bands[1:], upper_bands[1:]), dim=1
    )
    provisional = SelectorPayload(
        candidate_id=candidate_name,
        manifest_digest=fit_payload.manifest_digest,
        session_id=fit_payload.session_id,
        geometry_digest=fit_payload.geometry_digest,
        source_fit_digest=fit_payload.digest,
        source_calibration_digest=calibration_payload.digest,
        mean_uv=mean,
        proximal_covariance=proximal,
        selector_covariance=selector_covariance,
        global_bands=global_bands,
        per_camera_bands=per_camera_bands,
        score_anchors=score_anchors,
        calibration_sample_count=calibration_sample_count,
        joint_tail_order=joint_tail_order,
        joint_tail_quantile=tail_quantile,
        joint_upper_limit=upper_limit,
        joint_lower_limit=lower_limit,
        camera_index=fit_payload.camera_index,
        digest="",
    )
    return SelectorPayload(
        candidate_id=provisional.candidate_id,
        manifest_digest=provisional.manifest_digest,
        session_id=provisional.session_id,
        geometry_digest=provisional.geometry_digest,
        source_fit_digest=provisional.source_fit_digest,
        source_calibration_digest=provisional.source_calibration_digest,
        mean_uv=provisional.mean_uv,
        proximal_covariance=provisional.proximal_covariance,
        selector_covariance=provisional.selector_covariance,
        global_bands=provisional.global_bands,
        per_camera_bands=provisional.per_camera_bands,
        score_anchors=provisional.score_anchors,
        calibration_sample_count=provisional.calibration_sample_count,
        joint_tail_order=provisional.joint_tail_order,
        joint_tail_quantile=provisional.joint_tail_quantile,
        joint_upper_limit=provisional.joint_upper_limit,
        joint_lower_limit=provisional.joint_lower_limit,
        camera_index=provisional.camera_index,
        digest=_selector_digest(provisional),
    )


def binomial_confidence_interval(
    successes: int,
    sample_count: int,
    confidence: float = 0.95,
) -> tuple[float, float, str]:
    """Return an exact Clopper-Pearson interval, or documented Wilson fallback.

    SciPy is optional.  If its exact binomial implementation cannot be
    imported, the function uses the two-sided Wilson score interval described
    by :data:`WILSON_FALLBACK_NOTE`.
    """

    if isinstance(successes, bool) or not isinstance(successes, int):
        raise TypeError("successes must be an integer")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise TypeError("sample_count must be an integer")
    if sample_count < 1 or successes < 0 or successes > sample_count:
        raise ValueError("successes must lie in [0, sample_count] with sample_count positive")
    level = _finite_float(confidence, "confidence")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    try:
        from scipy.stats import binomtest

        interval = binomtest(successes, sample_count).proportion_ci(
            confidence_level=level, method="exact"
        )
        return float(interval.low), float(interval.high), "clopper-pearson-exact"
    except ImportError:
        from statistics import NormalDist

        z = NormalDist().inv_cdf(0.5 + 0.5 * level)
        proportion = successes / sample_count
        denominator = 1.0 + z * z / sample_count
        center = (proportion + z * z / (2.0 * sample_count)) / denominator
        radius = (
            z
            * math.sqrt(
                proportion * (1.0 - proportion) / sample_count
                + z * z / (4.0 * sample_count * sample_count)
            )
            / denominator
        )
        return max(0.0, center - radius), min(1.0, center + radius), "wilson-score"


def _coverage(success_mask: Tensor, confidence: float) -> BinomialCoverage:
    mask = torch.as_tensor(success_mask).detach().cpu().to(torch.bool).reshape(-1)
    successes = int(torch.count_nonzero(mask))
    sample_count = int(mask.numel())
    lower, upper, method = binomial_confidence_interval(
        successes, sample_count, confidence
    )
    return BinomialCoverage(
        successes=successes,
        sample_count=sample_count,
        coverage=successes / sample_count,
        confidence_interval=(lower, upper),
        interval_method=method,
    )


def audit_candidate_coverage(
    *,
    audit_payload: SessionAuditPayload,
    selector_payload: SelectorPayload,
    confidence: float = 0.95,
) -> CandidateAuditCoverage:
    """Evaluate candidate bands on the independent audit split only."""

    _verify_flowoff_payload(audit_payload, AUDIT_SCHEMA)
    _verify_selector(selector_payload)
    if (
        audit_payload.manifest_digest != selector_payload.manifest_digest
        or audit_payload.session_id != selector_payload.session_id
        or audit_payload.geometry_digest != selector_payload.geometry_digest
        or not torch.equal(audit_payload.camera_index, selector_payload.camera_index)
    ):
        raise ValueError("audit and selector must share one immutable session manifest")
    global_scores, camera_scores = _score_samples(
        audit_payload.flow_off_samples_uv,
        selector_payload.mean_uv,
        selector_payload.selector_covariance,
        selector_payload.camera_index,
    )
    global_mask = (global_scores >= selector_payload.global_bands[0]) & (
        global_scores <= selector_payload.global_bands[1]
    )
    camera_mask = (camera_scores >= selector_payload.per_camera_bands[:, 0]) & (
        camera_scores <= selector_payload.per_camera_bands[:, 1]
    )
    combined_scores = torch.cat((global_scores[:, None], camera_scores), dim=1)
    upper_nonconformity = torch.max(
        combined_scores / selector_payload.score_anchors[None, :], dim=1
    ).values
    lower_nonconformity = torch.max(
        selector_payload.score_anchors[None, :]
        / combined_scores.clamp_min(torch.finfo(torch.float64).tiny),
        dim=1,
    ).values
    joint_upper_mask = upper_nonconformity <= selector_payload.joint_upper_limit
    joint_lower_mask = lower_nonconformity <= selector_payload.joint_lower_limit
    joint_mask = joint_upper_mask & joint_lower_mask
    if not torch.equal(joint_mask, global_mask & torch.all(camera_mask, dim=1)):
        raise RuntimeError("derived score bands disagree with joint nonconformity")
    global_coverage = _coverage(global_mask, confidence)
    global_upper = _coverage(
        global_scores <= selector_payload.global_bands[1], confidence
    )
    global_lower = _coverage(
        global_scores >= selector_payload.global_bands[0], confidence
    )
    per_camera = tuple(
        _coverage(camera_mask[:, camera], confidence)
        for camera in range(camera_mask.shape[1])
    )
    joint_upper = _coverage(joint_upper_mask, confidence)
    joint_lower = _coverage(joint_lower_mask, confidence)
    joint = _coverage(joint_mask, confidence)
    finite_tail_coverage = selector_payload.joint_tail_order / (
        selector_payload.calibration_sample_count + 1
    )
    two_sided_lower_bound = max(0.0, 2.0 * finite_tail_coverage - 1.0)
    digest = stable_digest(
        global_scores,
        camera_scores,
        metadata={
            "schema": AUDIT_SCHEMA,
            "audit_digest": audit_payload.digest,
            "selector_digest": selector_payload.digest,
            "confidence": float(confidence),
            "global_successes": global_coverage.successes,
            "global_upper_successes": global_upper.successes,
            "global_lower_successes": global_lower.successes,
            "per_camera_successes": [item.successes for item in per_camera],
            "joint_upper_successes": joint_upper.successes,
            "joint_lower_successes": joint_lower.successes,
            "joint_successes": joint.successes,
            "finite_sample_two_sided_lower_bound": two_sided_lower_bound,
        },
    )
    return CandidateAuditCoverage(
        global_band=global_coverage,
        global_upper=global_upper,
        global_lower=global_lower,
        per_camera_bands=per_camera,
        joint_upper=joint_upper,
        joint_lower=joint_lower,
        joint_band=joint,
        nominal_two_sided_coverage_lower_bound=two_sided_lower_bound,
        digest=digest,
    )


def score_selector_residual(
    residual_uv: Tensor,
    selector_payload: SelectorPayload,
) -> SelectorResidualScores:
    """Score one reconstruction residual without exposing calibration samples."""

    _verify_selector(selector_payload)
    residual = _owned_tensor(residual_uv, dtype=torch.float64)
    if residual.shape != selector_payload.mean_uv.shape:
        raise ValueError("residual_uv must match selector detector shape")
    if not bool(torch.all(torch.isfinite(residual))):
        raise ValueError("residual_uv must contain finite values")
    global_scores, camera_scores = _score_samples(
        residual[None, :, :] + selector_payload.mean_uv[None, :, :],
        selector_payload.mean_uv,
        selector_payload.selector_covariance,
        selector_payload.camera_index,
    )
    return SelectorResidualScores(
        global_score=float(global_scores[0]),
        per_camera_scores=camera_scores[0],
    )


def selector_gate_checks(
    scores: SelectorResidualScores,
    selector_payload: SelectorPayload,
) -> dict[str, bool]:
    """Return observable upper/lower checks for global and camera scores."""

    _verify_selector(selector_payload)
    camera_scores = _owned_tensor(scores.per_camera_scores, dtype=torch.float64)
    if camera_scores.shape != (selector_payload.per_camera_bands.shape[0],):
        raise ValueError("per-camera score count must match selector bands")
    global_score = _finite_float(scores.global_score, "global_score")
    return {
        "global_upper": global_score <= float(selector_payload.global_bands[1]),
        "global_lower": global_score >= float(selector_payload.global_bands[0]),
        "camera_upper": bool(
            torch.all(camera_scores <= selector_payload.per_camera_bands[:, 1])
        ),
        "camera_lower": bool(
            torch.all(camera_scores >= selector_payload.per_camera_bands[:, 0])
        ),
    }


__all__ = [
    "CandidateAuditCoverage",
    "SelectorPayload",
    "SelectorResidualScores",
    "SessionAuditPayload",
    "SessionFlowOffFitPayload",
    "SessionPacket",
    "SessionThresholdCalibrationPayload",
    "audit_candidate_coverage",
    "binomial_confidence_interval",
    "build_session_packet",
    "calibrate_candidate_selector",
    "finite_sample_conformal_order",
    "finite_sample_conformal_threshold",
    "score_selector_residual",
    "selector_gate_checks",
    "stable_digest",
    "stable_seed",
    "verify_session_packet",
]
