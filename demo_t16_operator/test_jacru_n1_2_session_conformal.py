from dataclasses import FrozenInstanceError, fields

import pytest
import torch

from demo_t16_operator.jacru_n1_2_session_conformal import (
    CandidateAuditCoverage,
    SelectorPayload,
    SessionAuditPayload,
    SessionFlowOffFitPayload,
    SessionThresholdCalibrationPayload,
    audit_candidate_coverage,
    binomial_confidence_interval,
    build_session_packet,
    calibrate_candidate_selector,
    finite_sample_conformal_order,
    finite_sample_conformal_threshold,
    score_selector_residual,
    selector_gate_checks,
    stable_digest,
    stable_seed,
    verify_session_packet,
)


def _camera_index() -> torch.Tensor:
    return torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64)


def _scale() -> torch.Tensor:
    return torch.tensor(
        [
            [0.010, 0.018],
            [0.013, 0.021],
            [0.017, 0.026],
            [0.022, 0.031],
            [0.028, 0.039],
            [0.035, 0.048],
        ],
        dtype=torch.float64,
    )


def _fields(count: int = 3, multiplier: float = 1.0) -> torch.Tensor:
    base = torch.linspace(-0.4, 0.6, count * 12, dtype=torch.float64)
    return multiplier * base.reshape(count, 6, 2)


def _packet(**overrides):
    values = {
        "manifest_id": "manifest-opened-synthetic-v1",
        "session_id": "session-a",
        "geometry_digest": "geometry-a",
        "field_ids": ("smooth", "single", "double"),
        "camera_index": _camera_index(),
        "flow_on_fields_uv": _fields(),
        "session_scale_uv": _scale(),
        "camera_bias_relative_std": 1.4,
        "fit_repeats": 64,
        "calibration_repeats": 64,
        "audit_repeats": 128,
        "seed": 7103,
    }
    values.update(overrides)
    return build_session_packet(**values)


def test_stable_helpers_and_packet_are_deterministic() -> None:
    assert stable_seed("a", {"z": 2, "b": 1}) == stable_seed(
        "a", {"b": 1, "z": 2}
    )
    first = _packet()
    second = _packet()
    assert first.digest == second.digest
    assert first.fit.digest == second.fit.digest
    assert torch.equal(first.flow_on_observations_uv, second.flow_on_observations_uv)
    assert torch.equal(first.calibration.flow_off_samples_uv, second.calibration.flow_off_samples_uv)


def test_split_streams_are_independent_and_count_stable() -> None:
    packet = _packet()
    changed_fit_count = _packet(fit_repeats=96)
    assert not torch.equal(
        packet.fit.flow_off_samples_uv,
        packet.calibration.flow_off_samples_uv,
    )
    assert not torch.equal(
        packet.calibration.flow_off_samples_uv[:64],
        packet.audit.flow_off_samples_uv[:64],
    )
    assert torch.equal(
        packet.calibration.flow_off_samples_uv,
        changed_fit_count.calibration.flow_off_samples_uv,
    )
    assert torch.equal(
        packet.audit.flow_off_samples_uv,
        changed_fit_count.audit.flow_off_samples_uv,
    )


def test_session_scale_and_nuisance_are_target_scale_independent() -> None:
    first = _packet(flow_on_fields_uv=_fields(multiplier=1.0))
    rescaled = _packet(flow_on_fields_uv=_fields(multiplier=50.0))
    assert torch.equal(first.fit.flow_off_samples_uv, rescaled.fit.flow_off_samples_uv)
    assert torch.equal(
        first.calibration.flow_off_samples_uv,
        rescaled.calibration.flow_off_samples_uv,
    )
    first_noise = first.flow_on_observations_uv - _fields(multiplier=1.0)
    rescaled_noise = rescaled.flow_on_observations_uv - _fields(multiplier=50.0)
    assert torch.allclose(first_noise, rescaled_noise, atol=5e-14, rtol=0.0)
    low_variance = torch.var(first.fit.flow_off_samples_uv[:, 0, 0])
    high_variance = torch.var(first.fit.flow_off_samples_uv[:, -1, -1])
    assert high_variance > 8.0 * low_variance


def test_persistent_camera_component_bias_is_shared_with_flow_on_fields() -> None:
    packet = _packet(
        flow_on_fields_uv=_fields(count=3),
        fit_repeats=2048,
        camera_bias_relative_std=10.0,
        frame_camera_jitter_relative_std=0.0,
    )
    flow_on_residual = packet.flow_on_observations_uv - _fields(count=3)
    for camera in range(2):
        rays = _camera_index() == camera
        flow_on_mean = torch.mean(flow_on_residual[:, rays, :], dim=(0, 1))
        flow_off_mean = torch.mean(
            packet.fit.flow_off_samples_uv[:, rays, :], dim=(0, 1)
        )
        assert torch.allclose(flow_on_mean, flow_off_mean, atol=0.04, rtol=0.0)


def test_finite_sample_uses_exact_62nd_order_statistic() -> None:
    scores = torch.arange(64, 0, -1, dtype=torch.float64)
    assert finite_sample_conformal_order(64, 0.95) == 62
    assert finite_sample_conformal_threshold(scores, 0.95) == 62.0
    assert finite_sample_conformal_order(64, 0.05) == 4
    assert finite_sample_conformal_threshold(scores, 0.05) == 4.0


def test_selector_has_global_and_per_camera_band_shapes() -> None:
    packet = _packet()
    selector = calibrate_candidate_selector(
        fit_payload=packet.fit,
        calibration_payload=packet.calibration,
    )
    assert selector.global_bands.shape == (2,)
    assert selector.per_camera_bands.shape == (2, 2)
    assert selector.score_anchors.shape == (3,)
    assert selector.calibration_sample_count == 64
    assert selector.joint_tail_order == 64
    assert selector.joint_tail_quantile == pytest.approx(0.975)
    assert selector.proximal_covariance.shape == (12, 12)
    assert selector.selector_covariance.dtype == torch.float64
    assert torch.all(selector.per_camera_bands[:, 0] < selector.per_camera_bands[:, 1])


def test_payloads_are_frozen_and_capability_limited() -> None:
    packet = _packet()
    selector = calibrate_candidate_selector(
        fit_payload=packet.fit,
        calibration_payload=packet.calibration,
    )
    forbidden = {"truth", "clean", "target", "audit_samples", "nuisance"}
    for payload_type in (
        SessionFlowOffFitPayload,
        SessionThresholdCalibrationPayload,
        SessionAuditPayload,
    ):
        names = {item.name.lower() for item in fields(payload_type)}
        assert not any(token in name for name in names for token in forbidden)
    assert {item.name for item in fields(SelectorPayload)} == {
        "candidate_id",
        "manifest_digest",
        "session_id",
        "geometry_digest",
        "source_fit_digest",
        "source_calibration_digest",
        "mean_uv",
        "proximal_covariance",
        "selector_covariance",
        "global_bands",
        "per_camera_bands",
        "score_anchors",
        "calibration_sample_count",
        "joint_tail_order",
        "joint_tail_quantile",
        "joint_upper_limit",
        "joint_lower_limit",
        "camera_index",
        "digest",
    }
    assert not hasattr(selector, "audit")
    with pytest.raises(FrozenInstanceError):
        packet.fit.digest = "changed"


def test_thresholds_are_candidate_specific() -> None:
    packet = _packet()
    structured = calibrate_candidate_selector(
        fit_payload=packet.fit,
        calibration_payload=packet.calibration,
        selector_covariance_policy="structured",
        mean_policy="estimated",
    )
    isotropic_zero = calibrate_candidate_selector(
        fit_payload=packet.fit,
        calibration_payload=packet.calibration,
        selector_covariance_policy="isotropic",
        mean_policy="zero",
    )
    assert structured.digest != isotropic_zero.digest
    assert not torch.allclose(structured.global_bands, isotropic_zero.global_bands)
    assert not torch.allclose(
        structured.per_camera_bands, isotropic_zero.per_camera_bands
    )


def test_independent_audit_coverage_and_interval_bounds() -> None:
    packet = _packet(audit_repeats=512)
    selector = calibrate_candidate_selector(
        fit_payload=packet.fit,
        calibration_payload=packet.calibration,
    )
    result = audit_candidate_coverage(
        audit_payload=packet.audit,
        selector_payload=selector,
    )
    assert isinstance(result, CandidateAuditCoverage)
    for coverage in (
        result.global_band,
        result.global_upper,
        result.global_lower,
        result.joint_upper,
        result.joint_lower,
        result.joint_band,
        *result.per_camera_bands,
    ):
        lower, upper = coverage.confidence_interval
        assert 0.0 <= lower <= coverage.coverage <= upper <= 1.0
        assert coverage.interval_method in {
            "clopper-pearson-exact",
            "wilson-score",
        }
    assert result.global_band.sample_count == 512
    assert result.nominal_two_sided_coverage_lower_bound == pytest.approx(63.0 / 65.0)


def test_exact_binomial_interval_handles_boundary_counts() -> None:
    for successes in (0, 64):
        lower, upper, method = binomial_confidence_interval(successes, 64)
        assert 0.0 <= lower <= upper <= 1.0
        assert method == "clopper-pearson-exact"


def test_source_mutation_isolated_and_payload_mutation_detected() -> None:
    fields_input = _fields()
    scale_input = _scale()
    packet = _packet(flow_on_fields_uv=fields_input, session_scale_uv=scale_input)
    original = packet.fit.flow_off_samples_uv.clone()
    fields_input.zero_()
    scale_input.fill_(99.0)
    assert torch.equal(packet.fit.flow_off_samples_uv, original)

    packet.fit.flow_off_samples_uv[0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="digest mismatch"):
        calibrate_candidate_selector(
            fit_payload=packet.fit,
            calibration_payload=packet.calibration,
        )


def test_top_level_session_packet_mutation_is_detected() -> None:
    packet = _packet()
    verify_session_packet(packet)
    packet.flow_on_observations_uv[0, 0, 0] += 0.5
    with pytest.raises(ValueError, match="packet digest mismatch"):
        verify_session_packet(packet)


def test_digest_is_value_and_mutation_sensitive() -> None:
    value = torch.arange(8, dtype=torch.float64)
    before = stable_digest(value, metadata={"kind": "test"})
    value[3] += 0.25
    after = stable_digest(value, metadata={"kind": "test"})
    assert before != after
    assert before == stable_digest(torch.arange(8, dtype=torch.float64), metadata={"kind": "test"})


@pytest.mark.parametrize(
    "overrides, error, message",
    [
        ({"session_id": ""}, ValueError, "nonempty"),
        ({"seed": True}, TypeError, "integer"),
        ({"fit_repeats": 1}, ValueError, "at least two"),
        ({"field_ids": ("only-one",)}, ValueError, "two or three"),
        ({"field_ids": ("same", "same", "third")}, ValueError, "unique"),
        ({"camera_bias_relative_std": -0.1}, ValueError, "nonnegative"),
        ({"frame_camera_jitter_relative_std": -0.1}, ValueError, "nonnegative"),
        ({"session_scale_uv": torch.zeros((6, 2))}, ValueError, "positive finite"),
        (
            {"camera_index": torch.tensor([0, 0, 2, 2, 2, 2])},
            ValueError,
            "contiguous",
        ),
    ],
)
def test_session_builder_rejects_invalid_inputs(overrides, error, message) -> None:
    with pytest.raises(error, match=message):
        _packet(**overrides)


@pytest.mark.parametrize(
    "call, error, message",
    [
        (lambda: finite_sample_conformal_order(0, 0.95), ValueError, "positive"),
        (lambda: finite_sample_conformal_order(2, 0.95), ValueError, "exceeds"),
        (lambda: finite_sample_conformal_order(64, 1.0), ValueError, "\\(0, 1\\)"),
        (
            lambda: finite_sample_conformal_threshold(
                torch.tensor([1.0, float("nan")]), 0.5
            ),
            ValueError,
            "finite",
        ),
        (
            lambda: binomial_confidence_interval(65, 64),
            ValueError,
            "lie in",
        ),
    ],
)
def test_conformal_and_interval_helpers_reject_invalid_inputs(call, error, message) -> None:
    with pytest.raises(error, match=message):
        call()


def test_selector_rejects_cross_session_and_invalid_policy() -> None:
    first = _packet()
    second = _packet(session_id="session-b")
    with pytest.raises(ValueError, match="one session"):
        calibrate_candidate_selector(
            fit_payload=first.fit,
            calibration_payload=second.calibration,
        )
    with pytest.raises(ValueError, match="unsupported selector"):
        calibrate_candidate_selector(
            fit_payload=first.fit,
            calibration_payload=first.calibration,
            selector_covariance_policy="oracle",
        )


def test_joint_alpha_spending_uses_one_max_score_per_frame() -> None:
    packet = _packet()
    selector = calibrate_candidate_selector(
        fit_payload=packet.fit,
        calibration_payload=packet.calibration,
        target_two_sided_coverage=0.95,
    )
    assert selector.joint_tail_quantile == pytest.approx(0.975)
    assert selector.joint_tail_order == finite_sample_conformal_order(64, 0.975)
    assert selector.joint_tail_order == 64
    assert selector.global_bands[0] < selector.global_bands[1]
    assert torch.all(selector.per_camera_bands[:, 0] < selector.per_camera_bands[:, 1])


def test_audit_rejects_packet_substitution_even_with_same_camera_layout() -> None:
    first = _packet()
    second = _packet(session_id="session-b")
    selector = calibrate_candidate_selector(
        fit_payload=first.fit,
        calibration_payload=first.calibration,
    )
    with pytest.raises(ValueError, match="immutable session manifest"):
        audit_candidate_coverage(
            audit_payload=second.audit,
            selector_payload=selector,
        )


def test_observable_gate_detects_lower_tail_and_single_camera_upper_tail() -> None:
    packet = _packet()
    selector = calibrate_candidate_selector(
        fit_payload=packet.fit,
        calibration_payload=packet.calibration,
    )
    zero = score_selector_residual(torch.zeros((6, 2)), selector)
    zero_checks = selector_gate_checks(zero, selector)
    assert not zero_checks["global_lower"]
    assert not zero_checks["camera_lower"]

    direction = torch.zeros((6, 2), dtype=torch.float64)
    direction[_camera_index() == 0, 0] = 1.0
    witnessed = False
    for scale in torch.logspace(-4, 1, 300, dtype=torch.float64):
        scores = score_selector_residual(scale * direction, selector)
        checks = selector_gate_checks(scores, selector)
        if checks["global_upper"] and not checks["camera_upper"]:
            witnessed = True
            break
    assert witnessed, "per-camera upper gate must catch a localized residual before global failure"
