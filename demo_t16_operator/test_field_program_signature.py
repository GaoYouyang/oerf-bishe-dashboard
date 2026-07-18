from __future__ import annotations

import torch

from demo_t16_operator.automatic_discrete_multifidelity import SyntheticRayRig
from demo_t16_operator.field_dependent_ray import (
    sample_pupil_sobol,
    trace_field_dependent_rays,
)
from demo_t16_operator.field_program_signature import (
    _diagnostic_trace_with_stages,
    build_field_program_signature,
)


def _grid(size: int = 9) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, size, dtype=torch.float64)
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    return -torch.exp(-((xx / 0.42) ** 2 + (yy / 0.34) ** 2 + (zz / 0.38) ** 2))


def _rig() -> SyntheticRayRig:
    return SyntheticRayRig(
        rig_id="signature-test",
        view_angle_degrees=27.0,
        detector_u=0.04,
        detector_z=-0.03,
        aperture_radius=0.025,
        path_half_length=0.62,
        cone_u=0.02,
        cone_z=0.015,
        bend=0.0,
    )


def _signature(values: torch.Tensor, *, threshold: float):
    return build_field_program_signature(
        values,
        sample_pupil_sobol(3, seed=71),
        _rig(),
        difference_step=0.002,
        refractivity_scale=0.003,
        step_count=6,
        support_threshold=threshold,
        frustum_half_width_u=0.005,
        frustum_half_width_v=0.005,
    )


def test_signature_is_deterministic_and_covers_all_program_groups() -> None:
    values = _grid()
    first = _signature(values, threshold=0.1)
    second = _signature(values.clone(), threshold=0.1)
    assert first == second
    assert first.group_count == 4 * 6 + 2
    assert first.query_record_count == (4 * 6 * 3 + 2 * 6 * 3) * 7
    assert first.minimum_domain_margin >= 0.0
    assert first.minimum_stencil_margin >= 0.0
    assert first.maximum_direction_norm_error <= 1e-12


def test_support_branch_change_changes_ordered_digest() -> None:
    values = _grid()
    base = _signature(values, threshold=0.1)
    changed = _signature(0.25 * values, threshold=0.1)
    assert base.labels_sha256 == changed.labels_sha256
    assert base.support_bits_sha256 != changed.support_bits_sha256
    assert base.digest_sha256 != changed.digest_sha256


def test_ray_row_permutation_changes_ordered_signature_not_counts() -> None:
    values = _grid()
    rays = sample_pupil_sobol(3, seed=79)
    kwargs = {
        "difference_step": 0.002,
        "refractivity_scale": 0.003,
        "step_count": 6,
        "support_threshold": 0.1,
        "frustum_half_width_u": 0.005,
        "frustum_half_width_v": 0.005,
    }
    original = build_field_program_signature(values, rays, _rig(), **kwargs)
    permuted = build_field_program_signature(
        values, rays[torch.tensor([2, 1, 0])], _rig(), **kwargs
    )
    assert original.query_record_count == permuted.query_record_count
    assert original.support_true_count == permuted.support_true_count
    assert original.digest_sha256 != permuted.digest_sha256


def test_diagnostic_replay_matches_hash_bound_production_trace() -> None:
    values = _grid()
    states = sample_pupil_sobol(3, seed=73)
    plain = trace_field_dependent_rays(
        values,
        states,
        _rig(),
        gradient_mode="central",
        difference_step=0.002,
        refractivity_scale=0.003,
        step_count=6,
        create_graph=False,
    )
    audited, records = _diagnostic_trace_with_stages(
        values,
        states,
        _rig(),
        difference_step=0.002,
        refractivity_scale=0.003,
        step_count=6,
    )
    assert len(records) == 4 * 6
    assert torch.equal(plain.positions, audited.positions)
    assert torch.equal(plain.directions, audited.directions)
