from __future__ import annotations

from argparse import Namespace
import copy
import json
from pathlib import Path

import pytest
import torch

from demo_t16_operator.jacru_n1_2_session_conformal import (
    build_session_packet,
    calibrate_candidate_selector,
    score_selector_residual,
    selector_gate_checks,
)
from site_tools import run_jacru_n1_2_session_conformal_dual_reference as n12


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "demo_t16_operator/configs/"
    "jacru_n1_2_session_conformal_dual_reference_development_v1.json"
)
SOURCE = ROOT / "demo_t16_operator/configs/jacru_m2_learned_residual_t0_v1.json"


def _config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _source_one_seed():
    value = json.loads(SOURCE.read_text(encoding="utf-8"))
    for split in value["splits"].values():
        split["base_seeds"] = split["base_seeds"][:1]
    value["training"]["model_seeds"] = value["training"]["model_seeds"][:1]
    return value


def test_session_record_builder_reuses_one_packet_for_two_or_three_fields() -> None:
    records, packets, case_to_session, manifest = n12._prepare_session_records(
        _source_one_seed(), _config()
    )
    assert len(records) == 7
    assert len(packets) == 3
    assert len(case_to_session) == 7
    assert sorted(row["field_count"] for row in manifest) == [2, 2, 3]
    for row in manifest:
        session_records = [
            record
            for record in records
            if case_to_session[record.case.inference.case_id] == row["session_id"]
        ]
        assert len(session_records) == row["field_count"]
        assert packets[row["session_id"]].manifest_digest == row["manifest_digest"]
        assert not row["target_amplitude_used_for_scale"]


def test_registered_huber_reference_is_exactly_24_iterations() -> None:
    config = _config()
    source = _source_one_seed()
    step = int(config["registered_classical_reference"]["projection_step"])
    total = int(source["physical_budget"]["cgls_base_iterations"]) + 1 + step
    assert step == 11
    assert total == int(
        config["registered_classical_reference"]["total_forward_calls"]
    )
    assert total == 24


def _toy_selector():
    camera_index = torch.tensor([0, 0], dtype=torch.int64)
    fields = torch.zeros((2, 2, 2), dtype=torch.float64)
    scale = torch.full((2, 2), 0.04, dtype=torch.float64)
    packet = build_session_packet(
        manifest_id="toy-manifest",
        session_id="toy-session",
        geometry_digest="toy-geometry",
        field_ids=("field-a", "field-b"),
        camera_index=camera_index,
        flow_on_fields_uv=fields,
        session_scale_uv=scale,
        fit_repeats=64,
        calibration_repeats=64,
        audit_repeats=64,
        seed=31,
    )
    selector = calibrate_candidate_selector(
        fit_payload=packet.fit,
        calibration_payload=packet.calibration,
    )
    return packet, selector


def _toy_field_matrix():
    support = torch.tensor(
        [
            [[1, 1], [0, 0]],
            [[1, 0], [0, 0]],
        ],
        dtype=torch.bool,
    )
    matrix = torch.tensor(
        [
            [1.0, 0.2, 0.0],
            [0.0, 0.4, 1.0],
            [0.3, 0.0, 0.8],
            [0.5, 0.1, 0.2],
        ],
        dtype=torch.float64,
    )
    field = torch.zeros((2, 2, 2), dtype=torch.float64)
    field.masked_scatter_(support, torch.tensor([0.8, -0.2, 0.1], dtype=torch.float64))
    return support, matrix, field


def test_dense_ceiling_keeps_raw_when_raw_is_inside_frozen_joint_band() -> None:
    packet, selector = _toy_selector()
    support, matrix, field = _toy_field_matrix()
    residual = None
    for sample in packet.calibration.flow_off_samples_uv:
        candidate = sample - selector.mean_uv
        checks = selector_gate_checks(
            score_selector_residual(candidate, selector), selector
        )
        if n12._gate_policy_pass(checks, "joint_two_sided"):
            residual = candidate
            break
    assert residual is not None
    target = (matrix @ field.masked_select(support)).reshape(2, 2) - residual
    result = n12._dense_multigate_ceiling(
        initial_field=field,
        target_observation_uv=target,
        dense_active_matrix=matrix,
        support_mask=support,
        selector=selector,
        gate_policy="joint_two_sided",
        log10_alpha_bounds=(-8.0, 8.0),
        alpha_grid_count=33,
    )
    assert result.selector_valid
    assert result.fallback_reason == "RAW_ALREADY_INSIDE_FROZEN_GATE"
    assert torch.equal(result.field, field)


def test_dense_ceiling_fails_closed_when_lower_gate_has_no_feasible_alpha() -> None:
    _, selector = _toy_selector()
    support, matrix, field = _toy_field_matrix()
    target = (matrix @ field.masked_select(support)).reshape(2, 2)
    result = n12._dense_multigate_ceiling(
        initial_field=field,
        target_observation_uv=target,
        dense_active_matrix=matrix,
        support_mask=support,
        selector=selector,
        gate_policy="joint_two_sided",
        log10_alpha_bounds=(-8.0, 8.0),
        alpha_grid_count=33,
    )
    assert not result.selector_valid
    assert result.fallback_reason == "NO_ALPHA_SATISFIED_FROZEN_GATE_RETURN_RAW"
    assert torch.equal(result.field, field)


def test_mismatch_triangle_lower_gate_fails_closed_when_uninformative() -> None:
    _, selector = _toy_selector()
    scores = score_selector_residual(selector.mean_uv, selector)
    mismatch_scores = score_selector_residual(
        100.0 * torch.ones_like(selector.mean_uv), selector
    )
    checks = n12._gate_checks_with_optional_mismatch(
        scores=scores,
        selector=selector,
        evaluator_mismatch_scores=mismatch_scores,
    )
    assert not checks["global_lower_informative"]
    assert not checks["camera_lower_informative"]
    assert not n12._gate_policy_pass(checks, "joint_two_sided")


@pytest.mark.parametrize(
    "policy, checks, expected",
    [
        ("global_upper_only", {"global_upper": True}, True),
        (
            "joint_upper_only",
            {"global_upper": True, "camera_upper": False},
            False,
        ),
        (
            "joint_two_sided",
            {
                "global_upper": True,
                "camera_upper": True,
                "global_lower": True,
                "camera_lower": False,
            },
            False,
        ),
    ],
)
def test_gate_policy_contract(policy, checks, expected) -> None:
    assert n12._gate_policy_pass(checks, policy) is expected


def test_formal_mode_rejects_development_config_and_overrides() -> None:
    args = Namespace(
        run_mode="formal",
        epochs=None,
        seed_limit=None,
        replace_output=False,
    )
    with pytest.raises(RuntimeError, match="frozen formal config"):
        n12._validate_run_contract(args, _config())

    frozen = copy.deepcopy(_config())
    frozen["status"] = "FROZEN_BEFORE_FIRST_FORMAL_N1_2_EXECUTION"
    args.epochs = 1
    with pytest.raises(RuntimeError, match="forbids"):
        n12._validate_run_contract(args, frozen)
