from __future__ import annotations

from dataclasses import fields
import math

import pytest
import torch

from demo_t16_operator.certified_grouped_majorizer import (
    PRIMITIVE_COUNT,
    SCHEMA_VERSION,
    STATUS,
    DeploymentGeometry,
    GeometryPartitionStump,
    PartitionSpec,
    audit_partition_safety,
    build_diagonal_metric,
    build_grouped_majorizer,
    deployment_geometry_from_rig,
    generate_tiny_primitive_rigs,
    predefined_partitions,
    run_signed_pdhg_trajectory,
    select_partition,
    stable_rig_seed,
    validate_partition,
)


def _assignments() -> dict[str, str]:
    return {
        "train-a": "train",
        "train-b": "train",
        "cal-a": "safety_calibration",
        "fresh-a": "fresh_geometry_ood",
    }


def _rigs(assignments: dict[str, str] | None = None):
    return generate_tiny_primitive_rigs(
        split_assignments=assignments or _assignments(),
        geometry_seed=101,
        noise_seed=202,
        row_count=10,
        column_count=8,
        dtype=torch.float64,
    )


def test_schema_and_deployment_contract_exclude_sensitive_fields() -> None:
    assert SCHEMA_VERSION == "certified-grouped-majorizer-smoke-1.0"
    assert STATUS == "DEVELOPMENT_ONLY_SYNTHETIC_CERTIFIED_PARTITION_SMOKE"
    assert {field.name for field in fields(DeploymentGeometry)} == {
        "rig_id",
        "geometry_features",
    }
    rig = _rigs()[0]
    deployment = deployment_geometry_from_rig(rig)
    assert deployment.rig_id == rig.rig_id
    assert deployment.geometry_features.data_ptr() != rig.geometry_features.data_ptr()


def test_stable_sha_seed_is_order_and_unrelated_rig_invariant() -> None:
    original = _rigs()
    reordered_assignments = {
        "fresh-a": "fresh_geometry_ood",
        "train-b": "train",
        "inserted": "train",
        "cal-a": "safety_calibration",
        "train-a": "train",
    }
    changed = _rigs(reordered_assignments)
    original_by_id = {rig.rig_id: rig for rig in original}
    changed_by_id = {rig.rig_id: rig for rig in changed}
    for rig_id in original_by_id:
        left, right = original_by_id[rig_id], changed_by_id[rig_id]
        assert left.geometry_seed_sha256 == right.geometry_seed_sha256
        assert left.noise_seed_sha256 == right.noise_seed_sha256
        assert left.geometry_feature_sha256 == right.geometry_feature_sha256
        assert torch.equal(left.geometry_features, right.geometry_features)
        assert torch.equal(left.primitives, right.primitives)
        assert torch.equal(left.target, right.target)
    seed, digest = stable_rig_seed(101, "train-a", "train")
    assert 0 <= seed < 2**63 - 1
    assert len(digest) == 64


def test_partition_theorem_singleton_and_exact_identities() -> None:
    generator = torch.Generator().manual_seed(77)
    primitives = torch.randn((PRIMITIVE_COUNT, 7, 5), generator=generator, dtype=torch.float64)
    specs = predefined_partitions()
    signed = torch.sum(primitives, dim=0)
    singleton = build_grouped_majorizer(primitives, specs["singleton_factor"])
    exact = build_grouped_majorizer(primitives, specs["all_in_one_exact"])
    assert torch.allclose(
        singleton.matrix,
        torch.sum(torch.abs(primitives), dim=0),
        rtol=2e-15,
        atol=2e-15,
    )
    assert torch.equal(exact.matrix, torch.abs(signed))
    for spec in specs.values():
        majorizer = build_grouped_majorizer(primitives, spec)
        assert bool(torch.all(majorizer.matrix >= torch.abs(signed) - 1e-14))


def test_invalid_partitions_fail_before_majorizer_construction() -> None:
    duplicate = PartitionSpec("duplicate", ((0, 1), (1, 2), (3, 4, 5)))
    missing = PartitionSpec("missing", ((0, 1), (2, 3), (4,)))
    with pytest.raises(ValueError, match="exactly once"):
        validate_partition(duplicate, primitive_count=PRIMITIVE_COUNT)
    with pytest.raises(ValueError, match="exactly once"):
        validate_partition(missing, primitive_count=PRIMITIVE_COUNT)


def test_every_predefined_partition_has_zero_schur_violation_including_zero_support() -> None:
    specs = predefined_partitions()
    for rig in _rigs():
        for spec in specs.values():
            majorizer = build_grouped_majorizer(rig.primitives, spec)
            metric = build_diagonal_metric(majorizer, eta=0.7)
            audit = audit_partition_safety(rig.primitives, spec, metric, eta=0.7)
            assert audit["total_violation_count"] == 0
            assert audit["exact_masses_recomputed_from_signed_primitives"] is True

    primitives = torch.zeros((PRIMITIVE_COUNT, 5, 4), dtype=torch.float64)
    primitives[0, 1:4, 1:3] = 1.0
    primitives[1, 1:4, 1:3] = -0.4
    majorizer = build_grouped_majorizer(primitives, specs["paired_local"])
    metric = build_diagonal_metric(majorizer, eta=0.7)
    audit = audit_partition_safety(primitives, specs["paired_local"], metric, eta=0.7)
    assert audit["total_violation_count"] == 0
    assert torch.count_nonzero(metric.sigma == 0.0) == 2
    assert torch.count_nonzero(metric.tau == 0.0) == 2


def test_selector_runtime_type_gate_and_exact_oracle_exclusion() -> None:
    rig = _rigs()[0]
    selector = GeometryPartitionStump(
        feature_index=0,
        threshold=0.0,
        left_partition="paired_local",
        right_partition="paired_cross",
        allowed_partitions=("singleton_factor", "paired_local", "paired_cross"),
        exact_oracle_partition="all_in_one_exact",
    )
    selected = select_partition(selector, deployment_geometry_from_rig(rig))
    assert selected in selector.allowed_partitions
    with pytest.raises(TypeError, match="DeploymentGeometry"):
        select_partition(selector, rig)  # type: ignore[arg-type]
    unsafe_selector = GeometryPartitionStump(
        feature_index=0,
        threshold=0.0,
        left_partition="all_in_one_exact",
        right_partition="paired_cross",
        allowed_partitions=("all_in_one_exact", "paired_cross"),
        exact_oracle_partition="all_in_one_exact",
    )
    with pytest.raises(ValueError, match="exact all-in-one"):
        select_partition(unsafe_selector, deployment_geometry_from_rig(rig))


def test_cost_proxy_counts_each_primitive_once_and_solver_calls_are_equal() -> None:
    rig = _rigs()[0]
    specs = predefined_partitions()
    costs = {
        name: build_grouped_majorizer(rig.primitives, spec).construction_cost
        for name, spec in specs.items()
    }
    entries = rig.signed_matrix.numel()
    assert all(
        cost["primitive_component_accumulation_entries"] == PRIMITIVE_COUNT * entries
        for cost in costs.values()
    )
    assert all(cost["definition"] == "ANALYTIC_PROXY_NOT_WALL_TIME" for cost in costs.values())
    assert costs["singleton_factor"]["cost_proxy_units"] < costs["paired_local"]["cost_proxy_units"]
    assert costs["paired_local"]["cost_proxy_units"] < costs["all_in_one_exact"]["cost_proxy_units"]
    assert costs["all_in_one_exact"]["cost_proxy_units"] == max(
        cost["cost_proxy_units"] for cost in costs.values()
    )

    ledgers = []
    for name in ("singleton_factor", "paired_local", "all_in_one_exact"):
        majorizer = build_grouped_majorizer(rig.primitives, specs[name])
        metric = build_diagonal_metric(majorizer, eta=0.7)
        trajectory = run_signed_pdhg_trajectory(
            rig.signed_matrix,
            rig.target,
            rig.truth,
            metric,
            checkpoints=[0, 1, 2, 4, 8],
            theta=1.0,
        )
        assert all(math.isfinite(float(row["field_relative_l2"])) for row in trajectory.rows)
        ledgers.append(trajectory.ledger)
    assert ledgers[0] == ledgers[1] == ledgers[2]
