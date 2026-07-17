from __future__ import annotations

from pathlib import Path

import pytest

from site_tools.validate_psu_b0_factor_gate_b import (
    CHECKPOINTS,
    FAMILIES,
    METHODS,
    THRESHOLDS,
    ValidationError,
    Validator,
    _compare_nested,
    load_strict_json,
    parse_checksum_manifest,
    recompute_decision,
)


def _rows(*, factor_k32: float = 0.60) -> list[dict[str, object]]:
    profiles = {
        "scalar_a_only_pdhg": {4: 1.30, 8: 1.20, 16: 1.10, 32: 1.00},
        "view_block_a_only_pdhg": {4: 1.10, 8: 0.95, 16: 0.85, 32: 0.80},
        "voxel_factor_a_only_pdhg": {4: 0.90, 8: 0.80, 16: 0.70, 32: factor_k32},
        "graph_pcgls": {4: 0.70, 8: 0.65, 16: 0.60, 32: 0.55},
    }
    return [
        {
            "replicate": replicate,
            "sample_index": sample,
            "method": method,
            "iterations": iteration,
            "field_relative_l2": profiles[method][iteration],
        }
        for replicate in (0, 8)
        for sample in range(len(FAMILIES))
        for method in METHODS
        for iteration in CHECKPOINTS
    ]


def _timings(*, factor_seconds: float = 2.0) -> list[dict[str, object]]:
    return [
        {
            "replicate": replicate,
            "sample_index": sample,
            "voxel_factor_seconds": factor_seconds,
            "graph_seconds": 1.0,
        }
        for replicate in (0, 8)
        for sample in range(len(FAMILIES))
    ]


@pytest.mark.parametrize(
    "payload",
    [
        '{"a":1,"a":2}',
        '{"a":NaN}',
        '[1,2,3]',
    ],
)
def test_strict_json_rejects_ambiguous_or_nonobject_payloads(
    tmp_path: Path,
    payload: str,
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValidationError):
        load_strict_json(path)


def test_checksum_manifest_requires_exact_runner_file_set(tmp_path: Path) -> None:
    digest = "a" * 64
    path = tmp_path / "checksums.sha256"
    path.write_text(
        "".join(
            f"{digest}  {name}\n"
            for name in ("report.json", "metric_rows.csv", "audit.json")
        ),
        encoding="ascii",
    )
    assert set(parse_checksum_manifest(path)) == {
        "report.json",
        "metric_rows.csv",
        "audit.json",
    }
    path.write_text(f"{digest}  report.json\n", encoding="ascii")
    with pytest.raises(ValidationError, match="file set"):
        parse_checksum_manifest(path)


def test_independent_decision_recomputation_matches_frozen_pass_semantics() -> None:
    decision = recompute_decision(_rows(), _timings(), THRESHOLDS)
    assert decision["status"] == "GATE_B_E2_ORACLE_SCALE_CONDITIONING_SIGNAL_ONLY"
    assert decision["all_gates_passed"] is True
    assert decision["algorithm_superiority_claim_authorized"] is False


def test_independent_decision_recomputation_exposes_accuracy_and_cost_failures() -> None:
    decision = recompute_decision(
        _rows(factor_k32=1.05),
        _timings(factor_seconds=4.0),
        THRESHOLDS,
    )
    assert decision["status"] == "GATE_B_E2_MECHANISM_NO_GO"
    assert decision["gates"]["mean_reduction_vs_scalar"] is False
    assert decision["gates"]["single_sample_wall_time"] is False


def test_independent_decision_rejects_duplicate_timing_pairs() -> None:
    timings = _timings()
    with pytest.raises(ValidationError, match="duplicate"):
        recompute_decision(_rows(), [*timings, timings[0]], THRESHOLDS)


def test_nested_comparison_detects_reported_numeric_mutation() -> None:
    validator = Validator()
    with pytest.raises(ValidationError):
        _compare_nested(
            validator,
            {"gate": {"value": 0.8}},
            {"gate": {"value": 0.7}},
            "decision",
        )
