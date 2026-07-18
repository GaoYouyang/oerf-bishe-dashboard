from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import demo_t16_operator.run_n2_pvgr_n3_blind_analysis_recovery as recovery


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "demo_t16_operator/configs/n2_pvgr_n3_blind_analysis_recovery_v1.json"


def _config() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_recovery_contract_closes_every_result_dependent_change() -> None:
    config = _config()
    recovery._validate_recovery_contract(config)
    for forbidden in (
        "rerun_physical_cells",
        "change_thresholds",
        "change_seeds_or_factorial",
        "exclude_cells",
        "change_bootstrap_or_machine_decisions",
    ):
        drifted = copy.deepcopy(config)
        drifted["frozen_recovery_actions"][forbidden] = True
        with pytest.raises(ValueError, match="forbidden recovery action"):
            recovery._validate_recovery_contract(drifted)


def test_logical_query_count_accepts_only_frozen_equal_semantics() -> None:
    config = _config()
    assert (
        recovery.logical_query_count(
            {"logical_scalar_grid_point_queries": 101},
            "operator_consistent_homotopy",
            config,
        )
        == 101
    )
    assert (
        recovery.logical_query_count(
            {"total_field_point_queries": 100}, "picard_1", config
        )
        == 100
    )
    assert (
        recovery.logical_query_count(
            {
                "logical_scalar_grid_point_queries": 100,
                "total_field_point_queries": 100,
            },
            "picard_2",
            config,
        )
        == 100
    )
    with pytest.raises(ValueError, match="conflicting"):
        recovery.logical_query_count(
            {
                "logical_scalar_grid_point_queries": 100,
                "total_field_point_queries": 101,
            },
            "picard_2",
            config,
        )
    with pytest.raises(KeyError, match="no frozen"):
        recovery.logical_query_count({}, "picard_1", config)
    with pytest.raises(KeyError, match="no frozen"):
        recovery.logical_query_count(
            {"total_field_point_queries": 100}, "unregistered_method", config
        )


def test_recovered_query_ratios_match_original_predeclared_units() -> None:
    config = _config()
    rows = [
        {
            "query_accounting": {
                "operator_consistent_homotopy": {
                    "logical_scalar_grid_point_queries": 101
                },
                "picard_1": {"total_field_point_queries": 100},
                "picard_2": {"total_field_point_queries": 150},
                "high128": {"logical_scalar_grid_point_queries": 250},
            }
        }
    ]
    assert recovery._query_ratio(rows, "picard_1", config) == pytest.approx(0.4)
    assert recovery._query_ratio_vs_ocbh(rows, "picard_1", config) == pytest.approx(
        100 / 101
    )
    assert recovery._query_ratio_vs_ocbh(rows, "picard_2", config) == pytest.approx(
        150 / 101
    )


def test_flattened_query_ledger_preserves_original_and_canonical_fields() -> None:
    config = _config()
    rows = [
        {
            "cell_id": "cell-1",
            "query_accounting": {
                "picard_1": {
                    "total_field_point_queries": 100,
                    "query_unit": "scalar_grid_evaluation_at_one_coordinate",
                },
                "high128": {"logical_scalar_grid_point_queries": 250},
            },
        }
    ]
    flat = recovery._flatten_query_rows(rows, config)
    by_method = {row["method_id"]: row for row in flat}
    assert by_method["picard_1"]["total_field_point_queries"] == 100
    assert by_method["picard_1"]["logical_scalar_grid_point_queries"] == 100
    assert (
        by_method["picard_1"]["logical_query_count_source_field"]
        == "total_field_point_queries"
    )
    assert (
        by_method["high128"]["logical_query_count_source_field"]
        == "logical_scalar_grid_point_queries"
    )


def test_opaque_checkpoint_set_is_complete_without_payload_parsing() -> None:
    config = _config()
    work = ROOT / str(config["formal_work_output"])
    checkpoints = list(work.glob(str(config["checkpoint_glob"])))
    assert len(checkpoints) == int(config["expected_opaque_checkpoint_count"])
    assert not (ROOT / str(config["formal_output"])).exists()


def test_committed_recovery_attestation_validates_when_present() -> None:
    config = _config()
    attestation = ROOT / str(config["recovery_attestation"])
    if not attestation.exists():
        pytest.skip("blind-recovery attestation is created in the second commit")
    validated = recovery._validate_recovery_attestation(config, CONFIG)
    assert validated["checkpoint_payloads_parsed"] is False
    assert validated["opaque_checkpoint_count"] == 96
