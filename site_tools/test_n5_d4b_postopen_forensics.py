from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from site_tools.n5_d4b_postopen_forensics import (
    DEFAULT_OUTPUT,
    DEFAULT_PREREG_DIR,
    DEFAULT_RESULT_DIR,
    _contraction_methods,
    _point_location,
    run,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = (
    ROOT
    / "demo_t16_operator/results/"
    "n2_pvgr_n5_d4b_postopen_forensics_v1"
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_contraction_method_preserves_saved_float_semantics() -> None:
    jvp = np.array([1.0e16, 1.0, -1.0e16], dtype=np.float64)
    cotangent = np.ones(3, dtype=np.float64)
    tangent = np.array([1.0, 1.0], dtype=np.float64)
    vjp = np.array([0.5, 0.5], dtype=np.float64)
    methods = _contraction_methods(jvp, cotangent, tangent, vjp)

    assert methods["exact_binary_rational"]["lhs"] == 1.0
    assert methods["exact_binary_rational"]["rhs"] == 1.0
    assert methods["exact_binary_rational"]["relative_defect"] == 0.0
    assert methods["numpy_sum_float64_products"]["lhs"] == 0.0


def test_point_location_maps_stage_and_midpoint_order() -> None:
    layout = [
        {
            "group_index": 0,
            "label": "step=0,stage=k1",
            "point_start": 0,
            "point_stop": 4,
        },
        {
            "group_index": 64,
            "label": "curved_path_midpoints",
            "point_start": 4,
            "point_stop": 68,
        },
    ]
    stage = _point_location(2, layout, step_count=16)
    midpoint = _point_location(4 + 2 * 16 + 7, layout, step_count=16)

    assert (stage["step_index"], stage["stage"], stage["ray_index"]) == (0, "k1", 2)
    assert (midpoint["step_index"], midpoint["stage"], midpoint["ray_index"]) == (
        7,
        None,
        2,
    )


def test_committed_forensic_result_keeps_historical_decision_closed() -> None:
    result = _read_json(RESULT_DIR / "result.json")

    assert result["schema"] == "n2-pvgr-n5-d4b-postopen-forensics-1.0"
    assert result["diagnostic_decision"] == (
        "POSTOPEN_EXPLANATION_ONLY_NO_AUTHORIZATION"
    )
    assert result["historical_machine_decision"] == (
        "D4B_DERIVATIVE_CONTEXT_CHANGED_FAIL_CLOSED"
    )
    assert result["historical_decision_changed"] is False
    assert result["source_files_unchanged"] is True
    assert not any(result["forbidden_call_counts"].values())
    assert not any(result["claim_authorizations"].values())


def test_committed_dot_forensics_rule_out_final_reduction_order() -> None:
    result = _read_json(RESULT_DIR / "result.json")
    records = result["dot_contraction_forensics"]

    assert result["original_gate_threshold"] == 1.0e-10
    assert {record["map_id"] for record in records} == {
        "raw_curved_minus_straight",
        "paired_neumaier_residual",
    }
    exact_by_map = {
        record["map_id"]: record["exact_saved_array_relative_defect"]
        for record in records
    }
    assert exact_by_map["raw_curved_minus_straight"] == pytest.approx(
        1.8416778571393566e-10, rel=1e-15
    )
    assert exact_by_map["paired_neumaier_residual"] == pytest.approx(
        1.534312314569555e-10, rel=1e-15
    )
    assert all(
        not record["exact_saved_array_passes_original_threshold"]
        for record in records
    )
    assert result["dot_summary"] == {
        "failed_map_count": 2,
        "exact_saved_array_pass_count": 0,
        "final_contraction_rounding_explains_formal_failure": False,
        "component_signal_to_raw_residual_signal_ratio": pytest.approx(
            14466.672593632687
        ),
        "raw_residual_normwise_adjoint_defect": pytest.approx(
            5.205277793230917e-13
        ),
        "interpretation": "exact contractions of the stored float64 arrays remain above the frozen threshold; final reduction order is not a sufficient explanation",
    }

    decomposition = result["dot_context_decomposition"]
    assert len(decomposition) == 1
    maps = decomposition[0]["maps"]
    assert maps["curved_detector"]["relative_defect_by_dot_signal"] < 1e-13
    assert maps["straight_detector"]["relative_defect_by_dot_signal"] < 1e-13
    assert maps["raw_curved_minus_straight"][
        "relative_defect_by_dot_signal"
    ] > 1e-10
    assert decomposition[0]["derived"][
        "raw_absolute_defect_over_curved_absolute_defect"
    ] == pytest.approx(0.6307675506444103)


def test_committed_topology_replay_localizes_all_support_flips() -> None:
    result = _read_json(RESULT_DIR / "result.json")
    records = result["topology_support_forensics"]
    changed = {
        (record["cell_index"], record["direction_index"], item["side"]): item
        for record in records
        for item in record["perturbations"]
        if item["support_flip_count"]
    }
    expected = {
        (6, 0, "plus"): (4, 4, 0),
        (8, 0, "minus"): (1, 1, 0),
        (10, 0, "plus"): (3, 0, 3),
        (10, 0, "minus"): (1, 1, 0),
        (17, 0, "plus"): (2, 2, 0),
        (17, 0, "minus"): (4, 0, 4),
        (19, 0, "plus"): (2, 0, 2),
        (19, 0, "minus"): (2, 2, 0),
        (19, 1, "plus"): (2, 2, 0),
    }

    assert len(records) == 6
    assert set(changed) == set(expected)
    for key, counts in expected.items():
        item = changed[key]
        assert item["h"] == 0.01
        assert (
            item["support_flip_count"],
            item["support_0_to_1_count"],
            item["support_1_to_0_count"],
        ) == counts
        assert item["replay_matches_stored_hashes"] is True
        assert len(item["flips"]) == counts[0]
        for flip in item["flips"]:
            assert flip["offset_label"] in {
                "base",
                "+x",
                "+y",
                "+z",
                "-x",
                "-y",
                "-z",
            }
            assert flip["transition"] in {"0_to_1", "1_to_0"}
            assert 0 <= flip["ray_index"] < 4
            assert 0 <= flip["step_index"] < 16
    assert all(record["query_record_count"] == 2688 for record in records)
    assert all(
        record["largest_tested_two_sided_stable_h"] == 0.003
        and record["smallest_tested_changed_h"] == 0.01
        for record in records
    )
    assert result["topology_summary"] == {
        "failing_context_count": 6,
        "changed_signature_count": 9,
        "support_flip_count": 21,
        "support_0_to_1_count": 12,
        "support_1_to_0_count": 9,
        "all_replayed_hashes_match_frozen_rows": True,
        "all_changes_at_h_0_01": True,
        "all_contexts_two_sided_stable_at_or_below_h_0_003": True,
    }

    association = result["topology_fd_association"]
    assert association["topology_changed_map_gate_accounting"] == {
        "map_count": 24,
        "passing_map_count": 24,
        "all_map_gates_pass": True,
    }
    changed_required = association["topology_changed"][
        "required_h_max_map_relative_error"
    ]
    stable_required = association["topology_stable"][
        "required_h_max_map_relative_error"
    ]
    assert changed_required["count"] == 6
    assert stable_required["count"] == 58
    assert changed_required["maximum"] < 1e-6
    assert stable_required["maximum"] < 1e-6


def test_forensic_manifest_binds_all_public_artifacts() -> None:
    manifest = _read_json(RESULT_DIR / "manifest.json")
    for name, record in manifest["artifacts"].items():
        path = RESULT_DIR / name
        assert path.is_file()
        assert path.stat().st_size == record["bytes"]
        assert _sha256(path) == record["sha256"]


def test_forensic_runner_refuses_to_replace_committed_output() -> None:
    with pytest.raises(FileExistsError, match="refusing to replace"):
        run(DEFAULT_RESULT_DIR, DEFAULT_PREREG_DIR, DEFAULT_OUTPUT)
