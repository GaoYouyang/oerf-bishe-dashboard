from __future__ import annotations

import copy
import json

import pytest

import site_tools.validate_n2_pvgr_n3_grouped_factorial as validator


def _result() -> dict[str, object]:
    return json.loads(
        (validator.DEFAULT_RESULT / "result.json").read_text(encoding="utf-8")
    )


def test_current_bundle_passes_fail_closed_validation() -> None:
    report = validator.validate()
    assert report["status"] == "PASS_FAIL_CLOSED_BUNDLE_VALIDATION"
    assert report["machine_decision"] == (
        "GROUPED_FACTORIAL_FAIL_NO_FORWARD_AUTHORIZATION"
    )
    assert report["manifest_file_count"] == 41
    assert report["row_counts"]["physical_cells"] == 96
    assert report["row_counts"]["field_units"] == 8
    assert report["csv_row_counts"]["query_accounting.csv"] == 480
    assert report["claim_authorizations_all_false"] is True
    assert report["analysis_recovery_disclosed"] is True


def test_machine_decision_tampering_is_rejected() -> None:
    result = _result()
    tampered = copy.deepcopy(result)
    tampered["machine_decision"] = "OCBH_NOT_DOMINATED_CONDITIONAL_FIELD_VJP_GATE_NEXT"
    with pytest.raises(ValueError, match="machine decision"):
        validator._validate_machine_decision(tampered)


def test_dominance_boolean_tampering_is_rejected() -> None:
    result = _result()
    tampered = copy.deepcopy(result)
    tampered["picard_forward_dominance_rows"][0]["dominates_ocbh_forward_role"] = True
    with pytest.raises(ValueError, match="dominance gate summary"):
        validator._validate_machine_decision(tampered)


def test_nonfinite_json_number_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        validator._require_finite({"nested": [0.0, float("nan")]})
