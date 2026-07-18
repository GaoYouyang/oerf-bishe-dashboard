from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from site_tools.build_psu_n2_contract_rehearsal import (
    ALLOWED_FIELD_STATUSES,
    DECISION,
    SOURCE_PATHS,
    STATUS,
    build_from_tracked_sources,
    build_report,
)


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "site_tools/build_psu_n2_contract_rehearsal.py"


def _documents() -> dict[str, dict[str, object]]:
    return {
        ref: json.loads(path.read_text(encoding="utf-8"))
        for ref, path in SOURCE_PATHS.items()
    }


def _hashes() -> dict[str, str]:
    return {ref: "a" * 64 for ref in SOURCE_PATHS}


def test_tracked_sources_build_fail_closed_rehearsal() -> None:
    report = build_from_tracked_sources()
    assert report["status"] == STATUS
    assert report["decision"] == DECISION
    assert report["dataset_public_facts"]["view_count"] == 70
    assert report["dataset_public_facts"]["rotation_run_count"] == 10
    assert report["operator_interface_rehearsal"] == {
        "passed_its_own_interface_audit": True,
        "n2_operator_gate_passed": False,
        "reason": "An audited support operator is necessary but not sufficient for a bound N2 dataset record.",
    }
    assert all(not row["n2_gate_passed"] for row in report["gate_results"])
    assert not any(report["authorization"].values())


def test_field_statuses_and_known_public_negative_are_explicit() -> None:
    report = build_from_tracked_sources()
    rows = report["field_audit"]
    assert {row["status"] for row in rows.values()} <= ALLOWED_FIELD_STATUSES
    flowoff = rows["physical_mismatch.flow_off_repeats"]
    assert flowoff["status"] == "PUBLIC_NEGATIVE"
    assert flowoff["public_value"] == {
        "paper_reported_acquired_flow_off_frames_per_test": 2000,
        "independent_repeats_per_fixed_condition": 0,
        "temporal_covariance_authorized": False,
    }
    assert rows["physical_mismatch.primary"]["status"] == "FORBIDDEN_TO_INFER"
    assert rows["field_domain.truth_available"]["public_value"] is False


def test_report_is_privacy_safe_and_contains_no_local_path_tokens() -> None:
    rendered = json.dumps(build_from_tracked_sources(), ensure_ascii=False, sort_keys=True)
    forbidden = [
        r"/Users/",
        r"/home/",
        r"file://",
        r"~/",
        r"[A-Za-z]:\\",
        r"gaoyouyang",
        r"\.\./",
    ]
    for pattern in forbidden:
        assert re.search(pattern, rendered) is None


def test_source_snapshot_has_hashes_but_not_paths() -> None:
    report = build_from_tracked_sources()
    assert {row["ref"] for row in report["source_snapshot"]} == set(SOURCE_PATHS)
    for row in report["source_snapshot"]:
        assert re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
        assert "path" not in row


def test_flowoff_change_requires_manual_reaudit() -> None:
    documents = _documents()
    tampered = copy.deepcopy(documents)
    tampered["flowoff_inventory"]["temporal_repeat_assessment"][
        "independent_temporal_flowoff_frames_available_per_condition"
    ] = 50
    with pytest.raises(ValueError, match="flow-off inventory changed"):
        build_report(tampered, _hashes())


def test_unique_3d_truth_claim_change_requires_manual_reaudit() -> None:
    documents = _documents()
    tampered = copy.deepcopy(documents)
    tampered["heldout_protocol"]["claim_boundary"][
        "held_out_reprojection_is_unique_3d_truth"
    ] = True
    with pytest.raises(ValueError, match="reprojection truth boundary changed"):
        build_report(tampered, _hashes())


def test_cli_output_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    for output in (first, second):
        completed = subprocess.run(
            [sys.executable, str(BUILDER), "--output", str(output)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
    assert first.read_bytes() == second.read_bytes()
