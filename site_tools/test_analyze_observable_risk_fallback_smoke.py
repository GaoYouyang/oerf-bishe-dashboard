from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

import site_tools.analyze_observable_risk_fallback_smoke as analyzer
from site_tools.run_observable_risk_fallback_smoke import RISK_FIELDS


def _validated_report() -> dict[str, Any]:
    return {
        "schema_version": "observable-risk-fallback-smoke-report-1.4",
        "status": "DEVELOPMENT_ONLY_SYNTHETIC_RCCF_INTERFACE_GATE",
        "evidence_scope": "SYNTHETIC_FOUR_SPLIT_RCCF_CPU_MICRO_SMOKE_ONLY",
        "risk_calibration": {
            "risk_upper_bound": 0.9916666666666667,
            "takeover_coverage": 1.0 / 12.0,
            "takeover_coverage_lower_bound": 0.0006971110443230177,
            "authorized_takeover_coverage": 0.0,
            "authorized_takeover_coverage_lower_bound": 0.0,
            "development_gate_passed": False,
        },
        "aggregate": {
            "calibration_count": 12,
            "calibration_accepted_count": 1,
            "calibration_failure_count": 0,
            "fresh_count": 8,
            "fresh_takeover_count": 0,
            "fresh_takeover_coverage": 0.0,
            "fresh_fallback_rate": 1.0,
            "fresh_harm_count": 0,
            "fresh_selection_conditional_harm_rate": None,
            "fresh_worst_takeover_field_harm": None,
            "fresh_worst_takeover_residual_harm": None,
            "partition_audit_count": 136,
            "partition_audit_violation_count": 0,
            "operator_decomposition_mismatch_count": 0,
        },
        "gates": {
            "synthetic_micro_interface_gate_passed": False,
            "future_paper_gate_passed": False,
            "research_claim_authorized": False,
            "real_bost_claim_authorized": False,
            "generalization_claim_authorized": False,
            "paper_superiority_claim_authorized": False,
        },
        "claim_boundary": {
            "synthetic_interface_success_claimed": False,
            "real_bost_claimed": False,
            "generalization_claimed": False,
            "paper_superiority_claimed": False,
            "deeponet_fno_nerif_superiority_claimed": False,
        },
    }


def _risk_row(rig_id: str, split_role: str, *, fallback_used: bool) -> dict[str, Any]:
    return {
        "rig_id": rig_id,
        "split_role": split_role,
        "candidate_partition": "paired_local",
        "risk_score": 0.2 if not fallback_used else 0.8,
        "acceptance_threshold": 0.4,
        "support_gate_passed": True,
        "observed_field_harm_vs_fallback": 0.0,
        "observed_residual_harm_vs_fallback": 0.0,
        "harm_failure": False,
        "fallback_used": fallback_used,
        "selection_frozen_before_offline_evaluation": True,
    }


@pytest.fixture
def synthetic_bundle(tmp_path: Path) -> Path:
    root = tmp_path / "full-result"
    root.mkdir()
    config = {
        "development_gate": {
            "maximum_risk_upper": 0.5,
            "minimum_takeover_coverage": 0.25,
        }
    }
    (root / "config_snapshot.json").write_text(
        json.dumps(config) + "\n", encoding="utf-8", newline="\n"
    )
    rows = [
        _risk_row(f"cal-{index:02d}", "risk_calibration", fallback_used=index != 0)
        for index in range(12)
    ]
    rows.extend(
        _risk_row(f"fresh-{index:02d}", "fresh_geometry_ood", fallback_used=True)
        for index in range(8)
    )
    with (root / "risk_rows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(RISK_FIELDS), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return root


def test_run_requires_strict_source_and_writes_only_bounded_negative_bundle(
    synthetic_bundle: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, bool]] = []

    def fake_validate(
        root: Path, *, require_clean_source: bool = False
    ) -> dict[str, Any]:
        calls.append((root, require_clean_source))
        return _validated_report()

    monkeypatch.setattr(analyzer, "validate_result_bundle", fake_validate)
    output = tmp_path / "public"
    summary = analyzer.run(synthetic_bundle, output)

    assert calls == [(synthetic_bundle, True)]
    assert {path.name for path in output.iterdir()} == analyzer.PUBLIC_FILES
    assert summary["schema_version"] == analyzer.PUBLIC_SCHEMA
    assert summary["status"] == analyzer.PUBLIC_STATUS
    assert "FAILED" in summary["result_label"]
    assert "NO AUTHORITY" in summary["result_label"]
    assert summary["calibration"]["risk_upper_bound"] == pytest.approx(
        0.9916666666666667
    )
    assert summary["calibration"]["maximum_risk_upper_gate"] == 0.5
    assert summary["calibration"]["diagnostic_takeover_coverage"] == pytest.approx(
        1.0 / 12.0
    )
    assert summary["calibration"]["authorized_takeover_coverage"] == 0.0
    assert summary["fresh"]["fallback_count"] == 8
    assert summary["fresh"]["takeover_count"] == 0
    assert summary["certificates_and_decomposition"]["partition_audit_count"] == 136
    assert summary["certificates_and_decomposition"][
        "operator_decomposition_mismatch_count"
    ] == 0
    assert all(value is False for value in summary["claim_boundary"].values())
    assert summary["claim_boundary"]["timing_advantage_claimed"] is False
    assert (output / "diagnostic.png").stat().st_size > 20_000
    readme = (output / "README.md").read_text(encoding="utf-8")
    assert "FAILED / NO AUTHORITY / DEVELOPMENT ONLY / NEGATIVE RESULT" in readme
    assert "No timing comparison or timing advantage claim is made." in readme


def test_public_csv_schemas_and_text_line_endings_are_frozen(
    synthetic_bundle: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        analyzer,
        "validate_result_bundle",
        lambda _root, *, require_clean_source: _validated_report(),
    )
    output = tmp_path / "public"
    analyzer.run(synthetic_bundle, output)

    expected = {
        "calibration_inspection.csv": analyzer.CALIBRATION_FIELDS,
        "fresh_inspection.csv": analyzer.FRESH_FIELDS,
    }
    for name, fields in expected.items():
        payload = (output / name).read_bytes()
        assert b"\r" not in payload
        assert payload.endswith(b"\n")
        with (output / name).open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            assert tuple(reader.fieldnames or ()) == fields
            rows = list(reader)
        assert len(rows) == (12 if name.startswith("calibration") else 8)
        assert all(tuple(row) == fields for row in rows)

    for name in ("README.md", "summary.json"):
        payload = (output / name).read_bytes()
        assert b"\r" not in payload
        assert payload.endswith(b"\n")
    parsed = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert parsed["public_csv_schema"] == {
        name: list(fields) for name, fields in expected.items()
    }


def test_strict_validation_and_schema_fail_before_public_output(
    synthetic_bundle: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "public"

    def reject_dirty(_root: Path, *, require_clean_source: bool) -> dict[str, Any]:
        assert require_clean_source is True
        raise ValueError("clean-source validation requires a clean entry worktree")

    monkeypatch.setattr(analyzer, "validate_result_bundle", reject_dirty)
    with pytest.raises(ValueError, match="clean-source"):
        analyzer.run(synthetic_bundle, output)
    assert not output.exists()

    rows = list(csv.DictReader((synthetic_bundle / "risk_rows.csv").open(newline="")))
    with (synthetic_bundle / "risk_rows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fields = list(RISK_FIELDS[:-1])
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows([{field: row[field] for field in fields} for row in rows])
    monkeypatch.setattr(
        analyzer,
        "validate_result_bundle",
        lambda _root, *, require_clean_source: _validated_report(),
    )
    with pytest.raises(analyzer.PublicExportError, match="schema drift"):
        analyzer.run(synthetic_bundle, output)
    assert not output.exists()
