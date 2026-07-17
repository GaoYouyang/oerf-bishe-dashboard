from __future__ import annotations

import copy
import csv
from dataclasses import replace
import json
from pathlib import Path

import pytest

import site_tools.run_observable_risk_fallback_smoke as runner_module
from site_tools.run_observable_risk_fallback_smoke import (
    EXPECTED_OUTPUT_FILES,
    load_config,
    run_smoke,
)
from site_tools.validate_observable_risk_fallback_smoke import validate_result_bundle


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "demo_t16_operator"
    / "configs"
    / "observable_risk_fallback_smoke_v1.json"
)


def _config() -> dict[str, object]:
    return copy.deepcopy(load_config(CONFIG_PATH))


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_smoke_captures_provenance_before_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result"
    observations: list[bool] = []

    def fake_git_state() -> dict[str, object]:
        observations.append(output.exists())
        return {"source_commit": "0" * 40, "source_worktree_dirty": False}

    monkeypatch.setattr(runner_module, "_git_state", fake_git_state)
    report = run_smoke(_config(), output_dir=output)
    assert observations == [False]
    assert report["provenance"]["source_worktree_dirty"] is False


def test_smoke_writes_four_split_no_auth_evidence_under_time_budget(tmp_path: Path) -> None:
    output = tmp_path / "result"
    report = run_smoke(_config(), output_dir=output)
    validated = validate_result_bundle(output)
    assert validated["schema_version"] == report["schema_version"]
    assert {path.name for path in output.iterdir()} == EXPECTED_OUTPUT_FILES
    assert report["runtime"]["wall_time_seconds"] < 10.0
    assert report["split_contract"]["split_unit"] == "COMPLETE_RIG"
    assert set(report["split_contract"]["role_rig_ids"]) == {
        "train",
        "model_selection",
        "risk_calibration",
        "fresh_geometry_ood",
    }
    assert report["aggregate"]["calibration_count"] == 12
    assert report["aggregate"]["fresh_count"] == 8
    assert report["risk_calibration"]["multiplicity_correction"] == "BONFERRONI_FROZEN_FINITE_GRID"
    assert report["risk_calibration"]["threshold_candidate_count"] >= 2
    assert (
        report["aggregate"]["calibration_takeover_coverage_lower_bound"]
        <= report["aggregate"]["calibration_takeover_coverage"]
    )
    assert report["gates"]["synthetic_micro_interface_gate_passed"] is False
    assert report["gates"]["future_paper_gate_passed"] is False
    assert report["gates"]["research_claim_authorized"] is False
    assert report["gates"]["real_bost_claim_authorized"] is False
    assert not any(report["claim_boundary"].values())

    with (output / "selection_rows.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert all(row["fallback_partition"] == "paired_cross" for row in rows)
    assert all(row["candidate_partition"] != "all_in_one_exact" for row in rows)
    for field in (
        "uses_truth",
        "uses_target",
        "uses_primitives",
        "uses_signed_matrix",
        "uses_exact_abs_operator",
        "uses_solver_trajectory",
    ):
        assert all(row[field] == "False" for row in rows)
    for filename in (
        "geometry_manifest.csv",
        "partition_audit_rows.csv",
        "selection_rows.csv",
        "risk_rows.csv",
        "metric_rows.csv",
        "trajectory_rows.csv",
    ):
        assert b"\r" not in (output / filename).read_bytes()


def test_smoke_aborts_when_actual_solver_operator_differs_from_primitives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = runner_module.generate_four_way_rigs

    def mismatched_rigs(**kwargs: object):
        rigs = original(**kwargs)  # type: ignore[arg-type]
        changed = rigs[0].signed_matrix.clone()
        changed[0, 0] += 0.01
        rigs[0] = replace(rigs[0], signed_matrix=changed)
        return rigs

    monkeypatch.setattr(runner_module, "generate_four_way_rigs", mismatched_rigs)
    with pytest.raises(RuntimeError, match="actual solver operator"):
        run_smoke(_config(), output_dir=tmp_path / "result")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["rigs"]["assignments"].__setitem__("fresh-00", "train"),
            "fresh_geometry_ood",
        ),
        (
            lambda value: value["partitions"].__setitem__(
                "candidate_names",
                ["singleton_factor", "paired_local", "paired_cross", "all_in_one_exact"],
            ),
            "catalogue",
        ),
        (
            lambda value: value["claim_boundary"].__setitem__("paper_superiority_claimed", True),
            "claim",
        ),
        (
            lambda value: value["future_paper_gate"].__setitem__("minimum_fresh_rigs", 8),
            "sample floors",
        ),
        (
            lambda value: value["model"].__setitem__("fresh_sensitive_access_forbidden", False),
            "sensitive",
        ),
    ],
)
def test_config_fails_closed(
    tmp_path: Path, mutate: object, message: str
) -> None:
    value = _config()
    mutate(value)  # type: ignore[operator]
    path = tmp_path / "config.json"
    _write_json(path, value)
    with pytest.raises(ValueError, match=message):
        load_config(path)
