from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

import site_tools.run_certified_grouped_majorizer_smoke as runner_module
from site_tools.run_certified_grouped_majorizer_smoke import (
    CONFIG_SCHEMA_VERSION,
    EVIDENCE_SCOPE,
    EXPECTED_OUTPUT_FILES,
    METHODS,
    OUTPUT_PAYLOADS,
    REPORT_SCHEMA_VERSION,
    STATUS,
    load_config,
    run_smoke,
    validate_result_bundle,
)


def _config() -> dict[str, object]:
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "status": STATUS,
        "evidence_scope": EVIDENCE_SCOPE,
        "seeds": {"geometry": 101, "noise": 202},
        "rigs": {
            "row_count": 10,
            "column_count": 8,
            "primitive_count": 6,
            "split_unit": "COMPLETE_RIG",
            "random_ray_split_used": False,
            "assignments": {
                "train-00": "train",
                "train-01": "train",
                "train-02": "train",
                "train-03": "train",
                "cal-00": "safety_calibration",
                "cal-01": "safety_calibration",
                "fresh-00": "fresh_geometry_ood",
                "fresh-01": "fresh_geometry_ood",
                "fresh-02": "fresh_geometry_ood",
            },
        },
        "partitions": {
            "fixed_partition_names": ["paired_local", "paired_cross", "triad_bridge"],
            "global_candidate_names": [
                "singleton_factor",
                "paired_local",
                "paired_cross",
                "triad_bridge",
            ],
            "selector_candidate_names": [
                "singleton_factor",
                "paired_local",
                "paired_cross",
                "triad_bridge",
            ],
            "exact_oracle_name": "all_in_one_exact",
            "all_in_one_for_selector_forbidden": True,
            "cost_proxy_role": "ANALYTIC_PROXY_NOT_WALL_TIME",
        },
        "selector": {
            "model_class": "DEPTH_ONE_GEOMETRY_STUMP",
            "train_top_k": 16,
            "fresh_exact_truth_target_access_forbidden": True,
        },
        "solver": {"eta": 0.7, "theta": 1.0, "checkpoints": [0, 1, 2, 4, 8]},
        "runtime": {
            "device": "cpu",
            "dtype": "torch.float64",
            "timing_role": "MEASURED_SINGLE_RUN_NONCOMPARATIVE",
        },
        "claim_boundary": {
            "real_bost_claimed": False,
            "generalization_claimed": False,
            "paper_superiority_claimed": False,
            "exact_oracle_is_deployable": False,
        },
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _refresh_checksums(output: Path) -> None:
    (output / "checksums.sha256").write_text(
        "".join(
            f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}\n"
            for name in OUTPUT_PAYLOADS
        ),
        encoding="ascii",
    )


def test_smoke_freezes_git_state_before_writing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result"
    observations: list[bool] = []

    def fake_git_state() -> dict[str, object]:
        observations.append(output.exists())
        return {"source_commit": "0" * 40, "source_worktree_dirty": False}

    monkeypatch.setattr(runner_module, "_git_state", fake_git_state)
    report = runner_module.run_smoke(_config(), output_dir=output)

    assert observations == [False]
    assert report["provenance"]["source_worktree_dirty"] is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.__setitem__("unexpected", 1), "keys differ"),
        (lambda value: value["runtime"].__setitem__("device", "mps"), "runtime"),  # type: ignore[index]
        (
            lambda value: value["rigs"].__setitem__("random_ray_split_used", True),  # type: ignore[index]
            "complete-rig",
        ),
        (
            lambda value: value["selector"].__setitem__(  # type: ignore[index]
                "fresh_exact_truth_target_access_forbidden", False
            ),
            "fresh exact",
        ),
        (
            lambda value: value["partitions"].__setitem__(  # type: ignore[index]
                "selector_candidate_names",
                ["singleton_factor", "paired_local", "paired_cross", "all_in_one_exact"],
            ),
            "selector candidate",
        ),
        (
            lambda value: value["partitions"].__setitem__(  # type: ignore[index]
                "cost_proxy_role", "MEASURED_RUNTIME"
            ),
            "cost role",
        ),
    ],
)
def test_config_fails_closed(tmp_path: Path, mutate: object, message: str) -> None:
    value = _config()
    mutate(value)  # type: ignore[operator]
    path = tmp_path / "config.json"
    _write_json(path, value)
    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_smoke_writes_certified_geometry_only_negative_or_positive_evidence(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result"
    report = run_smoke(_config(), output_dir=output)
    validated = validate_result_bundle(output)
    assert validated["schema_version"] == REPORT_SCHEMA_VERSION
    assert report["mathematical_contract"]["certificate"].startswith("M_P >= abs(A)")
    assert report["selection"]["fresh_exact_truth_target_access"] is False
    assert report["selection"]["all_in_one_exact_available_to_selector"] is False
    assert "all_in_one_exact" not in report["selection"]["selector_allowed_partitions"]
    assert report["method_contracts"]["all_in_one_exact_oracle"].startswith("NONDEPLOYABLE")
    assert report["decision"]["all_partition_audits_zero_violation"] is True
    assert report["decision"]["selector_all_fresh_schur_safe"] is True
    assert report["decision"]["advantage_not_due_to_all_in_one_exact"] is True
    expected_gate = (
        report["decision"]["selector_beats_train_selected_fixed_on_every_fresh_rig"]
        and report["decision"]["selector_beats_train_selected_fixed_on_every_safety_rig"]
        and report["decision"]["geometry_adaptation_observed_on_fresh"]
        and report["decision"]["advantage_not_due_to_all_in_one_exact"]
        and report["decision"]["equal_fresh_A_A_transpose_call_budget"]
    )
    assert report["decision"]["research_claim_authorized"] is expected_gate
    assert report["decision"]["real_bost_claim_authorized"] is False
    assert {path.name for path in output.iterdir()} == EXPECTED_OUTPUT_FILES

    with (output / "metric_rows.csv").open(encoding="utf-8") as handle:
        metrics = list(csv.DictReader(handle))
    with (output / "geometry_manifest.csv").open(encoding="utf-8") as handle:
        geometry = list(csv.DictReader(handle))
    assert {row["method"] for row in metrics} == set(METHODS)
    assert len(metrics) == 3 * len(METHODS)
    assert len(geometry) == 9
    assert all(row["total_violation_count"] == "0" for row in metrics)
    assert all(row["cost_proxy_definition"] == "ANALYTIC_PROXY_NOT_WALL_TIME" for row in metrics)
    assert all(len(row["geometry_feature_sha256"]) == 64 for row in geometry)
    assert report["provenance"]["geometry_manifest_sha256"] == hashlib.sha256(
        (output / "geometry_manifest.csv").read_bytes()
    ).hexdigest()
    for rig_id in {row["rig_id"] for row in metrics}:
        call_rows = [row for row in metrics if row["rig_id"] == rig_id]
        assert len(
            {
                (
                    row["signed_forward_solver_calls"],
                    row["signed_transpose_solver_calls"],
                    row["signed_forward_evaluation_calls"],
                )
                for row in call_rows
            }
        ) == 1


def test_validator_rejects_safety_tamper_even_with_refreshed_checksums(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result"
    run_smoke(_config(), output_dir=output)
    audit_path = output / "partition_audit_rows.csv"
    with audit_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    rows[0]["total_violation_count"] = "1"
    with audit_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _refresh_checksums(output)
    with pytest.raises(ValueError, match="independent safety reconstruction mismatch"):
        validate_result_bundle(output)


def test_geometry_manifest_and_hash_are_assignment_reorder_invariant(
    tmp_path: Path,
) -> None:
    first = _config()
    second = copy.deepcopy(first)
    assignments = second["rigs"]["assignments"]  # type: ignore[index]
    second["rigs"]["assignments"] = dict(reversed(list(assignments.items())))  # type: ignore[index,union-attr]
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    first_report = run_smoke(first, output_dir=first_output)
    second_report = run_smoke(second, output_dir=second_output)
    assert (first_output / "geometry_manifest.csv").read_bytes() == (
        second_output / "geometry_manifest.csv"
    ).read_bytes()
    assert first_report["provenance"]["geometry_manifest_sha256"] == second_report[
        "provenance"
    ]["geometry_manifest_sha256"]


def test_validator_rejects_decision_tamper_even_with_refreshed_checksum(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result"
    report = run_smoke(_config(), output_dir=output)
    report["decision"]["research_claim_authorized"] = not report["decision"][
        "research_claim_authorized"
    ]
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    _refresh_checksums(output)
    with pytest.raises(ValueError, match="decision research_claim_authorized mismatch"):
        validate_result_bundle(output)


def test_validator_rejects_metric_replay_tamper_even_with_refreshed_checksum(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result"
    run_smoke(_config(), output_dir=output)
    metric_path = output / "metric_rows.csv"
    with metric_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    rows[0]["final_field_relative_l2"] = "999.0"
    with metric_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _refresh_checksums(output)
    with pytest.raises(ValueError, match="metric trajectory mismatch"):
        validate_result_bundle(output)


def test_validator_rejects_trajectory_replay_tamper_even_with_refreshed_checksum(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result"
    run_smoke(_config(), output_dir=output)
    trajectory_path = output / "trajectory_rows.csv"
    with trajectory_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    rows[0]["solution_l2"] = "123.0"
    with trajectory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _refresh_checksums(output)
    with pytest.raises(ValueError, match="trajectory replay mismatch"):
        validate_result_bundle(output)


def test_validator_rejects_construction_cost_tamper_even_with_refreshed_checksum(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result"
    run_smoke(_config(), output_dir=output)
    cost_path = output / "construction_cost_rows.csv"
    with cost_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    rows[0]["cost_proxy_units"] = "1"
    with cost_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _refresh_checksums(output)
    with pytest.raises(ValueError, match="construction cost mismatch"):
        validate_result_bundle(output)


def test_validator_rejects_aggregate_tamper_even_with_refreshed_checksum(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result"
    report = run_smoke(_config(), output_dir=output)
    report["aggregate_fresh_geometry_ood"]["singleton_factor"][
        "mean_final_field_relative_l2"
    ] = 999.0
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    _refresh_checksums(output)
    with pytest.raises(ValueError, match="aggregate mismatch"):
        validate_result_bundle(output)
