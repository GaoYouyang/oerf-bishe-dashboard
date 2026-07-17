"""Pure-helper tests for the preregistered PDHG screen runner."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import site_tools.run_psu_b0_pdhg_screen as runner

from site_tools.run_psu_b0_pdhg_screen import (
    EXPECTED_CONTROLS_AND_REFERENCES,
    EXPECTED_METRIC_ROWS,
    GRAPH_STAGES,
    OPENED_REPLICATES,
    PDHG_ITERATIONS,
    PDHGPreflightInvalid,
    PDHGStressContext,
    REACTION_FAMILIES,
    REQUIRED_SOURCE_PATH_KEYS,
    STRESS_TRAJECTORY_SPECS,
    all_method_ids,
    alpha_zero_controls,
    audit_metric_row_counts,
    audit_pdhg_performance_rows,
    audit_pdhg_stability_preflight,
    component_pcgls_candidate,
    graph_budget_frontier,
    graph_pcgls_candidates,
    historical_method_specs,
    independent_pdhg_wall_time_ratios,
    paired_pdhg_wall_time_ratios,
    pdhg_candidate_grid,
    pdhg_screen_decision,
    rank_pdhg_candidates,
    regularization_vector_count,
    require_valid_pdhg_stability_preflight,
    run_pdhg_stability_preflight,
    run_e1_test_attestation,
    validate_clean_worktree,
    validate_frozen_config,
    validate_geometry_audit_manifest,
    validate_infrastructure_amendment,
    validate_metric_row_counts,
    validate_source_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "demo_t16_operator/configs/psu_b0_pdhg_scale_smoke_v1.json"
)
V2_CONFIG_PATH = ROOT / (
    "demo_t16_operator/configs/"
    "psu_b0_pdhg_scale_smoke_v2_infrastructure_amendment.json"
)


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_fixture(tmp_path: Path) -> tuple[dict, dict[str, Path]]:
    config = deepcopy(_config())
    paths: dict[str, Path] = {}
    digests: dict[str, str] = {}
    for source_key in REQUIRED_SOURCE_PATH_KEYS:
        path = tmp_path / f"{source_key}.txt"
        payload = f"frozen:{source_key}\n".encode()
        path.write_bytes(payload)
        config[source_key] = path.name
        paths[source_key] = path
        digests[source_key] = _sha256_bytes(payload)
    config["source_sha256"] = digests
    return config, paths


def _geometry_fixture(tmp_path: Path) -> tuple[dict, Path, Path]:
    config = deepcopy(_config())
    view_root = tmp_path / "geometry"
    view_root.mkdir()
    manifest_config = config["geometry_audit_manifest"]
    required_count = int(manifest_config["required_view_count"])
    required_status = str(manifest_config["required_view_bundle_status"])
    content = {
        "view_count": required_count,
        "resume_audit": {"sha256_verified": True},
        "views": [
            {"view_index": index, "bundle_status": required_status}
            for index in range(required_count)
        ],
    }
    payload = json.dumps(content, sort_keys=True).encode()
    path = view_root / str(manifest_config["filename"])
    path.write_bytes(payload)
    manifest_config["path"] = str(path.relative_to(tmp_path))
    manifest_config["sha256"] = _sha256_bytes(payload)
    return config, view_root, path


def _stable_stress_row(**kwargs: object) -> dict:
    iterations = int(kwargs["iterations"])
    return {
        "replicate": int(kwargs["replicate"]),
        "trajectory_id": str(kwargs["trajectory_id"]),
        "penalty": str(kwargs["penalty"]),
        "alpha": float(kwargs["alpha"]),
        "eta_label": str(kwargs["eta_label"]),
        "eta": float(kwargs["eta"]),
        "iterations": iterations,
        "truth_used": False,
        "sample_count": 1,
        "is_finite": True,
        "call_ledger_valid": True,
        "field_observations": [
            {
                "sample_index": 0,
                "maximum_relative_data_objective": 1.0,
                "head_fixed_point_median": 1.0,
                "tail_fixed_point_median": 0.5,
                "tail_data_objective_median": 0.5,
                "tail_to_head_fixed_point_ratio": 0.5,
                "maximum_dual_feasibility_violation": 0.0,
                "support_gauge_ok": True,
                "alpha_zero_edge_dual_exact_zero": (
                    True if float(kwargs["alpha"]) == 0.0 else None
                ),
            }
        ],
        "solver_forward_calls": iterations,
        "solver_adjoint_calls": iterations,
        "history_scorer_forward_calls": 0,
        "observed_physical_forward_calls": iterations,
        "observed_physical_adjoint_calls": iterations,
        "observed_gradient_calls": iterations,
        "observed_gradient_adjoint_calls": iterations,
    }


def _stress_contexts() -> list[PDHGStressContext]:
    return [
        PDHGStressContext(
            replicate=replicate,
            operator=object(),
            observation_b0=torch.zeros((1, 1, 2)),
            sigma_by_view=torch.ones((1, 1)),
            view_mask=torch.ones((1, 1)),
            rays_per_view=1,
            measurement_scale=1.0,
            regularization_vector_count=4,
            norm_estimate=object(),
        )
        for replicate in OPENED_REPLICATES
    ]


def test_cpu_float64_export_moves_device_before_dtype_cast() -> None:
    class MPSLikeTensor:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def detach(self) -> "MPSLikeTensor":
            return self

        def to(self, *args: object, **kwargs: object) -> "MPSLikeTensor":
            self.calls.append((args, kwargs))
            if len(args) > 1 or ("device" in kwargs and "dtype" in kwargs):
                raise TypeError("MPS cannot combine CPU transfer and float64 cast")
            return self

        def numpy(self) -> str:
            return "cpu-float64-array"

    value = MPSLikeTensor()

    exported = runner._cpu_float64_numpy(value)  # type: ignore[arg-type]

    assert exported == "cpu-float64-array"
    assert value.calls == [
        ((), {"device": "cpu"}),
        ((), {"dtype": torch.float64}),
    ]


def _ranking_summaries() -> list[dict]:
    summaries = []
    for index, candidate in enumerate(pdhg_candidate_grid(_config())):
        summaries.append(
            {
                **candidate,
                "ranking_valid": True,
                "mean_field_gain_vs_graph_budget_frontier_percent": 1.1,
                "field_gain_vs_graph_budget_frontier_p10_percent": -0.5,
                "field_harm_vs_graph_budget_frontier_over_one_percent_rate": 0.0,
                "worst_field_gain_vs_graph_budget_frontier_percent": -2.0,
                "mean_gradient_gain_vs_graph_budget_frontier_percent": 0.1,
                "mean_front_gain_vs_graph_budget_frontier": 0.01,
                "front_critical_mean_front_gain_vs_graph_budget_frontier": 0.01,
                "replicate_0_mean_field_gain_percent": 0.2,
                "replicate_8_mean_field_gain_percent": 0.3,
                "median_wall_time_ratio_vs_graph_frontier": 1.5,
                "fixture_order": index,
            }
        )
    return summaries


def _formal_metric_rows() -> list[dict]:
    rows = []
    for candidate in pdhg_candidate_grid(_config()):
        for replicate in OPENED_REPLICATES:
            for sample_index, family in enumerate(REACTION_FAMILIES):
                rows.append(
                    {
                        **candidate,
                        "replicate": replicate,
                        "sample_index": sample_index,
                        "reaction_family": family,
                        "field_relative_l2": 0.5,
                        "gradient_relative_l2": 0.6,
                        "front_top10_f1": 0.7,
                        "whitened_data_objective": 1.0,
                        "regularization_penalty": 0.25,
                        "total_objective": 1.25,
                        "primal_update_norm": 0.1,
                        "dual_feasibility_violation": 0.0,
                        "solver_elapsed_seconds": 0.2,
                        "output_amplitude_scale": 1.0,
                        "evaluation_forward_calls": 1,
                        "is_finite": True,
                        "support_gauge_ok": True,
                        "execution_status": "ok",
                    }
                )
    return rows


def test_frozen_grid_has_exact_count_and_identifiers() -> None:
    config = _config()
    audit = validate_frozen_config(config)
    candidates = pdhg_candidate_grid(config)

    assert audit["formal_candidate_count"] == 32
    assert len(candidates) == 32
    assert len({row["candidate_id"] for row in candidates}) == 32
    assert candidates[0]["candidate_id"] == "pdhg_tv_a1of256_k4"
    assert candidates[-1]["candidate_id"] == "pdhg_huber_a1of4_k32"
    assert {row["iterations"] for row in candidates} == set(PDHG_ITERATIONS)
    assert all(
        row["forward_calls"] == row["iterations"]
        and row["adjoint_calls"] == row["iterations"]
        for row in candidates
    )
    assert len(alpha_zero_controls(config)) == 4
    assert component_pcgls_candidate(config)["candidate_id"] == "component_s3_k4"
    assert (
        len(graph_pcgls_candidates(config))
        + 1
        + len(historical_method_specs(config))
        == 13
    )
    assert EXPECTED_CONTROLS_AND_REFERENCES == 4 + 13 == 17
    assert len(all_method_ids(config)) == 49
    assert config["geometry_audit_manifest"]["sha256"] == (
        "3d5b19cd0a52e9706660d63102ace1a37d8eb0728e4af5cdbc7312c75fb23261"
    )


def test_v2_infrastructure_amendment_binds_parent_files_and_metadata() -> None:
    config = json.loads(V2_CONFIG_PATH.read_text(encoding="utf-8"))

    frozen_audit = validate_frozen_config(config)
    parent_audit = validate_infrastructure_amendment(root=ROOT, config=config)

    assert frozen_audit["amendment"]["active"] is True
    assert parent_audit["active"] is True
    assert set(parent_audit["verified_parent_files"]) == {
        "parent_config",
        "parent_protocol",
        "parent_public_summary",
    }
    tampered = deepcopy(config)
    tampered["amendment"]["solver_math_unchanged"] = False
    with pytest.raises(ValueError, match="amendment metadata is not exact"):
        validate_frozen_config(tampered)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("synthetic_scale_by_view_uses_clean_truth", False),
        ("experimental_deployment_scale_available", True),
    ],
)
def test_truth_derived_scale_boundary_is_fail_closed(key: str, value: bool) -> None:
    config = deepcopy(_config())
    config["claim_boundary"][key] = value

    with pytest.raises(ValueError, match=key):
        validate_frozen_config(config)


def test_graph_frontier_is_one_pooled_stage_not_per_field_oracle() -> None:
    rows = []
    for stage in GRAPH_STAGES:
        if stage == 2:
            errors = (0.1, 0.9)
        elif stage == 3:
            errors = (0.4, 0.4)
        else:
            errors = (0.8, 0.8)
        for sample, error in enumerate(errors):
            rows.append(
                {
                    "candidate_id": f"graph_s3_k{stage}",
                    "replicate": 0,
                    "reaction_family": f"fixture_{sample}",
                    "field_relative_l2": error,
                }
            )

    frontier = graph_budget_frontier(
        rows,
        expected_fields_per_method=2,
    )

    assert {row["candidate_id"] for row in frontier.values()} == {
        "graph_s3_k3"
    }
    assert frontier[4]["pooled_mean_field_relative_l2"] == pytest.approx(0.4)
    # A forbidden per-field oracle would mix k2's 0.1 and k3's 0.4 -> 0.25.
    assert frontier[4]["pooled_mean_field_relative_l2"] != pytest.approx(0.25)


def test_ranking_and_one_winner_decision_follow_frozen_gate_order() -> None:
    summaries = _ranking_summaries()
    preferred_id = "pdhg_huber_a1of64_k16"
    preferred = next(row for row in summaries if row["candidate_id"] == preferred_id)
    preferred["mean_field_gain_vs_graph_budget_frontier_percent"] = 1.6
    preferred["field_gain_vs_graph_budget_frontier_p10_percent"] = -0.2

    ranking = rank_pdhg_candidates(summaries)
    decision = pdhg_screen_decision(
        ranking,
        count_audit={"valid": True},
        winner_timing_ratio=2.0,
    )

    assert ranking[0]["candidate_id"] == preferred_id
    assert decision["winner_candidate_id"] == preferred_id
    assert decision["status"] == "POSTOPEN_PDHG_SCALE_SIGNAL_ONLY"
    assert decision["fresh_authorized"] is False
    assert decision["neural_training_authorized"] is False

    scale_failure = deepcopy(ranking)
    scale_failure[0][
        "mean_field_gain_vs_graph_budget_frontier_percent"
    ] = 0.99
    failed = pdhg_screen_decision(
        scale_failure,
        count_audit={"valid": True},
        winner_timing_ratio=2.0,
    )
    assert failed["status"] == "POSTOPEN_PDHG_SCALE_NO_GO"

    structure_failure = deepcopy(ranking)
    structure_failure[0][
        "mean_gradient_gain_vs_graph_budget_frontier_percent"
    ] = -0.01
    failed = pdhg_screen_decision(
        structure_failure,
        count_audit={"valid": True},
        winner_timing_ratio=2.0,
    )
    assert failed["status"] == "POSTOPEN_PDHG_STRUCTURE_NO_GO"

    invalid = deepcopy(ranking)
    invalid[-1]["ranking_valid"] = False
    failed = pdhg_screen_decision(
        invalid,
        count_audit={"valid": True},
        winner_timing_ratio=2.0,
    )
    assert failed["status"] == "PDHG_PREFLIGHT_INVALID"


def test_metric_row_count_is_fail_closed_for_missing_or_duplicate_field() -> None:
    methods = all_method_ids(_config())
    rows = [
        {
            "candidate_id": method,
            "replicate": replicate,
            "reaction_family": family,
        }
        for method in methods
        for replicate in OPENED_REPLICATES
        for family in REACTION_FAMILIES
    ]
    audit = validate_metric_row_counts(rows, method_ids=methods)
    assert audit["valid"] is True
    assert audit["observed_row_count"] == EXPECTED_METRIC_ROWS == 784

    missing = rows[:-1]
    failed_audit = audit_metric_row_counts(missing, method_ids=methods)
    assert failed_audit["valid"] is False
    assert failed_audit["missing_row_count"] == 1
    with pytest.raises(ValueError, match="metric row ledger is incomplete"):
        validate_metric_row_counts(missing, method_ids=methods)

    duplicate = [*rows[:-1], rows[0]]
    failed_audit = audit_metric_row_counts(duplicate, method_ids=methods)
    assert failed_audit["valid"] is False
    assert failed_audit["missing_row_count"] == 1
    assert failed_audit["duplicate_key_count"] == 1


@pytest.mark.parametrize(
    "failure_mode",
    ["nonfinite", "support", "dual", "objective", "status", "calls", "timing"],
)
def test_formal_pdhg_rows_are_strictly_validated_before_ranking(
    failure_mode: str,
) -> None:
    rows = _formal_metric_rows()
    assert audit_pdhg_performance_rows(rows)["valid"] is True
    target = rows[0]
    if failure_mode == "nonfinite":
        target["field_relative_l2"] = float("nan")
    elif failure_mode == "support":
        target["support_gauge_ok"] = False
    elif failure_mode == "dual":
        target["dual_feasibility_violation"] = 1e-3
    elif failure_mode == "objective":
        target["total_objective"] = 99.0
    elif failure_mode == "status":
        target["execution_status"] = "solver_failed"
    elif failure_mode == "calls":
        target["forward_calls"] += 1
    else:
        target["solver_elapsed_seconds"] = 0.0

    audit = audit_pdhg_performance_rows(rows)
    assert audit["valid"] is False
    assert audit["failure_count"] >= 1


def test_source_sha256_manifest_verifies_every_frozen_relative_input(
    tmp_path: Path,
) -> None:
    config, paths = _source_fixture(tmp_path)

    audit = validate_source_sha256(root=tmp_path, config=config)

    assert set(audit) == {path.name for path in paths.values()}
    assert audit[paths["source_smoke_config"].name] == config["source_sha256"][
        "source_smoke_config"
    ]


@pytest.mark.parametrize("failure_mode", ["tampered", "missing", "digest_key"])
def test_source_sha256_manifest_fails_closed_for_changed_or_missing_input(
    tmp_path: Path,
    failure_mode: str,
) -> None:
    config, paths = _source_fixture(tmp_path)
    target = paths["source_smoke_config"]
    if failure_mode == "tampered":
        target.write_text("changed\n", encoding="utf-8")
        match = "SHA-256 mismatch"
    elif failure_mode == "missing":
        target.unlink()
        match = "source input is missing"
    else:
        del config["source_sha256"]["source_smoke_config"]
        match = "exactly cover"

    with pytest.raises(ValueError, match=match):
        validate_source_sha256(root=tmp_path, config=config)


def test_geometry_manifest_hash_and_semantics_are_verified(
    tmp_path: Path,
) -> None:
    config, view_root, _ = _geometry_fixture(tmp_path)

    audit = validate_geometry_audit_manifest(
        root=tmp_path,
        view_root=view_root,
        config=config,
    )

    assert audit["actual_sha256"] == config["geometry_audit_manifest"]["sha256"]
    assert audit["view_count_valid"] is True
    assert audit["resume_sha256_verified"] is True
    assert audit["all_view_bundle_statuses_valid"] is True


@pytest.mark.parametrize(
    "failure_mode", ["tampered", "missing", "wrong_path", "scope"]
)
def test_geometry_manifest_fails_closed_before_geometry_load(
    tmp_path: Path,
    failure_mode: str,
) -> None:
    config, view_root, path = _geometry_fixture(tmp_path)
    if failure_mode == "tampered":
        path.write_bytes(path.read_bytes() + b"\n")
        match = "SHA-256 mismatch"
    elif failure_mode == "missing":
        path.unlink()
        match = "manifest is missing"
    elif failure_mode == "wrong_path":
        other = tmp_path / "elsewhere"
        other.mkdir()
        config["geometry_audit_manifest"]["path"] = str(
            (other / path.name).relative_to(tmp_path)
        )
        match = "path does not match"
    else:
        config["geometry_audit_manifest"]["verification_scope"] = "wrong"
        match = "verification_scope"

    with pytest.raises(ValueError, match=match):
        validate_geometry_audit_manifest(
            root=tmp_path,
            view_root=view_root,
            config=config,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("view_count", "view_count mismatch"),
        ("resume", "sha256_verified must be true"),
        ("bundle", "invalid bundle_status"),
    ],
)
def test_geometry_manifest_semantics_fail_even_with_matching_hash(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    config, view_root, path = _geometry_fixture(tmp_path)
    content = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "view_count":
        content["view_count"] -= 1
    elif mutation == "resume":
        content["resume_audit"]["sha256_verified"] = False
    else:
        content["views"][0]["bundle_status"] = "UNVERIFIED"
    payload = json.dumps(content, sort_keys=True).encode()
    path.write_bytes(payload)
    config["geometry_audit_manifest"]["sha256"] = _sha256_bytes(payload)

    with pytest.raises(ValueError, match=match):
        validate_geometry_audit_manifest(
            root=tmp_path,
            view_root=view_root,
            config=config,
        )


def test_clean_worktree_audit_binds_execution_to_head(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        output = "abc123\n" if command[1:3] == ["rev-parse", "HEAD"] else ""
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    assert validate_clean_worktree(ROOT) == {
        "git_commit": "abc123",
        "worktree_clean": True,
    }


def test_clean_worktree_audit_rejects_dirty_state(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        output = (
            "abc123\n"
            if command[1:3] == ["rev-parse", "HEAD"]
            else " M site_tools/run_psu_b0_pdhg_screen.py\n"
        )
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="clean git worktree"):
        validate_clean_worktree(ROOT)


def test_e1_attestation_binds_junit_nodes_and_rejects_changed_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node_names = ["fixture.suite::test_case"]
    node_sha = hashlib.sha256(
        json.dumps(node_names, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    config = deepcopy(_config())
    config["preflight"]["e1_expected_test_case_count"] = 1
    config["preflight"]["e1_expected_node_sha256"] = node_sha

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return SimpleNamespace(
                stdout="abc123\n", stderr="", returncode=0
            )
        junit_argument = next(
            value for value in command if value.startswith("--junitxml=")
        )
        junit_path = Path(junit_argument.split("=", 1)[1])
        junit_path.write_text(
            '<testsuites><testsuite tests="1" failures="0" errors="0" '
            'skipped="0"><testcase classname="fixture.suite" '
            'name="test_case"/></testsuite></testsuites>',
            encoding="utf-8",
        )
        return SimpleNamespace(stdout="1 passed\n", stderr="", returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    audit = run_e1_test_attestation(ROOT, config=config)
    assert audit["valid"] is True
    assert audit["counts"] == {
        "tests": 1,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }

    config["preflight"]["e1_expected_test_case_count"] = 2
    with pytest.raises(PDHGPreflightInvalid, match="E1 pytest"):
        run_e1_test_attestation(ROOT, config=config)


def test_regularization_count_uses_potential_forward_neumann_sites() -> None:
    support = torch.zeros((3, 3, 3), dtype=torch.bool)
    support[1, 1, 1] = True

    assert int(torch.count_nonzero(support)) == 1
    assert regularization_vector_count(support) == 4


def test_independent_k_wall_time_does_not_inherit_max_k_elapsed() -> None:
    candidates = [
        {"candidate_id": "candidate_k4", "iterations": 4},
        {"candidate_id": "candidate_k32", "iterations": 32},
    ]
    frontier = {
        4: {"candidate_id": "graph_s3_k4"},
        32: {"candidate_id": "graph_s3_k32"},
    }
    pdhg_timing = {
        (0, "candidate_k4"): 4.0,
        (8, "candidate_k4"): 6.0,
        (0, "candidate_k32"): 32.0,
        (8, "candidate_k32"): 36.0,
    }
    graph_timing = {
        (0, "graph_s3_k4"): 2.0,
        (8, "graph_s3_k4"): 2.0,
        (0, "graph_s3_k32"): 4.0,
        (8, "graph_s3_k32"): 4.0,
    }

    ratios = independent_pdhg_wall_time_ratios(
        candidates,
        frontier=frontier,
        pdhg_timing=pdhg_timing,
        graph_timing=graph_timing,
    )

    assert ratios["candidate_k4"] == pytest.approx(2.5)
    assert ratios["candidate_k32"] == pytest.approx(8.5)
    assert ratios["candidate_k4"] != ratios["candidate_k32"]


def test_paired_timing_uses_median_of_ratios_and_requires_ab_ba() -> None:
    candidates = [{"candidate_id": "candidate_k4", "iterations": 4}]
    timing = {
        (0, "candidate_k4"): {
            "candidate_seconds": 2.0,
            "baseline_seconds": 1.0,
            "order": "candidate_then_baseline",
        },
        (8, "candidate_k4"): {
            "candidate_seconds": 9.0,
            "baseline_seconds": 3.0,
            "order": "baseline_then_candidate",
        },
    }

    ratios = paired_pdhg_wall_time_ratios(
        candidates,
        paired_timing=timing,
    )

    assert ratios["candidate_k4"] == pytest.approx(2.5)
    assert ratios["candidate_k4"] != pytest.approx(11.0 / 4.0)
    invalid = deepcopy(timing)
    invalid[(8, "candidate_k4")]["order"] = "candidate_then_baseline"
    with pytest.raises(ValueError, match="counterbalance"):
        paired_pdhg_wall_time_ratios(candidates, paired_timing=invalid)


def test_real_stress_orchestrator_consumes_all_three_trajectories_and_both_etas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = deepcopy(_config())
    config["solver"]["conservative_stress_eta"] = 0.37
    config["preflight"]["eta_stress_gate"]["etas"] = [0.9, 0.37]
    calls: list[dict] = []

    def fake_stress(**kwargs: object) -> dict:
        calls.append(dict(kwargs))
        return _stable_stress_row(**kwargs)

    monkeypatch.setattr(runner, "_run_pdhg_stress_trajectory", fake_stress)
    contexts = _stress_contexts()

    audit = run_pdhg_stability_preflight(
        contexts=contexts,
        config=config,
        device=torch.device("cpu"),
    )

    assert audit["valid"] is True
    assert len(calls) == 2 * 3 * 2 == 12
    assert {
        (call["replicate"], call["trajectory_id"], call["eta_label"])
        for call in calls
    } == {
        (replicate, trajectory_id, eta_label)
        for replicate in OPENED_REPLICATES
        for trajectory_id in STRESS_TRAJECTORY_SPECS
        for eta_label in ("conservative", "primary")
    }
    assert {
        call["eta"] for call in calls if call["eta_label"] == "conservative"
    } == {0.37}
    assert all(call["iterations"] == 32 for call in calls)
    assert all(call["head_iteration_first"] == 1 for call in calls)
    assert all(call["head_iteration_last"] == 8 for call in calls)
    assert all(call["tail_iteration_first"] == 25 for call in calls)
    assert all(call["tail_iteration_last"] == 32 for call in calls)
    data_only = [
        call for call in calls if call["trajectory_id"] == "pdhg_data_only_k32"
    ]
    assert all(call["penalty"] == "tv" and call["alpha"] == 0.0 for call in data_only)
    assert audit["call_ledger"]["observed_physical_forward_calls"] == 384
    assert audit["call_ledger"]["observed_physical_adjoint_calls"] == 384
    assert audit["call_ledger"]["history_scorer_forward_calls"] == 0


def test_stress_orchestrator_rejects_mapping_that_can_carry_truth() -> None:
    context = {
        "replicate": 0,
        "graph_operator": object(),
        "b0": object(),
        "truth": torch.ones((1, 1, 2, 2, 2)),
    }

    with pytest.raises(TypeError, match="truth-inaccessible"):
        run_pdhg_stability_preflight(
            contexts=[context],  # type: ignore[list-item]
            config=_config(),
            device=torch.device("cpu"),
        )


def test_stress_audit_rejects_nonfinite_constraints_and_relative_instability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()

    monkeypatch.setattr(
        runner,
        "_run_pdhg_stress_trajectory",
        lambda **kwargs: _stable_stress_row(**kwargs),
    )
    contexts = _stress_contexts()
    valid = run_pdhg_stability_preflight(
        contexts=contexts,
        config=config,
        device=torch.device("cpu"),
    )
    gate = config["preflight"]["eta_stress_gate"]

    mutations = []
    nonfinite = deepcopy(valid["runs"])
    nonfinite[0]["is_finite"] = False
    mutations.append(nonfinite)

    support_failure = deepcopy(valid["runs"])
    support_failure[0]["field_observations"][0]["support_gauge_ok"] = False
    mutations.append(support_failure)

    dual_failure = deepcopy(valid["runs"])
    dual_failure[0]["field_observations"][0][
        "maximum_dual_feasibility_violation"
    ] = 2e-7
    mutations.append(dual_failure)

    alpha_zero_failure = deepcopy(valid["runs"])
    data_only = next(
        row
        for row in alpha_zero_failure
        if row["trajectory_id"] == "pdhg_data_only_k32"
    )
    data_only["field_observations"][0][
        "alpha_zero_edge_dual_exact_zero"
    ] = False
    mutations.append(alpha_zero_failure)

    unstable = deepcopy(valid["runs"])
    primary = next(
        row
        for row in unstable
        if row["replicate"] == 0
        and row["trajectory_id"] == "pdhg_tv_a1of4_k32"
        and row["eta_label"] == "primary"
    )
    conservative = next(
        row
        for row in unstable
        if row["replicate"] == 0
        and row["trajectory_id"] == "pdhg_tv_a1of4_k32"
        and row["eta_label"] == "conservative"
    )
    primary["field_observations"][0]["tail_fixed_point_median"] = 11.0
    primary["field_observations"][0]["tail_data_objective_median"] = 3.0
    conservative["field_observations"][0]["tail_fixed_point_median"] = 1.0
    conservative["field_observations"][0]["tail_data_objective_median"] = 1.0
    mutations.append(unstable)

    for rows in mutations:
        audit = audit_pdhg_stability_preflight(
            rows,
            gate=gate,
            primary_eta=0.9,
            conservative_eta=0.5,
        )
        assert audit["status"] == "PDHG_PREFLIGHT_INVALID"
        assert audit["valid"] is False
        with pytest.raises(PDHGPreflightInvalid):
            require_valid_pdhg_stability_preflight(audit)


def test_prefix_trajectory_passes_bound_norm_and_separates_scorer_f_and_d(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOperator:
        spacing_xyz = (1.0, 1.0, 1.0)
        support = torch.ones((3, 3, 3), dtype=torch.float32)

        def __init__(self) -> None:
            self.forward_calls = 0
            self.adjoint_calls = 0

        def reset_call_counts(self) -> None:
            self.forward_calls = 0
            self.adjoint_calls = 0

        def call_report(self) -> dict[str, int]:
            return {
                "forward_calls": self.forward_calls,
                "adjoint_calls": self.adjoint_calls,
            }

        def __call__(self, volume: torch.Tensor) -> torch.Tensor:
            self.forward_calls += 1
            return torch.zeros((len(volume), 2, 2), dtype=volume.dtype)

        def adjoint(self, residual: torch.Tensor) -> torch.Tensor:
            self.adjoint_calls += 1
            return torch.zeros((len(residual), 1, 3, 3, 3), dtype=residual.dtype)

    operator = FakeOperator()
    norm_estimate = SimpleNamespace(
        data_norm_squared_upper=4.0,
        gradient_norm_squared_upper=9.0,
    )
    captured: dict[str, object] = {}

    def fake_solver(
        passed_operator: FakeOperator,
        observation: torch.Tensor,
        **kwargs: object,
    ) -> SimpleNamespace:
        captured.update(kwargs)
        count = int(kwargs["iterations"])
        volume = torch.zeros((len(observation), 1, 3, 3, 3))
        for _ in range(count):
            passed_operator(volume)
            passed_operator.adjoint(observation)
        checkpoints = tuple(int(value) for value in kwargs["checkpoint_iterations"])
        history = [
            {
                "edge_dual_maximum_norm": torch.zeros(len(observation)),
                "primal_update_norm": torch.zeros(len(observation)),
            }
            for _ in range(count)
        ]
        return SimpleNamespace(
            checkpoint_volumes={checkpoint: volume.clone() for checkpoint in checkpoints},
            history=history,
            step_contract_value=0.5,
            gradient_calls=count,
            gradient_adjoint_calls=count,
        )

    scorer_gradient_calls = 0

    def fake_edge_penalty(volume: torch.Tensor, **_: object) -> torch.Tensor:
        nonlocal scorer_gradient_calls
        scorer_gradient_calls += 1
        return torch.zeros(len(volume))

    monkeypatch.setattr(runner, "primal_dual_reconstruction", fake_solver)
    monkeypatch.setattr(runner, "isotropic_edge_penalty", fake_edge_penalty)
    candidates = pdhg_candidate_grid(_config())[:4]
    observation = torch.zeros((len(REACTION_FAMILIES), 2, 2))
    truth = torch.ones((len(REACTION_FAMILIES), 1, 3, 3, 3))

    rows, ledger, result = runner._run_pdhg_trajectory(
        operator=operator,
        observation_b0=observation,
        truth=truth,
        output_scale=torch.ones(len(REACTION_FAMILIES)),
        trajectory_candidates=candidates,
        ones_sigma=torch.ones((len(REACTION_FAMILIES), 1)),
        ones_mask=torch.ones((len(REACTION_FAMILIES), 1)),
        rays_per_view=2,
        measurement_scale=1.0,
        regularization_vector_count=4,
        norm_estimate=norm_estimate,
        eta=0.9,
        theta=1.0,
        huber_delta=0.5,
        device=torch.device("cpu"),
    )

    assert captured["norm_estimate"] is norm_estimate
    assert "data_norm_squared_upper" not in captured
    assert "gradient_norm_squared_upper" not in captured
    assert len(rows) == 4 * len(REACTION_FAMILIES)
    assert ledger["checkpoint_iterations"] == [4, 8, 16, 32]
    assert ledger["physical_solver_forward_calls"] == 32
    assert ledger["physical_solver_adjoint_calls"] == 32
    assert ledger["physical_solver_gradient_calls"] == 32
    assert ledger["physical_solver_gradient_adjoint_calls"] == 32
    assert ledger["prefix_forward_calls_saved"] == 28
    assert ledger["checkpoint_metric_forward_calls"] == 4
    assert ledger["checkpoint_metric_gradient_calls"] == 4
    assert ledger["checkpoint_metric_adjoint_calls"] == 0
    assert ledger["checkpoint_metric_gradient_adjoint_calls"] == 0
    assert operator.call_report() == {"forward_calls": 36, "adjoint_calls": 32}
    assert scorer_gradient_calls == 4
    assert result.gradient_calls == 32
    assert result.gradient_adjoint_calls == 32


def test_preflight_invalid_bundle_is_atomic_and_keeps_failure_evidence(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "formal-output"
    error = PDHGPreflightInvalid(
        {
            "status": "PDHG_PREFLIGHT_INVALID",
            "valid": False,
            "failures": ["fixture instability"],
            "call_ledger": {"solver_forward_calls": 384},
        },
        evidence={
            "config": {"sha256": "a" * 64},
            "protocol": {"actual_sha256": "b" * 64},
            "git": {"git_commit": "abc123", "worktree_clean": True},
            "environment": {"device": "mps", "dtype": "float32"},
        },
    )

    payload = runner._write_preflight_invalid_bundle(output_dir, error)

    assert output_dir.is_dir()
    assert payload["status"] == "PDHG_PREFLIGHT_INVALID"
    assert payload["truth_based_performance_rows_generated"] is False
    assert payload["performance_ranking_generated"] is False
    assert payload["performance_metric_row_count"] == 0
    stored = json.loads(
        (output_dir / "preflight_invalid.json").read_text(encoding="utf-8")
    )
    assert stored["audit"]["failures"] == ["fixture instability"]
    assert stored["evidence"]["git"]["worktree_clean"] is True
    assert (output_dir / "README.md").is_file()
    assert (output_dir / "checksums.sha256").is_file()
    assert not list(tmp_path.glob(".formal-output.preflight-invalid-*"))


def test_success_bundle_is_atomic_and_cleans_partial_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "formal-output"
    report = {
        "decision": {
            "status": "POSTOPEN_PDHG_SCALE_SIGNAL_ONLY",
            "winner_candidate_id": "pdhg_tv_a1of16_k8",
        },
        "formal_candidate_count": 32,
        "method_count": 49,
        "metric_row_count": 784,
        "configuration_audit": {
            "e1_attestation": {
                "valid": True,
                "junit_xml": "<testsuites tests=\"116\" />\n",
            }
        },
    }
    keyword_arguments = {
        "report": report,
        "rows": [{"method_id": "candidate", "field_error": 0.1}],
        "ranking": [{"rank": 1, "candidate_id": "candidate"}],
        "methods": [{"method_id": "candidate", "family": "pdhg"}],
        "call_ledger": {"actual_forward_calls": 1},
        "timing_bundle": {
            "environment": {"device": "cpu", "dtype": "float32"},
            "paired_ratios": [1.0],
        },
    }
    original_write_checksums = runner._write_checksums

    def fail_checksums(*_: object, **__: object) -> None:
        raise OSError("fixture interrupted before atomic publish")

    monkeypatch.setattr(runner, "_write_checksums", fail_checksums)
    with pytest.raises(OSError, match="fixture interrupted"):
        runner._write_success_bundle(output_dir, **keyword_arguments)
    assert not output_dir.exists()
    assert not list(tmp_path.glob(".formal-output.success-staging-*"))

    monkeypatch.setattr(runner, "_write_checksums", original_write_checksums)
    runner._write_success_bundle(output_dir, **keyword_arguments)
    assert output_dir.is_dir()
    expected_files = {
        "README.md",
        "candidate_summaries.csv",
        "checksums.sha256",
        "e1_attestation.json",
        "e1_pytest.xml",
        "environment.json",
        "method_summaries.csv",
        "metric_rows.csv",
        "operator_call_ledger.json",
        "report.json",
        "timing_audit.json",
    }
    assert {path.name for path in output_dir.iterdir()} == expected_files
    timing_audit = json.loads(
        (output_dir / "timing_audit.json").read_text(encoding="utf-8")
    )
    assert "environment" not in timing_audit
    assert not list(tmp_path.glob(".formal-output.success-staging-*"))
