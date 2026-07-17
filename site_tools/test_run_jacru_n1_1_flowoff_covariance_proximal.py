from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from site_tools import run_jacru_n1_1_flowoff_covariance_proximal as n11


SOURCE_PATHS = {
    "source_t0_config": Path("t0/config.json"),
    "source_t0_summary": Path("t0/results/summary.json"),
    "source_m2_7_config": Path("m2_7/config.json"),
    "source_m2_7_summary": Path("m2_7/results/summary.json"),
    "source_m2_8_config": Path("m2_8/config.json"),
    "source_m2_8_summary": Path("m2_8/results/summary.json"),
    "source_n1_0_config": Path("n1_0/config.json"),
    "source_n1_0_summary": Path("n1_0/results/summary.json"),
    "implementation_calibration_module": Path("implementation/calibration.py"),
    "implementation_dense_assembler": Path("implementation/dense_assembler.py"),
    "implementation_runner": Path("implementation/runner.py"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_validation_fixture(root: Path) -> dict[str, Any]:
    config: dict[str, Any] = {
        "source_t0_config": str(SOURCE_PATHS["source_t0_config"]),
        "source_t0_results": str(SOURCE_PATHS["source_t0_summary"].parent),
        "source_m2_7_config": str(SOURCE_PATHS["source_m2_7_config"]),
        "source_m2_7_results": str(SOURCE_PATHS["source_m2_7_summary"].parent),
        "source_m2_8_config": str(SOURCE_PATHS["source_m2_8_config"]),
        "source_m2_8_results": str(SOURCE_PATHS["source_m2_8_summary"].parent),
        "source_n1_0_config": str(SOURCE_PATHS["source_n1_0_config"]),
        "source_n1_0_results": str(SOURCE_PATHS["source_n1_0_summary"].parent),
        "implementation_calibration_module": str(
            SOURCE_PATHS["implementation_calibration_module"]
        ),
        "implementation_dense_assembler": str(
            SOURCE_PATHS["implementation_dense_assembler"]
        ),
        "implementation_runner": str(SOURCE_PATHS["implementation_runner"]),
    }
    for key, relative_path in SOURCE_PATHS.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"fixture": key}
        if key == "source_n1_0_summary":
            payload = {
                "status": "N1_0_OBSERVABLE_DISCREPANCY_STOPPING_NO_GO",
                "authorization": {"continue_flow_off_covariance_research": True},
            }
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        config[f"{key}_sha256"] = _sha256(path)
    return config


def _metric_row(
    *,
    candidate_id: str = "candidate",
    method: str = "jacru_m2",
    model_seed: int = 17,
    split: str = "development",
    field_gain: float = 0.08,
    h1_gain: float = 0.04,
    clean_ratio: float = 1.05,
    measured_ratio: float = 0.8,
    field_harm: bool = False,
    target_crossed: bool = True,
    raw_no_correction: bool = False,
    alpha: float = 1.0,
    closure_error: float = 1e-12,
    correction_norm: float = 2.0,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "method": method,
        "model_seed": model_seed,
        "split": split,
        "field_gain_to_best_matched": field_gain,
        "h1_gain_to_best_matched": h1_gain,
        "clean_reprojection_ratio_to_base": clean_ratio,
        "measured_reprojection_ratio_to_cgls": measured_ratio,
        "field_harm": field_harm,
        "target_crossed": target_crossed,
        "raw_no_correction": raw_no_correction,
        "alpha": alpha,
        "residual_closure_relative_error": closure_error,
        "correction_norm": correction_norm,
    }


def _decision_config() -> dict[str, Any]:
    return {
        "methods": ["jacru_m2"],
        "candidates": [
            {
                "id": "candidate",
                "calibration_mode": "paired_static",
                "uses_truth": False,
                "uses_exact_nuisance": False,
            }
        ],
        "decision_gates": {
            "development_field_gain_minimum": 0.05,
            "development_h1_gain_minimum": 0.03,
            "development_clean_reprojection_ratio_to_base_mean_maximum": 1.1,
            "development_clean_reprojection_ratio_to_base_worst_maximum": 1.5,
            "ood_field_gain_minimum": 0.02,
            "ood_h1_gain_minimum": 0.0,
            "ood_clean_reprojection_ratio_to_base_mean_maximum": 1.15,
            "ood_clean_reprojection_ratio_to_base_worst_maximum": 1.75,
            "field_harm_rate_maximum": 0.05,
            "worst_field_gain_minimum": -0.05,
            "minimum_target_crossing_rate": 0.95,
            "maximum_residual_closure_relative_error": 1e-10,
            "require_all_model_seed_mean_field_gains_positive": True,
        },
    }


def test_source_hash_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _source_validation_fixture(tmp_path)
    monkeypatch.setattr(n11, "ROOT", tmp_path)

    observed = n11._validate_sources(config)
    assert observed == {key: tmp_path / value for key, value in SOURCE_PATHS.items()}

    drifted = tmp_path / SOURCE_PATHS["source_m2_8_config"]
    drifted.write_text("drifted after freeze\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"source_m2_8_config hash drifted"):
        n11._validate_sources(config)


def test_k10_baseline_loader_selects_two_matched_references_per_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = tmp_path / "m2_7_results"
    results.mkdir()
    path = results / "matched_baseline_rows.csv"
    rows: list[dict[str, Any]] = []
    for case_index in range(30):
        case_id = f"case_{case_index:02d}"
        for kind in ("cgls_matched", "huber_pdhg_matched"):
            rows.append(
                {
                    "case_id": case_id,
                    "baseline_kind": kind,
                    "projection_iterations": 10,
                    "sentinel": f"{case_id}:{kind}:k10",
                }
            )
        rows.append(
            {
                "case_id": case_id,
                "baseline_kind": "cgls_matched",
                "projection_iterations": 8,
                "sentinel": "must-not-be-selected",
            }
        )
        rows.append(
            {
                "case_id": case_id,
                "baseline_kind": "unmatched_debug_only",
                "projection_iterations": 10,
                "sentinel": "must-not-be-selected",
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    monkeypatch.setattr(n11, "ROOT", tmp_path)
    loaded = n11._load_matched_baselines({"source_m2_7_results": "m2_7_results"})

    assert len(loaded) == 60
    assert set(kind for _, kind in loaded) == {
        "cgls_matched",
        "huber_pdhg_matched",
    }
    assert all(row["projection_iterations"] == "10" for row in loaded.values())
    assert all(row["sentinel"].endswith(":k10") for row in loaded.values())


@pytest.mark.parametrize(
    (
        "mean_policy",
        "proximal_policy",
        "selector_policy",
        "expected_mean",
        "expected_proximal",
        "expected_selector",
    ),
    [
        (
            "estimated_flowoff",
            "estimated_structured",
            "estimated_isotropic",
            "candidate_mean",
            "structured",
            "isotropic",
        ),
        (
            "zero",
            "estimated_isotropic",
            "estimated_structured",
            "zero",
            "isotropic",
            "structured",
        ),
        (
            "exact_persistent_bias_oracle",
            "exact_generator_oracle",
            "exact_generator_oracle",
            "exact_mean",
            "exact",
            "exact",
        ),
    ],
)
def test_candidate_policy_branches_select_only_declared_components(
    monkeypatch: pytest.MonkeyPatch,
    mean_policy: str,
    proximal_policy: str,
    selector_policy: str,
    expected_mean: str,
    expected_proximal: str,
    expected_selector: str,
) -> None:
    candidate_mean = torch.tensor([1.0, -2.0], dtype=torch.float64)
    exact_mean = torch.tensor([3.0, 4.0], dtype=torch.float64)
    structured = torch.diag(torch.tensor([2.0, 8.0], dtype=torch.float64))
    isotropic = 17.0 * torch.eye(2, dtype=torch.float64)
    exact = torch.diag(torch.tensor([5.0, 10.0], dtype=torch.float64))
    selection_samples = torch.tensor(
        [[1.0, -2.0], [2.0, -1.0]], dtype=torch.float64
    )
    calibration = {
        "candidate_mean": candidate_mean,
        "exact_mean": exact_mean,
        "estimate": SimpleNamespace(covariance=structured),
        "exact_covariance": exact,
        "payload": SimpleNamespace(selection_samples_uv=selection_samples),
    }
    tensors = {
        "candidate_mean": candidate_mean,
        "zero": torch.zeros_like(candidate_mean),
        "exact_mean": exact_mean,
        "structured": structured,
        "isotropic": isotropic,
        "exact": exact,
    }
    monkeypatch.setattr(n11, "isotropic_covariance_like", lambda _: isotropic)
    score_call: dict[str, Any] = {}

    def fake_score(
        samples_uv: torch.Tensor,
        *,
        mean_uv: torch.Tensor,
        covariance: torch.Tensor,
        quantile: float,
    ) -> tuple[float, torch.Tensor]:
        score_call.update(
            samples_uv=samples_uv,
            mean_uv=mean_uv,
            covariance=covariance,
            quantile=quantile,
        )
        return 12.5, torch.tensor([1.0], dtype=torch.float64)

    monkeypatch.setattr(n11, "_score_calibration_samples", fake_score)
    mean, proximal, selector, threshold = n11._candidate_components(
        candidate={
            "mean_policy": mean_policy,
            "proximal_covariance_policy": proximal_policy,
            "selector_covariance_policy": selector_policy,
            "discrepancy_quantile": 0.95,
        },
        calibration=calibration,
    )

    torch.testing.assert_close(mean, tensors[expected_mean])
    torch.testing.assert_close(proximal, tensors[expected_proximal])
    torch.testing.assert_close(selector, tensors[expected_selector])
    assert threshold == pytest.approx(12.5)
    assert score_call["samples_uv"] is selection_samples
    assert score_call["mean_uv"] is mean
    assert score_call["covariance"] is selector
    assert score_call["quantile"] == pytest.approx(0.95)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("mean_policy", "unsupported mean policy"),
        ("proximal_covariance_policy", "unsupported proximal covariance policy"),
        ("selector_covariance_policy", "unsupported selector covariance policy"),
    ],
)
def test_candidate_policy_rejects_unknown_branch(field: str, message: str) -> None:
    calibration = {
        "candidate_mean": torch.zeros(2, dtype=torch.float64),
        "exact_mean": torch.ones(2, dtype=torch.float64),
        "estimate": SimpleNamespace(covariance=torch.eye(2, dtype=torch.float64)),
        "exact_covariance": 2.0 * torch.eye(2, dtype=torch.float64),
        "payload": SimpleNamespace(selection_samples_uv=torch.zeros((2, 2))),
    }
    candidate = {
        "mean_policy": "estimated_flowoff",
        "proximal_covariance_policy": "estimated_structured",
        "selector_covariance_policy": "estimated_structured",
        "discrepancy_quantile": 0.95,
    }
    candidate[field] = "undeclared_policy"
    with pytest.raises(ValueError, match=message):
        n11._candidate_components(candidate=candidate, calibration=calibration)


def test_aggregate_reports_tail_harm_crossing_and_finite_alpha_statistics() -> None:
    rows = [
        _metric_row(
            field_gain=0.1,
            h1_gain=0.2,
            clean_ratio=1.0,
            measured_ratio=0.8,
            field_harm=False,
            target_crossed=True,
            raw_no_correction=False,
            alpha=100.0,
            closure_error=1e-12,
            correction_norm=2.0,
        ),
        _metric_row(
            field_gain=0.3,
            h1_gain=0.4,
            clean_ratio=1.2,
            measured_ratio=0.6,
            field_harm=True,
            target_crossed=False,
            raw_no_correction=True,
            alpha=math.inf,
            closure_error=2e-12,
            correction_norm=4.0,
        ),
    ]

    aggregate = n11._aggregate(rows)

    assert len(aggregate) == 1
    row = aggregate[0]
    assert row["case_count"] == 2
    assert row["field_gain_mean"] == pytest.approx(0.2)
    assert row["h1_gain_mean"] == pytest.approx(0.3)
    assert row["clean_reprojection_ratio_to_base_mean"] == pytest.approx(1.1)
    assert row["clean_reprojection_ratio_to_base_maximum"] == pytest.approx(1.2)
    assert row["measured_reprojection_ratio_to_cgls_mean"] == pytest.approx(0.7)
    assert row["field_harm_rate"] == pytest.approx(0.5)
    assert row["worst_field_gain"] == pytest.approx(0.1)
    assert row["target_crossing_rate"] == pytest.approx(0.5)
    assert row["raw_no_correction_rate"] == pytest.approx(0.5)
    assert row["log10_alpha_mean_finite"] == pytest.approx(2.0)
    assert row["residual_closure_relative_error_maximum"] == pytest.approx(2e-12)
    assert row["correction_norm_mean"] == pytest.approx(3.0)


def test_calibration_decisions_fail_closed_when_covariance_is_not_spd() -> None:
    config = {
        "flowoff_calibration": {"discrepancy_quantile": 0.95},
        "calibration_gates": {
            "audit_coverage_mean_minimum": 0.85,
            "audit_coverage_p90_error_maximum": 0.2,
            "condition_number_maximum": 10.0,
        },
    }
    rows = []
    for mode in ("paired_static", "unpaired_distribution"):
        rows.extend(
            {
                "mode": mode,
                "empirical_audit_coverage": coverage,
                "estimated_condition_number": condition,
                "estimated_minimum_eigenvalue": minimum_eigenvalue,
            }
            for coverage, condition, minimum_eigenvalue in (
                (0.9, 4.0, 0.2),
                (1.0, 5.0, 0.1 if mode == "paired_static" else -0.1),
            )
        )

    decisions = n11._calibration_decisions(rows, config)

    assert decisions["paired_static"]["passed"] is True
    assert decisions["unpaired_distribution"]["passed"] is False
    assert decisions["unpaired_distribution"]["checks"]["covariance_spd"] is False


def test_decisions_pass_all_gates_then_fail_closed_on_ood_gain() -> None:
    config = _decision_config()
    rows = []
    for seed, development_gain, ood_gain in ((17, 0.08, 0.03), (29, 0.06, 0.025)):
        rows.append(
            _metric_row(
                model_seed=seed,
                split="development",
                field_gain=development_gain,
                h1_gain=0.04,
            )
        )
        rows.append(
            _metric_row(
                model_seed=seed,
                split="ood",
                field_gain=ood_gain,
                h1_gain=0.01,
                clean_ratio=1.1,
            )
        )
    calibration = {"paired_static": {"passed": True}}

    passing = n11._decisions(
        rows=rows, config=config, calibration_decisions=calibration
    )
    assert len(passing) == 1
    assert passing[0]["passed"] is True
    assert all(passing[0]["checks"].values())
    assert passing[0]["dense_ceiling_only"] is True

    for row in rows:
        if row["split"] == "ood":
            row["field_gain_to_best_matched"] = 0.0
    failing = n11._decisions(
        rows=rows, config=config, calibration_decisions=calibration
    )
    assert failing[0]["passed"] is False
    assert failing[0]["checks"]["ood_field_gain"] is False
    assert failing[0]["checks"]["all_seed_means_positive"] is False


def test_frozen_config_has_seven_labeled_candidates_and_strict_claim_boundaries() -> None:
    config = n11._read_json(n11.DEFAULT_CONFIG)
    candidates = config["candidates"]

    assert config["status"] == "FROZEN_BEFORE_FIRST_FORMAL_N1_1_EXECUTION"
    assert [candidate["id"] for candidate in candidates] == [
        "paired_isotropic_sensor",
        "paired_structured_sensor",
        "unpaired_isotropic_sensor",
        "unpaired_structured_sensor",
        "paired_exact_mean_iid_sensor_oracle",
        "unpaired_exact_covariance_sensor_oracle",
        "paired_structured_truth_residual_oracle",
    ]
    assert len({candidate["id"] for candidate in candidates}) == 7
    assert {
        candidate["id"] for candidate in candidates if candidate["uses_truth"]
    } == {"paired_structured_truth_residual_oracle"}
    assert {
        candidate["id"]
        for candidate in candidates
        if candidate["uses_exact_nuisance"]
    } == {
        "paired_exact_mean_iid_sensor_oracle",
        "unpaired_exact_covariance_sensor_oracle",
    }

    boundary = config["claim_boundary"]
    assert boundary["uses_only_opened_synthetic_t0"] is True
    assert boundary["proposal_checkpoint_selection_used_development_truth"] is True
    assert boundary["flowoff_payload_exposes_hidden_scale_or_nuisance"] is False
    assert boundary["paired_flowoff_assumes_same_session_bias_stability"] is True
    assert (
        boundary[
            "clean_reprojection_is_same_voxel_operator_against_continuous_clean_target"
        ]
        is True
    )
    assert boundary["clean_reprojection_is_independent_renderer"] is False
    assert boundary["dense_aa_t_ceiling_is_deployable"] is False
    assert boundary["may_claim_runtime_or_efficiency"] is False
    assert boundary["may_claim_method_superiority"] is False
    assert boundary["may_claim_real_bost_generalization"] is False
    assert boundary["may_open_fresh_or_final"] is False

    limitations = config["current_selector_limitations"]
    assert limitations["uses_only_global_whitened_discrepancy"] is True
    assert limitations["has_per_camera_upper_gate"] is False
    assert limitations["has_lower_discrepancy_gate"] is False
    assert "mechanism diagnostic only" in limitations["interpretation"]
    assert "camera-wise tail protection" in limitations["interpretation"]
    assert "lower gate" in limitations["interpretation"]
