from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / (
    "docs/poolfire_exact_k1_dual_view_ridge_v142_4_public_summary.json"
)
EXECUTION_STATUS = (
    "PASS_INDEPENDENT_EXECUTION_EXACT_K1_DUAL_VIEW_RIDGE_V142_4_1"
)
SCIENTIFIC_DECISION = "FAIL_SHARED_LINEAR_RIDGE_REPRESENTATION_V142_4"
ARMS = ("formal_view", "independent_view", "joint_ls_warm_restart_k1")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _arm(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "complete_gate_pass": bool(row["complete_gate_pass"]),
        "passing_cells": int(row["passing_cells"]),
        "passing_trajectories": int(row["passing_trajectories"]),
        "maximum_metric_ratio": float(row["maximum_metric_ratio"]),
        "exact_calls_per_sample": {
            "A": int(row["exact_calls_per_sample"]["A"]),
            "AT": int(row["exact_calls_per_sample"]["AT"]),
        },
        "trajectory_summary": row["trajectory_summary"],
    }


def build(validation_report: Path, output: Path) -> dict[str, Any]:
    source = json.loads(Path(validation_report).read_text(encoding="utf-8"))
    checks = source.get("checks", {})
    _require(source.get("status") == EXECUTION_STATUS, "independent execution changed")
    _require(
        source.get("scientific_decision") == SCIENTIFIC_DECISION,
        "scientific decision changed",
    )
    _require(len(checks) == 19 and all(checks.values()), "integrity checks incomplete")
    _require(
        int(source.get("sample_count", -1)) == 3700
        and int(source.get("trajectory_count", -1)) == 5,
        "evaluation roster changed",
    )
    _require(
        set(source.get("arm_results", {})) == set(ARMS),
        "evaluated arm roster changed",
    )
    _require(
        source.get("algorithm_breakthrough") is False
        and source.get("resource_gate_authorized") is False,
        "claim boundary changed",
    )

    summary = {
        "schema_version": "oerf-public-exact-k1-dual-view-ridge-v142.4.1",
        "updated": "2026-08-14",
        "execution_status": EXECUTION_STATUS,
        "scientific_decision": SCIENTIFIC_DECISION,
        "plain_verdict_zh": (
            "共享线性方向 ridge 在两个独立几何特征视图下给出几乎相同的结果，"
            "但都只通过 1/3700 个单元、0/5 条完整轨迹，因此按结果前合同关闭。"
        ),
        "plain_verdict_en": (
            "The shared linear direction ridge gives nearly identical results under "
            "two independently rebuilt geometry-feature views, but both pass only "
            "1/3,700 cells and 0/5 complete trajectories, so the preregistered branch closes."
        ),
        "evaluation": {
            "sample_count": 3700,
            "trajectory_count": 5,
            "trajectory_order": [
                "p=14kw_size=05",
                "p=22kw_size=03",
                "p=33kw_size=01",
                "p=45kw_size=05",
                "p=58kw_size=03",
            ],
            "reference": "Zero-CGLS K4",
            "reference_exact_calls_per_sample": {"A": 4, "AT": 4},
            "candidate_exact_calls_per_sample": {"A": 3, "AT": 3},
            "metrics": source["metrics"],
            "per_cell_ratio_limit": 1.05,
            "trajectory_p90_ratio_limit": 1.02,
            "trajectory_worst_ratio_limit": 1.05,
        },
        "dual_view_reproducibility": {
            "maximum_per_cell_metric_ratio_difference": float(
                source["maximum_metric_ratio_difference_between_views"]
            ),
            "frozen_limit": float(source["metric_stability_limit"]),
            "pass": bool(source["metric_stability_pass"]),
            "exact_k1_dual_absolute_difference": float(
                source["prediction_stage_maximum_diagnostics"][
                    "exact_k1_dual_absolute"
                ]
            ),
            "exact_k1_residual_absolute_difference": float(
                source["prediction_stage_maximum_diagnostics"][
                    "exact_k1_residual_absolute"
                ]
            ),
            "formal_prediction_recompute_absolute_difference": float(
                source["prediction_stage_maximum_diagnostics"][
                    "formal_prediction_recompute_absolute"
                ]
            ),
            "cross_view_dual_relative_l2_difference": float(
                source["prediction_stage_maximum_diagnostics"][
                    "cross_view_dual_relative_l2_diagnostic"
                ]
            ),
        },
        "arm_results": {
            name: _arm(source["arm_results"][name]) for name in ARMS
        },
        "controls": {
            "passing_same_or_lower_call_controls": source[
                "passing_same_or_lower_call_controls"
            ],
            "dominating_same_or_lower_call_controls": source[
                "dominating_same_or_lower_call_controls"
            ],
            "note_zh": (
                "没有同价或更便宜对照通过完整 matched-accuracy 门；scaled_bp_k2 "
                "虽也失败，但在冻结支配判据下仍优于当前线性预测器。"
            ),
            "note_en": (
                "No same-or-lower-cost control passes the complete matched-accuracy gate. "
                "Scaled BP K2 also fails, yet still dominates the current linear predictor "
                "under the frozen dominance rule."
            ),
        },
        "integrity": {
            "checks_passed": sum(bool(value) for value in checks.values()),
            "checks_total": len(checks),
            "prediction_sealed_before_truth": bool(
                checks["independent_prediction_before_truth"]
            ),
            "both_views_physically_replayed": bool(
                checks["both_views_physically_replayed"]
            ),
            "first_full_replay_report_inconclusive_due_to_serialization_only": True,
            "partial_arrays_reused": False,
            "shared_frozen_physics_kernels": bool(
                source["shared_frozen_physics_kernels"]
            ),
            "end_to_end_physics_independence_proven": bool(
                source["end_to_end_physics_independence_proven"]
            ),
        },
        "route_action": {
            "shared_linear_ridge_closed": True,
            "post_result_target_switch_allowed": False,
            "lambda_retuning_allowed": False,
            "larger_model_rescue_allowed": False,
            "resource_gate_authorized": False,
            "external_gate_authorized": False,
            "next_question_zh": (
                "固定 teacher 的容量存在，但其系数不能被当前部署可见特征做跨轨迹线性预测。"
                "下一步只能另冻物理上不同、可证伪的表示或安全门，不能在本分支调参挽救。"
            ),
            "next_question_en": (
                "The fixed teacher has mechanism capacity, but its coefficients are not "
                "cross-trajectory linearly predictable from the current deployment-visible "
                "features. Any next step must preregister a physically different, falsifiable "
                "representation or safety gate rather than retune this branch."
            ),
        },
        "claim_boundary": {
            "fixed_teacher_mechanism_capacity_proven": True,
            "deployable_linear_predictor_proven": False,
            "matched_accuracy_call_reduction_proven": False,
            "algorithm_breakthrough": False,
            "paper_success": False,
            "external_generalization": False,
            "resource_speedup": False,
            "curved_ray_validated": False,
            "real_bost": False,
        },
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args.validation_report, args.output)
    print(
        json.dumps(
            {
                "scientific_decision": result["scientific_decision"],
                "sample_count": result["evaluation"]["sample_count"],
                "algorithm_breakthrough": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
