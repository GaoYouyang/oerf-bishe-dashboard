#!/usr/bin/env python3
"""Select and check bounded observable PCGLS stopping-rule families."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "psu-b0-covariance-stopping-rule-audit-report-1.0"
STATUS = "POSTOPEN_OBSERVABLE_STOPPING_RULE_AUDIT_COMPLETE"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    numeric = {
        "replicate",
        "sample_index",
        "stage",
        "relative_whitened_residual_objective",
        "residual_objective_reduction_from_previous_stage",
        "pcgls_alpha",
        "previous_pcgls_beta",
        "relative_volume_update",
        "volume_gradient_to_field_norm",
        "field_gain_vs_component_s3_k4_percent",
        "gradient_gain_vs_component_s3_k4_percent",
        "front_gain_vs_component_s3_k4",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in numeric:
            row[key] = (
                int(row[key])
                if key in {"replicate", "sample_index", "stage"}
                else float(row[key])
            )
    return rows


def _group_trajectories(
    rows: list[dict[str, Any]],
) -> dict[tuple[int, int], dict[int, dict[str, Any]]]:
    grouped: dict[tuple[int, int], dict[int, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (int(row["replicate"]), int(row["sample_index"])),
            {},
        )[int(row["stage"])] = row
    if any(set(stages) != {2, 3, 4, 5} for stages in grouped.values()):
        raise ValueError("each field must contain stages 2, 3, 4, and 5")
    return grouped


def choose_one_threshold(
    stages: dict[int, dict[str, Any]],
    *,
    feature: str,
    threshold: float,
    direction: str,
) -> dict[str, Any]:
    value = float(stages[4][feature])
    condition = (
        value <= float(threshold)
        if direction == "le"
        else value >= float(threshold)
    )
    return stages[5] if condition else stages[4]


def choose_rollback_continue(
    stages: dict[int, dict[str, Any]],
    *,
    beta_minimum: float,
    rollback_update_minimum: float,
    extension_reduction_maximum: float,
) -> dict[str, Any]:
    stage_four = stages[4]
    if float(stage_four["previous_pcgls_beta"]) >= float(beta_minimum):
        if float(stage_four["relative_volume_update"]) >= float(
            rollback_update_minimum
        ):
            return stages[3]
        return stages[5]
    if float(
        stage_four["residual_objective_reduction_from_previous_stage"]
    ) <= float(extension_reduction_maximum):
        return stages[5]
    return stage_four


def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
    field = np.asarray(
        [
            float(row["field_gain_vs_component_s3_k4_percent"])
            for row in selected
        ]
    )
    stages = np.asarray([int(row["stage"]) for row in selected])
    counts = {
        str(stage): int(np.sum(stages == stage))
        for stage in (3, 4, 5)
    }
    return {
        "field_count": len(selected),
        "mean_field_gain_percent": float(np.mean(field)),
        "field_gain_p10_percent": float(np.quantile(field, 0.1)),
        "field_harm_over_one_percent_rate": float(
            np.mean(field < -1.0)
        ),
        "worst_field_gain_percent": float(np.min(field)),
        "mean_gradient_gain_percent": float(
            np.mean(
                [
                    float(
                        row[
                            "gradient_gain_vs_component_s3_k4_percent"
                        ]
                    )
                    for row in selected
                ]
            )
        ),
        "mean_front_f1_gain": float(
            np.mean(
                [
                    float(row["front_gain_vs_component_s3_k4"])
                    for row in selected
                ]
            )
        ),
        "mean_stage": float(np.mean(stages)),
        "stage_counts": counts,
    }


def gate_checks(
    summary: dict[str, Any],
    *,
    gates: dict[str, Any],
) -> dict[str, bool]:
    return {
        "mean_field_gain": float(summary["mean_field_gain_percent"])
        >= float(gates["mean_field_gain_percent_minimum"]),
        "field_p10": float(summary["field_gain_p10_percent"])
        >= float(gates["field_gain_p10_percent_minimum"]),
        "field_harm": float(
            summary["field_harm_over_one_percent_rate"]
        )
        <= float(
            gates["field_harm_over_one_percent_rate_maximum"]
        ),
        "worst_field": float(summary["worst_field_gain_percent"])
        >= float(gates["worst_field_gain_percent_minimum"]),
        "mean_gradient_gain": float(
            summary["mean_gradient_gain_percent"]
        )
        >= float(gates["mean_gradient_gain_percent_minimum"]),
        "mean_front_gain": float(summary["mean_front_f1_gain"])
        >= float(gates["mean_front_f1_gain_minimum"]),
        "mean_stage": float(summary["mean_stage"])
        <= float(gates["mean_stage_maximum"]),
    }


def _rule_grid(
    config: dict[str, Any],
    *,
    selection: dict[tuple[int, int], dict[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rules = []
    quantiles = [
        float(value) for value in config["one_threshold_quantiles"]
    ]
    for feature in config["one_threshold_features"]:
        values = np.asarray(
            [float(stages[4][feature]) for stages in selection.values()]
        )
        for quantile in quantiles:
            threshold = float(np.quantile(values, quantile))
            for direction in ("le", "ge"):
                rules.append(
                    {
                        "rule_id": (
                            f"one_{feature}_{direction}_q"
                            f"{quantile:g}"
                        ),
                        "family": "one_threshold_stage4_or_5",
                        "feature": feature,
                        "direction": direction,
                        "threshold": threshold,
                    }
                )
    grid = config["rollback_continue_grid"]
    for beta in grid["previous_pcgls_beta_minimum"]:
        for update in grid[
            "relative_volume_update_rollback_minimum"
        ]:
            for reduction in grid[
                "stage_five_extension_reduction_maximum"
            ]:
                rules.append(
                    {
                        "rule_id": (
                            f"rollback_b{float(beta):g}_"
                            f"u{float(update):g}_d{float(reduction):g}"
                        ),
                        "family": "rollback_or_continue_from_stage4",
                        "beta_minimum": float(beta),
                        "rollback_update_minimum": float(update),
                        "extension_reduction_maximum": float(reduction),
                    }
                )
    return rules


def _apply_rule(
    trajectories: dict[tuple[int, int], dict[int, dict[str, Any]]],
    rule: dict[str, Any],
) -> list[dict[str, Any]]:
    if rule["family"] == "one_threshold_stage4_or_5":
        return [
            choose_one_threshold(
                stages,
                feature=str(rule["feature"]),
                threshold=float(rule["threshold"]),
                direction=str(rule["direction"]),
            )
            for stages in trajectories.values()
        ]
    if rule["family"] == "rollback_or_continue_from_stage4":
        return [
            choose_rollback_continue(
                stages,
                beta_minimum=float(rule["beta_minimum"]),
                rollback_update_minimum=float(
                    rule["rollback_update_minimum"]
                ),
                extension_reduction_maximum=float(
                    rule["extension_reduction_maximum"]
                ),
            )
            for stages in trajectories.values()
        ]
    raise ValueError(f"unknown rule family: {rule['family']}")


def run_audit(
    *,
    root: Path,
    config_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    feature_report_path = root / str(config["source_feature_report"])
    trajectory_path = root / str(config["source_trajectory_rows"])
    rows = _read_rows(trajectory_path)
    grouped = _group_trajectories(rows)
    selection = {
        key: value
        for key, value in grouped.items()
        if value[4]["split"] == "selection"
    }
    diagnostic = {
        key: value
        for key, value in grouped.items()
        if value[4]["split"] == "opened_diagnostic_check"
    }
    rules = _rule_grid(config, selection=selection)
    evaluated = []
    passing_selection = []
    for rule in rules:
        selection_summary = summarize(_apply_rule(selection, rule))
        checks = gate_checks(
            selection_summary,
            gates=config["selection_gates"],
        )
        row = {
            **rule,
            "selection": {
                **selection_summary,
                "gate_checks": checks,
                "all_gates_pass": bool(all(checks.values())),
            },
        }
        evaluated.append(row)
        if all(checks.values()):
            passing_selection.append(row)
    selected = None
    if passing_selection:
        selected = min(
            passing_selection,
            key=lambda row: (
                -float(row["selection"]["mean_field_gain_percent"]),
                -float(row["selection"]["worst_field_gain_percent"]),
                float(row["selection"]["mean_stage"]),
                str(row["rule_id"]),
            ),
        )
    diagnostic_summary = None
    diagnostic_pass = False
    action_rows = []
    if selected is not None:
        chosen = _apply_rule(diagnostic, selected)
        diagnostic_summary = summarize(chosen)
        checks = gate_checks(
            diagnostic_summary,
            gates=config["selection_gates"],
        )
        diagnostic_summary["gate_checks"] = checks
        diagnostic_summary["all_gates_pass"] = bool(
            all(checks.values())
        )
        diagnostic_pass = bool(all(checks.values()))
        for row in _apply_rule(selection, selected) + chosen:
            action_rows.append(
                {
                    "replicate": row["replicate"],
                    "split": row["split"],
                    "sample_index": row["sample_index"],
                    "reaction_family": row["reaction_family"],
                    "selected_stage": row["stage"],
                    "field_gain_vs_component_s3_k4_percent": row[
                        "field_gain_vs_component_s3_k4_percent"
                    ],
                    "gradient_gain_vs_component_s3_k4_percent": row[
                        "gradient_gain_vs_component_s3_k4_percent"
                    ],
                    "front_gain_vs_component_s3_k4": row[
                        "front_gain_vs_component_s3_k4"
                    ],
                }
            )
    selection_family_counts = {
        family: sum(
            row["family"] == family
            and row["selection"]["all_gates_pass"]
            for row in evaluated
        )
        for family in {
            str(row["family"]) for row in evaluated
        }
    }
    authorize = bool(selected is not None and diagnostic_pass)
    report = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "evidence_scope": config["evidence_scope"],
        "config_sha256": _sha256(config_path),
        "source_feature_report_sha256": _sha256(feature_report_path),
        "source_trajectory_rows_sha256": _sha256(trajectory_path),
        "configuration": config,
        "rule_count": len(rules),
        "selection_passing_rule_count_by_family": selection_family_counts,
        "constant_stage_context": {
            str(stage): {
                "selection": summarize(
                    [value[stage] for value in selection.values()]
                ),
                "opened_diagnostic_check": summarize(
                    [value[stage] for value in diagnostic.values()]
                ),
            }
            for stage in (3, 4, 5)
        },
        "decision": {
            "selected_rule": selected,
            "opened_diagnostic_check": diagnostic_summary,
            "fresh_preregistration_authorized": authorize,
            "verdict": (
                "POSTOPEN_RULE_PASSED_BOTH_HALVES_FRESH_PREREGISTRATION_ONLY"
                if authorize
                else "OBSERVABLE_POOLED_STOPPING_RULE_NO_GO"
            ),
        },
        "claim_boundary": config["claim_boundary"],
    }
    return report, action_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = (
        args.config
        if args.config.is_absolute()
        else root / args.config
    )
    report, action_rows = run_audit(
        root=root,
        config_path=config_path,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    action_path = args.output_dir / "selected_action_rows.csv"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(action_path, action_rows)
    print(
        json.dumps(
            {
                "status": report["status"],
                "verdict": report["decision"]["verdict"],
                "selected_rule": (
                    None
                    if report["decision"]["selected_rule"] is None
                    else report["decision"]["selected_rule"]["rule_id"]
                ),
                "fresh_preregistration_authorized": report["decision"][
                    "fresh_preregistration_authorized"
                ],
                "report": str(report_path),
                "selected_action_rows": (
                    str(action_path) if action_rows else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
