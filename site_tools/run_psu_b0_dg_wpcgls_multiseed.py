#!/usr/bin/env python3
"""Run a fresh replicate-clustered detector-whitening development pilot."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from site_tools.run_psu_b0_dg_wpcgls_smoke import (
    _load_json,
    run_smoke,
)


SCHEMA = "psu-b0-dg-wpcgls-multiseed-report-1.0"
STATUS = "FRESH_MULTI_SEED_DEVELOPMENT_PILOT_COMPLETE"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _paired_gain_rows(
    rows: list[dict[str, Any]],
    *,
    baseline_method: str,
) -> list[dict[str, Any]]:
    baseline = {
        (
            int(row["replicate"]),
            str(row["noise_family"]),
            int(row["sample_index"]),
        ): row
        for row in rows
        if row["method"] == baseline_method
    }
    output = []
    for row in rows:
        key = (
            int(row["replicate"]),
            str(row["noise_family"]),
            int(row["sample_index"]),
        )
        reference = baseline[key]
        reference_field = float(reference["field_relative_l2"])
        output.append(
            {
                **row,
                "field_gain_vs_baseline_percent": float(
                    100.0
                    * (
                        reference_field
                        - float(row["field_relative_l2"])
                    )
                    / max(reference_field, 1e-12)
                ),
                "gradient_gain_vs_baseline_percent": float(
                    100.0
                    * (
                        float(reference["gradient_relative_l2"])
                        - float(row["gradient_relative_l2"])
                    )
                    / max(
                        float(reference["gradient_relative_l2"]),
                        1e-12,
                    )
                ),
                "front_f1_gain_vs_baseline": float(
                    float(row["front_top10_f1"])
                    - float(reference["front_top10_f1"])
                ),
            }
        )
    return output


def _mean_ci95(values: np.ndarray) -> tuple[float, float, float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or len(vector) < 2:
        raise ValueError("confidence interval needs at least two values")
    mean = float(np.mean(vector))
    standard_error = float(
        np.std(vector, ddof=1) / np.sqrt(len(vector))
    )
    critical = float(student_t.ppf(0.975, len(vector) - 1))
    half_width = critical * standard_error
    return mean, mean - half_width, mean + half_width


def _aggregate(
    gain_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    replicate_groups: dict[
        tuple[int, str, str],
        list[dict[str, Any]],
    ] = {}
    for row in gain_rows:
        replicate_groups.setdefault(
            (
                int(row["replicate"]),
                str(row["noise_family"]),
                str(row["method"]),
            ),
            [],
        ).append(row)
    replicate_rows = []
    for (replicate, noise_family, method), selected in sorted(
        replicate_groups.items()
    ):
        field_gain = np.asarray(
            [
                float(row["field_gain_vs_baseline_percent"])
                for row in selected
            ],
            dtype=np.float64,
        )
        gradient_gain = np.asarray(
            [
                float(row["gradient_gain_vs_baseline_percent"])
                for row in selected
            ],
            dtype=np.float64,
        )
        front_gain = np.asarray(
            [
                float(row["front_f1_gain_vs_baseline"])
                for row in selected
            ],
            dtype=np.float64,
        )
        replicate_rows.append(
            {
                "replicate": replicate,
                "noise_family": noise_family,
                "method": method,
                "field_gain_percent_mean": float(np.mean(field_gain)),
                "field_gain_percent_median": float(np.median(field_gain)),
                "gradient_gain_percent_mean": float(
                    np.mean(gradient_gain)
                ),
                "front_f1_gain_mean": float(np.mean(front_gain)),
                "field_harm_over_one_percent_rate": float(
                    np.mean(field_gain < -1.0)
                ),
            }
        )

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in replicate_rows:
        groups.setdefault(
            (str(row["noise_family"]), str(row["method"])),
            [],
        ).append(row)
    summaries = []
    for (noise_family, method), selected in sorted(groups.items()):
        replicate_gain = np.asarray(
            [
                float(row["field_gain_percent_mean"])
                for row in selected
            ],
            dtype=np.float64,
        )
        mean, lower, upper = _mean_ci95(replicate_gain)
        field_rows = [
            row
            for row in gain_rows
            if row["noise_family"] == noise_family
            and row["method"] == method
        ]
        field_gain = np.asarray(
            [
                float(row["field_gain_vs_baseline_percent"])
                for row in field_rows
            ],
            dtype=np.float64,
        )
        gradient_gain = np.asarray(
            [
                float(row["gradient_gain_vs_baseline_percent"])
                for row in field_rows
            ],
            dtype=np.float64,
        )
        front_gain = np.asarray(
            [
                float(row["front_f1_gain_vs_baseline"])
                for row in field_rows
            ],
            dtype=np.float64,
        )
        summaries.append(
            {
                "noise_family": noise_family,
                "method": method,
                "replicate_count": len(selected),
                "field_count": len(field_rows),
                "replicate_mean_field_gain_percent": mean,
                "replicate_mean_field_gain_ci95_lower": lower,
                "replicate_mean_field_gain_ci95_upper": upper,
                "field_gain_percent_median": float(
                    np.median(field_gain)
                ),
                "field_gain_percent_p10": float(
                    np.quantile(field_gain, 0.1)
                ),
                "field_harm_over_one_percent_rate": float(
                    np.mean(field_gain < -1.0)
                ),
                "gradient_gain_percent_mean": float(
                    np.mean(gradient_gain)
                ),
                "front_f1_gain_mean": float(np.mean(front_gain)),
            }
        )
    return replicate_rows, summaries


def _family_summaries(
    gain_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in gain_rows:
        groups.setdefault(
            (
                str(row["noise_family"]),
                str(row["method"]),
                str(row["reaction_family"]),
            ),
            [],
        ).append(row)
    output = []
    for (noise_family, method, reaction_family), selected in sorted(
        groups.items()
    ):
        field = np.asarray(
            [
                float(row["field_gain_vs_baseline_percent"])
                for row in selected
            ],
            dtype=np.float64,
        )
        output.append(
            {
                "noise_family": noise_family,
                "method": method,
                "reaction_family": reaction_family,
                "replicate_count": len(selected),
                "field_gain_percent_mean": float(np.mean(field)),
                "field_gain_percent_median": float(np.median(field)),
                "field_gain_percent_p10": float(
                    np.quantile(field, 0.1)
                ),
                "field_harm_over_one_percent_rate": float(
                    np.mean(field < -1.0)
                ),
            }
        )
    return output


def _decision(
    summaries: list[dict[str, Any]],
    *,
    criteria: dict[str, Any],
) -> dict[str, Any]:
    lookup = {
        (str(row["noise_family"]), str(row["method"])): row
        for row in summaries
    }
    heat = lookup[("graph_heat", "dg_covgate")]
    oracle = lookup[("graph_heat", "oracle_covariance")]
    wrong = lookup[("graph_heat", "wrong_graph_assumption")]
    iid = lookup[("component_iid", "dg_covgate")]
    heat_gain = float(heat["replicate_mean_field_gain_percent"])
    oracle_gain = float(oracle["replicate_mean_field_gain_percent"])
    wrong_gain = float(wrong["replicate_mean_field_gain_percent"])
    retention = heat_gain / max(oracle_gain, 1e-12)
    checks = {
        "graph_heat_mean_gain": heat_gain
        >= float(
            criteria[
                "graph_heat_candidate_mean_gain_percent_minimum"
            ]
        ),
        "graph_heat_ci95_lower": float(
            heat["replicate_mean_field_gain_ci95_lower"]
        )
        >= float(
            criteria[
                "graph_heat_candidate_gain_ci95_lower_minimum"
            ]
        ),
        "graph_heat_p10": float(heat["field_gain_percent_p10"])
        >= float(
            criteria[
                "graph_heat_candidate_p10_field_gain_percent_minimum"
            ]
        ),
        "graph_heat_harm_rate": float(
            heat["field_harm_over_one_percent_rate"]
        )
        <= float(
            criteria[
                "graph_heat_candidate_harm_over_one_percent_rate_maximum"
            ]
        ),
        "graph_heat_front_gain": float(heat["front_f1_gain_mean"])
        >= float(
            criteria[
                "graph_heat_candidate_front_f1_gain_mean_minimum"
            ]
        ),
        "oracle_gain_retention": retention
        >= float(
            criteria["graph_heat_oracle_gain_retention_minimum"]
        ),
        "candidate_beats_wrong_model": heat_gain - wrong_gain
        >= float(
            criteria[
                "graph_heat_candidate_minus_wrong_mean_gain_percent_minimum"
            ]
        ),
        "candidate_harm_not_above_wrong": (
            not bool(
                criteria[
                    "graph_heat_candidate_harm_rate_not_above_wrong"
                ]
            )
            or float(heat["field_harm_over_one_percent_rate"])
            <= float(wrong["field_harm_over_one_percent_rate"])
        ),
        "iid_mean_neutral": abs(
            float(iid["replicate_mean_field_gain_percent"])
        )
        <= float(
            criteria[
                "component_iid_candidate_absolute_mean_gain_percent_maximum"
            ]
        ),
        "iid_harm_rate": float(
            iid["field_harm_over_one_percent_rate"]
        )
        <= float(
            criteria[
                "component_iid_candidate_harm_over_one_percent_rate_maximum"
            ]
        ),
    }
    return {
        "checks": checks,
        "all_checks_pass": bool(all(checks.values())),
        "graph_heat_candidate_mean_gain_percent": heat_gain,
        "graph_heat_candidate_ci95": [
            float(heat["replicate_mean_field_gain_ci95_lower"]),
            float(heat["replicate_mean_field_gain_ci95_upper"]),
        ],
        "graph_heat_oracle_mean_gain_percent": oracle_gain,
        "graph_heat_wrong_model_mean_gain_percent": wrong_gain,
        "oracle_gain_retention": float(retention),
        "verdict": (
            "MULTI_SEED_MECHANISM_REPLICATED_ADVANCE_TO_MODEL_MISMATCH_AND_REAL_REPEAT_GATE"
            if all(checks.values())
            else "MULTI_SEED_GATE_FAILED_NO_ADVANCEMENT_CLAIM"
        ),
    }


def run_multiseed(
    *,
    root: Path,
    config_path: Path,
    view_root: Path,
    device_name: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    config = _load_json(config_path)
    smoke_path = root / str(config["source_smoke_config"])
    smoke_template = _load_json(smoke_path)
    replicate_config = config["replicates"]
    replicate_count = int(replicate_config["count"])
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    replicate_reports: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix="psu_b0_dg_wpcgls_multiseed_"
    ) as temporary:
        temporary_root = Path(temporary)
        for replicate in range(replicate_count):
            derived = copy.deepcopy(smoke_template)
            derived["status"] = "DERIVED_FRESH_MULTI_SEED_REPLICATE"
            derived["covariance_calibration"]["base_seed"] = (
                int(replicate_config["calibration_seed_start"])
                + replicate
                * int(replicate_config["calibration_seed_stride"])
            )
            derived["data"]["field_seed_start"] = (
                int(replicate_config["field_seed_start"])
                + replicate
                * int(replicate_config["field_seed_stride"])
            )
            derived["data"]["flowon_noise_seed"] = (
                int(replicate_config["flowon_noise_seed_start"])
                + replicate
                * int(replicate_config["flowon_noise_seed_stride"])
            )
            derived_path = temporary_root / f"replicate_{replicate:02d}.json"
            derived_path.write_text(
                json.dumps(derived, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            replicate_report, replicate_rows = run_smoke(
                root=root,
                config_path=derived_path,
                view_root=view_root,
                device_name=device_name,
            )
            for row in replicate_rows:
                rows.append({"replicate": replicate, **row})
            replicate_reports.append(
                {
                    "replicate": replicate,
                    "derived_config_sha256": _sha256(derived_path),
                    "calibration_seed": derived[
                        "covariance_calibration"
                    ]["base_seed"],
                    "field_seed_start": derived["data"][
                        "field_seed_start"
                    ],
                    "flowon_noise_seed": derived["data"][
                        "flowon_noise_seed"
                    ],
                    "mechanism_verdict": replicate_report[
                        "mechanism_decision"
                    ]["verdict"],
                    "wall_seconds": replicate_report["runtime"][
                        "wall_seconds"
                    ],
                    "all_solver_calls_match": all(
                        int(row["forward_calls"])
                        == int(smoke_template["solver"]["stages"])
                        and int(row["adjoint_calls"])
                        == int(smoke_template["solver"]["stages"])
                        for row in replicate_rows
                    ),
                }
            )

    baseline_method = str(
        config["primary_comparison"]["baseline"]
    )
    gain_rows = _paired_gain_rows(
        rows,
        baseline_method=baseline_method,
    )
    replicate_rows, summaries = _aggregate(gain_rows)
    family_summaries = _family_summaries(gain_rows)
    decision = _decision(summaries, criteria=config["gates"])
    calls_match = all(
        bool(row["all_solver_calls_match"])
        for row in replicate_reports
    )
    if bool(config["gates"]["all_solver_calls_exactly_match"]):
        decision["checks"]["all_solver_calls_match"] = calls_match
        decision["all_checks_pass"] = bool(
            all(decision["checks"].values())
        )
        if not decision["all_checks_pass"]:
            decision["verdict"] = (
                "MULTI_SEED_GATE_FAILED_NO_ADVANCEMENT_CLAIM"
            )
    report = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "evidence_scope": config["evidence_scope"],
        "config_sha256": _sha256(config_path),
        "source_smoke_config_sha256": _sha256(smoke_path),
        "source_smoke_code_commit": config[
            "source_smoke_code_commit"
        ],
        "replicates": replicate_reports,
        "primary_comparison": config["primary_comparison"],
        "summaries": summaries,
        "family_summaries": family_summaries,
        "decision": decision,
        "runtime": {
            "device": device_name,
            "wall_seconds": float(time.perf_counter() - started),
            "replicate_wall_seconds_sum": float(
                sum(
                    float(row["wall_seconds"])
                    for row in replicate_reports
                )
            ),
        },
        "claim_boundary": config["claim_boundary"],
    }
    return report, gain_rows, replicate_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = (
        args.config
        if args.config.is_absolute()
        else root / args.config
    )
    report, rows, replicate_rows = run_multiseed(
        root=root,
        config_path=config_path,
        view_root=args.view_root.resolve(),
        device_name=str(args.device),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    metric_path = args.output_dir / "metric_rows.csv"
    replicate_path = args.output_dir / "replicate_summaries.csv"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(metric_path, rows)
    _write_csv(replicate_path, replicate_rows)
    print(
        json.dumps(
            {
                "status": report["status"],
                "verdict": report["decision"]["verdict"],
                "report": str(report_path),
                "metric_rows": str(metric_path),
                "replicate_summaries": str(replicate_path),
                "wall_seconds": report["runtime"]["wall_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
