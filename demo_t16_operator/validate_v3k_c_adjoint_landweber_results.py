#!/usr/bin/env python3
"""Independently validate the committed v3k-C numerical-gate artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "v3k_c_adjoint_landweber_gate.json"
EXPECTED_STATUS = "ADJOINT_DIRECTION_VALID_FIXED_LANDWEBER_REQUIRED_BASELINE"
METHODS = (
    "locked_fno_raw",
    "feasible_fno",
    "landweber_standard",
    "landweber_jacobi",
)
COMPARISONS = (
    ("feasible_fno", "locked_fno_raw"),
    ("landweber_standard", "feasible_fno"),
    ("landweber_jacobi", "feasible_fno"),
    ("landweber_standard", "landweber_jacobi"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--result-dir", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(actual: float, expected: float, message: str) -> None:
    if not np.isclose(float(actual), float(expected), atol=1e-10, rtol=1e-9):
        raise AssertionError(f"{message}: {actual} != {expected}")


def bootstrap_interval(
    values: np.ndarray, seed: int, replicates: int
) -> tuple[float, float]:
    rng = np.random.default_rng(int(seed))
    sampled = rng.integers(0, len(values), size=(int(replicates), len(values)))
    estimates = values[sampled].mean(axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def validate_checksums(result_dir: Path) -> None:
    manifest = result_dir / "v3k_c_adjoint_landweber_checksums.sha256"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    require(len(lines) == 11, "checksum manifest must cover eleven public assets")
    for line in lines:
        expected, name = line.split("  ", 1)
        path = result_dir / name
        require(path.is_file(), f"missing checksum target: {name}")
        require(sha256(path) == expected, f"checksum mismatch: {name}")


def collapsed_metrics(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str, int], dict[str, float]]:
    metrics = (
        "field_rel_l2",
        "audit_reprojection_rel_l2",
        "used_noisy_reprojection_rel_l2",
        "used_clean_reprojection_rel_l2",
        "correction_vs_feasible_relative_l2",
        "outside_support_relative_l2",
        "negative_voxel_fraction",
    )
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["source_split"], int(row["source_index"]))].append(
            row
        )
    return {
        key: {
            metric: float(np.mean([float(row[metric]) for row in subset]))
            for metric in metrics
        }
        for key, subset in grouped.items()
    }


def validate_selection(
    screen: list[dict[str, str]], commit: dict, config: dict
) -> None:
    expected_rows = (
        len(config["methods"])
        * len(config["numerical_protocol"]["step_fractions"])
        * len(config["numerical_protocol"]["iteration_counts"])
    )
    require(len(screen) == expected_rows == 112, "unexpected validation-screen rows")
    require(commit["selection_split"] == "val", "selection must use val only")
    require(commit["audit_camera_used_for_selection"] is False, "audit leaked")
    require(
        commit["audit_or_reprojection_metrics_computed_by_selection_function"]
        is False,
        "selection function computed a sealed diagnostic",
    )
    require(
        commit["test_field_or_metric_rows_present_at_commit"] is False,
        "test rows leaked into selection commit",
    )
    require(
        all(
            "audit" not in key and "reprojection" not in key
            for key in screen[0]
        ),
        "sealed diagnostic column leaked into validation screen",
    )
    for method in config["methods"]:
        candidates = [row for row in screen if row["method"] == method]
        winner = min(
            candidates,
            key=lambda row: (
                float(row["source_field_mean_rel_l2"]),
                int(row["iterations"]),
                float(row["step_fraction"]),
            ),
        )
        selected = commit["selected"][method]
        close(
            float(winner["source_field_mean_rel_l2"]),
            float(selected["source_field_mean_rel_l2"]),
            f"{method} selection metric",
        )
        require(int(winner["iterations"]) == int(selected["iterations"]), "iteration")
        close(winner["step_fraction"], selected["step_fraction"], "step fraction")
        require(float(selected["step_fraction"]) < 2.0, "unstable normalized step")


def validate_summaries(
    samples: list[dict[str, str]], summaries: list[dict[str, str]]
) -> dict[tuple[str, str, int], dict[str, float]]:
    collapsed = collapsed_metrics(samples)
    splits = sorted({row["source_split"] for row in samples})
    for row in summaries:
        split = row["source_split"]
        method = row["method"]
        sources = sorted(
            source
            for candidate, candidate_split, source in collapsed
            if candidate == method and candidate_split == split
        )
        values = np.asarray(
            [collapsed[(method, split, source)]["field_rel_l2"] for source in sources]
        )
        close(np.mean(values), row["mean_field_rel_l2"], "summary mean")
        close(np.median(values), row["median_field_rel_l2"], "summary median")
        close(np.quantile(values, 0.9), row["p90_field_rel_l2"], "summary p90")
    require(len(summaries) == len(splits) * len(METHODS) == 20, "summary rows")
    return collapsed


def validate_pairwise(
    samples: list[dict[str, str]],
    pairwise: list[dict[str, str]],
    collapsed: dict[tuple[str, str, int], dict[str, float]],
    config: dict,
) -> None:
    splits = sorted({row["source_split"] for row in samples})
    require(len(pairwise) == len(splits) * len(COMPARISONS) == 20, "pairwise rows")
    gate = config["gate"]
    for row in pairwise:
        split = row["source_split"]
        candidate = row["candidate"]
        comparator = row["comparator"]
        comparison_index = COMPARISONS.index((candidate, comparator))
        sources = sorted(
            {
                int(sample["source_index"])
                for sample in samples
                if sample["source_split"] == split
            }
        )
        field_gains = np.asarray(
            [
                100.0
                * (
                    collapsed[(comparator, split, source)]["field_rel_l2"]
                    - collapsed[(candidate, split, source)]["field_rel_l2"]
                )
                / max(collapsed[(comparator, split, source)]["field_rel_l2"], 1e-12)
                for source in sources
            ]
        )
        audit_gains = np.asarray(
            [
                100.0
                * (
                    collapsed[(comparator, split, source)][
                        "audit_reprojection_rel_l2"
                    ]
                    - collapsed[(candidate, split, source)][
                        "audit_reprojection_rel_l2"
                    ]
                )
                / max(
                    collapsed[(comparator, split, source)][
                        "audit_reprojection_rel_l2"
                    ],
                    1e-12,
                )
                for source in sources
            ]
        )
        seed = int(gate["bootstrap_seed"]) + 10 * splits.index(split) + comparison_index
        field_ci = bootstrap_interval(field_gains, seed, gate["bootstrap_replicates"])
        audit_ci = bootstrap_interval(
            audit_gains, seed + 1000, gate["bootstrap_replicates"]
        )
        close(np.mean(field_gains), row["mean_field_gain_pct"], "field gain")
        close(field_ci[0], row["field_cluster_ci95_low_pct"], "field CI low")
        close(field_ci[1], row["field_cluster_ci95_high_pct"], "field CI high")
        close(np.median(field_gains), row["median_field_gain_pct"], "field median")
        close(np.quantile(field_gains, 0.1), row["p10_field_gain_pct"], "field p10")
        close(np.mean(field_gains > 0), row["positive_field_fraction"], "coverage")
        close(np.mean(field_gains < -1), row["harm_rate_gt_1pct"], "harm rate")
        close(np.mean(audit_gains), row["mean_audit_gain_pct"], "audit gain")
        close(audit_ci[0], row["audit_cluster_ci95_low_pct"], "audit CI low")
        close(audit_ci[1], row["audit_cluster_ci95_high_pct"], "audit CI high")


def validate_layouts(
    samples: list[dict[str, str]], layouts: list[dict[str, str]], candidate: str
) -> None:
    indexed = {
        (
            row["method"],
            row["source_split"],
            row["geometry_id"],
            int(row["source_index"]),
        ): row
        for row in samples
    }
    for row in layouts:
        split = row["source_split"]
        geometry = row["geometry_id"]
        sources = sorted(
            {
                int(sample["source_index"])
                for sample in samples
                if sample["method"] == candidate
                and sample["source_split"] == split
                and sample["geometry_id"] == geometry
            }
        )
        field = []
        audit = []
        for source in sources:
            selected = indexed[(candidate, split, geometry, source)]
            control = indexed[("feasible_fno", split, geometry, source)]
            field.append(
                100.0
                * (float(control["field_rel_l2"]) - float(selected["field_rel_l2"]))
                / max(float(control["field_rel_l2"]), 1e-12)
            )
            audit.append(
                100.0
                * (
                    float(control["audit_reprojection_rel_l2"])
                    - float(selected["audit_reprojection_rel_l2"])
                )
                / max(float(control["audit_reprojection_rel_l2"]), 1e-12)
            )
        require(len(sources) == int(row["source_field_count"]), "layout source count")
        close(np.mean(field), row["mean_field_gain_pct"], "layout field mean")
        close(np.min(field), row["minimum_field_gain_pct"], "layout field minimum")
        close(np.mean(np.asarray(field) > 0), row["positive_field_fraction"], "layout coverage")
        close(np.mean(np.asarray(field) < -1), row["harm_rate_gt_1pct"], "layout harm")
        close(np.mean(audit), row["mean_audit_gain_pct"], "layout audit mean")


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    result_dir = args.result_dir or ROOT / "results" / str(config["output_dir"])
    dashboard = read_json(result_dir / "v3k_c_adjoint_landweber_dashboard.json")
    report = read_json(result_dir / "v3k_c_adjoint_landweber_report.json")
    commit = read_json(result_dir / "v3k_c_selection_commit.json")
    checks = read_csv(result_dir / "v3k_c_adjoint_checks.csv")
    screen = read_csv(result_dir / "v3k_c_validation_screen.csv")
    manifest = read_csv(result_dir / "v3k_c_pair_manifest.csv")
    samples = read_csv(result_dir / "v3k_c_sample_metrics.csv")
    summaries = read_csv(result_dir / "v3k_c_split_summary.csv")
    pairwise = read_csv(result_dir / "v3k_c_pairwise_summary.csv")
    layouts = read_csv(result_dir / "v3k_c_layout_summary.csv")
    validate_checksums(result_dir)

    require(len(checks) == 28, "expected one numerical check per geometry")
    require(len(manifest) == 672, "unexpected pair-manifest rows")
    require(len(samples) == 2688, "unexpected sample-metric rows")
    require(not list(result_dir.glob("*.pt")), "public result contains checkpoint")
    require(not list(result_dir.glob("*.npz")), "public result contains private array")
    for path in result_dir.iterdir():
        if path.suffix in {".csv", ".json", ".sha256"}:
            text = path.read_text(encoding="utf-8")
            require("/Users/" not in text, f"absolute private path leaked in {path.name}")

    validate_selection(screen, commit, config)
    collapsed = validate_summaries(samples, summaries)
    validate_pairwise(samples, pairwise, collapsed, config)

    require(dashboard["scientific_status"] == EXPECTED_STATUS, "status mismatch")
    require(report["status"] == EXPECTED_STATUS, "report status mismatch")
    require(dashboard["zero_learning_gate_pass"] is True, "numerical gate failed")
    require(all(dashboard["gate_checks"].values()), "one or more gate checks failed")
    require(
        max(float(row["adjoint_inner_product_relative_error"]) for row in checks)
        <= config["gate"]["maximum_adjoint_relative_error"],
        "adjoint identity exceeded tolerance",
    )
    require(
        max(float(row["finite_difference_gradient_relative_error"]) for row in checks)
        <= config["gate"]["maximum_gradient_relative_error"],
        "finite-difference check exceeded tolerance",
    )
    require(all(int(row["audit_camera_active"]) == 0 for row in checks), "audit leak")
    require(report["provenance"]["base_checkpoint_drift"] == 0, "base drift")
    require(
        report["provenance"]["base_checkpoint_sha256_before"]
        == report["provenance"]["base_checkpoint_sha256_after"],
        "checkpoint hash changed",
    )
    require(
        dashboard["next_decision"][
            "learned_scalar_development_prototype_authorized"
        ]
        is True,
        "bounded scalar prototype should be authorized",
    )
    require(
        dashboard["next_decision"][
            "learned_scalar_confirmatory_training_authorized"
        ]
        is False,
        "confirmatory scalar training was prematurely authorized",
    )
    require(
        dashboard["next_decision"]["learned_voxel_preconditioner_authorized"] is False,
        "voxel learning was prematurely authorized",
    )
    require(
        dashboard["next_decision"]["superiority_training_authorized"] is False,
        "superiority was prematurely authorized",
    )
    require(config["claims_boundary"]["superiority_tested"] is False, "claim leak")
    require(config["claims_boundary"]["blind_final_opened"] is False, "blind leak")
    for method in ("feasible_fno", "landweber_standard", "landweber_jacobi"):
        method_rows = [row for row in samples if row["method"] == method]
        require(
            max(float(row["outside_support_relative_l2"]) for row in method_rows) < 1e-12,
            f"{method} escaped known support",
        )
        require(
            max(float(row["negative_voxel_fraction"]) for row in method_rows) == 0.0,
            f"{method} contains negative voxels",
        )

    preferred = dashboard["preferred_numerical_method"]
    validate_layouts(samples, layouts, f"landweber_{preferred}")
    worst = min(layouts, key=lambda row: float(row["mean_field_gain_pct"]))
    require(
        dashboard["worst_layout"]["source_split"] == worst["source_split"]
        and dashboard["worst_layout"]["geometry_id"] == worst["geometry_id"],
        "worst-layout record mismatch",
    )
    selected = commit["selected"][preferred]
    print(
        json.dumps(
            {
                "status": "PASS",
                "scientific_result": dashboard["scientific_status"],
                "checked_geometries": len(checks),
                "validation_grid_rows": len(screen),
                "pair_manifest_rows": len(manifest),
                "sample_metric_rows": len(samples),
                "preferred_method": preferred,
                "selected_step_fraction": selected["step_fraction"],
                "selected_iterations": selected["iterations"],
                "layout_summary_rows": len(layouts),
                "public_checksum_targets": 11,
                "base_checkpoint_drift": report["provenance"][
                    "base_checkpoint_drift"
                ],
                "superiority_authorized": dashboard["next_decision"][
                    "superiority_training_authorized"
                ],
                "confirmatory_scalar_authorized": dashboard["next_decision"][
                    "learned_scalar_confirmatory_training_authorized"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
