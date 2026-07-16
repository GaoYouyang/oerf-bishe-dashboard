#!/usr/bin/env python3
"""Independently validate the committed v3k-D strong-control artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "v3k_d_strong_numerical_controls.json"
EXPECTED_STATUS = "STRONG_NUMERICAL_CONTROLS_COMPLETE_BB_AND_FRESH_LOCK_STILL_REQUIRED"
METHODS = (
    "locked_fno_raw",
    "feasible_fno",
    "ridge_raw",
    "feasible_ridge",
    "fno_geometry",
    "fno_global",
    "fno_quadratic",
    "fno_lookup",
    "ridge_geometry",
    "ridge_global",
    "ridge_quadratic",
    "ridge_lookup",
)
PUBLIC_COUNT = 12


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
    manifest = result_dir / "v3k_d_strong_controls_checksums.sha256"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    require(len(lines) == PUBLIC_COUNT, "checksum target count")
    for line in lines:
        expected, name = line.split("  ", 1)
        path = result_dir / name
        require(path.is_file(), f"missing checksum target: {name}")
        require(sha256(path) == expected, f"checksum mismatch: {name}")


def select_screen(rows: list[dict[str, str]]) -> dict[str, str]:
    require(rows, "empty selection cell")
    return min(
        rows,
        key=lambda row: (
            float(row["source_field_mean_rel_l2"]),
            int(row["operator_a_calls_per_sample"])
            + int(row["operator_at_calls_per_sample"]),
            int(row["iterations"]),
            float(row["step_fraction"] or 0.0),
            float(row["ridge_relative"] or 0.0),
        ),
    )


def validate_selected_row(
    actual: dict[str, object], expected: dict[str, str], role: str
) -> None:
    for key in (
        "start",
        "method",
        "spectral_regime",
        "iterations",
        "operator_a_calls_per_sample",
        "operator_at_calls_per_sample",
    ):
        require(str(actual[key]) == str(expected[key]), f"{role} mismatch: {key}")
    for key in (
        "ridge_relative",
        "step_fraction",
        "source_field_mean_rel_l2",
        "source_field_median_rel_l2",
    ):
        if str(expected[key]) != "":
            close(float(actual[key]), float(expected[key]), f"{role} mismatch: {key}")


def validate_selection(
    screen: list[dict[str, str]], commit: dict, config: dict
) -> None:
    protocol = config["selection_protocol"]
    expected = 2 * len(protocol["ridge_relative_grid"])
    expected += 2 * (
        4
        * len(protocol["step_fractions"])
        * len(protocol["iteration_counts"])
        + len(protocol["iteration_counts"])
    )
    require(len(screen) == expected == 474, "validation screen row count")
    require({row["selection_split"] for row in screen} == {"val"}, "non-val row in screen")
    require(
        all(row["selection_uses_audit_or_reprojection"] == "False" for row in screen),
        "selection diagnostic leak flag",
    )
    require(commit["selection_split"] == "val", "selection split")
    require(commit["audit_camera_used_for_selection"] is False, "audit camera leak")
    require(
        commit["audit_or_reprojection_metrics_computed_by_selection_function"] is False,
        "selection computed sealed metrics",
    )
    require(
        commit["test_field_or_metric_rows_present_at_commit"] is False,
        "test row leak in commit",
    )
    require(commit["truth_oracle_is_non_deployable"] is True, "oracle role")

    selected = commit["selected"]
    for method in ("ridge_raw", "feasible_ridge"):
        winner = select_screen([row for row in screen if row["method"] == method])
        validate_selected_row(selected[method], winner, method)
    solver_names = {
        "geometry": "geometry_landweber",
        "global": "global_landweber",
        "quadratic": "quadratic_step",
    }
    for start in ("fno", "ridge"):
        for short, solver in solver_names.items():
            winner = select_screen(
                [
                    row
                    for row in screen
                    if row["start"] == start
                    and row["method"] == solver
                    and row["spectral_regime"] == "all"
                ]
            )
            validate_selected_row(
                selected["methods"][f"{start}_{short}"], winner, f"{start}_{short}"
            )
        lookup = selected["methods"][f"{start}_lookup"]
        for regime in ("low", "high"):
            winner = select_screen(
                [
                    row
                    for row in screen
                    if row["start"] == start
                    and row["method"] == "spectral_lookup"
                    and row["spectral_regime"] == regime
                ]
            )
            validate_selected_row(
                lookup["regime_table"][regime], winner, f"{start} lookup {regime}"
            )

    manifest = commit["spectral_regime_manifest"]
    require(len(manifest) == 28, "spectral geometry manifest count")
    spectral = sorted({float(row["spectral_constant"]) for row in manifest})
    gaps = np.diff(spectral)
    boundary = 0.5 * (spectral[int(np.argmax(gaps))] + spectral[int(np.argmax(gaps)) + 1])
    close(boundary, commit["operator_only_spectral_boundary"], "spectral boundary")
    require({row["spectral_regime"] for row in manifest} == {"low", "high"}, "regimes")
    for row in manifest:
        expected_regime = (
            "low" if float(row["spectral_constant"]) <= boundary else "high"
        )
        require(row["spectral_regime"] == expected_regime, "regime assignment")


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
        grouped[(row["method"], row["source_split"], int(row["source_index"]))].append(row)
    return {
        key: {
            metric: float(np.mean([float(row[metric]) for row in subset]))
            for metric in metrics
        }
        for key, subset in grouped.items()
    }


def validate_validation_scores(
    collapsed: dict[tuple[str, str, int], dict[str, float]], commit: dict
) -> None:
    scores = commit["selected"]["validation_scores"]
    for method in METHODS:
        sources = sorted(
            source
            for candidate, split, source in collapsed
            if candidate == method and split == "val"
        )
        values = np.asarray(
            [collapsed[(method, "val", source)]["field_rel_l2"] for source in sources]
        )
        close(np.mean(values), scores[method]["source_field_mean_rel_l2"], "val score mean")
        close(np.median(values), scores[method]["source_field_median_rel_l2"], "val score median")
        require(len(sources) == int(scores[method]["independent_field_count"]), "field count")


def validate_summaries(
    summaries: list[dict[str, str]],
    collapsed: dict[tuple[str, str, int], dict[str, float]],
) -> None:
    splits = sorted({split for _, split, _ in collapsed})
    require(len(summaries) == len(splits) * len(METHODS) == 60, "summary row count")
    for row in summaries:
        method = row["method"]
        split = row["source_split"]
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


def validate_pairwise(
    samples: list[dict[str, str]],
    pairwise: list[dict[str, str]],
    collapsed: dict[tuple[str, str, int], dict[str, float]],
    config: dict,
) -> None:
    splits = sorted({row["source_split"] for row in samples})
    first_split = splits[0]
    comparisons = [
        (row["candidate"], row["comparator"])
        for row in pairwise
        if row["source_split"] == first_split
    ]
    require(len(pairwise) == len(splits) * len(comparisons), "pairwise row count")
    require(len(comparisons) == len(set(comparisons)), "duplicate comparison")
    for row in pairwise:
        split = row["source_split"]
        candidate = row["candidate"]
        comparator = row["comparator"]
        comparison_index = comparisons.index((candidate, comparator))
        sources = sorted(
            {int(sample["source_index"]) for sample in samples if sample["source_split"] == split}
        )
        field = np.asarray(
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
        audit = np.asarray(
            [
                100.0
                * (
                    collapsed[(comparator, split, source)]["audit_reprojection_rel_l2"]
                    - collapsed[(candidate, split, source)]["audit_reprojection_rel_l2"]
                )
                / max(
                    collapsed[(comparator, split, source)]["audit_reprojection_rel_l2"],
                    1e-12,
                )
                for source in sources
            ]
        )
        seed = int(config["gate"]["bootstrap_seed"]) + 10 * splits.index(split) + comparison_index
        field_ci = bootstrap_interval(field, seed, config["gate"]["bootstrap_replicates"])
        audit_ci = bootstrap_interval(audit, seed + 1000, config["gate"]["bootstrap_replicates"])
        close(np.mean(field), row["mean_field_gain_pct"], "pair field mean")
        close(field_ci[0], row["field_cluster_ci95_low_pct"], "pair field CI low")
        close(field_ci[1], row["field_cluster_ci95_high_pct"], "pair field CI high")
        close(np.mean(field > 0.0), row["positive_field_fraction"], "pair coverage")
        close(np.mean(field < -1.0), row["harm_rate_gt_1pct"], "pair harm")
        close(np.mean(audit), row["mean_audit_gain_pct"], "pair audit mean")
        close(audit_ci[0], row["audit_cluster_ci95_low_pct"], "pair audit CI low")
        close(audit_ci[1], row["audit_cluster_ci95_high_pct"], "pair audit CI high")


def validate_layouts(
    samples: list[dict[str, str]], layouts: list[dict[str, str]], champion: str
) -> None:
    indexed = {
        (row["method"], row["source_split"], row["geometry_id"], int(row["source_index"])): row
        for row in samples
    }
    require(len(layouts) == 20, "layout row count")
    for row in layouts:
        split = row["source_split"]
        geometry = row["geometry_id"]
        sources = sorted(
            {
                int(sample["source_index"])
                for sample in samples
                if sample["method"] == champion
                and sample["source_split"] == split
                and sample["geometry_id"] == geometry
            }
        )
        field = []
        audit = []
        for source in sources:
            candidate = indexed[(champion, split, geometry, source)]
            comparator = indexed[("feasible_fno", split, geometry, source)]
            field.append(
                100.0
                * (float(comparator["field_rel_l2"]) - float(candidate["field_rel_l2"]))
                / max(float(comparator["field_rel_l2"]), 1e-12)
            )
            audit.append(
                100.0
                * (
                    float(comparator["audit_reprojection_rel_l2"])
                    - float(candidate["audit_reprojection_rel_l2"])
                )
                / max(float(comparator["audit_reprojection_rel_l2"]), 1e-12)
            )
        close(np.mean(field), row["mean_field_gain_pct"], "layout field mean")
        close(np.min(field), row["minimum_field_gain_pct"], "layout field minimum")
        close(np.mean(np.asarray(field) > 0), row["positive_field_fraction"], "layout coverage")
        close(np.mean(np.asarray(field) < -1), row["harm_rate_gt_1pct"], "layout harm")
        close(np.mean(audit), row["mean_audit_gain_pct"], "layout audit")


def selected_calls(selection: dict, method: str) -> tuple[int, int]:
    if method not in selection["methods"]:
        return 0, 0
    choice = selection["methods"][method]
    if choice["method"] == "spectral_lookup":
        iteration = max(int(cell["iterations"]) for cell in choice["regime_table"].values())
        return iteration, iteration
    iteration = int(choice["iterations"])
    if choice["method"] == "quadratic_step":
        return 2 * iteration + 1, iteration
    return iteration, iteration


def validate_call_ledger(rows: list[dict[str, str]], selection: dict) -> None:
    splits = {row["source_split"] for row in rows}
    require(len(rows) == len(splits) * len(METHODS) == 60, "call ledger rows")
    for row in rows:
        expected_a, expected_at = selected_calls(selection, row["method"])
        require(int(row["operator_a_calls_per_sample"]) == expected_a, "A call ledger")
        require(int(row["operator_at_calls_per_sample"]) == expected_at, "A^T call ledger")
        require(row["evaluation_metric_forward_calls_excluded"] == "True", "evaluation calls")
        require(float(row["initialization_runtime_seconds"]) >= 0.0, "init runtime")
        require(float(row["refinement_runtime_seconds"]) >= 0.0, "refine runtime")
        if row["method"] not in selection["methods"]:
            close(row["refinement_runtime_seconds"], 0.0, "control refinement runtime")


def validate_frontier(selection: dict) -> None:
    scores = selection["validation_scores"]
    frontier = selection["operator_call_frontier"]
    budgets = sorted({sum(selected_calls(selection, method)) for method in METHODS})
    require([int(row["maximum_operator_calls_per_sample"]) for row in frontier] == budgets, "frontier budgets")
    for row in frontier:
        budget = int(row["maximum_operator_calls_per_sample"])
        eligible = [method for method in METHODS if sum(selected_calls(selection, method)) <= budget]
        winner = min(
            eligible,
            key=lambda method: (
                float(scores[method]["source_field_mean_rel_l2"]),
                sum(selected_calls(selection, method)),
                method,
            ),
        )
        require(row["best_method"] == winner, "frontier winner")
        close(row["source_field_mean_rel_l2"], scores[winner]["source_field_mean_rel_l2"], "frontier score")


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    result_dir = args.result_dir or ROOT / "results" / str(config["output_dir"])
    dashboard = read_json(result_dir / "v3k_d_strong_controls_dashboard.json")
    report = read_json(result_dir / "v3k_d_strong_controls_report.json")
    commit = read_json(result_dir / "v3k_d_selection_commit.json")
    screen = read_csv(result_dir / "v3k_d_validation_screen.csv")
    manifest = read_csv(result_dir / "v3k_d_pair_manifest.csv")
    samples = read_csv(result_dir / "v3k_d_sample_metrics.csv")
    summaries = read_csv(result_dir / "v3k_d_split_summary.csv")
    pairwise = read_csv(result_dir / "v3k_d_pairwise_summary.csv")
    layouts = read_csv(result_dir / "v3k_d_layout_summary.csv")
    quadratic = read_csv(result_dir / "v3k_d_quadratic_step_audit.csv")
    calls = read_csv(result_dir / "v3k_d_operator_call_ledger.csv")
    validate_checksums(result_dir)

    require(len(manifest) == 672, "pair manifest rows")
    require(len(samples) == 8064, "sample metric rows")
    require(len(quadratic) == 10, "quadratic audit rows")
    require(not list(result_dir.glob("*.pt")), "public result contains checkpoint")
    require(not list(result_dir.glob("*.npz")), "public result contains private array")
    for path in result_dir.iterdir():
        if path.suffix in {".csv", ".json", ".sha256"}:
            text = path.read_text(encoding="utf-8")
            require("/Users/" not in text, f"private absolute path leaked in {path.name}")

    validate_selection(screen, commit, config)
    collapsed = collapsed_metrics(samples)
    validate_validation_scores(collapsed, commit)
    validate_summaries(summaries, collapsed)
    validate_pairwise(samples, pairwise, collapsed, config)
    champion = str(commit["selected"]["champion_method"])
    validate_layouts(samples, layouts, champion)
    validate_call_ledger(calls, commit["selected"])
    validate_frontier(commit["selected"])

    scores = commit["selected"]["validation_scores"]
    expected_champion = min(
        METHODS,
        key=lambda method: (
            float(scores[method]["source_field_mean_rel_l2"]),
            sum(selected_calls(commit["selected"], method)),
            method,
        ),
    )
    require(champion == expected_champion == dashboard["champion_method"], "champion")
    require(dashboard["scientific_status"] == EXPECTED_STATUS, "dashboard status")
    require(report["status"] == EXPECTED_STATUS, "report status")
    require(dashboard["strong_numerical_controls_complete"] is True, "control gate")
    require(all(dashboard["gate_checks"].values()), "failed gate check")
    require(report["provenance"]["base_checkpoint_drift"] == 0, "checkpoint drift")
    require(
        report["provenance"]["base_checkpoint_sha256_before"]
        == report["provenance"]["base_checkpoint_sha256_after"],
        "checkpoint hash changed",
    )
    require(
        dashboard["next_decision"]["learned_scalar_development_training_authorized"] is False,
        "scalar training prematurely authorized",
    )
    require(
        dashboard["next_decision"]["learned_scalar_confirmatory_training_authorized"] is False,
        "confirmatory training prematurely authorized",
    )
    require(dashboard["next_decision"]["barzilai_borwein_control_required_next"] is True, "BB gate")
    require(config["claims_boundary"]["barzilai_borwein_control_tested"] is False, "BB claim")
    require(config["claims_boundary"]["learned_step_tested"] is False, "learned claim")
    require(config["claims_boundary"]["fresh_blind_opened"] is False, "blind claim")
    require(config["claims_boundary"]["private_dataset_public"] is False, "private claim")

    for method in METHODS:
        if method in {"locked_fno_raw", "ridge_raw"}:
            continue
        method_rows = [row for row in samples if row["method"] == method]
        require(
            max(float(row["outside_support_relative_l2"]) for row in method_rows) < 1e-12,
            f"{method} escaped support",
        )
        require(
            max(float(row["negative_voxel_fraction"]) for row in method_rows) == 0.0,
            f"{method} has negative voxels",
        )
    require(
        max(float(row["projected_objective_increase_fraction"]) for row in quadratic)
        <= config["gate"]["maximum_quadratic_step_projected_objective_increase_fraction"],
        "quadratic objective gate",
    )
    require(
        all("before projection only" in row["wording_boundary"] for row in quadratic),
        "quadratic wording boundary",
    )
    worst = min(layouts, key=lambda row: float(row["mean_field_gain_pct"]))
    require(
        dashboard["worst_layout"]["source_split"] == worst["source_split"]
        and dashboard["worst_layout"]["geometry_id"] == worst["geometry_id"],
        "worst layout mismatch",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "scientific_result": dashboard["scientific_status"],
                "validation_grid_rows": len(screen),
                "pair_manifest_rows": len(manifest),
                "sample_metric_rows": len(samples),
                "summary_rows": len(summaries),
                "pairwise_rows": len(pairwise),
                "champion": champion,
                "operator_call_frontier": commit["selected"]["operator_call_frontier"],
                "public_checksum_targets": PUBLIC_COUNT,
                "base_checkpoint_drift": report["provenance"]["base_checkpoint_drift"],
                "learned_scalar_authorized": dashboard["next_decision"][
                    "learned_scalar_development_training_authorized"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
