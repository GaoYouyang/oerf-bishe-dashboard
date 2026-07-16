#!/usr/bin/env python3
"""Independently validate the public v3k-E projected-BB artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "v3k_e_projected_bb_gate.json"
EXPECTED_STATUS = (
    "PBB_MEAN_ACCELERATION_BUT_NOISE_OOD_AND_TAIL_UNSAFE_"
    "DISCREPANCY_CONTROL_REQUIRED"
)
METHODS = (
    "feasible_fno",
    "fno_geometry",
    "fno_quadratic",
    "fno_pbb_strict",
    "fno_pbb_wide",
    "feasible_ridge",
    "ridge_geometry",
    "ridge_quadratic",
    "ridge_pbb_strict",
    "ridge_pbb_wide",
)
COMPARISONS = (
    ("fno_pbb_strict", "fno_geometry"),
    ("fno_pbb_wide", "fno_geometry"),
    ("fno_pbb_wide", "fno_quadratic"),
    ("fno_pbb_wide", "fno_pbb_strict"),
    ("ridge_pbb_strict", "ridge_geometry"),
    ("ridge_pbb_wide", "ridge_geometry"),
    ("ridge_pbb_wide", "ridge_quadratic"),
    ("ridge_pbb_wide", "ridge_pbb_strict"),
    ("fno_pbb_wide", "ridge_pbb_wide"),
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
    manifest = result_dir / "v3k_e_projected_bb_checksums.sha256"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    require(len(lines) == PUBLIC_COUNT, "checksum target count")
    for line in lines:
        expected, name = line.split("  ", 1)
        path = result_dir / name
        require(path.is_file(), f"missing checksum target: {name}")
        require(sha256(path) == expected, f"checksum mismatch: {name}")


def select_screen(
    rows: list[dict[str, str]], variants: list[str]
) -> dict[str, str]:
    require(bool(rows), "empty PBB selection cell")
    order = {variant: index for index, variant in enumerate(variants)}
    return min(
        rows,
        key=lambda row: (
            float(row["source_field_mean_rel_l2"]),
            int(row["operator_a_calls_per_sample"])
            + int(row["operator_at_calls_per_sample"]),
            int(row["iterations"]),
            order[row["bb_variant"]],
            float(row["normalized_step_max_bound"]),
        ),
    )


def validate_selection(
    screen: list[dict[str, str]], commit: dict, config: dict
) -> None:
    protocol = config["selection_protocol"]
    expected_per_start = len(protocol["variants"]) * len(
        protocol["iteration_counts"]
    ) * (1 + len(protocol["wide_normalized_step_max_grid"]))
    require(len(screen) == 2 * expected_per_start == 180, "selection row count")
    require({row["selection_split"] for row in screen} == {"val"}, "selection split")
    require(
        all(row["selection_uses_audit_or_reprojection"] == "False" for row in screen),
        "selection metric leak flag",
    )
    require(commit["selection_split"] == "val", "commit selection split")
    require(commit["audit_camera_used_for_selection"] is False, "audit camera leak")
    require(
        commit["audit_or_reprojection_metrics_computed_by_selection_function"]
        is False,
        "selection computed sealed metric",
    )
    require(
        commit["test_field_or_metric_rows_present_at_commit"] is False,
        "test rows present at selection commit",
    )
    require(
        commit["strict_and_wide_selected_independently"] is True,
        "strict/wide selection independence",
    )

    variants = [str(value) for value in protocol["variants"]]
    for start in ("fno", "ridge"):
        for mode in ("strict", "wide"):
            winner = select_screen(
                [
                    row
                    for row in screen
                    if row["start"] == start and row["bb_mode"] == mode
                ],
                variants,
            )
            actual = commit["selected_projected_bb"][f"{start}_{mode}"]
            for key in (
                "start",
                "bb_mode",
                "bb_variant",
                "iterations",
                "operator_a_calls_per_sample",
                "operator_at_calls_per_sample",
            ):
                require(str(actual[key]) == winner[key], f"selection mismatch: {start} {mode} {key}")
            for key in (
                "initial_step_fraction",
                "normalized_step_min_bound",
                "normalized_step_max_bound",
                "normalized_step_observed_min",
                "normalized_step_observed_max",
                "source_field_mean_rel_l2",
                "source_field_median_rel_l2",
                "clipped_high_fraction",
            ):
                close(actual[key], winner[key], f"selection mismatch: {start} {mode} {key}")
            require(
                actual["public_method"] == f"{start}_pbb_{mode}",
                "public method mapping",
            )


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


def validate_metrics_and_summaries(
    rows: list[dict[str, str]],
    summaries: list[dict[str, str]],
    commit: dict,
) -> dict[tuple[str, str, int], dict[str, float]]:
    require(len(rows) == 6720, "sample metric row count")
    require({row["method"] for row in rows} == set(METHODS), "sample methods")
    require(
        all(float(row["outside_support_relative_l2"]) <= 1e-12 for row in rows),
        "hard support violation",
    )
    require(
        all(float(row["negative_voxel_fraction"]) == 0.0 for row in rows),
        "nonnegativity violation",
    )
    collapsed = collapsed_metrics(rows)
    splits = sorted({split for _, split, _ in collapsed})
    require(len(splits) == 5, "split count")
    require(len(summaries) == 50, "split summary row count")

    summary_index = {(row["method"], row["source_split"]): row for row in summaries}
    for split in splits:
        for method in METHODS:
            sources = sorted(
                source
                for candidate, candidate_split, source in collapsed
                if candidate == method and candidate_split == split
            )
            values = np.asarray(
                [collapsed[(method, split, source)]["field_rel_l2"] for source in sources]
            )
            row = summary_index[(method, split)]
            close(np.mean(values), row["mean_field_rel_l2"], "summary mean")
            close(np.median(values), row["median_field_rel_l2"], "summary median")
            close(np.quantile(values, 0.9), row["p90_field_rel_l2"], "summary p90")
            require(int(row["independent_field_count"]) == len(sources), "summary n")

    scores = commit["validation_scores"]
    for method in METHODS:
        sources = sorted(
            source
            for candidate, split, source in collapsed
            if candidate == method and split == "val"
        )
        values = np.asarray(
            [collapsed[(method, "val", source)]["field_rel_l2"] for source in sources]
        )
        close(np.mean(values), scores[method]["source_field_mean_rel_l2"], "commit val mean")
        close(np.median(values), scores[method]["source_field_median_rel_l2"], "commit val median")
        require(len(sources) == int(scores[method]["independent_field_count"]), "commit val n")
    return collapsed


def validate_pairwise(
    rows: list[dict[str, str]],
    collapsed: dict[tuple[str, str, int], dict[str, float]],
    config: dict,
) -> None:
    require(len(rows) == 45, "pairwise row count")
    index = {
        (row["source_split"], row["candidate"], row["comparator"]): row
        for row in rows
    }
    splits = sorted({split for _, split, _ in collapsed})
    gate = config["gate"]
    for split_index, split in enumerate(splits):
        for comparison_index, (candidate, comparator) in enumerate(COMPARISONS):
            sources = sorted(
                source
                for method, candidate_split, source in collapsed
                if method == candidate and candidate_split == split
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
            row = index[(split, candidate, comparator)]
            close(np.mean(field), row["mean_field_gain_pct"], "pair field mean")
            close(np.median(field), row["median_field_gain_pct"], "pair field median")
            close(np.quantile(field, 0.1), row["p10_field_gain_pct"], "pair field p10")
            close(np.mean(field > 0.0), row["positive_field_fraction"], "pair positive")
            close(np.mean(field < -1.0), row["harm_rate_gt_1pct"], "pair harm")
            close(np.mean(audit), row["mean_audit_gain_pct"], "pair audit mean")
            seed = int(gate["bootstrap_seed"]) + 10 * split_index + comparison_index
            field_ci = bootstrap_interval(field, seed, int(gate["bootstrap_replicates"]))
            audit_ci = bootstrap_interval(audit, seed + 1000, int(gate["bootstrap_replicates"]))
            close(field_ci[0], row["field_cluster_ci95_low_pct"], "field CI low")
            close(field_ci[1], row["field_cluster_ci95_high_pct"], "field CI high")
            close(audit_ci[0], row["audit_cluster_ci95_low_pct"], "audit CI low")
            close(audit_ci[1], row["audit_cluster_ci95_high_pct"], "audit CI high")


def validate_audits_and_calls(
    audits: list[dict[str, str]], calls: list[dict[str, str]], commit: dict
) -> None:
    require(len(audits) == 20, "BB audit row count")
    require(len(calls) == 50, "call ledger row count")
    call_index = {(row["source_split"], row["method"]): row for row in calls}
    selected = {
        value["public_method"]: value
        for value in commit["selected_projected_bb"].values()
    }
    for row in audits:
        method = row["method"]
        choice = selected[method]
        iterations = int(row["iterations"])
        layout_count = int(row["layout_count"])
        require(iterations == int(choice["iterations"]), "audit iteration")
        require(int(row["step_sample_count"]) == iterations * layout_count, "audit step count")
        close(
            sum(
                float(row[key])
                for key in (
                    "bb1_step_fraction",
                    "bb2_step_fraction",
                    "initial_step_fraction_of_steps",
                    "fallback_step_fraction",
                )
            ),
            1.0,
            "step source fractions",
        )
        lower = float(row["normalized_step_min_bound"])
        upper = float(row["normalized_step_max_bound"])
        require(float(row["normalized_step_observed_min"]) >= lower - 1e-12, "step lower bound")
        require(float(row["normalized_step_observed_max"]) <= upper + 1e-12, "step upper bound")
        require(row["convergence_claimed"] == "False", "unsupported convergence claim")
        ledger = call_index[(row["source_split"], method)]
        require(int(ledger["operator_a_calls_per_sample"]) == iterations, "PBB A calls")
        require(int(ledger["operator_at_calls_per_sample"]) == iterations, "PBB AT calls")
        require(int(ledger["total_refinement_operator_calls_per_sample"]) == 2 * iterations, "PBB total calls")

    val_strict = next(
        row for row in audits if row["source_split"] == "val" and row["method"] == "fno_pbb_strict"
    )
    require(float(val_strict["clipped_high_fraction"]) > 0.9, "strict PBB degeneracy")
    val_wide = call_index[("val", "fno_pbb_wide")]
    val_geometry = call_index[("val", "fno_geometry")]
    require(
        int(val_wide["total_refinement_operator_calls_per_sample"])
        == int(val_geometry["total_refinement_operator_calls_per_sample"]),
        "same-call comparison",
    )


def validate_public_boundary(result_dir: Path, dashboard: dict, report: dict) -> None:
    require(dashboard["scientific_status"] == EXPECTED_STATUS, "dashboard status")
    require(report["status"] == EXPECTED_STATUS, "report status")
    require(dashboard["publishable_mechanism_gate_passed"] is False, "gate must fail")
    require(
        dashboard["next_decision"]["learned_scalar_development_training_authorized"]
        is False,
        "learned training incorrectly authorized",
    )
    require(
        dashboard["novelty_boundary"]["bb1988_is_baseline_not_innovation"] is True,
        "BB novelty boundary",
    )
    require(report["private_assets"]["private_dataset_published"] is False, "private data publication")
    require(report["private_assets"]["base_checkpoint_published"] is False, "checkpoint publication")
    for path in result_dir.iterdir():
        if path.suffix not in {".json", ".csv", ".sha256"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for forbidden in ("/Users/", ".npz", ".pt", "webvpn", "password"):
            require(forbidden not in text, f"private marker in {path.name}: {forbidden}")


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    result_dir = args.result_dir or ROOT / "results" / str(config["output_dir"])
    validate_checksums(result_dir)
    screen = read_csv(result_dir / "v3k_e_validation_screen.csv")
    commit = read_json(result_dir / "v3k_e_selection_commit.json")
    metrics = read_csv(result_dir / "v3k_e_sample_metrics.csv")
    summaries = read_csv(result_dir / "v3k_e_split_summary.csv")
    pairwise = read_csv(result_dir / "v3k_e_pairwise_summary.csv")
    audits = read_csv(result_dir / "v3k_e_bb_step_audit.csv")
    calls = read_csv(result_dir / "v3k_e_operator_call_ledger.csv")
    manifest = read_csv(result_dir / "v3k_e_pair_manifest.csv")
    layouts = read_csv(result_dir / "v3k_e_layout_summary.csv")
    dashboard = read_json(result_dir / "v3k_e_projected_bb_dashboard.json")
    report = read_json(result_dir / "v3k_e_projected_bb_report.json")

    require(len(manifest) == 672, "pair manifest row count")
    require(len(layouts) == 20, "layout summary row count")
    validate_selection(screen, commit, config)
    collapsed = validate_metrics_and_summaries(metrics, summaries, commit)
    validate_pairwise(pairwise, collapsed, config)
    validate_audits_and_calls(audits, calls, commit)
    validate_public_boundary(result_dir, dashboard, report)
    print(
        json.dumps(
            {
                "status": "VALIDATED",
                "scientific_status": dashboard["scientific_status"],
                "selection_rows": len(screen),
                "sample_metric_rows": len(metrics),
                "pairwise_rows": len(pairwise),
                "checksum_targets": PUBLIC_COUNT,
                "private_assets_published": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
