#!/usr/bin/env python3
"""Independently validate the committed v3k-B mechanism-pilot artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "v3k_b_voxel_ray_set_pilot.json"
EXPECTED_STATUS = "VOXEL_RAY_ANGLE_PAIRING_HARM_STOP_ATTENTION_SCALING"


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
    checksum_path = result_dir / "v3k_b_voxel_ray_set_checksums.sha256"
    lines = checksum_path.read_text(encoding="utf-8").splitlines()
    require(len(lines) == 10, "checksum manifest must cover ten public assets")
    for line in lines:
        expected, name = line.split("  ", 1)
        path = result_dir / name
        require(path.is_file(), f"missing checksum target: {name}")
        require(sha256(path) == expected, f"checksum mismatch: {name}")


def collapsed_errors(
    sample_rows: list[dict[str, str]],
) -> dict[tuple[str, int, str, int], float]:
    grouped: dict[tuple[str, int, str, int], list[float]] = defaultdict(list)
    for row in sample_rows:
        key = (
            row["method"],
            int(row["model_seed"]),
            row["source_split"],
            int(row["source_index"]),
        )
        grouped[key].append(float(row["field_rel_l2"]))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def validate_pairwise(
    sample_rows: list[dict[str, str]],
    pairwise_rows: list[dict[str, str]],
    config: dict,
) -> None:
    errors = collapsed_errors(sample_rows)
    seeds = [int(value) for value in config["model_seeds"]]
    gate = config["mechanism_gate"]
    for row_index, row in enumerate(pairwise_rows):
        split = row["source_split"]
        comparator = row["comparator"]
        sources = sorted(
            {
                int(sample["source_index"])
                for sample in sample_rows
                if sample["source_split"] == split
            }
        )
        field_collapsed = []
        all_gains = []
        seed_means = []
        for source in sources:
            gains = []
            for seed in seeds:
                candidate = errors[("correct_ray_set", seed, split, source)]
                control = errors[(comparator, seed, split, source)]
                gain = 100.0 * (control - candidate) / max(control, 1e-12)
                gains.append(gain)
                all_gains.append(gain)
            field_collapsed.append(float(np.mean(gains)))
        for seed in seeds:
            seed_means.append(
                float(
                    np.mean(
                        [
                            100.0
                            * (
                                errors[(comparator, seed, split, source)]
                                - errors[("correct_ray_set", seed, split, source)]
                            )
                            / max(errors[(comparator, seed, split, source)], 1e-12)
                            for source in sources
                        ]
                    )
                )
            )
        ci_low, ci_high = bootstrap_interval(
            np.asarray(field_collapsed),
            int(gate["bootstrap_seed"]) + row_index,
            int(gate["bootstrap_replicates"]),
        )
        close(np.mean(field_collapsed), row["mean_field_gain_pct"], "pairwise mean")
        close(ci_low, row["field_cluster_ci95_low_pct"], "pairwise CI low")
        close(ci_high, row["field_cluster_ci95_high_pct"], "pairwise CI high")
        close(np.median(field_collapsed), row["median_field_gain_pct"], "pairwise median")
        close(np.quantile(field_collapsed, 0.1), row["p10_field_gain_pct"], "pairwise p10")
        close(
            np.mean(np.asarray(all_gains) < -1.0),
            row["harm_rate_gt_1pct"],
            "pairwise harm rate",
        )
        require(
            int(row["positive_seed_count"]) == sum(value > 0.0 for value in seed_means),
            "positive seed count mismatch",
        )


def validate_swap(
    rows: list[dict[str, str]], dashboard: dict, config: dict
) -> None:
    summary = dashboard["same_model_ray_set_swap"]
    seeds = [int(value) for value in config["model_seeds"]]
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["model_seed"]), int(row["source_index"]))].append(
            float(row["correct_ray_set_field_gain_pct"])
        )
    sources = sorted({source for _, source in grouped})
    field_collapsed = [
        float(np.mean([np.mean(grouped[(seed, source)]) for seed in seeds]))
        for source in sources
    ]
    seed_means = [
        float(np.mean([np.mean(grouped[(seed, source)]) for source in sources]))
        for seed in seeds
    ]
    ci_low, ci_high = bootstrap_interval(
        np.asarray(field_collapsed),
        int(config["mechanism_gate"]["bootstrap_seed"]) + 20_000,
        int(config["mechanism_gate"]["bootstrap_replicates"]),
    )
    close(
        np.mean(field_collapsed),
        summary["mean_correct_ray_set_field_gain_pct"],
        "swap field gain",
    )
    close(ci_low, summary["field_cluster_ci95_low_pct"], "swap CI low")
    close(ci_high, summary["field_cluster_ci95_high_pct"], "swap CI high")
    close(
        np.mean([float(row["attention_swap_mean_l1"]) for row in rows]),
        summary["mean_attention_swap_l1"],
        "swap attention mean",
    )
    close(
        np.mean([float(row["correction_swap_relative_pct"]) for row in rows]),
        summary["mean_correction_swap_relative_pct"],
        "swap correction mean",
    )
    require(
        int(summary["positive_seed_count"]) == sum(value > 0.0 for value in seed_means),
        "swap positive seed count mismatch",
    )


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    result_dir = args.result_dir or ROOT / "results" / str(config["output_dir"])
    dashboard = read_json(result_dir / "v3k_b_voxel_ray_set_dashboard.json")
    report = read_json(result_dir / "v3k_b_voxel_ray_set_report.json")
    history = read_csv(result_dir / "v3k_b_training_history.csv")
    samples = read_csv(result_dir / "v3k_b_sample_metrics.csv")
    pairwise = read_csv(result_dir / "v3k_b_pairwise_mechanism.csv")
    swap = read_csv(result_dir / "v3k_b_same_model_ray_swap.csv")
    pairing = read_csv(result_dir / "v3k_b_ray_angle_pairing_derangement.csv")
    manifest = read_csv(result_dir / "v3k_b_pair_manifest.csv")
    validate_checksums(result_dir)

    require(dashboard["scientific_status"] == EXPECTED_STATUS, "unexpected status")
    require(report["status"] == EXPECTED_STATUS, "report status mismatch")
    require(dashboard["voxel_ray_set_mechanism_gate_pass"] is False, "gate must fail")
    require(len(dashboard["training_records"]) == 12, "expected 12 training runs")
    require(len(history) == 288, "expected 288 history rows")
    require(len(samples) == 10080, "expected 10080 sample rows")
    require(len(pairwise) == 20, "expected 20 pairwise rows")
    require(len(swap) == 480, "expected 480 swap rows")
    require(len(pairing) == 168, "expected 168 pairing rows")
    require(len(manifest) == 1312, "expected 1312 pair-manifest rows")
    require(not list(result_dir.glob("*.pt")), "public result contains checkpoints")
    require(
        {int(row["trainable_parameters"]) for row in dashboard["training_records"]}
        == {915},
        "unexpected trainable parameter count",
    )
    require(
        Counter(row["method"] for row in history)
        == Counter({method: 72 for method in config["methods"]}),
        "history is not balanced across methods",
    )
    require(all(row["fixed_point"] == "False" for row in pairing), "pairing fixed point")
    require(
        all(int(row["active_camera_count"]) == 6 for row in pairing),
        "pairing camera budget mismatch",
    )
    require(report["provenance"]["base_checkpoint_drift"] == 0, "base drift")
    require(
        report["provenance"]["base_checkpoint_sha256_before"]
        == report["provenance"]["base_checkpoint_sha256_after"],
        "base checkpoint hash changed",
    )
    require(dashboard["claims_boundary"]["superiority_tested"] is False, "claim leak")
    require(dashboard["claims_boundary"]["blind_final_opened"] is False, "blind leak")
    require(
        dashboard["mechanism_diagnosis"]["local_ray_values_help_vs_geometry_only"]
        is True,
        "ray evidence diagnosis mismatch",
    )
    require(
        dashboard["mechanism_diagnosis"]["correct_pairing_harms_vs_shuffled_pairing"]
        is True,
        "pairing harm diagnosis mismatch",
    )
    validate_pairwise(samples, pairwise, config)
    validate_swap(swap, dashboard, config)
    print(
        json.dumps(
            {
                "status": "PASS",
                "scientific_result": dashboard["scientific_status"],
                "training_runs": len(dashboard["training_records"]),
                "training_history_rows": len(history),
                "sample_metric_rows": len(samples),
                "same_model_swap_rows": len(swap),
                "base_checkpoint_drift": report["provenance"][
                    "base_checkpoint_drift"
                ],
                "superiority_authorized": dashboard["next_decision"][
                    "superiority_training_authorized"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
