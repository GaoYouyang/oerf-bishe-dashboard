#!/usr/bin/env python3
"""Independently validate the public v3k-F noise-stopping artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "v3k_f_noise_stopping_gate.json"
EXPECTED_STATUS = (
    "DISCREPANCY_RESCUES_MEAN_NOISE_OOD_BUT_TAIL_OR_FRESH_GATE_BLOCKS_LEARNING"
)
METHODS = {
    "feasible_fno",
    "fno_geometry",
    "fno_pbb_fixed64",
    "fno_pbb_discrepancy",
    "fno_pbb_camera_discrepancy",
    "fno_pbb_ncp",
    "fno_pbb_hybrid",
    "fno_pbb_generator_sigma",
    "fno_pbb_truth_oracle",
}
STOPPING_METHODS = METHODS - {"feasible_fno", "fno_geometry"}
SPLITS = {
    "val_tune",
    "val_lock",
    "test_iid",
    "test_noise_ood",
    "test_family_ood",
    "test_joint_ood",
}
PUBLIC_COUNT = 11


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


def validate_checksums(result_dir: Path) -> None:
    lines = (result_dir / "v3k_f_noise_stopping_checksums.sha256").read_text(
        encoding="utf-8"
    ).splitlines()
    require(len(lines) == PUBLIC_COUNT, "checksum target count")
    for line in lines:
        expected, name = line.split("  ", 1)
        path = result_dir / name
        require(path.is_file(), f"missing checksum target: {name}")
        require(sha256(path) == expected, f"checksum mismatch: {name}")


def validate_no_private_leak(result_dir: Path) -> None:
    forbidden = ("/Users/", "private_library", ".npz", ".pt", "webvpn")
    for path in result_dir.iterdir():
        if path.suffix not in {".csv", ".json", ".sha256"}:
            continue
        payload = path.read_text(encoding="utf-8", errors="ignore")
        for marker in forbidden:
            require(marker not in payload, f"private marker {marker!r} in {path.name}")


def validate_selection(config: dict) -> dict:
    selection_dir = ROOT / "results" / str(config["selection_output_dir"])
    assignment = read_csv(selection_dir / "v3k_f_validation_assignment.csv")
    screen = read_csv(selection_dir / "v3k_f_selection_screen.csv")
    commit = read_json(selection_dir / "v3k_f_selection_commit.json")
    require(commit["test_dataset_constructed"] is False, "selection saw test data")
    require(len(assignment) == 40, "validation field assignment count")
    roles = defaultdict(list)
    for row in assignment:
        roles[row["validation_role"]].append(int(row["source_index"]))
    require(len(roles["tune"]) == 24, "V_tune field count")
    require(len(roles["lock"]) == 16, "V_lock field count")
    require(set(roles["tune"]).isdisjoint(roles["lock"]), "validation role overlap")
    require(len(screen) == 215, "selection screen row count")
    require({row["selection_role"] for row in screen} == {"v_tune"}, "selection role")
    require(
        all(row["test_domain_used"] == "False" for row in screen),
        "selection screen test leak",
    )
    require(
        all(row["per_sample_truth_used_at_runtime"] == "False" for row in screen),
        "deployable runtime truth leak",
    )
    selected = commit["selected"]
    require(float(selected["self_discrepancy"]["tau"]) == 0.7, "primary tau")
    require(
        int(selected["self_discrepancy"]["independent_field_count"]) == 24,
        "selection statistical unit",
    )
    return commit


def validate_roles_and_counts(
    metrics: list[dict[str, str]],
    pairs: list[dict[str, str]],
    validation_roles: list[dict[str, str]],
) -> None:
    require(len(pairs) == 672, "pair count")
    require(len(metrics) == 672 * len(METHODS), "metric row count")
    require({row["method"] for row in metrics} == METHODS, "metric methods")
    require({row["source_split"] for row in metrics} == SPLITS, "metric splits")
    require(len(validation_roles) == 40, "audit validation role count")
    role_by_source = {
        int(row["source_index"]): row["validation_role"] for row in validation_roles
    }
    layouts_by_source: dict[int, set[int]] = defaultdict(set)
    pair_role_by_source: dict[int, set[str]] = defaultdict(set)
    for row in pairs:
        source = int(row["source_index"])
        layouts_by_source[source].add(int(row["pair_index"]))
        pair_role_by_source[source].add(row["evaluation_role"])
    for source, role in role_by_source.items():
        require(len(layouts_by_source[source]) == 4, "four layouts per validation field")
        expected = "val_tune" if role == "tune" else "val_lock"
        require(pair_role_by_source[source] == {expected}, "field crossed validation roles")


def validate_stopping(rows: list[dict[str, str]], maximum: int) -> None:
    require(len(rows) == 672 * len(STOPPING_METHODS), "stopping row count")
    require({row["method"] for row in rows} == STOPPING_METHODS, "stopping methods")
    for row in rows:
        stop = int(row["stop_iteration"])
        a_calls = int(row["a_calls"])
        at_calls = int(row["at_calls"])
        require(0 <= stop <= maximum, "stop iteration range")
        if row["method"] == "fno_pbb_truth_oracle":
            require((a_calls, at_calls) == (maximum, maximum), "oracle audit path cost")
        elif stop == maximum:
            require((a_calls, at_calls) == (maximum, maximum), "forced cap cost")
        else:
            require((a_calls, at_calls) == (stop + 1, stop), "first-crossing cost")
        require(
            int(row["total_operator_calls"]) == a_calls + at_calls,
            "total operator call arithmetic",
        )
        if row["method"] in {
            "fno_pbb_discrepancy",
            "fno_pbb_camera_discrepancy",
            "fno_pbb_ncp",
            "fno_pbb_hybrid",
        }:
            require(row["deployable_input_only"] == "True", "deployable label")
            require(row["oracle"] == "False", "deployable method marked oracle")


def collapsed_field_errors(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str, int], float]:
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["source_split"], row["method"], int(row["source_index"]))].append(
            float(row["field_rel_l2"])
        )
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def validate_primary_claim(metrics: list[dict[str, str]], dashboard: dict) -> None:
    collapsed = collapsed_field_errors(metrics)
    for split in SPLITS:
        sources = sorted(
            source
            for candidate_split, method, source in collapsed
            if candidate_split == split and method == "fno_pbb_discrepancy"
        )
        candidate = np.asarray(
            [collapsed[(split, "fno_pbb_discrepancy", source)] for source in sources]
        )
        comparator = np.asarray(
            [collapsed[(split, "fno_geometry", source)] for source in sources]
        )
        gains = 100.0 * (comparator - candidate) / comparator
        reported = dashboard["primary_vs_fixed_landweber"][split]
        close(np.mean(gains), reported["mean_field_gain_pct"], f"{split} mean gain")
        close(
            np.mean(gains < -1.0),
            reported["harm_rate_gt_1pct"],
            f"{split} harm tail",
        )
        require(len(sources) == int(reported["independent_field_count"]), "field count")


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    result_dir = args.result_dir or ROOT / "results" / str(config["audit_output_dir"])
    selection = validate_selection(config)
    validate_checksums(result_dir)
    validate_no_private_leak(result_dir)
    metrics = read_csv(result_dir / "v3k_f_sample_metrics.csv")
    pairs = read_csv(result_dir / "v3k_f_pair_manifest.csv")
    validation_roles = read_csv(result_dir / "v3k_f_validation_roles.csv")
    stopping = read_csv(result_dir / "v3k_f_stopping_rows.csv")
    dashboard = read_json(result_dir / "v3k_f_noise_stopping_dashboard.json")
    validate_roles_and_counts(metrics, pairs, validation_roles)
    validate_stopping(stopping, int(config["frozen_pbb"]["maximum_iterations"]))
    validate_primary_claim(metrics, dashboard)
    require(dashboard["scientific_status"] == EXPECTED_STATUS, "scientific status")
    require(dashboard["deterministic_development_gate_passed"] is False, "gate status")
    require(dashboard["learned_stopping_authorized"] is False, "learned-stop gate")
    require(dashboard["gate_checks"]["lock_tail_safe"] is False, "lock tail result")
    require(
        dashboard["gate_checks"]["noise_ood_tail_safe"] is False,
        "noise-OOD tail result",
    )
    require(selection["test_dataset_constructed"] is False, "selection provenance")
    print(
        json.dumps(
            {
                "status": "PASS",
                "independent_metric_rows": len(metrics),
                "independent_stopping_rows": len(stopping),
                "validated_scientific_status": EXPECTED_STATUS,
                "private_path_scan": "PASS",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
