#!/usr/bin/env python3
"""Validate the committed M3B geometry-transfer and uncertainty package."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from demo_m3b.run_m3b_geometry_uq import (
    MODEL_FEATURE_SETS,
    SELECTOR_ORDER,
    UQ_SCORE_FIELDS,
    geometry_descriptors,
    largest_gap_midpoint,
)


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "geometry_uq.json"
RESULTS = ROOT / "results" / "geometry_uq"


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = json.loads((RESULTS / "geometry_report.json").read_text(encoding="utf-8"))
    raw = read_csv("geometry_raw.csv")
    selected = read_csv("geometry_selected.csv")
    summary = read_csv("geometry_summary.csv")
    folds = read_csv("geometry_fold_summary.csv")
    uq = read_csv("geometry_uq_cells.csv")
    uq_audit = read_csv("geometry_uq_audit.csv")
    risk = read_csv("geometry_risk_coverage.csv")
    grid = config["grid"]

    expected_cells = (
        len(grid["families"])
        * len(grid["dynamics"])
        * len(grid["noise_levels"])
        * len(grid["view_counts"])
        * len(grid["geometries"])
        * int(grid["seed_count"])
    )
    ranks = [int(value) for value in grid["ranks"]]
    expected_candidates = expected_cells * len(ranks)
    expected_selected = expected_cells * len(SELECTOR_ORDER)
    expected_uq = expected_cells * len(MODEL_FEATURE_SETS)
    expected_risk = (
        len(MODEL_FEATURE_SETS)
        * len(UQ_SCORE_FIELDS)
        * len(config["selectors"]["risk_coverages"])
    )

    assert len(raw) == expected_candidates, (len(raw), expected_candidates)
    assert len(selected) == expected_selected
    assert len(summary) == len(SELECTOR_ORDER)
    assert len(folds) == len(SELECTOR_ORDER) * len(grid["geometries"])
    assert len(uq) == expected_uq
    assert len(uq_audit) == len(MODEL_FEATURE_SETS) * len(UQ_SCORE_FIELDS)
    assert len(risk) == expected_risk
    assert len({(row["cell_id"], row["rank"]) for row in raw}) == len(raw)
    assert len({(row["cell_id"], row["selector"]) for row in selected}) == len(selected)
    assert len({(row["cell_id"], row["mode"]) for row in uq}) == len(uq)

    by_cell: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        by_cell[row["cell_id"]].append(row)
    assert len(by_cell) == expected_cells
    for rows in by_cell.values():
        assert sorted(int(row["rank"]) for row in rows) == ranks
        assert len({row["geometry"] for row in rows}) == 1
        assert len({row["true_angles_deg"] for row in rows}) == 1
        assert len({row["reconstruction_angles_deg"] for row in rows}) == 1
        true_angles = np.asarray(json.loads(rows[0]["true_angles_deg"]), dtype=float)
        reconstruction_angles = np.asarray(
            json.loads(rows[0]["reconstruction_angles_deg"]), dtype=float
        )
        assert len(true_angles) == int(rows[0]["view_count"])
        assert len(reconstruction_angles) == len(true_angles)
        if rows[0]["geometry"] == "calibration_offset_2deg":
            circular_offset = np.mod(reconstruction_angles - true_angles, 180.0)
            assert np.allclose(circular_offset, 2.0)
        else:
            assert np.allclose(reconstruction_angles, true_angles)
        expected_heldout = largest_gap_midpoint(true_angles)
        assert math.isclose(float(rows[0]["heldout_angle_deg"]), expected_heldout, abs_tol=1e-10)
        descriptors = geometry_descriptors(reconstruction_angles)
        for name, expected in descriptors.items():
            assert math.isclose(float(rows[0][name]), expected, rel_tol=1e-10, abs_tol=1e-10)
        oracle_rank = int(rows[0]["oracle_rank"])
        oracle_error = min(float(row["field_rel_l2"]) for row in rows)
        assert all(int(row["oracle_rank"]) == oracle_rank for row in rows)
        assert all(
            math.isclose(float(row["oracle_field_rel_l2"]), oracle_error, rel_tol=1e-10)
            for row in rows
        )
        full = max(rows, key=lambda row: int(row["rank"]))
        assert math.isclose(float(full["field_error_ratio_to_full"]), 1.0, rel_tol=1e-9)
        assert min(float(row["oracle_regret_pct"]) for row in rows) >= -1e-9

    assert set(Counter(row["selector"] for row in selected).values()) == {expected_cells}
    assert set(Counter(row["mode"] for row in uq).values()) == {expected_cells}
    assert all(int(row["selected_rank"]) == 3 for row in selected if row["selector"] == "fixed_rank3")
    assert min(float(row["regret_pct"]) for row in selected) >= -1e-9
    for row in uq:
        percentiles = [
            float(row["prediction_std_percentile"]),
            float(row["vote_entropy_percentile"]),
            float(row["inverse_margin_percentile"]),
            float(row["predicted_risk_percentile"]),
        ]
        assert all(0.0 <= value <= 1.0 for value in percentiles)
        assert math.isclose(float(row["combined_uncertainty"]), float(np.mean(percentiles)))
        assert float(row["selected_regret_pct"]) >= -1e-9
        assert float(row["full_rank_regret_pct"]) >= -1e-9
        votes = json.loads(row["rank_votes"])
        assert len(votes) == len(grid["geometries"]) - 1
        assert set(votes).issubset(set(ranks))

    forbidden = {
        "geometry",
        "family",
        "dynamics",
        "noise_level",
        "field_rel_l2",
        "oracle_rank",
        "target_log_error_ratio",
    }
    expected_geometries = set(grid["geometries"])
    for heldout_geometry, audit in report["model_selection_audit"].items():
        assert audit["heldout_geometry"] == heldout_geometry
        assert set(audit["training_geometries"]) == expected_geometries - {heldout_geometry}
        for mode in MODEL_FEATURE_SETS:
            features = set(audit[mode]["features"])
            assert not features.intersection(forbidden)
            assert features == set(MODEL_FEATURE_SETS[mode])
            members = audit[mode]["ensemble_members"]
            assert len(members) == len(expected_geometries) - 1
            for member in members:
                omitted = member["omitted_training_geometry"]
                assert set(member["fit_geometries"]) == expected_geometries - {
                    heldout_geometry,
                    omitted,
                }
    for mode, fold_models in report["models"].items():
        for heldout_geometry, bundle in fold_models.items():
            assert heldout_geometry in expected_geometries
            assert not set(bundle["central"]["features"]).intersection(forbidden)
            assert len(bundle["ensemble"]) == len(expected_geometries) - 1
            assert all(
                set(model["features"]) == set(MODEL_FEATURE_SETS[mode])
                for model in bundle["ensemble"]
            )

    for row in risk:
        accepted = int(row["accepted_cells"])
        rejected = int(row["rejected_cells"])
        assert accepted + rejected == expected_cells
        assert math.isclose(float(row["actual_coverage"]), accepted / expected_cells)
        assert float(row["selective_mean_regret_pct"]) >= -1e-9
        assert float(row["full_fallback_system_mean_regret_pct"]) >= -1e-9

    design = report["design"]
    assert int(design["observation_cells"]) == expected_cells
    assert int(design["candidate_rows"]) == expected_candidates
    assert int(design["selector_rows"]) == expected_selected
    assert int(design["uncertainty_rows"]) == expected_uq

    manifest_lines = (RESULTS / "geometry_uq_checksums.sha256").read_text(
        encoding="ascii"
    ).splitlines()
    assert len(manifest_lines) == 8
    for line in manifest_lines:
        expected_digest, filename = line.split("  ", 1)
        actual_digest = hashlib.sha256((RESULTS / filename).read_bytes()).hexdigest()
        assert actual_digest == expected_digest, filename

    print("M3B geometry-transfer validation passed")
    print(f"observation_cells={expected_cells}")
    print(f"candidate_rows={len(raw)}")
    print(f"selector_rows={len(selected)}")
    print(f"uncertainty_rows={len(uq)}")
    print(f"risk_coverage_rows={len(risk)}")
    print("outer_geometry_leakage=none_detected")
    print("checksum_manifest=verified")


if __name__ == "__main__":
    main()
