#!/usr/bin/env python3
"""Validate the committed M3B family-selector result package."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "family_selector.json"
RESULTS = ROOT / "results" / "family_selector"


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = json.loads((RESULTS / "family_selector_report.json").read_text(encoding="utf-8"))
    raw = read_csv("family_selector_raw.csv")
    selected = read_csv("family_selector_selected.csv")
    ablation = read_csv("family_selector_ablation.csv")
    grid = config["grid"]

    expected_cells = (
        len(grid["families"])
        * len(grid["dynamics"])
        * len(grid["noise_levels"])
        * len(grid["view_counts"])
        * int(grid["seed_count"])
    )
    ranks = [int(value) for value in grid["ranks"]]
    expected_candidates = expected_cells * len(ranks)
    selector_count = 7
    ablation_count = 5

    assert len(raw) == expected_candidates, (len(raw), expected_candidates)
    assert len(selected) == expected_cells * selector_count
    assert len(ablation) == expected_cells * ablation_count
    assert len({(row["cell_id"], row["rank"]) for row in raw}) == len(raw)
    assert len({(row["cell_id"], row["selector"]) for row in selected}) == len(selected)
    assert len({(row["cell_id"], row["selector"]) for row in ablation}) == len(ablation)

    by_cell: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        by_cell[row["cell_id"]].append(row)
    assert len(by_cell) == expected_cells
    for rows in by_cell.values():
        assert sorted(int(row["rank"]) for row in rows) == ranks
        oracle_rank = int(rows[0]["oracle_rank"])
        assert all(int(row["oracle_rank"]) == oracle_rank for row in rows)
        oracle_error = min(float(row["field_rel_l2"]) for row in rows)
        assert abs(float(rows[0]["oracle_field_rel_l2"]) - oracle_error) < 1e-12
        full = max(rows, key=lambda row: int(row["rank"]))
        assert abs(float(full["field_error_ratio_to_full"]) - 1.0) < 1e-10
        assert min(float(row["oracle_regret_pct"]) for row in rows) >= -1e-9

    selector_counts = Counter(row["selector"] for row in selected)
    assert set(selector_counts.values()) == {expected_cells}
    fixed_rows = [row for row in selected if row["selector"] == "fixed_rank3"]
    assert all(int(row["selected_rank"]) == 3 for row in fixed_rows)
    assert min(float(row["regret_pct"]) for row in selected) >= -1e-9

    forbidden = {"family", "dynamics", "noise_level", "field_rel_l2", "oracle_rank", "target_log_error_ratio"}
    for heldout_family, audit in report["model_selection_audit"].items():
        assert audit["heldout_family"] == heldout_family
        assert heldout_family not in audit["training_families"]
        assert set(audit["training_families"]) == set(grid["families"]) - {heldout_family}
    for mode, fold_models in report["models"].items():
        for heldout_family, model in fold_models.items():
            features = set(model["features"])
            assert not features.intersection(forbidden)
            if mode == "with_holdout":
                assert "log_heldout_residual" in features
            else:
                assert "log_heldout_residual" not in features
            assert heldout_family in grid["families"]

    design = report["design"]
    assert int(design["observation_cells"]) == expected_cells
    assert int(design["candidate_rows"]) == expected_candidates
    assert int(design["selector_rows"]) == expected_cells * selector_count
    assert int(report["feature_ablation_rows"]) == expected_cells * ablation_count

    manifest_lines = (RESULTS / "family_selector_checksums.sha256").read_text(encoding="ascii").splitlines()
    assert len(manifest_lines) == 7
    for line in manifest_lines:
        expected_digest, filename = line.split("  ", 1)
        actual_digest = hashlib.sha256((RESULTS / filename).read_bytes()).hexdigest()
        assert actual_digest == expected_digest, filename

    print("M3B family-selector validation passed")
    print(f"observation_cells={expected_cells}")
    print(f"candidate_rows={len(raw)}")
    print(f"selector_rows={len(selected)}")
    print(f"ablation_rows={len(ablation)}")
    print("checksum_manifest=verified")
    print(f"oracle_rank_counts={report['oracle_rank_counts']}")


if __name__ == "__main__":
    main()
