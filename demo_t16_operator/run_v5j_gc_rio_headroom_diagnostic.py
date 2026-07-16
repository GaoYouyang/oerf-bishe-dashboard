#!/usr/bin/env python3
"""Post-open development headroom diagnosis for GC-RIO.

This script intentionally reads development truth and clean residuals only
after the v5h/v5i decisions. It never constructs the design-lock rigs, and its
oracle rows are diagnostic ceilings rather than deployable methods.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from demo_t16_operator.gc_rio.data import build_dataset
    from demo_t16_operator.gc_rio.protocol import make_development_config, sha256_json
else:
    from .gc_rio.data import build_dataset
    from .gc_rio.protocol import make_development_config, sha256_json


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "v5h_gc_rio_development.json"
OUTPUT_DIR = ROOT / "results" / "v5j_gc_rio_headroom_diagnostic"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def paired_target_projection(
    operators: Sequence[np.ndarray],
    clean_residuals: Sequence[np.ndarray],
    support: np.ndarray,
    *,
    relative_ridge: float = 1e-4,
) -> np.ndarray:
    """Fit one support field to both clean target residuals of a field."""

    matrix = np.concatenate([np.asarray(item, dtype=np.float64) for item in operators])
    target = np.concatenate(
        [np.asarray(item, dtype=np.float64) for item in clean_residuals]
    )
    mask = np.asarray(support, dtype=bool)
    active = matrix[:, mask]
    mean_diagonal = float(np.mean(np.sum(np.square(active), axis=0)))
    ridge = max(float(relative_ridge) * mean_diagonal, 1e-10)
    normal = active.T @ active + ridge * np.eye(active.shape[1])
    fitted = np.linalg.solve(normal, active.T @ target)
    field = np.zeros(matrix.shape[1], dtype=np.float64)
    field[mask] = fitted
    return field.astype(np.float32)


def _row_metric(prediction: np.ndarray, target: np.ndarray, sigma: float) -> float:
    return float(
        np.sqrt(
            np.mean(
                ((np.asarray(prediction) - np.asarray(target)) / float(sigma)) ** 2
            )
        )
    )


def _aggregate(rows: Sequence[Mapping[str, Any]], metric: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["method"]),
            str(row["split"]),
            str(row["rig_id"]),
            str(row["family"]),
        )
        groups[key].append(float(row[metric]))
    cells = [
        {
            "method": method,
            "split": split,
            "rig_id": rig_id,
            "family": family,
            "row_count": len(values),
            metric: float(np.mean(values)),
        }
        for (method, split, rig_id, family), values in sorted(groups.items())
    ]
    return cells


def run() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    bundle = build_dataset(make_development_config(config))
    if any(row["split"] == "design_lock" for row in bundle.rows):
        raise RuntimeError("design-lock rows must remain unconstructed")
    by_field: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bundle.rows:
        by_field[str(row["field_uid"])].append(row)
    paired_fields: dict[str, np.ndarray] = {}
    for field_uid, rows in by_field.items():
        if len(rows) != 2:
            raise RuntimeError(f"field {field_uid} must have two target rows")
        paired_fields[field_uid] = paired_target_projection(
            [row["target_operator"] for row in rows],
            [row["clean_target_residual"] for row in rows],
            rows[0]["support"],
        )

    records: list[dict[str, Any]] = []
    for row in bundle.rows:
        truth = bundle.truth_fields[str(row["field_uid"])]
        true_field_correction = truth - row["base_field"]
        predictions = {
            "zero_correction": np.zeros_like(row["target_residual_label"]),
            "analytic_source_correction": row["target_operator"]
            @ row["analytic_correction"],
            "true_field_correction_oracle": row["target_operator"]
            @ true_field_correction,
            "paired_target_field_oracle": row["target_operator"]
            @ paired_fields[str(row["field_uid"])],
            "clean_measurement_oracle": row["clean_target_residual"],
        }
        for method, prediction in predictions.items():
            records.append(
                {
                    "method": method,
                    "split": row["split"],
                    "rig_id": row["rig_id"],
                    "family": row["family"],
                    "field_uid": row["field_uid"],
                    "target_view": int(row["target_view"]),
                    "noisy_whitened_rmse": _row_metric(
                        prediction,
                        row["target_residual_label"],
                        float(row["target_sigma"]),
                    ),
                    "clean_whitened_rmse": _row_metric(
                        prediction,
                        row["clean_target_residual"],
                        float(row["target_sigma"]),
                    ),
                }
            )
    noisy_cells = _aggregate(records, "noisy_whitened_rmse")
    clean_cells = _aggregate(records, "clean_whitened_rmse")
    summaries: list[dict[str, Any]] = []
    for method in sorted({str(row["method"]) for row in records}):
        for split in ("train", "validation"):
            noisy = [
                float(row["noisy_whitened_rmse"])
                for row in noisy_cells
                if row["method"] == method and row["split"] == split
            ]
            clean = [
                float(row["clean_whitened_rmse"])
                for row in clean_cells
                if row["method"] == method and row["split"] == split
            ]
            summaries.append(
                {
                    "method": method,
                    "split": split,
                    "rig_family_cells": len(noisy),
                    "cell_mean_noisy_whitened_rmse": float(np.mean(noisy)),
                    "cell_mean_clean_whitened_rmse": float(np.mean(clean)),
                }
            )
    validation = {
        row["method"]: row
        for row in summaries
        if row["split"] == "validation"
    }
    analytic = float(
        validation["analytic_source_correction"]["cell_mean_noisy_whitened_rmse"]
    )
    headroom = {
        method: 1.0
        - float(row["cell_mean_noisy_whitened_rmse"]) / max(analytic, 1e-12)
        for method, row in validation.items()
    }
    report = {
        "schema": "v5j-gc-rio-postopen-headroom-1",
        "evidence_label": "postopen_development_oracle_diagnostic",
        "config_sha256": sha256_json(config),
        "design_lock_rows_constructed": 0,
        "validation_truth_opened_after_v5h_v5i_decisions": True,
        "row_count": len(bundle.rows),
        "summary": summaries,
        "validation_noisy_headroom_vs_analytic": headroom,
        "interpretation_rules": {
            "true_field_correction_oracle": "Tests whether the physical truth-field difference survives model-operator mismatch.",
            "paired_target_field_oracle": "Uses both clean targets and is an unattainable representability ceiling.",
            "clean_measurement_oracle": "Uses the clean target residual and is an unattainable noise floor.",
        },
        "claim_boundary": [
            "All oracle methods read development truth or clean target residuals and are not deployable.",
            "No design-lock rig, target or label was constructed or opened.",
            "The generator remains a synthetic weak-deflection model, not OERF evidence.",
            "These rows may choose a new development mechanism but cannot support a confirmatory claim.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "oracle_rows.csv", records)
    _write_csv(OUTPUT_DIR / "oracle_summary.csv", summaries)
    _write_json(OUTPUT_DIR / "report.json", report)
    return report


def main() -> None:
    report = run()
    print(
        json.dumps(
            {
                "validation_noisy_headroom_vs_analytic": report[
                    "validation_noisy_headroom_vs_analytic"
                ],
                "design_lock_rows_constructed": report[
                    "design_lock_rows_constructed"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
