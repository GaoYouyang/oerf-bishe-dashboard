#!/usr/bin/env python3
"""Post-hoc mechanism diagnosis for the CG-PDNO trust fresh run.

This script is deliberately labelled development-only. It uses validation to
screen a residual-ratio acceptance threshold and reports the already-created
test split only as a diagnostic. It does not authorize a superiority claim.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "results" / "cg_pdno_trust_fresh"
THRESHOLDS = [0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rows() -> list[dict[str, str]]:
    with (RESULT / "sample_metrics.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def field_summary(source: list[dict[str, str]], threshold: float | None) -> dict[str, float]:
    grouped: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    for row in source:
        model = float(row["model_relative_l2"])
        fallback = float(row["fallback_relative_l2"])
        ratio = float(row["model_whitened_residual_rms"]) / float(
            row["fallback_whitened_residual_rms"]
        )
        selected = model if threshold is None or ratio <= threshold else fallback
        grouped[int(row["sample_index"])].append((selected, fallback, ratio))
    gains = []
    accepted = []
    for values in grouped.values():
        selected = float(np.mean([value[0] for value in values]))
        fallback = float(np.mean([value[1] for value in values]))
        gains.append(100.0 * (fallback - selected) / fallback)
        accepted.append(float(np.mean([value[2] <= threshold for value in values])) if threshold else 1.0)
    return {
        "source_field_count": len(grouped),
        "mean_gain_percent": float(np.mean(gains)),
        "p10_gain_percent": float(np.quantile(gains, 0.1)),
        "harm_rate_over_1_percent": float(np.mean(np.asarray(gains) < -1.0)),
        "mean_acceptance_fraction_across_seeds": float(np.mean(accepted)),
    }


def main() -> None:
    source = rows()
    by_split = {
        split: [row for row in source if row["split"] == split]
        for split in {row["split"] for row in source}
    }
    screen = []
    for threshold in THRESHOLDS:
        item = {"threshold": threshold, **field_summary(by_split["validation"], threshold)}
        screen.append(item)
    admissible = [row for row in screen if row["harm_rate_over_1_percent"] == 0.0]
    selected = sorted(
        admissible,
        key=lambda row: (-row["mean_gain_percent"], row["threshold"]),
    )[0]
    threshold = float(selected["threshold"])

    raw_validation = field_summary(by_split["validation"], None)
    raw_test = field_summary(by_split["test"], None)
    gated_test = field_summary(by_split["test"], threshold)
    pooled_gain = np.asarray([float(row["relative_gain_percent"]) for row in source])
    pooled_residual_ratio = np.asarray(
        [
            float(row["model_whitened_residual_rms"])
            / float(row["fallback_whitened_residual_rms"])
            for row in source
        ]
    )
    pooled_correction = np.asarray(
        [float(row["correction_to_fallback_norm"]) for row in source]
    )
    failure_groups: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in by_split["validation"]:
        failure_groups[(int(row["sample_index"]), row["geometry_id"])].append(row)
    failures = []
    for (index, geometry), values in failure_groups.items():
        gain = float(np.mean([float(row["relative_gain_percent"]) for row in values]))
        if gain < -1.0:
            failures.append(
                {
                    "sample_index": index,
                    "geometry_id": geometry,
                    "mean_gain_percent": gain,
                    "mean_residual_ratio": float(
                        np.mean(
                            [
                                float(row["model_whitened_residual_rms"])
                                / float(row["fallback_whitened_residual_rms"])
                                for row in values
                            ]
                        )
                    ),
                }
            )
    report = {
        "evidence_label": "posthoc_development_diagnostic",
        "claim_status": "NOT_AUTHORIZED_FOR_SUPERIORITY_OR_BLIND_TEST",
        "selection_unit": "source field after averaging three optimization seeds",
        "threshold_rule": "maximize validation mean gain subject to zero source-field harm over 1 percent; lower threshold breaks ties",
        "screen": screen,
        "selected_threshold": threshold,
        "raw_validation": raw_validation,
        "gated_validation": selected,
        "raw_test": raw_test,
        "gated_test_diagnostic": gated_test,
        "pooled_correlations": {
            "gain_vs_residual_ratio": float(np.corrcoef(pooled_gain, pooled_residual_ratio)[0, 1]),
            "gain_vs_correction_norm": float(np.corrcoef(pooled_gain, pooled_correction)[0, 1]),
        },
        "validation_failure_fields": failures,
        "call_accounting_warning": (
            "The current recurrent CG-PDNO needs a complete learned path, a complete fallback path "
            "and two final forward evaluations for this gate. Refactor to a shared deterministic "
            "base plus one learned correction before treating the gate as an algorithm."
        ),
    }
    output = RESULT / "residual_gate_diagnostic.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    checksum = RESULT / "residual_gate_diagnostic.sha256"
    checksum.write_text(f"{sha256(output)}  {output.name}\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
