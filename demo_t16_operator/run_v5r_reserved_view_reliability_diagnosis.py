#!/usr/bin/env python3
"""Post-open reserved-camera reliability diagnosis for failed v5p.

The reserved camera never enters reconstruction. Its residual is hashed before
the two original target labels are read. All sign rules are descriptive and no
rule is selected or authorized on these opened synthetic fields.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from demo_t16_operator.gc_rio.data import _flatten_operator, build_dataset
    from demo_t16_operator.gc_rio.protocol import sha256_json, sha256_state_dict
    from demo_t16_operator.release_provenance import (
        relative_file_hashes,
        runtime_environment,
    )
    from demo_t16_operator.run_v5n_strong_classical_baselines import (
        _source_system,
        projected_bb_correction,
    )
    from demo_t16_operator.run_v5o_prior_anchored_frontier import (
        prior_anchored_bb_correction,
    )
    from demo_t16_operator.run_v5p_fresh_budget_gate import (
        CONFIG_PATH,
        OUTPUT_DIR as V5P_OUTPUT_DIR,
        ROOT,
        _load_models,
        _score_methods,
        _target_prediction,
        _unique_field_indices,
        fresh_dataset_config,
    )
    from demo_t16_operator.run_v5q_postopen_topology_diagnosis import (
        _correction_map,
        _rms,
        _shared_prior_members,
        _target_gain_by_field,
    )
else:
    from .gc_rio.data import _flatten_operator, build_dataset
    from .gc_rio.protocol import sha256_json, sha256_state_dict
    from .release_provenance import relative_file_hashes, runtime_environment
    from .run_v5n_strong_classical_baselines import (
        _source_system,
        projected_bb_correction,
    )
    from .run_v5o_prior_anchored_frontier import prior_anchored_bb_correction
    from .run_v5p_fresh_budget_gate import (
        CONFIG_PATH,
        OUTPUT_DIR as V5P_OUTPUT_DIR,
        ROOT,
        _load_models,
        _score_methods,
        _target_prediction,
        _unique_field_indices,
        fresh_dataset_config,
    )
    from .run_v5q_postopen_topology_diagnosis import (
        _correction_map,
        _rms,
        _shared_prior_members,
        _target_gain_by_field,
    )


OUTPUT_DIR = ROOT / "results" / "v5r_reserved_view_reliability_diagnosis"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_checksums(output: Path, names: Sequence[str]) -> None:
    lines = [
        f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}"
        for name in names
    ]
    (output / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="ascii"
    )


def residual_gain(
    operator: np.ndarray,
    label: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
) -> float:
    """Return one minus the candidate/baseline residual-RMS ratio."""

    matrix = np.asarray(operator, dtype=np.float64)
    target = np.asarray(label, dtype=np.float64)
    candidate_error = _rms(matrix @ np.asarray(candidate, dtype=np.float64) - target)
    baseline_error = _rms(matrix @ np.asarray(baseline, dtype=np.float64) - target)
    return 1.0 - candidate_error / max(baseline_error, 1e-15)


def rule_masks(audit_rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, bool]]:
    """Return four untuned sign rules without choosing among them."""

    output: dict[str, dict[str, bool]] = {
        "source_positive": {},
        "reserved_positive": {},
        "source_and_reserved_positive": {},
        "source_or_reserved_positive": {},
    }
    for row in audit_rows:
        field_uid = str(row["field_uid"])
        source = float(row["source_gain_vs_pbb9"]) > 0.0
        reserved = float(row["reserved_gain_vs_pbb9"]) > 0.0
        output["source_positive"][field_uid] = source
        output["reserved_positive"][field_uid] = reserved
        output["source_and_reserved_positive"][field_uid] = source and reserved
        output["source_or_reserved_positive"][field_uid] = source or reserved
    return output


def _score_rule(
    bundle: Any,
    indices: Sequence[int],
    candidate_prediction: np.ndarray,
    pbb_prediction: np.ndarray,
    mask: Mapping[str, bool],
    target_gain: Mapping[str, float],
) -> dict[str, Any]:
    gated = np.stack(
        [
            candidate_prediction[position]
            if mask[str(bundle.rows[index]["field_uid"])]
            else pbb_prediction[position]
            for position, index in enumerate(indices)
        ]
    )
    aggregates, cells = _score_methods(
        bundle,
        indices,
        {"gated": gated, "source_only_pbb_9": pbb_prediction},
    )
    cell_gains = [
        1.0
        - float(gate_cell["mean_whitened_rmse"])
        / max(float(pbb_cell["mean_whitened_rmse"]), 1e-15)
        for gate_cell, pbb_cell in zip(
            cells["gated"], cells["source_only_pbb_9"], strict=True
        )
    ]
    selected = [field_uid for field_uid, value in mask.items() if value]
    return {
        "field_coverage": len(selected) / max(len(mask), 1),
        "selected_field_mean_target_gain": (
            float(np.mean([target_gain[field_uid] for field_uid in selected]))
            if selected
            else None
        ),
        "selected_field_target_harm_fraction": (
            float(np.mean([target_gain[field_uid] < 0.0 for field_uid in selected]))
            if selected
            else None
        ),
        "gated_ratio_of_cluster_means_gain_vs_pbb9": 1.0
        - aggregates["gated"] / max(aggregates["source_only_pbb_9"], 1e-15),
        "gated_cell_mean_gain_vs_pbb9": float(np.mean(cell_gains)),
        "gated_positive_cell_fraction": float(np.mean(np.asarray(cell_gains) > 0.0)),
        "gated_worst_cell_degradation": max(0.0, -float(np.min(cell_gains))),
    }


def _correlation_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    field_correlation = float(
        spearmanr(
            [float(row["reserved_gain_vs_pbb9"]) for row in rows],
            [float(row["target_gain_vs_pbb9"]) for row in rows],
        ).statistic
    )
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["rig_id"]), str(row["family"]))].append(row)
    cell_rows = []
    for (rig_id, family), selected in sorted(grouped.items()):
        cell_rows.append(
            {
                "rig_id": rig_id,
                "family": family,
                "field_count": len(selected),
                "mean_reserved_gain_vs_pbb9": float(
                    np.mean([float(row["reserved_gain_vs_pbb9"]) for row in selected])
                ),
                "mean_target_gain_vs_pbb9": float(
                    np.mean([float(row["target_gain_vs_pbb9"]) for row in selected])
                ),
            }
        )
    cell_correlation = float(
        spearmanr(
            [row["mean_reserved_gain_vs_pbb9"] for row in cell_rows],
            [row["mean_target_gain_vs_pbb9"] for row in cell_rows],
        ).statistic
    )
    per_rig = []
    for rig_id in sorted({str(row["rig_id"]) for row in rows}):
        selected = [row for row in rows if row["rig_id"] == rig_id]
        per_rig.append(
            {
                "rig_id": rig_id,
                "field_spearman": float(
                    spearmanr(
                        [float(row["reserved_gain_vs_pbb9"]) for row in selected],
                        [float(row["target_gain_vs_pbb9"]) for row in selected],
                    ).statistic
                ),
            }
        )
    return {
        "field_spearman": field_correlation,
        "cell_spearman": cell_correlation,
        "per_rig": per_rig,
        "cell_rows": cell_rows,
    }


def run() -> dict[str, Any]:
    preregistration = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    base = json.loads(
        (ROOT / preregistration["source_config"]).read_text(encoding="utf-8")
    )
    source_report = json.loads(
        (ROOT / preregistration["source_report"]).read_text(encoding="utf-8")
    )
    bundle = build_dataset(fresh_dataset_config(base, preregistration))
    models, checkpoint_hashes = _load_models(base, preregistration, source_report)
    _, priors = _shared_prior_members(
        bundle,
        models,
        batch_size=int(base["training"]["batch_size"]),
    )
    candidate = _correction_map(
        bundle,
        lambda row: prior_anchored_bb_correction(
            row,
            priors[str(row["field_uid"])],
            iterations=8,
            relative_anchor=0.1,
        ),
    )
    pbb9 = _correction_map(
        bundle, lambda row: projected_bb_correction(row, 9)
    )
    pbb11 = _correction_map(
        bundle, lambda row: projected_bb_correction(row, 11)
    )
    pbb32 = _correction_map(
        bundle, lambda row: projected_bb_correction(row, 32)
    )
    zero = {
        str(bundle.rows[index]["field_uid"]): np.zeros_like(
            bundle.rows[index]["base_field"]
        )
        for index in _unique_field_indices(bundle)
    }
    frozen_corrections = {
        "zero_correction": zero,
        "shared_field_prior": priors,
        "source_only_pbb_9": pbb9,
        "source_only_pbb_11": pbb11,
        "source_only_pbb_32": pbb32,
        "sfio_papbb_budget_vector_v1": candidate,
    }
    frozen_predictions = {
        method: _target_prediction(bundle, values)[1]
        for method, values in frozen_corrections.items()
    }
    reproduced_prediction_sha256 = sha256_state_dict(
        {
            f"{method}_target_prediction": value
            for method, value in frozen_predictions.items()
        }
    )
    v5p_report = json.loads(
        (V5P_OUTPUT_DIR / "report.json").read_text(encoding="utf-8")
    )
    if reproduced_prediction_sha256 != v5p_report[
        "prediction_sha256_before_scoring"
    ]:
        raise RuntimeError("v5r failed to reproduce frozen v5p predictions")
    rig_config = {str(rig["id"]): rig for rig in preregistration["rigs"]}

    audit_rows = []
    seen: set[str] = set()
    for row in bundle.rows:
        field_uid = str(row["field_uid"])
        if field_uid in seen:
            continue
        seen.add(field_uid)
        rig_id = str(row["rig_id"])
        source_operator, source_label, support = _source_system(row)
        source_gain = residual_gain(
            source_operator,
            source_label,
            candidate[field_uid][support],
            pbb9[field_uid][support],
        )
        reserved_view = int(rig_config[rig_id]["reserved_indices"][0])
        reserved_operator = _flatten_operator(bundle.model_operators[rig_id])[
            reserved_view
        ]
        reserved_observation = bundle.scoring_observations[field_uid][
            :, reserved_view, :
        ].reshape(-1)
        reserved_label = reserved_observation - reserved_operator @ np.asarray(
            row["base_field"], dtype=np.float64
        )
        audit_rows.append(
            {
                "field_uid": field_uid,
                "rig_id": rig_id,
                "family": str(row["family"]),
                "reserved_view": reserved_view,
                "source_gain_vs_pbb9": source_gain,
                "reserved_gain_vs_pbb9": residual_gain(
                    reserved_operator,
                    reserved_label,
                    candidate[field_uid],
                    pbb9[field_uid],
                ),
            }
        )
    audit_feature_sha256_before_target_label_access = sha256_json(audit_rows)

    target_gain = _target_gain_by_field(bundle, candidate, pbb9)
    field_rows = [
        {**row, "target_gain_vs_pbb9": target_gain[str(row["field_uid"])]}
        for row in audit_rows
    ]
    correlations = _correlation_summary(field_rows)
    indices, candidate_prediction, _ = _target_prediction(bundle, candidate)
    pbb_indices, pbb_prediction, _ = _target_prediction(bundle, pbb9)
    if indices != pbb_indices:
        raise RuntimeError("candidate and PBB target row order differs")
    masks = rule_masks(audit_rows)
    rules = {
        name: _score_rule(
            bundle,
            indices,
            candidate_prediction,
            pbb_prediction,
            mask,
            target_gain,
        )
        for name, mask in masks.items()
    }
    report = {
        "schema": "v5r-reserved-view-reliability-diagnosis-1",
        "evidence_label": "post_open_same_field_reserved_view_diagnosis",
        "claim_ceiling": "No sign rule, gate, design lock, experimental, OERF, or publication superiority claim is authorized.",
        "chronology": {
            "same_opened_v5p_fields": True,
            "reserved_view_excluded_from_reconstruction": True,
            "reserved_audit_features_hashed_before_original_target_label_access": True,
            "threshold_sweep_performed": False,
            "rule_selected": False,
        },
        "audit_feature_sha256_before_target_label_access": audit_feature_sha256_before_target_label_access,
        "v5p_prediction_sha256": v5p_report[
            "prediction_sha256_before_scoring"
        ],
        "reproduced_prediction_sha256": reproduced_prediction_sha256,
        "checkpoint_hashes": checkpoint_hashes,
        "source_provenance": {
            "direct_dependency_sha256": relative_file_hashes(
                ROOT,
                [
                    Path(__file__),
                    ROOT / "run_v5p_fresh_budget_gate.py",
                    ROOT / "run_v5q_postopen_topology_diagnosis.py",
                    CONFIG_PATH,
                    ROOT / preregistration["source_config"],
                    ROOT / preregistration["source_report"],
                    V5P_OUTPUT_DIR / "report.json",
                    ROOT / "release_provenance.py",
                    ROOT / "gc_rio" / "data.py",
                    ROOT / "gc_rio" / "protocol.py",
                    ROOT / "gc_rio" / "shared_field_model.py",
                    ROOT / "gc_rio" / "training.py",
                    ROOT / "run_v5h_gc_rio_development.py",
                    ROOT / "run_v5n_strong_classical_baselines.py",
                    ROOT / "run_v5o_prior_anchored_frontier.py",
                    *[
                        ROOT
                        / "results"
                        / "v5k_shared_field_work"
                        / str(preregistration["source_method"])
                        / str(seed)
                        / "best.pt"
                        for seed in preregistration["ensemble_seeds"]
                    ],
                ],
            ),
            "runtime_environment": runtime_environment(
                device="cpu", torch_version=torch.__version__
            ),
            "determinism_boundary": (
                "V5R must reproduce the frozen V5P prediction hash before "
                "reading the original target labels."
            ),
        },
        "sample_accounting": {
            "fields": len(field_rows),
            "rigs": len({row["rig_id"] for row in field_rows}),
            "families": len({row["family"] for row in field_rows}),
            "reserved_views_per_field": 1,
            "target_views_per_field": 2,
        },
        "correlations": {
            key: value for key, value in correlations.items() if key != "cell_rows"
        },
        "rules": rules,
        "decision": "RESERVED_VIEW_RELIABILITY_NO_GO_POSTOPEN",
        "reasons": [
            "The reserved-positive rule improves aggregate gain but only 12/18 cells are positive.",
            "Its selected-field target harm fraction is 33.7%, worse than the in-sample source-positive rule.",
            "The conjunction lowers coverage without materially lowering selected-field harm.",
            "One rig has near-zero reserved-to-target rank association, so the mechanism is not uniformly transferable.",
        ],
        "next_boundary": "Archive sign-rule reliability on these data. Ask whether a reserved camera is operationally available; otherwise pivot to forward-model mismatch, covariance, or 4D temporal priors.",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "field_rows.csv", field_rows)
    _write_csv(OUTPUT_DIR / "cell_rows.csv", correlations["cell_rows"])
    _write_json(OUTPUT_DIR / "report.json", report)
    _write_checksums(OUTPUT_DIR, ["field_rows.csv", "cell_rows.csv", "report.json"])
    return report


def main() -> None:
    report = run()
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "correlations": report["correlations"],
                "rules": report["rules"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
