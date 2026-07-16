#!/usr/bin/env python3
"""Diagnose an exact active-view fallback around the opened PSU pilot."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import resource
import time
from typing import Any

import torch

from demo_t16_operator.psu_b0_spectral_preconditioner import (
    ActiveViewSupportEnvelopeDirection,
    FixedSobolevDirection,
    PositiveSpectralDirection,
)
from demo_t16_operator.psu_b0_streaming_operator import (
    zero_outer_boundary_support,
)
from site_tools.run_psu_b0_real_interface_audit import (
    _make_operator,
    load_real_support_geometry,
)
from site_tools.run_psu_b0_spectral_preconditioner_pilot import (
    SyntheticSplit,
    _aggregate,
    _build_split,
    _evaluate,
    _load_json,
    _sha256,
)


PRIVATE_SCHEMA = "psu-b0-support-envelope-postopen-private-report-1.0"
PUBLIC_SCHEMA = "psu-b0-support-envelope-postopen-public-summary-1.0"
STATUS = "POSTOPEN_SUPPORT_ENVELOPE_DIAGNOSIS_COMPLETE_NOT_FRESH"


def _max_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if platform.system() == "Darwin" else raw * 1024


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_checkpoint_model(
    *,
    checkpoint: Path,
    config: dict[str, Any],
    selected_strength: float,
    device: torch.device,
) -> tuple[int, PositiveSpectralDirection, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    seed = int(payload["seed"])
    if float(payload["base_sobolev_strength"]) != float(selected_strength):
        raise ValueError("checkpoint Sobolev strength does not match the summary")
    model = PositiveSpectralDirection(
        (int(config["geometry"]["grid_size"]),) * 3,
        view_count=int(config["geometry"]["view_count"]),
        hidden=int(config["model"]["hidden"]),
        embedding_width=int(config["model"]["view_embedding_width"]),
        maximum_log_gain=float(config["model"]["maximum_log_correction"]),
        base_sobolev_strength=float(selected_strength),
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return seed, model, {
        "filename": checkpoint.name,
        "sha256": _file_sha256(checkpoint),
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
    }


def _build_splits(
    *,
    config: dict[str, Any],
    true_operator: Any,
    nominal_operator: Any,
    device: torch.device,
) -> dict[str, SyntheticSplit]:
    splits: dict[str, SyntheticSplit] = {}
    train_validation_masks: set[str] = set()
    for name, spec in config["data"]["splits"].items():
        forbidden = (
            train_validation_masks
            if name in {"train", "validation"}
            else set(train_validation_masks)
        )
        split, used_masks = _build_split(
            name=name,
            spec=spec,
            config=config,
            true_operator=true_operator,
            nominal_operator=nominal_operator,
            device=device,
            forbidden_masks=forbidden,
        )
        splits[name] = split
        if name in {"train", "validation"}:
            train_validation_masks = used_masks
    return splits


def _paired_equivalence(
    rows: list[dict[str, Any]],
    *,
    left_method: str,
    right_method: str,
    split: str,
) -> dict[str, Any]:
    keys = (
        "field_relative_l2",
        "gradient_relative_l2",
        "front_top10_f1",
        "combined_loss",
        "measurement_relative_l2",
    )
    left = {
        row["sample_id"]: row
        for row in rows
        if row["split"] == split and row["method"] == left_method
    }
    right = {
        row["sample_id"]: row
        for row in rows
        if row["split"] == split and row["method"] == right_method
    }
    if set(left) != set(right) or not left:
        raise ValueError("paired methods do not cover the same nonempty samples")
    maximum = {
        key: float(
            max(
                abs(float(left[sample][key]) - float(right[sample][key]))
                for sample in left
            )
        )
        for key in keys
    }
    return {
        "split": split,
        "left_method": left_method,
        "right_method": right_method,
        "sample_count": len(left),
        "maximum_absolute_metric_difference": maximum,
        "maximum_over_metrics": float(max(maximum.values())),
    }


def build_public_summary(private: dict[str, Any]) -> dict[str, Any]:
    """Strip local paths, hashes, checkpoints, masks, and per-sample rows."""

    return {
        "schema_version": PUBLIC_SCHEMA,
        "status": private["status"],
        "evidence_scope": private["evidence_scope"],
        "configuration": copy.deepcopy(private["configuration_public"]),
        "dataset": copy.deepcopy(private["dataset_public"]),
        "aggregates": copy.deepcopy(private["aggregates"]),
        "equivalence_diagnostics": copy.deepcopy(
            private["equivalence_diagnostics"]
        ),
        "execution": copy.deepcopy(private["execution_public"]),
        "gates": copy.deepcopy(private["gates"]),
        "claim_boundary": copy.deepcopy(private["claim_boundary"]),
        "public_export_policy": {
            "contains_local_paths": False,
            "contains_checkpoint_hashes_or_weights": False,
            "contains_per_sample_metrics": False,
            "contains_ray_coordinates_or_masks": False,
            "contains_observations_or_volumes": False,
            "fresh_candidate_gate": False,
        },
    }


def run_diagnosis(
    *,
    config_path: Path,
    pilot_summary_path: Path,
    view_root: Path,
    checkpoint_dir: Path,
    device_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path)
    pilot_summary = _load_json(pilot_summary_path)
    device = torch.device(device_name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    torch.set_num_threads(8)
    selected_strength = float(
        pilot_summary["sobolev_selection"]["selected_strength"]
    )
    minimum_active = int(
        config["data"]["splits"]["train"]["active_view_range"][0]
    )
    maximum_active = int(
        config["data"]["splits"]["train"]["active_view_range"][1]
    )
    grid_size = int(config["geometry"]["grid_size"])
    rays_per_view = int(config["geometry"]["rays_per_view"])

    started = time.perf_counter()
    support = zero_outer_boundary_support(
        (grid_size,) * 3,
        dtype=torch.float32,
    ).to(device)
    true_geometry, true_provenance = load_real_support_geometry(
        view_root,
        rays_per_view=rays_per_view,
        sample_count=int(config["geometry"]["true_finite_aperture_sample_count"]),
    )
    nominal_geometry, nominal_provenance = load_real_support_geometry(
        view_root,
        rays_per_view=rays_per_view,
        sample_count=int(
            config["geometry"]["nominal_finite_aperture_sample_count"]
        ),
    )
    true_operator = _make_operator(
        true_geometry,
        grid_size=grid_size,
        dtype=torch.float32,
    ).to(device)
    nominal_operator = _make_operator(
        nominal_geometry,
        grid_size=grid_size,
        dtype=torch.float32,
    ).to(device)
    true_operator.support.copy_(support)
    nominal_operator.support.copy_(support)
    splits = _build_splits(
        config=config,
        true_operator=true_operator,
        nominal_operator=nominal_operator,
        device=device,
    )

    fallback = FixedSobolevDirection(
        (grid_size,) * 3,
        strength=selected_strength,
    ).to(device)
    checkpoints = sorted(checkpoint_dir.glob("learned_seed_*.pt"))
    expected_seeds = sorted(int(value) for value in config["training"]["seeds"])
    if len(checkpoints) != len(expected_seeds):
        raise ValueError("checkpoint count does not match the frozen seed list")
    models: dict[int, PositiveSpectralDirection] = {}
    checkpoint_records = []
    for checkpoint in checkpoints:
        seed, model, record = _load_checkpoint_model(
            checkpoint=checkpoint,
            config=config,
            selected_strength=selected_strength,
            device=device,
        )
        models[seed] = model
        checkpoint_records.append({"seed": seed, **record})
    if sorted(models) != expected_seeds:
        raise ValueError("checkpoint seeds do not match the frozen seed list")

    rows: list[dict[str, Any]] = []
    execution = []
    audit_splits = [
        name
        for name in splits
        if name.startswith("test_")
    ]
    for split_name in audit_splits:
        baseline_rows, baseline_execution = _evaluate(
            method="sobolev_selected",
            split=splits[split_name],
            operator=nominal_operator,
            config=config,
            device=device,
            direction=fallback,
        )
        rows.extend(baseline_rows)
        execution.append(baseline_execution)
        for seed, model in models.items():
            raw_method = f"raw_seed_{seed}"
            raw_rows, raw_execution = _evaluate(
                method=raw_method,
                split=splits[split_name],
                operator=nominal_operator,
                config=config,
                device=device,
                direction=model,
            )
            rows.extend(raw_rows)
            execution.append(raw_execution)
            envelope = ActiveViewSupportEnvelopeDirection(
                candidate=model,
                fallback=fallback,
                minimum_active_views=minimum_active,
                maximum_active_views=maximum_active,
            )
            enveloped_rows, enveloped_execution = _evaluate(
                method=f"enveloped_seed_{seed}",
                split=splits[split_name],
                operator=nominal_operator,
                config=config,
                device=device,
                direction=envelope,
            )
            rows.extend(enveloped_rows)
            execution.append(enveloped_execution)

    aggregates = _aggregate(rows, baseline_method="sobolev_selected")
    equivalence = []
    inside_splits = []
    outside_splits = []
    coverage_rows = []
    for split_name in audit_splits:
        split = splits[split_name]
        active_count = torch.sum(split.view_mask > 0.5, dim=1)
        trust = (
            (active_count >= minimum_active)
            & (active_count <= maximum_active)
        )
        coverage = float(torch.mean(trust.to(torch.float32)))
        coverage_rows.append(
            {
                "split": split_name,
                "sample_count": len(split.truth),
                "support_envelope_coverage": coverage,
                "active_view_count_minimum": int(active_count.min()),
                "active_view_count_maximum": int(active_count.max()),
            }
        )
        if coverage == 1.0:
            inside_splits.append(split_name)
        elif coverage == 0.0:
            outside_splits.append(split_name)
        else:
            raise ValueError("the frozen diagnosis expects split-level coverage")
        for seed in expected_seeds:
            equivalence.append(
                _paired_equivalence(
                    rows,
                    left_method=f"enveloped_seed_{seed}",
                    right_method=(
                        f"raw_seed_{seed}"
                        if coverage == 1.0
                        else "sobolev_selected"
                    ),
                    split=split_name,
                )
            )

    continuous_tolerance = 1e-6
    front_f1_tolerance = 5e-4

    def metric_equivalent(row: dict[str, Any]) -> bool:
        differences = row["maximum_absolute_metric_difference"]
        return (
            float(differences["field_relative_l2"]) <= continuous_tolerance
            and float(differences["gradient_relative_l2"])
            <= continuous_tolerance
            and float(differences["combined_loss"]) <= continuous_tolerance
            and float(differences["measurement_relative_l2"])
            <= continuous_tolerance
            and float(differences["front_top10_f1"]) <= front_f1_tolerance
        )

    inside_equivalence = all(
        metric_equivalent(row)
        for row in equivalence
        if row["split"] in inside_splits
    )
    outside_equivalence = all(
        metric_equivalent(row)
        for row in equivalence
        if row["split"] in outside_splits
    )
    call_match = all(
        row["logical_calls_per_sample"]
        == {
            "forward": int(config["solver"]["stages"]),
            "adjoint": int(config["solver"]["stages"]),
        }
        for row in execution
        if row["method"].startswith(("raw_seed_", "enveloped_seed_"))
    )
    monotone = all(
        row["data_objective_monotone"]
        for row in execution
    )
    gates = {
        "checkpoint_seed_set_matches_frozen_pilot": sorted(models)
        == expected_seeds,
        "inside_support_matches_raw_candidate": inside_equivalence,
        "outside_support_matches_sobolev_fallback": outside_equivalence,
        "logical_forward_adjoint_calls_match": call_match,
        "line_search_data_objective_monotone": monotone,
        "opened_audit_values_not_reclassified_as_fresh": True,
        "development_rotation_40_not_accessed": True,
        "final_audit_not_accessed": True,
    }
    private = {
        "schema_version": PRIVATE_SCHEMA,
        "status": STATUS,
        "evidence_scope": (
            "POSTOPEN_MECHANISM_DIAGNOSIS_ON_PREVIOUSLY_OPENED_SYNTHETIC_"
            "AUDITS_WITH_REAL_PSU_SUPPORT_GEOMETRY"
        ),
        "configuration_private": {
            "config_path": str(config_path.resolve()),
            "config_sha256": _sha256(config_path),
            "pilot_summary_path": str(pilot_summary_path.resolve()),
            "pilot_summary_sha256": _sha256(pilot_summary_path),
            "view_root": str(view_root.resolve()),
            "checkpoint_dir": str(checkpoint_dir.resolve()),
            "device": device_name,
        },
        "configuration_public": {
            "grid_shape_zyx": [grid_size, grid_size, grid_size],
            "view_count": int(config["geometry"]["view_count"]),
            "rays_per_view": rays_per_view,
            "stages": int(config["solver"]["stages"]),
            "selected_sobolev_strength": selected_strength,
            "minimum_active_views": minimum_active,
            "maximum_active_views": maximum_active,
            "seed_count": len(expected_seeds),
            "parameter_count_per_candidate": checkpoint_records[0][
                "parameter_count"
            ],
            "logical_calls_per_sample": {
                "forward": int(config["solver"]["stages"]),
                "adjoint": int(config["solver"]["stages"]),
            },
            "equivalence_tolerances": {
                "continuous_metrics_absolute": continuous_tolerance,
                "front_top10_f1_absolute": front_f1_tolerance,
            },
        },
        "dataset_private": {
            "true_geometry_provenance": true_provenance,
            "nominal_geometry_provenance": nominal_provenance,
            "checkpoint_records": checkpoint_records,
            "per_sample_metrics": rows,
        },
        "dataset_public": {
            "source_dataset_doi": "10.26208/1VE2-5C19",
            "real_psu_measurement_values_used": False,
            "analytic_truth_is_cfd": False,
            "audit_values_previously_opened": True,
            "split_support_coverage": coverage_rows,
        },
        "aggregates": aggregates,
        "equivalence_diagnostics": equivalence,
        "execution_private": {
            "wall_seconds": float(time.perf_counter() - started),
            "process_max_rss_bytes": int(_max_rss_bytes()),
            "evaluation_records": execution,
        },
        "execution_public": {
            "wall_seconds": float(time.perf_counter() - started),
            "process_max_rss_bytes": int(_max_rss_bytes()),
            "inside_support_splits": inside_splits,
            "outside_support_splits": outside_splits,
        },
        "gates": gates,
        "claim_boundary": {
            "fresh_candidate_gate": False,
            "algorithm_superiority": False,
            "experimental_field_truth": False,
            "view_count_only_is_sufficient_ood_detection": False,
            "exact_declared_fallback_mechanism_verified": (
                inside_equivalence and outside_equivalence
            ),
        },
    }
    return private, build_public_summary(private)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pilot-summary", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--public-output", type=Path)
    args = parser.parse_args()
    private, public = run_diagnosis(
        config_path=args.config,
        pilot_summary_path=args.pilot_summary,
        view_root=args.view_root,
        checkpoint_dir=args.checkpoint_dir,
        device_name=args.device,
    )
    if args.private_output is not None:
        args.private_output.parent.mkdir(parents=True, exist_ok=True)
        args.private_output.write_text(
            json.dumps(private, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    if args.public_output is not None:
        args.public_output.parent.mkdir(parents=True, exist_ok=True)
        args.public_output.write_text(
            json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
