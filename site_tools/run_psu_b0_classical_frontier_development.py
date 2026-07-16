#!/usr/bin/env python3
"""Build a call-accounted PSU B0 classical frontier after the OCRRG audit."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from demo_t16_operator.psu_b0_classical_baselines import (
    preconditioned_cgls_reconstruction,
    quadratic_tikhonov_reconstruction,
)
from demo_t16_operator.psu_b0_spectral_preconditioner import (
    exact_line_search_reconstruction,
    normalized_field_loss,
    weighted_cgls_reconstruction,
)
from demo_t16_operator.psu_b0_streaming_operator import (
    zero_outer_boundary_support,
)
from site_tools.run_psu_b0_real_interface_audit import (
    _make_operator,
    load_real_support_geometry,
)
from site_tools.run_psu_b0_residual_risk_development import (
    DevelopmentSplit,
    _build_development_split,
    _synchronize,
)
from site_tools.run_psu_b0_residual_risk_fresh import _build_fresh_splits
from site_tools.run_psu_b0_spectral_preconditioner_pilot import (
    _aggregate,
    _expanded_measurement_values,
    _field_metrics,
    _load_json,
    _unique_masks,
)


PRIVATE_SCHEMA = "psu-b0-classical-frontier-development-private-1.0"
PUBLIC_SCHEMA = "psu-b0-classical-frontier-development-public-1.0"
STATUS = "POSTOPEN_CLASSICAL_FRONTIER_DEVELOPMENT_ONLY_NO_SUPERIORITY_CLAIM"


def select_regularization(
    screen: list[dict[str, Any]],
    *,
    regularizer: str,
) -> dict[str, Any]:
    rows = [row for row in screen if row["regularizer"] == regularizer]
    if not rows:
        raise ValueError(f"no selection rows for regularizer {regularizer}")
    return dict(
        min(
            rows,
            key=lambda row: (
                float(row["mean_combined_loss"]),
                float(row["mean_field_relative_l2"]),
                float(row["regularization_lambda"]),
            ),
        )
    )


def _verify_split_metadata(
    wrapped: DevelopmentSplit,
    source_rows: list[dict[str, Any]],
) -> None:
    baseline = {
        str(row["sample_id"]): row
        for row in source_rows
        if row["method"] == "sobolev_selected"
        and row["split"] == wrapped.data.name
    }
    if set(baseline) != set(wrapped.data.sample_ids):
        raise ValueError(f"regenerated sample identifiers drifted for {wrapped.data.name}")
    for index, sample_id in enumerate(wrapped.data.sample_ids):
        row = baseline[str(sample_id)]
        checks = (
            str(row["family"]) == str(wrapped.data.families[index]),
            str(row["truth_operator"]) == str(wrapped.data.truth_operator),
            int(row["active_view_count"])
            == int(torch.sum(wrapped.data.view_mask[index] > 0.5)),
            abs(
                float(row["relative_noise"])
                - float(wrapped.data.relative_noise[index])
            )
            <= 1e-7,
        )
        if not all(checks):
            raise ValueError(f"regenerated metadata drifted for {sample_id}")


def _evaluate_solver(
    *,
    method: str,
    wrapped: DevelopmentSplit,
    operator: Any,
    source_config: dict[str, Any],
    device: torch.device,
    solver: str,
    stages: int,
    regularizer: str | None = None,
    regularization_lambda: float | None = None,
    direction: Any | None = None,
    batch_size: int = 12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split = wrapped.data
    rays_per_view = int(source_config["geometry"]["rays_per_view"])
    rows: list[dict[str, Any]] = []
    monotone = True
    started = time.perf_counter()
    operator.reset_call_counts()
    for start in range(0, len(split.truth), int(batch_size)):
        stop = min(start + int(batch_size), len(split.truth))
        truth = split.truth[start:stop].to(device)
        observation = split.observation_uv[start:stop].to(device)
        sigma = split.sigma_by_view[start:stop].to(device)
        mask = split.view_mask[start:stop].to(device)
        with torch.no_grad():
            if solver == "cgls":
                result = weighted_cgls_reconstruction(
                    operator,
                    observation,
                    sigma_by_view=sigma,
                    view_mask=mask,
                    rays_per_view=rays_per_view,
                    stages=int(stages),
                )
            elif solver == "tikhonov":
                if regularizer is None or regularization_lambda is None:
                    raise ValueError("Tikhonov evaluation needs a regularizer and lambda")
                result = quadratic_tikhonov_reconstruction(
                    operator,
                    observation,
                    sigma_by_view=sigma,
                    view_mask=mask,
                    rays_per_view=rays_per_view,
                    stages=int(stages),
                    regularizer=regularizer,  # type: ignore[arg-type]
                    regularization_lambda=float(regularization_lambda),
                )
            elif solver == "line_search":
                if direction is None:
                    raise ValueError("line-search evaluation needs a direction")
                result = exact_line_search_reconstruction(
                    operator,
                    observation,
                    sigma_by_view=sigma,
                    view_mask=mask,
                    rays_per_view=rays_per_view,
                    stages=int(stages),
                    direction=direction,
                )
            elif solver == "pcgls":
                if direction is None:
                    raise ValueError("PCGLS evaluation needs a preconditioner")
                result = preconditioned_cgls_reconstruction(
                    operator,
                    observation,
                    sigma_by_view=sigma,
                    view_mask=mask,
                    rays_per_view=rays_per_view,
                    stages=int(stages),
                    preconditioner=direction,
                )
            else:
                raise ValueError(
                    "solver must be cgls, tikhonov, line_search, or pcgls"
                )
        metrics = _field_metrics(result.volume, truth)
        combined = normalized_field_loss(
            result.volume,
            truth,
            gradient_weight=float(source_config["training"]["gradient_weight"]),
        )
        active = _expanded_measurement_values(mask, rays_per_view=rays_per_view)
        expanded_sigma = _expanded_measurement_values(
            sigma,
            rays_per_view=rays_per_view,
        )
        measurement = torch.linalg.vector_norm(
            (result.residual_uv / expanded_sigma).flatten(1),
            dim=1,
        ) / torch.linalg.vector_norm(
            (active * observation / expanded_sigma).flatten(1),
            dim=1,
        ).clamp_min(1e-12)
        for history in result.history:
            if "relative_total_objective_before" in history:
                monotone &= bool(
                    torch.all(
                        history["relative_total_objective_after"]
                        <= history["relative_total_objective_before"] + 2e-5
                    )
                )
            elif "relative_objective_before" in history:
                monotone &= bool(
                    torch.all(
                        history["relative_objective_after"]
                        <= history["relative_objective_before"] + 2e-5
                    )
                )
        for offset, index in enumerate(range(start, stop)):
            rows.append(
                {
                    "sample_id": split.sample_ids[index],
                    "split": split.name,
                    "family": split.families[index],
                    "noise_profile": wrapped.noise_profiles[index],
                    "truth_operator": split.truth_operator,
                    "relative_noise": float(split.relative_noise[index]),
                    "active_view_count": int(
                        torch.sum(split.view_mask[index] > 0.5)
                    ),
                    "operator_mismatch_relative_l2": float(
                        split.operator_mismatch_relative_l2[index]
                    ),
                    "method": method,
                    "field_relative_l2": float(
                        metrics["field_relative_l2"][offset]
                    ),
                    "gradient_relative_l2": float(
                        metrics["gradient_relative_l2"][offset]
                    ),
                    "front_top10_f1": float(metrics["front_top10_f1"][offset]),
                    "combined_loss": float(combined[offset]),
                    "measurement_relative_l2": float(measurement[offset]),
                }
            )
    _synchronize(device)
    calls = operator.call_report()
    expected_adjoint = int(stages) + int(solver == "cgls")
    return rows, {
        "method": method,
        "split": split.name,
        "solver": solver,
        "stages": int(stages),
        "regularizer": regularizer,
        "regularization_lambda": regularization_lambda,
        "wall_seconds": float(time.perf_counter() - started),
        "batch_invocations": {
            "forward": int(calls["forward_calls"]),
            "adjoint": int(calls["adjoint_calls"]),
        },
        "logical_calls_per_sample": {
            "forward": int(stages),
            "adjoint": expected_adjoint,
        },
        "regularized_or_data_objective_monotone": bool(monotone),
    }


def _screen_record(
    rows: list[dict[str, Any]],
    *,
    regularizer: str,
    regularization_lambda: float,
) -> dict[str, Any]:
    return {
        "regularizer": regularizer,
        "regularization_lambda": float(regularization_lambda),
        "sample_count": len(rows),
        "mean_combined_loss": float(
            np.mean([row["combined_loss"] for row in rows])
        ),
        "mean_field_relative_l2": float(
            np.mean([row["field_relative_l2"] for row in rows])
        ),
        "mean_gradient_relative_l2": float(
            np.mean([row["gradient_relative_l2"] for row in rows])
        ),
        "mean_front_top10_f1": float(
            np.mean([row["front_top10_f1"] for row in rows])
        ),
        "mean_measurement_relative_l2": float(
            np.mean([row["measurement_relative_l2"] for row in rows])
        ),
    }


def _stratified_aggregates(
    rows: list[dict[str, Any]],
    *,
    baseline_method: str = "sobolev_selected",
) -> list[dict[str, Any]]:
    baseline = {
        (str(row["split"]), str(row["sample_id"])): float(
            row["field_relative_l2"]
        )
        for row in rows
        if row["method"] == baseline_method
    }
    output = []
    for key_name in ("active_view_count", "family", "noise_profile"):
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for row in rows:
            if key_name not in row:
                continue
            key = (
                str(row["split"]),
                str(row["method"]),
                str(row[key_name]),
            )
            groups.setdefault(key, []).append(row)
        for (split, method, value), group in sorted(groups.items()):
            gain = np.asarray(
                [
                    100.0
                    * (
                        baseline[(split, str(row["sample_id"]))]
                        - float(row["field_relative_l2"])
                    )
                    / max(
                        baseline[(split, str(row["sample_id"]))],
                        1e-12,
                    )
                    for row in group
                ],
                dtype=np.float64,
            )
            output.append(
                {
                    "split": split,
                    "method": method,
                    "stratum": key_name,
                    "value": value,
                    "sample_count": len(group),
                    "field_gain_vs_sobolev_mean_percent": float(np.mean(gain)),
                    "field_gain_vs_sobolev_p10_percent": float(
                        np.quantile(gain, 0.10)
                    ),
                    "field_harm_over_one_percent_rate": float(
                        np.mean(gain < -1.0)
                    ),
                }
            )
    return output


def _frontier_comparison(
    aggregates: list[dict[str, Any]],
    *,
    classical_methods: set[str],
) -> list[dict[str, Any]]:
    by_split: dict[str, list[dict[str, Any]]] = {}
    for row in aggregates:
        by_split.setdefault(str(row["split"]), []).append(row)
    output = []
    for split, rows in sorted(by_split.items()):
        classical = [row for row in rows if row["method"] in classical_methods]
        learned = [
            row
            for row in rows
            if str(row["method"]).startswith(("raw_seed_", "gated_seed_"))
        ]
        if not classical or not learned:
            continue
        best_classical = min(
            classical,
            key=lambda row: float(row["field_relative_l2_mean"]),
        )
        learned_rows = []
        for row in learned:
            learned_rows.append(
                {
                    "method": row["method"],
                    "field_relative_l2_mean": row["field_relative_l2_mean"],
                    "relative_field_error_vs_best_classical_percent": (
                        100.0
                        * (
                            float(row["field_relative_l2_mean"])
                            - float(best_classical["field_relative_l2_mean"])
                        )
                        / max(
                            float(best_classical["field_relative_l2_mean"]),
                            1e-12,
                        )
                    ),
                    "beats_best_classical_mean_field_error": (
                        float(row["field_relative_l2_mean"])
                        < float(best_classical["field_relative_l2_mean"])
                    ),
                }
            )
        output.append(
            {
                "split": split,
                "best_classical_method": best_classical["method"],
                "best_classical_field_relative_l2_mean": best_classical[
                    "field_relative_l2_mean"
                ],
                "learned_comparison": learned_rows,
            }
        )
    return output


def build_public_summary(private: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PUBLIC_SCHEMA,
        "status": private["status"],
        "evidence_scope": private["evidence_scope"],
        "configuration": copy.deepcopy(private["configuration_public"]),
        "regeneration_checks": copy.deepcopy(private["regeneration_checks"]),
        "selection_screen": copy.deepcopy(private["selection_screen"]),
        "selected_regularization": copy.deepcopy(
            private["selected_regularization"]
        ),
        "aggregates": copy.deepcopy(private["aggregates"]),
        "stratified_aggregates": copy.deepcopy(
            private["stratified_aggregates"]
        ),
        "frontier_comparison": copy.deepcopy(private["frontier_comparison"]),
        "execution": copy.deepcopy(private["execution"]),
        "claim_boundary": copy.deepcopy(private["claim_boundary"]),
    }


def run_frontier(
    *,
    root: Path,
    config_path: Path,
    development_report_path: Path,
    fresh_report_path: Path,
    view_root: Path,
    device_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path)
    development_config = _load_json(
        root / str(config["source_development_config"])
    )
    preregistration = _load_json(
        root / str(config["source_fresh_preregistration"])
    )
    source_config = _load_json(
        root / str(development_config["source_pilot"]["config"])
    )
    development_report = _load_json(development_report_path)
    fresh_report = _load_json(fresh_report_path)
    device = torch.device(device_name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    torch.set_num_threads(8)
    started = time.perf_counter()

    geometry = development_config["geometry"]
    grid_size = int(geometry["grid_size"])
    rays_per_view = int(geometry["rays_per_view"])
    support = zero_outer_boundary_support((grid_size,) * 3).to(device)
    true_geometry, _ = load_real_support_geometry(
        view_root,
        rays_per_view=rays_per_view,
        sample_count=int(geometry["true_finite_aperture_sample_count"]),
    )
    nominal_geometry, _ = load_real_support_geometry(
        view_root,
        rays_per_view=rays_per_view,
        sample_count=int(geometry["nominal_finite_aperture_sample_count"]),
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

    active_range = tuple(int(value) for value in geometry["active_view_range"])
    used_masks: set[str] = set()
    train_spec = development_config["development_splits"]["risk_train"]
    _, used_masks = _unique_masks(
        count=int(train_spec["count"]),
        view_count=int(geometry["view_count"]),
        minimum_active=active_range[0],
        maximum_active=active_range[1],
        seed=int(train_spec["mask_seed"]),
        forbidden=used_masks,
    )
    development_splits: dict[str, DevelopmentSplit] = {}
    for split_name in ("risk_validation", "risk_calibration"):
        wrapped, used_masks = _build_development_split(
            name=split_name,
            spec=development_config["development_splits"][split_name],
            config=development_config,
            source_config=source_config,
            true_operator=true_operator,
            nominal_operator=nominal_operator,
            device=device,
            forbidden_masks=used_masks,
        )
        development_splits[split_name] = wrapped
    fresh_splits, _, _ = _build_fresh_splits(
        preregistration=preregistration,
        development_config=development_config,
        source_config=source_config,
        true_operator=true_operator,
        nominal_operator=nominal_operator,
        device=device,
    )

    development_rows = development_report["dataset_private"]["metric_rows"]
    fresh_rows = fresh_report["dataset_private"]["per_sample_metrics"]
    for wrapped in development_splits.values():
        _verify_split_metadata(wrapped, development_rows)
    for wrapped in fresh_splits.values():
        _verify_split_metadata(wrapped, fresh_rows)

    screen = []
    execution = []
    validation = development_splits[str(config["selection_split"])]
    stages = int(config["solver"]["tikhonov_stages"])
    for regularizer, grid_key in (
        ("identity", "l2_lambda_grid"),
        ("h1", "h1_lambda_grid"),
    ):
        for value in config["solver"][grid_key]:
            method = f"screen_{regularizer}_{float(value):.8g}"
            rows, ledger = _evaluate_solver(
                method=method,
                wrapped=validation,
                operator=nominal_operator,
                source_config=source_config,
                device=device,
                solver="tikhonov",
                stages=stages,
                regularizer=regularizer,
                regularization_lambda=float(value),
            )
            screen.append(
                _screen_record(
                    rows,
                    regularizer=regularizer,
                    regularization_lambda=float(value),
                )
            )
            execution.append(ledger)
    selected = {
        regularizer: select_regularization(screen, regularizer=regularizer)
        for regularizer in ("identity", "h1")
    }

    generated_rows: list[dict[str, Any]] = []
    all_splits = {**development_splits, **fresh_splits}
    for split_name, wrapped in all_splits.items():
        for regularizer in ("identity", "h1"):
            value = float(selected[regularizer]["regularization_lambda"])
            method = f"tikhonov_{regularizer}_selected"
            rows, ledger = _evaluate_solver(
                method=method,
                wrapped=wrapped,
                operator=nominal_operator,
                source_config=source_config,
                device=device,
                solver="tikhonov",
                stages=stages,
                regularizer=regularizer,
                regularization_lambda=value,
            )
            generated_rows.extend(rows)
            execution.append(ledger)
        for cgls_stages in config["solver"]["cgls_stage_counts"]:
            method = f"cgls_{int(cgls_stages)}"
            rows, ledger = _evaluate_solver(
                method=method,
                wrapped=wrapped,
                operator=nominal_operator,
                source_config=source_config,
                device=device,
                solver="cgls",
                stages=int(cgls_stages),
            )
            generated_rows.extend(rows)
            execution.append(ledger)

    retained_old_rows = [
        dict(row)
        for row in development_rows
        if row["split"] in development_splits
    ] + [dict(row) for row in fresh_rows]
    combined_rows = retained_old_rows + generated_rows
    aggregates = _aggregate(
        combined_rows,
        baseline_method="sobolev_selected",
    )
    selected_methods = {
        "sobolev_selected",
        "tikhonov_identity_selected",
        "tikhonov_h1_selected",
        "cgls_3",
        "cgls_4",
    }
    stratified = _stratified_aggregates(
        [
            row
            for row in combined_rows
            if row["method"] in selected_methods
            or str(row["method"]).startswith(("raw_seed_", "gated_seed_"))
        ]
    )
    frontier = _frontier_comparison(
        aggregates,
        classical_methods=selected_methods,
    )
    calls_valid = all(
        bool(row["regularized_or_data_objective_monotone"])
        and row["logical_calls_per_sample"]
        in (
            {"forward": 4, "adjoint": 4},
            {"forward": 3, "adjoint": 4},
            {"forward": 4, "adjoint": 5},
        )
        for row in execution
    )
    private = {
        "schema_version": PRIVATE_SCHEMA,
        "status": STATUS,
        "evidence_scope": (
            "REAL_PSU_SUPPORT_GEOMETRY_WITH_ANALYTIC_REACTION_MORPHOLOGY_"
            "AND_SYNTHETIC_CAMERA_NOISE_POSTOPEN_CLASSICAL_DEVELOPMENT"
        ),
        "configuration_private": {
            "root": str(root.resolve()),
            "config_path": str(config_path.resolve()),
            "development_report_path": str(development_report_path.resolve()),
            "fresh_report_path": str(fresh_report_path.resolve()),
            "view_root": str(view_root.resolve()),
            "device": device_name,
        },
        "configuration_public": copy.deepcopy(config),
        "regeneration_checks": {
            "development_metadata_matches_frozen_rows": True,
            "fresh_metadata_matches_frozen_rows": True,
            "selection_uses_only_risk_validation": True,
            "risk_calibration_not_used_for_selection": True,
            "fresh_values_were_already_open": True,
            "operator_call_ledgers_and_objective_checks_pass": calls_valid,
        },
        "selection_screen": screen,
        "selected_regularization": selected,
        "metric_rows_private": combined_rows,
        "aggregates": aggregates,
        "stratified_aggregates": stratified,
        "frontier_comparison": frontier,
        "execution": execution,
        "runtime": {
            "wall_seconds": float(time.perf_counter() - started),
        },
        "claim_boundary": {
            "postopen_development_only": True,
            "fresh_diagnostic_is_confirmatory": False,
            "tv_baseline_included": False,
            "hybrid_parameter_choice_included": False,
            "experimental_field_truth_used": False,
            "real_psu_measurement_values_used": False,
            "analytic_morphology_is_cfd": False,
            "algorithm_superiority": False,
        },
    }
    return private, build_public_summary(private)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "demo_t16_operator/configs/"
            "psu_b0_classical_frontier_development_v1.json"
        ),
    )
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--fresh-report", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--public-output", type=Path)
    args = parser.parse_args()
    private, public = run_frontier(
        root=args.root,
        config_path=args.config,
        development_report_path=args.development_report,
        fresh_report_path=args.fresh_report,
        view_root=args.view_root,
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
