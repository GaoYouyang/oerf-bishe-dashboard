#!/usr/bin/env python3
"""Develop a gain-regressed observable spectral-expert PCGLS mixture."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import platform
import resource
import time
from typing import Any

import numpy as np
import torch

from demo_t16_operator.psu_b0_morphology_spectral_experts import (
    MORPHOLOGY_EXPERT_SCHEMA,
    ObservableMorphologyExpertFactory,
)
from demo_t16_operator.psu_b0_streaming_operator import (
    zero_outer_boundary_support,
)
from site_tools.run_psu_b0_conditioned_pcgls_development import (
    paired_gain_summary,
)
from site_tools.run_psu_b0_classical_frontier_development import (
    _verify_split_metadata,
)
from site_tools.run_psu_b0_observable_morphology_probe import (
    fit_ridge_multioutput,
    score_predictions,
)
from site_tools.run_psu_b0_omse_pcgls_development import (
    _best_diagnostic,
    _development_gate,
    _evaluate_integrated_factory,
    _evaluate_mixture_scores,
    _expert_bank,
    _expert_gain_targets,
    _oof_regression_scores,
    _paired_screen_metrics,
    select_omse_screen_candidate,
)
from site_tools.run_psu_b0_real_interface_audit import (
    _make_operator,
    load_real_support_geometry,
)
from site_tools.run_psu_b0_residual_risk_development import (
    DevelopmentSplit,
    _build_development_split,
)
from site_tools.run_psu_b0_spectral_preconditioner_pilot import (
    _load_json,
)


PRIVATE_SCHEMA = "psu-b0-ogse-pcgls-development-private-2.0"
PUBLIC_SCHEMA = "psu-b0-ogse-pcgls-development-public-2.0"
STATUS = "OGSE_PCGLS_DEVELOPMENT_COMPLETE_FRESH_NOT_USED"


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def greedy_oracle_expert_bank(
    *,
    sample_ids: list[str],
    candidate_rows: list[dict[str, Any]],
    baseline_candidate_id: str,
    bank_size: int,
) -> dict[str, Any]:
    """Greedily maximize train-only per-sample oracle field improvement."""

    if int(bank_size) < 1:
        raise ValueError("bank_size must be positive")
    candidate_ids = sorted(
        {str(row["candidate_id"]) for row in candidate_rows}
    )
    if baseline_candidate_id not in candidate_ids:
        raise ValueError("baseline candidate is absent")
    if int(bank_size) > len(candidate_ids):
        raise ValueError("bank_size exceeds the candidate family")
    lookup = {
        (str(row["sample_id"]), str(row["candidate_id"])): float(
            row["field_relative_l2"]
        )
        for row in candidate_rows
    }
    baseline = np.asarray(
        [
            lookup[(sample_id, baseline_candidate_id)]
            for sample_id in sample_ids
        ],
        dtype=np.float64,
    )
    selected = [str(baseline_candidate_id)]
    current = baseline.copy()
    trajectory = []
    for size in range(1, int(bank_size) + 1):
        gain = 100.0 * (baseline - current) / np.maximum(baseline, 1e-12)
        trajectory.append(
            {
                "bank_size": size,
                "added_candidate_id": selected[-1],
                "mean_oracle_field_gain_percent": float(np.mean(gain)),
                "p10_oracle_field_gain_percent": float(
                    np.quantile(gain, 0.10)
                ),
                "minimum_oracle_field_gain_percent": float(np.min(gain)),
            }
        )
        if size == int(bank_size):
            break
        best = None
        for candidate_id in candidate_ids:
            if candidate_id in selected:
                continue
            candidate = np.asarray(
                [
                    lookup[(sample_id, candidate_id)]
                    for sample_id in sample_ids
                ],
                dtype=np.float64,
            )
            updated = np.minimum(current, candidate)
            candidate_gain = (
                100.0
                * (baseline - updated)
                / np.maximum(baseline, 1e-12)
            )
            key = (
                float(np.mean(candidate_gain)),
                float(np.quantile(candidate_gain, 0.10)),
                candidate_id,
            )
            if best is None or key > best[0]:
                best = (key, candidate_id, updated)
        if best is None:
            raise RuntimeError("greedy expert-bank selection stalled")
        selected.append(best[1])
        current = best[2]
    return {
        "candidate_ids": selected,
        "trajectory": trajectory,
    }


def build_public_summary(private: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PUBLIC_SCHEMA,
        "algorithm_schema": MORPHOLOGY_EXPERT_SCHEMA,
        "status": private["status"],
        "evidence_scope": private["evidence_scope"],
        "configuration": copy.deepcopy(private["configuration_public"]),
        "regeneration_checks": copy.deepcopy(
            private["regeneration_checks"]
        ),
        "expert_banks": copy.deepcopy(private["expert_banks"]),
        "selection_screen": copy.deepcopy(private["selection_screen"]),
        "selected_strict_candidate": copy.deepcopy(
            private["selected_strict_candidate"]
        ),
        "best_diagnostic_candidate": copy.deepcopy(
            private["best_diagnostic_candidate"]
        ),
        "paired_gain_summary": copy.deepcopy(
            private["paired_gain_summary"]
        ),
        "development_gates": copy.deepcopy(
            private["development_gates"]
        ),
        "execution_summary": copy.deepcopy(
            private["execution_summary"]
        ),
        "runtime": copy.deepcopy(private["runtime"]),
        "claim_boundary": copy.deepcopy(private["claim_boundary"]),
    }


def run_development(
    *,
    root: Path,
    config_path: Path,
    development_report_path: Path,
    probe_private_report_path: Path,
    headroom_private_report_path: Path,
    view_root: Path,
    device_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path)
    development_config = _load_json(
        root / str(config["source_development_config"])
    )
    source_config = _load_json(
        root / str(development_config["source_pilot"]["config"])
    )
    headroom_public = _load_json(
        root / str(config["source_headroom_public_summary"])
    )
    probe_private = _load_json(probe_private_report_path)
    headroom_private = _load_json(headroom_private_report_path)
    development_report = _load_json(development_report_path)
    baseline_id = str(
        config["expert_bank"]["baseline_candidate_id"]
    )
    features = np.asarray(
        probe_private["features_private"]["risk_train"][
            "initial_normal_spectrum"
        ],
        dtype=np.float64,
    )
    train = development_config["development_splits"]["risk_train"]
    train_count = int(train["count"])
    families = [
        str(train["families"][index % len(train["families"])])
        for index in range(train_count)
    ]
    family_index = {
        family: index for index, family in enumerate(sorted(set(families)))
    }
    fold_labels = np.asarray(
        [family_index[family] for family in families],
        dtype=np.int64,
    )
    train_sample_ids = [
        f"risk_train-{index:03d}" for index in range(train_count)
    ]
    train_candidate_rows = headroom_private[
        "candidate_metric_rows_private"
    ]["risk_train"]
    bank_records = {}
    for bank_size in config["expert_bank"]["sizes"]:
        bank_records[str(int(bank_size))] = greedy_oracle_expert_bank(
            sample_ids=train_sample_ids,
            candidate_rows=train_candidate_rows,
            baseline_candidate_id=baseline_id,
            bank_size=int(bank_size),
        )

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
        sample_count=int(
            geometry["true_finite_aperture_sample_count"]
        ),
    )
    nominal_geometry, _ = load_real_support_geometry(
        view_root,
        rays_per_view=rays_per_view,
        sample_count=int(
            geometry["nominal_finite_aperture_sample_count"]
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
    used_masks: set[str] = set()
    splits: dict[str, DevelopmentSplit] = {}
    for split_name in (
        "risk_train",
        "risk_validation",
        "risk_calibration",
    ):
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
        splits[split_name] = wrapped
    source_rows = development_report["dataset_private"]["metric_rows"]
    for wrapped in splits.values():
        _verify_split_metadata(wrapped, source_rows)

    screen = []
    execution = []
    screen_config = config["selector_screen"]
    for bank_size_text, bank_record in bank_records.items():
        expert_ids = list(bank_record["candidate_ids"])
        baseline_index = expert_ids.index(baseline_id)
        expert_logs = _expert_bank(
            headroom_public=headroom_public,
            expert_ids=expert_ids,
            grid_size=grid_size,
            device=device,
        )
        gain_targets = _expert_gain_targets(
            sample_ids=train_sample_ids,
            candidate_rows=train_candidate_rows,
            expert_ids=expert_ids,
            baseline_candidate_id=baseline_id,
        )
        baseline_rows, ledger = _evaluate_mixture_scores(
            method=f"static_pcgls4_bank_{bank_size_text}",
            wrapped=splits["risk_train"],
            operator=nominal_operator,
            source_config=source_config,
            device=device,
            scores=np.zeros((train_count, len(expert_ids))),
            expert_log_gains=expert_logs,
            baseline_expert_index=baseline_index,
            temperature=1.0,
            confidence_threshold=1e30,
            maximum_blend=0.0,
        )
        execution.append(ledger)
        for regularization in screen_config["ridge_lambda_grid"]:
            scores = _oof_regression_scores(
                features,
                gain_targets,
                fold_labels,
                fold_count=int(screen_config["folds"]),
                regularization=float(regularization),
            )
            _, margins = score_predictions(scores)
            thresholds = [
                float(np.quantile(margins, float(quantile)))
                for quantile in screen_config[
                    "confidence_margin_quantiles"
                ]
            ]
            for temperature in screen_config["temperature_grid"]:
                for maximum_blend in screen_config["maximum_blend_grid"]:
                    for quantile_index, threshold in enumerate(thresholds):
                        method = (
                            f"screen_bank{bank_size_text}_"
                            f"l{float(regularization):g}_"
                            f"t{float(temperature):g}_"
                            f"b{float(maximum_blend):g}_q{quantile_index}"
                        )
                        rows, ledger = _evaluate_mixture_scores(
                            method=method,
                            wrapped=splits["risk_train"],
                            operator=nominal_operator,
                            source_config=source_config,
                            device=device,
                            scores=scores,
                            expert_log_gains=expert_logs,
                            baseline_expert_index=baseline_index,
                            temperature=float(temperature),
                            confidence_threshold=float(threshold),
                            maximum_blend=float(maximum_blend),
                        )
                        metrics = _paired_screen_metrics(
                            rows,
                            baseline_rows,
                            scores=scores,
                            margins=margins,
                            confidence_threshold=float(threshold),
                            baseline_expert_index=baseline_index,
                        )
                        screen.append(
                            {
                                "bank_size": int(bank_size_text),
                                "expert_candidate_ids": expert_ids,
                                "regularization": float(regularization),
                                "temperature": float(temperature),
                                "maximum_blend": float(maximum_blend),
                                "confidence_threshold": float(threshold),
                                "confidence_quantile": quantile_index,
                                **metrics,
                            }
                        )
                        execution.append(ledger)
    selected = select_omse_screen_candidate(
        screen,
        gate=config["strict_oof_gate"],
    )
    diagnostic = _best_diagnostic(screen)

    transfer_rows = []
    transfer_execution = []
    selection_models = {}
    for route_name, route in (
        ("strict", selected),
        ("diagnostic", diagnostic),
    ):
        if route_name == "strict" and not bool(
            route.get("strict_gate_pass")
        ):
            bank_size = int(config["expert_bank"]["sizes"][0])
            route_parameters = {
                "bank_size": bank_size,
                "expert_candidate_ids": bank_records[str(bank_size)][
                    "candidate_ids"
                ],
                "regularization": 0.1,
                "temperature": 1.0,
                "maximum_blend": 0.0,
                "confidence_threshold": 1e30,
            }
        else:
            route_parameters = route
        expert_ids = list(route_parameters["expert_candidate_ids"])
        baseline_index = expert_ids.index(baseline_id)
        expert_logs = _expert_bank(
            headroom_public=headroom_public,
            expert_ids=expert_ids,
            grid_size=grid_size,
            device=device,
        )
        gain_targets = _expert_gain_targets(
            sample_ids=train_sample_ids,
            candidate_rows=train_candidate_rows,
            expert_ids=expert_ids,
            baseline_candidate_id=baseline_id,
        )
        ridge = fit_ridge_multioutput(
            features,
            gain_targets,
            regularization=float(route_parameters["regularization"]),
        )
        factory = ObservableMorphologyExpertFactory(
            expert_log_gains=expert_logs,
            expert_candidate_ids=expert_ids,
            baseline_expert_index=baseline_index,
            feature_mean=torch.as_tensor(ridge["mean"]),
            feature_scale=torch.as_tensor(ridge["scale"]),
            ridge_weights=torch.as_tensor(ridge["weights"]),
            temperature=float(route_parameters["temperature"]),
            confidence_threshold=float(
                route_parameters["confidence_threshold"]
            ),
            maximum_blend=float(route_parameters["maximum_blend"]),
        ).to(device)
        selection_models[route_name] = {
            "parameters": copy.deepcopy(route_parameters),
            "ridge_mean_private": np.asarray(ridge["mean"]).tolist(),
            "ridge_scale_private": np.asarray(ridge["scale"]).tolist(),
            "ridge_weights_private": np.asarray(ridge["weights"]).tolist(),
        }
        for split_name in ("risk_validation", "risk_calibration"):
            rows, ledger = _evaluate_integrated_factory(
                method=f"ogse_{route_name}",
                wrapped=splits[split_name],
                operator=nominal_operator,
                source_config=source_config,
                device=device,
                factory=factory,
            )
            transfer_rows.extend(rows)
            transfer_execution.append(ledger)

    reference_bank = bank_records[str(config["expert_bank"]["sizes"][0])][
        "candidate_ids"
    ]
    reference_logs = _expert_bank(
        headroom_public=headroom_public,
        expert_ids=reference_bank,
        grid_size=grid_size,
        device=device,
    )
    reference_baseline_index = reference_bank.index(baseline_id)
    for split_name in ("risk_validation", "risk_calibration"):
        rows, ledger = _evaluate_mixture_scores(
            method="static_pcgls4",
            wrapped=splits[split_name],
            operator=nominal_operator,
            source_config=source_config,
            device=device,
            scores=np.zeros(
                (len(splits[split_name].data.truth), len(reference_bank))
            ),
            expert_log_gains=reference_logs,
            baseline_expert_index=reference_baseline_index,
            temperature=1.0,
            confidence_threshold=1e30,
            maximum_blend=0.0,
        )
        transfer_rows.extend(rows)
        transfer_execution.append(ledger)
    summaries = []
    for method_index, method in enumerate(
        ("ogse_strict", "ogse_diagnostic")
    ):
        for split_index, split_name in enumerate(
            ("risk_validation", "risk_calibration")
        ):
            summaries.append(
                paired_gain_summary(
                    transfer_rows,
                    split=split_name,
                    candidate_method=method,
                    bootstrap_seed=20263200
                    + 100 * method_index
                    + split_index,
                )
            )
    development_gates = [
        _development_gate(
            summaries,
            method=method,
            gate=config["development_gate"],
        )
        for method in ("ogse_strict", "ogse_diagnostic")
    ]
    execution.extend(transfer_execution)
    ledgers_valid = all(
        row["logical_calls_per_sample"] == {
            "forward": 4,
            "adjoint": 4,
        }
        and bool(row["data_objective_monotone"])
        and float(row["gain_minimum"]) > 0.0
        and float(row["gain_geometric_mean_maximum_defect"]) <= 2e-5
        for row in execution
    )
    private = {
        "schema_version": PRIVATE_SCHEMA,
        "algorithm_schema": MORPHOLOGY_EXPERT_SCHEMA,
        "status": STATUS,
        "evidence_scope": (
            "REAL_PSU_SUPPORT_GEOMETRY_WITH_ANALYTIC_REACTION_MORPHOLOGY_"
            "AND_SYNTHETIC_CAMERA_NOISE_POSTOPEN_OGSE_DEVELOPMENT_ONLY"
        ),
        "configuration_private": {
            "root": str(root.resolve()),
            "config_path": str(config_path.resolve()),
            "development_report_path": str(
                development_report_path.resolve()
            ),
            "probe_private_report_path": str(
                probe_private_report_path.resolve()
            ),
            "headroom_private_report_path": str(
                headroom_private_report_path.resolve()
            ),
            "view_root": str(view_root.resolve()),
            "device": str(device),
        },
        "configuration_public": copy.deepcopy(config),
        "regeneration_checks": {
            "development_metadata_matches_frozen_rows": True,
            "expert_banks_selected_on_risk_train_only": True,
            "gain_regression_oof_screen_uses_risk_train_only": True,
            "integrated_factory_shares_first_adjoint": True,
            "opened_fresh_not_loaded": True,
            "fixed_spd_and_call_ledgers_pass": bool(ledgers_valid),
        },
        "expert_banks": bank_records,
        "selection_screen": screen,
        "selected_strict_candidate": selected,
        "best_diagnostic_candidate": diagnostic,
        "selection_models_private": selection_models,
        "transfer_metric_rows_private": transfer_rows,
        "paired_gain_summary": summaries,
        "development_gates": development_gates,
        "execution": execution,
        "execution_summary": {
            "screen_candidate_count": len(screen),
            "bank_count": len(bank_records),
            "logical_calls_per_candidate_sample": {
                "forward": 4,
                "adjoint": 4,
            },
        },
        "runtime": {
            "wall_seconds": float(time.perf_counter() - started),
            "maximum_rss_bytes": int(_max_rss_bytes()),
        },
        "claim_boundary": {
            "postopen_development_diagnostic_only": True,
            "validation_and_calibration_are_postopen_diagnostics": True,
            "fresh_values_loaded": False,
            "fresh_repeat_authorized": False,
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
            "psu_b0_ogse_pcgls_development_v2.json"
        ),
    )
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--probe-private-report", type=Path, required=True)
    parser.add_argument("--headroom-private-report", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--public-output", type=Path)
    args = parser.parse_args()
    private, public = run_development(
        root=args.root,
        config_path=args.config,
        development_report_path=args.development_report,
        probe_private_report_path=args.probe_private_report,
        headroom_private_report_path=args.headroom_private_report,
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
