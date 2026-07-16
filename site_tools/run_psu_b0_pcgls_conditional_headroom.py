#!/usr/bin/env python3
"""Audit conditional headroom inside a finite fixed-SPD PCGLS family."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import platform
import resource
import time
from typing import Any, Iterable

import numpy as np
import torch

from demo_t16_operator.psu_b0_classical_baselines import (
    GeneralizedSobolevDirection,
)
from demo_t16_operator.psu_b0_streaming_operator import (
    zero_outer_boundary_support,
)
from site_tools.run_psu_b0_conditioned_pcgls_development import (
    _evaluate,
    paired_gain_summary,
)
from site_tools.run_psu_b0_classical_frontier_development import (
    _verify_split_metadata,
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


PRIVATE_SCHEMA = "psu-b0-pcgls-conditional-headroom-private-1.0"
PUBLIC_SCHEMA = "psu-b0-pcgls-conditional-headroom-public-1.0"
STATUS = "PCGLS_CONDITIONAL_HEADROOM_DEVELOPMENT_COMPLETE_FRESH_NOT_USED"


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def candidate_grid(
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    section = config["candidate_grid"]
    output = []
    for strength in section["strengths"]:
        for epsilon in section["epsilons"]:
            for pattern, weights in section[
                "axis_weight_patterns_xyz"
            ].items():
                candidate_id = (
                    f"pcgls4_s{float(strength):g}_"
                    f"e{float(epsilon):g}_{pattern}"
                )
                output.append(
                    {
                        "candidate_id": candidate_id,
                        "strength": float(strength),
                        "epsilon": float(epsilon),
                        "axis_pattern": str(pattern),
                        "axis_weights_xyz": [
                            float(value) for value in weights
                        ],
                    }
                )
    identifiers = [str(row["candidate_id"]) for row in output]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate grid contains duplicate identifiers")
    return output


def relative_noise_bin(
    value: float,
    *,
    edges: Iterable[float],
    labels: Iterable[str],
) -> str:
    edge_values = tuple(float(item) for item in edges)
    label_values = tuple(str(item) for item in labels)
    if len(label_values) != len(edge_values) + 1:
        raise ValueError("noise-bin labels must outnumber edges by one")
    if tuple(sorted(edge_values)) != edge_values:
        raise ValueError("noise-bin edges must be increasing")
    numeric = float(value)
    for edge, label in zip(edge_values, label_values):
        if numeric <= edge:
            return label
    return label_values[-1]


def annotate_noise_bins(
    rows: list[dict[str, Any]],
    *,
    edges: Iterable[float],
    labels: Iterable[str],
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        copied = dict(row)
        copied["relative_noise_bin"] = relative_noise_bin(
            float(row["relative_noise"]),
            edges=edges,
            labels=labels,
        )
        output.append(copied)
    return output


def _group_key(
    row: dict[str, Any],
    keys: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(str(row[key]) for key in keys)


def select_candidates_by_group(
    rows: list[dict[str, Any]],
    *,
    group_keys: tuple[str, ...],
) -> dict[tuple[str, ...], str]:
    """Select one candidate per train stratum using the frozen loss rule."""

    grouped: dict[tuple[str, ...], dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        key = _group_key(row, group_keys)
        candidate_id = str(row["candidate_id"])
        grouped.setdefault(key, {}).setdefault(candidate_id, []).append(row)
    if not grouped:
        raise ValueError("cannot select candidates from empty rows")
    output: dict[tuple[str, ...], str] = {}
    for key, candidates in sorted(grouped.items()):
        expected = {str(row["sample_id"]) for values in candidates.values() for row in values}
        scores = []
        for candidate_id, values in candidates.items():
            sample_ids = {str(row["sample_id"]) for row in values}
            if sample_ids != expected:
                raise ValueError("candidate rows do not cover the same group")
            scores.append(
                (
                    float(np.mean([row["combined_loss"] for row in values])),
                    float(
                        np.mean(
                            [row["field_relative_l2"] for row in values]
                        )
                    ),
                    candidate_id,
                )
            )
        output[key] = min(scores)[2]
    return output


def _row_lookup(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    output = {}
    for row in rows:
        key = (str(row["sample_id"]), str(row["candidate_id"]))
        if key in output:
            raise ValueError(f"duplicate candidate row: {key}")
        output[key] = row
    return output


def materialize_group_strategy(
    rows: list[dict[str, Any]],
    *,
    method: str,
    mapping: dict[tuple[str, ...], str],
    group_keys: tuple[str, ...],
    fallback_candidate_id: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    lookup = _row_lookup(rows)
    sample_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_rows.setdefault(str(row["sample_id"]), row)
    output = []
    usage: dict[str, int] = {}
    for sample_id, anchor in sorted(sample_rows.items()):
        candidate_id = mapping.get(
            _group_key(anchor, group_keys),
            str(fallback_candidate_id),
        )
        selected = dict(lookup[(sample_id, candidate_id)])
        selected["method"] = str(method)
        selected["selected_candidate_id"] = candidate_id
        output.append(selected)
        usage[candidate_id] = usage.get(candidate_id, 0) + 1
    return output, usage


def materialize_sample_oracle(
    rows: list[dict[str, Any]],
    *,
    method: str,
    metric: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["sample_id"]), []).append(row)
    output = []
    usage: dict[str, int] = {}
    for sample_id, candidates in sorted(grouped.items()):
        selected = min(
            candidates,
            key=lambda row: (
                float(row[metric]),
                float(row["field_relative_l2"]),
                str(row["candidate_id"]),
            ),
        )
        copied = dict(selected)
        copied["method"] = str(method)
        copied["selected_candidate_id"] = str(selected["candidate_id"])
        output.append(copied)
        candidate_id = str(selected["candidate_id"])
        usage[candidate_id] = usage.get(candidate_id, 0) + 1
    return output, usage


def _mapping_for_public(
    mapping: dict[tuple[str, ...], str],
) -> list[dict[str, Any]]:
    return [
        {
            "stratum": list(key),
            "candidate_id": value,
        }
        for key, value in sorted(mapping.items())
    ]


def _candidate_usage_for_public(
    usage: dict[str, int],
) -> list[dict[str, Any]]:
    return [
        {"candidate_id": candidate_id, "sample_count": int(count)}
        for candidate_id, count in sorted(
            usage.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def _strategy_decision(
    summaries: list[dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    lookup = {
        (str(row["candidate_method"]), str(row["split"])): row
        for row in summaries
    }
    oracle = [
        lookup[("sample_field_truth_oracle", split)]
        for split in ("risk_validation", "risk_calibration")
    ]
    view = [
        lookup[("train_view_count", split)]
        for split in ("risk_validation", "risk_calibration")
    ]
    oracle_minimum = min(
        float(row["mean_field_gain_percent"]) for row in oracle
    )
    view_minimum = min(float(row["mean_field_gain_percent"]) for row in view)
    if oracle_minimum < float(threshold):
        code = "FINITE_SPECTRAL_FAMILY_HEADROOM_SMALL"
        next_route = "PRIORITIZE_TV_STOPPING_OR_NEW_PRIOR"
    elif view_minimum < float(threshold):
        code = "ORACLE_HEADROOM_EXISTS_OBSERVABLE_STRATA_DO_NOT_TRANSFER"
        next_route = "IMPROVE_PHYSICAL_FEATURES_BEFORE_MODEL_CAPACITY"
    else:
        code = "OBSERVABLE_CONDITIONAL_HEADROOM_EXISTS"
        next_route = "DESIGN_SMALL_AUDITABLE_SELECTOR"
    return {
        "decision_code": code,
        "next_route": next_route,
        "meaningful_headroom_threshold_percent": float(threshold),
        "minimum_sample_field_oracle_gain_percent": oracle_minimum,
        "minimum_train_view_count_gain_percent": view_minimum,
    }


def build_public_summary(private: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PUBLIC_SCHEMA,
        "status": private["status"],
        "evidence_scope": private["evidence_scope"],
        "configuration": copy.deepcopy(private["configuration_public"]),
        "regeneration_checks": copy.deepcopy(
            private["regeneration_checks"]
        ),
        "candidate_grid": copy.deepcopy(private["candidate_grid"]),
        "train_selection": copy.deepcopy(private["train_selection"]),
        "strategy_usage": copy.deepcopy(private["strategy_usage"]),
        "paired_gain_summary": copy.deepcopy(
            private["paired_gain_summary"]
        ),
        "decision": copy.deepcopy(private["decision"]),
        "execution_summary": copy.deepcopy(private["execution_summary"]),
        "runtime": copy.deepcopy(private["runtime"]),
        "claim_boundary": copy.deepcopy(private["claim_boundary"]),
    }


def run_headroom_audit(
    *,
    root: Path,
    config_path: Path,
    development_report_path: Path,
    view_root: Path,
    device_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path)
    development_config = _load_json(
        root / str(config["source_development_config"])
    )
    source_config = _load_json(
        root
        / str(
            development_config["source_pilot"]["config"]
        )
    )
    strong_frontier = _load_json(
        root / str(config["source_strong_frontier"])
    )
    development_report = _load_json(development_report_path)
    baseline_config = config["solver"]["baseline"]
    selected = strong_frontier["selected_candidates"]["pcgls_4"][
        "parameters"
    ]
    expected = (
        float(baseline_config["strength"]),
        float(baseline_config["epsilon"]),
        int(config["solver"]["stages"]),
    )
    actual = (
        float(selected["strength"]),
        float(selected["epsilon"]),
        int(selected["stages"]),
    )
    if actual != expected:
        raise ValueError("headroom baseline drifted from strong frontier")
    candidates = candidate_grid(config)
    candidate_by_id = {
        str(row["candidate_id"]): row for row in candidates
    }
    baseline_id = str(baseline_config["candidate_id"])
    if baseline_id not in candidate_by_id:
        raise ValueError("baseline is absent from the finite candidate grid")

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

    edges = config["observable_strata"]["relative_noise_bin_edges"]
    labels = config["observable_strata"]["relative_noise_bin_labels"]
    rows_by_split: dict[str, list[dict[str, Any]]] = {
        name: [] for name in splits
    }
    execution = []
    for candidate in candidates:
        direction = GeneralizedSobolevDirection(
            (grid_size,) * 3,
            strength=float(candidate["strength"]),
            epsilon=float(candidate["epsilon"]),
            axis_weights_xyz=tuple(
                float(value)
                for value in candidate["axis_weights_xyz"]
            ),
        ).to(device)
        for split_name, wrapped in splits.items():
            rows, ledger = _evaluate(
                method=f"candidate_{candidate['candidate_id']}",
                wrapped=wrapped,
                operator=nominal_operator,
                source_config=source_config,
                device=device,
                model=None,
                static_direction=direction,
                batch_size=12,
            )
            for row in rows:
                row["candidate_id"] = str(candidate["candidate_id"])
            rows_by_split[split_name].extend(
                annotate_noise_bins(rows, edges=edges, labels=labels)
            )
            ledger["candidate_id"] = str(candidate["candidate_id"])
            execution.append(ledger)

    train_rows = rows_by_split["risk_train"]
    global_mapping = select_candidates_by_group(
        train_rows,
        group_keys=(),
    )
    global_candidate = global_mapping[()]
    view_mapping = select_candidates_by_group(
        train_rows,
        group_keys=("active_view_count",),
    )
    view_noise_mapping = select_candidates_by_group(
        train_rows,
        group_keys=("active_view_count", "relative_noise_bin"),
    )
    family_mapping = select_candidates_by_group(
        train_rows,
        group_keys=("family",),
    )

    all_strategy_rows: list[dict[str, Any]] = []
    strategy_usage: dict[str, dict[str, list[dict[str, Any]]]] = {}
    evaluation_splits = ("risk_validation", "risk_calibration")
    for split_name in evaluation_splits:
        target = rows_by_split[split_name]
        baseline_rows, baseline_usage = materialize_group_strategy(
            target,
            method="static_pcgls4",
            mapping={(): baseline_id},
            group_keys=(),
            fallback_candidate_id=baseline_id,
        )
        all_strategy_rows.extend(baseline_rows)
        strategy_usage.setdefault("static_pcgls4", {})[split_name] = (
            _candidate_usage_for_public(baseline_usage)
        )
        strategies = (
            (
                "train_global",
                {(): global_candidate},
                (),
            ),
            (
                "train_view_count",
                view_mapping,
                ("active_view_count",),
            ),
            (
                "train_view_count_plus_noise",
                view_noise_mapping,
                ("active_view_count", "relative_noise_bin"),
            ),
            (
                "train_family_label",
                family_mapping,
                ("family",),
            ),
        )
        for method, mapping, keys in strategies:
            selected_rows, usage = materialize_group_strategy(
                target,
                method=method,
                mapping=mapping,
                group_keys=keys,
                fallback_candidate_id=global_candidate,
            )
            all_strategy_rows.extend(selected_rows)
            strategy_usage.setdefault(method, {})[split_name] = (
                _candidate_usage_for_public(usage)
            )

        split_family_mapping = select_candidates_by_group(
            target,
            group_keys=("family",),
        )
        split_family_rows, usage = materialize_group_strategy(
            target,
            method="split_family_truth_oracle",
            mapping=split_family_mapping,
            group_keys=("family",),
            fallback_candidate_id=global_candidate,
        )
        all_strategy_rows.extend(split_family_rows)
        strategy_usage.setdefault(
            "split_family_truth_oracle", {}
        )[split_name] = _candidate_usage_for_public(usage)

        for method, metric in (
            ("sample_combined_truth_oracle", "combined_loss"),
            ("sample_field_truth_oracle", "field_relative_l2"),
        ):
            oracle_rows, usage = materialize_sample_oracle(
                target,
                method=method,
                metric=metric,
            )
            all_strategy_rows.extend(oracle_rows)
            strategy_usage.setdefault(method, {})[split_name] = (
                _candidate_usage_for_public(usage)
            )

    methods = [
        "train_global",
        "train_view_count",
        "train_view_count_plus_noise",
        "train_family_label",
        "split_family_truth_oracle",
        "sample_combined_truth_oracle",
        "sample_field_truth_oracle",
    ]
    summaries = []
    for method_index, method in enumerate(methods):
        for split_index, split_name in enumerate(evaluation_splits):
            summaries.append(
                paired_gain_summary(
                    all_strategy_rows,
                    split=split_name,
                    candidate_method=method,
                    bootstrap_seed=20262900
                    + 100 * method_index
                    + split_index,
                )
            )
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
    decision = _strategy_decision(
        summaries,
        threshold=float(
            config["decision_thresholds"][
                "meaningful_mean_field_headroom_percent"
            ]
        ),
    )
    private = {
        "schema_version": PRIVATE_SCHEMA,
        "status": STATUS,
        "evidence_scope": (
            "REAL_PSU_SUPPORT_GEOMETRY_WITH_ANALYTIC_REACTION_MORPHOLOGY_"
            "AND_SYNTHETIC_CAMERA_NOISE_POSTOPEN_DEVELOPMENT_HEADROOM_ONLY"
        ),
        "configuration_private": {
            "root": str(root.resolve()),
            "config_path": str(config_path.resolve()),
            "development_report_path": str(
                development_report_path.resolve()
            ),
            "view_root": str(view_root.resolve()),
            "device": str(device),
        },
        "configuration_public": copy.deepcopy(config),
        "regeneration_checks": {
            "all_development_split_metadata_match_frozen_rows": True,
            "baseline_matches_validation_selected_pcgls4": True,
            "selection_uses_risk_train_only": True,
            "opened_fresh_not_loaded": True,
            "candidate_call_ledgers_and_spd_checks_pass": bool(
                ledgers_valid
            ),
        },
        "candidate_grid": {
            "candidate_count": len(candidates),
            "candidates": copy.deepcopy(candidates),
        },
        "train_selection": {
            "global": global_candidate,
            "view_count": _mapping_for_public(view_mapping),
            "view_count_plus_noise": _mapping_for_public(
                view_noise_mapping
            ),
            "family_label_non_deployable": _mapping_for_public(
                family_mapping
            ),
        },
        "strategy_usage": strategy_usage,
        "candidate_metric_rows_private": rows_by_split,
        "strategy_metric_rows_private": all_strategy_rows,
        "paired_gain_summary": summaries,
        "decision": decision,
        "execution": execution,
        "execution_summary": {
            "candidate_evaluations": len(execution),
            "candidate_count": len(candidates),
            "split_count": len(splits),
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
            "fresh_values_loaded": False,
            "per_sample_oracle_is_deployable": False,
            "split_family_oracle_is_deployable": False,
            "train_family_label_is_available_on_real_measurements": False,
            "relative_noise_is_validated_on_real_measurements": False,
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
            "psu_b0_pcgls_conditional_headroom_v1.json"
        ),
    )
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--public-output", type=Path)
    args = parser.parse_args()
    private, public = run_headroom_audit(
        root=args.root,
        config_path=args.config,
        development_report_path=args.development_report,
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
