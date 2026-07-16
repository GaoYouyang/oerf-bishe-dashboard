#!/usr/bin/env python3
"""Audit frozen v3k-F stopping rules after the selection commit exists."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .noise_stopping import effective_operator_calls, gather_path
    from . import run_v3k_d_strong_numerical_controls as v3d
    from .v3k_f_noise_stopping_common import (
        DEPLOYABLE_METHODS,
        METHOD_LABELS,
        ORACLE_METHODS,
        ROOT,
        build_bundle,
        load_context,
        read_json,
        stop_indices,
        truth_oracle_stop,
        validation_roles,
    )
except ImportError:
    from noise_stopping import effective_operator_calls, gather_path
    import run_v3k_d_strong_numerical_controls as v3d
    from v3k_f_noise_stopping_common import (
        DEPLOYABLE_METHODS,
        METHOD_LABELS,
        ORACLE_METHODS,
        ROOT,
        build_bundle,
        load_context,
        read_json,
        stop_indices,
        truth_oracle_stop,
        validation_roles,
    )


DEFAULT_CONFIG = ROOT / "configs" / "v3k_f_noise_stopping_gate.json"
PUBLIC_FILES = [
    "v3k_f_pair_manifest.csv",
    "v3k_f_validation_roles.csv",
    "v3k_f_sample_metrics.csv",
    "v3k_f_split_summary.csv",
    "v3k_f_pairwise_summary.csv",
    "v3k_f_layout_summary.csv",
    "v3k_f_stopping_rows.csv",
    "v3k_f_operator_call_ledger.csv",
    "v3k_f_noise_stopping_dashboard.json",
    "v3k_f_noise_stopping_report.json",
    "t16_v3k_f_noise_stopping_gate.png",
]
FAMILY_TO_METHOD = {
    "self_discrepancy": "fno_pbb_discrepancy",
    "camera_discrepancy": "fno_pbb_camera_discrepancy",
    "ncp": "fno_pbb_ncp",
    "hybrid": "fno_pbb_hybrid",
    "generator_discrepancy": "fno_pbb_generator_sigma",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device")
    return parser.parse_args()


def selected_parameters(choice: dict[str, object]) -> dict[str, float]:
    return {
        key: float(choice[key])
        for key in ("tau", "camera_max_factor", "ncp_multiplier")
        if choice.get(key) is not None
    }


def role_for_sources(
    sources: np.ndarray, tune_rows: np.ndarray
) -> dict[int, str]:
    output = {}
    for source in np.unique(sources):
        selected = tune_rows[sources == source]
        if not np.all(selected == selected[0]):
            raise RuntimeError("a validation field was split across tune and lock")
        output[int(source)] = "val_tune" if bool(selected[0]) else "val_lock"
    return output


def predictions_and_stops(
    bundle: dict[str, object],
    selected: dict[str, dict[str, object]],
    ncp_thresholds: dict[str, float],
    maximum: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    predictions = {
        "feasible_fno": bundle["baseline"]["feasible_fno"],
        "fno_geometry": bundle["baseline"]["fno_geometry"],
        "fno_pbb_fixed64": bundle["path"][maximum],
    }
    stops = {"fno_pbb_fixed64": np.full(len(bundle["sources"]), maximum, dtype=np.int64)}
    for family, method in FAMILY_TO_METHOD.items():
        stop = stop_indices(
            family,
            bundle,
            selected_parameters(selected[family]),
            ncp_thresholds,
            maximum,
        )
        stops[method] = stop
        predictions[method] = gather_path(bundle["path"], stop)
    oracle_stop = truth_oracle_stop(bundle)
    stops["fno_pbb_truth_oracle"] = oracle_stop
    predictions["fno_pbb_truth_oracle"] = gather_path(bundle["path"], oracle_stop)
    return predictions, stops


def stopping_rows(
    bundle: dict[str, object],
    stops: dict[str, np.ndarray],
    maximum: int,
    split_roles: dict[int, str] | None,
) -> list[dict[str, object]]:
    rows = []
    pairs = bundle["dataset"].pairs
    for method, stop in stops.items():
        if method == "fno_pbb_truth_oracle":
            a_calls = np.full(len(stop), maximum, dtype=np.int64)
            at_calls = np.full(len(stop), maximum, dtype=np.int64)
            decision_input = "truth field and complete path; oracle only"
        else:
            a_calls, at_calls = effective_operator_calls(stop, maximum)
            decision_input = {
                "fno_pbb_fixed64": "fixed cap",
                "fno_pbb_discrepancy": "active y, q, mask, residual",
                "fno_pbb_camera_discrepancy": "active y, q, mask, per-camera residual",
                "fno_pbb_ncp": "active residual detector-axis periodogram",
                "fno_pbb_hybrid": "active discrepancy and NCP",
                "fno_pbb_generator_sigma": "clean full-view RMS; oracle only",
            }[method]
        for index, pair in enumerate(pairs):
            source = int(pair["source_index"])
            checked = min(int(stop[index]), maximum - 1)
            rows.append(
                {
                    "source_split": split_roles[source]
                    if split_roles is not None
                    else str(pair["source_split"]),
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "source_index": source,
                    "sample_seed": int(bundle["sample_seeds"][index]),
                    "pair_index": int(pair["pair_index"]),
                    "geometry_id": str(pair["geometry_id"]),
                    "noise_level": float(bundle["noise_level"][index]),
                    "stop_iteration": int(stop[index]),
                    "last_residual_check_iteration": checked,
                    "forced_cap": int(stop[index] == maximum),
                    "a_calls": int(a_calls[index]),
                    "at_calls": int(at_calls[index]),
                    "total_operator_calls": int(a_calls[index] + at_calls[index]),
                    "discrepancy_pooled_at_last_check": float(
                        bundle["self_stats"]["discrepancy_pooled"][checked, index]
                    ),
                    "discrepancy_camera_max_at_last_check": float(
                        bundle["self_stats"]["discrepancy_camera_max"][checked, index]
                    ),
                    "ncp_camera_mean_at_last_check": float(
                        bundle["self_stats"]["ncp_camera_mean"][checked, index]
                    ),
                    "ncp_camera_max_at_last_check": float(
                        bundle["self_stats"]["ncp_camera_max"][checked, index]
                    ),
                    "deployable_input_only": method in DEPLOYABLE_METHODS
                    or method == "fno_pbb_fixed64",
                    "oracle": method in ORACLE_METHODS,
                    "decision_input": decision_input,
                    "audit_execution_precomputed_full_path": True,
                }
            )
    return rows


def call_ledger(
    stopping: list[dict[str, object]],
    baseline_selection: dict[str, object],
    splits: list[str],
) -> list[dict[str, object]]:
    rows = []
    for split in splits:
        subset = [row for row in stopping if row["source_split"] == split]
        for method in [
            "fno_pbb_fixed64",
            "fno_pbb_discrepancy",
            "fno_pbb_camera_discrepancy",
            "fno_pbb_ncp",
            "fno_pbb_hybrid",
            "fno_pbb_generator_sigma",
            "fno_pbb_truth_oracle",
        ]:
            values = [row for row in subset if row["method"] == method]
            calls = np.asarray([row["total_operator_calls"] for row in values])
            stops = np.asarray([row["stop_iteration"] for row in values])
            rows.append(
                {
                    "source_split": split,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "sample_layout_count": len(values),
                    "mean_stop_iteration": float(np.mean(stops)),
                    "median_stop_iteration": float(np.median(stops)),
                    "p90_stop_iteration": float(np.quantile(stops, 0.9)),
                    "forced_cap_fraction": float(np.mean(stops == 64)),
                    "mean_total_operator_calls": float(np.mean(calls)),
                    "median_total_operator_calls": float(np.median(calls)),
                    "p90_total_operator_calls": float(np.quantile(calls, 0.9)),
                    "maximum_total_operator_calls": int(np.max(calls)),
                    "deployable_input_only": method in DEPLOYABLE_METHODS
                    or method == "fno_pbb_fixed64",
                    "oracle": method in ORACLE_METHODS,
                    "wall_time_speedup_claimed": False,
                }
            )
        _, a_calls, at_calls = v3d_baseline_calls("fno_geometry", baseline_selection)
        rows.append(
            {
                "source_split": split,
                "method": "fno_geometry",
                "method_label": METHOD_LABELS["fno_geometry"],
                "sample_layout_count": len(
                    {int(row["pair_index"]) for row in subset}
                ),
                "mean_stop_iteration": int(a_calls),
                "median_stop_iteration": int(a_calls),
                "p90_stop_iteration": int(a_calls),
                "forced_cap_fraction": 1.0,
                "mean_total_operator_calls": int(a_calls + at_calls),
                "median_total_operator_calls": int(a_calls + at_calls),
                "p90_total_operator_calls": int(a_calls + at_calls),
                "maximum_total_operator_calls": int(a_calls + at_calls),
                "deployable_input_only": True,
                "oracle": False,
                "wall_time_speedup_claimed": False,
            }
        )
    return rows


def v3d_baseline_calls(
    method: str, baseline_selection: dict[str, object]
) -> tuple[int, int, int]:
    try:
        from .run_v3k_e_projected_bb_gate import baseline_calls
    except ImportError:
        from run_v3k_e_projected_bb_gate import baseline_calls
    return baseline_calls(method, baseline_selection)


def correction_references() -> dict[str, str]:
    return {method: "feasible_fno" for method in METHOD_LABELS}


def comparisons() -> list[tuple[str, str]]:
    candidates = [
        "fno_pbb_fixed64",
        "fno_pbb_discrepancy",
        "fno_pbb_camera_discrepancy",
        "fno_pbb_ncp",
        "fno_pbb_hybrid",
        "fno_pbb_generator_sigma",
        "fno_pbb_truth_oracle",
    ]
    output = [(method, "fno_geometry") for method in candidates]
    output.extend(
        (method, "fno_pbb_fixed64")
        for method in candidates
        if method != "fno_pbb_fixed64"
    )
    return output


def find_pair(
    rows: list[dict[str, object]], split: str, candidate: str, comparator: str
) -> dict[str, object]:
    return v3d.find_pair(rows, split, candidate, comparator)


def plot_results(
    path: Path,
    summary: list[dict[str, object]],
    pairwise: list[dict[str, object]],
    ledger: list[dict[str, object]],
) -> None:
    splits = [
        "val_lock",
        "test_iid",
        "test_noise_ood",
        "test_family_ood",
        "test_joint_ood",
    ]
    methods = [
        "fno_geometry",
        "fno_pbb_fixed64",
        "fno_pbb_discrepancy",
        "fno_pbb_ncp",
        "fno_pbb_hybrid",
    ]
    lookup = {
        (str(row["source_split"]), str(row["method"])): row for row in summary
    }
    call_lookup = {
        (str(row["source_split"]), str(row["method"])): row for row in ledger
    }
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), constrained_layout=True)
    x = np.arange(len(splits))
    width = 0.15
    colors = ["#7a8b8b", "#436f7a", "#197278", "#d08c60", "#7c5c8c"]
    for offset, (method, color) in enumerate(zip(methods, colors)):
        axes[0].bar(
            x + (offset - 2) * width,
            [lookup[(split, method)]["mean_field_rel_l2"] for split in splits],
            width,
            label=method.replace("fno_pbb_", ""),
            color=color,
        )
    axes[0].set_xticks(x, [value.replace("test_", "") for value in splits], rotation=28, ha="right")
    axes[0].set_title("Field error after field collapse")
    axes[0].set_ylabel("mean relative L2")
    axes[0].legend(fontsize=6)

    for method, color in zip(methods[1:], colors[1:]):
        gains = [
            find_pair(pairwise, split, method, "fno_geometry")["mean_field_gain_pct"]
            for split in splits
        ]
        axes[1].plot(x, gains, marker="o", label=method.replace("fno_pbb_", ""), color=color)
    axes[1].axhline(0.0, color="#58666e", linewidth=1)
    axes[1].set_xticks(x, [value.replace("test_", "") for value in splits], rotation=28, ha="right")
    axes[1].set_title("Gain vs fixed Landweber")
    axes[1].set_ylabel("mean field gain (%)")
    axes[1].legend(fontsize=6)

    for method, color in zip(methods[1:], colors[1:]):
        axes[2].plot(
            x,
            [call_lookup[(split, method)]["mean_total_operator_calls"] for split in splits],
            marker="o",
            label=method.replace("fno_pbb_", ""),
            color=color,
        )
    axes[2].axhline(128, color="#58666e", linewidth=1, linestyle="--")
    axes[2].set_xticks(x, [value.replace("test_", "") for value in splits], rotation=28, ha="right")
    axes[2].set_title("Deployment-equivalent call budget")
    axes[2].set_ylabel("mean A + A^T calls")

    noise_methods = methods[1:]
    noise_pairs = [
        find_pair(pairwise, "test_noise_ood", method, "fno_geometry")
        for method in noise_methods
    ]
    axes[3].barh(
        [method.replace("fno_pbb_", "") for method in noise_methods],
        [100 * float(row["harm_rate_gt_1pct"]) for row in noise_pairs],
        color=colors[1:],
    )
    axes[3].axvline(10, color="#b8564f", linewidth=1, linestyle="--")
    axes[3].set_title("Noise-OOD harm tail")
    axes[3].set_xlabel("fields degraded >1% vs Landweber (%)")
    fig.suptitle("v3k-F: discrepancy rescues noise; NCP alone is morphology-unsafe", fontsize=14)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    selection_dir = ROOT / "results" / str(config["selection_output_dir"])
    selection_path = selection_dir / "v3k_f_selection_commit.json"
    selection = read_json(selection_path)
    if selection.get("test_dataset_constructed") is not False:
        raise RuntimeError("selection commit does not prove a test-free phase")
    context = load_context(config, args.device)
    if selection["provenance"]["pbb_selection_commit_sha256"] != v3d.sha256(
        context["pbb_path"]
    ):
        raise RuntimeError("frozen PBB selection drifted after v3k-F selection")
    selected = selection["selected"]
    maximum = int(config["frozen_pbb"]["maximum_iterations"])

    sample_rows: list[dict[str, object]] = []
    stop_rows: list[dict[str, object]] = []
    pair_manifest: list[dict[str, object]] = []
    validation_manifest: list[dict[str, object]] = []
    split_names: list[str] = []
    design = config["pair_design"]
    for split, partition in design["evaluation_geometry_partition_by_split"].items():
        bundle = build_bundle(context, str(split), str(partition))
        split_roles = None
        if split == "val":
            tune_rows, validation_manifest = validation_roles(bundle, config)
            split_roles = role_for_sources(bundle["sources"], tune_rows)
            split_names.extend(["val_tune", "val_lock"])
        else:
            split_names.append(str(split))
        predictions, stops = predictions_and_stops(
            bundle, selected, context["ncp_thresholds"], maximum
        )
        metrics = v3d.metric_rows(
            predictions,
            bundle["dataset"],
            context["data"],
            context["factory"],
            labels=METHOD_LABELS,
            correction_reference=correction_references(),
        )
        if split_roles is not None:
            for row in metrics:
                row["source_split"] = split_roles[int(row["source_index"])]
        sample_rows.extend(metrics)
        stop_rows.extend(stopping_rows(bundle, stops, maximum, split_roles))
        for pair in bundle["pairs"]:
            source = int(pair["source_index"])
            pair_manifest.append(
                {
                    **pair,
                    "evaluation_role": split_roles[source]
                    if split_roles is not None
                    else "post_selection_development_audit",
                    "selection_commit_preexisted": True,
                }
            )

    summary, pairwise = v3d.summarize(
        sample_rows,
        config,
        labels=METHOD_LABELS,
        comparisons=comparisons(),
    )
    layout_summary = v3d.summarize_layouts(
        sample_rows, "fno_pbb_discrepancy", "fno_geometry"
    )
    ledger = call_ledger(
        stop_rows, context["baseline_selection"], sorted(set(split_names))
    )
    primary_vs_landweber = {
        split: find_pair(
            pairwise, split, "fno_pbb_discrepancy", "fno_geometry"
        )
        for split in sorted(set(split_names))
    }
    ncp_vs_landweber = {
        split: find_pair(pairwise, split, "fno_pbb_ncp", "fno_geometry")
        for split in sorted(set(split_names))
    }
    primary_calls = {
        str(row["source_split"]): row
        for row in ledger
        if row["method"] == "fno_pbb_discrepancy"
    }
    gate = config["gate"]
    lock = primary_vs_landweber["val_lock"]
    noise = primary_vs_landweber["test_noise_ood"]
    gate_checks = {
        "lock_mean_gain_nonnegative": float(lock["mean_field_gain_pct"])
        >= float(gate["minimum_lock_mean_gain_vs_fixed_landweber_pct"]),
        "noise_ood_mean_gain_nonnegative": float(noise["mean_field_gain_pct"])
        >= float(gate["minimum_noise_ood_mean_gain_vs_fixed_landweber_pct"]),
        "lock_tail_safe": float(lock["harm_rate_gt_1pct"])
        <= float(gate["maximum_lock_harm_rate_gt_1pct"]),
        "noise_ood_tail_safe": float(noise["harm_rate_gt_1pct"])
        <= float(gate["maximum_noise_ood_harm_rate_gt_1pct"]),
        "every_layout_mean_safe": all(
            float(row["mean_field_gain_pct"])
            >= float(gate["minimum_each_layout_mean_gain_vs_fixed_landweber_pct"])
            for row in layout_summary
        ),
        "maximum_call_budget_respected": max(
            int(row["maximum_total_operator_calls"])
            for row in primary_calls.values()
        )
        <= int(gate["maximum_worst_operator_calls"]),
        "lock_median_call_budget_respected": float(
            primary_calls["val_lock"]["median_total_operator_calls"]
        )
        <= float(gate["maximum_median_operator_calls"]),
        "fresh_blind_confirmation_present": False,
        "real_sensor_noise_calibration_present": False,
    }
    deterministic_development_gate = all(
        value
        for key, value in gate_checks.items()
        if key
        not in {
            "fresh_blind_confirmation_present",
            "real_sensor_noise_calibration_present",
        }
    )
    learned_stopping_authorized = all(gate_checks.values())
    status = (
        "DISCREPANCY_RESCUES_MEAN_NOISE_OOD_BUT_TAIL_OR_FRESH_GATE_BLOCKS_LEARNING"
        if float(noise["mean_field_gain_pct"]) >= 0.0
        else "DETERMINISTIC_NOISE_STOPPING_STILL_BEHIND_FIXED_LANDWEBER"
    )

    output_dir = ROOT / "results" / str(config["audit_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    v3d.write_csv(output_dir / PUBLIC_FILES[0], pair_manifest)
    v3d.write_csv(output_dir / PUBLIC_FILES[1], validation_manifest)
    v3d.write_csv(output_dir / PUBLIC_FILES[2], sample_rows)
    v3d.write_csv(output_dir / PUBLIC_FILES[3], summary)
    v3d.write_csv(output_dir / PUBLIC_FILES[4], pairwise)
    v3d.write_csv(output_dir / PUBLIC_FILES[5], layout_summary)
    v3d.write_csv(output_dir / PUBLIC_FILES[6], stop_rows)
    v3d.write_csv(output_dir / PUBLIC_FILES[7], ledger)
    plot_results(output_dir / PUBLIC_FILES[10], summary, pairwise, ledger)

    dashboard = {
        "experiment": config["name"],
        "scientific_status": status,
        "deterministic_development_gate_passed": deterministic_development_gate,
        "learned_stopping_authorized": learned_stopping_authorized,
        "selected_stopping_rules": selected,
        "ncp_white_noise_reference": context["ncp_thresholds"],
        "primary_method": "fno_pbb_discrepancy",
        "primary_vs_fixed_landweber": primary_vs_landweber,
        "ncp_vs_fixed_landweber": ncp_vs_landweber,
        "gate_checks": gate_checks,
        "gate_thresholds": gate,
        "operator_call_ledger": ledger,
        "split_summary": summary,
        "pairwise_summary": pairwise,
        "layout_summary": layout_summary,
        "interpretation": {
            "positive": "active-observation discrepancy stopping repairs the mean noise-OOD failure while using fewer mean operator calls than fixed PBB",
            "negative": "NCP alone mistakes structured thin-front residuals for nonwhite noise and can stop too early on morphology OOD",
            "mechanism": "residual magnitude carries useful noise information, while whiteness is confounded by unresolved signal morphology in this small BOST forward model",
            "camera_balance": "the selected max-camera constraint is nonbinding on V_tune; it is not evidence for a new camera-aware algorithm",
        },
        "next_decision": {
            "learned_stopping_development_training_authorized": False,
            "reason": "existing domains are development evidence and the tail/fresh/real-calibration gates remain binding",
            "next_algorithmic_control": "prewhitened heteroscedastic discrepancy with real or independently simulated camera covariance",
            "next_model_candidate": "a small deployment-input risk head may be tested only after fresh deterministic confirmation",
            "required_fresh_lock": "new fields, noise realizations, layouts, and an independently frozen sensor-noise model",
        },
        "novelty_boundary": {
            "morozov_discrepancy_is_not_novel": True,
            "ncp_residual_whiteness_is_not_novel": True,
            "current_combination_claimed_novel": False,
            "candidate_thesis_contribution": "BOST-specific geometry and covariance conditioned stopping with fair forward-adjoint accounting, if fresh data confirm it",
        },
        "claims_boundary": config["claims_boundary"],
    }
    (output_dir / PUBLIC_FILES[8]).write_text(
        json.dumps(dashboard, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report = {
        "status": status,
        "dashboard": dashboard,
        "protocol": {
            "selection_program": "run_v3k_f_noise_stopping_select.py",
            "audit_program": "run_v3k_f_noise_stopping_audit.py",
            "selection_commit_written_before_test_construction": True,
            "validation_grouping": "whole source fields; four layouts never cross V_tune/V_lock",
            "residual_check": "x_0 through x_63 forward is reused for the corresponding gradient; forced x_64 cap adds no diagnostic forward",
            "call_contract": "stop at x_k<64 costs (k+1) A and k A^T; forced x_64 costs 64 A and 64 A^T",
            "statistical_unit": "source field after four-layout collapse",
            "wall_time_speedup_claimed": False,
        },
        "provenance": {
            "config_sha256": v3d.sha256(args.config),
            "selection_commit_sha256": v3d.sha256(selection_path),
            "pbb_selection_commit_sha256": v3d.sha256(context["pbb_path"]),
            "baseline_selection_commit_sha256": v3d.sha256(context["baseline_path"]),
            "private_dataset_sha256": v3d.sha256(context["private_path"]),
            "base_checkpoint_sha256": v3d.sha256(context["checkpoint_path"]),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "public_assets": PUBLIC_FILES,
        "private_assets": {
            "private_dataset_published": False,
            "checkpoint_weights_published": False,
            "new_checkpoint_count": 0,
        },
    }
    (output_dir / PUBLIC_FILES[9]).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    checksums = [
        f"{v3d.sha256(output_dir / name)}  {name}" for name in PUBLIC_FILES
    ]
    (output_dir / "v3k_f_noise_stopping_checksums.sha256").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": status,
                "deterministic_development_gate_passed": deterministic_development_gate,
                "learned_stopping_authorized": learned_stopping_authorized,
                "primary_vs_fixed_landweber": primary_vs_landweber,
                "gate_checks": gate_checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
