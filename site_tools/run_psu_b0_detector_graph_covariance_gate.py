#!/usr/bin/env python3
"""Plan flow-off acquisition with a graph-covariance calibration gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import resource
import time
from typing import Any, Iterable

import numpy as np

from demo_t16_operator.detector_graph_covariance import (
    empirical_covariance_rank_upper_bound,
    evaluate_covariance_fit,
    fit_amplitude_modulated_graph_covariance,
    fit_component_iid_covariance,
    fit_diagonal_shrinkage_covariance,
    fit_gated_graph_covariance,
    fit_graph_separable_covariance,
    fit_low_rank_drift_covariance,
    detector_graph_spectral_basis,
    simulate_flowoff_repeats,
)
from demo_t16_operator.psu_b0_detector_graph_features import (
    build_detector_knn_graph,
    detector_graph_diagnostics,
)
from site_tools.audit_psu_flowoff_repeat_inventory import (
    summarize_archive_index,
)
from site_tools.run_psu_b0_real_interface_audit import (
    load_real_support_geometry,
)
from site_tools.run_psu_b0_spectral_preconditioner_pilot import (
    _load_json,
)


SCHEMA = "psu-b0-detector-graph-covariance-gate-report-1.0"
STATUS = "DG_COVGATE_ACQUISITION_PLANNING_COMPLETE_SYNTHETIC_ONLY"


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty metric")
    return {
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.1)),
        "p90": float(np.quantile(array, 0.9)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["truth_family"]),
            int(row["repeat_count"]),
            str(row["model"]),
        )
        groups.setdefault(key, []).append(row)
    output = []
    for key in sorted(groups, key=lambda item: (item[1], item[0], item[2])):
        family, repeat_count, model = key
        selected = groups[key]
        gain = _quantiles(
            float(row["nll_gain_vs_component_iid"]) for row in selected
        )
        harm = _quantiles(
            max(0.0, -float(row["nll_gain_vs_component_iid"]))
            for row in selected
        )
        coverage_error = _quantiles(
            float(row["absolute_coverage_error"]) for row in selected
        )
        energy_error = _quantiles(
            float(row["absolute_log_energy_ratio"]) for row in selected
        )
        output.append(
            {
                "truth_family": family,
                "repeat_count": repeat_count,
                "model": model,
                "trial_count": len(selected),
                "nll_gain_vs_component_iid": gain,
                "nll_harm_vs_component_iid": harm,
                "absolute_coverage_error": coverage_error,
                "absolute_log_energy_ratio": energy_error,
                "graph_activation_rate": float(
                    np.mean(
                        [
                            bool(row["graph_activated"])
                            for row in selected
                        ]
                    )
                ),
                "amplitude_activation_rate": float(
                    np.mean(
                        [
                            bool(row["amplitude_activated"])
                            for row in selected
                        ]
                    )
                ),
                "low_rank_drift_activation_rate": float(
                    np.mean(
                        [
                            bool(row["low_rank_drift_activated"])
                            for row in selected
                        ]
                    )
                ),
                "selected_spatial_fraction": _quantiles(
                    float(row["selected_spatial_fraction"])
                    for row in selected
                ),
            }
        )
    return output


def _acquisition_decision(
    summaries: list[dict[str, Any]],
    *,
    repeat_counts: list[int],
    criteria: dict[str, Any],
) -> dict[str, Any]:
    lookup = {
        (
            str(row["truth_family"]),
            int(row["repeat_count"]),
            str(row["model"]),
        ): row
        for row in summaries
    }
    rows = []
    for repeat_count in repeat_counts:
        gated = {
            family: lookup[(family, repeat_count, "dg_covgate")]
            for family in (
                "component_iid",
                "graph_heat",
                "nonstationary_drift",
            )
        }
        heat = gated["graph_heat"]
        iid = gated["component_iid"]
        maximum_coverage_error = max(
            row["absolute_coverage_error"]["p90"]
            for row in gated.values()
        )
        checks = {
            "graph_heat_median_gain": (
                heat["nll_gain_vs_component_iid"]["median"]
                >= float(criteria["minimum_graph_heat_median_nll_gain"])
            ),
            "graph_heat_p10_nonnegative": (
                heat["nll_gain_vs_component_iid"]["p10"]
                >= float(criteria["minimum_graph_heat_p10_nll_gain"])
            ),
            "iid_p90_harm": (
                iid["nll_harm_vs_component_iid"]["p90"]
                <= float(criteria["maximum_iid_p90_nll_harm"])
            ),
            "graph_heat_activation": (
                heat["graph_activation_rate"]
                >= float(criteria["minimum_graph_heat_activation_rate"])
            ),
            "iid_false_activation": (
                iid["graph_activation_rate"]
                <= float(criteria["maximum_iid_activation_rate"])
            ),
            "all_family_coverage": (
                maximum_coverage_error
                <= float(criteria["maximum_all_family_p90_coverage_error"])
            ),
        }
        rows.append(
            {
                "repeat_count": repeat_count,
                "checks": checks,
                "all_checks_pass": bool(all(checks.values())),
                "graph_heat_nll_gain_median": heat[
                    "nll_gain_vs_component_iid"
                ]["median"],
                "graph_heat_nll_gain_p10": heat[
                    "nll_gain_vs_component_iid"
                ]["p10"],
                "iid_nll_harm_p90": iid[
                    "nll_harm_vs_component_iid"
                ]["p90"],
                "graph_heat_activation_rate": heat[
                    "graph_activation_rate"
                ],
                "iid_activation_rate": iid["graph_activation_rate"],
                "maximum_all_family_coverage_error_p90": (
                    maximum_coverage_error
                ),
            }
        )
    passing = [
        row["repeat_count"] for row in rows if row["all_checks_pass"]
    ]
    minimum = min(passing) if passing else None
    return {
        "criteria": criteria,
        "repeat_count_rows": rows,
        "minimum_repeat_count_passing_synthetic_gate": minimum,
        "recommended_flowoff_repeat_count": (
            minimum if minimum is not None else f">{max(repeat_counts)}"
        ),
        "recommended_reserved_validation_fraction": 0.25,
        "real_data_authorized": False,
        "interpretation": (
            "A passing count is only a synthetic acquisition-planning result "
            "on measured detector geometry. Real use still requires a sealed "
            "flow-off validation block and residual-correlation checks."
        ),
    }


def run_experiment(
    *,
    root: Path,
    config_path: Path,
    view_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    config = _load_json(config_path)
    detector_config = _load_json(
        root / str(config["source_detector_config"])
    )
    graph_config = detector_config["detector_graph"]
    geometry_config = config["geometry"]
    view_count = int(geometry_config["view_count"])
    rays_per_view = int(geometry_config["rays_per_view"])
    geometry, _ = load_real_support_geometry(
        view_root,
        rays_per_view=rays_per_view,
        sample_count=int(geometry_config["finite_aperture_sample_count"]),
    )
    graph = build_detector_knn_graph(
        geometry["detector_xy"],
        view_count=view_count,
        rays_per_view=rays_per_view,
        neighbor_count=int(graph_config["neighbor_count"]),
        least_squares_ridge=float(graph_config["least_squares_ridge"]),
    )
    graph_grid = {
        key: tuple(float(value) for value in config["graph_grid"][key])
        for key in (
            "diffusion_times",
            "spatial_fractions",
            "correlations",
            "variance_ratios",
        )
    }
    simulation = config["simulation"]
    repeat_counts = [
        int(value) for value in simulation["repeat_counts"]
    ]
    test_repeat_count = int(simulation["sealed_test_repeat_count"])
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    truth_metadata: dict[str, Any] = {}

    for view in range(view_count):
        laplacian_eigenvalues, eigenvectors = (
            detector_graph_spectral_basis(graph, view_index=view)
        )
        detector_xy = (
            graph["detector_xy"][view].detach().cpu().numpy()
        )
        for family_index, family in enumerate(simulation["families"]):
            for repeat_count in repeat_counts:
                for seed_index in range(int(simulation["seed_count"])):
                    seed = (
                        int(simulation["base_seed"])
                        + 1_000_000 * family_index
                        + 10_000 * repeat_count
                        + 100 * seed_index
                        + view
                    )
                    rng = np.random.default_rng(seed)
                    calibration, metadata = simulate_flowoff_repeats(
                        family=str(family),
                        repeat_count=repeat_count,
                        rng=rng,
                        laplacian_eigenvalues=laplacian_eigenvalues,
                        eigenvectors=eigenvectors,
                        detector_xy=detector_xy,
                    )
                    sealed_test, _ = simulate_flowoff_repeats(
                        family=str(family),
                        repeat_count=test_repeat_count,
                        rng=rng,
                        laplacian_eigenvalues=laplacian_eigenvalues,
                        eigenvectors=eigenvectors,
                        detector_xy=detector_xy,
                    )
                    truth_metadata[str(family)] = metadata
                    iid_fit = fit_component_iid_covariance(
                        calibration,
                        laplacian_eigenvalues=laplacian_eigenvalues,
                        eigenvectors=eigenvectors,
                        correlations=graph_grid["correlations"],
                        variance_ratios=graph_grid["variance_ratios"],
                    )
                    diagonal_fit = fit_diagonal_shrinkage_covariance(
                        calibration,
                        prior_strength=float(
                            config["diagonal_shrinkage"]["prior_strength"]
                        ),
                    )
                    graph_fit = fit_graph_separable_covariance(
                        calibration,
                        laplacian_eigenvalues=laplacian_eigenvalues,
                        eigenvectors=eigenvectors,
                        diffusion_times=graph_grid["diffusion_times"],
                        spatial_fractions=graph_grid["spatial_fractions"],
                        correlations=graph_grid["correlations"],
                        variance_ratios=graph_grid["variance_ratios"],
                    )
                    amplitude_fit = (
                        fit_amplitude_modulated_graph_covariance(
                            calibration,
                            laplacian_eigenvalues=laplacian_eigenvalues,
                            eigenvectors=eigenvectors,
                            graph_grid=graph_grid,
                            smoothing_strengths=config["gated_model"][
                                "amplitude_smoothing_strengths"
                            ],
                        )
                    )
                    low_rank_fit = fit_low_rank_drift_covariance(
                        calibration,
                        laplacian_eigenvalues=laplacian_eigenvalues,
                        eigenvectors=eigenvectors,
                        graph_grid=graph_grid,
                        amplitude_smoothing_strengths=config[
                            "gated_model"
                        ]["amplitude_smoothing_strengths"],
                        rank_options=config["gated_model"][
                            "low_rank_drift"
                        ]["rank_options"],
                        shrinkage_strengths=config["gated_model"][
                            "low_rank_drift"
                        ]["shrinkage_strengths"],
                        base_amplitude_fit=amplitude_fit,
                    )
                    gated_fit, gate = fit_gated_graph_covariance(
                        calibration,
                        laplacian_eigenvalues=laplacian_eigenvalues,
                        eigenvectors=eigenvectors,
                        graph_grid=graph_grid,
                        validation_fraction=float(
                            config["gated_model"]["validation_fraction"]
                        ),
                        minimum_validation_gain_per_dimension=float(
                            config["gated_model"][
                                "minimum_validation_gain_per_dimension"
                            ]
                        ),
                        full_graph_fit=graph_fit,
                        full_iid_fit=iid_fit,
                        amplitude_smoothing_strengths=config[
                            "gated_model"
                        ]["amplitude_smoothing_strengths"],
                        full_amplitude_fit=amplitude_fit,
                        low_rank_drift=config["gated_model"][
                            "low_rank_drift"
                        ],
                        full_low_rank_fit=low_rank_fit,
                    )
                    fits = (
                        ("component_iid", iid_fit, False),
                        ("diagonal_shrinkage", diagonal_fit, False),
                        ("always_graph", graph_fit, True),
                        (
                            "always_amplitude_graph",
                            amplitude_fit,
                            True,
                        ),
                        (
                            "always_low_rank_drift",
                            low_rank_fit,
                            True,
                        ),
                        (
                            "dg_covgate",
                            gated_fit,
                            bool(gate["graph_activated"]),
                        ),
                    )
                    evaluations = {
                        model: evaluate_covariance_fit(
                            fit,
                            sealed_test,
                            eigenvectors=eigenvectors,
                        )
                        for model, fit, _ in fits
                    }
                    baseline_nll = evaluations["component_iid"][
                        "mean_nll_per_dimension"
                    ]
                    for model, fit, activated in fits:
                        evaluation = evaluations[model]
                        rows.append(
                            {
                                "truth_family": str(family),
                                "repeat_count": repeat_count,
                                "seed_index": seed_index,
                                "view_index": view,
                                "model": model,
                                "mean_nll_per_dimension": evaluation[
                                    "mean_nll_per_dimension"
                                ],
                                "nll_gain_vs_component_iid": (
                                    baseline_nll
                                    - evaluation["mean_nll_per_dimension"]
                                ),
                                "whitened_energy_ratio": evaluation[
                                    "whitened_energy_ratio"
                                ],
                                "absolute_log_energy_ratio": evaluation[
                                    "absolute_log_energy_ratio"
                                ],
                                "chi_square_95_coverage": evaluation[
                                    "chi_square_95_coverage"
                                ],
                                "absolute_coverage_error": evaluation[
                                    "absolute_coverage_error"
                                ],
                                "graph_activated": activated,
                                "amplitude_activated": bool(
                                    gate["amplitude_activated"]
                                    if model == "dg_covgate"
                                    else model
                                    == "always_amplitude_graph"
                                ),
                                "low_rank_drift_activated": bool(
                                    gate["low_rank_drift_activated"]
                                    if model == "dg_covgate"
                                    else model == "always_low_rank_drift"
                                ),
                                "selected_spatial_fraction": float(
                                    fit.parameters.get(
                                        "spatial_fraction",
                                        0.0,
                                    )
                                ),
                                "selected_diffusion_time": float(
                                    fit.parameters.get(
                                        "diffusion_time",
                                        0.0,
                                    )
                                ),
                                "selected_component_correlation": float(
                                    fit.parameters.get(
                                        "component_correlation",
                                        0.0,
                                    )
                                ),
                                "selected_component_variance_ratio": float(
                                    fit.parameters.get(
                                        "component_variance_ratio",
                                        1.0,
                                    )
                                ),
                                "selected_amplitude_log_standard_deviation": float(
                                    fit.parameters.get(
                                        "amplitude_log_standard_deviation",
                                        0.0,
                                    )
                                ),
                                "selected_low_rank_eigenvalue_sum": float(
                                    fit.parameters.get(
                                        "retained_low_rank_eigenvalue_sum",
                                        0.0,
                                    )
                                ),
                                "gate_validation_gain_per_dimension": float(
                                    gate[
                                        "validation_nll_gain_per_dimension"
                                    ]
                                    if model == "dg_covgate"
                                    else 0.0
                                ),
                            }
                        )

    summaries = _summarize_rows(rows)
    archive_summary = summarize_archive_index(
        (root / str(config["source_archive_index"])).read_text(
            encoding="utf-8",
            errors="replace",
        )
    )
    dimension = 2 * rays_per_view
    acquisition = _acquisition_decision(
        summaries,
        repeat_counts=repeat_counts,
        criteria=config["acquisition_gate"],
    )
    report = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "evidence_scope": (
            "REAL_PSU_NINE_VIEW_DETECTOR_GRAPH_WITH_SYNTHETIC_FLOWOFF_"
            "COVARIANCE_FAMILIES_NO_REAL_TEMPORAL_REPEATS_NO_RECONSTRUCTION"
        ),
        "configuration": config,
        "detector_graph_diagnostics": detector_graph_diagnostics(graph),
        "public_archive_repeat_inventory": archive_summary,
        "truth_family_metadata": truth_metadata,
        "empirical_covariance_rank_ceiling": [
            {
                "repeat_count": count,
                "measurement_dimension_per_camera": dimension,
                "maximum_centered_empirical_covariance_rank": (
                    empirical_covariance_rank_upper_bound(
                        repeat_count=count,
                        dimension=dimension,
                    )
                ),
                "rank_fraction": float(
                    empirical_covariance_rank_upper_bound(
                        repeat_count=count,
                        dimension=dimension,
                    )
                    / dimension
                ),
            }
            for count in repeat_counts
        ],
        "summary": summaries,
        "acquisition_gate": acquisition,
        "runtime": {
            "wall_seconds": float(time.perf_counter() - started),
            "maximum_rss_bytes": int(_max_rss_bytes()),
            "trial_rows": len(rows),
        },
        "claim_boundary": config["claim_boundary"],
    }
    return report, rows, summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "demo_t16_operator/configs/"
            "psu_b0_detector_graph_covariance_gate_v1.json"
        ),
    )
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "demo_t16_operator/results/"
            "psu_b0_detector_graph_covariance_gate"
        ),
    )
    args = parser.parse_args()
    report, rows, summaries = run_experiment(
        root=args.root,
        config_path=args.config,
        view_root=args.view_root,
    )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    rows_path = output / "trial_rows.csv"
    summary_path = output / "summary_rows.csv"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    _write_csv(rows_path, rows)
    flattened = []
    for row in summaries:
        flattened.append(
            {
                "truth_family": row["truth_family"],
                "repeat_count": row["repeat_count"],
                "model": row["model"],
                "trial_count": row["trial_count"],
                "nll_gain_median": row[
                    "nll_gain_vs_component_iid"
                ]["median"],
                "nll_gain_p10": row[
                    "nll_gain_vs_component_iid"
                ]["p10"],
                "nll_gain_p90": row[
                    "nll_gain_vs_component_iid"
                ]["p90"],
                "nll_harm_p90": row[
                    "nll_harm_vs_component_iid"
                ]["p90"],
                "coverage_error_p90": row[
                    "absolute_coverage_error"
                ]["p90"],
                "energy_error_median": row[
                    "absolute_log_energy_ratio"
                ]["median"],
                "graph_activation_rate": row[
                    "graph_activation_rate"
                ],
                "amplitude_activation_rate": row[
                    "amplitude_activation_rate"
                ],
                "low_rank_drift_activation_rate": row[
                    "low_rank_drift_activation_rate"
                ],
                "spatial_fraction_median": row[
                    "selected_spatial_fraction"
                ]["median"],
            }
        )
    _write_csv(summary_path, flattened)
    checksums = {
        path.name: _sha256(path)
        for path in (report_path, rows_path, summary_path)
    }
    (output / "checksums.sha256").write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(checksums.items())
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output_dir": str(output),
                "wall_seconds": report["runtime"]["wall_seconds"],
                "minimum_repeat_count_passing_synthetic_gate": report[
                    "acquisition_gate"
                ]["minimum_repeat_count_passing_synthetic_gate"],
                "trial_rows": len(rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
