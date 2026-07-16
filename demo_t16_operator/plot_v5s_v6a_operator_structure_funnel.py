#!/usr/bin/env python3
"""Plot the operator-structure evidence funnel from v5s through v6a."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "operator_structure_funnel_v5s_v6a.png"


def _report(name: str) -> dict:
    return json.loads((RESULTS / name / "report.json").read_text(encoding="utf-8"))


def main() -> None:
    v5s = _report("v5s_dco_low_rank_screening")
    v5u = _report("v5u_calibrated_renderer_residual_screening")
    v5v = _report("v5v_camera_local_kernel_correction")
    v5w = _report("v5w_clean_aperture_kernel_screening")
    v5x = _report("v5x_ray_conditioned_voxel_kernel")
    v5z = _report("v5z_stabilized_direct_ray_kernel")
    v6a = _report("v6a_ray_kernel_hypernetwork_development")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#b7c5c2",
            "axes.labelcolor": "#26383e",
            "xtick.color": "#53666c",
            "ytick.color": "#53666c",
            "figure.facecolor": "#f3f6f4",
            "axes.facecolor": "#ffffff",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    figure.suptitle(
        "Operator-correction evidence funnel: structure improved, development gate still failed",
        fontsize=16,
        fontweight="bold",
        color="#17252b",
    )

    axis = axes[0, 0]
    labels = ["V5S global\nHOSVD", "V5U calibrated\nHOSVD", "V5V camera\nkernel"]
    candidate = [
        v5s["development_summary"]["gc_biloc_ridge"]["mean_relative_discrepancy_error"],
        v5u["development_summary"]["cal_hosvd_ridge"]["mean_relative_renderer_residual_error"],
        v5v["development_summary"]["camera_local_kernel_geometry_ridge"]["mean_relative_renderer_residual_error"],
    ]
    baseline = [
        v5s["development_summary"]["full_matrix_geometry_ridge"]["mean_relative_discrepancy_error"],
        v5u["development_summary"]["full_matrix_geometry_ridge"]["mean_relative_renderer_residual_error"],
        v5v["development_summary"]["full_matrix_geometry_ridge"]["mean_relative_renderer_residual_error"],
    ]
    x = np.arange(len(labels))
    axis.bar(x - 0.2, baseline, 0.4, color="#146f66", label="full-matrix ridge")
    axis.bar(x + 0.2, candidate, 0.4, color="#b45e52", label="structured candidate")
    axis.set_xticks(x, labels)
    axis.set_ylim(0.0, 1.02)
    axis.set_ylabel("Mean relative discrepancy error")
    axis.set_title("Mixed mismatch: global and camera-level structures both fail")
    axis.grid(axis="y", alpha=0.16)
    axis.legend(fontsize=8)

    axis = axes[0, 1]
    oracle_labels = ["V5S\nglobal", "V5U\ncal-HOSVD", "V5V\nleft kernel", "V5W\nright kernel", "V5X\nray-wise"]
    oracle = [
        v5s["development_summary"]["oracle_shared_subspace"]["mean_relative_discrepancy_error"],
        v5u["development_summary"]["oracle_shared_subspace"]["mean_relative_renderer_residual_error"],
        v5v["development_summary"]["oracle_camera_local_kernel"]["mean_relative_renderer_residual_error"],
        v5w["best_right_oracle"]["mean_error"],
        v5x["best_oracle"]["mean_relative_aperture_error"],
    ]
    bars = axis.bar(np.arange(len(oracle)), oracle, color=["#8a989b", "#6d83a5", "#b45e52", "#8a651b", "#315f93"])
    axis.axhline(0.35, color="#8a651b", linestyle="--", linewidth=1, label="strict oracle reference")
    axis.set_xticks(np.arange(len(oracle)), oracle_labels)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Oracle representation error")
    axis.set_title("Ray-conditioned voxel kernels reveal the first near-signal")
    axis.grid(axis="y", alpha=0.16)
    axis.legend(fontsize=8)
    for bar, value in zip(bars, oracle, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", fontsize=8)

    axis = axes[1, 0]
    clean_labels = ["full ridge", "V5Z linear", "V6A hypernet", "V5X oracle"]
    mean_values = [
        v6a["development_summary"]["full_matrix_geometry_ridge"]["mean_relative_aperture_residual_error"],
        v5z["development_summary"]["direct_ray_conditioned_kernel"]["mean_relative_aperture_residual_error"],
        v6a["development_summary"]["hypernetwork_ensemble"]["mean_relative_aperture_residual_error"],
        v5x["best_oracle"]["mean_relative_aperture_error"],
    ]
    worst_values = [
        v6a["development_summary"]["full_matrix_geometry_ridge"]["worst_relative_aperture_residual_error"],
        v5z["development_summary"]["direct_ray_conditioned_kernel"]["worst_relative_aperture_residual_error"],
        v6a["development_summary"]["hypernetwork_ensemble"]["worst_relative_aperture_residual_error"],
        v5x["best_oracle"]["worst_relative_aperture_error"],
    ]
    x = np.arange(len(clean_labels))
    axis.bar(x - 0.2, mean_values, 0.4, color="#315f93", label="mean")
    axis.bar(x + 0.2, worst_values, 0.4, color="#b45e52", label="worst rig")
    axis.set_xticks(x, clean_labels, rotation=18, ha="right")
    axis.set_ylabel("Finite-aperture residual error")
    axis.set_title("Hypernetwork helps mean and tail, but wins only 6/12 rigs")
    axis.grid(axis="y", alpha=0.16)
    axis.legend(fontsize=8)

    axis = axes[1, 1]
    model_labels = ["full ridge", "V5Z linear", "V6A hypernet"]
    parameters = [
        v6a["model_size"]["full_matrix_predictor_coefficients"],
        v5z["model_size"]["direct_model_coefficients"],
        v6a["model_size"]["parameters_per_seed"],
    ]
    errors = [mean_values[0], mean_values[1], mean_values[2]]
    axis.scatter(parameters, errors, s=[85, 85, 110], color=["#146f66", "#315f93", "#b45e52"])
    for label, x_value, y_value in zip(model_labels, parameters, errors, strict=True):
        axis.annotate(label, (x_value, y_value), xytext=(7, 5), textcoords="offset points", fontsize=9)
    axis.set_xscale("log")
    axis.set_xlim(400, 2_000_000)
    axis.set_ylim(0.70, 0.84)
    axis.set_xlabel("Predictor parameters (log scale)")
    axis.set_ylabel("Mean finite-aperture residual error")
    axis.set_title("Compact models improve the Pareto frontier, not the pass/fail verdict")
    axis.grid(alpha=0.16)

    figure.text(
        0.5,
        0.005,
        "All V5T-V6A structure choices use opened synthetic rigs; no panel is fresh, inverse, or OERF evidence.",
        ha="center",
        fontsize=9,
        color="#53666c",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
