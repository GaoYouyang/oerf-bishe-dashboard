#!/usr/bin/env python3
"""Plot the PSU B0 interface gate from public aggregate and synthetic artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PUBLIC_SCHEMA = "psu-b0-real-support-interface-public-summary-1.0"
FIXTURE_SCHEMA = "psu-b0-reconstruction-interface-fixture-1.0"
FIGURE_SCHEMA = "psu-b0-reconstruction-interface-figure-1.0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path, schema: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != schema:
        raise ValueError(f"unexpected schema in {path.name}")
    return value


def build_figure(
    *,
    public_summary_path: Path,
    fixture_report_path: Path,
    fixture_arrays_path: Path,
    output_stem: Path,
) -> dict[str, Any]:
    public = _load_json(public_summary_path, PUBLIC_SCHEMA)
    fixture = _load_json(fixture_report_path, FIXTURE_SCHEMA)
    arrays = np.load(fixture_arrays_path, allow_pickle=False)
    residual = np.asarray(arrays["residual_history"], dtype=np.float64)
    if residual.ndim != 1 or residual.size < 2 or np.any(residual <= 0):
        raise ValueError("fixture residual history must be positive and one-dimensional")
    profiles = public["grid_profiles"]
    if len(profiles) != 2:
        raise ValueError("the frozen figure expects 16^3 and 32^3 profiles")

    labels: list[str] = []
    errors: list[float] = []
    colors: list[str] = []
    for row in profiles:
        size = int(row["grid_shape_zyx"][0])
        labels.extend((f"{size} CPU64", f"{size} MPS32"))
        errors.extend(
            (
                float(row["cpu_float64_adjoint_relative_error"]),
                float(row["mps_float32_adjoint_relative_error"]),
            )
        )
        colors.extend(("#315f93", "#16817a"))

    figure, axes = plt.subplots(1, 4, figsize=(14.2, 3.65), constrained_layout=True)
    axes[0].bar(np.arange(len(errors)), errors, color=colors, width=0.68)
    axes[0].set_yscale("log")
    axes[0].axhline(1e-11, color="#315f93", linestyle="--", linewidth=1)
    axes[0].axhline(2e-5, color="#16817a", linestyle=":", linewidth=1)
    axes[0].set_xticks(np.arange(len(labels)), labels, rotation=28, ha="right")
    axes[0].set_ylabel("Relative dot-product defect")
    axes[0].set_title("A. Exact adjoint gate")
    axes[0].grid(axis="y", alpha=0.22)

    x = np.arange(len(profiles))
    width = 0.19
    runtime_rows = {
        "CPU F": [
            1000.0 * float(row["cpu_profile"]["forward_seconds_median"])
            for row in profiles
        ],
        "CPU Aᵀ": [
            1000.0 * float(row["cpu_profile"]["adjoint_seconds_median"])
            for row in profiles
        ],
        "MPS F": [
            1000.0 * float(row["mps_profile"]["forward_seconds_median"])
            for row in profiles
        ],
        "MPS Aᵀ": [
            1000.0 * float(row["mps_profile"]["adjoint_seconds_median"])
            for row in profiles
        ],
    }
    runtime_colors = ("#315f93", "#8ba6c8", "#16817a", "#7fc6b8")
    for index, ((name, values), color) in enumerate(
        zip(runtime_rows.items(), runtime_colors, strict=True)
    ):
        axes[1].bar(
            x + (index - 1.5) * width,
            values,
            width=width,
            label=name,
            color=color,
        )
    axes[1].set_xticks(
        x,
        [f"{int(row['grid_shape_zyx'][0])}³" for row in profiles],
    )
    axes[1].set_ylabel("Median milliseconds")
    axes[1].set_title("B. 2,304-ray call profile")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    axes[1].grid(axis="y", alpha=0.22)

    current_mib = [
        float(row["mps_profile"]["mps_current_allocated_bytes"]) / 1024**2
        for row in profiles
    ]
    driver_mib = [
        float(row["mps_profile"]["mps_driver_allocated_bytes"]) / 1024**2
        for row in profiles
    ]
    axes[2].bar(x - 0.16, current_mib, width=0.32, color="#16817a", label="Current")
    axes[2].bar(x + 0.16, driver_mib, width=0.32, color="#d49b32", label="Driver")
    axes[2].set_xticks(
        x,
        [f"{int(row['grid_shape_zyx'][0])}³" for row in profiles],
    )
    axes[2].set_ylabel("MPS memory snapshot (MiB)")
    axes[2].set_title("C. Small-fixture memory")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(axis="y", alpha=0.22)

    axes[3].semilogy(
        np.arange(residual.size),
        residual,
        color="#8b3f4d",
        linewidth=2.2,
    )
    final_residual = float(fixture["metrics"]["final_measurement_relative_l2"])
    axes[3].scatter(
        [residual.size],
        [final_residual],
        color="#8b3f4d",
        s=28,
        zorder=3,
    )
    axes[3].set_xlabel("Fixed Landweber iteration")
    axes[3].set_ylabel("Synthetic support relative L2")
    axes[3].set_title("D. Interface closure only")
    axes[3].grid(alpha=0.22)
    axes[3].annotate(
        f"final {final_residual:.4f}\nfield L2 {fixture['metrics']['field_relative_l2_fixture_truth_only']:.3f}",
        xy=(residual.size, final_residual),
        xytext=(0.56, 0.45),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#8b3f4d"},
        fontsize=8,
    )

    geometry = public["aggregate_geometry"]
    figure.suptitle(
        "PSU B0 reconstruction interface gate: real support geometry + synthetic inverse",
        fontsize=13,
    )
    figure.text(
        0.5,
        0.01,
        (
            f"Real geometry: {geometry['selected_ray_count']:,} rays, "
            f"{geometry['total_sample_count']:,} fixed-denominator samples. "
            "No development/final audit access; no reconstruction superiority claim."
        ),
        ha="center",
        fontsize=9,
        color="#444444",
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("png", "pdf", "svg"):
        path = output_stem.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=220 if suffix == "png" else None)
        outputs.append(
            {
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    plt.close(figure)
    manifest = {
        "schema_version": FIGURE_SCHEMA,
        "status": "FIGURE_COMPLETE_INTERFACE_GATE_ONLY",
        "inputs": {
            "public_summary": {
                "filename": public_summary_path.name,
                "sha256": _sha256(public_summary_path),
            },
            "fixture_report": {
                "filename": fixture_report_path.name,
                "sha256": _sha256(fixture_report_path),
            },
            "fixture_arrays": {
                "filename": fixture_arrays_path.name,
                "sha256": _sha256(fixture_arrays_path),
            },
        },
        "outputs": outputs,
        "claim_boundary": {
            "real_geometry_inverse_run": False,
            "development_or_final_audit_opened": False,
            "algorithm_superiority": False,
        },
    }
    manifest_path = output_stem.parent / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-summary", type=Path, required=True)
    parser.add_argument("--fixture-report", type=Path, required=True)
    parser.add_argument("--fixture-arrays", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_figure(
        public_summary_path=args.public_summary,
        fixture_report_path=args.fixture_report,
        fixture_arrays_path=args.fixture_arrays,
        output_stem=args.output_stem,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

