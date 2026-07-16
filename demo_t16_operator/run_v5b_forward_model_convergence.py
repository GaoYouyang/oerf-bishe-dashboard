#!/usr/bin/env python3
"""Audit finite-aperture renderer convergence before fitting any field model.

This deterministic development audit asks whether an off-grid high-resolution
truth operator is closest to the physically nearest radius in a lower-cost
candidate bank.  It separates native renderer normalization, one shared
physical scale, and a profiled scalar gain.  No field, noise, neural model, or
confirmatory claim is involved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .finite_aperture_bost import (
        build_finite_aperture_operator_bank,
        finite_aperture_reference_scale,
    )
except ImportError:
    from finite_aperture_bost import (
        build_finite_aperture_operator_bank,
        finite_aperture_reference_scale,
    )


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "v5b_forward_model_convergence.json"
DEFAULT_OUTPUT = ROOT / "results" / "v5b_forward_model_convergence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _flatten_views(operator: np.ndarray, views: Sequence[int]) -> np.ndarray:
    matrix = np.asarray(operator, dtype=np.float64)
    if matrix.ndim != 4:
        raise ValueError("operator must have shape [depth,view,detector,voxel]")
    indices = np.asarray(tuple(int(value) for value in views), dtype=int)
    if indices.size == 0 or np.any(indices < 0) or np.any(indices >= matrix.shape[1]):
        raise ValueError("views are empty or outside the operator")
    return matrix[:, indices].reshape(-1)


def rank_operator_bank(
    truth_operator: np.ndarray,
    candidate_bank: np.ndarray,
    views: Sequence[int],
    *,
    candidate_reference_scale: float,
    truth_reference_scale: float,
) -> dict[str, np.ndarray]:
    """Return native, shared-scale and scalar-gain-profiled distances."""

    bank = np.asarray(candidate_bank, dtype=np.float64)
    if bank.ndim != 5:
        raise ValueError("candidate_bank must have a leading radius dimension")
    truth = _flatten_views(truth_operator, views)
    denominator = max(float(np.linalg.norm(truth)), 1e-15)
    scale_ratio = float(candidate_reference_scale) / float(truth_reference_scale)
    if not np.isfinite(scale_ratio) or scale_ratio <= 0.0:
        raise ValueError("reference scales must be finite and strictly positive")

    native: list[float] = []
    shared: list[float] = []
    profiled: list[float] = []
    gains: list[float] = []
    for operator in bank:
        candidate = _flatten_views(operator, views)
        native.append(float(np.linalg.norm(candidate - truth) / denominator))
        shared_candidate = scale_ratio * candidate
        shared.append(float(np.linalg.norm(shared_candidate - truth) / denominator))
        gain = float(np.dot(candidate, truth) / max(np.dot(candidate, candidate), 1e-30))
        gains.append(gain)
        profiled.append(float(np.linalg.norm(gain * candidate - truth) / denominator))
    return {
        "native": np.asarray(native),
        "shared_scale": np.asarray(shared),
        "profiled_gain": np.asarray(profiled),
        "profiled_gain_values": np.asarray(gains),
    }


def _second_margin(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    if len(ordered) < 2:
        raise ValueError("at least two candidate radii are required")
    return float(ordered[1] - ordered[0])


def run_audit(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    n, depth = int(config["grid_size"]), int(config["depth"])
    candidate_radii = np.asarray(config["candidate_aperture_radii"], dtype=float)
    truth_radii = np.asarray(config["true_aperture_radii"], dtype=float)
    rows: list[dict[str, Any]] = []

    for rig in config["rigs"]:
        angles = np.asarray(rig["angles_degrees"], dtype=float)
        views = tuple(int(value) for value in rig["fit_camera_indices"])
        common = {
            "cone_u": float(rig["cone_u"]),
            "cone_z": float(rig["cone_z"]),
            "bend": float(rig["bend"]),
        }
        truth_path = int(rig["truth_path_samples"])
        truth_aperture = int(rig["truth_aperture_samples"])
        truth_scale = finite_aperture_reference_scale(
            n,
            depth,
            angles,
            aperture_samples=truth_aperture,
            path_samples=truth_path,
            **common,
        )
        truth_bank = build_finite_aperture_operator_bank(
            n,
            depth,
            angles,
            truth_radii,
            aperture_samples=truth_aperture,
            path_samples=truth_path,
            normalization_scale=truth_scale,
            **common,
        )
        for path_samples in config["reconstruction_path_samples"]:
            for aperture_samples in config["reconstruction_aperture_samples"]:
                reconstruction_scale = finite_aperture_reference_scale(
                    n,
                    depth,
                    angles,
                    aperture_samples=int(aperture_samples),
                    path_samples=int(path_samples),
                    **common,
                )
                reconstruction_bank = build_finite_aperture_operator_bank(
                    n,
                    depth,
                    angles,
                    candidate_radii,
                    aperture_samples=int(aperture_samples),
                    path_samples=int(path_samples),
                    normalization_scale=reconstruction_scale,
                    **common,
                )
                for truth_index, true_radius in enumerate(truth_radii):
                    nearest = int(np.argmin(np.abs(candidate_radii - true_radius)))
                    metrics = rank_operator_bank(
                        truth_bank[truth_index],
                        reconstruction_bank,
                        views,
                        candidate_reference_scale=reconstruction_scale,
                        truth_reference_scale=truth_scale,
                    )
                    row: dict[str, Any] = {
                        "rig_id": str(rig["id"]),
                        "true_aperture_radius": float(true_radius),
                        "nearest_candidate_radius": float(candidate_radii[nearest]),
                        "reconstruction_path_samples": int(path_samples),
                        "reconstruction_aperture_samples": int(aperture_samples),
                        "truth_path_samples": truth_path,
                        "truth_aperture_samples": truth_aperture,
                        "reconstruction_reference_scale": reconstruction_scale,
                        "truth_reference_scale": truth_scale,
                        "reconstruction_over_truth_scale": reconstruction_scale / truth_scale,
                    }
                    for name in ("native", "shared_scale", "profiled_gain"):
                        values = metrics[name]
                        best = int(np.argmin(values))
                        row[f"{name}_best_radius"] = float(candidate_radii[best])
                        row[f"{name}_nearest_match"] = best == nearest
                        row[f"{name}_best_distance"] = float(values[best])
                        row[f"{name}_nearest_distance"] = float(values[nearest])
                        row[f"{name}_nearest_over_best"] = float(
                            values[nearest] / max(values[best], 1e-15)
                        )
                        row[f"{name}_ranking_margin"] = _second_margin(values)
                    gain_values = metrics["profiled_gain_values"]
                    row["profiled_gain_at_best_radius"] = float(
                        gain_values[int(np.argmin(metrics["profiled_gain"]))]
                    )
                    rows.append(row)

    aggregate: list[dict[str, Any]] = []
    for path_samples in config["reconstruction_path_samples"]:
        for aperture_samples in config["reconstruction_aperture_samples"]:
            group = [
                row
                for row in rows
                if row["reconstruction_path_samples"] == int(path_samples)
                and row["reconstruction_aperture_samples"] == int(aperture_samples)
            ]
            value: dict[str, Any] = {
                "reconstruction_path_samples": int(path_samples),
                "reconstruction_aperture_samples": int(aperture_samples),
                "rig_radius_blocks": len(group),
                "mean_absolute_scale_mismatch_percent": float(
                    np.mean(
                        [
                            abs(float(row["reconstruction_over_truth_scale"]) - 1.0)
                            * 100.0
                            for row in group
                        ]
                    )
                ),
            }
            for name in ("native", "shared_scale", "profiled_gain"):
                matches = [bool(row[f"{name}_nearest_match"]) for row in group]
                value[f"{name}_match_rate"] = float(np.mean(matches))
                value[f"{name}_all_blocks_match"] = bool(all(matches))
                value[f"{name}_mean_best_distance"] = float(
                    np.mean([row[f"{name}_best_distance"] for row in group])
                )
                value[f"{name}_minimum_ranking_margin"] = float(
                    np.min([row[f"{name}_ranking_margin"] for row in group])
                )
            aggregate.append(value)
    return rows, aggregate


def write_figure(path: Path, aggregate: list[dict[str, Any]]) -> None:
    path_values = sorted({int(row["reconstruction_path_samples"]) for row in aggregate})
    aperture_values = sorted(
        {int(row["reconstruction_aperture_samples"]) for row in aggregate}
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), constrained_layout=True)
    for axis, name, title in zip(
        axes,
        ("native", "shared_scale", "profiled_gain"),
        ("Native normalization", "Shared physical scale", "Profiled scalar gain"),
        strict=True,
    ):
        matrix = np.zeros((len(aperture_values), len(path_values)), dtype=float)
        for row in aggregate:
            i = aperture_values.index(int(row["reconstruction_aperture_samples"]))
            j = path_values.index(int(row["reconstruction_path_samples"]))
            matrix[i, j] = float(row[f"{name}_match_rate"])
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white" if matrix[i, j] < 0.65 else "black", fontsize=9)
        axis.set_title(title)
        axis.set_xticks(range(len(path_values)), labels=path_values)
        axis.set_yticks(range(len(aperture_values)), labels=aperture_values)
        axis.set_xlabel("Path samples")
        axis.set_ylabel("Aperture samples")
    figure.colorbar(image, ax=axes, shrink=0.82, label="Nearest-radius match rate")
    figure.suptitle("v5b forward-model convergence gate (6 rig-radius blocks)")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rows, aggregate = run_audit(config)
    write_csv(output / "operator_convergence_rows.csv", rows)
    write_csv(output / "operator_convergence_aggregate.csv", aggregate)
    write_figure(output / "v5b_forward_model_convergence.png", aggregate)
    (output / "config_snapshot.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    safe = [
        row
        for row in aggregate
        if row["native_all_blocks_match"]
        and row["shared_scale_all_blocks_match"]
        and row["profiled_gain_all_blocks_match"]
    ]
    ranked = sorted(
        aggregate,
        key=lambda row: (
            min(
                row["native_match_rate"],
                row["shared_scale_match_rate"],
                row["profiled_gain_match_rate"],
            ),
            row["native_match_rate"]
            + row["shared_scale_match_rate"]
            + row["profiled_gain_match_rate"],
            -row["profiled_gain_mean_best_distance"],
            -row["reconstruction_aperture_samples"],
            -row["reconstruction_path_samples"],
        ),
        reverse=True,
    )
    report = {
        "claim_status": config["claim_status"],
        "evidence_label": config["evidence_label"],
        "claim_boundary": (
            "deterministic 8x8x5 prescribed weak-deflection operator audit; no field, "
            "noise, reconstruction, neural operator, nonlinear ray tracing, CFD, real "
            "BOST or confirmatory superiority claim"
        ),
        "scientific_question": (
            "Is the reconstruction renderer converged enough that off-grid high-"
            "resolution truth is ranked by the physically nearest candidate radius?"
        ),
        "row_count": len(rows),
        "deterministic_rig_radius_blocks": len(config["rigs"])
        * len(config["true_aperture_radii"]),
        "aggregate_setting_count": len(aggregate),
        "safe_setting_count": len(safe),
        "safe_settings": safe,
        "descriptive_best_setting": ranked[0],
        "interpretation": (
            "A joint field/radius algorithm must not be tuned until at least one "
            "renderer setting ranks every rig-radius block correctly under native, "
            "shared-scale and profiled-gain distances. This is a numerical convergence "
            "gate, not evidence that radius is identifiable from noisy BOS data."
        ),
        "source_hashes": {
            "runner": sha256(Path(__file__)),
            "operator": sha256(ROOT / "finite_aperture_bost.py"),
            "config": sha256(args.config),
        },
    }
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    artifact_paths = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "checksums.sha256"
    )
    (output / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in artifact_paths),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
