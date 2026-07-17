#!/usr/bin/env python3
"""Audit whether JACRU-M0 interface scores pre-exist before seeing data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from demo_t16_operator.jacru_synthetic_fixture import (
    JACRUSyntheticFixtureConfig,
    build_jacru_synthetic_case,
)
from demo_t16_operator.jump_aware_cone_ray_unrolling import (
    JumpAwareConfig,
    JumpAwareConeRayField,
)
from demo_t16_operator.multi_interface_metrics import multi_interface_level_set_metrics
from demo_t16_operator.psu_b0_streaming_operator import zero_outer_boundary_support


REPORT_SCHEMA = "jacru-m0-data-free-initialization-audit-1.0"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected one JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_config(config: dict[str, Any]) -> JACRUSyntheticFixtureConfig:
    fixture = config["fixture"]
    return JACRUSyntheticFixtureConfig(
        grid_shape=tuple(int(value) for value in fixture["grid_shape"]),
        detector_shape=tuple(int(value) for value in fixture["detector_shape"]),
        samples_per_ray=int(fixture["samples_per_ray"]),
        noise_relative_std=float(fixture["noise_relative_std"]),
        camera_bias_relative_std=float(fixture["camera_bias_relative_std"]),
        enable_noise=bool(fixture["enable_noise"]),
        enable_camera_bias=bool(fixture["enable_camera_bias"]),
    )


def _spacing(config: JACRUSyntheticFixtureConfig) -> tuple[float, float, float]:
    return tuple(
        (config.domain_maximum_xyz[index] - config.domain_minimum_xyz[index])
        / (config.grid_shape[2 - index] - 1)
        for index in range(3)
    )


def _truth_levels(case) -> np.ndarray:
    return case.evaluation.level_sets[0].movedim(0, -1).cpu().numpy()


def _score(
    predicted: list[np.ndarray],
    truth: np.ndarray,
    *,
    spacing_xyz: tuple[float, float, float],
) -> dict[str, Any]:
    return multi_interface_level_set_metrics(
        predicted,
        truth,
        spacing_xyz=spacing_xyz,
    )


def _random_planes(
    *,
    count: int,
    seed: int,
    config: JACRUSyntheticFixtureConfig,
) -> list[np.ndarray]:
    rng = np.random.default_rng(int(seed))
    nz, ny, nx = config.grid_shape
    x = np.linspace(config.domain_minimum_xyz[0], config.domain_maximum_xyz[0], nx)
    y = np.linspace(config.domain_minimum_xyz[1], config.domain_maximum_xyz[1], ny)
    z = np.linspace(config.domain_minimum_xyz[2], config.domain_maximum_xyz[2], nz)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    coordinates = np.stack((xx, yy, zz), axis=-1)
    planes = []
    for _ in range(int(count)):
        normal = rng.normal(size=3)
        normal /= np.linalg.norm(normal)
        offset = rng.uniform(-0.35, 0.35)
        planes.append(np.einsum("...c,c->...", coordinates, normal) - offset)
    return planes


def _read_final_rows(path: Path) -> dict[tuple[int, str, str], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (int(row["base_seed"]), row["family"], row["method"]): row
        for row in rows
    }


def _plot(rows: list[dict[str, Any]], output: Path) -> None:
    single = [row for row in rows if row["family"] == "single_interface"]
    figure, axes = plt.subplots(1, len(single), figsize=(6.4 * len(single), 4.8), constrained_layout=True)
    if len(single) == 1:
        axes = [axes]
    for axis, row in zip(axes, single, strict=True):
        values = np.asarray(row["random_plane_f1_values"], dtype=np.float64)
        axis.hist(values, bins=16, color="#94a3b8", alpha=0.85, label="random planes")
        axis.axvline(row["initial_f1_at_1dx"], color="#d97706", linewidth=2.5, label="data-free init")
        axis.axvline(row["final_f1_at_1dx"], color="#be123c", linewidth=2.5, label="final JACRU-bias")
        axis.set_title(f"single interface | seed {row['base_seed']}")
        axis.set_xlabel("surface F1 @ 1dx")
        axis.set_ylabel("random-plane count")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
        axis.text(
            0.02,
            0.95,
            f"init={row['initial_f1_at_1dx']:.3f}\n"
            f"final={row['final_f1_at_1dx']:.3f}\n"
            f"delta={row['final_minus_initial_f1']:+.3f}\n"
            f"init percentile={row['initial_f1_random_percentile']:.1%}",
            transform=axis.transAxes,
            va="top",
            family="monospace",
        )
    figure.suptitle(
        "JACRU-M0 data-free initialization audit | interface score before observations",
        fontsize=13,
    )
    figure.savefig(output, dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def run_audit(
    *,
    config_path: Path,
    metric_rows_path: Path,
    output_dir: Path,
    random_plane_count: int = 128,
) -> dict[str, Any]:
    config = _load_json(config_path)
    fixture = _fixture_config(config)
    spacing_xyz = _spacing(fixture)
    final_rows = _read_final_rows(metric_rows_path)
    rows: list[dict[str, Any]] = []
    for seed in config["cases"]["base_seeds"]:
        for family in config["cases"]["families"]:
            case = build_jacru_synthetic_case(
                family=family,
                split=str(config["cases"]["split"]),
                base_seed=int(seed),
                config=fixture,
            )
            method = config["methods"]["jacru_bias"]
            support = zero_outer_boundary_support(
                fixture.grid_shape,
                dtype=torch.float64,
            )
            model = JumpAwareConeRayField(
                JumpAwareConfig(
                    grid_shape=fixture.grid_shape,
                    spacing_xyz=spacing_xyz,
                    epsilon=float(method["epsilon_voxels"]) * spacing_xyz[0],
                    outer_steps=int(method["outer_steps"]),
                    field_updates_per_outer=int(method["field_updates_per_outer"]),
                    interface_updates_per_outer=int(method["interface_updates_per_outer"]),
                    bias_updates_per_outer=int(method["bias_updates_per_outer"]),
                    seed=int(seed),
                ),
                batch_size=1,
                dtype=torch.float64,
                support=support,
                camera_group_count=case.inference.geometry.camera_count,
            )
            truth = _truth_levels(case)
            initial_level = model.phase_field[0, 0].detach().cpu().numpy()
            initial_active = bool(model.active_gate()[0])
            initial = _score(
                [initial_level] if initial_active else [],
                truth,
                spacing_xyz=spacing_xyz,
            )
            final = final_rows[(int(seed), family, "jacru_bias")]
            row: dict[str, Any] = {
                "base_seed": int(seed),
                "family": family,
                "initial_gate_probability": float(
                    model.gate_probability()[0, 0].detach()
                ),
                "initial_gate_active": initial_active,
                "initial_f1_at_1dx": float(initial["penalized_surface_f1_at_1dx"]),
                "initial_assd": float(initial["penalized_surface_assd"]),
                "initial_false_positive_count": int(initial["false_positive_count"]),
                "final_f1_at_1dx": float(final["surface_f1_at_1dx"]),
                "final_assd": float(final["surface_assd"]),
            }
            row["final_minus_initial_f1"] = row["final_f1_at_1dx"] - row["initial_f1_at_1dx"]
            row["initial_minus_final_assd"] = row["initial_assd"] - row["final_assd"]
            if family == "single_interface":
                random_f1 = []
                for plane in _random_planes(
                    count=int(random_plane_count),
                    seed=int(seed) + 9001,
                    config=fixture,
                ):
                    metric = _score([plane], truth, spacing_xyz=spacing_xyz)
                    random_f1.append(float(metric["penalized_surface_f1_at_1dx"]))
                row["random_plane_f1_values"] = random_f1
                row["random_plane_f1_mean"] = float(np.mean(random_f1))
                row["random_plane_f1_p95"] = float(np.quantile(random_f1, 0.95))
                row["initial_f1_random_percentile"] = float(
                    np.mean(np.asarray(random_f1) <= row["initial_f1_at_1dx"])
                )
            else:
                row["random_plane_f1_values"] = []
                row["random_plane_f1_mean"] = None
                row["random_plane_f1_p95"] = None
                row["initial_f1_random_percentile"] = None
            rows.append(row)

    single = [row for row in rows if row["family"] == "single_interface"]
    smooth = [row for row in rows if row["family"] == "smooth_no_interface"]
    initial_f1_mean = float(np.mean([row["initial_f1_at_1dx"] for row in single]))
    final_f1_mean = float(np.mean([row["final_f1_at_1dx"] for row in single]))
    delta_mean = final_f1_mean - initial_f1_mean
    false_positive_rate = float(
        np.mean([row["initial_false_positive_count"] > 0 for row in smooth])
    )
    confounded = initial_f1_mean >= 0.8 and delta_mean <= 0.1
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": (
            "INTERFACE_SCORE_CONFOUNDED_BY_DATA_FREE_INITIALIZATION"
            if confounded
            else "INITIALIZATION_CONFOUND_NOT_ESTABLISHED"
        ),
        "source_config_sha256": _sha256(config_path),
        "source_metric_rows_sha256": _sha256(metric_rows_path),
        "random_plane_count_per_single_interface_case": int(random_plane_count),
        "rows": [
            {key: value for key, value in row.items() if key != "random_plane_f1_values"}
            for row in rows
        ],
        "aggregate": {
            "single_interface_initial_f1_at_1dx_mean": initial_f1_mean,
            "single_interface_final_f1_at_1dx_mean": final_f1_mean,
            "single_interface_final_minus_initial_f1_mean": delta_mean,
            "smooth_initial_false_positive_rate": false_positive_rate,
        },
        "decision": {
            "confounded": confounded,
            "interface_gate_credit_authorized": False,
            "required_fix": (
                "score improvement over data-free initialization, randomize orientation, "
                "and start the deployment gate below threshold"
            ),
        },
        "claim_boundary": {
            "uses_observations_for_initialization_score": False,
            "proves_general_benchmark_leakage": False,
            "invalidates_the_current_m0_interface_gain_claim": confounded,
            "is_algorithm_superiority_evidence": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _plot(rows, output_dir / "diagnostic.png")
    (output_dir / "README.md").write_text(
        "# JACRU-M0 initialization audit\n\n"
        "This audit scores the declared interface before any observation is used. "
        "It checks whether the apparent interface gain is inherited from a geometry-"
        "aligned initialization rather than learned from BOST data.\n",
        encoding="ascii",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metric-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-plane-count", type=int, default=128)
    args = parser.parse_args()
    report = run_audit(
        config_path=args.config,
        metric_rows_path=args.metric_rows,
        output_dir=args.output,
        random_plane_count=args.random_plane_count,
    )
    print(json.dumps({"status": report["status"], **report["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
