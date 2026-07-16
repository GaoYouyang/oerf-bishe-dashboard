#!/usr/bin/env python3
"""Extract observable graph-whitened PCGLS trajectories on opened seeds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import resource
import time
from typing import Any

import numpy as np
import torch

from demo_t16_operator.detector_covariance_whitening import (
    DetectorCovarianceWhitening,
    WhitenedMeasurementOperator,
)
from demo_t16_operator.detector_graph_covariance import (
    detector_graph_spectral_basis,
)
from demo_t16_operator.psu_b0_classical_baselines import (
    GeneralizedSobolevDirection,
    preconditioned_cgls_reconstruction,
    preconditioned_cgls_trajectory,
)
from demo_t16_operator.psu_b0_detector_graph_features import (
    build_detector_knn_graph,
)
from demo_t16_operator.psu_b0_reaction_phantoms import (
    reaction_morphology_batch,
)
from demo_t16_operator.psu_b0_reconstruction_interface import (
    finite_difference_gradient,
)
from demo_t16_operator.psu_b0_streaming_operator import (
    zero_outer_boundary_support,
)
from site_tools.run_psu_b0_covariance_conditioned_pcgls_diagnosis import (
    _derive_smoke_config,
    validate_partition,
)
from site_tools.run_psu_b0_dg_wpcgls_smoke import (
    _calibrate,
    _flowon_unit_noise,
    _load_json,
    _synchronize,
)
from site_tools.run_psu_b0_real_interface_audit import (
    _make_operator,
    load_real_support_geometry,
)
from site_tools.run_psu_b0_spectral_preconditioner_pilot import (
    _field_metrics,
)


SCHEMA = "psu-b0-covariance-stopping-feature-audit-report-1.0"
STATUS = "POSTOPEN_OBSERVABLE_PCGLS_TRAJECTORY_EXTRACTED"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _per_sample_norm(volume: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(volume.flatten(1), dim=1)


def _trajectory_feature_rows(
    *,
    replicate: int,
    split: str,
    families: list[str],
    prepared_observation: torch.Tensor,
    trajectory: dict[int, Any],
    baseline_metrics: dict[str, torch.Tensor],
    spacing_xyz: tuple[float, float, float],
) -> list[dict[str, Any]]:
    initial_objective = torch.sum(
        prepared_observation.square(),
        dim=(1, 2),
    ).clamp_min(1e-20)
    previous_objective = torch.ones_like(initial_objective)
    previous_volume = torch.zeros_like(
        trajectory[min(trajectory)].volume
    )
    rows = []
    for stage, result in sorted(trajectory.items()):
        residual_objective = torch.sum(
            result.residual_uv.square(),
            dim=(1, 2),
        ) / initial_objective
        update = result.volume - previous_volume
        volume_norm = _per_sample_norm(result.volume).clamp_min(1e-20)
        update_norm = _per_sample_norm(update)
        gradient = finite_difference_gradient(
            result.volume[:, 0],
            spacing_xyz=spacing_xyz,
        )
        gradient_norm = torch.linalg.vector_norm(
            gradient.flatten(1),
            dim=1,
        )
        alpha = result.history[-1]["alpha"]
        previous_beta = (
            torch.zeros_like(alpha)
            if len(result.history) < 2
            else result.history[-2].get(
                "beta",
                torch.zeros_like(alpha),
            )
        )
        for index, family in enumerate(families):
            rows.append(
                {
                    "replicate": int(replicate),
                    "split": split,
                    "sample_index": index,
                    "reaction_family": family,
                    "stage": int(stage),
                    "baseline_field_relative_l2": float(
                        baseline_metrics["field_relative_l2"][index]
                    ),
                    "baseline_gradient_relative_l2": float(
                        baseline_metrics["gradient_relative_l2"][index]
                    ),
                    "baseline_front_top10_f1": float(
                        baseline_metrics["front_top10_f1"][index]
                    ),
                    "relative_whitened_residual_objective": float(
                        residual_objective[index]
                    ),
                    "residual_objective_reduction_from_previous_stage": float(
                        previous_objective[index]
                        - residual_objective[index]
                    ),
                    "pcgls_alpha": float(alpha[index]),
                    "previous_pcgls_beta": float(previous_beta[index]),
                    "relative_volume_update": float(
                        update_norm[index] / volume_norm[index]
                    ),
                    "volume_gradient_to_field_norm": float(
                        gradient_norm[index] / volume_norm[index]
                    ),
                }
            )
        previous_objective = residual_objective
        previous_volume = result.volume
    return rows


def run_feature_audit(
    *,
    root: Path,
    config_path: Path,
    view_root: Path,
    device_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = _load_json(config_path)
    diagnosis_path = root / str(config["source_diagnosis_config"])
    multiseed_path = root / str(config["source_multiseed_config"])
    smoke_path = root / str(config["source_smoke_config"])
    multiseed = _load_json(multiseed_path)
    smoke = _load_json(smoke_path)
    replicate_count = int(multiseed["replicates"]["count"])
    partition = validate_partition(
        config,
        replicate_count=replicate_count,
    )
    device = torch.device(device_name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    torch.set_num_threads(8)
    started = time.perf_counter()

    geometry_config = smoke["geometry"]
    view_count = int(geometry_config["view_count"])
    rays_per_view = int(geometry_config["rays_per_view"])
    grid_size = int(geometry_config["grid_size"])
    geometry, _ = load_real_support_geometry(
        view_root,
        rays_per_view=rays_per_view,
        sample_count=int(
            geometry_config["finite_aperture_sample_count"]
        ),
    )
    operator = _make_operator(
        geometry,
        grid_size=grid_size,
        dtype=torch.float32,
    ).to(device)
    operator.support.copy_(
        zero_outer_boundary_support((grid_size,) * 3).to(device)
    )
    detector_config = _load_json(
        root / str(smoke["source_detector_config"])
    )
    graph_config = detector_config["detector_graph"]
    graph = build_detector_knn_graph(
        geometry["detector_xy"],
        view_count=view_count,
        rays_per_view=rays_per_view,
        neighbor_count=int(graph_config["neighbor_count"]),
        least_squares_ridge=float(
            graph_config["least_squares_ridge"]
        ),
    )
    eigenpairs = [
        detector_graph_spectral_basis(graph, view_index=view)
        for view in range(view_count)
    ]
    eigenvectors_by_view = [vectors for _, vectors in eigenpairs]
    graph_family_index = [
        str(value) for value in smoke["data"]["noise_families"]
    ].index("graph_heat")
    families = [str(value) for value in smoke["data"]["reaction_families"]]
    graph_spec = config["graph_trajectory"]
    baseline_spec = config["fair_baseline"]
    checkpoints = [
        int(value) for value in graph_spec["checkpoint_stages"]
    ]
    rows: list[dict[str, Any]] = []
    ledger = []

    for replicate in range(replicate_count):
        derived = _derive_smoke_config(
            smoke,
            multiseed,
            replicate=replicate,
        )
        field_seeds = [
            int(derived["data"]["field_seed_start"]) + index
            for index in range(len(families))
        ]
        truth = reaction_morphology_batch(
            grid_size=grid_size,
            families=families,
            seeds=field_seeds,
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            clean = operator(truth).detach()
        clean_by_view = clean.reshape(
            len(truth),
            view_count,
            rays_per_view,
            2,
        )
        view_rms = torch.sqrt(
            torch.mean(
                clean_by_view.square(),
                dim=(2, 3),
            ).clamp_min(1e-20)
        )
        floor = (
            float(derived["data"]["minimum_view_signal_fraction"])
            * torch.mean(view_rms, dim=1, keepdim=True)
        )
        levels = tuple(
            float(value)
            for value in derived["data"]["relative_noise_levels"]
        )
        relative_noise = torch.as_tensor(
            [
                levels[index % len(levels)]
                for index in range(len(truth))
            ],
            dtype=torch.float32,
            device=device,
        )
        view_factors = torch.as_tensor(
            derived["data"]["view_noise_factors"],
            dtype=torch.float32,
            device=device,
        )
        scale_by_view = (
            relative_noise[:, None]
            * torch.maximum(view_rms, floor)
            * view_factors[None]
        ).clamp_min(1e-8)
        fits, gate_rows = _calibrate(
            family="graph_heat",
            family_index=graph_family_index,
            config=derived,
            graph=graph,
            eigenpairs=eigenpairs,
        )
        unit_noise = _flowon_unit_noise(
            family="graph_heat",
            family_index=graph_family_index,
            sample_count=len(truth),
            config=derived,
            graph=graph,
            eigenpairs=eigenpairs,
        ).to(device)
        observation = (
            clean
            + unit_noise
            * scale_by_view.repeat_interleave(
                rays_per_view,
                dim=1,
            )[:, :, None]
        )

        component_whitening = DetectorCovarianceWhitening(
            fits["component_iid"],
            eigenvectors_by_view=eigenvectors_by_view,
            scale_by_view=scale_by_view,
            predictive_mean_correction=True,
            dtype=torch.float32,
        ).to(device)
        component_operator = WhitenedMeasurementOperator(
            operator,
            component_whitening,
        ).to(device)
        component_prepared = component_operator.prepare_observation(
            observation
        )
        component_direction = GeneralizedSobolevDirection(
            (grid_size,) * 3,
            strength=float(baseline_spec["sobolev_strength"]),
            epsilon=float(baseline_spec["sobolev_epsilon"]),
        ).to(device)
        component_operator.reset_call_counts()
        with torch.no_grad():
            baseline = preconditioned_cgls_reconstruction(
                component_operator,
                component_prepared,
                sigma_by_view=torch.ones(
                    (len(truth), view_count),
                    dtype=torch.float32,
                    device=device,
                ),
                view_mask=torch.ones(
                    (len(truth), view_count),
                    dtype=torch.float32,
                    device=device,
                ),
                rays_per_view=rays_per_view,
                stages=int(baseline_spec["stages"]),
                preconditioner=component_direction,
            )
        component_call_report = component_operator.call_report()
        expected_baseline_calls = int(baseline_spec["stages"])
        if component_call_report != {
            "forward_calls": expected_baseline_calls,
            "adjoint_calls": expected_baseline_calls,
        }:
            raise ValueError("fair baseline violated its call budget")
        baseline_metrics = _field_metrics(baseline.volume, truth)

        graph_whitening = DetectorCovarianceWhitening(
            fits["dg_covgate"],
            eigenvectors_by_view=eigenvectors_by_view,
            scale_by_view=scale_by_view,
            predictive_mean_correction=True,
            dtype=torch.float32,
        ).to(device)
        graph_operator = WhitenedMeasurementOperator(
            operator,
            graph_whitening,
        ).to(device)
        graph_prepared = graph_operator.prepare_observation(observation)
        graph_direction = GeneralizedSobolevDirection(
            (grid_size,) * 3,
            strength=float(graph_spec["sobolev_strength"]),
            epsilon=float(graph_spec["sobolev_epsilon"]),
        ).to(device)
        graph_operator.reset_call_counts()
        _synchronize(device)
        solve_started = time.perf_counter()
        with torch.no_grad():
            trajectory = preconditioned_cgls_trajectory(
                graph_operator,
                graph_prepared,
                sigma_by_view=torch.ones(
                    (len(truth), view_count),
                    dtype=torch.float32,
                    device=device,
                ),
                view_mask=torch.ones(
                    (len(truth), view_count),
                    dtype=torch.float32,
                    device=device,
                ),
                rays_per_view=rays_per_view,
                checkpoint_stages=checkpoints,
                preconditioner=graph_direction,
            )
        _synchronize(device)
        graph_metrics = {
            stage: _field_metrics(result.volume, truth)
            for stage, result in trajectory.items()
        }
        graph_call_report = graph_operator.call_report()
        maximum_checkpoint = max(checkpoints)
        if graph_call_report != {
            "forward_calls": maximum_checkpoint,
            "adjoint_calls": maximum_checkpoint,
        }:
            raise ValueError("graph trajectory violated its call budget")
        feature_rows = _trajectory_feature_rows(
            replicate=replicate,
            split=partition[replicate],
            families=families,
            prepared_observation=graph_prepared,
            trajectory=trajectory,
            baseline_metrics=baseline_metrics,
            spacing_xyz=tuple(
                float(value) for value in operator.spacing_xyz
            ),
        )
        for row in feature_rows:
            stage = int(row["stage"])
            index = int(row["sample_index"])
            metrics = graph_metrics[stage]
            baseline_field = float(
                baseline_metrics["field_relative_l2"][index]
            )
            baseline_gradient = float(
                baseline_metrics["gradient_relative_l2"][index]
            )
            row.update(
                {
                    "graph_field_relative_l2": float(
                        metrics["field_relative_l2"][index]
                    ),
                    "graph_gradient_relative_l2": float(
                        metrics["gradient_relative_l2"][index]
                    ),
                    "graph_front_top10_f1": float(
                        metrics["front_top10_f1"][index]
                    ),
                    "field_gain_vs_component_s3_k4_percent": float(
                        100.0
                        * (
                            baseline_field
                            - float(metrics["field_relative_l2"][index])
                        )
                        / max(baseline_field, 1e-12)
                    ),
                    "gradient_gain_vs_component_s3_k4_percent": float(
                        100.0
                        * (
                            baseline_gradient
                            - float(
                                metrics["gradient_relative_l2"][index]
                            )
                        )
                        / max(baseline_gradient, 1e-12)
                    ),
                    "front_gain_vs_component_s3_k4": float(
                        float(metrics["front_top10_f1"][index])
                        - float(
                            baseline_metrics["front_top10_f1"][index]
                        )
                    ),
                }
            )
        rows.extend(feature_rows)
        ledger.append(
            {
                "replicate": replicate,
                "split": partition[replicate],
                "graph_activated_view_count": int(
                    sum(bool(row["graph_activated"]) for row in gate_rows)
                ),
                "component_call_report": component_call_report,
                "graph_call_report": graph_call_report,
                "graph_trajectory_seconds": float(
                    time.perf_counter() - solve_started
                ),
            }
        )

    report = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "evidence_scope": config["evidence_scope"],
        "config_sha256": _sha256(config_path),
        "source_diagnosis_config_sha256": _sha256(diagnosis_path),
        "source_diagnosis_report_sha256": _sha256(
            root / str(config["source_diagnosis_report"])
        ),
        "configuration": config,
        "replicate_ledger": ledger,
        "row_count": len(rows),
        "runtime": {
            "device": device_name,
            "torch_version": torch.__version__,
            "wall_seconds": float(time.perf_counter() - started),
            "maximum_rss_bytes": _max_rss_bytes(),
        },
        "claim_boundary": config["claim_boundary"],
    }
    return report, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = (
        args.config
        if args.config.is_absolute()
        else root / args.config
    )
    report, rows = run_feature_audit(
        root=root,
        config_path=config_path,
        view_root=args.view_root.resolve(),
        device_name=str(args.device),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    row_path = args.output_dir / "trajectory_rows.csv"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(row_path, rows)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(report_path),
                "trajectory_rows": str(row_path),
                "row_count": report["row_count"],
                "wall_seconds": report["runtime"]["wall_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
