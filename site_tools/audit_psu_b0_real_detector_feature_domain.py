#!/usr/bin/env python3
"""Audit whether real PSU deflections occupy the synthetic detector-feature domain."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import platform
import resource
import time
from typing import Any

import numpy as np
import torch

from demo_t16_operator.psu_b0_detector_graph_features import (
    DETECTOR_GRAPH_FEATURE_SCHEMA,
    build_detector_knn_graph,
    detector_graph_diagnostics,
    detector_graph_front_features,
)
from demo_t16_operator.psu_b0_streaming_operator import (
    zero_outer_boundary_support,
)
from site_tools.run_psu_b0_real_interface_audit import (
    MASK_STATUS,
    VIEW_BUNDLE_STATUS,
    _make_operator,
    deterministic_quantile_indices,
    load_real_support_geometry,
)
from site_tools.run_psu_b0_residual_risk_development import (
    DevelopmentSplit,
    _build_development_split,
)
from site_tools.run_psu_b0_spectral_preconditioner_pilot import (
    _load_json,
)


PRIVATE_SCHEMA = "psu-b0-real-detector-feature-domain-private-1.0"
PUBLIC_SCHEMA = "psu-b0-real-detector-feature-domain-public-1.0"
STATUS = "REAL_MEASUREMENT_FEATURE_DOMAIN_AUDIT_COMPLETE_NO_RECONSTRUCTION"


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def deterministic_camera_subset_masks(
    *,
    view_count: int,
    minimum_active: int,
    maximum_active: int,
) -> np.ndarray:
    views = int(view_count)
    minimum = int(minimum_active)
    maximum = int(maximum_active)
    if views < 1 or not 1 <= minimum <= maximum <= views:
        raise ValueError("active-view range is invalid")
    rows = []
    for count in range(minimum, maximum + 1):
        for active in itertools.combinations(range(views), count):
            row = np.zeros(views, dtype=np.float32)
            row[list(active)] = 1.0
            rows.append(row)
    return np.stack(rows, axis=0)


def _load_real_deflections(
    view_root: Path,
    *,
    rays_per_view: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    view_dirs = sorted(path for path in view_root.glob("view_*") if path.is_dir())
    if len(view_dirs) != 9:
        raise ValueError("expected exactly nine PSU support views")
    observations = []
    provenance = []
    for view_dir in view_dirs:
        bundle_dir = view_dir / "bundle"
        mask_dir = view_dir / "corrected_masks"
        bundle_manifest = _load_json(
            bundle_dir / "view_bundle_manifest.json"
        )
        mask_manifest = _load_json(
            mask_dir / "corrected_view_masks_manifest.json"
        )
        if bundle_manifest.get("status") != VIEW_BUNDLE_STATUS:
            raise ValueError(f"unverified bundle: {view_dir.name}")
        if mask_manifest.get("status") != MASK_STATUS:
            raise ValueError(f"unverified masks: {view_dir.name}")
        active = np.load(
            mask_dir / "amask_all_zero_based.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        selected = deterministic_quantile_indices(active, int(rays_per_view))
        u = np.asarray(
            np.load(
                bundle_dir / "epsu_all.npy",
                mmap_mode="r",
                allow_pickle=False,
            )[selected, 0],
            dtype=np.float64,
        )
        v = np.asarray(
            np.load(
                bundle_dir / "epsv_all.npy",
                mmap_mode="r",
                allow_pickle=False,
            )[selected, 0],
            dtype=np.float64,
        )
        observation = np.stack((u, v), axis=1)
        if not np.all(np.isfinite(observation)):
            raise ValueError(f"nonfinite real deflections: {view_dir.name}")
        observations.append(observation)
        provenance.append(
            {
                "view_id_zero_based": int(
                    bundle_manifest["view"]["view_id_zero_based"]
                ),
                "selected_index_sha256": _array_sha256(selected),
                "selected_deflection_sha256": _array_sha256(observation),
                "selected_ray_count": int(len(selected)),
            }
        )
    return np.concatenate(observations, axis=0), provenance


def _self_normalized_features(
    observation_uv: torch.Tensor,
    *,
    view_mask: torch.Tensor,
    graph: dict[str, torch.Tensor],
    projection_u_xyz: torch.Tensor,
    projection_v_xyz: torch.Tensor,
    rays_per_view: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    batch = len(observation_uv)
    view_count = int(view_mask.shape[1])
    views = observation_uv.reshape(
        batch,
        view_count,
        int(rays_per_view),
        2,
    )
    rms = torch.sqrt(
        torch.mean(views.square(), dim=(2, 3)).clamp_min(1e-20)
    )
    features, names = detector_graph_front_features(
        observation_uv,
        sigma_by_view=rms,
        view_mask=view_mask,
        graph=graph,
        projection_u_xyz=projection_u_xyz,
        projection_v_xyz=projection_v_xyz,
        rays_per_view=rays_per_view,
    )
    return features.detach().cpu().numpy(), names


def robust_domain_comparison(
    reference: np.ndarray,
    candidates: np.ndarray,
    *,
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    train = np.asarray(reference, dtype=np.float64)
    values = np.asarray(candidates, dtype=np.float64)
    if (
        train.ndim != 2
        or values.ndim != 2
        or train.shape[1] != values.shape[1]
        or train.shape[1] != len(feature_names)
    ):
        raise ValueError("domain comparison arrays do not align")
    center = np.median(train, axis=0)
    mad = 1.4826 * np.median(np.abs(train - center), axis=0)
    standard = np.std(train, axis=0, ddof=0)
    scale = np.maximum(mad, 0.25 * standard)
    informative = scale > 1e-8
    if not np.any(informative):
        raise ValueError("reference features have no informative dimensions")
    reference_z = (train[:, informative] - center[informative]) / scale[informative]
    candidate_z = (
        values[:, informative] - center[informative]
    ) / scale[informative]
    normalized_dimension = float(np.sqrt(np.count_nonzero(informative)))
    center_distance = (
        np.linalg.vector_norm(candidate_z, axis=1)
        / normalized_dimension
    )
    nearest_distance = (
        np.min(
            np.linalg.vector_norm(
                candidate_z[:, None, :] - reference_z[None, :, :],
                axis=2,
            ),
            axis=1,
        )
        / normalized_dimension
    )
    lower = np.quantile(train, 0.025, axis=0)
    upper = np.quantile(train, 0.975, axis=0)
    outside = (values < lower[None]) | (values > upper[None])
    outside_fraction = np.mean(outside, axis=0)
    ranked = np.argsort(outside_fraction)[::-1]
    return {
        "reference_count": int(len(train)),
        "candidate_count": int(len(values)),
        "informative_feature_count": int(np.count_nonzero(informative)),
        "robust_center_distance": {
            "median": float(np.median(center_distance)),
            "p90": float(np.quantile(center_distance, 0.9)),
            "maximum": float(np.max(center_distance)),
        },
        "nearest_reference_distance": {
            "median": float(np.median(nearest_distance)),
            "p90": float(np.quantile(nearest_distance, 0.9)),
            "maximum": float(np.max(nearest_distance)),
        },
        "candidate_rows_outside_any_train_95pct_feature_envelope": float(
            np.mean(np.any(outside[:, informative], axis=1))
        ),
        "mean_informative_feature_outside_fraction": float(
            np.mean(outside_fraction[informative])
        ),
        "highest_outside_fraction_features": [
            {
                "feature": str(feature_names[index]),
                "outside_fraction": float(outside_fraction[index]),
                "train_q025": float(lower[index]),
                "train_q975": float(upper[index]),
                "candidate_median": float(np.median(values[:, index])),
            }
            for index in ranked[:10]
            if informative[index]
        ],
    }


def build_public_summary(private: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PUBLIC_SCHEMA,
        "feature_schema": private["feature_schema"],
        "status": private["status"],
        "evidence_scope": private["evidence_scope"],
        "configuration": copy.deepcopy(private["configuration_public"]),
        "detector_graph_diagnostics": copy.deepcopy(
            private["detector_graph_diagnostics"]
        ),
        "synthetic_internal_comparisons": copy.deepcopy(
            private["synthetic_internal_comparisons"]
        ),
        "real_measurement_comparison": copy.deepcopy(
            private["real_measurement_comparison"]
        ),
        "decision": copy.deepcopy(private["decision"]),
        "runtime": copy.deepcopy(private["runtime"]),
        "claim_boundary": copy.deepcopy(private["claim_boundary"]),
    }


def run_audit(
    *,
    root: Path,
    config_path: Path,
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
    geometry_config = development_config["geometry"]
    device = torch.device(device_name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    started = time.perf_counter()
    grid_size = int(geometry_config["grid_size"])
    rays_per_view = int(geometry_config["rays_per_view"])
    view_count = int(geometry_config["view_count"])
    true_geometry, _ = load_real_support_geometry(
        view_root,
        rays_per_view=rays_per_view,
        sample_count=int(
            geometry_config["true_finite_aperture_sample_count"]
        ),
    )
    nominal_geometry, _ = load_real_support_geometry(
        view_root,
        rays_per_view=rays_per_view,
        sample_count=int(
            geometry_config["nominal_finite_aperture_sample_count"]
        ),
    )
    graph_config = config["detector_graph"]
    graph = build_detector_knn_graph(
        nominal_geometry["detector_xy"],
        view_count=view_count,
        rays_per_view=rays_per_view,
        neighbor_count=int(graph_config["neighbor_count"]),
        least_squares_ridge=float(graph_config["least_squares_ridge"]),
    )
    support = zero_outer_boundary_support((grid_size,) * 3).to(device)
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
    synthetic_features = {}
    feature_names: tuple[str, ...] = ()
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
        values, feature_names = _self_normalized_features(
            wrapped.data.observation_uv.to(device),
            view_mask=wrapped.data.view_mask.to(device),
            graph=graph,
            projection_u_xyz=nominal_operator.projection_u,
            projection_v_xyz=nominal_operator.projection_v,
            rays_per_view=rays_per_view,
        )
        synthetic_features[split_name] = values

    real_observation, real_provenance = _load_real_deflections(
        view_root,
        rays_per_view=rays_per_view,
    )
    active_range = tuple(
        int(value) for value in geometry_config["active_view_range"]
    )
    real_masks = deterministic_camera_subset_masks(
        view_count=view_count,
        minimum_active=active_range[0],
        maximum_active=active_range[1],
    )
    real_batch = torch.as_tensor(
        np.repeat(real_observation[None], len(real_masks), axis=0),
        dtype=torch.float32,
        device=device,
    )
    real_features, real_feature_names = _self_normalized_features(
        real_batch,
        view_mask=torch.as_tensor(real_masks, device=device),
        graph=graph,
        projection_u_xyz=nominal_operator.projection_u,
        projection_v_xyz=nominal_operator.projection_v,
        rays_per_view=rays_per_view,
    )
    if feature_names != real_feature_names:
        raise ValueError("real and synthetic feature schemas disagree")

    train = synthetic_features["risk_train"]
    validation_comparison = robust_domain_comparison(
        train,
        synthetic_features["risk_validation"],
        feature_names=feature_names,
    )
    calibration_comparison = robust_domain_comparison(
        train,
        synthetic_features["risk_calibration"],
        feature_names=feature_names,
    )
    real_comparison = robust_domain_comparison(
        train,
        real_features,
        feature_names=feature_names,
    )
    private = {
        "schema_version": PRIVATE_SCHEMA,
        "feature_schema": DETECTOR_GRAPH_FEATURE_SCHEMA,
        "status": STATUS,
        "evidence_scope": (
            "ONE_REAL_PSU_FLIGHT_BODY_FIELD_WITH_130_DETERMINISTIC_"
            "CAMERA_SUBSETS_COMPARED_TO_SELF_NORMALIZED_SYNTHETIC_"
            "DETECTOR_GRAPH_FEATURES_NO_RECONSTRUCTION"
        ),
        "configuration_private": {
            "root": str(root.resolve()),
            "config_path": str(config_path.resolve()),
            "view_root": str(view_root.resolve()),
            "device": str(device),
        },
        "configuration_public": {
            "view_count": view_count,
            "rays_per_view": rays_per_view,
            "active_view_range": list(active_range),
            "real_physical_field_count": 1,
            "real_camera_subset_count": int(len(real_masks)),
            "normalization": "PER_VIEW_VECTOR_RMS_NOT_NOISE_WHITENING",
            "reference_split": "risk_train",
        },
        "detector_graph_diagnostics": detector_graph_diagnostics(graph),
        "feature_names": list(feature_names),
        "synthetic_features_private": {
            key: value.tolist()
            for key, value in synthetic_features.items()
        },
        "real_features_private": real_features.tolist(),
        "real_camera_masks_private": real_masks.tolist(),
        "real_view_provenance_private": real_provenance,
        "synthetic_internal_comparisons": {
            "risk_validation_vs_risk_train": validation_comparison,
            "risk_calibration_vs_risk_train": calibration_comparison,
        },
        "real_measurement_comparison": real_comparison,
        "decision": {
            "synthetic_envelope_is_realistically_validated": False,
            "route_training_authorized": False,
            "next_required_evidence": (
                "FLOW_OFF_REFERENCE_REPEATS_OR_GRAPH_CORRELATED_NOISE_"
                "MODEL_BEFORE_ANY_SET_ENCODER"
            ),
        },
        "runtime": {
            "wall_seconds": float(time.perf_counter() - started),
            "maximum_rss_bytes": int(_max_rss_bytes()),
        },
        "claim_boundary": {
            "real_measurement_values_used_for_descriptor_audit": True,
            "real_measurement_values_used_for_training": False,
            "camera_subsets_are_independent_physical_fields": False,
            "real_physical_field_count": 1,
            "per_view_rms_is_measured_noise_sigma": False,
            "three_dimensional_truth_used": False,
            "reconstruction_performed": False,
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
            "demo_t16_operator/configs/psu_b0_detector_pose_probe_v1.json"
        ),
    )
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--public-output", type=Path)
    args = parser.parse_args()
    private, public = run_audit(
        root=args.root,
        config_path=args.config,
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
