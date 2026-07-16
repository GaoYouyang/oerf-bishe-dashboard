"""Small, leakage-resistant development data for v5h GC-RIO."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .. import finite_aperture_bost, independent_reaction_bost, rig_shared_profile
from ..finite_aperture_bost import build_finite_aperture_operator_bank
from ..independent_reaction_bost import correlated_camera_noise, make_reaction_field, reaction_support
from ..rig_shared_profile import fit_support_ridge, whitened_normal_mean_diagonal


FORBIDDEN_PREDICTOR_KEYS = frozenset(
    {"source_observation", "target_observation", "truth_field", "clean_observation", "true_sigma"}
)


def _hash(value: Any) -> str:
    if isinstance(value, np.ndarray):
        payload = np.ascontiguousarray(value).view(np.uint8).tobytes()
    else:
        payload = json.dumps(value, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def paired_difference_mad(replicates: np.ndarray) -> np.ndarray:
    """Estimate per-camera sigma from paired zero-signal repeats."""

    values = np.asarray(replicates, dtype=np.float64)
    if values.ndim != 4 or values.shape[0] < 2 or values.shape[0] % 2:
        raise ValueError("replicates must be even [repeat,z,view,x], with >= 2")
    differences = values[0::2] - values[1::2]
    center = np.median(differences, axis=(0, 1, 3), keepdims=True)
    mad = np.median(np.abs(differences - center), axis=(0, 1, 3))
    return np.maximum(mad / (0.6744897501960817 * np.sqrt(2.0)), 1e-7).astype(np.float32)


def _cfg(config: Mapping[str, Any], name: str, default: Any) -> Any:
    return config[name] if name in config else default


def _flatten_operator(operator: np.ndarray) -> np.ndarray:
    # [D,V,X,P] -> [V,D*X,P]
    return np.asarray(operator).transpose(1, 0, 2, 3).reshape(operator.shape[1], -1, operator.shape[-1]).astype(np.float32)


def _flatten_observation(observation: np.ndarray) -> np.ndarray:
    # [D,V,X] -> [V,D*X]
    return np.asarray(observation).transpose(1, 0, 2).reshape(observation.shape[1], -1).astype(np.float32)


def _validate_rigs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rigs = config.get("rigs")
    if not isinstance(rigs, list) or not rigs:
        raise ValueError("config.rigs must be a nonempty list")
    names = {"train", "validation", "design_lock"}
    seen: set[str] = set()
    output = []
    for raw in rigs:
        rig = dict(raw)
        rig_id = str(rig.get("id", ""))
        split = str(rig.get("split", ""))
        if not rig_id or rig_id in seen or split not in names:
            raise ValueError("rig ids must be unique and splits must be train/validation/design_lock")
        seen.add(rig_id)
        source = tuple(int(x) for x in rig.get("source_indices", (0, 1, 2, 3)))
        target = tuple(int(x) for x in rig.get("target_indices", (4, 5)))
        reserved = tuple(int(x) for x in rig.get("reserved_indices", (6,)))
        if tuple(sorted(source + target + reserved)) != tuple(range(7)) or len(source) != 4 or len(target) != 2 or len(reserved) != 1:
            raise ValueError(f"rig {rig_id} must partition views into 4/2/1")
        if len(set(source + target + reserved)) != 7:
            raise ValueError(f"rig {rig_id} view groups overlap")
        angles = np.asarray(rig.get("angles_degrees", np.linspace(0, 180, 7, endpoint=False)), dtype=np.float32)
        if angles.shape != (7,) or not np.all(np.isfinite(angles)):
            raise ValueError(f"rig {rig_id} needs seven finite angles_degrees")
        rig.update({"id": rig_id, "split": split, "source_indices": source, "target_indices": target, "reserved_indices": reserved, "angles_degrees": angles})
        output.append(rig)
    return output


def _families_by_split(config: Mapping[str, Any], rigs: Sequence[dict[str, Any]]) -> dict[str, tuple[str, ...]]:
    names = ("train", "validation", "design_lock")
    split_config = config.get("splits", {})
    default = tuple(str(x) for x in config.get("families", ("expanding_kernel", "jet_shear")))
    result = {name: tuple(str(x) for x in split_config.get(name, {}).get("families", default)) for name in names}
    owners: dict[str, str] = {}
    for split, families in result.items():
        for family in families:
            if family in owners and owners[family] != split:
                raise ValueError(f"family {family} occurs in multiple splits")
            owners[family] = split
    for rig in rigs:
        if not result[rig["split"]]:
            raise ValueError(f"rig {rig['id']} has no family")
    return result


@dataclass
class GCRIODatasetBundle:
    rows: list[dict[str, Any]]
    model_operators: dict[str, np.ndarray]
    truth_operators: dict[str, np.ndarray]
    truth_fields: dict[str, np.ndarray]
    clean_observations: dict[str, np.ndarray]
    scoring_observations: dict[str, np.ndarray]
    noise_provenance: dict[str, dict[str, Any]]
    manifest: dict[str, Any]

    @property
    def predictor_keys(self) -> frozenset[str]:
        return frozenset({"source_operator", "target_operator", "source_residual", "source_sigma", "target_sigma", "support", "base_field", "analytic_correction"})

    def predictor_batch(self, indices: Sequence[int]) -> dict[str, np.ndarray]:
        selected = [self.rows[int(i)] for i in indices]
        output = {
            "source_operator": np.stack([r["source_operator"] for r in selected]).astype(np.float32),
            "target_operator": np.stack([r["target_operator"] for r in selected]).astype(np.float32),
            "source_residual": np.stack([r["source_residual"] for r in selected]).astype(np.float32),
            "source_sigma": np.stack([r["source_sigma"] for r in selected]).astype(np.float32),
            "target_sigma": np.asarray([r["target_sigma"] for r in selected], dtype=np.float32),
            "support": np.stack([r["support"] for r in selected]).astype(np.float32),
            "base_field": np.stack([r["base_field"] for r in selected]).astype(np.float32),
            "analytic_correction": np.stack([r["analytic_correction"] for r in selected]).astype(np.float32),
        }
        if FORBIDDEN_PREDICTOR_KEYS & output.keys():
            raise RuntimeError("predictor firewall violated")
        return output

    def scoring_batch(self, indices: Sequence[int]) -> dict[str, np.ndarray]:
        selected = [self.rows[int(i)] for i in indices]
        return {
            "row_index": np.asarray([r["row_index"] for r in selected], dtype=np.int64),
            "target_residual_label": np.stack([r["target_residual_label"] for r in selected]).astype(np.float32),
            "truth_field": np.stack([self.truth_fields[r["field_uid"]] for r in selected]).astype(np.float32),
            "clean_target_residual": np.stack([r["clean_target_residual"] for r in selected]).astype(np.float32),
            "target_observation": np.stack([r["target_observation"] for r in selected]).astype(np.float32),
        }


def build_dataset(config: Mapping[str, Any]) -> GCRIODatasetBundle:
    """Build deterministic v5h data from an explicit rig list."""

    seed = int(_cfg(config, "seed", 0))
    n, depth = int(_cfg(config, "grid_size", 4)), int(_cfg(config, "depth", 3))
    if n < 3 or depth < 2:
        raise ValueError("grid_size >= 3 and depth >= 2 are required")
    rigs = _validate_rigs(config)
    families_by_split = _families_by_split(config, rigs)
    fields_per_family = int(_cfg(config, "fields_per_family", 1))
    if fields_per_family < 1:
        raise ValueError("fields_per_family must be positive")
    sigma = np.asarray(_cfg(config, "camera_sigma", _cfg(config, "noise_sigma", [0.02] * 7)), dtype=np.float64)
    if sigma.shape != (7,) or np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("camera_sigma must contain seven positive absolute scales")
    repeats = int(_cfg(config, "flow_off_replicates", 8))
    if repeats < 2 or repeats % 2:
        raise ValueError("flow_off_replicates must be even and >= 2")
    corr = float(_cfg(config, "correlation_fraction", 0.35))
    signal_fraction = float(_cfg(config, "signal_fraction", 0.15))
    support_threshold = float(_cfg(config, "support_threshold", 0.5))
    kappa = float(_cfg(config, "correction_kappa", 1e-3))
    truth_fields: dict[str, np.ndarray] = {}
    clean_observations: dict[str, np.ndarray] = {}
    scoring_observations: dict[str, np.ndarray] = {}
    model_operators: dict[str, np.ndarray] = {}
    truth_operators: dict[str, np.ndarray] = {}
    provenance: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    operator_hashes: dict[str, str] = {}
    for rig_index, rig in enumerate(rigs):
        rig_id = rig["id"]
        model_radius = (float(rig.get("model_radius", _cfg(config, "reconstruction_radius", 0.07))),)
        truth_radius = (float(rig.get("truth_radius", _cfg(config, "truth_radius", 0.10))),)
        model_operator = build_finite_aperture_operator_bank(n, depth, rig["angles_degrees"], model_radius, aperture_samples=int(rig.get("model_aperture_samples", 5)), path_samples=int(rig.get("model_path_samples", 8)), cone_u=float(rig.get("cone_u", 0.07)), cone_z=float(rig.get("cone_z", 0.05)), bend=float(rig.get("bend", 0.035)))[0]
        truth_operator = build_finite_aperture_operator_bank(n, depth, rig["angles_degrees"], truth_radius, aperture_samples=int(rig.get("truth_aperture_samples", 5)), path_samples=int(rig.get("truth_path_samples", 8)), cone_u=float(rig.get("truth_cone_u", rig.get("cone_u", 0.07))), cone_z=float(rig.get("truth_cone_z", rig.get("cone_z", 0.05))), bend=float(rig.get("truth_bend", rig.get("bend", 0.035))))[0]
        model_operators[rig_id], truth_operators[rig_id] = model_operator, truth_operator
        operator_hashes[rig_id] = _hash(model_operator)
        rng = np.random.default_rng(seed + 100003 * (rig_index + 1))
        zero = np.zeros((depth, 7, n), dtype=np.float32)
        flow = np.stack([correlated_camera_noise(zero, sigma.astype(np.float32), rng, correlation_fraction=corr, signal_fraction=0.0) for _ in range(repeats)])
        sigma_hat = paired_difference_mad(flow)
        provenance[rig_id] = {"true_sigma": sigma.astype(np.float32), "flow_off_hash": _hash(flow), "correlation_fraction": corr, "signal_fraction": signal_fraction, "noise_estimate_mismatch": True}
        model_flat = _flatten_operator(model_operator)
        truth_flat = _flatten_operator(truth_operator)
        support = (reaction_support(n, depth) >= support_threshold).reshape(-1)
        if not np.any(support):
            raise ValueError("support_threshold selects no independent reaction support voxels")
        for family_index, family in enumerate(families_by_split[rig["split"]]):
            for field_index in range(fields_per_family):
                field_uid = f"{rig_id}:{family}:{field_index}"
                field_rng = np.random.default_rng(seed + rig_index * 1009 + family_index * 97 + field_index)
                field = make_reaction_field(family, n, depth, field_rng).astype(np.float32)
                truth_clean = np.einsum("vmp,p->vm", truth_flat, field.reshape(-1), optimize=True).astype(np.float32)
                truth_clean = truth_clean.reshape(7, depth, n).transpose(1, 0, 2)
                observed = truth_clean + correlated_camera_noise(truth_clean, sigma.astype(np.float32), rng, correlation_fraction=corr, signal_fraction=signal_fraction)
                source_views = rig["source_indices"]
                source_operator_4d = model_operator[:, source_views]
                source_observation_3d = observed[:, source_views]
                ridge = max(float(_cfg(config, "ridge_lambda", 0.0)), whitened_normal_mean_diagonal(source_operator_4d, source_observation_3d, sigma_hat[list(source_views)], range(len(source_views)), support) * 1e-3)
                baseline_fit = fit_support_ridge(source_operator_4d, source_observation_3d, sigma_hat[list(source_views)], range(len(source_views)), support, ridge)
                baseline = baseline_fit.field.astype(np.float32)
                source_flat = _flatten_observation(source_observation_3d)
                source_clean_fit = np.einsum("smp,p->sm", model_flat[list(source_views)], baseline.reshape(-1), optimize=True).astype(np.float32)
                source_residual = source_flat - source_clean_fit
                correction_lambda = max(kappa * whitened_normal_mean_diagonal(source_operator_4d, source_residual.transpose(1, 0).reshape(depth, len(source_views), n), sigma_hat[list(source_views)], range(len(source_views)), support), 1e-8)
                # Fit residuals with the same support and preregistered kappa-scaled penalty.
                correction_fit = fit_support_ridge(source_operator_4d, source_observation_3d - np.einsum("dvxp,p->dvx", source_operator_4d, baseline.reshape(-1)).astype(np.float32), sigma_hat[list(source_views)], range(len(source_views)), support, correction_lambda)
                analytic = correction_fit.field.reshape(-1).astype(np.float32)
                truth_fields[field_uid] = field.reshape(-1).astype(np.float32)
                clean_observations[field_uid] = truth_clean
                scoring_observations[field_uid] = observed
                for target_view in rig["target_indices"]:
                    target_op = model_flat[target_view]
                    target_clean_model = target_op @ baseline.reshape(-1)
                    target_clean_truth = truth_clean[:, target_view].reshape(-1)
                    target_noisy = observed[:, target_view].reshape(-1)
                    rows.append({"row_index": len(rows), "rig_id": rig_id, "field_uid": field_uid, "family": family, "split": rig["split"], "role": "target", "target_view": target_view, "source_operator": model_flat[list(source_views)], "target_operator": target_op, "source_residual": source_residual, "source_sigma": sigma_hat[list(source_views)], "target_sigma": sigma_hat[target_view], "support": support.astype(np.float32), "base_field": baseline.reshape(-1).astype(np.float32), "analytic_correction": analytic, "target_observation": target_noisy, "target_residual_label": target_noisy - target_op @ baseline.reshape(-1), "clean_target_residual": target_clean_truth - target_clean_model})
    if len(set(operator_hashes.values())) < min(2, len(rigs)):
        raise ValueError("rigs must produce at least two different model operator hashes")
    generator_sources = {name: _hash(inspect.getsource(module)) for name, module in (("independent_reaction_bost", independent_reaction_bost), ("finite_aperture_bost", finite_aperture_bost), ("rig_shared_profile", rig_shared_profile))}
    manifest = {"schema": "v5h-gc-rio-dev-2", "seed": seed, "split_family_sets": {k: list(v) for k, v in families_by_split.items()}, "split_rig_sets": {split: [r["id"] for r in rigs if r["split"] == split] for split in ("train", "validation", "design_lock")}, "field_split_map": {uid: row["split"] for uid, row in ((r["field_uid"], r) for r in rows)}, "operator_hashes": operator_hashes, "flow_off_hashes": {k: v["flow_off_hash"] for k, v in provenance.items()}, "predictor_row_hash": _hash([{k: v for k, v in row.items() if k not in {"target_observation", "target_residual_label"}} for row in rows]), "generator_source_hashes": generator_sources}
    return GCRIODatasetBundle(rows, model_operators, truth_operators, truth_fields, clean_observations, scoring_observations, provenance, manifest)
