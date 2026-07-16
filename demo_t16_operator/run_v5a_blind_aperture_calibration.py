#!/usr/bin/env python3
"""First-open blind finite-aperture calibration gate for weak BOST.

Synthetic observations are generated with a deterministic finite-aperture
operator, while deployment candidates only receive a bank of approximate
operators and the visible camera measurements. Cross-view residuals select or
softly marginalize the calibration bank. A truth-based selector is used only
on ``independent_select``; the family-held-out lock is not constructed until
the selection commit is on disk.

This is a tiny linear weak-deflection surrogate. It is not nonlinear ray
tracing, a full depth-of-field model, or a NeRIF/TDBOST reproduction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .finite_aperture_bost import build_finite_aperture_operator_bank
    from .independent_reaction_bost import (
        array_sha256,
        correlated_camera_noise,
        make_reaction_field,
        reaction_support,
    )
    from .measurement_contract import BOSTBatch, DenseVolumeLinearBOST
    from .run_base_correction_independent_gate import (
        front_f1,
        gradient_relative_l2,
        projected_bb,
        projected_fista,
        relative_l2,
    )
except ImportError:
    from finite_aperture_bost import build_finite_aperture_operator_bank
    from independent_reaction_bost import (
        array_sha256,
        correlated_camera_noise,
        make_reaction_field,
        reaction_support,
    )
    from measurement_contract import BOSTBatch, DenseVolumeLinearBOST
    from run_base_correction_independent_gate import (
        front_f1,
        gradient_relative_l2,
        projected_bb,
        projected_fista,
        relative_l2,
    )


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "v5a_blind_aperture_calibration.json"
DEFAULT_OUTPUT = ROOT / "results" / "v5a_blind_aperture_calibration"
BASELINE_NAMES = (
    "pinhole_pbb_4",
    "pinhole_pbb_equal_calls",
    "pinhole_fista_equal_calls",
)


@dataclass(frozen=True)
class MismatchSplit:
    batch: BOSTBatch
    audit_mask: torch.Tensor
    true_radius_indices: np.ndarray
    true_radii: np.ndarray
    family_names: tuple[str, ...]
    calibration_ids: tuple[str, ...]
    true_operator_bank: np.ndarray
    reconstruction_operator_bank: np.ndarray


class SamplewiseDenseVolumeLinearBOST(torch.nn.Module):
    """Exact-adjoint dense operator with one calibration matrix per sample."""

    def __init__(self, operator: torch.Tensor, volume_shape: tuple[int, int, int]):
        super().__init__()
        matrix = torch.as_tensor(operator)
        if matrix.ndim != 5:
            raise ValueError(
                "samplewise operator must have shape [batch,depth,view,detector,voxel]"
            )
        if matrix.shape[-1] != int(np.prod(volume_shape)):
            raise ValueError("samplewise operator voxels and volume shape disagree")
        if matrix.shape[1] != int(volume_shape[0]):
            raise ValueError("samplewise detector depth and volume depth disagree")
        self.register_buffer("operator", matrix)
        self.volume_shape = tuple(int(value) for value in volume_shape)

    def forward(self, volume: torch.Tensor, batch: BOSTBatch) -> torch.Tensor:
        batch.validate()
        values = volume[:, 0] if volume.ndim == 5 else volume
        if values.shape[0] != self.operator.shape[0]:
            raise ValueError("samplewise operator and volume batch sizes disagree")
        projected = torch.einsum("bdvnp,bp->bdvn", self.operator, values.flatten(1))
        return projected * batch.view_mask[:, None, :, None]

    def adjoint(self, residual: torch.Tensor, batch: BOSTBatch) -> torch.Tensor:
        batch.validate()
        if residual.shape != batch.observation.shape:
            raise ValueError("residual and observation shapes disagree")
        weighted = residual * batch.view_mask[:, None, :, None]
        flat = torch.einsum("bdvnp,bdvn->bp", self.operator, weighted)
        return flat.reshape(len(residual), 1, *self.volume_shape)

    def weighted_gradient(
        self, volume: torch.Tensor, batch: BOSTBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = batch.observation - self.forward(volume, batch)
        active = batch.active_observation_mask()
        sigma = batch.expanded_noise_std()
        weighted = torch.where(active, residual / sigma.square(), torch.zeros_like(residual))
        return self.adjoint(weighted, batch), residual

    @torch.no_grad()
    def weighted_lipschitz_exact(self, batch: BOSTBatch) -> torch.Tensor:
        batch.validate()
        sigma = batch.expanded_noise_std().to(self.operator)
        active = batch.active_observation_mask().to(self.operator)
        rows = []
        for sample in range(len(batch.geometry_ids)):
            scale = active[sample] / sigma[sample]
            weighted = self.operator[sample] * scale[:, :, :, None]
            rows.append(torch.linalg.svdvals(weighted.flatten(0, 2))[0].square())
        return torch.stack(rows).clamp_min(1e-8)

    @torch.no_grad()
    def adjoint_relative_error(self, batch: BOSTBatch, seed: int = 0) -> float:
        generator = torch.Generator().manual_seed(int(seed))
        volume = torch.randn(
            (len(batch.geometry_ids), 1, *self.volume_shape),
            generator=generator,
            dtype=self.operator.dtype,
        )
        residual = torch.randn(batch.observation.shape, generator=generator)
        residual = residual * batch.view_mask[:, None, :, None]
        lhs = torch.sum(self.forward(volume, batch) * residual)
        rhs = torch.sum(volume * self.adjoint(residual, batch))
        scale = torch.maximum(torch.abs(lhs), torch.abs(rhs)).clamp_min(1e-12)
        return float(torch.abs(lhs - rhs) / scale)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="calibrate on select without constructing the first-open lock",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def unique_view_masks(
    count: int,
    views: int,
    audit_camera: int,
    budgets: list[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:
    if not 0 <= audit_camera < views:
        raise ValueError("audit camera index is outside the rig")
    eligible = [index for index in range(views) if index != audit_camera]
    patterns: list[np.ndarray] = []
    for budget in sorted(set(int(value) for value in budgets)):
        if not 2 <= budget <= len(eligible):
            raise ValueError("cross-view calibration needs 2..views-1 active cameras")
        for active in itertools.combinations(eligible, budget):
            mask = np.zeros(views, dtype=np.float32)
            mask[list(active)] = 1.0
            patterns.append(mask)
    if count > len(patterns):
        raise ValueError("requested more unique view masks than the rig permits")
    order = rng.permutation(len(patterns))[:count]
    masks = np.stack([patterns[int(index)] for index in order])
    identifiers = ["m" + "".join(str(int(value)) for value in row) for row in masks]
    return masks, identifiers


def balanced_assignments(size: int, classes: int, rng: np.random.Generator) -> np.ndarray:
    values = np.tile(np.arange(classes), int(np.ceil(size / classes)))[:size]
    rng.shuffle(values)
    return values.astype(int)


def reconstruction_signal_rms(clean: np.ndarray, view_mask: np.ndarray) -> float:
    """Estimate the synthetic noise scale from reconstruction cameras only."""

    visible = np.asarray(view_mask, dtype=bool)
    if clean.ndim != 3 or visible.shape != (clean.shape[1],):
        raise ValueError("clean observation and view mask shapes disagree")
    if not np.any(visible):
        raise ValueError("noise scaling requires at least one reconstruction camera")
    return float(np.sqrt(np.mean(clean[:, visible] ** 2)) + 1e-8)


def build_operator_banks(config: dict, split: str) -> tuple[np.ndarray, np.ndarray]:
    rig = config["independent_rigs"][split]
    n, depth = int(config["grid_size"]), int(config["depth"])
    angles = np.asarray(rig["angles_degrees"], dtype=np.float64)
    common = {
        "cone_u": float(rig["cone_u"]),
        "cone_z": float(rig["cone_z"]),
        "bend": float(rig["bend"]),
    }
    reconstruction = build_finite_aperture_operator_bank(
        n,
        depth,
        angles,
        config["candidate_aperture_radii"],
        aperture_samples=int(rig["reconstruction_aperture_samples"]),
        path_samples=int(rig["reconstruction_path_samples"]),
        **common,
    )
    truth = build_finite_aperture_operator_bank(
        n,
        depth,
        angles,
        rig["true_aperture_radii"],
        aperture_samples=int(rig["truth_aperture_samples"]),
        path_samples=int(rig["truth_path_samples"]),
        **common,
    )
    return reconstruction, truth


def build_mismatch_split(
    config: dict,
    split: str,
    reconstruction_bank: np.ndarray,
    truth_bank: np.ndarray,
) -> MismatchSplit:
    offsets = {"independent_select": 0, "independent_lock": 1}
    rng = np.random.default_rng(int(config["data_seed"]) + offsets[split] * 1_000_000)
    rig = config["independent_rigs"][split]
    angles = np.asarray(rig["angles_degrees"], dtype=np.float32)
    count = int(config["counts"][split])
    audit_camera = int(rig["audit_camera_index"])
    masks, mask_ids = unique_view_masks(
        count,
        len(angles),
        audit_camera,
        [int(value) for value in config["active_views"][split]],
        rng,
    )
    radius_indices = balanced_assignments(count, truth_bank.shape[0], rng)
    families = tuple(str(value) for value in config["families"][split])
    family_indices = balanced_assignments(count, len(families), rng)
    levels = np.asarray(config["relative_noise"][split], dtype=float)
    noise_indices = balanced_assignments(count, len(levels), rng)
    factors = np.asarray(config["camera_noise_factors"], dtype=np.float32)
    if len(factors) != len(angles):
        raise ValueError("camera_noise_factors and rig views disagree")

    n, depth = int(config["grid_size"]), int(config["depth"])
    fields, observations, sigmas = [], [], []
    family_names: list[str] = []
    calibration_ids: list[str] = []
    geometry_ids: list[str] = []
    true_radii = np.asarray(rig["true_aperture_radii"], dtype=float)
    for index in range(count):
        family = families[int(family_indices[index])]
        field = make_reaction_field(family, n, depth, rng)
        radius_index = int(radius_indices[index])
        matrix = truth_bank[radius_index]
        clean = np.einsum("dvnp,p->dvn", matrix, field.reshape(-1), optimize=True)
        reconstruction_views = masks[index].astype(bool)
        keep = reconstruction_views.copy()
        keep[audit_camera] = True
        signal_rms = reconstruction_signal_rms(clean, reconstruction_views)
        sigma = float(levels[int(noise_indices[index])]) * signal_rms * factors
        noisy = clean + correlated_camera_noise(
            clean,
            sigma,
            rng,
            correlation_fraction=float(rig["correlation_fraction"]),
            signal_fraction=float(rig["signal_fraction"]),
        )
        noisy[:, ~keep] = 0.0
        calibration_id = f"{split}:aperture={true_radii[radius_index]:.5f}"
        geometry_id = f"{split}:{mask_ids[index]}:field={index:03d}"
        fields.append(field.astype(np.float32))
        observations.append(noisy.astype(np.float32))
        sigmas.append(sigma.astype(np.float32))
        family_names.append(family)
        calibration_ids.append(calibration_id)
        geometry_ids.append(geometry_id)

    audit_mask = np.zeros_like(masks)
    audit_mask[:, audit_camera] = 1.0
    batch = BOSTBatch(
        observation=torch.from_numpy(np.stack(observations)),
        view_mask=torch.from_numpy(masks),
        noise_std=torch.from_numpy(np.stack(sigmas))[:, None, :, None],
        view_angles_degrees=torch.from_numpy(
            np.broadcast_to(angles, (count, len(angles))).copy()
        ),
        support=torch.from_numpy(reaction_support(n, depth))[None, None],
        geometry_ids=tuple(geometry_ids),
        truth=torch.from_numpy(np.stack(fields))[:, None],
    ).validate()
    return MismatchSplit(
        batch=batch,
        audit_mask=torch.from_numpy(audit_mask),
        true_radius_indices=radius_indices,
        true_radii=true_radii[radius_indices],
        family_names=tuple(family_names),
        calibration_ids=tuple(calibration_ids),
        true_operator_bank=truth_bank,
        reconstruction_operator_bank=reconstruction_bank,
    )


def remask(batch: BOSTBatch, mask: torch.Tensor, suffix: str) -> BOSTBatch:
    return BOSTBatch(
        observation=batch.observation,
        view_mask=mask.to(batch.view_mask),
        noise_std=batch.noise_std,
        view_angles_degrees=batch.view_angles_degrees,
        support=batch.support,
        geometry_ids=tuple(f"{value}:{suffix}" for value in batch.geometry_ids),
        truth=batch.truth,
    ).validate()


def subset_batch(batch: BOSTBatch, indices: np.ndarray, suffix: str) -> BOSTBatch:
    index = torch.as_tensor(indices, dtype=torch.long)
    support = batch.support if batch.support.shape[0] == 1 else batch.support[index]
    noise = batch.noise_std if batch.noise_std.shape[0] == 1 else batch.noise_std[index]
    return BOSTBatch(
        observation=batch.observation[index],
        view_mask=batch.view_mask[index],
        noise_std=noise,
        view_angles_degrees=batch.view_angles_degrees[index],
        support=support,
        geometry_ids=tuple(f"{batch.geometry_ids[int(i)]}:{suffix}" for i in indices),
        truth=None if batch.truth is None else batch.truth[index],
    ).validate()


def cross_view_masks(batch: BOSTBatch, fold: int, folds: int) -> tuple[torch.Tensor, torch.Tensor]:
    fit = batch.view_mask.clone()
    held = torch.zeros_like(fit)
    for sample, identifier in enumerate(batch.geometry_ids):
        active = torch.where(batch.view_mask[sample] > 0.5)[0]
        if len(active) < 2:
            raise ValueError("cross-view folds require at least two reconstruction cameras")
        offset = sum(identifier.encode("utf-8")) % len(active)
        stride = max(1, len(active) // int(folds))
        camera = int(active[(offset + int(fold) * stride) % len(active)])
        fit[sample, camera] = 0.0
        held[sample, camera] = 1.0
    return fit, held


def exact_lipschitz(config: dict, operator: object, batch: BOSTBatch) -> torch.Tensor:
    safety = float(config["algorithm"]["lipschitz_safety_factor"])
    return safety * operator.weighted_lipschitz_exact(batch)


def whitened_rms(field: torch.Tensor, operator: object, batch: BOSTBatch) -> np.ndarray:
    residual = batch.observation - operator.forward(field, batch)
    white = batch.whitened(residual)
    count = batch.active_observation_mask().flatten(1).sum(dim=1).clamp_min(1)
    return torch.sqrt(torch.sum(white.square(), dim=(1, 2, 3)) / count).cpu().numpy()


@torch.no_grad()
def reconstruct_operator_bank(config: dict, split: MismatchSplit) -> dict[str, object]:
    algorithm = config["algorithm"]
    probe_stages = int(algorithm["probe_stages"])
    if algorithm.get("probe_solver") != "projected_bb":
        raise ValueError("v5a cross-view probes currently require projected_bb")
    folds = int(algorithm["cross_view_folds"])
    bb_min = float(algorithm["bb_normalized_step_min"])
    bb_max = float(algorithm["bb_normalized_step_max"])
    operators = [
        DenseVolumeLinearBOST(torch.from_numpy(matrix), tuple(split.batch.truth.shape[2:]))
        for matrix in split.reconstruction_operator_bank
    ]
    full_fields, full_scores, cv_scores = [], [], []
    full_lipschitz = []
    for operator in operators:
        lipschitz = exact_lipschitz(config, operator, split.batch)
        full_lipschitz.append(lipschitz)
        field = projected_bb(
            split.batch, operator, lipschitz, probe_stages, bb_min, bb_max
        )
        full_fields.append(field)
        full_scores.append(whitened_rms(field, operator, split.batch))
        numerator = np.zeros(len(split.batch.geometry_ids), dtype=np.float64)
        denominator = np.zeros_like(numerator)
        for fold in range(folds):
            fit_mask, held_mask = cross_view_masks(split.batch, fold, folds)
            fit_batch = remask(split.batch, fit_mask, f"fit{fold}")
            held_batch = remask(split.batch, held_mask, f"held{fold}")
            fit_lipschitz = exact_lipschitz(config, operator, fit_batch)
            fit_field = projected_bb(
                fit_batch, operator, fit_lipschitz, probe_stages, bb_min, bb_max
            )
            residual = held_batch.observation - operator.forward(fit_field, held_batch)
            white = held_batch.whitened(residual)
            numerator += torch.sum(white.square(), dim=(1, 2, 3)).cpu().numpy()
            denominator += held_batch.active_observation_mask().flatten(1).sum(dim=1).cpu().numpy()
        cv_scores.append(np.sqrt(numerator / np.maximum(denominator, 1.0)))
    return {
        "operators": operators,
        "fields": torch.stack(full_fields),
        "full_scores": np.stack(full_scores),
        "cv_scores": np.stack(cv_scores),
        "full_lipschitz": full_lipschitz,
        "diagnostic_forward_calls": len(operators) * (folds + 1) * probe_stages,
        "diagnostic_adjoint_calls": len(operators) * (folds + 1) * probe_stages,
        "exact_spectral_decompositions": len(split.batch.geometry_ids)
        * len(operators)
        * (folds + 1),
    }


def gather_fields(fields: torch.Tensor, indices: np.ndarray) -> torch.Tensor:
    chosen = torch.as_tensor(indices, dtype=torch.long)
    sample = torch.arange(fields.shape[1], dtype=torch.long)
    return fields[chosen, sample]


def normalized_score(score: np.ndarray) -> np.ndarray:
    minimum = np.min(score, axis=0, keepdims=True)
    centered = score - minimum
    scale = np.median(centered, axis=0, keepdims=True)
    fallback = np.mean(centered, axis=0, keepdims=True)
    scale = np.where(scale > 1e-10, scale, fallback)
    return centered / np.maximum(scale, 1e-10)


def soft_weights(score: np.ndarray, temperature: float) -> np.ndarray:
    logits = -normalized_score(score) / float(temperature)
    logits -= np.max(logits, axis=0, keepdims=True)
    values = np.exp(logits)
    return values / np.sum(values, axis=0, keepdims=True)


def margin_confidence(score: np.ndarray) -> np.ndarray:
    ordered = np.sort(score, axis=0)
    scale = np.median(np.abs(score), axis=0)
    return (ordered[1] - ordered[0]) / np.maximum(scale, 1e-10)


def entropy_confidence(weights: np.ndarray) -> np.ndarray:
    entropy = -np.sum(weights * np.log(np.maximum(weights, 1e-12)), axis=0)
    return 1.0 - entropy / np.log(weights.shape[0])


def samplewise_operator_from_weights(
    bank: np.ndarray, weights: np.ndarray, volume_shape: tuple[int, int, int]
) -> SamplewiseDenseVolumeLinearBOST:
    if weights.ndim != 2 or weights.shape[0] != bank.shape[0]:
        raise ValueError("operator weights have an incompatible shape")
    matrix = np.einsum("kb,kdvnp->bdvnp", weights, bank, optimize=True).astype(
        np.float32
    )
    return SamplewiseDenseVolumeLinearBOST(torch.from_numpy(matrix), volume_shape)


def one_hot_weights(indices: np.ndarray, classes: int) -> np.ndarray:
    weights = np.zeros((int(classes), len(indices)), dtype=np.float32)
    weights[indices.astype(int), np.arange(len(indices))] = 1.0
    return weights


@torch.no_grad()
def reconstruct_samplewise(
    config: dict,
    split: MismatchSplit,
    operator: SamplewiseDenseVolumeLinearBOST,
    stages: int,
    *,
    solver: str,
) -> torch.Tensor:
    algorithm = config["algorithm"]
    lipschitz = exact_lipschitz(config, operator, split.batch)
    if solver == "projected_bb":
        return projected_bb(
            split.batch,
            operator,
            lipschitz,
            int(stages),
            float(algorithm["bb_normalized_step_min"]),
            float(algorithm["bb_normalized_step_max"]),
        )
    if solver == "projected_fista":
        return projected_fista(split.batch, operator, lipschitz, int(stages))
    raise ValueError(f"unknown samplewise solver: {solver}")


@torch.no_grad()
def oracle_true_reconstruction(
    config: dict, split: MismatchSplit, stages: int, *, solver: str
) -> torch.Tensor:
    matrices = split.true_operator_bank[split.true_radius_indices]
    operator = SamplewiseDenseVolumeLinearBOST(
        torch.from_numpy(matrices), tuple(split.batch.truth.shape[2:])
    )
    return reconstruct_samplewise(
        config,
        split,
        operator,
        int(stages),
        solver=solver,
    )


@torch.no_grad()
def method_fields(config: dict, split: MismatchSplit, bank: dict[str, object]) -> dict[str, object]:
    algorithm = config["algorithm"]
    probe_fields = bank["fields"]
    cv_scores = np.asarray(bank["cv_scores"])
    full_scores = np.asarray(bank["full_scores"])
    radii = np.asarray(config["candidate_aperture_radii"], dtype=float)
    hard_cv_index = np.argmin(cv_scores, axis=0)
    hard_full_index = np.argmin(full_scores, axis=0)
    diagnostic_calls = int(bank["diagnostic_forward_calls"])
    total_calls = int(algorithm["total_call_budget"])
    final_stages = total_calls - diagnostic_calls
    if final_stages < 2:
        raise ValueError("total_call_budget leaves fewer than two final PBB stages")
    nominal_operator = bank["operators"][0]
    nominal_lipschitz = bank["full_lipschitz"][0]
    volume_shape = tuple(split.batch.truth.shape[2:])
    final_solver = str(algorithm["final_solver"])

    cv_operator = samplewise_operator_from_weights(
        split.reconstruction_operator_bank,
        one_hot_weights(hard_cv_index, len(radii)),
        volume_shape,
    )
    full_operator = samplewise_operator_from_weights(
        split.reconstruction_operator_bank,
        one_hot_weights(hard_full_index, len(radii)),
        volume_shape,
    )
    uniform_weights = np.full(
        (len(radii), len(split.batch.geometry_ids)), 1.0 / len(radii), dtype=np.float32
    )
    uniform_operator = samplewise_operator_from_weights(
        split.reconstruction_operator_bank, uniform_weights, volume_shape
    )
    methods: dict[str, torch.Tensor] = {
        "pinhole_pbb_4": projected_bb(
            split.batch,
            nominal_operator,
            nominal_lipschitz,
            4,
            float(algorithm["bb_normalized_step_min"]),
            float(algorithm["bb_normalized_step_max"]),
        ),
        "pinhole_pbb_equal_calls": projected_bb(
            split.batch,
            nominal_operator,
            nominal_lipschitz,
            total_calls,
            float(algorithm["bb_normalized_step_min"]),
            float(algorithm["bb_normalized_step_max"]),
        ),
        "pinhole_fista_equal_calls": projected_fista(
            split.batch, nominal_operator, nominal_lipschitz, total_calls
        ),
        "cv_hard": reconstruct_samplewise(
            config, split, cv_operator, final_stages, solver=final_solver
        ),
        "full_residual_hard": reconstruct_samplewise(
            config, split, full_operator, final_stages, solver=final_solver
        ),
        "uniform_operator_mean": reconstruct_samplewise(
            config, split, uniform_operator, final_stages, solver=final_solver
        ),
    }
    confidence: dict[str, np.ndarray] = {
        "cv_hard": margin_confidence(cv_scores),
        "full_residual_hard": margin_confidence(full_scores),
        "uniform_operator_mean": np.zeros(probe_fields.shape[1], dtype=float),
    }
    estimated_radius: dict[str, np.ndarray] = {
        "cv_hard": radii[hard_cv_index],
        "full_residual_hard": radii[hard_full_index],
        "uniform_operator_mean": np.full(probe_fields.shape[1], float(np.mean(radii))),
    }
    for temperature in algorithm["soft_temperatures"]:
        value = float(temperature)
        name = f"cv_soft_t{value:g}"
        weights = soft_weights(cv_scores, value)
        soft_operator = samplewise_operator_from_weights(
            split.reconstruction_operator_bank, weights, volume_shape
        )
        methods[name] = reconstruct_samplewise(
            config, split, soft_operator, final_stages, solver=final_solver
        )
        confidence[name] = entropy_confidence(weights)
        estimated_radius[name] = np.sum(weights * radii[:, None], axis=0)
    nearest = np.argmin(np.abs(radii[:, None] - split.true_radii[None, :]), axis=0)
    nearest_operator = samplewise_operator_from_weights(
        split.reconstruction_operator_bank,
        one_hot_weights(nearest, len(radii)),
        volume_shape,
    )
    methods["oracle_nearest_bank_radius"] = reconstruct_samplewise(
        config, split, nearest_operator, total_calls, solver=final_solver
    )
    methods["oracle_true_operator_4"] = oracle_true_reconstruction(
        config, split, 4, solver="projected_bb"
    )
    methods["oracle_true_operator"] = oracle_true_reconstruction(
        config, split, total_calls, solver=final_solver
    )
    return {
        "fields": methods,
        "confidence": confidence,
        "estimated_radius": estimated_radius,
        "hard_cv_index": hard_cv_index,
        "hard_full_index": hard_full_index,
        "diagnostic_calls": diagnostic_calls,
        "final_stages": final_stages,
    }


def centroid_error(prediction: torch.Tensor, truth: torch.Tensor) -> np.ndarray:
    depth, height, width = truth.shape[2:]
    z = torch.linspace(-1.0, 1.0, depth, dtype=truth.dtype)
    y = torch.linspace(-1.0, 1.0, height, dtype=truth.dtype)
    x = torch.linspace(-1.0, 1.0, width, dtype=truth.dtype)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    coordinates = torch.stack([zz, yy, xx]).to(truth)

    def center(values: torch.Tensor) -> torch.Tensor:
        mass = values.sum(dim=(1, 2, 3, 4)).clamp_min(1e-12)
        return torch.stack(
            [torch.sum(values[:, 0] * axis, dim=(1, 2, 3)) / mass for axis in coordinates],
            dim=1,
        )

    return torch.linalg.vector_norm(center(prediction) - center(truth), dim=1).cpu().numpy()


def audit_true_reprojection_rms(field: torch.Tensor, split: MismatchSplit) -> np.ndarray:
    observation = split.batch.observation.cpu().numpy()
    sigma = split.batch.expanded_noise_std().cpu().numpy()
    audit = split.audit_mask.cpu().numpy().astype(bool)
    rows = []
    for sample in range(len(split.batch.geometry_ids)):
        matrix = split.true_operator_bank[int(split.true_radius_indices[sample])]
        predicted = np.einsum(
            "dvnp,p->dvn", matrix, field[sample, 0].cpu().numpy().reshape(-1), optimize=True
        )
        mask = np.broadcast_to(audit[sample][None, :, None], predicted.shape)
        white = (observation[sample] - predicted) / sigma[sample]
        rows.append(float(np.sqrt(np.mean(white[mask] ** 2))))
    return np.asarray(rows)


@torch.no_grad()
def metrics_for_field(
    field: torch.Tensor,
    split: MismatchSplit,
    nominal_operator: DenseVolumeLinearBOST,
) -> dict[str, np.ndarray]:
    truth = split.batch.truth
    support = split.batch.expanded_support()
    truth_mass = truth.sum(dim=(1, 2, 3, 4)).abs().clamp_min(1e-12)
    return {
        "relative_l2": relative_l2(field, truth).cpu().numpy(),
        "gradient_relative_l2": gradient_relative_l2(field, truth).cpu().numpy(),
        "front_f1": front_f1(field, truth, support).cpu().numpy(),
        "mass_relative_error": (
            (field.sum(dim=(1, 2, 3, 4)) - truth.sum(dim=(1, 2, 3, 4))).abs()
            / truth_mass
        ).cpu().numpy(),
        "centroid_error": centroid_error(field, truth),
        "nominal_pinhole_residual_rms": whitened_rms(field, nominal_operator, split.batch),
        "audit_true_operator_residual_rms": audit_true_reprojection_rms(field, split),
    }


def evaluate_all_methods(
    split: MismatchSplit,
    fields: dict[str, torch.Tensor],
    nominal_operator: DenseVolumeLinearBOST,
) -> dict[str, dict[str, np.ndarray]]:
    return {
        name: metrics_for_field(field, split, nominal_operator)
        for name, field in fields.items()
    }


def threshold_candidates(confidence: np.ndarray, count: int) -> np.ndarray:
    values = np.unique(np.quantile(confidence, np.linspace(0.0, 1.0, int(count))))
    lower = float(np.min(confidence)) - max(1e-9, abs(float(np.min(confidence))) * 1e-6)
    upper = float(np.max(confidence)) + max(1e-9, abs(float(np.max(confidence))) * 1e-6)
    return np.concatenate(([lower], values, [upper]))


def gain_for_gate(
    raw_error: np.ndarray,
    baseline_error: np.ndarray,
    confidence: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    accepted = confidence >= float(threshold)
    candidate_error = np.where(accepted, raw_error, baseline_error)
    gain = 100.0 * (baseline_error - candidate_error) / np.maximum(baseline_error, 1e-12)
    return gain, accepted


def calibrate_selection(
    config: dict,
    metrics: dict[str, dict[str, np.ndarray]],
    confidence: dict[str, np.ndarray],
    baseline: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    constraints = config["selection_gate"]
    rows: list[dict[str, object]] = []
    for method, values in confidence.items():
        for threshold in threshold_candidates(values, int(constraints["quantile_count"])):
            gain, accepted = gain_for_gate(
                metrics[method]["relative_l2"],
                metrics[baseline]["relative_l2"],
                values,
                float(threshold),
            )
            row = {
                "method": method,
                "threshold": float(threshold),
                "coverage": float(np.mean(accepted)),
                "mean_gain_percent": float(np.mean(gain)),
                "p10_gain_percent": float(np.quantile(gain, 0.10)),
                "harm_rate_over_1_percent": float(np.mean(gain < -1.0)),
            }
            row["feasible"] = bool(
                row["coverage"] >= float(constraints["minimum_coverage"])
                and row["p10_gain_percent"] >= float(constraints["minimum_p10_gain_percent"])
                and row["harm_rate_over_1_percent"]
                <= float(constraints["maximum_harm_rate_over_1_percent"])
            )
            rows.append(row)
    feasible = [row for row in rows if row["feasible"]]
    if feasible:
        chosen = max(
            feasible,
            key=lambda row: (
                row["mean_gain_percent"],
                row["coverage"],
                -row["harm_rate_over_1_percent"],
            ),
        )
        reason = "best_select_gain_subject_to_predeclared_coverage_and_tail_constraints"
    else:
        first_method = sorted(confidence)[0]
        values = confidence[first_method]
        chosen = {
            "method": first_method,
            "threshold": float(np.max(values)) + max(1e-9, abs(float(np.max(values))) * 1e-6),
            "coverage": 0.0,
            "mean_gain_percent": 0.0,
            "p10_gain_percent": 0.0,
            "harm_rate_over_1_percent": 0.0,
            "feasible": False,
        }
        reason = "no_feasible_operator_selector_abstain_all"
    return {**chosen, "selection_reason": reason}, rows


def apply_selection(
    fields: dict[str, torch.Tensor],
    confidence: dict[str, np.ndarray],
    selection: dict[str, object],
    baseline: str,
) -> tuple[torch.Tensor, np.ndarray]:
    method = str(selection["method"])
    accepted = confidence[method] >= float(selection["threshold"])
    mask = torch.from_numpy(accepted).to(fields[baseline].device)
    candidate = torch.where(
        mask[:, None, None, None, None], fields[method], fields[baseline]
    )
    return candidate, accepted


def split_summary(
    metrics: dict[str, dict[str, np.ndarray]],
    baseline: str,
    candidate: str,
    accepted: np.ndarray,
) -> dict[str, float]:
    base_error = metrics[baseline]["relative_l2"]
    candidate_error = metrics[candidate]["relative_l2"]
    gain = 100.0 * (base_error - candidate_error) / np.maximum(base_error, 1e-12)
    audit_base = metrics[baseline]["audit_true_operator_residual_rms"]
    audit_candidate = metrics[candidate]["audit_true_operator_residual_rms"]
    audit_change = 100.0 * (audit_candidate - audit_base) / np.maximum(audit_base, 1e-12)
    return {
        "candidate_mean_relative_l2": float(np.mean(candidate_error)),
        "baseline_mean_relative_l2": float(np.mean(base_error)),
        "mean_gain_percent": float(np.mean(gain)),
        "p10_gain_percent": float(np.quantile(gain, 0.10)),
        "harm_rate_over_1_percent": float(np.mean(gain < -1.0)),
        "coverage": float(np.mean(accepted)),
        "candidate_mean_gradient_relative_l2": float(
            np.mean(metrics[candidate]["gradient_relative_l2"])
        ),
        "baseline_mean_gradient_relative_l2": float(
            np.mean(metrics[baseline]["gradient_relative_l2"])
        ),
        "candidate_mean_front_f1": float(np.mean(metrics[candidate]["front_f1"])),
        "baseline_mean_front_f1": float(np.mean(metrics[baseline]["front_f1"])),
        "candidate_mean_mass_relative_error": float(
            np.mean(metrics[candidate]["mass_relative_error"])
        ),
        "baseline_mean_mass_relative_error": float(
            np.mean(metrics[baseline]["mass_relative_error"])
        ),
        "candidate_mean_centroid_error": float(np.mean(metrics[candidate]["centroid_error"])),
        "baseline_mean_centroid_error": float(np.mean(metrics[baseline]["centroid_error"])),
        "candidate_mean_audit_true_residual_rms": float(np.mean(audit_candidate)),
        "baseline_mean_audit_true_residual_rms": float(np.mean(audit_base)),
        "audit_reprojection_change_percent": float(np.mean(audit_change)),
    }


def mismatch_penalty(metrics: dict[str, dict[str, np.ndarray]]) -> float:
    pinhole = np.mean(metrics["pinhole_pbb_4"]["relative_l2"])
    oracle = np.mean(metrics["oracle_true_operator_4"]["relative_l2"])
    return float(100.0 * (pinhole - oracle) / max(oracle, 1e-12))


def sample_rows(
    split_name: str,
    split: MismatchSplit,
    metrics: dict[str, dict[str, np.ndarray]],
    methods: dict[str, object],
    selection: dict[str, object],
    baseline: str,
    accepted: np.ndarray,
) -> list[dict[str, object]]:
    chosen_method = str(selection["method"])
    names = [baseline, chosen_method, "selected_candidate", "oracle_true_operator"]
    rows = []
    for index, identifier in enumerate(split.batch.geometry_ids):
        row: dict[str, object] = {
            "split": split_name,
            "sample_index": index,
            "geometry_id": identifier,
            "family": split.family_names[index],
            "calibration_id": split.calibration_ids[index],
            "true_aperture_radius": float(split.true_radii[index]),
            "active_reconstruction_views": int(split.batch.view_mask[index].sum()),
            "accepted": bool(accepted[index]),
            "selected_method": chosen_method,
            "selected_baseline": baseline,
            "confidence": float(methods["confidence"][chosen_method][index]),
            "estimated_aperture_radius": float(
                methods["estimated_radius"][chosen_method][index]
            ),
        }
        for name in dict.fromkeys(names):
            for metric, values in metrics[name].items():
                row[f"{name}_{metric}"] = float(values[index])
        base_error = metrics[baseline]["relative_l2"][index]
        row["selected_gain_percent"] = float(
            100.0
            * (base_error - metrics["selected_candidate"]["relative_l2"][index])
            / max(base_error, 1e-12)
        )
        rows.append(row)
    return rows


def operator_manifest(
    config: dict,
    split_name: str,
    split: MismatchSplit,
) -> dict[str, object]:
    return {
        "split": split_name,
        "candidate_aperture_radii": config["candidate_aperture_radii"],
        "true_aperture_radii": config["independent_rigs"][split_name][
            "true_aperture_radii"
        ],
        "reconstruction_operator_sha256": [
            array_sha256(matrix) for matrix in split.reconstruction_operator_bank
        ],
        "truth_operator_sha256": [
            array_sha256(matrix) for matrix in split.true_operator_bank
        ],
        "truth_reconstruction_hash_overlap": sorted(
            set(array_sha256(matrix) for matrix in split.reconstruction_operator_bank)
            & set(array_sha256(matrix) for matrix in split.true_operator_bank)
        ),
        "audit_camera_index": int(
            config["independent_rigs"][split_name]["audit_camera_index"]
        ),
        "audit_camera_excluded_from_noise_scale": True,
    }


def write_figure(
    path: Path,
    split: MismatchSplit,
    bank: dict[str, object],
    methods: dict[str, object],
    metrics: dict[str, dict[str, np.ndarray]],
    selection: dict[str, object],
    baseline: str,
) -> None:
    method = str(selection["method"])
    gain = 100.0 * (
        metrics[baseline]["relative_l2"] - metrics["selected_candidate"]["relative_l2"]
    ) / np.maximum(metrics[baseline]["relative_l2"], 1e-12)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.0))
    axes[0].scatter(
        split.true_radii,
        methods["estimated_radius"][method],
        c=split.batch.view_mask.sum(dim=1).cpu().numpy(),
        cmap="viridis",
    )
    axes[0].plot([0, 0.17], [0, 0.17], color="black", lw=1)
    axes[0].set(
        title="Blind aperture estimate",
        xlabel="truth-only aperture radius",
        ylabel="estimated radius",
    )
    axes[1].scatter(methods["confidence"][method], gain, alpha=0.78)
    axes[1].axvline(float(selection["threshold"]), color="tab:red", ls="--")
    axes[1].axhline(0.0, color="black", lw=1)
    axes[1].set(title="Selection confidence", xlabel="confidence", ylabel="selected gain (%)")
    labels = ["pinhole", "equal PBB", "equal FISTA", "selected", "oracle A"]
    names = [
        "pinhole_pbb_4",
        "pinhole_pbb_equal_calls",
        "pinhole_fista_equal_calls",
        "selected_candidate",
        "oracle_true_operator",
    ]
    axes[2].bar(
        np.arange(len(names)),
        [np.mean(metrics[name]["relative_l2"]) for name in names],
        color=["#67758b", "#52677e", "#3d536b", "#168a75", "#d0933b"],
    )
    axes[2].set_xticks(np.arange(len(names)), labels, rotation=28, ha="right")
    axes[2].set(title="Field error", ylabel="mean relative L2")
    axes[3].hist(gain, bins=12, color="#168a75", alpha=0.82)
    axes[3].axvline(0.0, color="black", lw=1)
    axes[3].set(title="Selected gain distribution", xlabel="gain (%)", ylabel="count")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config_snapshot.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    started = time.perf_counter()

    select_reconstruction_bank, select_truth_bank = build_operator_banks(
        config, "independent_select"
    )
    select = build_mismatch_split(
        config,
        "independent_select",
        select_reconstruction_bank,
        select_truth_bank,
    )
    select_bank = reconstruct_operator_bank(config, select)
    select_methods = method_fields(config, select, select_bank)
    nominal_select_operator = select_bank["operators"][0]
    select_metrics = evaluate_all_methods(
        select, select_methods["fields"], nominal_select_operator
    )
    baseline_means = {
        name: float(np.mean(select_metrics[name]["relative_l2"]))
        for name in BASELINE_NAMES
    }
    selected_baseline = min(baseline_means, key=baseline_means.get)
    selection, calibration_rows = calibrate_selection(
        config,
        select_metrics,
        select_methods["confidence"],
        selected_baseline,
    )
    selected_field, select_accepted = apply_selection(
        select_methods["fields"],
        select_methods["confidence"],
        selection,
        selected_baseline,
    )
    select_methods["fields"]["selected_candidate"] = selected_field
    select_metrics["selected_candidate"] = metrics_for_field(
        selected_field, select, nominal_select_operator
    )
    select_summary = split_summary(
        select_metrics, selected_baseline, "selected_candidate", select_accepted
    )
    select_manifest = operator_manifest(config, "independent_select", select)

    selection_path = output / "selection_commit.json"
    selection_commit = {
        "created_before_independent_lock": True,
        "selection_role": "select truth used only for baseline, method and confidence threshold selection",
        "selected_baseline": selected_baseline,
        "baseline_mean_relative_l2": baseline_means,
        "selected_operator_method": selection,
        "select_mismatch_penalty_percent": mismatch_penalty(select_metrics),
        "audit_camera_used_by_reconstruction_or_selection": False,
        "audit_camera_excluded_from_noise_scale": True,
        "config_sha256": sha256(config_path),
        "select_operator_manifest": select_manifest,
    }
    selection_path.write_text(
        json.dumps(selection_commit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    calibration_path = output / "selection_calibration.csv"
    write_csv(calibration_path, calibration_rows)

    if args.selection_only:
        report_path = output / "selection_only_report.json"
        report_path.write_text(
            json.dumps(
                {
                    "evidence_label": config["evidence_label"],
                    "status": "DEVELOPMENT_SELECT_ONLY_LOCK_NOT_CONSTRUCTED",
                    "selection": selection_commit,
                    "select_summary": select_summary,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(report_path.read_text(encoding="utf-8"))
        return

    # Nothing below this line exists before the durable selection commit.
    lock_reconstruction_bank, lock_truth_bank = build_operator_banks(
        config, "independent_lock"
    )
    lock = build_mismatch_split(
        config,
        "independent_lock",
        lock_reconstruction_bank,
        lock_truth_bank,
    )
    lock_bank = reconstruct_operator_bank(config, lock)
    lock_methods = method_fields(config, lock, lock_bank)
    nominal_lock_operator = lock_bank["operators"][0]
    lock_metrics = evaluate_all_methods(lock, lock_methods["fields"], nominal_lock_operator)
    lock_field, lock_accepted = apply_selection(
        lock_methods["fields"],
        lock_methods["confidence"],
        selection,
        selected_baseline,
    )
    lock_methods["fields"]["selected_candidate"] = lock_field
    lock_metrics["selected_candidate"] = metrics_for_field(
        lock_field, lock, nominal_lock_operator
    )
    lock_summary = split_summary(
        lock_metrics, selected_baseline, "selected_candidate", lock_accepted
    )
    lock_manifest = operator_manifest(config, "independent_lock", lock)
    lock_mismatch_penalty = mismatch_penalty(lock_metrics)

    gate = config["claim_gate"]
    gate_checks = {
        "mismatch_is_nontrivial": lock_mismatch_penalty
        >= float(gate["minimum_mismatch_penalty_percent"]),
        "mean_gain": lock_summary["mean_gain_percent"]
        >= float(gate["minimum_mean_gain_percent"]),
        "p10_gain": lock_summary["p10_gain_percent"]
        >= float(gate["minimum_p10_gain_percent"]),
        "harm_rate": lock_summary["harm_rate_over_1_percent"]
        <= float(gate["maximum_harm_rate_over_1_percent"]),
        "audit_reprojection": lock_summary["audit_reprojection_change_percent"]
        <= float(gate["maximum_audit_reprojection_increase_percent"]),
    }
    claim_status = (
        "SYNTHETIC_WEAK_BOST_BLIND_APERTURE_GATE_PASSED_REAL_OPTICS_AND_LEARNED_BASELINES_REQUIRED"
        if all(gate_checks.values())
        else "SYNTHETIC_WEAK_BOST_BLIND_APERTURE_GATE_FAILED_OR_INCOMPLETE"
    )

    select_rows = sample_rows(
        "independent_select",
        select,
        select_metrics,
        select_methods,
        selection,
        selected_baseline,
        select_accepted,
    )
    lock_rows = sample_rows(
        "independent_lock",
        lock,
        lock_metrics,
        lock_methods,
        selection,
        selected_baseline,
        lock_accepted,
    )
    samples_path = output / "sample_metrics.csv"
    summary_path = output / "summary.csv"
    manifest_path = output / "operator_manifest.json"
    write_csv(samples_path, select_rows + lock_rows)
    write_csv(
        summary_path,
        [
            {"split": "independent_select", "baseline": selected_baseline, **select_summary},
            {"split": "independent_lock", "baseline": selected_baseline, **lock_summary},
        ],
    )
    manifest_path.write_text(
        json.dumps(
            {
                "select": select_manifest,
                "lock": lock_manifest,
                "select_lock_calibration_id_overlap": sorted(
                    set(select.calibration_ids) & set(lock.calibration_ids)
                ),
                "select_lock_geometry_id_overlap": sorted(
                    set(select.batch.geometry_ids) & set(lock.batch.geometry_ids)
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    figure_path = output / "v5a_blind_aperture_calibration.png"
    write_figure(
        figure_path,
        lock,
        lock_bank,
        lock_methods,
        lock_metrics,
        selection,
        selected_baseline,
    )

    report_path = output / "report.json"
    elapsed = time.perf_counter() - started
    report = {
        "evidence_label": config["evidence_label"],
        "claim_status": claim_status,
        "claim_boundary": "tiny synthetic linear weak-deflection finite-aperture surrogate; no nonlinear ray tracing, real BOS, CFD, NeRIF/TDBOST reproduction, DeepONet/FNO comparison or calibrated probability guarantee",
        "lock_status": "FIRST_OPEN; baseline, method and threshold committed before lock operators, fields and labels were constructed",
        "selected_baseline": selected_baseline,
        "selection": selection,
        "gate_checks": gate_checks,
        "select_mismatch_penalty_percent": mismatch_penalty(select_metrics),
        "lock_mismatch_penalty_percent": lock_mismatch_penalty,
        "select_summary": select_summary,
        "lock_summary": lock_summary,
        "operator_audit": {
            "select": select_manifest,
            "lock": lock_manifest,
            "select_nominal_adjoint_relative_error": nominal_select_operator.adjoint_relative_error(
                select.batch, seed=91
            ),
            "lock_nominal_adjoint_relative_error": nominal_lock_operator.adjoint_relative_error(
                lock.batch, seed=92
            ),
            "truth_reconstruction_operator_equal": False,
            "audit_camera_used_by_reconstruction_or_selection": False,
            "audit_camera_excluded_from_noise_scale": True,
        },
        "call_accounting": {
            "blind_candidate_forward": int(config["algorithm"]["total_call_budget"]),
            "blind_candidate_adjoint": int(config["algorithm"]["total_call_budget"]),
            "diagnostic_forward": int(lock_bank["diagnostic_forward_calls"]),
            "diagnostic_adjoint": int(lock_bank["diagnostic_adjoint_calls"]),
            "final_selected_operator_forward": int(lock_methods["final_stages"]),
            "final_selected_operator_adjoint": int(lock_methods["final_stages"]),
            "pinhole_equal_call_baselines_forward": int(
                config["algorithm"]["total_call_budget"]
            ),
            "pinhole_equal_call_baselines_adjoint": int(
                config["algorithm"]["total_call_budget"]
            ),
            "candidate_bank_size": len(config["candidate_aperture_radii"]),
            "cross_view_folds": int(config["algorithm"]["cross_view_folds"]),
            "probe_stages_per_operator": int(config["algorithm"]["probe_stages"]),
            "total_call_budget": int(config["algorithm"]["total_call_budget"]),
            "lock_exact_spectral_decompositions": int(
                lock_bank["exact_spectral_decompositions"]
                + len(lock.batch.geometry_ids)
            ),
            "truth_only_audit_forward_per_method": 1,
            "oracle_true_operator_is_not_deployable": True,
        },
        "novelty_boundary": {
            "not_new": "blind calibration, joint calibration/reconstruction and operator correction already exist",
            "candidate_space": "BOST-specific operator-level stability or cross-view selective correction under controlled optical mismatch",
        },
        "runtime": {
            "elapsed_seconds": elapsed,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
        },
        "selection_commit_sha256": sha256(selection_path),
        "source_sha256": {
            "runner": sha256(Path(__file__)),
            "finite_aperture_generator": sha256(ROOT / "finite_aperture_bost.py"),
            "measurement_contract": sha256(ROOT / "measurement_contract.py"),
        },
        "config": config,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    files = [
        config_path,
        selection_path,
        calibration_path,
        samples_path,
        summary_path,
        manifest_path,
        figure_path,
        report_path,
    ]
    checksum_path = output / "checksums.sha256"
    checksum_path.write_text(
        "\n".join(f"{sha256(path)}  {path.name}" for path in files) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "claim_status": claim_status,
                "selected_baseline": selected_baseline,
                "selection": selection,
                "lock_mismatch_penalty_percent": lock_mismatch_penalty,
                "lock": lock_summary,
                "gate_checks": gate_checks,
                "elapsed_seconds": elapsed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
