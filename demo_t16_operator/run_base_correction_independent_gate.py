#!/usr/bin/env python3
"""First-open independent-generator gate for Base-Correction CG-PDNO."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .bost_physics import build_forward_matrix, forward_volume, make_phantom, support_window
    from .cg_pdno import BaseCorrectionCGPDNO, PBBBaseCorrectionCGPDNO
    from .independent_reaction_bost import (
        array_sha256,
        build_curved_cone_operator,
        correlated_camera_noise,
        make_reaction_field,
        reaction_support,
    )
    from .measurement_contract import (
        BOSTBatch,
        DenseVolumeLinearBOST,
        DepthSeparableLinearBOST,
    )
except ImportError:
    from bost_physics import build_forward_matrix, forward_volume, make_phantom, support_window
    from cg_pdno import BaseCorrectionCGPDNO, PBBBaseCorrectionCGPDNO
    from independent_reaction_bost import (
        array_sha256,
        build_curved_cone_operator,
        correlated_camera_noise,
        make_reaction_field,
        reaction_support,
    )
    from measurement_contract import BOSTBatch, DenseVolumeLinearBOST, DepthSeparableLinearBOST


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "base_correction_independent_gate.json"
DEFAULT_OUTPUT = ROOT / "results" / "base_correction_independent_gate"
METHODS = ("candidate", "fixed_pg", "projected_bb", "fista")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
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


def make_masks(
    count: int,
    views: int,
    active_counts: list[int],
    rng: np.random.Generator,
    *,
    prefix: str,
    forbidden: set[str],
    unique_within_split: bool = False,
) -> tuple[np.ndarray, list[str]]:
    rows: list[np.ndarray] = []
    identifiers: list[str] = []
    seen: set[str] = set()
    attempts = 0
    while len(rows) < count:
        active = int(active_counts[len(rows) % len(active_counts)])
        selected = np.sort(rng.choice(views, size=active, replace=False))
        mask = np.zeros(views, dtype=np.float32)
        mask[selected] = 1.0
        identifier = prefix + "_" + "".join("1" if value else "0" for value in mask)
        attempts += 1
        if identifier in forbidden or (unique_within_split and identifier in seen):
            if attempts > 50000:
                raise RuntimeError("could not construct a disjoint geometry pool")
            continue
        rows.append(mask)
        identifiers.append(identifier)
        seen.add(identifier)
    return np.stack(rows), identifiers


def analytic_batch(
    config: dict,
    split: str,
    operator_np: np.ndarray,
    forbidden: set[str],
) -> BOSTBatch:
    offsets = {"train": 0, "validation": 1}
    rng = np.random.default_rng(int(config["data_seed"]) + offsets[split] * 100_000)
    count = int(config["counts"][split])
    angles = np.asarray(config["train_view_angles_degrees"], dtype=np.float32)
    masks, geometry_ids = make_masks(
        count,
        len(angles),
        [int(value) for value in config["active_views"][split]],
        rng,
        prefix="analytic",
        forbidden=forbidden,
    )
    n = int(config["grid_size"])
    depth = int(config["depth"])
    factors = np.asarray(config["camera_noise_factors"], dtype=np.float32)
    families = list(config["families"][split])
    levels = [float(value) for value in config["relative_noise"][split]]
    fields, observations, sigmas = [], [], []
    for index in range(count):
        field = make_phantom(families[index % len(families)], n, depth, rng)
        clean = forward_volume(field, operator_np).astype(np.float32)
        active = masks[index] > 0.5
        signal_rms = float(np.sqrt(np.mean(clean[:, active] ** 2)) + 1e-8)
        sigma = levels[index % len(levels)] * signal_rms * factors
        noisy = clean + rng.normal(scale=sigma[None, :, None], size=clean.shape).astype(
            np.float32
        )
        noisy[:, ~active] = 0.0
        fields.append(field)
        observations.append(noisy)
        sigmas.append(sigma)
    return BOSTBatch(
        observation=torch.from_numpy(np.stack(observations)),
        view_mask=torch.from_numpy(masks),
        noise_std=torch.from_numpy(np.stack(sigmas))[:, None, :, None],
        view_angles_degrees=torch.from_numpy(
            np.broadcast_to(angles, (count, len(angles))).copy()
        ),
        support=torch.from_numpy(support_window(n, depth).astype(np.float32))[None, None],
        geometry_ids=tuple(geometry_ids),
        truth=torch.from_numpy(np.stack(fields))[:, None],
    ).validate()


def independent_batch(
    config: dict,
    split: str,
    operator_np: np.ndarray,
    forbidden: set[str],
) -> BOSTBatch:
    offsets = {"independent_select": 2, "independent_lock": 3}
    rng = np.random.default_rng(int(config["data_seed"]) + offsets[split] * 100_000)
    rig = config["independent_rigs"][split]
    angles = np.asarray(rig["angles_degrees"], dtype=np.float32)
    rig_hash = array_sha256(operator_np)[:10]
    count = int(config["counts"][split])
    masks, geometry_ids = make_masks(
        count,
        len(angles),
        [int(value) for value in config["active_views"][split]],
        rng,
        prefix=f"{split}_{rig_hash}",
        forbidden=forbidden,
        unique_within_split=True,
    )
    n = int(config["grid_size"])
    depth = int(config["depth"])
    factors = np.asarray(config["camera_noise_factors"], dtype=np.float32)
    families = list(config["families"][split])
    levels = [float(value) for value in config["relative_noise"][split]]
    fields, observations, sigmas = [], [], []
    for index in range(count):
        field = make_reaction_field(families[index % len(families)], n, depth, rng)
        clean = np.einsum("dvnp,p->dvn", operator_np, field.reshape(-1), optimize=True)
        active = masks[index] > 0.5
        signal_rms = float(np.sqrt(np.mean(clean[:, active] ** 2)) + 1e-8)
        sigma = levels[index % len(levels)] * signal_rms * factors
        gain = rng.normal(scale=float(rig["gain_drift_std"]), size=len(angles))
        drifted = clean * (1.0 + gain[None, :, None])
        noisy = drifted + correlated_camera_noise(
            drifted,
            sigma,
            rng,
            correlation_fraction=float(rig["correlation_fraction"]),
            signal_fraction=float(rig["signal_fraction"]),
        )
        noisy[:, ~active] = 0.0
        fields.append(field)
        observations.append(noisy.astype(np.float32))
        sigmas.append(sigma)
    return BOSTBatch(
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


def relative_l2(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    numerator = torch.linalg.vector_norm((prediction - truth).flatten(1), dim=1)
    denominator = torch.linalg.vector_norm(truth.flatten(1), dim=1).clamp_min(1e-12)
    return numerator / denominator


def gradient_relative_l2(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    numerator = torch.zeros(len(prediction), dtype=prediction.dtype)
    denominator = torch.zeros_like(numerator)
    for dimension in (2, 3, 4):
        pred_diff = torch.diff(prediction, dim=dimension)
        truth_diff = torch.diff(truth, dim=dimension)
        numerator += torch.sum((pred_diff - truth_diff) ** 2, dim=(1, 2, 3, 4))
        denominator += torch.sum(truth_diff**2, dim=(1, 2, 3, 4))
    return torch.sqrt(numerator / denominator.clamp_min(1e-18))


def gradient_training_loss(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    losses = [
        torch.mean((torch.diff(prediction, dim=dimension) - torch.diff(truth, dim=dimension)) ** 2)
        for dimension in (2, 3, 4)
    ]
    return torch.stack(losses).mean()


def make_model(config: dict) -> BaseCorrectionCGPDNO | PBBBaseCorrectionCGPDNO:
    family = config.get("model_family", "fixed_pg_base")
    if family == "fixed_pg_base":
        return BaseCorrectionCGPDNO(**config["model"])
    if family == "projected_bb_base":
        return PBBBaseCorrectionCGPDNO(**config["model"])
    raise ValueError(f"unknown model_family: {family}")


def front_f1(prediction: torch.Tensor, truth: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    def magnitude(values: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(values)
        for dimension in (2, 3, 4):
            difference = torch.diff(values, dim=dimension)
            padding = [0, 0, 0, 0, 0, 0]
            padding[2 * (4 - dimension) + 1] = 1
            result = result + torch.nn.functional.pad(difference, padding).square()
        return torch.sqrt(result.clamp_min(0.0))

    pred_gradient = magnitude(prediction)
    truth_gradient = magnitude(truth)
    mask = support.expand_as(truth) > 0.1
    rows = []
    for index in range(len(truth)):
        valid_truth = truth_gradient[index][mask[index]]
        valid_pred = pred_gradient[index][mask[index]]
        truth_threshold = torch.quantile(valid_truth, 0.80)
        pred_threshold = torch.quantile(valid_pred, 0.80)
        truth_front = (truth_gradient[index] >= truth_threshold) & mask[index]
        pred_front = (pred_gradient[index] >= pred_threshold) & mask[index]
        true_positive = torch.sum(truth_front & pred_front).to(truth.dtype)
        precision = true_positive / torch.sum(pred_front).clamp_min(1)
        recall = true_positive / torch.sum(truth_front).clamp_min(1)
        rows.append(2.0 * precision * recall / (precision + recall).clamp_min(1e-12))
    return torch.stack(rows)


def projected_bb(
    batch: BOSTBatch,
    operator: object,
    lipschitz: torch.Tensor,
    stages: int,
    normalized_min: float,
    normalized_max: float,
) -> torch.Tensor:
    support = batch.expanded_support().to(batch.observation)
    current = torch.zeros_like(batch.truth)
    previous_field = None
    previous_gradient = None
    for _ in range(int(stages)):
        gradient, _ = operator.weighted_gradient(current, batch)
        if previous_field is None:
            step = 1.0 / lipschitz
        else:
            displacement = current - previous_field
            gradient_change = previous_gradient - gradient
            numerator = torch.sum(displacement * displacement, dim=(1, 2, 3, 4))
            denominator = torch.sum(
                displacement * gradient_change, dim=(1, 2, 3, 4)
            )
            bb = numerator / denominator.clamp_min(1e-18)
            fallback = 1.0 / lipschitz
            step = torch.where(denominator > 1e-12, bb, fallback)
            normalized = torch.clamp(
                step * lipschitz, min=float(normalized_min), max=float(normalized_max)
            )
            step = normalized / lipschitz
        next_field = torch.clamp(
            current + step[:, None, None, None, None] * gradient, min=0.0
        ) * support
        previous_field, previous_gradient = current, gradient
        current = next_field
    return current


def fixed_projected_gradient(
    batch: BOSTBatch,
    operator: object,
    lipschitz: torch.Tensor,
    stages: int,
) -> torch.Tensor:
    support = batch.expanded_support().to(batch.observation)
    current = torch.zeros_like(batch.truth)
    for _ in range(int(stages)):
        gradient, _ = operator.weighted_gradient(current, batch)
        current = torch.clamp(
            current + gradient / lipschitz[:, None, None, None, None], min=0.0
        ) * support
    return current


def projected_fista(
    batch: BOSTBatch,
    operator: object,
    lipschitz: torch.Tensor,
    stages: int,
) -> torch.Tensor:
    support = batch.expanded_support().to(batch.observation)
    current = torch.zeros_like(batch.truth)
    extrapolated = current.clone()
    momentum = 1.0
    for _ in range(int(stages)):
        gradient, _ = operator.weighted_gradient(extrapolated, batch)
        next_field = torch.clamp(
            extrapolated + gradient / lipschitz[:, None, None, None, None], min=0.0
        ) * support
        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
        extrapolated = next_field + ((momentum - 1.0) / next_momentum) * (
            next_field - current
        )
        current = next_field
        momentum = next_momentum
    return current


def train_seed(
    config: dict,
    seed: int,
    train: BOSTBatch,
    validation: BOSTBatch,
    operator: object,
    train_lipschitz: torch.Tensor,
    validation_lipschitz: torch.Tensor,
) -> tuple[BaseCorrectionCGPDNO | PBBBaseCorrectionCGPDNO, list[dict[str, object]], int]:
    torch.manual_seed(int(seed))
    model = make_model(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    best_state = deepcopy(model.state_dict())
    best_validation = float("inf")
    best_epoch = 0
    history: list[dict[str, object]] = []
    warm = torch.zeros_like(train.truth)
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(train, operator, warm, train_lipschitz)
        field_loss = torch.mean((output["prediction"] - train.truth) ** 2)
        gradient_loss = gradient_training_loss(output["prediction"], train.truth)
        loss = field_loss + float(config["training"]["gradient_loss_weight"]) * gradient_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        every = int(config["training"]["validation_every"])
        if epoch == 1 or epoch % every == 0 or epoch == int(config["training"]["epochs"]):
            model.eval()
            with torch.no_grad():
                validation_output = model(
                    validation,
                    operator,
                    torch.zeros_like(validation.truth),
                    validation_lipschitz,
                )
                validation_error = float(
                    relative_l2(validation_output["prediction"], validation.truth).mean()
                )
                train_error = float(relative_l2(output["prediction"], train.truth).mean())
            history.append(
                {
                    "seed": int(seed),
                    "epoch": int(epoch),
                    "train_relative_l2": train_error,
                    "validation_relative_l2": validation_error,
                    "validation_acceptance_rate": float(
                        validation_output["acceptance_gate"].mean()
                    ),
                }
            )
            if validation_error < best_validation:
                best_validation = validation_error
                best_epoch = int(epoch)
                best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, history, best_epoch


@torch.no_grad()
def evaluate(
    model: BaseCorrectionCGPDNO | PBBBaseCorrectionCGPDNO,
    batch: BOSTBatch,
    operator: object,
    lipschitz: torch.Tensor,
    baseline_config: dict,
) -> dict[str, object]:
    warm = torch.zeros_like(batch.truth)
    output = model(batch, operator, warm, lipschitz)
    fixed_pg = fixed_projected_gradient(batch, operator, lipschitz, model.stages)
    pbb = projected_bb(
        batch,
        operator,
        lipschitz,
        model.stages,
        float(baseline_config["bb_normalized_step_min"]),
        float(baseline_config["bb_normalized_step_max"]),
    )
    methods = {
        "candidate": output["prediction"],
        "fixed_pg": fixed_pg,
        "projected_bb": pbb,
        "fista": projected_fista(batch, operator, lipschitz, model.stages),
    }
    expected_fallback = pbb if isinstance(model, PBBBaseCorrectionCGPDNO) else fixed_pg
    if not torch.allclose(
        output["deterministic_fallback"], expected_fallback, rtol=2e-5, atol=2e-6
    ):
        raise RuntimeError("model fallback and declared deterministic baseline disagree")
    support = batch.expanded_support()
    active_count = batch.active_observation_mask().flatten(1).sum(dim=1).clamp_min(1)
    metrics: dict[str, dict[str, np.ndarray]] = {}
    objectives: dict[str, torch.Tensor] = {}
    for name, field in methods.items():
        residual = batch.observation - operator.forward(field, batch)
        white = batch.whitened(residual)
        objectives[name] = 0.5 * torch.sum(white.square(), dim=(1, 2, 3))
        metrics[name] = {
            "relative_l2": relative_l2(field, batch.truth).cpu().numpy(),
            "gradient_relative_l2": gradient_relative_l2(field, batch.truth).cpu().numpy(),
            "front_f1": front_f1(field, batch.truth, support).cpu().numpy(),
            "whitened_residual_rms": torch.sqrt(
                torch.sum(white.square(), dim=(1, 2, 3)) / active_count
            ).cpu().numpy(),
        }
    base_residual = batch.observation - operator.forward(output["shared_base"], batch)
    base_objective = 0.5 * torch.sum(batch.whitened(base_residual).square(), dim=(1, 2, 3))
    violation = objectives["candidate"] > base_objective * (1.0 + 1e-6) + 1e-6
    return {
        "metrics": metrics,
        "acceptance_gate": output["acceptance_gate"].cpu().numpy(),
        "descent_gate": output.get(
            "descent_gate", output["acceptance_gate"]
        ).cpu().numpy(),
        "saturation_gate": output.get(
            "saturation_gate", torch.ones_like(output["acceptance_gate"])
        ).cpu().numpy(),
        "candidate_alpha": output["candidate_alpha"].cpu().numpy(),
        "correction_ratio": output["raw_correction_ratio"].cpu().numpy(),
        "candidate_bound": output["candidate_bound"].cpu().numpy(),
        "fallback_bound": output["fallback_bound"].cpu().numpy(),
        "certificate_violation": violation.cpu().numpy(),
        "forward_calls": int(output["forward_calls"]),
        "adjoint_calls": int(output["adjoint_calls"]),
        "audit_forward_calls_per_method": 1,
    }


def append_evaluation(
    seed: int,
    split: str,
    batch: BOSTBatch,
    result: dict[str, object],
    sample_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
) -> None:
    metrics = result["metrics"]
    for baseline in ("fixed_pg", "projected_bb", "fista"):
        gain = 100.0 * (
            metrics[baseline]["relative_l2"] - metrics["candidate"]["relative_l2"]
        ) / metrics[baseline]["relative_l2"]
        summary_rows.append(
            {
                "seed": int(seed),
                "split": split,
                "baseline": baseline,
                "candidate_mean_relative_l2": float(
                    np.mean(metrics["candidate"]["relative_l2"])
                ),
                "baseline_mean_relative_l2": float(
                    np.mean(metrics[baseline]["relative_l2"])
                ),
                "mean_gain_percent": float(np.mean(gain)),
                "p10_gain_percent": float(np.quantile(gain, 0.10)),
                "harm_rate_over_1_percent": float(np.mean(gain < -1.0)),
                "candidate_mean_gradient_relative_l2": float(
                    np.mean(metrics["candidate"]["gradient_relative_l2"])
                ),
                "baseline_mean_gradient_relative_l2": float(
                    np.mean(metrics[baseline]["gradient_relative_l2"])
                ),
                "candidate_mean_front_f1": float(np.mean(metrics["candidate"]["front_f1"])),
                "baseline_mean_front_f1": float(np.mean(metrics[baseline]["front_f1"])),
                "acceptance_rate": float(np.mean(result["acceptance_gate"])),
                "descent_gate_rate": float(np.mean(result["descent_gate"])),
                "saturation_gate_rate": float(np.mean(result["saturation_gate"])),
                "certificate_violation_rate": float(
                    np.mean(result["certificate_violation"])
                ),
                "forward_calls": int(result["forward_calls"]),
                "adjoint_calls": int(result["adjoint_calls"]),
            }
        )
    for index, geometry_id in enumerate(batch.geometry_ids):
        row: dict[str, object] = {
            "seed": int(seed),
            "split": split,
            "sample_index": int(index),
            "geometry_id": geometry_id,
            "active_views": int(batch.view_mask[index].sum()),
            "acceptance_gate": float(result["acceptance_gate"][index]),
            "descent_gate": float(result["descent_gate"][index]),
            "saturation_gate": float(result["saturation_gate"][index]),
            "candidate_alpha": float(result["candidate_alpha"][index]),
            "correction_ratio": float(result["correction_ratio"][index]),
            "candidate_bound": float(result["candidate_bound"][index]),
            "fallback_bound": float(result["fallback_bound"][index]),
            "certificate_violation": bool(result["certificate_violation"][index]),
        }
        for method in METHODS:
            for metric, values in metrics[method].items():
                row[f"{method}_{metric}"] = float(values[index])
        for baseline in ("fixed_pg", "projected_bb", "fista"):
            row[f"gain_vs_{baseline}_percent"] = float(
                100.0
                * (
                    metrics[baseline]["relative_l2"][index]
                    - metrics["candidate"]["relative_l2"][index]
                )
                / metrics[baseline]["relative_l2"][index]
            )
        sample_rows.append(row)


def aggregate_summary(rows: list[dict[str, object]], split: str, baseline: str) -> dict:
    selected = [row for row in rows if row["split"] == split and row["baseline"] == baseline]
    return {
        "candidate_mean_relative_l2": float(
            np.mean([row["candidate_mean_relative_l2"] for row in selected])
        ),
        "baseline_mean_relative_l2": float(
            np.mean([row["baseline_mean_relative_l2"] for row in selected])
        ),
        "mean_gain_percent": float(np.mean([row["mean_gain_percent"] for row in selected])),
        "minimum_seed_p10_gain_percent": float(
            np.min([row["p10_gain_percent"] for row in selected])
        ),
        "maximum_seed_harm_rate_over_1_percent": float(
            np.max([row["harm_rate_over_1_percent"] for row in selected])
        ),
        "mean_acceptance_rate": float(np.mean([row["acceptance_rate"] for row in selected])),
        "maximum_certificate_violation_rate": float(
            np.max([row["certificate_violation_rate"] for row in selected])
        ),
    }


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    config_snapshot = output / "config_snapshot.json"
    config_snapshot.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    n, depth = int(config["grid_size"]), int(config["depth"])
    train_angles = np.asarray(config["train_view_angles_degrees"], dtype=np.float32)
    train_matrix = build_forward_matrix(n, train_angles)
    train_operator = DepthSeparableLinearBOST(
        torch.from_numpy(train_matrix), (depth, n, n)
    )
    train = analytic_batch(config, "train", train_matrix, set())
    train_ids = set(train.geometry_ids)
    validation = analytic_batch(config, "validation", train_matrix, train_ids)
    train_lipschitz = float(config["lipschitz_safety_factor"]) * train_operator.weighted_lipschitz(
        train, power_iterations=int(config["power_iterations"])
    )
    validation_lipschitz = float(
        config["lipschitz_safety_factor"]
    ) * train_operator.weighted_lipschitz(
        validation, power_iterations=int(config["power_iterations"])
    )

    select_rig = config["independent_rigs"]["independent_select"]
    select_matrix = build_curved_cone_operator(
        n,
        depth,
        np.asarray(select_rig["angles_degrees"], dtype=np.float32),
        path_samples=int(select_rig["path_samples"]),
        cone_u=float(select_rig["cone_u"]),
        cone_z=float(select_rig["cone_z"]),
        bend=float(select_rig["bend"]),
    )
    select_operator = DenseVolumeLinearBOST(torch.from_numpy(select_matrix), (depth, n, n))
    select = independent_batch(config, "independent_select", select_matrix, set())
    select_lipschitz = float(
        config["lipschitz_safety_factor"]
    ) * select_operator.weighted_lipschitz(
        select, power_iterations=int(config["power_iterations"])
    )

    models: dict[int, BaseCorrectionCGPDNO | PBBBaseCorrectionCGPDNO] = {}
    history_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    best_epochs: dict[str, int] = {}
    started = time.perf_counter()
    for seed_value in config["training"]["seeds"]:
        seed = int(seed_value)
        model, history, best_epoch = train_seed(
            config,
            seed,
            train,
            validation,
            train_operator,
            train_lipschitz,
            validation_lipschitz,
        )
        models[seed] = model
        best_epochs[str(seed)] = int(best_epoch)
        history_rows.extend(history)
        for split, batch, operator, lipschitz in [
            ("validation", validation, train_operator, validation_lipschitz),
            ("independent_select", select, select_operator, select_lipschitz),
        ]:
            result = evaluate(
                model, batch, operator, lipschitz, config["numerical_baselines"]
            )
            append_evaluation(seed, split, batch, result, sample_rows, summary_rows)

    baseline_means = {
        baseline: float(
            np.mean(
                [
                    row["baseline_mean_relative_l2"]
                    for row in summary_rows
                    if row["split"] == "independent_select" and row["baseline"] == baseline
                ]
            )
        )
        for baseline in ("fixed_pg", "projected_bb", "fista")
    }
    selected_baseline = min(baseline_means, key=baseline_means.get)
    selection_path = output / "selection_commit.json"
    selection = {
        "created_before_independent_lock": True,
        "selection_role": "independent_select_only",
        "selected_deterministic_baseline": selected_baseline,
        "baseline_mean_relative_l2": baseline_means,
        "best_epoch_by_seed": best_epochs,
        "config_sha256": sha256(config_snapshot),
        "train_operator_sha256": array_sha256(train_matrix),
        "select_operator_sha256": array_sha256(select_matrix),
    }
    selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")

    lock_rig = config["independent_rigs"]["independent_lock"]
    lock_matrix = build_curved_cone_operator(
        n,
        depth,
        np.asarray(lock_rig["angles_degrees"], dtype=np.float32),
        path_samples=int(lock_rig["path_samples"]),
        cone_u=float(lock_rig["cone_u"]),
        cone_z=float(lock_rig["cone_z"]),
        bend=float(lock_rig["bend"]),
    )
    lock_operator = DenseVolumeLinearBOST(torch.from_numpy(lock_matrix), (depth, n, n))
    lock = independent_batch(config, "independent_lock", lock_matrix, set(select.geometry_ids))
    lock_lipschitz = float(config["lipschitz_safety_factor"]) * lock_operator.weighted_lipschitz(
        lock, power_iterations=int(config["power_iterations"])
    )
    for seed, model in models.items():
        lock_result = evaluate(
            model, lock, lock_operator, lock_lipschitz, config["numerical_baselines"]
        )
        append_evaluation(
            seed, "independent_lock", lock, lock_result, sample_rows, summary_rows
        )
    elapsed = time.perf_counter() - started

    aggregates = {
        split: {
            baseline: aggregate_summary(summary_rows, split, baseline)
            for baseline in ("fixed_pg", "projected_bb", "fista")
        }
        for split in ("validation", "independent_select", "independent_lock")
    }
    lock_gate = aggregates["independent_lock"][selected_baseline]
    gate_config = config["claim_gate"]
    gate_checks = {
        "mean_gain": lock_gate["mean_gain_percent"]
        >= float(gate_config["minimum_mean_gain_percent"]),
        "p10_gain": lock_gate["minimum_seed_p10_gain_percent"]
        >= float(gate_config["minimum_p10_gain_percent"]),
        "harm_rate": lock_gate["maximum_seed_harm_rate_over_1_percent"]
        <= float(gate_config["maximum_harm_rate_over_1_percent"]),
        "certificate": lock_gate["maximum_certificate_violation_rate"]
        <= float(gate_config["maximum_certificate_violation_rate"]),
    }
    claim_status = (
        "INDEPENDENT_LINEAR_GATE_PASSED_STRONG_LEARNED_BASELINES_STILL_REQUIRED"
        if all(gate_checks.values())
        else "INDEPENDENT_LINEAR_GATE_FAILED_OR_INCOMPLETE"
    )

    history_path = output / "history.csv"
    sample_path = output / "sample_metrics.csv"
    summary_path = output / "summary.csv"
    write_csv(history_path, history_rows)
    write_csv(sample_path, sample_rows)
    write_csv(summary_path, summary_rows)

    figure_path = output / "base_correction_independent_gate.png"
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    for seed in config["training"]["seeds"]:
        rows = [row for row in history_rows if int(row["seed"]) == int(seed)]
        axes[0].plot(
            [row["epoch"] for row in rows],
            [row["validation_relative_l2"] for row in rows],
            marker="o",
            ms=2.5,
            label=f"seed {seed}",
        )
    axes[0].set(title="Analytic validation lock", xlabel="epoch", ylabel="relative L2")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.25)
    for axis, split in zip(axes[1:], ["independent_select", "independent_lock"]):
        data = [
            [
                row["mean_gain_percent"]
                for row in summary_rows
                if row["split"] == split and row["baseline"] == baseline
            ]
            for baseline in ("fixed_pg", "projected_bb", "fista")
        ]
        axis.boxplot(data, tick_labels=["fixed", "PBB", "FISTA"])
        axis.axhline(0.0, color="black", lw=1)
        axis.set(title=split.replace("independent_", "") + " generator", ylabel="mean gain (%)")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    report_path = output / "report.json"
    report = {
        "evidence_label": config["evidence_label"],
        "claim_status": claim_status,
        "claim_boundary": "independent prescribed linear cone/curved rays and reaction-field families; no nonlinear refraction, CFD truth, OpenBOST, OERF, FNO, FFNO, DeepONet or Learned Primal-Dual comparison yet",
        "lock_status": "FIRST_OPEN; any later architecture change requires a new rig and seed partition",
        "selected_deterministic_baseline": selected_baseline,
        "gate_checks": gate_checks,
        "aggregates": aggregates,
        "operator_audit": {
            "train_kind": "depth-separable straight weak-deflection",
            "select_kind": "fully-3D prescribed curved/cone weak-deflection",
            "lock_kind": "stronger fully-3D prescribed curved/cone weak-deflection",
            "train_sha256": array_sha256(train_matrix),
            "select_sha256": array_sha256(select_matrix),
            "lock_sha256": array_sha256(lock_matrix),
            "train_select_equal": bool(np.array_equal(train_matrix.reshape(-1), select_matrix.reshape(-1)))
            if train_matrix.size == select_matrix.size
            else False,
            "select_lock_equal": bool(np.array_equal(select_matrix, lock_matrix)),
            "select_adjoint_relative_error": select_operator.adjoint_relative_error(select, 71),
            "lock_adjoint_relative_error": lock_operator.adjoint_relative_error(lock, 72),
        },
        "geometry_overlap": {
            "train_validation": sorted(train_ids & set(validation.geometry_ids)),
            "select_lock": sorted(set(select.geometry_ids) & set(lock.geometry_ids)),
        },
        "call_accounting": {
            "candidate_forward": int(config["model"]["stages"]),
            "candidate_adjoint": int(config["model"]["stages"]),
            "fixed_pg_forward": int(config["model"]["stages"]),
            "fixed_pg_adjoint": int(config["model"]["stages"]),
            "projected_bb_forward": int(config["model"]["stages"]),
            "projected_bb_adjoint": int(config["model"]["stages"]),
            "fista_forward": int(config["model"]["stages"]),
            "fista_adjoint": int(config["model"]["stages"]),
            "metric_only_forward_per_method": 1,
        },
        "training": {
            "best_epoch_by_seed": best_epochs,
            "elapsed_seconds": elapsed,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "parameter_count": sum(parameter.numel() for parameter in next(iter(models.values())).parameters()),
        },
        "selection_commit_sha256": sha256(selection_path),
        "config": config,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    public_files = [
        config_snapshot,
        selection_path,
        history_path,
        sample_path,
        summary_path,
        figure_path,
        report_path,
    ]
    checksum_path = output / "checksums.sha256"
    checksum_path.write_text(
        "\n".join(f"{sha256(path)}  {path.name}" for path in public_files) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "claim_status": claim_status,
                "selected_baseline": selected_baseline,
                "lock": lock_gate,
                "gate_checks": gate_checks,
                "elapsed_seconds": elapsed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
