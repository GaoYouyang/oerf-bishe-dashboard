#!/usr/bin/env python3
"""Run the preregistered positive-spectral PSU B0 development pilot."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
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

from demo_t16_operator.psu_b0_reaction_phantoms import (
    PHANTOM_SCHEMA,
    reaction_morphology_batch,
)
from demo_t16_operator.psu_b0_reconstruction_interface import (
    finite_difference_gradient,
)
from demo_t16_operator.psu_b0_spectral_preconditioner import (
    FixedSobolevDirection,
    IdentityDirection,
    PositiveSpectralDirection,
    SPECTRAL_PRECONDITIONER_SCHEMA,
    exact_line_search_reconstruction,
    normalized_field_loss,
    weighted_cgls_reconstruction,
)
from demo_t16_operator.psu_b0_streaming_operator import (
    zero_outer_boundary_support,
)
from site_tools.run_psu_b0_real_interface_audit import (
    _make_operator,
    load_real_support_geometry,
)


PRIVATE_SCHEMA = "psu-b0-spectral-preconditioner-pilot-private-report-1.0"
PUBLIC_SCHEMA = "psu-b0-spectral-preconditioner-pilot-public-summary-1.0"


@dataclass(frozen=True)
class SyntheticSplit:
    name: str
    sample_ids: tuple[str, ...]
    families: tuple[str, ...]
    truth: torch.Tensor
    observation_uv: torch.Tensor
    sigma_by_view: torch.Tensor
    view_mask: torch.Tensor
    relative_noise: torch.Tensor
    truth_operator: str
    operator_mismatch_relative_l2: torch.Tensor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _max_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if platform.system() == "Darwin" else raw * 1024


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config must be a JSON object")
    return value


def _unique_masks(
    *,
    count: int,
    view_count: int,
    minimum_active: int,
    maximum_active: int,
    seed: int,
    forbidden: set[str],
) -> tuple[torch.Tensor, set[str]]:
    if not 1 <= minimum_active <= maximum_active <= view_count:
        raise ValueError("active view range is invalid")
    candidates: list[tuple[int, ...]] = []
    for active_count in range(minimum_active, maximum_active + 1):
        candidates.extend(itertools.combinations(range(view_count), active_count))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(candidates)
    rows = []
    keys = set(forbidden)
    for active in candidates:
        key = "".join("1" if index in active else "0" for index in range(view_count))
        if key in keys:
            continue
        row = np.zeros(view_count, dtype=np.float32)
        row[list(active)] = 1.0
        rows.append(row)
        keys.add(key)
        if len(rows) == int(count):
            break
    if len(rows) != int(count):
        raise ValueError("not enough unique masks for the requested split")
    return torch.from_numpy(np.stack(rows)), keys


def _batched_forward(
    operator: Any,
    values: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    rows = []
    with torch.no_grad():
        for start in range(0, len(values), int(batch_size)):
            batch = values[start : start + int(batch_size)].to(device)
            rows.append(operator(batch).detach().cpu())
    return torch.cat(rows)


def _build_split(
    *,
    name: str,
    spec: dict[str, Any],
    config: dict[str, Any],
    true_operator: Any,
    nominal_operator: Any,
    device: torch.device,
    forbidden_masks: set[str],
) -> tuple[SyntheticSplit, set[str]]:
    count = int(spec["count"])
    grid_size = int(config["geometry"]["grid_size"])
    view_count = int(config["geometry"]["view_count"])
    rays_per_view = int(config["geometry"]["rays_per_view"])
    families = tuple(
        str(spec["families"][index % len(spec["families"])])
        for index in range(count)
    )
    field_seeds = tuple(int(spec["field_seed_start"]) + index for index in range(count))
    truth = reaction_morphology_batch(
        grid_size=grid_size,
        families=families,
        seeds=field_seeds,
        dtype=torch.float32,
        device="cpu",
    )
    active_range = tuple(int(value) for value in spec["active_view_range"])
    mask, used_masks = _unique_masks(
        count=count,
        view_count=view_count,
        minimum_active=active_range[0],
        maximum_active=active_range[1],
        seed=int(spec["mask_seed"]),
        forbidden=forbidden_masks,
    )
    generation_batch = int(config["training"]["batch_size"])
    nominal_signal = _batched_forward(
        nominal_operator,
        truth,
        batch_size=generation_batch,
        device=device,
    )
    if str(spec["truth_operator"]) == "qmc32":
        true_signal = _batched_forward(
            true_operator,
            truth,
            batch_size=generation_batch,
            device=device,
        )
    elif str(spec["truth_operator"]) == "qmc8":
        true_signal = nominal_signal.clone()
    else:
        raise ValueError("truth_operator must be qmc32 or qmc8")
    difference = torch.linalg.vector_norm(
        (true_signal - nominal_signal).flatten(1),
        dim=1,
    ) / torch.linalg.vector_norm(true_signal.flatten(1), dim=1).clamp_min(1e-12)
    view_signal = true_signal.reshape(count, view_count, rays_per_view, 2)
    view_rms = torch.sqrt(
        torch.mean(view_signal.square(), dim=(2, 3)).clamp_min(1e-20)
    )
    global_floor = 0.10 * torch.mean(view_rms, dim=1, keepdim=True)
    relative_levels = tuple(float(value) for value in spec["relative_noise_levels"])
    relative_noise = torch.as_tensor(
        [relative_levels[index % len(relative_levels)] for index in range(count)],
        dtype=torch.float32,
    )
    view_factors = torch.as_tensor(
        config["data"]["view_noise_factors"],
        dtype=torch.float32,
    )
    sigma = (
        relative_noise[:, None]
        * torch.maximum(view_rms, global_floor)
        * view_factors[None]
    ).clamp_min(1e-8)
    generator = torch.Generator().manual_seed(int(spec["noise_seed"]))
    noise = torch.randn(
        true_signal.shape,
        generator=generator,
        dtype=true_signal.dtype,
    )
    noise = noise * sigma.repeat_interleave(rays_per_view, dim=1)[:, :, None]
    observation = true_signal + noise
    sample_ids = tuple(f"{name}-{index:03d}" for index in range(count))
    return (
        SyntheticSplit(
            name=name,
            sample_ids=sample_ids,
            families=families,
            truth=truth,
            observation_uv=observation,
            sigma_by_view=sigma,
            view_mask=mask,
            relative_noise=relative_noise,
            truth_operator=str(spec["truth_operator"]),
            operator_mismatch_relative_l2=difference,
        ),
        used_masks,
    )


def _expanded_measurement_values(
    values: torch.Tensor,
    *,
    rays_per_view: int,
) -> torch.Tensor:
    return values.repeat_interleave(int(rays_per_view), dim=1)[:, :, None]


def _field_metrics(
    prediction: torch.Tensor,
    truth: torch.Tensor,
) -> dict[str, torch.Tensor]:
    difference = prediction - truth
    field = torch.linalg.vector_norm(difference.flatten(1), dim=1)
    field = field / torch.linalg.vector_norm(truth.flatten(1), dim=1).clamp_min(1e-12)
    grid_shape = truth.shape[-3:]
    spacing = tuple(2.0 / (size - 1) for size in grid_shape[::-1])
    prediction_gradient = finite_difference_gradient(
        prediction[:, 0],
        spacing_xyz=spacing,
    )
    truth_gradient = finite_difference_gradient(
        truth[:, 0],
        spacing_xyz=spacing,
    )
    gradient = torch.linalg.vector_norm(
        (prediction_gradient - truth_gradient).flatten(1),
        dim=1,
    )
    gradient = gradient / torch.linalg.vector_norm(
        truth_gradient.flatten(1),
        dim=1,
    ).clamp_min(1e-12)
    predicted_magnitude = torch.sqrt(
        torch.sum(prediction_gradient.square(), dim=1)
    ).flatten(1)
    truth_magnitude = torch.sqrt(torch.sum(truth_gradient.square(), dim=1)).flatten(1)
    predicted_threshold = torch.quantile(predicted_magnitude, 0.90, dim=1)
    truth_threshold = torch.quantile(truth_magnitude, 0.90, dim=1)
    predicted_front = predicted_magnitude >= predicted_threshold[:, None]
    truth_front = truth_magnitude >= truth_threshold[:, None]
    intersection = torch.sum(predicted_front & truth_front, dim=1).to(torch.float32)
    precision = intersection / torch.sum(predicted_front, dim=1).clamp_min(1)
    recall = intersection / torch.sum(truth_front, dim=1).clamp_min(1)
    front_f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "field_relative_l2": field,
        "gradient_relative_l2": gradient,
        "front_top10_f1": front_f1,
    }


def _evaluate(
    *,
    method: str,
    split: SyntheticSplit,
    operator: Any,
    config: dict[str, Any],
    device: torch.device,
    direction: Any | None = None,
    batch_size: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    size = int(batch_size or config["training"]["batch_size"])
    rays_per_view = int(config["geometry"]["rays_per_view"])
    stages = int(config["solver"]["stages"])
    rows: list[dict[str, Any]] = []
    monotone = True
    gain_minimum = float("inf")
    gain_maximum = 0.0
    started = time.perf_counter()
    operator.reset_call_counts()
    for start in range(0, len(split.truth), size):
        stop = min(start + size, len(split.truth))
        truth = split.truth[start:stop].to(device)
        observation = split.observation_uv[start:stop].to(device)
        sigma = split.sigma_by_view[start:stop].to(device)
        mask = split.view_mask[start:stop].to(device)
        with torch.no_grad():
            if method == "cgls":
                result = weighted_cgls_reconstruction(
                    operator,
                    observation,
                    sigma_by_view=sigma,
                    view_mask=mask,
                    rays_per_view=rays_per_view,
                    stages=stages,
                )
            else:
                if direction is None:
                    raise ValueError("line-search methods require a direction")
                result = exact_line_search_reconstruction(
                    operator,
                    observation,
                    sigma_by_view=sigma,
                    view_mask=mask,
                    rays_per_view=rays_per_view,
                    stages=stages,
                    direction=direction,
                )
        metrics = _field_metrics(result.volume, truth)
        combined = normalized_field_loss(
            result.volume,
            truth,
            gradient_weight=float(config["training"]["gradient_weight"]),
        )
        active = _expanded_measurement_values(mask, rays_per_view=rays_per_view)
        expanded_sigma = _expanded_measurement_values(
            sigma,
            rays_per_view=rays_per_view,
        )
        measurement = torch.linalg.vector_norm(
            (result.residual_uv / expanded_sigma).flatten(1),
            dim=1,
        ) / torch.linalg.vector_norm(
            (active * observation / expanded_sigma).flatten(1),
            dim=1,
        ).clamp_min(1e-12)
        for history in result.history:
            if "relative_objective_before" in history:
                monotone &= bool(
                    torch.all(
                        history["relative_objective_after"]
                        <= history["relative_objective_before"] + 2e-5
                    )
                )
            if "gain_minimum" in history:
                gain_minimum = min(
                    gain_minimum,
                    float(torch.min(history["gain_minimum"])),
                )
                gain_maximum = max(
                    gain_maximum,
                    float(torch.max(history["gain_maximum"])),
                )
        for offset, index in enumerate(range(start, stop)):
            rows.append(
                {
                    "sample_id": split.sample_ids[index],
                    "split": split.name,
                    "family": split.families[index],
                    "truth_operator": split.truth_operator,
                    "relative_noise": float(split.relative_noise[index]),
                    "active_view_count": int(torch.sum(split.view_mask[index] > 0.5)),
                    "operator_mismatch_relative_l2": float(
                        split.operator_mismatch_relative_l2[index]
                    ),
                    "method": method,
                    "field_relative_l2": float(
                        metrics["field_relative_l2"][offset]
                    ),
                    "gradient_relative_l2": float(
                        metrics["gradient_relative_l2"][offset]
                    ),
                    "front_top10_f1": float(metrics["front_top10_f1"][offset]),
                    "combined_loss": float(combined[offset]),
                    "measurement_relative_l2": float(measurement[offset]),
                }
            )
    _synchronize(device)
    calls = operator.call_report()
    return rows, {
        "method": method,
        "split": split.name,
        "wall_seconds": float(time.perf_counter() - started),
        "batch_invocations": {
            "forward": int(calls["forward_calls"]),
            "adjoint": int(calls["adjoint_calls"]),
        },
        "logical_calls_per_sample": (
            {"forward": stages, "adjoint": stages + 1}
            if method == "cgls"
            else {"forward": stages, "adjoint": stages}
        ),
        "data_objective_monotone": bool(monotone),
        "gain_minimum": None if gain_minimum == float("inf") else gain_minimum,
        "gain_maximum": None if gain_maximum == 0.0 else gain_maximum,
    }


def _validation_loss(
    *,
    split: SyntheticSplit,
    operator: Any,
    config: dict[str, Any],
    device: torch.device,
    direction: Any,
) -> float:
    rows, _ = _evaluate(
        method="validation_probe",
        split=split,
        operator=operator,
        config=config,
        device=device,
        direction=direction,
    )
    return float(np.mean([row["combined_loss"] for row in rows]))


def _train_seed(
    *,
    seed: int,
    train: SyntheticSplit,
    validation: SyntheticSplit,
    operator: Any,
    config: dict[str, Any],
    base_strength: float,
    device: torch.device,
) -> tuple[PositiveSpectralDirection, dict[str, Any]]:
    torch.manual_seed(int(seed))
    model = PositiveSpectralDirection(
        (int(config["geometry"]["grid_size"]),) * 3,
        view_count=int(config["geometry"]["view_count"]),
        hidden=int(config["model"]["hidden"]),
        embedding_width=int(config["model"]["view_embedding_width"]),
        maximum_log_gain=float(config["model"]["maximum_log_correction"]),
        base_sobolev_strength=float(base_strength),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(config["training"]["epochs"])
    batch_size = int(config["training"]["batch_size"])
    validation_every = int(config["training"]["validation_every"])
    controller_weight = float(config["training"]["controller_l2_weight"])
    generator = torch.Generator().manual_seed(int(seed) + 9000)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    records: list[dict[str, float | int]] = []
    operator.reset_call_counts()
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(train.truth), generator=generator)
        model.train()
        losses = []
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            truth = train.truth[indices].to(device)
            observation = train.observation_uv[indices].to(device)
            sigma = train.sigma_by_view[indices].to(device)
            mask = train.view_mask[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            result = exact_line_search_reconstruction(
                operator,
                observation,
                sigma_by_view=sigma,
                view_mask=mask,
                rays_per_view=int(config["geometry"]["rays_per_view"]),
                stages=int(config["solver"]["stages"]),
                direction=model,
            )
            loss = normalized_field_loss(
                result.volume,
                truth,
                gradient_weight=float(config["training"]["gradient_weight"]),
            ).mean()
            coefficients = torch.stack(
                [
                    row["controller_coefficients"].square().mean()
                    for row in result.history
                ]
            ).mean()
            objective = loss + controller_weight * coefficients
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch % validation_every == 0 or epoch == epochs:
            model.eval()
            validation_loss = _validation_loss(
                split=validation,
                operator=operator,
                config=config,
                device=device,
                direction=model,
            )
            records.append(
                {
                    "epoch": epoch,
                    "train_combined_loss": float(np.mean(losses)),
                    "validation_combined_loss": validation_loss,
                }
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
    if best_state is None:
        raise RuntimeError("training never produced a validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    _synchronize(device)
    calls = operator.call_report()
    report = {
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_validation_combined_loss": float(best_loss),
        "training_wall_seconds": float(time.perf_counter() - started),
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "checkpoint_sha256": _state_sha256(best_state),
        "batch_invocations": {
            "forward": int(calls["forward_calls"]),
            "adjoint": int(calls["adjoint_calls"]),
        },
        "learning_curve": records,
    }
    return model, report


def _aggregate(
    rows: list[dict[str, Any]],
    *,
    baseline_method: str,
) -> list[dict[str, Any]]:
    baseline = {
        (row["split"], row["sample_id"]): float(row["field_relative_l2"])
        for row in rows
        if row["method"] == baseline_method
    }
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["split"], row["method"]), []).append(row)
    output = []
    for (split, method), values in sorted(groups.items()):
        field = np.asarray([row["field_relative_l2"] for row in values])
        gradient = np.asarray([row["gradient_relative_l2"] for row in values])
        front = np.asarray([row["front_top10_f1"] for row in values])
        combined = np.asarray([row["combined_loss"] for row in values])
        measurement = np.asarray([row["measurement_relative_l2"] for row in values])
        gains = np.asarray(
            [
                100.0
                * (
                    baseline[(split, row["sample_id"])]
                    - float(row["field_relative_l2"])
                )
                / max(baseline[(split, row["sample_id"])], 1e-12)
                for row in values
            ]
        )
        output.append(
            {
                "split": split,
                "method": method,
                "sample_count": len(values),
                "field_relative_l2_mean": float(np.mean(field)),
                "field_relative_l2_median": float(np.median(field)),
                "field_relative_l2_p90": float(np.quantile(field, 0.90)),
                "gradient_relative_l2_mean": float(np.mean(gradient)),
                "front_top10_f1_mean": float(np.mean(front)),
                "combined_loss_mean": float(np.mean(combined)),
                "measurement_relative_l2_mean": float(np.mean(measurement)),
                "field_gain_vs_sobolev_mean_percent": float(np.mean(gains)),
                "field_gain_vs_sobolev_p10_percent": float(np.quantile(gains, 0.10)),
                "field_harm_over_one_percent_rate": float(np.mean(gains < -1.0)),
            }
        )
    return output


def _candidate_gates(
    aggregates: list[dict[str, Any]],
    *,
    seeds: list[int],
    config: dict[str, Any],
) -> dict[str, Any]:
    by_key = {
        (row["split"], row["method"]): row
        for row in aggregates
    }
    thresholds = config["audit_gates"]
    seed_rows = []
    passed = 0
    for seed in seeds:
        method = f"learned_seed_{seed}"
        iid = by_key[("test_iid", method)]
        family = by_key[("test_family_ood", method)]
        joint = by_key[("test_joint_ood", method)]
        exact = by_key[("test_exact_operator_control", method)]
        gates = {
            "iid_mean_gain": iid["field_gain_vs_sobolev_mean_percent"]
            >= float(thresholds["iid_mean_field_gain_percent_minimum"]),
            "family_ood_mean_gain": family["field_gain_vs_sobolev_mean_percent"]
            >= float(thresholds["family_ood_mean_field_gain_percent_minimum"]),
            "joint_ood_mean_gain": joint["field_gain_vs_sobolev_mean_percent"]
            >= float(thresholds["joint_ood_mean_field_gain_percent_minimum"]),
            "iid_harm_rate": iid["field_harm_over_one_percent_rate"]
            <= float(thresholds["iid_harm_over_one_percent_rate_maximum"]),
            "exact_operator_no_material_harm": exact[
                "field_gain_vs_sobolev_mean_percent"
            ]
            >= float(
                thresholds["exact_operator_control_mean_gain_percent_minimum"]
            ),
        }
        seed_pass = all(gates.values())
        passed += int(seed_pass)
        seed_rows.append(
            {
                "seed": int(seed),
                "method": method,
                "gates": gates,
                "pass": seed_pass,
            }
        )
    required = int(thresholds["seeds_required_to_pass"])
    return {
        "per_seed": seed_rows,
        "passing_seed_count": int(passed),
        "required_passing_seed_count": required,
        "candidate_gate_pass": passed >= required,
    }


def build_public_summary(private: dict[str, Any]) -> dict[str, Any]:
    """Remove paths, masks, observations, per-sample values, and checkpoints."""

    return {
        "schema_version": PUBLIC_SCHEMA,
        "status": private["status"],
        "evidence_scope": private["evidence_scope"],
        "configuration": copy.deepcopy(private["configuration_public"]),
        "dataset": copy.deepcopy(private["dataset_public"]),
        "sobolev_selection": copy.deepcopy(private["sobolev_selection"]),
        "training": [
            {
                key: value
                for key, value in row.items()
                if key
                in {
                    "seed",
                    "best_epoch",
                    "best_validation_combined_loss",
                    "training_wall_seconds",
                    "parameter_count",
                    "learning_curve",
                }
            }
            for row in private["training"]
        ],
        "aggregates": copy.deepcopy(private["aggregates"]),
        "execution": copy.deepcopy(private["execution"]),
        "candidate_gates": copy.deepcopy(private["candidate_gates"]),
        "gates": copy.deepcopy(private["gates"]),
        "claim_boundary": copy.deepcopy(private["claim_boundary"]),
        "public_export_policy": {
            "contains_local_paths": False,
            "contains_ray_coordinates_or_masks": False,
            "contains_observation_or_volume_values": False,
            "contains_per_sample_metrics": False,
            "contains_checkpoint_hashes_or_weights": False,
            "contains_development_rotation_40_or_final_audit_values": False,
        },
    }


def run_pilot(
    *,
    config_path: Path,
    view_root: Path,
    checkpoint_dir: Path | None,
    device_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path)
    device = torch.device(device_name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    torch.set_num_threads(8)
    grid_size = int(config["geometry"]["grid_size"])
    rays_per_view = int(config["geometry"]["rays_per_view"])
    support = zero_outer_boundary_support(
        (grid_size,) * 3,
        dtype=torch.float32,
    ).to(device)
    geometry_started = time.perf_counter()
    true_geometry, true_provenance = load_real_support_geometry(
        view_root,
        rays_per_view=rays_per_view,
        sample_count=int(config["geometry"]["true_finite_aperture_sample_count"]),
    )
    nominal_geometry, nominal_provenance = load_real_support_geometry(
        view_root,
        rays_per_view=rays_per_view,
        sample_count=int(config["geometry"]["nominal_finite_aperture_sample_count"]),
    )
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
    geometry_seconds = time.perf_counter() - geometry_started

    splits: dict[str, SyntheticSplit] = {}
    train_validation_masks: set[str] = set()
    split_mask_keys: dict[str, set[str]] = {}
    data_started = time.perf_counter()
    for name, spec in config["data"]["splits"].items():
        forbidden = (
            train_validation_masks
            if name in {"train", "validation"}
            else set(train_validation_masks)
        )
        split, used_masks = _build_split(
            name=name,
            spec=spec,
            config=config,
            true_operator=true_operator,
            nominal_operator=nominal_operator,
            device=device,
            forbidden_masks=forbidden,
        )
        splits[name] = split
        split_keys = used_masks - forbidden
        split_mask_keys[name] = split_keys
        if name in {"train", "validation"}:
            train_validation_masks = used_masks
    data_seconds = time.perf_counter() - data_started

    validation = splits["validation"]
    strength_rows = []
    best_strength = None
    best_validation = float("inf")
    for strength in config["solver"]["sobolev_strength_grid"]:
        direction = FixedSobolevDirection(
            (grid_size,) * 3,
            strength=float(strength),
        ).to(device)
        value = _validation_loss(
            split=validation,
            operator=nominal_operator,
            config=config,
            device=device,
            direction=direction,
        )
        strength_rows.append(
            {
                "strength": float(strength),
                "validation_combined_loss": float(value),
            }
        )
        if value < best_validation:
            best_validation = value
            best_strength = float(strength)
    if best_strength is None:
        raise RuntimeError("Sobolev selection did not produce a result")

    training_records = []
    models: dict[int, PositiveSpectralDirection] = {}
    for seed in [int(value) for value in config["training"]["seeds"]]:
        model, training = _train_seed(
            seed=seed,
            train=splits["train"],
            validation=validation,
            operator=nominal_operator,
            config=config,
            base_strength=best_strength,
            device=device,
        )
        models[seed] = model
        training_records.append(training)
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "seed": seed,
                    "base_sobolev_strength": best_strength,
                    "config_sha256": _sha256(config_path),
                },
                checkpoint_dir / f"learned_seed_{seed}.pt",
            )

    all_rows: list[dict[str, Any]] = []
    execution_rows = []
    evaluation_splits = [
        name for name in splits if name not in {"train"}
    ]
    methods: list[tuple[str, Any | None]] = [
        ("identity", IdentityDirection()),
        (
            "sobolev_selected",
            FixedSobolevDirection(
                (grid_size,) * 3,
                strength=best_strength,
            ).to(device),
        ),
        ("cgls", None),
    ]
    methods.extend(
        (f"learned_seed_{seed}", model)
        for seed, model in models.items()
    )
    for split_name in evaluation_splits:
        for method, direction in methods:
            rows, execution = _evaluate(
                method=method,
                split=splits[split_name],
                operator=nominal_operator,
                config=config,
                device=device,
                direction=direction,
            )
            all_rows.extend(rows)
            execution_rows.append(execution)

    aggregates = _aggregate(
        all_rows,
        baseline_method="sobolev_selected",
    )
    candidate = _candidate_gates(
        aggregates,
        seeds=[int(value) for value in config["training"]["seeds"]],
        config=config,
    )
    monotone = all(
        row["data_objective_monotone"]
        for row in execution_rows
        if row["method"] != "cgls"
    )
    learned_calls_match = all(
        row["logical_calls_per_sample"]
        == {
            "forward": int(config["solver"]["stages"]),
            "adjoint": int(config["solver"]["stages"]),
        }
        for row in execution_rows
        if row["method"].startswith("learned_seed_")
    )
    train_validation_disjoint = split_mask_keys["train"].isdisjoint(
        split_mask_keys["validation"]
    )
    audit_excludes_train_validation = all(
        keys.isdisjoint(train_validation_masks)
        for name, keys in split_mask_keys.items()
        if name not in {"train", "validation"}
    )
    all_finite = all(
        np.isfinite(float(row[key]))
        for row in all_rows
        for key in (
            "field_relative_l2",
            "gradient_relative_l2",
            "front_top10_f1",
            "combined_loss",
            "measurement_relative_l2",
        )
    )
    gates = {
        "all_metrics_finite": all_finite,
        "train_validation_masks_disjoint": train_validation_disjoint,
        "every_audit_split_excludes_train_and_validation_masks": (
            audit_excludes_train_validation
        ),
        "line_search_data_objective_monotone": monotone,
        "learned_and_fixed_sobolev_calls_match": learned_calls_match,
        "sobolev_selected_on_validation_only": True,
        "audit_splits_not_used_for_training_or_epoch_selection": True,
        "development_rotation_40_not_accessed": True,
        "final_audit_not_accessed": True,
    }
    public_split_rows = []
    for split in splits.values():
        mismatch = split.operator_mismatch_relative_l2.numpy()
        public_split_rows.append(
            {
                "name": split.name,
                "sample_count": len(split.truth),
                "families": sorted(set(split.families)),
                "truth_operator": split.truth_operator,
                "active_view_count_minimum": int(
                    torch.sum(split.view_mask, dim=1).min()
                ),
                "active_view_count_maximum": int(
                    torch.sum(split.view_mask, dim=1).max()
                ),
                "relative_noise_minimum": float(split.relative_noise.min()),
                "relative_noise_maximum": float(split.relative_noise.max()),
                "operator_mismatch_relative_l2_mean": float(np.mean(mismatch)),
                "operator_mismatch_relative_l2_maximum": float(np.max(mismatch)),
            }
        )
    private = {
        "schema_version": PRIVATE_SCHEMA,
        "status": (
            "SPECTRAL_PRECONDITIONER_PILOT_CANDIDATE_PASS_SYNTHETIC_ONLY"
            if all(gates.values()) and candidate["candidate_gate_pass"]
            else "SPECTRAL_PRECONDITIONER_PILOT_CANDIDATE_NO_GO_OR_INCOMPLETE"
        ),
        "evidence_scope": config["evidence_scope"],
        "configuration_private": {
            "config_path": str(config_path.resolve()),
            "config_sha256": _sha256(config_path),
            "view_root": str(view_root.resolve()),
            "device": device_name,
            "torch_threads": int(torch.get_num_threads()),
        },
        "configuration_public": {
            "config_filename": config_path.name,
            "config_sha256": _sha256(config_path),
            "grid_shape_zyx": [grid_size, grid_size, grid_size],
            "view_count": int(config["geometry"]["view_count"]),
            "rays_per_view": rays_per_view,
            "true_finite_aperture_sample_count": int(
                config["geometry"]["true_finite_aperture_sample_count"]
            ),
            "nominal_finite_aperture_sample_count": int(
                config["geometry"]["nominal_finite_aperture_sample_count"]
            ),
            "stages": int(config["solver"]["stages"]),
            "dtype": "float32",
            "device": device_name,
            "phantom_schema": PHANTOM_SCHEMA,
            "preconditioner_schema": SPECTRAL_PRECONDITIONER_SCHEMA,
        },
        "dataset_private": {
            "true_geometry_provenance": true_provenance,
            "nominal_geometry_provenance": nominal_provenance,
            "per_sample_metrics": all_rows,
        },
        "dataset_public": {
            "name": (
                "PSU support-view geometry with deterministic analytic "
                "reaction-flow morphology proxies"
            ),
            "source_dataset_doi": "10.26208/1VE2-5C19",
            "real_psu_measurement_values_used": False,
            "analytic_truth_is_cfd": False,
            "mask_contract": config["data"]["mask_contract"],
            "train_validation_masks_disjoint": train_validation_disjoint,
            "every_audit_split_excludes_train_and_validation_masks": (
                audit_excludes_train_validation
            ),
            "splits": public_split_rows,
        },
        "sobolev_selection": {
            "selection_split": "validation",
            "grid": strength_rows,
            "selected_strength": best_strength,
            "selected_validation_combined_loss": best_validation,
        },
        "training": training_records,
        "aggregates": aggregates,
        "execution": {
            "geometry_build_seconds": float(geometry_seconds),
            "dataset_generation_seconds": float(data_seconds),
            "evaluation_records": execution_rows,
            "process_max_rss_bytes": int(_max_rss_bytes()),
            "host": {
                "machine": platform.machine(),
                "platform": platform.platform(),
                "torch_version": torch.__version__,
            },
        },
        "candidate_gates": candidate,
        "gates": gates,
        "claim_boundary": copy.deepcopy(config["claim_boundary"]),
    }
    public = build_public_summary(private)
    return private, public


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--public-output", type=Path)
    args = parser.parse_args()
    private, public = run_pilot(
        config_path=args.config,
        view_root=args.view_root,
        checkpoint_dir=args.checkpoint_dir,
        device_name=args.device,
    )
    if args.private_output is not None:
        args.private_output.parent.mkdir(parents=True, exist_ok=True)
        args.private_output.write_text(
            json.dumps(private, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.public_output is not None:
        args.public_output.parent.mkdir(parents=True, exist_ok=True)
        args.public_output.write_text(
            json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
