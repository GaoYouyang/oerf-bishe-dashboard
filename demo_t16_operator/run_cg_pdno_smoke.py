#!/usr/bin/env python3
"""Train a tiny CG-PDNO implementation smoke on controlled phantoms.

This run intentionally remains an inverse-crime engineering check.  It proves
that the generic contract trains, that the zero-init path equals a deterministic
prewhitened solver, and that metrics/results are reproducible.  It cannot be
used as evidence of real BOST or public-dataset superiority.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .bost_physics import build_forward_matrix, forward_volume, make_phantom, support_window
    from .cg_pdno import CGPDNO
    from .measurement_contract import (
        BOSTBatch,
        DepthSeparableLinearBOST,
        inverse_variance_trust_budget,
    )
except ImportError:
    from bost_physics import build_forward_matrix, forward_volume, make_phantom, support_window
    from cg_pdno import CGPDNO
    from measurement_contract import (
        BOSTBatch,
        DepthSeparableLinearBOST,
        inverse_variance_trust_budget,
    )


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "cg_pdno_smoke.json"
RESULT_DIR = ROOT / "results" / "cg_pdno_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    return parser.parse_args()


def read_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_masks(
    count: int,
    views: int,
    active_counts: list[int],
    rng: np.random.Generator,
    forbidden: set[str],
) -> tuple[np.ndarray, list[str]]:
    rows: list[np.ndarray] = []
    identifiers: list[str] = []
    attempts = 0
    while len(rows) < count:
        active = int(active_counts[len(rows) % len(active_counts)])
        selected = np.sort(rng.choice(views, size=active, replace=False))
        mask = np.zeros(views, dtype=np.float32)
        mask[selected] = 1.0
        identifier = "g_" + "".join("1" if value else "0" for value in mask > 0.5)
        attempts += 1
        if identifier in forbidden:
            if attempts > 20000:
                raise RuntimeError("could not generate a disjoint geometry pool")
            continue
        rows.append(mask)
        identifiers.append(identifier)
    return np.stack(rows), identifiers


def build_batch(
    config: dict,
    split: str,
    operator_np: np.ndarray,
    forbidden_geometries: set[str],
) -> BOSTBatch:
    split_order = {"train": 0, "validation": 1, "test": 2}
    seed = int(config["data_seed"]) + 100_000 * split_order[split]
    rng = np.random.default_rng(seed)
    count = int(config["counts"][split])
    angles = np.asarray(config["view_angles_degrees"], dtype=np.float32)
    masks, geometry_ids = make_masks(
        count,
        len(angles),
        [int(value) for value in config["active_views"][split]],
        rng,
        forbidden_geometries,
    )
    fields = []
    observations = []
    sigmas = []
    camera_factors = np.asarray(config["camera_noise_factors"], dtype=np.float32)
    families = list(config["families"][split])
    noise_levels = [float(value) for value in config["relative_noise"][split]]
    n = int(config["grid_size"])
    depth = int(config["depth"])
    for index in range(count):
        field = make_phantom(families[index % len(families)], n, depth, rng)
        clean = forward_volume(field, operator_np).astype(np.float32)
        active = masks[index] > 0.5
        signal_rms = float(np.sqrt(np.mean(clean[:, active] ** 2)) + 1e-7)
        q = noise_levels[index % len(noise_levels)]
        sigma = q * signal_rms * camera_factors
        noisy = clean + rng.normal(
            scale=sigma[None, :, None], size=clean.shape
        ).astype(np.float32)
        noisy[:, ~active] = 0.0
        fields.append(field)
        observations.append(noisy)
        sigmas.append(sigma)
    support = support_window(n, depth).astype(np.float32)
    return BOSTBatch(
        observation=torch.from_numpy(np.stack(observations)),
        view_mask=torch.from_numpy(masks),
        noise_std=torch.from_numpy(np.stack(sigmas))[:, None, :, None],
        view_angles_degrees=torch.from_numpy(
            np.broadcast_to(angles, (count, len(angles))).copy()
        ),
        support=torch.from_numpy(support)[None, None],
        geometry_ids=tuple(geometry_ids),
        truth=torch.from_numpy(np.stack(fields))[:, None],
    ).validate()


def relative_l2(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    numerator = torch.linalg.vector_norm((prediction - truth).flatten(1), dim=1)
    denominator = torch.linalg.vector_norm(truth.flatten(1), dim=1).clamp_min(1e-12)
    return numerator / denominator


@torch.no_grad()
def evaluate(
    model: CGPDNO,
    batch: BOSTBatch,
    operator: DepthSeparableLinearBOST,
    lipschitz: torch.Tensor,
    trust_guard: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    start = torch.zeros_like(batch.truth)
    output = model(batch, operator, start, lipschitz)
    fallback = model.deterministic_fallback(batch, operator, start, lipschitz)
    active_count = batch.active_observation_mask().flatten(1).sum(dim=1).clamp_min(1)

    def whitened_residual_rms(field: torch.Tensor) -> torch.Tensor:
        residual = batch.observation - operator.forward(field, batch)
        return torch.sqrt(batch.whitened(residual).square().flatten(1).sum(dim=1) / active_count)

    fallback_norm = torch.linalg.vector_norm(fallback.flatten(1), dim=1).clamp_min(1e-12)
    result = {
        "model": relative_l2(output["prediction"], batch.truth).cpu().numpy(),
        "fallback": relative_l2(fallback, batch.truth).cpu().numpy(),
        "model_whitened_residual_rms": whitened_residual_rms(output["prediction"]).cpu().numpy(),
        "fallback_whitened_residual_rms": whitened_residual_rms(fallback).cpu().numpy(),
        "correction_to_fallback_norm": (
            torch.linalg.vector_norm((output["prediction"] - fallback).flatten(1), dim=1)
            / fallback_norm
        ).cpu().numpy(),
        "step": output["normalized_step"].cpu().numpy(),
        "proximal_scale": output["proximal_scale"].cpu().numpy(),
    }
    if trust_guard is not None:
        budget = inverse_variance_trust_budget(
            batch,
            reference_noise=float(trust_guard["reference_noise"]),
            power=float(trust_guard["power"]),
            minimum=float(trust_guard.get("minimum", 0.0)),
        )
        guarded = model(batch, operator, start, lipschitz, trust_budget=budget)
        result["guarded"] = relative_l2(guarded["prediction"], batch.truth).cpu().numpy()
        result["guarded_whitened_residual_rms"] = whitened_residual_rms(
            guarded["prediction"]
        ).cpu().numpy()
        result["trust_budget"] = budget.cpu().numpy()
    return result


def train_seed(
    config: dict,
    seed: int,
    train: BOSTBatch,
    validation: BOSTBatch,
    operator: DepthSeparableLinearBOST,
    train_lipschitz: torch.Tensor,
    validation_lipschitz: torch.Tensor,
) -> tuple[CGPDNO, list[dict[str, float]]]:
    torch.manual_seed(int(seed))
    model = CGPDNO(**config["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(config["training"]["epochs"])
    every = int(config["training"]["validation_every"])
    best_state = deepcopy(model.state_dict())
    best_validation = float("inf")
    history: list[dict[str, float]] = []
    start = torch.zeros_like(train.truth)
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(train, operator, start, train_lipschitz)
        field_loss = torch.mean((output["prediction"] - train.truth) ** 2)
        field_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        if epoch == 1 or epoch % every == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_output = model(
                    validation,
                    operator,
                    torch.zeros_like(validation.truth),
                    validation_lipschitz,
                )
                val_error = float(relative_l2(val_output["prediction"], validation.truth).mean())
                train_error = float(relative_l2(output["prediction"], train.truth).mean())
            history.append(
                {
                    "seed": int(seed),
                    "epoch": int(epoch),
                    "train_relative_l2": train_error,
                    "validation_relative_l2": val_error,
                }
            )
            if val_error < best_validation:
                best_validation = val_error
                best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, history


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    result_dir = args.output_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    angles = np.asarray(config["view_angles_degrees"], dtype=np.float32)
    operator_np = build_forward_matrix(int(config["grid_size"]), angles)
    operator = DepthSeparableLinearBOST(
        torch.from_numpy(operator_np),
        (int(config["depth"]), int(config["grid_size"]), int(config["grid_size"])),
    )
    train = build_batch(config, "train", operator_np, set())
    train_geometries = set(train.geometry_ids)
    validation = build_batch(config, "validation", operator_np, train_geometries)
    development_geometries = train_geometries | set(validation.geometry_ids)
    test = build_batch(config, "test", operator_np, development_geometries)
    train_lipschitz = operator.weighted_lipschitz(
        train, power_iterations=int(config["power_iterations"])
    )
    validation_lipschitz = operator.weighted_lipschitz(
        validation, power_iterations=int(config["power_iterations"])
    )
    test_lipschitz = operator.weighted_lipschitz(
        test, power_iterations=int(config["power_iterations"])
    )

    history_rows: list[dict[str, float]] = []
    sample_rows: list[dict[str, object]] = []
    summaries = []
    for seed in config["training"]["seeds"]:
        model, history = train_seed(
            config,
            int(seed),
            train,
            validation,
            operator,
            train_lipschitz,
            validation_lipschitz,
        )
        history_rows.extend(history)
        for split, batch, lipschitz in [
            ("validation", validation, validation_lipschitz),
            ("test", test, test_lipschitz),
        ]:
            result = evaluate(
                model,
                batch,
                operator,
                lipschitz,
                trust_guard=config.get("trust_guard"),
            )
            gain = 100.0 * (result["fallback"] - result["model"]) / result["fallback"]
            summary = {
                "seed": int(seed),
                "split": split,
                "model_mean_relative_l2": float(np.mean(result["model"])),
                "fallback_mean_relative_l2": float(np.mean(result["fallback"])),
                "mean_relative_gain_percent": float(np.mean(gain)),
                "harm_rate_over_1_percent": float(np.mean(gain < -1.0)),
            }
            if "guarded" in result:
                guarded_gain = (
                    100.0 * (result["fallback"] - result["guarded"]) / result["fallback"]
                )
                summary.update(
                    {
                        "guarded_mean_relative_l2": float(np.mean(result["guarded"])),
                        "guarded_mean_relative_gain_percent": float(np.mean(guarded_gain)),
                        "guarded_harm_rate_over_1_percent": float(
                            np.mean(guarded_gain < -1.0)
                        ),
                        "mean_trust_budget": float(np.mean(result["trust_budget"])),
                    }
                )
            summaries.append(summary)
            for index, geometry_id in enumerate(batch.geometry_ids):
                sample = {
                    "seed": int(seed),
                    "split": split,
                    "sample_index": int(index),
                    "geometry_id": geometry_id,
                    "active_views": int(batch.view_mask[index].sum()),
                    "model_relative_l2": float(result["model"][index]),
                    "fallback_relative_l2": float(result["fallback"][index]),
                    "relative_gain_percent": float(gain[index]),
                    "model_whitened_residual_rms": float(
                        result["model_whitened_residual_rms"][index]
                    ),
                    "fallback_whitened_residual_rms": float(
                        result["fallback_whitened_residual_rms"][index]
                    ),
                    "correction_to_fallback_norm": float(
                        result["correction_to_fallback_norm"][index]
                    ),
                }
                if "guarded" in result:
                    sample.update(
                        {
                            "guarded_relative_l2": float(result["guarded"][index]),
                            "guarded_relative_gain_percent": float(guarded_gain[index]),
                            "trust_budget": float(result["trust_budget"][index]),
                            "guarded_whitened_residual_rms": float(
                                result["guarded_whitened_residual_rms"][index]
                            ),
                        }
                    )
                sample_rows.append(sample)

    history_path = result_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(history_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(history_rows)
    sample_path = result_dir / "sample_metrics.csv"
    with sample_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(sample_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(sample_rows)

    figure_path = result_dir / "cg_pdno_smoke_learning.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for seed in config["training"]["seeds"]:
        rows = [row for row in history_rows if int(row["seed"]) == int(seed)]
        axes[0].plot(
            [row["epoch"] for row in rows],
            [row["validation_relative_l2"] for row in rows],
            marker="o",
            ms=3,
            label=f"seed {seed}",
        )
    axes[0].set(xlabel="epoch", ylabel="validation relative L2", title="Validation-locked smoke")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    validation_gain = [
        row["mean_relative_gain_percent"] for row in summaries if row["split"] == "validation"
    ]
    test_gain = [row["mean_relative_gain_percent"] for row in summaries if row["split"] == "test"]
    plot_values = [validation_gain, test_gain]
    plot_labels = ["val raw", "test raw"]
    if "trust_guard" in config:
        plot_values.extend(
            [
                [
                    row["guarded_mean_relative_gain_percent"]
                    for row in summaries
                    if row["split"] == "validation"
                ],
                [
                    row["guarded_mean_relative_gain_percent"]
                    for row in summaries
                    if row["split"] == "test"
                ],
            ]
        )
        plot_labels.extend(["val guarded", "test guarded"])
    axes[1].boxplot(plot_values, tick_labels=plot_labels)
    axes[1].axhline(0.0, color="black", lw=1)
    axes[1].set(ylabel="mean gain vs fixed 4-step solver (%)", title="Engineering signal only")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)

    report_path = result_dir / "report.json"
    report = {
        "evidence_label": config["evidence_label"],
        "claim_status": "NOT_AUTHORIZED_FOR_SUPERIORITY",
        "reason": "same linear generator/operator, small controlled phantoms, validation-selected checkpoints and no public or real BOST field truth",
        "contract": {
            "input": "observation + view mask + calibrated diagonal noise + view angles + support",
            "output": "3D nonnegative field",
            "forward_calls": int(config["model"]["stages"]),
            "adjoint_calls": int(config["model"]["stages"]),
            "learned_stop": False,
            "trust_guard": config.get("trust_guard"),
            "deterministic_fallback": "prewhitened projected gradient with fixed normalized step",
        },
        "adjoint_relative_error": operator.adjoint_relative_error(validation, seed=41),
        "geometry_overlap": {
            "train_validation": sorted(train_geometries & set(validation.geometry_ids)),
            "development_test": sorted(development_geometries & set(test.geometry_ids)),
        },
        "summaries": summaries,
        "aggregate": {
            split: {
                "model_mean_relative_l2": float(
                    np.mean([row["model_mean_relative_l2"] for row in summaries if row["split"] == split])
                ),
                "fallback_mean_relative_l2": float(
                    np.mean([row["fallback_mean_relative_l2"] for row in summaries if row["split"] == split])
                ),
                "mean_relative_gain_percent": float(
                    np.mean([row["mean_relative_gain_percent"] for row in summaries if row["split"] == split])
                ),
                "maximum_harm_rate_over_1_percent": float(
                    np.max([row["harm_rate_over_1_percent"] for row in summaries if row["split"] == split])
                ),
            }
            for split in ["validation", "test"]
        },
        "guarded_aggregate": {
            split: {
                "model_mean_relative_l2": float(
                    np.mean(
                        [
                            row["guarded_mean_relative_l2"]
                            for row in summaries
                            if row["split"] == split
                        ]
                    )
                ),
                "mean_relative_gain_percent": float(
                    np.mean(
                        [
                            row["guarded_mean_relative_gain_percent"]
                            for row in summaries
                            if row["split"] == split
                        ]
                    )
                ),
                "maximum_harm_rate_over_1_percent": float(
                    np.max(
                        [
                            row["guarded_harm_rate_over_1_percent"]
                            for row in summaries
                            if row["split"] == split
                        ]
                    )
                ),
                "mean_trust_budget": float(
                    np.mean(
                        [row["mean_trust_budget"] for row in summaries if row["split"] == split]
                    )
                ),
            }
            for split in ["validation", "test"]
        }
        if "trust_guard" in config
        else None,
        "config": config,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    checksums = {
        path.name: sha256(path)
        for path in [history_path, sample_path, figure_path, report_path]
    }
    (result_dir / "checksums.json").write_text(
        json.dumps(checksums, indent=2), encoding="utf-8"
    )
    print(json.dumps({"raw": report["aggregate"], "guarded": report["guarded_aggregate"]}, indent=2))
    print(f"adjoint_relative_error={report['adjoint_relative_error']:.3e}")


if __name__ == "__main__":
    main()
