#!/usr/bin/env python3
"""Train a tiny covariance/geometry-conditioned unrolled BOST mechanism test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .bost_physics import build_forward_matrix, make_grid_3d, make_phantom, support_window
    from .candidate_operator_models import CovarianceGeometryUnrolledOperator
except ImportError:
    from bost_physics import build_forward_matrix, make_grid_3d, make_phantom, support_window
    from candidate_operator_models import CovarianceGeometryUnrolledOperator


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "candidate_operator_smoke.json"


@dataclass
class Bundle:
    field: torch.Tensor
    observation: torch.Tensor
    operator: torch.Tensor
    geometry: torch.Tensor
    active_mask: torch.Tensor
    noise_std: torch.Tensor
    noise_level: torch.Tensor
    support: torch.Tensor
    voxel_features: torch.Tensor
    seeds: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def select_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def geometry_table(angles: np.ndarray, detector_count: int, depth: int) -> np.ndarray:
    detector = np.linspace(-1.0, 1.0, detector_count, dtype=np.float32)
    z_axis = np.linspace(-1.0, 1.0, depth, dtype=np.float32)
    rows = []
    for z in z_axis:
        for angle in angles:
            radians = np.deg2rad(float(angle))
            for coordinate in detector:
                rows.append([np.sin(radians), np.cos(radians), coordinate, z])
    return np.asarray(rows, dtype=np.float32)


def block_operator(n: int, depth: int, angles: np.ndarray) -> tuple[np.ndarray, float]:
    slice_operator = build_forward_matrix(n, angles).reshape(len(angles) * n, n * n)
    matrix = np.kron(np.eye(depth, dtype=np.float32), slice_operator).astype(np.float32)
    spectral = float(np.linalg.svd(matrix, compute_uv=False)[0])
    return matrix / spectral, spectral


def camera_mask(
    rng: np.random.Generator,
    view_count: int,
    active_counts: list[int],
    limited_arc: bool,
) -> np.ndarray:
    count = int(active_counts[int(rng.integers(0, len(active_counts)))])
    mask = np.zeros(view_count, dtype=np.float32)
    if limited_arc:
        start = int(rng.integers(0, view_count - count + 1))
        mask[start : start + count] = 1.0
    else:
        mask[rng.choice(view_count, size=count, replace=False)] = 1.0
    return mask


def ray_mask(mask: np.ndarray, detector_count: int, depth: int) -> np.ndarray:
    return np.tile(np.repeat(mask, detector_count), depth).astype(np.float32)


def noise_profile(
    rng: np.random.Generator,
    view_count: int,
    detector_count: int,
    depth: int,
    severity: float,
) -> np.ndarray:
    detector = np.linspace(-1.0, 1.0, detector_count, dtype=np.float32)
    edge = 0.65 + severity * 0.85 * np.abs(detector) ** 1.5
    camera = rng.lognormal(mean=0.0, sigma=0.18 * severity, size=view_count).astype(np.float32)
    camera /= np.exp(np.mean(np.log(camera)))
    profile = camera[:, None] * edge[None, :]
    profile /= np.sqrt(np.mean(profile**2))
    return np.tile(profile.reshape(-1), depth).astype(np.float32)


def split_spec(name: str, config: dict) -> dict[str, object]:
    if name in {"train", "validation", "iid"}:
        return {
            "families": list(config["train_families"]),
            "noise": list(config["train_noise_levels"]),
            "limited_arc": False,
            "shifted_geometry": False,
            "noise_severity": 1.0,
        }
    if name == "noise_ood":
        return {
            "families": list(config["train_families"]),
            "noise": [0.10, 0.14],
            "limited_arc": False,
            "shifted_geometry": False,
            "noise_severity": 1.8,
        }
    if name == "family_ood":
        return {
            "families": ["thin_front"],
            "noise": list(config["train_noise_levels"]),
            "limited_arc": False,
            "shifted_geometry": False,
            "noise_severity": 1.0,
        }
    if name == "layout_ood":
        return {
            "families": list(config["train_families"]),
            "noise": list(config["train_noise_levels"]),
            "limited_arc": True,
            "shifted_geometry": True,
            "noise_severity": 1.0,
        }
    if name == "joint_ood":
        return {
            "families": ["thin_front"],
            "noise": [0.10, 0.14],
            "limited_arc": True,
            "shifted_geometry": True,
            "noise_severity": 1.8,
        }
    raise ValueError(f"unknown split {name}")


def make_bundle(name: str, count: int, config: dict, seed_offset: int) -> tuple[Bundle, float]:
    n = int(config["grid_size"])
    depth = int(config["depth"])
    views = int(config["candidate_views"])
    spec = split_spec(name, config)
    offset = 7.5 if bool(spec["shifted_geometry"]) else 0.0
    angles = np.linspace(0.0, 180.0, views, endpoint=False, dtype=np.float32) + offset
    operator, spectral = block_operator(n, depth, angles)
    geometry = geometry_table(angles, n, depth)
    support = support_window(n, depth).astype(np.float32).reshape(-1)
    xx, yy, zz = make_grid_3d(n, depth)
    voxel_features = np.stack([xx, yy, zz], axis=-1).astype(np.float32).reshape(-1, 3)
    fields = []
    observations = []
    masks = []
    stds = []
    noise_levels = []
    operators = []
    geometries = []
    seeds = []
    base_seed = int(config["seed"]) + int(seed_offset)
    families = list(spec["families"])
    noises = list(spec["noise"])
    active_counts = [int(value) for value in config["train_active_views"]]
    for index in range(count):
        seed = base_seed + index
        rng = np.random.default_rng(seed)
        family = str(families[index % len(families)])
        relative_noise = float(noises[index % len(noises)])
        field = make_phantom(family, n, depth, rng).reshape(-1)
        mask_camera = camera_mask(
            rng,
            views,
            active_counts,
            limited_arc=bool(spec["limited_arc"]),
        )
        mask = ray_mask(mask_camera, n, depth)
        std = noise_profile(
            rng,
            views,
            n,
            depth,
            severity=float(spec["noise_severity"]),
        )
        clean = operator @ field
        active_rms = float(np.sqrt(np.mean(clean[mask > 0.5] ** 2)) + 1e-8)
        noisy = clean + rng.normal(scale=relative_noise * active_rms * std).astype(np.float32)
        fields.append(field)
        observations.append(noisy)
        masks.append(mask)
        stds.append(std)
        noise_levels.append(relative_noise)
        operators.append(operator)
        geometries.append(geometry)
        seeds.append(seed)
    return (
        Bundle(
            field=torch.from_numpy(np.asarray(fields, dtype=np.float32)),
            observation=torch.from_numpy(np.asarray(observations, dtype=np.float32)),
            operator=torch.from_numpy(np.asarray(operators, dtype=np.float32)),
            geometry=torch.from_numpy(np.asarray(geometries, dtype=np.float32)),
            active_mask=torch.from_numpy(np.asarray(masks, dtype=np.float32)),
            noise_std=torch.from_numpy(np.asarray(stds, dtype=np.float32)),
            noise_level=torch.from_numpy(np.asarray(noise_levels, dtype=np.float32)),
            support=torch.from_numpy(support),
            voxel_features=torch.from_numpy(voxel_features),
            seeds=np.asarray(seeds, dtype=np.int64),
        ),
        spectral,
    )


def subset(bundle: Bundle, indices: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    tensor_index = torch.from_numpy(indices.astype(np.int64))
    return {
        "field": bundle.field[tensor_index].to(device),
        "observation": bundle.observation[tensor_index].to(device),
        "operator": bundle.operator[tensor_index].to(device),
        "geometry": bundle.geometry[tensor_index].to(device),
        "active_mask": bundle.active_mask[tensor_index].to(device),
        "noise_std": bundle.noise_std[tensor_index].to(device),
        "noise_level": bundle.noise_level[tensor_index].to(device),
        "support": bundle.support.to(device),
        "voxel_features": bundle.voxel_features.to(device),
    }


def model_prediction(
    model: CovarianceGeometryUnrolledOperator,
    batch: dict[str, torch.Tensor],
    use_noise: bool,
    guard: dict[str, float] | None = None,
) -> torch.Tensor:
    budget = None
    if guard is not None:
        reference = float(guard["reference_noise"])
        power = float(guard["power"])
        budget = torch.clamp((reference / batch["noise_level"].clamp_min(reference)) ** power, max=1.0)
    return model(
        observation=batch["observation"],
        operator=batch["operator"],
        geometry=batch["geometry"],
        active_mask=batch["active_mask"],
        support=batch["support"],
        voxel_features=batch["voxel_features"],
        correction_budget=budget,
        noise_std=batch["noise_std"] if use_noise else None,
    )


def relative_l2(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(prediction - truth, dim=1) / torch.linalg.vector_norm(
        truth, dim=1
    ).clamp_min(1e-8)


@torch.no_grad()
def evaluate(
    model: CovarianceGeometryUnrolledOperator,
    bundle: Bundle,
    device: torch.device,
    batch_size: int,
    use_noise: bool,
    guard: dict[str, float] | None = None,
) -> np.ndarray:
    model.eval()
    values = []
    for start in range(0, len(bundle.field), batch_size):
        indices = np.arange(start, min(start + batch_size, len(bundle.field)))
        batch = subset(bundle, indices, device)
        values.append(
            relative_l2(model_prediction(model, batch, use_noise, guard=guard), batch["field"]).cpu()
        )
    return torch.cat(values).numpy()


def make_model(config: dict, device: torch.device) -> CovarianceGeometryUnrolledOperator:
    n = int(config["grid_size"])
    depth = int(config["depth"])
    spec = config["model"]
    return CovarianceGeometryUnrolledOperator(
        voxel_count=depth * n * n,
        geometry_features=4,
        voxel_features=3,
        iterations=int(spec["iterations"]),
        context_features=int(spec["context_features"]),
        hidden_features=int(spec["hidden_features"]),
        initial_step=float(spec["initial_step"]),
        correction_scale=float(spec["correction_scale"]),
    ).to(device)


def train_model(
    config: dict,
    train: Bundle,
    validation: Bundle,
    device: torch.device,
) -> tuple[CovarianceGeometryUnrolledOperator, list[dict[str, float]], int]:
    model = make_model(config, device)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    batch_size = int(training["batch_size"])
    generator = np.random.default_rng(int(config["seed"]) + 991)
    best_epoch = 0
    best_error = float("inf")
    best_state = None
    history = []
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        order = generator.permutation(len(train.field))
        losses = []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch = subset(train, indices, device)
            prediction = model_prediction(model, batch, use_noise=True)
            loss = torch.mean((prediction - batch["field"]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_error = float(
            np.mean(evaluate(model, validation, device, batch_size, use_noise=True))
        )
        history.append(
            {
                "epoch": epoch,
                "train_mse": float(np.mean(losses)),
                "validation_relative_l2": validation_error,
            }
        )
        if validation_error < best_error:
            best_error = validation_error
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, history, best_epoch


def baseline_model(config: dict, device: torch.device) -> CovarianceGeometryUnrolledOperator:
    model = make_model(config, device)
    model.eval()
    return model


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output_dir or ROOT / "results" / "candidate_operator_smoke"
    output.mkdir(parents=True, exist_ok=True)
    config_snapshot = output / "config_snapshot.json"
    config_snapshot.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    device = select_device(args.device)

    train, train_scale = make_bundle("train", int(config["train_fields"]), config, 0)
    validation, validation_scale = make_bundle(
        "validation", int(config["validation_fields"]), config, 100_000
    )
    model, history, best_epoch = train_model(config, train, validation, device)
    selection_commit = {
        "best_epoch": best_epoch,
        "selection_metric": "mean source-field relative L2 on validation",
        "test_domains_constructed_before_selection": False,
        "train_field_count": len(train.field),
        "validation_field_count": len(validation.field),
        "train_operator_spectral_scale": train_scale,
        "validation_operator_spectral_scale": validation_scale,
    }
    (output / "selection_commit.json").write_text(
        json.dumps(selection_commit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    test_seed_base = int(config.get("test_seed_base_offset", 200_000))
    test_bundles = {
        name: make_bundle(
            name,
            int(config["test_fields_per_domain"]),
            config,
            test_seed_base + 100_000 * index,
        )[0]
        for index, name in enumerate(config["test_domains"])
    }
    all_bundles = {"validation": validation, **test_bundles}
    deterministic = baseline_model(config, device)
    batch_size = int(config["training"]["batch_size"])
    sample_rows = []
    summary_rows = []
    for split, bundle in all_bundles.items():
        methods = {
            "unweighted_pg4": evaluate(
                deterministic, bundle, device, batch_size, use_noise=False
            ),
            "prewhitened_pg4": evaluate(
                deterministic, bundle, device, batch_size, use_noise=True
            ),
            "cg_unrolled4": evaluate(model, bundle, device, batch_size, use_noise=True),
        }
        if "correction_guard" in config:
            methods["cg_unrolled4_guarded"] = evaluate(
                model,
                bundle,
                device,
                batch_size,
                use_noise=True,
                guard=config["correction_guard"],
            )
        for method, errors in methods.items():
            for seed, error in zip(bundle.seeds, errors):
                sample_rows.append(
                    {
                        "source_split": split,
                        "method": method,
                        "source_seed": int(seed),
                        "field_relative_l2": float(error),
                        "forward_calls": int(config["model"]["iterations"]),
                        "adjoint_calls": int(config["model"]["iterations"]),
                    }
                )
            comparator = methods["prewhitened_pg4"]
            gain = 100.0 * (comparator - errors) / comparator
            summary_rows.append(
                {
                    "source_split": split,
                    "method": method,
                    "field_count": len(errors),
                    "mean_relative_l2": float(np.mean(errors)),
                    "median_relative_l2": float(np.median(errors)),
                    "p90_relative_l2": float(np.quantile(errors, 0.9)),
                    "mean_gain_vs_prewhitened_pg4_pct": float(np.mean(gain)),
                    "harm_rate_gt_1pct_vs_prewhitened_pg4": float(np.mean(gain < -1.0)),
                }
            )
    write_csv(output / "history.csv", history)
    write_csv(output / "sample_metrics.csv", sample_rows)
    write_csv(output / "summary.csv", summary_rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].plot([row["epoch"] for row in history], [row["validation_relative_l2"] for row in history])
    axes[0].axvline(best_epoch, color="#a34f43", linestyle="--", label=f"selected {best_epoch}")
    axes[0].set(xlabel="epoch", ylabel="validation relative L2", title="Validation-only selection")
    axes[0].legend()
    domains = list(all_bundles)
    x = np.arange(len(domains))
    width = 0.25
    colors = ["#607178", "#315f93", "#146f66"]
    plotted_methods = ["unweighted_pg4", "prewhitened_pg4", "cg_unrolled4"]
    if "correction_guard" in config:
        plotted_methods.append("cg_unrolled4_guarded")
    width = 0.8 / len(plotted_methods)
    colors = ["#607178", "#315f93", "#a34f43", "#146f66"]
    for index, method in enumerate(plotted_methods):
        values = [
            next(
                row["mean_relative_l2"]
                for row in summary_rows
                if row["source_split"] == split and row["method"] == method
            )
            for split in domains
        ]
        offset = (index - (len(plotted_methods) - 1) / 2.0) * width
        axes[1].bar(x + offset, values, width, label=method, color=colors[index])
    axes[1].set_xticks(x, domains, rotation=28, ha="right")
    axes[1].set(ylabel="mean relative L2", title="Equal 4F + 4F^T mechanism smoke")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    figure_path = output / "candidate_operator_smoke.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    report = {
        "status": "MECHANISM_SMOKE_ONLY",
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "selection_commit": selection_commit,
        "test_seed_base_offset": test_seed_base,
        "summary": summary_rows,
        "claims_boundary": config["claims_boundary"],
        "interpretation_rule": (
            "learned gain is only interesting after subtracting deterministic prewhitening; "
            "no DeepONet/FNO/NeRIF superiority claim is allowed from this smoke"
        ),
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    public_files = [
        "config_snapshot.json",
        "selection_commit.json",
        "history.csv",
        "sample_metrics.csv",
        "summary.csv",
        "candidate_operator_smoke.png",
        "report.json",
    ]
    (output / "checksums.sha256").write_text(
        "\n".join(f"{sha256(output / name)}  {name}" for name in public_files) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
