#!/usr/bin/env python3
"""Measure cross-architecture cost without turning partial FLOP counts into claims."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional

try:
    from .data import load_npz
    from .direct_operator_data import (
        prepare_direct_operator_data,
        replace_lift_with_ridge,
    )
    from .own_algorithm_data import append_ray_view_channels
    from .own_algorithm_models import ZeroInitializedRaySetAdapter
    from .run_direct_operator_pilot import tune_classical_baselines
    from .run_own_algorithm_benchmark import make_benchmark_model
    from .train_eval import (
        choose_device,
        gradient_mse,
        masked_relative_projection_loss,
        project_torch,
        set_seed,
        synchronize,
    )
except ImportError:
    from data import load_npz
    from direct_operator_data import prepare_direct_operator_data, replace_lift_with_ridge
    from own_algorithm_data import append_ray_view_channels
    from own_algorithm_models import ZeroInitializedRaySetAdapter
    from run_direct_operator_pilot import tune_classical_baselines
    from run_own_algorithm_benchmark import make_benchmark_model
    from train_eval import (
        choose_device,
        gradient_mse,
        masked_relative_projection_loss,
        project_torch,
        set_seed,
        synchronize,
    )


ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = ROOT / "configs"
DEFAULT_CONFIG = CONFIG_ROOT / "v3e_compute_accounting.json"
LABELS = {
    "ridge_unet_aug": "ridge residual 3D U-Net",
    "ridge_fno_aug": "ridge residual FNO",
    "ridge_deeponet": "ridge residual DeepONet",
    "ray_set_operator": "provisional ray-set operator",
    "zero_init_ray_set_adapter": "frozen-FNO zero-init ray-set adapter",
}
OUTPUT_FILES = [
    "v3e_compute_trials.csv",
    "v3e_compute_profiles.csv",
    "v3e_fno_error_compute_checkpoints.csv",
    "v3e_fno_time_to_target.csv",
    "v3e_compute_readiness.csv",
    "v3e_compute_dashboard.json",
    "v3e_compute_report.json",
    "t16_v3e_compute_accounting.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--repeat-limit", type=int)
    parser.add_argument("--model-limit", type=int)
    parser.add_argument("--worker-model", choices=list(LABELS))
    parser.add_argument("--worker-repeat", type=int)
    parser.add_argument("--worker-output", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def runtime_environment(requested_device: str) -> dict[str, object]:
    memory = command_output(["sysctl", "-n", "hw.memsize"])
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hardware_model": command_output(["sysctl", "-n", "hw.model"]),
        "processor": command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
        or platform.processor(),
        "physical_memory_bytes": None if memory is None else int(memory),
        "torch": torch.__version__,
        "neuraloperator": importlib.metadata.version("neuraloperator"),
        "numpy": np.__version__,
        "requested_device": requested_device,
    }


def write_v3e_checksums(output_dir: Path, filenames: list[str]) -> None:
    lines = []
    for filename in filenames:
        digest = hashlib.sha256((output_dir / filename).read_bytes()).hexdigest()
        lines.append(f"{digest}  {filename}")
    (output_dir / "v3e_compute_checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="ascii"
    )


def prepare_k_budget_data(experiment: dict[str, Any]) -> dict[str, np.ndarray]:
    source = load_npz(ROOT / "results" / str(experiment["source_dataset"]))
    direct = prepare_direct_operator_data(
        source,
        [int(experiment["total_budget"])],
        int(experiment["fixed_query_index"]),
        int(experiment["audit_query_index"]),
    )
    selected, champions, _ = tune_classical_baselines(
        direct,
        [int(experiment["total_budget"])],
        [float(value) for value in experiment["ridge_relative_grid"]],
    )
    if set(champions.values()) != {"ridge"}:
        raise RuntimeError("v3e expects validation-locked ridge input")
    return append_ray_view_channels(replace_lift_with_ridge(direct, selected))


def benchmark_config(experiment: dict[str, Any]) -> dict[str, Any]:
    dataset = read_json(CONFIG_ROOT / str(experiment["dataset_config"]))
    own = read_json(CONFIG_ROOT / str(experiment["own_algorithm_config"]))
    dataset["own_models"] = copy.deepcopy(own["models"])
    return dataset


def model_metadata(data: dict[str, np.ndarray]) -> dict[str, object]:
    names = [str(value) for value in data["input_channel_names"].tolist()]
    return {
        "view_start": int(data["ray_view_channel_start"]),
        "view_count": int(data["ray_view_channel_count"]),
        "mask_start": names.index("camera_0_active"),
        "angle_sin_start": int(data["ray_angle_sin_channel_start"]),
        "angle_cos_start": int(data["ray_angle_cos_channel_start"]),
        "coordinates": tuple(names.index(axis) for axis in ("z", "y", "x")),
    }


def make_cost_model(
    method: str,
    config: dict[str, Any],
    data: dict[str, np.ndarray],
) -> nn.Module:
    if method != "zero_init_ray_set_adapter":
        return make_benchmark_model(method, config, data)
    metadata = model_metadata(data)
    base = make_benchmark_model("ridge_fno_aug", config, data)
    return ZeroInitializedRaySetAdapter(
        base_operator=base,
        view_count=int(metadata["view_count"]),
        view_channel_start=int(metadata["view_start"]),
        mask_channel_start=int(metadata["mask_start"]),
        angle_sin_channel_start=int(metadata["angle_sin_start"]),
        angle_cos_channel_start=int(metadata["angle_cos_start"]),
        coordinate_channels=metadata["coordinates"],
    )


class AnalyticForwardCounter:
    """Shape-based partial operation count with explicit coverage boundaries."""

    def __init__(self) -> None:
        self.dense_macs = 0
        self.spectral_complex_macs = 0
        self.fft_flops_estimate = 0.0
        self.component_rows: list[dict[str, object]] = []
        self.handles: list[Any] = []

    def _record(self, module: nn.Module, category: str, value: float) -> None:
        if value <= 0:
            return
        self.component_rows.append(
            {
                "module": type(module).__name__,
                "category": category,
                "value": float(value),
            }
        )

    def _conv_hook(self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        source = inputs[0]
        kernel = math.prod(int(value) for value in module.kernel_size)
        if isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            spatial = math.prod(int(value) for value in source.shape[2:])
            macs = (
                int(source.shape[0])
                * spatial
                * int(module.in_channels)
                * (int(module.out_channels) // int(module.groups))
                * kernel
            )
        else:
            macs = int(output.numel()) * (int(module.in_channels) // int(module.groups)) * kernel
        self.dense_macs += int(macs)
        self._record(module, "dense_real_macs", macs)

    def _linear_hook(self, module: nn.Linear, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        macs = int(output.numel()) * int(module.in_features)
        self.dense_macs += macs
        self._record(module, "dense_real_macs", macs)

    def _spectral_hook(self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        source = inputs[0]
        batch = int(source.shape[0])
        spatial = math.prod(int(value) for value in source.shape[2:])
        retained_modes = math.prod(int(value) for value in module.n_modes)
        complex_macs = (
            batch
            * int(module.in_channels)
            * int(module.out_channels)
            * retained_modes
        )
        fft_flops = (
            2.5
            * spatial
            * math.log2(spatial)
            * batch
            * (int(module.in_channels) + int(module.out_channels))
        )
        self.spectral_complex_macs += complex_macs
        self.fft_flops_estimate += fft_flops
        self._record(module, "spectral_complex_macs", complex_macs)
        self._record(module, "fft_real_flops_estimate", fft_flops)

    def _custom_hook(self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        source = inputs[0]
        name = type(module).__name__
        if name == "GridDeepONetResidual":
            rank = int(module.branch[-1].out_features)
            voxels = math.prod(int(value) for value in source.shape[2:])
            macs = int(source.shape[0]) * voxels * rank
        elif name in {"RaySetResidualOperator", "RaySetAttentionEncoder"}:
            if name == "RaySetResidualOperator":
                features = int(module.view_encoder[0].out_channels)
            else:
                features = int(module.view_encoder[0].out_channels)
            voxels = math.prod(int(value) for value in source.shape[2:])
            # query-key dot, weighted value sum and weighted variance sum
            macs = int(source.shape[0]) * int(module.view_count) * features * voxels * 3
        else:
            return
        self.dense_macs += macs
        self._record(module, "explicit_contraction_real_macs", macs)

    def attach(self, model: nn.Module) -> None:
        convolution_types = (
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
        )
        for module in model.modules():
            if isinstance(module, convolution_types):
                self.handles.append(module.register_forward_hook(self._conv_hook))
            elif isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_hook(self._linear_hook))
            elif type(module).__name__ == "SpectralConv":
                self.handles.append(module.register_forward_hook(self._spectral_hook))
            elif type(module).__name__ in {
                "GridDeepONetResidual",
                "RaySetResidualOperator",
                "RaySetAttentionEncoder",
            }:
                self.handles.append(module.register_forward_hook(self._custom_hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def summary(self) -> dict[str, object]:
        estimated = (
            2.0 * self.dense_macs
            + 6.0 * self.spectral_complex_macs
            + self.fft_flops_estimate
        )
        return {
            "forward_dense_real_macs": int(self.dense_macs),
            "forward_spectral_complex_macs": int(self.spectral_complex_macs),
            "forward_fft_real_flops_estimate": float(self.fft_flops_estimate),
            "forward_estimated_flops_v1": float(estimated),
            "component_count": len(self.component_rows),
        }


def analytic_forward_profile(model: nn.Module, sample: torch.Tensor) -> dict[str, object]:
    counter = AnalyticForwardCounter()
    counter.attach(model)
    model.eval()
    with torch.no_grad():
        output = model(sample)
    counter.close()
    return {**counter.summary(), "output_shape": list(output.shape)}


class MPSMemorySampler:
    def __init__(self, interval_ms: float):
        self.interval = max(float(interval_ms) / 1000.0, 0.0005)
        self.allocated: list[int] = []
        self.driver: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def observe(self) -> None:
        if not torch.backends.mps.is_available():
            return
        try:
            self.allocated.append(int(torch.mps.current_allocated_memory()))
            self.driver.append(int(torch.mps.driver_allocated_memory()))
        except RuntimeError:
            return

    def _run(self) -> None:
        while not self._stop.is_set():
            self.observe()
            time.sleep(self.interval)

    def __enter__(self) -> "MPSMemorySampler":
        self.observe()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.observe()

    def summary(self) -> dict[str, int | None]:
        return {
            "mps_peak_allocated_bytes_observed": max(self.allocated) if self.allocated else None,
            "mps_peak_driver_bytes_observed": max(self.driver) if self.driver else None,
            "mps_memory_sample_count": len(self.allocated),
        }


def tensor_bytes(values: list[torch.Tensor]) -> int:
    return sum(int(value.numel()) * int(value.element_size()) for value in values)


def percentile_summary(values: list[float], prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_p10_ms": float(np.quantile(array, 0.10)),
        f"{prefix}_p50_ms": float(np.quantile(array, 0.50)),
        f"{prefix}_p90_ms": float(np.quantile(array, 0.90)),
        f"{prefix}_mean_ms": float(np.mean(array)),
    }


def current_process_rss_bytes() -> int | None:
    try:
        value = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True
        ).strip()
        return int(value) * 1024
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def process_peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def full_training_loss(
    model: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    observation: torch.Tensor,
    view_mask: torch.Tensor,
    operator: torch.Tensor,
    training: dict[str, Any],
) -> torch.Tensor:
    prediction = model(x)
    projected = project_torch(prediction, operator)
    field_loss = functional.mse_loss(prediction, target)
    gradient_loss = gradient_mse(prediction, target)
    projection_loss = masked_relative_projection_loss(
        projected, observation, view_mask
    )
    outside = (x[:, 1:2] < 0.02).to(prediction.dtype)
    boundary_loss = torch.mean((prediction * outside) ** 2)
    return (
        field_loss
        + float(training["lambda_gradient"]) * gradient_loss
        + float(training["lambda_reprojection"]) * projection_loss
        + float(training["lambda_boundary"]) * boundary_loss
    )


def worker_record(
    experiment: dict[str, Any],
    method: str,
    repeat: int,
    requested_device: str,
) -> dict[str, object]:
    seed = int(experiment["seed"]) + list(experiment["methods"]).index(method) * 101 + repeat
    set_seed(seed)
    data = prepare_k_budget_data(experiment)
    config = benchmark_config(experiment)
    train_id = int(np.where(np.asarray(data["split_names"]) == "train")[0][0])
    indices = np.flatnonzero(
        (data["total_budget"] == int(experiment["total_budget"]))
        & (data["split_id"] == train_id)
    )
    training_batch = int(experiment["training_batch_size"])
    if len(indices) < training_batch:
        raise RuntimeError("not enough K-budget training examples for compute probe")

    inference_batch = int(experiment["inference_batch_size"])
    cpu_probe = torch.from_numpy(data["inputs"][indices[:inference_batch]]).clone()
    # Neuraloperator's positional embedding caches the first device grid outside
    # the state dict. Keep the CPU shape probe and timed device model separate.
    analytic_model = make_cost_model(method, config, data)
    analytic = analytic_forward_profile(analytic_model, cpu_probe)
    del analytic_model
    set_seed(seed)
    model = make_cost_model(method, config, data)

    parameters = list(model.parameters())
    buffers = list(model.buffers())
    trainable = [value for value in parameters if value.requires_grad]
    parameter_bytes = tensor_bytes(parameters)
    trainable_parameter_bytes = tensor_bytes(trainable)
    buffer_bytes = tensor_bytes(buffers)
    total_parameters = sum(int(value.numel()) for value in parameters)
    trainable_parameters = sum(int(value.numel()) for value in trainable)

    device = choose_device(requested_device)
    model = model.to(device)
    x_inference = cpu_probe.to(device)
    selected = indices[:training_batch]
    x_train = torch.from_numpy(data["inputs"][selected]).to(device)
    target = torch.from_numpy(data["field"][selected, None]).to(device)
    observation = torch.from_numpy(data["observation"][selected]).to(device)
    view_mask = torch.from_numpy(data["view_mask"][selected]).to(device)
    operator = torch.from_numpy(data["forward_matrix"]).to(device)
    synchronize(device)
    if device.type == "mps":
        torch.mps.empty_cache()
        synchronize(device)
        resident_allocated = int(torch.mps.current_allocated_memory())
        resident_driver = int(torch.mps.driver_allocated_memory())
    else:
        resident_allocated = None
        resident_driver = None
    resident_rss = current_process_rss_bytes()

    model.eval()
    with MPSMemorySampler(float(experiment["memory_poll_interval_ms"])) as memory:
        with torch.no_grad():
            for _ in range(int(experiment["inference_warmup_steps"])):
                model(x_inference)
            synchronize(device)
            inference_times = []
            for _ in range(int(experiment["inference_measure_steps"])):
                start = time.perf_counter()
                model(x_inference)
                synchronize(device)
                inference_times.append((time.perf_counter() - start) * 1000.0)
                memory.observe()
    inference_memory = memory.summary()

    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    model.train()
    with MPSMemorySampler(float(experiment["memory_poll_interval_ms"])) as memory:
        for _ in range(int(experiment["training_warmup_steps"])):
            optimizer.zero_grad(set_to_none=True)
            loss = full_training_loss(
                model, x_train, target, observation, view_mask, operator, config["training"]
            )
            loss.backward()
            optimizer.step()
        synchronize(device)
        training_times = []
        for _ in range(int(experiment["training_measure_steps"])):
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = full_training_loss(
                model, x_train, target, observation, view_mask, operator, config["training"]
            )
            loss.backward()
            optimizer.step()
            synchronize(device)
            training_times.append((time.perf_counter() - start) * 1000.0)
            memory.observe()
    training_memory = memory.summary()

    inference_stats = percentile_summary(inference_times, "inference")
    training_stats = percentile_summary(training_times, "training_step")
    training_stats["training_step_p50_ms_per_sample"] = (
        training_stats["training_step_p50_ms"] / training_batch
    )
    return {
        "method": method,
        "label": LABELS[method],
        "repeat": repeat,
        "worker_pid": os.getpid(),
        "seed": seed,
        "device": str(device),
        "dtype": str(x_train.dtype),
        "grid": "8x16x16",
        "input_channels": int(x_train.shape[1]),
        "inference_batch_size": inference_batch,
        "training_batch_size": training_batch,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "frozen_parameters": total_parameters - trainable_parameters,
        "parameter_bytes": parameter_bytes,
        "trainable_parameter_bytes": trainable_parameter_bytes,
        "buffer_bytes": buffer_bytes,
        "training_state_bytes_floor": (
            parameter_bytes + buffer_bytes + 3 * trainable_parameter_bytes
        ),
        **analytic,
        **inference_stats,
        **training_stats,
        "inference_peak_mps_allocated_bytes_observed": inference_memory[
            "mps_peak_allocated_bytes_observed"
        ],
        "inference_peak_mps_driver_bytes_observed": inference_memory[
            "mps_peak_driver_bytes_observed"
        ],
        "training_peak_mps_allocated_bytes_observed": training_memory[
            "mps_peak_allocated_bytes_observed"
        ],
        "training_peak_mps_driver_bytes_observed": training_memory[
            "mps_peak_driver_bytes_observed"
        ],
        "resident_mps_allocated_bytes": resident_allocated,
        "resident_mps_driver_bytes": resident_driver,
        "training_mps_increment_bytes_observed": (
            None
            if resident_allocated is None
            or training_memory["mps_peak_allocated_bytes_observed"] is None
            else int(training_memory["mps_peak_allocated_bytes_observed"])
            - resident_allocated
        ),
        "mps_memory_sample_count": int(inference_memory["mps_memory_sample_count"])
        + int(training_memory["mps_memory_sample_count"]),
        "resident_process_rss_bytes": resident_rss,
        "process_peak_rss_bytes": process_peak_rss_bytes(),
        "checkpoint_weights_used": False,
        "timing_includes_data_loading": False,
        "training_timing_includes_full_physics_loss": True,
    }


def aggregate_trials(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    timing_metrics = [
        "inference_p50_ms",
        "inference_p90_ms",
        "training_step_p50_ms",
        "training_step_p90_ms",
        "training_step_p50_ms_per_sample",
    ]
    memory_metrics = [
        "training_peak_mps_allocated_bytes_observed",
        "training_peak_mps_driver_bytes_observed",
        "training_mps_increment_bytes_observed",
        "process_peak_rss_bytes",
    ]
    for method in dict.fromkeys(str(row["method"]) for row in rows):
        subset = [row for row in rows if str(row["method"]) == method]
        first = subset[0]
        record: dict[str, object] = {
            key: first[key]
            for key in (
                "method",
                "label",
                "device",
                "dtype",
                "grid",
                "input_channels",
                "inference_batch_size",
                "training_batch_size",
                "total_parameters",
                "trainable_parameters",
                "frozen_parameters",
                "parameter_bytes",
                "trainable_parameter_bytes",
                "buffer_bytes",
                "training_state_bytes_floor",
                "forward_dense_real_macs",
                "forward_spectral_complex_macs",
                "forward_fft_real_flops_estimate",
                "forward_estimated_flops_v1",
            )
        }
        record["worker_repeats"] = len(subset)
        for metric in timing_metrics:
            values = np.asarray([float(row[metric]) for row in subset])
            record[metric] = float(np.median(values))
            record[f"{metric}_worker_min"] = float(np.min(values))
            record[f"{metric}_worker_max"] = float(np.max(values))
            record[f"{metric}_worker_spread_pct"] = float(
                100.0 * (np.max(values) - np.min(values)) / max(np.median(values), 1e-12)
            )
        for metric in memory_metrics:
            values = [row[metric] for row in subset if row[metric] is not None]
            record[metric] = None if not values else int(max(int(value) for value in values))
        record["inference_samples_per_second"] = float(
            1000.0 * int(record["inference_batch_size"])
            / float(record["inference_p50_ms"])
        )
        record["training_samples_per_second"] = float(
            1000.0 * int(record["training_batch_size"])
            / float(record["training_step_p50_ms"])
        )
        output.append(record)
    return output


def fno_frontier(
    experiment: dict[str, Any]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    source = ROOT / "results" / "v3d_fno_optimizer_audit" / "v3d_optimizer_validation_summary.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    checkpoints: list[dict[str, object]] = []
    fixed = {int(value) for value in experiment["fixed_epoch_checkpoints"]}
    for row in rows:
        epoch = int(row["cumulative_epochs"])
        if epoch in fixed:
            checkpoints.append(
                {
                    "strategy": row["strategy"],
                    "cumulative_epochs": epoch,
                    "mean_validation_rel_l2": float(row["mean_validation_rel_l2"]),
                    "mean_cumulative_train_seconds": float(row["mean_cumulative_train_seconds"]),
                    "mean_selected_checkpoint_epoch": float(row["mean_selected_checkpoint_epoch"]),
                    "source": "v3d_validation_only_prefix_best",
                }
            )
    targets: list[dict[str, object]] = []
    for strategy in sorted({row["strategy"] for row in rows}):
        subset = sorted(
            (row for row in rows if row["strategy"] == strategy),
            key=lambda row: int(row["cumulative_epochs"]),
        )
        for target in experiment["fno_time_to_target_validation_rel_l2"]:
            crossing = next(
                (
                    row
                    for row in subset
                    if float(row["mean_validation_rel_l2"]) <= float(target)
                ),
                None,
            )
            targets.append(
                {
                    "strategy": strategy,
                    "target_validation_rel_l2": float(target),
                    "target_reached": crossing is not None,
                    "first_endpoint_epoch": None
                    if crossing is None
                    else int(crossing["cumulative_epochs"]),
                    "mean_cumulative_train_seconds": None
                    if crossing is None
                    else float(crossing["mean_cumulative_train_seconds"]),
                    "selection_scope": "validation_only",
                }
            )
    return checkpoints, targets


def readiness_rows(profiles: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for profile in profiles:
        method = str(profile["method"])
        rows.append(
            {
                "method": method,
                "cost_profile_complete": True,
                "fixed_240_epoch_validation_curve_complete": method == "ridge_fno_aug",
                "matched_optimizer_protocol_complete": method == "ridge_fno_aug",
                "matched_error_compute_frontier_complete": False,
                "real_or_group_geometry_data_complete": False,
                "confirmatory_superiority_eligible": False,
                "next_required_evidence": (
                    "geometry manifest plus matched 60/120/180/240 learning curve"
                    if method != "ridge_fno_aug"
                    else "candidate matched curve plus geometry manifest"
                ),
            }
        )
    return rows


def plot_profiles(profiles: list[dict[str, object]], path: Path) -> None:
    labels = [str(row["label"]) for row in profiles]
    x = np.arange(len(profiles))
    colors = ["#52727b", "#16827a", "#a6624e", "#765f91", "#c28a35"]
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.2), constrained_layout=True)

    params = np.asarray([float(row["trainable_parameters"]) / 1000.0 for row in profiles])
    total = np.asarray([float(row["total_parameters"]) / 1000.0 for row in profiles])
    axes[0].bar(x, total, color="#ccd5d6", label="total")
    axes[0].bar(x, params, color=colors[: len(profiles)], label="trainable")
    axes[0].set_ylabel("parameters (thousand)")
    axes[0].set_title("Model and trainable size")
    axes[0].legend(fontsize=8)

    inference = np.asarray([float(row["inference_p50_ms"]) for row in profiles])
    training = np.asarray([float(row["training_step_p50_ms_per_sample"]) for row in profiles])
    width = 0.36
    axes[1].bar(x - width / 2, inference, width, color="#16827a", label="inference B=1")
    axes[1].bar(x + width / 2, training, width, color="#c28a35", label="train / sample")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("synchronized MPS latency (ms, log)")
    axes[1].set_title("Measured latency")
    axes[1].legend(fontsize=8)

    flops = np.asarray([float(row["forward_estimated_flops_v1"]) / 1e6 for row in profiles])
    memory = np.asarray(
        [float(row["training_peak_mps_allocated_bytes_observed"] or 0) / 2**20 for row in profiles]
    )
    axes[2].scatter(
        flops,
        memory,
        s=110,
        c=colors[: len(profiles)],
        edgecolors="white",
        linewidths=1.2,
    )
    annotation_offsets = {
        "ridge_unet_aug": (6, 5),
        "ridge_fno_aug": (6, 9),
        "ridge_deeponet": (6, -12),
        "ray_set_operator": (6, 6),
        "zero_init_ray_set_adapter": (6, 6),
    }
    short_labels = {
        "ridge_unet_aug": "U-Net",
        "ridge_fno_aug": "FNO",
        "ridge_deeponet": "DeepONet",
        "ray_set_operator": "ray-set",
        "zero_init_ray_set_adapter": "adapter",
    }
    for index, row in enumerate(profiles):
        method = str(row["method"])
        axes[2].annotate(
            short_labels[method],
            (flops[index], memory[index]),
            xytext=annotation_offsets[method],
            textcoords="offset points",
            fontsize=8,
        )
    axes[2].set_xlabel("estimated forward FLOPs v1 (million)")
    axes[2].set_ylabel("observed MPS training allocation (MiB)")
    axes[2].set_title("Cost coordinates, not accuracy")
    axes[2].grid(True, alpha=0.22)

    for axis in axes[:2]:
        axis.set_xticks(x, [label.replace("ridge residual ", "") for label in labels], rotation=27, ha="right")
        axis.grid(True, axis="y", alpha=0.2)
    fig.suptitle("T16 v3e cross-architecture compute accounting (8x16x16, K=6)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_worker(args: argparse.Namespace, experiment: dict[str, Any]) -> None:
    if args.worker_output is None or args.worker_repeat is None:
        raise ValueError("worker mode requires --worker-output and --worker-repeat")
    requested = args.device or str(experiment["device"])
    record = worker_record(
        experiment, str(args.worker_model), int(args.worker_repeat), requested
    )
    args.worker_output.write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def run_driver(args: argparse.Namespace, experiment: dict[str, Any]) -> None:
    output_dir = args.output_dir or ROOT / "results" / str(experiment["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = [str(value) for value in experiment["methods"]]
    if args.model_limit is not None:
        methods = methods[: int(args.model_limit)]
    repeats = int(experiment["worker_repeats"])
    if args.repeat_limit is not None:
        repeats = min(repeats, int(args.repeat_limit))
    requested = args.device or str(experiment["device"])

    trials: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="t16-v3e-") as temp:
        temp_root = Path(temp)
        for method in methods:
            for repeat in range(repeats):
                worker_output = temp_root / f"{method}-{repeat}.json"
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--config",
                    str(args.config.resolve()),
                    "--worker-model",
                    method,
                    "--worker-repeat",
                    str(repeat),
                    "--worker-output",
                    str(worker_output),
                    "--device",
                    requested,
                ]
                completed = subprocess.run(
                    command,
                    cwd=ROOT.parent,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"worker failed for {method}/{repeat}:\n{completed.stdout}\n{completed.stderr}"
                    )
                row = read_json(worker_output)
                trials.append(row)
                print(
                    f"{method} repeat={repeat}: "
                    f"infer={float(row['inference_p50_ms']):.3f} ms "
                    f"train={float(row['training_step_p50_ms']):.3f} ms",
                    flush=True,
                )

    profiles = aggregate_trials(trials)
    checkpoints, targets = fno_frontier(experiment)
    readiness = readiness_rows(profiles)
    write_csv(output_dir / "v3e_compute_trials.csv", trials)
    write_csv(output_dir / "v3e_compute_profiles.csv", profiles)
    write_csv(output_dir / "v3e_fno_error_compute_checkpoints.csv", checkpoints)
    write_csv(output_dir / "v3e_fno_time_to_target.csv", targets)
    write_csv(output_dir / "v3e_compute_readiness.csv", readiness)
    plot_profiles(profiles, output_dir / "t16_v3e_compute_accounting.png")

    provenance = {
        "experiment_config_sha256": sha256_file(args.config.resolve()),
        "dataset_config_sha256": sha256_file(
            CONFIG_ROOT / str(experiment["dataset_config"])
        ),
        "own_algorithm_config_sha256": sha256_file(
            CONFIG_ROOT / str(experiment["own_algorithm_config"])
        ),
        "compute_script_sha256": sha256_file(Path(__file__).resolve()),
        "models_script_sha256": sha256_file(ROOT / "models.py"),
        "own_models_script_sha256": sha256_file(ROOT / "own_algorithm_models.py"),
        "requirements_sha256": sha256_file(ROOT / "requirements.txt"),
        "source_dataset_npz_sha256": sha256_file(
            ROOT / "results" / str(experiment["source_dataset"])
        ),
        "source_dataset_npz_public": False,
        "checkpoint_weights_used": False,
        "git_source_base_commit": command_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT.parent
        ),
        "git_worktree_dirty_at_run": bool(
            command_output(["git", "status", "--porcelain"], cwd=ROOT.parent)
        ),
    }
    environment = runtime_environment(requested)
    dashboard = {
        "experiment": experiment["name"],
        "scientific_status": "COST_SCHEMA_COMPLETE_CROSS_ARCHITECTURE_SUPERIORITY_LOCKED",
        "total_budget": int(experiment["total_budget"]),
        "grid": "8x16x16",
        "worker_repeats": repeats,
        "method_count": len(methods),
        "compute_contract": experiment["compute_contract"],
        "profiles": profiles,
        "fno_error_compute_checkpoints": checkpoints,
        "fno_time_to_target": targets,
        "readiness": readiness,
        "cross_architecture_superiority_gate_pass": False,
        "environment": environment,
        "provenance": provenance,
    }
    (output_dir / "v3e_compute_dashboard.json").write_text(
        json.dumps(dashboard, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    report = {
        "status": "completed_cross_architecture_compute_accounting_v1",
        "scientific_status": dashboard["scientific_status"],
        "environment": environment,
        "protocol": {
            "fresh_process_per_model_repeat": True,
            "identical_K6_inputs_and_grid": True,
            "synchronized_device_timing": True,
            "inference_batch_size": int(experiment["inference_batch_size"]),
            "training_batch_size": int(experiment["training_batch_size"]),
            "training_step_includes_full_physics_loss_backward_and_adamw": True,
            "random_initialization_for_cost_only": True,
            "validation_or_test_errors_used_for_cost_ranking": False,
        },
        "compute_contract": experiment["compute_contract"],
        "profiles": profiles,
        "readiness": readiness,
        "claims_boundary": [
            "The FLOP estimate is a versioned analytical estimate, not a hardware counter.",
            "Normalization, activation, pooling, softmax, indexing, most elementwise work and optimizer arithmetic are excluded.",
            "MPS memory is a sampled observed peak; short allocator spikes may be missed.",
            "Only the FNO has a matched 24-to-240 validation trajectory, so no cross-architecture error-compute superiority is claimed.",
            "The zero-initialized adapter is included as a cost envelope despite its prior negative development result.",
            "The dataset is an inspected small linear synthetic development set; real or group geometry remains required.",
        ],
        "provenance": provenance,
    }
    (output_dir / "v3e_compute_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_v3e_checksums(output_dir, OUTPUT_FILES)
    print(
        json.dumps(
            {
                "status": dashboard["scientific_status"],
                "methods": methods,
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    experiment = read_json(args.config)
    if args.worker_model is not None:
        run_worker(args, experiment)
    else:
        run_driver(args, experiment)


if __name__ == "__main__":
    main()
