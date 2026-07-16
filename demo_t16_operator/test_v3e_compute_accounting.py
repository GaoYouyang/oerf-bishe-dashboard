from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from demo_t16_operator.run_v3e_compute_accounting import (
    AnalyticForwardCounter,
    aggregate_trials,
)


ROOT = Path(__file__).resolve().parent


class TinyCounterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(3, 4, kernel_size=3, padding=1, bias=False)
        self.linear = nn.Linear(4, 5, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = self.conv(x).mean(dim=(2, 3, 4))
        return self.linear(values)


class TinyTransposeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(
            2, 4, kernel_size=2, stride=2, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


def test_compute_config_freezes_expected_methods_and_claim_boundary() -> None:
    config = json.loads(
        (ROOT / "configs" / "v3e_compute_accounting.json").read_text(encoding="utf-8")
    )
    assert config["methods"] == [
        "ridge_unet_aug",
        "ridge_fno_aug",
        "ridge_deeponet",
        "ray_set_operator",
        "zero_init_ray_set_adapter",
    ]
    assert config["worker_repeats"] >= 3
    assert "softmax" in config["compute_contract"]["excluded_from_estimate"]
    assert "not an exact hardware FLOP count" in config["compute_contract"]["claim_scope"]


def test_compute_script_keeps_cpu_probe_and_timed_model_separate() -> None:
    source = (ROOT / "run_v3e_compute_accounting.py").read_text(encoding="utf-8")
    assert "analytic_model = make_cost_model" in source
    assert "del analytic_model" in source
    assert source.count("make_cost_model(method, config, data)") >= 2


def test_analytic_counter_counts_conv_and_linear_macs_exactly() -> None:
    model = TinyCounterModel()
    counter = AnalyticForwardCounter()
    counter.attach(model)
    model(torch.zeros(2, 3, 4, 4, 4))
    counter.close()
    expected_conv = 2 * 4 * 4 * 4 * 4 * 3 * 3 * 3 * 3
    expected_linear = 2 * 5 * 4
    assert counter.dense_macs == expected_conv + expected_linear
    assert counter.spectral_complex_macs == 0
    assert counter.fft_flops_estimate == 0.0
    assert counter.summary()["forward_estimated_flops_v1"] == 2 * (
        expected_conv + expected_linear
    )


def test_transposed_convolution_counts_input_scatter_macs() -> None:
    model = TinyTransposeModel()
    counter = AnalyticForwardCounter()
    counter.attach(model)
    output = model(torch.zeros(1, 2, 2, 2, 2))
    counter.close()
    assert output.shape == (1, 4, 4, 4, 4)
    # Each input scalar is scattered through Cout * kernel_volume weights.
    expected = 1 * 2 * (2 * 2 * 2) * 4 * (2 * 2 * 2)
    assert counter.dense_macs == expected
    assert counter.summary()["forward_estimated_flops_v1"] == 2 * expected


def test_aggregate_trials_uses_median_timing_and_max_memory() -> None:
    base = {
        "method": "ridge_fno_aug",
        "label": "FNO",
        "device": "mps",
        "dtype": "torch.float32",
        "grid": "8x16x16",
        "input_channels": 42,
        "inference_batch_size": 1,
        "training_batch_size": 12,
        "total_parameters": 10,
        "trainable_parameters": 10,
        "frozen_parameters": 0,
        "parameter_bytes": 40,
        "trainable_parameter_bytes": 40,
        "buffer_bytes": 0,
        "training_state_bytes_floor": 160,
        "forward_dense_real_macs": 100,
        "forward_spectral_complex_macs": 10,
        "forward_fft_real_flops_estimate": 20.0,
        "forward_estimated_flops_v1": 280.0,
        "inference_p50_ms": 2.0,
        "inference_p90_ms": 3.0,
        "training_step_p50_ms": 24.0,
        "training_step_p90_ms": 30.0,
        "training_step_p50_ms_per_sample": 2.0,
        "training_peak_mps_allocated_bytes_observed": 100,
        "training_peak_mps_driver_bytes_observed": 200,
        "training_mps_increment_bytes_observed": 50,
        "process_peak_rss_bytes": 500,
    }
    rows = []
    for shift in (-0.2, 0.0, 0.4):
        row = dict(base)
        for key in (
            "inference_p50_ms",
            "inference_p90_ms",
            "training_step_p50_ms",
            "training_step_p90_ms",
            "training_step_p50_ms_per_sample",
        ):
            row[key] = float(row[key]) + shift
        row["training_peak_mps_allocated_bytes_observed"] += int(shift * 10)
        rows.append(row)
    result = aggregate_trials(rows)[0]
    assert result["worker_repeats"] == 3
    assert result["inference_p50_ms"] == 2.0
    assert result["training_peak_mps_allocated_bytes_observed"] == 104
    assert result["training_samples_per_second"] == 500.0
