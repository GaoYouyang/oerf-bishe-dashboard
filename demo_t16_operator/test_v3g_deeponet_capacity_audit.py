from __future__ import annotations

import json
from pathlib import Path

from demo_t16_operator.own_algorithm_models import GridDeepONetResidual
from demo_t16_operator.run_v3g_deeponet_capacity_audit import (
    select_screen_champion,
)


ROOT = Path(__file__).resolve().parent


def model_parameters(rank: int, pool_shape: tuple[int, int, int]) -> int:
    model = GridDeepONetResidual(
        view_channel_start=3,
        view_count=9,
        mask_channel_start=12,
        angle_sin_channel_start=21,
        angle_cos_channel_start=30,
        coordinate_channels=(39, 40, 41),
        rank=rank,
        pool_shape=pool_shape,
    )
    return sum(parameter.numel() for parameter in model.parameters())


def test_v3g_config_pre_registers_bounded_factorial_screen() -> None:
    config = json.loads(
        (ROOT / "configs" / "v3g_deeponet_capacity_audit.json").read_text(
            encoding="utf-8"
        )
    )
    screened = [row for row in config["variants"] if row["screen"]]
    excluded = [row for row in config["variants"] if not row["screen"]]
    assert len(screened) == 8
    assert len(excluded) == 2
    assert config["screen_learning_rates"] == [0.001, 0.002, 0.004]
    assert config["training_seeds"] == [20260901, 20260902, 20260903]
    assert config["base_epochs"] == 24
    assert config["parameter_cap_ratio"] == 1.5
    assert config["validation_plateau_rule"][
        "test_and_q_audit_locked_out_of_selection"
    ]


def test_v3g_parameter_cap_includes_screen_and_excludes_high_resolution() -> None:
    reference = model_parameters(48, (4, 4, 4))
    cap = int(reference * 1.5)
    assert reference == 49313
    assert model_parameters(96, (4, 4, 4)) <= cap
    assert model_parameters(48, (8, 4, 4)) > cap
    assert model_parameters(48, (4, 8, 8)) > cap


def test_screen_selection_uses_global_seed_mean_then_capacity() -> None:
    rows = [
        {
            "variant_id": "large",
            "learning_rate": 0.001,
            "parameter_count": 20,
            "best_validation_rel_l2": 0.10,
            "endpoint_validation_rel_l2": 0.11,
            "train_seconds": 1.0,
        },
        {
            "variant_id": "large",
            "learning_rate": 0.001,
            "parameter_count": 20,
            "best_validation_rel_l2": 0.20,
            "endpoint_validation_rel_l2": 0.21,
            "train_seconds": 1.0,
        },
        {
            "variant_id": "small",
            "learning_rate": 0.002,
            "parameter_count": 10,
            "best_validation_rel_l2": 0.14,
            "endpoint_validation_rel_l2": 0.15,
            "train_seconds": 1.0,
        },
        {
            "variant_id": "small",
            "learning_rate": 0.002,
            "parameter_count": 10,
            "best_validation_rel_l2": 0.16,
            "endpoint_validation_rel_l2": 0.17,
            "train_seconds": 1.0,
        },
    ]
    variant, learning_rate, annotated, summaries = select_screen_champion(
        rows, expected_seed_count=2
    )
    assert variant == "small"
    assert learning_rate == 0.002
    assert sum(bool(row["global_validation_champion"]) for row in annotated) == 2
    assert sum(bool(row["global_validation_champion"]) for row in summaries) == 1

