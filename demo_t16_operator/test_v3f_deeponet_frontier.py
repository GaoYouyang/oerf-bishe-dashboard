from __future__ import annotations

import json
from pathlib import Path

from demo_t16_operator.run_v3f_deeponet_frontier import (
    build_dominance_diagnostic,
    select_global_learning_rate,
    selected_architecture,
)


ROOT = Path(__file__).resolve().parent


def test_v3f_config_locks_matching_and_claim_boundaries() -> None:
    config = json.loads(
        (ROOT / "configs" / "v3f_deeponet_frontier.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["fixed_epoch_checkpoints"] == [60, 120, 180, 240]
    assert config["base_learning_rate_grid"] == [0.0005, 0.001, 0.002, 0.004]
    assert config["validation_plateau_rule"]["test_and_q_audit_locked_out_of_selection"]
    assert config["development_gate"]["require_every_seed_mean_positive"]


def test_learning_rate_selection_uses_global_seed_mean() -> None:
    rows = [
        {"learning_rate": 0.001, "best_validation_rel_l2": 0.20},
        {"learning_rate": 0.001, "best_validation_rel_l2": 0.22},
        {"learning_rate": 0.002, "best_validation_rel_l2": 0.18},
        {"learning_rate": 0.002, "best_validation_rel_l2": 0.20},
    ]
    selected, annotated = select_global_learning_rate(rows, expected_seed_count=2)
    assert selected == 0.002
    assert sum(bool(row["global_validation_champion"]) for row in annotated) == 2


def test_architecture_selection_ignores_post_selection_dev2() -> None:
    deep = [
        {
            "strategy": "deep_champion",
            "cumulative_epochs": 240,
            "mean_validation_rel_l2": 0.09,
        }
    ]
    fno = [
        {
            "strategy": "fno_champion",
            "cumulative_epochs": 240,
            "mean_validation_rel_l2": 0.10,
        }
    ]
    winner, values = selected_architecture(
        deep, "deep_champion", fno, "fno_champion", 240
    )
    assert winner == "deeponet"
    assert values == {"deeponet": 0.09, "fno": 0.10}
