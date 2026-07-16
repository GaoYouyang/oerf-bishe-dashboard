from __future__ import annotations

from demo_t16_operator.run_v3d_fno_optimizer_audit import (
    batch_order_contract_sha256,
    build_pairwise_summary,
    build_strategy_summary,
    choose_validation_champion,
    strategy_ids,
    summarize_strategies,
)
from demo_t16_operator.train_eval import sample_weighted_mean


EXPERIMENT = {
    "base_epochs": 24,
    "max_total_epochs": 48,
    "strategies": [
        {"id": "a", "label": "A"},
        {"id": "b", "label": "B"},
    ],
    "validation_plateau_rule": {
        "metric": "prefix_best_validation_relative_l2",
        "maximum_mean_relative_improvement_pct_per_block": 0.5,
        "maximum_seed_relative_improvement_pct_per_block": 1.0,
        "required_consecutive_final_plateau_blocks": 2,
        "selection_uses_validation_only": True,
        "test_and_q_audit_locked_out_of_selection": True,
    },
}


def make_rows() -> list[dict]:
    output = []
    values = {
        "a": {24: 0.20, 36: 0.18, 48: 0.1798},
        "b": {24: 0.20, 36: 0.175, 48: 0.17},
    }
    for strategy, epochs in values.items():
        previous = None
        for epoch, value in epochs.items():
            improvement = 0.0 if previous is None else 100 * (previous - value) / previous
            for seed in (1, 2, 3):
                output.append(
                    {
                        "strategy": strategy,
                        "model_seed": seed,
                        "cumulative_epochs": epoch,
                        "selected_validation_rel_l2": value,
                        "endpoint_validation_rel_l2": value + 0.01,
                        "relative_validation_improvement_pct": improvement,
                        "retained_previous_checkpoint": improvement == 0.0,
                        "selected_checkpoint_epoch": epoch,
                        "cumulative_train_seconds": float(epoch),
                    }
                )
            previous = value
    return output


def test_strategy_ids_reject_duplicates() -> None:
    assert strategy_ids(EXPERIMENT) == ["a", "b"]


def test_validation_champion_uses_final_validation_only() -> None:
    summary, _ = summarize_strategies(make_rows(), EXPERIMENT)
    assert choose_validation_champion(summary, 48) == "b"


def test_strategy_summary_keeps_plateau_separate_from_champion() -> None:
    summary, decisions = summarize_strategies(make_rows(), EXPERIMENT)
    champion = choose_validation_champion(summary, 48)
    rows = build_strategy_summary(summary, decisions, champion, EXPERIMENT)
    lookup = {row["strategy"]: row for row in rows}
    assert lookup["b"]["validation_champion"] is True
    assert lookup["b"]["plateau_reached"] is False
    assert lookup["a"]["validation_champion"] is False


def test_pairwise_summary_uses_field_level_candidate_minus_comparator() -> None:
    clusters = []
    for strategy, errors in {"a": (0.20, 0.30), "b": (0.18, 0.27)}.items():
        for source_index, value in enumerate(errors):
            clusters.append(
                {
                    "strategy": strategy,
                    "source_index": source_index,
                    "source_split": "test_iid",
                    "model_seed_count": 3,
                    "field_rel_l2": value,
                    "audit_reprojection_rel_l2": value,
                }
            )
    experiment = {**EXPERIMENT, "bootstrap_seed": 7, "bootstrap_replicates": 100}
    row = build_pairwise_summary(clusters, "b", experiment)[0]
    assert row["candidate"] == "b"
    assert row["comparator"] == "a"
    assert row["mean_field_superiority_pct"] > 0.0
    assert row["worst_domain"] == "test_iid"


def test_batch_order_contract_is_reproducible_and_seed_sensitive() -> None:
    first = batch_order_contract_sha256(160, 12, 12, 20270001)
    assert first == batch_order_contract_sha256(160, 12, 12, 20270001)
    assert first != batch_order_contract_sha256(160, 12, 12, 20270002)
    assert first != batch_order_contract_sha256(
        160, 12, 12, 20270001, sample_indices=list(range(1, 161))
    )


def test_validation_batch_means_are_weighted_by_sample_count() -> None:
    assert sample_weighted_mean([1.0, 3.0], [12, 4]) == 1.5
