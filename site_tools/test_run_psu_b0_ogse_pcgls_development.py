from __future__ import annotations

from site_tools.run_psu_b0_ogse_pcgls_development import (
    greedy_oracle_expert_bank,
)


def test_greedy_bank_keeps_baseline_and_adds_complementary_expert() -> None:
    rows = []
    values = {
        "base": [1.0, 1.0, 1.0],
        "left": [0.5, 1.2, 1.0],
        "right": [1.1, 0.4, 0.9],
    }
    for candidate, errors in values.items():
        for index, error in enumerate(errors):
            rows.append(
                {
                    "sample_id": f"s{index}",
                    "candidate_id": candidate,
                    "field_relative_l2": error,
                }
            )
    result = greedy_oracle_expert_bank(
        sample_ids=["s0", "s1", "s2"],
        candidate_rows=rows,
        baseline_candidate_id="base",
        bank_size=2,
    )
    assert result["candidate_ids"][0] == "base"
    assert result["candidate_ids"][1] in {"left", "right"}
    assert result["trajectory"][-1]["mean_oracle_field_gain_percent"] > 0.0
