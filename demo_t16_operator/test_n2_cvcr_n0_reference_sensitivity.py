from __future__ import annotations

from demo_t16_operator.run_n2_cvcr_n0_reference_sensitivity import (
    DESCRIPTIVE_LAST_STEP_THRESHOLD,
    ORDER_LADDER,
)


def test_reference_ladder_is_strictly_increasing_and_ends_at_4096() -> None:
    counts = [radial * angular for radial, angular in ORDER_LADDER]
    assert counts == sorted(set(counts))
    assert counts[0] == 1024
    assert counts[-1] == 4096


def test_postopen_threshold_is_descriptive_and_stricter_than_original_gate() -> None:
    assert DESCRIPTIVE_LAST_STEP_THRESHOLD == 0.001
    assert DESCRIPTIVE_LAST_STEP_THRESHOLD < 0.003
