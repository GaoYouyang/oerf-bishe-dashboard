from __future__ import annotations

from pathlib import Path

import pytest

from site_tools import audit_jacru_n1_1_raw_center_postopen as audit


def _row(*, split: str, gain: float, harm: bool, exact: bool = False) -> dict:
    return {
        "candidate_id": "sensor",
        "method": "jacru_m2",
        "split": split,
        "field_gain_to_raw": gain,
        "h1_gain_to_raw": gain / 2.0,
        "field_harm_vs_raw": harm,
        "uses_truth": False,
        "uses_exact_nuisance": exact,
    }


def test_aggregate_preserves_raw_center_tail_statistics() -> None:
    rows = [
        _row(split="development", gain=0.08, harm=False),
        _row(split="development", gain=-0.12, harm=True),
    ]
    [value] = audit._aggregate(rows)
    assert value["field_gain_to_raw_mean"] == pytest.approx(-0.02)
    assert value["h1_gain_to_raw_mean"] == pytest.approx(-0.01)
    assert value["field_harm_rate_vs_raw"] == pytest.approx(0.5)
    assert value["worst_field_gain_to_raw"] == pytest.approx(-0.12)


def test_postopen_decision_requires_both_splits_and_never_authorizes() -> None:
    aggregates = audit._aggregate(
        [
            _row(split="development", gain=0.08, harm=False),
            _row(split="ood", gain=0.03, harm=False),
        ]
    )
    [passing] = audit._postopen_diagnostic_decisions(aggregates)
    assert passing["diagnostic_passed"] is True
    assert passing["may_authorize_method"] is False

    aggregates[0]["worst_field_gain_to_raw"] = -0.2
    [failing] = audit._postopen_diagnostic_decisions(aggregates)
    assert failing["diagnostic_passed"] is False
    assert failing["checks"]["development_worst_at_least_minus_5pct"] is False


def test_packet_checksum_mutation_fails_closed(tmp_path: Path) -> None:
    payload = tmp_path / "payload.txt"
    payload.write_text("frozen\n", encoding="utf-8")
    (tmp_path / "checksums.sha256").write_text(
        f"{audit._sha256(payload)}  {payload.name}\n", encoding="utf-8"
    )
    audit._verify_packet_checksums(tmp_path)
    payload.write_text("mutated\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source checksum mismatch"):
        audit._verify_packet_checksums(tmp_path)
