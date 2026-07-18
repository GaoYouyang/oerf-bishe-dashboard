from __future__ import annotations

import json
from pathlib import Path

import pytest

from demo_t16_operator.run_n2_pvgr_n5_d4c_msra_development import (
    ROOT,
    _validate_config,
    run,
)


CONFIG = (
    ROOT
    / "demo_t16_operator/configs/"
    "n2_pvgr_n5_d4c_msra_development_preregistered_v1.json"
)


def _small_config(tmp_path: Path) -> Path:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["trial_count"] = 4
    config["probe_counts"] = [1, 2, 4]
    config["gamma_threshold_grid"] = [0.5, 2.0]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def test_config_forbids_claim_authorization() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claim_authorizations"]["field_derivative_interface"] = True
    with pytest.raises(ValueError, match="pre-authorize"):
        _validate_config(config)


def test_development_run_reports_blind_spots_without_selecting_threshold(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result"
    result = run(
        _small_config(tmp_path), output, require_committed_source=False
    )

    assert result["machine_decision"] == (
        "D4C_MSRA_DEVELOPMENT_CHARACTERIZATION_ONLY_NO_AUTHORIZATION"
    )
    assert result["threshold_selected"] is None
    assert result["threshold_selection_forbidden"] is True
    assert result["counts"]["trial_count"] == 4
    assert result["counts"]["scenario_count"] == 11
    assert result["headline_diagnostics"]["low_signal_traditional_reject_count"] == 4
    assert result["headline_diagnostics"]["diagnostic_support_report_only_count"] == 4
    assert result["headline_diagnostics"]["hard_branch_reject_count"] == 4
    assert result["retrospective_d4b"]["historical_decision_changed"] is False
    assert all(value is False for value in result["claim_authorizations"].values())

    for name in (
        "result.json",
        "synthetic_rows.csv",
        "threshold_probe_rows.csv",
        "n2_pvgr_n5_d4c_msra_development.png",
        "summary.md",
        "manifest.json",
    ):
        assert (output / name).is_file()


def test_development_run_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    with pytest.raises(FileExistsError):
        run(_small_config(tmp_path), output, require_committed_source=False)
