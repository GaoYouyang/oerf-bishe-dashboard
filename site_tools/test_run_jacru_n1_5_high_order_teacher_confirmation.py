from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from site_tools import run_jacru_n1_5_high_order_teacher_confirmation as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "demo_t16_operator/configs/jacru_n1_5_high_order_teacher_confirmation_v1.json"


def _payloads():
    config = json.loads(CONFIG.read_text())
    source = json.loads((ROOT / config["source_t0_config"]).read_text())
    postopen = json.loads(
        (ROOT / config["source_postopen_results"] / "summary.json").read_text()
    )
    return config, source, postopen


def test_seed_derivation_matches_frozen_declaration() -> None:
    config, _, _ = _payloads()
    assert runner._derive_confirmation_seeds(config) == config["confirmation"]["base_seeds"]


def test_config_accepts_frozen_confirmation_contract() -> None:
    config, source, postopen = _payloads()
    runner._validate_config(config, source, postopen, None)


def test_config_rejects_candidate_drift() -> None:
    config, source, postopen = _payloads()
    changed = copy.deepcopy(config)
    changed["selected_candidate"]["beta"] = 1.0
    with pytest.raises(ValueError, match="drifted"):
        runner._validate_config(changed, source, postopen, None)


def test_config_rejects_prior_seed_overlap() -> None:
    config, source, postopen = _payloads()
    changed = copy.deepcopy(config)
    changed["confirmation"]["base_seeds"][0] = source["splits"]["development"]["base_seeds"][0]
    with pytest.raises(ValueError, match="derivation|overlap"):
        runner._validate_config(changed, source, postopen, None)


def test_seed_limited_smoke_writes_immutable_package(tmp_path: Path) -> None:
    output = tmp_path / "confirmation-smoke"
    old_argv = runner.sys.argv
    runner.sys.argv = [
        "run_jacru_n1_5_high_order_teacher_confirmation.py",
        "--config",
        str(CONFIG),
        "--output-dir",
        str(output),
        "--seed-limit",
        "1",
    ]
    try:
        assert runner.main() == 0
    finally:
        runner.sys.argv = old_argv
    expected = {
        "README.md",
        "case_manifest.csv",
        "case_metrics.csv",
        "checksums.sha256",
        "diagnostic.pdf",
        "diagnostic.png",
        "geometry_cluster_metrics.csv",
        "provenance.json",
        "summary.json",
    }
    assert {path.name for path in output.iterdir()} == expected
    summary = json.loads((output / "summary.json").read_text())
    assert summary["candidate_was_frozen_before_open"] is True
    assert summary["candidate_changed_after_open"] is False
    assert summary["confirmation_geometry_cluster_count"] == 1
    runner.sys.argv = [
        "run_jacru_n1_5_high_order_teacher_confirmation.py",
        "--config",
        str(CONFIG),
        "--output-dir",
        str(output),
        "--seed-limit",
        "1",
    ]
    try:
        with pytest.raises(FileExistsError, match="immutable"):
            runner.main()
    finally:
        runner.sys.argv = old_argv
