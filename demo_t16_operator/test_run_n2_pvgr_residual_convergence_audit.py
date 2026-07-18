from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

try:
    from .run_n2_pvgr_residual_convergence_audit import DEFAULT_CONFIG, ROOT, run
except ImportError:
    from run_n2_pvgr_residual_convergence_audit import DEFAULT_CONFIG, ROOT, run


def _reduced_contract(tmp_path: Path) -> tuple[Path, Path]:
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    source_path = ROOT / config["source_config"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["population_count"] = 8
    source["dimensionless_stress_scale_multipliers"] = [1]
    source["development_cases"] = source["development_cases"][:1]

    reduced_source = tmp_path / "source.json"
    reduced_source.write_text(
        json.dumps(source, indent=2) + "\n",
        encoding="utf-8",
    )
    config.update(
        {
            "source_config": str(reduced_source),
            "step_counts": [8, 16, 32],
            "accepted_execution_step_count": 16,
            "reference_step_count": 32,
            "maximum_residual_relative_l2": 0.5,
            "maximum_residual_variance_ratio_deviation": 0.5,
            "hard_conclusion": "TEST_RESIDUAL_TARGET_ACCEPTED",
        }
    )
    reduced_config = tmp_path / "config.json"
    reduced_config.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    return reduced_config, reduced_source


def test_reduced_audit_is_replayable_and_hash_complete(tmp_path: Path) -> None:
    config_path, _ = _reduced_contract(tmp_path)
    output = tmp_path / "result"
    result = run(config_path, output)

    assert result["machine_decision"] == "TEST_RESIDUAL_TARGET_ACCEPTED"
    assert result["case_scale_count"] == 1
    assert result["accepted_execution_step_screen_count"] == 1
    assert result["accepted_execution_step_screen_required_count"] == 1
    assert result["reserved_audit_families_not_opened"] == [
        "oblique_compression_sheet",
        "shock_expansion_pair",
    ]
    convergence = result["rows"][0]["convergence"]
    reference = next(item for item in convergence if item["step_count"] == 32)
    assert reference["high_relative_l2_to_reference"] == pytest.approx(0.0)
    assert reference["medium_relative_l2_to_reference"] == pytest.approx(0.0)
    assert reference["residual_relative_l2_to_reference"] == pytest.approx(0.0)
    assert reference["residual_variance_ratio_to_reference"] == pytest.approx(1.0)

    expected = {
        "config_snapshot.json",
        "manifest.json",
        "metrics.csv",
        "n2_pvgr_residual_convergence_audit.png",
        "result.json",
        "summary.md",
    }
    assert expected == {path.name for path in output.iterdir()}
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for name, expected_hash in manifest["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected_hash
    assert b"\r\n" not in (output / "metrics.csv").read_bytes()


def test_audit_refuses_reserved_family_before_execution(tmp_path: Path) -> None:
    config_path, source_path = _reduced_contract(tmp_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["development_cases"][0]["phantom_family"] = "shock_expansion_pair"
    source_path.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="reserved audit family"):
        run(config_path, tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()
