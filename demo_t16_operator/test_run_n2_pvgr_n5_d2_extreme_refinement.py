from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import demo_t16_operator.run_n2_pvgr_n5_d2_extreme_refinement as d2
import site_tools.validate_n2_pvgr_n5_d2_extreme_refinement as d2_validator


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "demo_t16_operator/configs/n2_pvgr_n5_d2_extreme_refinement_preregistered_v1.json"
)


def _config() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _entry(array: np.ndarray) -> dict[str, object]:
    return {
        "values": array.tolist(),
        "sha256": d2.d1._array_sha256(array),
        "l2_norm": float(np.linalg.norm(array)),
    }


def _methods(value: float, *, raw_offset: float = 0.0) -> dict[str, object]:
    base = np.full((256, 2), value, dtype=np.float64)
    return {
        method: _entry(
            base + raw_offset if method == "raw_separate_subtraction" else base
        )
        for method in d2.METHODS
    }


def _level(value: float, *, raw_offset: float = 0.0) -> dict[str, object]:
    return {
        "methods": _methods(value, raw_offset=raw_offset),
        "diagnostics": {
            "finite_ray_fraction": 1.0,
            "minimum_domain_margin": 0.2,
            "minimum_stencil_margin": 0.19,
            "maximum_direction_norm_error": 2e-16,
        },
    }


def test_contract_reuses_exact_d1_cells_and_parent_bundle() -> None:
    config = _config()
    d2._validate_contract(config)
    _, result, levels, _, cells = d2._load_parent_contract(config)
    assert result["machine_decision"] == (
        "D1_ACCUMULATION_ORDER_TOO_SMALL_TO_EXPLAIN_N4_FLOOR"
    )
    assert len(levels) == 8
    assert [cell["cell_id"] for cell in cells] == config["selected_cells"]


def test_contract_rejects_level_gate_and_method_drift() -> None:
    config = _config()
    config["new_step_counts"][1] = 16384
    with pytest.raises(ValueError, match="new levels drifted"):
        d2._validate_contract(config)

    config = _config()
    config["gates"]["maximum_adjacent_contraction_ratio"] = 0.75
    with pytest.raises(ValueError, match="gates drifted"):
        d2._validate_contract(config)

    config = _config()
    config["reference_method"] = "raw_separate_subtraction"
    with pytest.raises(ValueError, match="reference method drifted"):
        d2._validate_contract(config)

    config = _config()
    config["decision_contract"][
        "richardson_and_geometric_tail_values_are_indicators_not_proofs"
    ] = False
    with pytest.raises(ValueError, match="decision contract drifted"):
        d2._validate_contract(config)


def test_cell_metrics_recompute_tail_and_pass_all_frozen_gates() -> None:
    config = _config()
    cell = {
        "cell_id": config["selected_cells"][0],
        "pair_id": "p04",
        "role": "matched_control",
    }
    parent = {"methods": _methods(1.0004)}
    levels = {
        4096: _level(1.0001),
        8192: _level(1.000025, raw_offset=1e-9),
    }
    result = d2._cell_metrics(cell, parent, levels, config)
    independently_recomputed = d2_validator._recompute_cell(
        cell,
        parent,
        levels,
        config,
    )
    assert result["all_gates_pass"]
    assert result["adjacent_contraction_ratio"] == pytest.approx(0.25, rel=2e-4)
    assert result["observed_order"] == pytest.approx(2.0, rel=2e-4)
    assert result["fixed_second_order_richardson_correction_relative_l2"] == (
        pytest.approx(result["d4096_to_d8192_reference_relative_l2"] / 3.0)
    )
    assert independently_recomputed["gates"] == result["gates"]
    assert independently_recomputed["adjacent_contraction_ratio"] == pytest.approx(
        result["adjacent_contraction_ratio"], rel=1e-14
    )


def test_decision_requires_every_selected_cell() -> None:
    rows = [
        {"cell_id": f"cell-{index}", "all_gates_pass": True}
        for index in range(4)
    ]
    assert d2._decision(rows) == "D2_SELECTED_TAIL_RESOLVED_AT_H8192"
    rows[2]["all_gates_pass"] = False
    assert d2._decision(rows) == "D2_SELECTED_TAIL_NOT_RESOLVED_AT_H8192"


def test_manifest_hashes_staging_but_points_to_final_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(d2, "ROOT", tmp_path)
    staging = tmp_path / "formal.staging"
    output = tmp_path / "formal"
    staging.mkdir()
    source = tmp_path / "source.txt"
    source.write_text("source\n", encoding="utf-8")
    artifact = staging / "result.json"
    artifact.write_text('{"valid": true}\n', encoding="utf-8")
    manifest = d2._manifest(
        staging,
        output,
        {"source": source},
        ["result.json"],
    )
    assert manifest["files"]["result.json"]["path"] == "formal/result.json"
    assert manifest["files"]["result.json"]["sha256"] == d2._sha256(artifact)


def test_committed_attestation_validates_when_present() -> None:
    config = _config()
    attestation = ROOT / str(config["pre_registration_attestation"])
    if not attestation.exists():
        assert not (ROOT / str(config["formal_output"])).exists()
        pytest.skip("N5-D2 attestation is created after the protocol commit")
    result = d2._validate_preregistration(config, CONFIG)
    assert result["formal_results_absent_at_creation"] is True
