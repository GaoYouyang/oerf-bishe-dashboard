from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

import demo_t16_operator.run_n2_pvgr_n4_1_opaque_artifact_recovery as recovery


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "demo_t16_operator/configs/" "n2_pvgr_n4_1_opaque_artifact_recovery_v1.json"
)


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_recovery_contract_matches_opaque_inventory_without_parsing_payloads() -> None:
    config = _config()
    recovery._validate_contract(config)
    paths = recovery._checkpoint_paths(config)
    assert len(paths) == 105
    assert sum(path.name == "H2048.json" for path in paths) == 9
    assert not config["disclosure"][
        "checkpoint_payloads_parsed_by_recovery_author_before_protocol_freeze"
    ]
    assert not config["frozen_recovery_actions"]["rerun_any_numerical_level"]


def test_checkpoint_merkle_root_is_path_and_byte_sensitive(tmp_path: Path) -> None:
    first = tmp_path / "levels/a/H256.json"
    second = tmp_path / "levels/b/H512.json"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")
    paths = [first, second]
    root = recovery._checkpoint_merkle_root(paths, tmp_path)
    assert root == recovery._checkpoint_merkle_root(list(reversed(paths)), tmp_path)
    second.write_text("changed", encoding="utf-8")
    assert root != recovery._checkpoint_merkle_root(paths, tmp_path)


def _decision(role: str) -> dict:
    return {
        "role": role,
        "h1024_output_contraction": {"contraction_ratio": 0.25},
        "h1024_matched_residual_contraction": {"contraction_ratio": 0.30},
        "h1024_all_gates_pass": role == "matched_control",
        "requires_h2048_escalation": role == "n3_failure",
        "final_cellwise_reference_authorized": True,
    }


def test_fixed_plot_renders_nonblank_png_with_explicit_category_keys(
    tmp_path: Path,
) -> None:
    path = tmp_path / "recovery.png"
    decisions = [_decision("n3_failure"), _decision("matched_control")]
    costs = [
        {"step_count": step, "wall_seconds": step / 256}
        for step in (256, 512, 1024, 2048)
    ]
    recovery._fixed_plot(path, decisions, costs)
    with Image.open(path) as image:
        assert image.width >= 1200
        assert image.height >= 350
        assert image.getbbox() is not None


def test_committed_recovery_attestation_validates_when_present() -> None:
    config = _config()
    recovery._validate_contract(config)
    attestation = ROOT / config["recovery_attestation"]
    if not attestation.exists():
        assert not (ROOT / config["formal_output"]).exists()
        pytest.skip(
            "recovery attestation is created after the recovery protocol commit"
        )
    validated = recovery._validate_attestation(config, CONFIG)
    assert validated["formal_output_absent_at_creation"] is True
    assert validated["checkpoint_payloads_parsed"] is False
