from __future__ import annotations

import hashlib
import json
from pathlib import Path

from site_tools.run_psu_b0_residual_risk_fresh import build_public_summary


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    ROOT
    / "demo_t16_operator"
    / "configs"
    / "psu_b0_residual_risk_fresh_prereg_v1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_fresh_preregistration_freezes_nonzero_coverage_and_heldout_stress() -> None:
    config = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    assert config["status"] == "FROZEN_BEFORE_FRESH_AUDIT"
    frozen = config["frozen_source"]
    assert _sha256(ROOT / frozen["development_config"]) == frozen[
        "development_config_sha256"
    ]
    assert _sha256(ROOT / frozen["risk_module"]) == frozen["risk_module_sha256"]
    development = json.loads(
        (ROOT / frozen["development_config"]).read_text(encoding="utf-8")
    )
    development_families = {
        family
        for split in development["development_splits"].values()
        for family in split["families"]
    }
    heldout = set(config["fresh_splits"]["fresh_family_ood"]["families"])
    assert heldout
    assert not (heldout & development_families)
    assert set(config["fresh_splits"]["fresh_joint_ood"]["families"]) == heldout
    for split, gates in config["fresh_gates"]["per_seed"].items():
        assert split in config["fresh_splits"]
        assert float(gates["coverage_minimum"]) > 0.0
    assert config["fresh_gates"]["outside_support"][
        "candidate_coverage_required"
    ] == 0.0
    for profile in config["camera_noise_profiles"].values():
        squared = sum(float(value) ** 2 for value in profile.values())
        assert squared <= 1.0


def test_public_fresh_summary_excludes_private_evidence() -> None:
    private = {
        "status": "x",
        "evidence_scope": "y",
        "frozen_protocol_public": {"feature_count": 16},
        "dataset_public": {"splits": []},
        "aggregates": [],
        "outside_support_equivalence": [],
        "candidate_gates": {"candidate_gate_pass": False},
        "execution_public": {"wall_seconds": 1.0},
        "gates": {"all_metrics_finite": True},
        "claim_boundary": {"fresh_values_opened": True},
        "configuration_private": {"root": "/private/path"},
        "dataset_private": {
            "per_sample_metrics": [{"observation": [1, 2]}],
            "checkpoint_records": [{"sha256": "secret"}],
        },
    }
    public = build_public_summary(private)
    payload = json.dumps(public, sort_keys=True)
    assert "/private/path" not in payload
    assert '"observation":' not in payload
    assert "secret" not in payload
    assert public["public_export_policy"]["contains_per_sample_features_or_metrics"] is False
