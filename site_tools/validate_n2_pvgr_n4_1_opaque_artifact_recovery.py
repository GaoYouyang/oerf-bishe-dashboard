#!/usr/bin/env python3
"""Validate the N4.1 opaque-checkpoint artifact recovery and rendered figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import demo_t16_operator.run_n2_pvgr_n4_1_opaque_artifact_recovery as recovery  # noqa: E402
import site_tools.validate_n2_pvgr_n4_1_evaluator_convergence as result_validator  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "demo_t16_operator/configs/" "n2_pvgr_n4_1_opaque_artifact_recovery_v1.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(config_path: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    recovery._validate_contract(config)
    attestation = recovery._validate_attestation(config, config_path)
    output = ROOT / config["formal_output"]
    amendment_config = ROOT / config["original_amendment_config"]
    base_report = result_validator.validate(amendment_config, output)
    result = _read_json(output / "result.json")
    manifest = _read_json(output / "manifest.json")
    attestation_path = ROOT / config["recovery_attestation"]
    expected_metadata = recovery._recovery_metadata(
        config, attestation, attestation_path
    )
    if result.get("artifact_recovery") != expected_metadata:
        raise ValueError("N4.1 result recovery metadata drifted")
    if manifest.get("artifact_recovery") != expected_metadata:
        raise ValueError("N4.1 manifest recovery metadata drifted")
    if result["artifact_recovery"]["numerical_levels_rerun"] is not False:
        raise ValueError("N4.1 recovery claims a numerical rerun")
    if (
        result["artifact_recovery"]["scientific_or_machine_decision_change"]
        is not False
    ):
        raise ValueError("N4.1 recovery changed a scientific decision")
    if result["artifact_recovery"]["plot_change"] != (
        "bar_x_dict_to_list_of_keys_only"
    ):
        raise ValueError("N4.1 recovery plot-change declaration drifted")
    for key, relative in config["attested_files"].items():
        manifest_key = f"recovery_attested_{key}"
        entry = manifest["files"].get(manifest_key)
        path = ROOT / relative
        if not entry or entry["path"] != relative:
            raise ValueError(f"N4.1 recovery manifest entry missing: {key}")
        if entry["sha256"] != recovery._sha256(path):
            raise ValueError(f"N4.1 recovery manifest hash drifted: {key}")
    recovery_entry = manifest["files"].get("recovery_attestation")
    if not recovery_entry or recovery_entry["sha256"] != recovery._sha256(
        attestation_path
    ):
        raise ValueError("N4.1 recovery attestation manifest entry drifted")
    summary = (output / "summary.md").read_text(encoding="utf-8")
    if "Artifact recovery: attested opaque-checkpoint reuse" not in summary:
        raise ValueError("N4.1 summary omitted recovery disclosure")
    figure_path = output / result["figure"]
    with Image.open(figure_path) as image:
        image.verify()
    with Image.open(figure_path).convert("RGB") as image:
        if image.width < 1200 or image.height < 350:
            raise ValueError("N4.1 recovery figure dimensions are unexpectedly small")
        extrema = ImageStat.Stat(image).extrema
        if not any(high - low > 40 for low, high in extrema):
            raise ValueError("N4.1 recovery figure appears blank")
    report = {
        "schema": "n2-pvgr-n4-1-opaque-artifact-recovery-validation-1.0",
        "valid": True,
        "machine_decision": result["machine_decision"],
        "opaque_checkpoint_count": attestation["opaque_checkpoint_count"],
        "opaque_h2048_checkpoint_count": attestation["opaque_h2048_checkpoint_count"],
        "opaque_checkpoint_merkle_root": attestation["opaque_checkpoint_merkle_root"],
        "numerical_levels_rerun": False,
        "scientific_or_machine_decision_change": False,
        "figure_verified_nonblank": True,
        "base_result_validation": base_report["valid"],
    }
    (output / "recovery_validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(json.dumps(validate(args.config.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
