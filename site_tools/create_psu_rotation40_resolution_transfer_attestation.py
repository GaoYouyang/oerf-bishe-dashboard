#!/usr/bin/env python3
"""Create the one-time pre-result attestation for the rotation-40 resolution audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(ROOT_BOOTSTRAP))

from site_tools.run_psu_rotation40_resolution_transfer import (
    ATTESTATION_SCHEMA,
    ROOT,
    _attested_external_inputs,
    validate_config,
)


DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator/configs/psu_rotation40_resolution_transfer_preregistered_v1.json"
)


def _git(*args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    if binary:
        return result.stdout
    return result.stdout.decode("utf-8").strip()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def build_attestation(
    config_path: Path,
    *,
    protocol_commit: str,
    output_path: Path,
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", protocol_commit):
        raise ValueError("protocol_commit must be a full lowercase Git SHA")
    if str(_git("rev-parse", f"{protocol_commit}^{{commit}}")) != protocol_commit:
        raise ValueError("protocol_commit did not resolve to itself")
    if str(_git("rev-parse", "HEAD")) != protocol_commit:
        raise RuntimeError("attestation must be created at the protocol commit")
    config_path = config_path.resolve()
    config_relative = _relative(config_path)
    if config_relative != (
        "demo_t16_operator/configs/psu_rotation40_resolution_transfer_preregistered_v1.json"
    ):
        raise ValueError("unexpected resolution-transfer config path")
    current_config = config_path.read_bytes()
    frozen_config = _git("show", f"{protocol_commit}:{config_relative}", binary=True)
    if current_config != frozen_config:
        raise RuntimeError("current config differs from the protocol commit")
    config = json.loads(current_config.decode("utf-8"))
    validate_config(config)
    expected_output = (ROOT / config["pre_registration_attestation"]).resolve()
    if output_path.resolve() != expected_output:
        raise ValueError("attestation output path does not match the config")
    if output_path.exists():
        raise FileExistsError("resolution-transfer attestation already exists")
    formal_output = (ROOT / config["formal_output"]).resolve()
    if formal_output.exists():
        raise FileExistsError("formal result exists before attestation")

    attested_files: dict[str, dict[str, Any]] = {}
    for key, relative_value in config["attested_files"].items():
        relative = str(relative_value)
        current_path = (ROOT / relative).resolve()
        if ROOT not in current_path.parents or not current_path.is_file():
            raise ValueError(f"attested file is absent or escapes repository: {key}")
        current = current_path.read_bytes()
        frozen = _git("show", f"{protocol_commit}:{relative}", binary=True)
        if current != frozen:
            raise RuntimeError(f"attested file differs from protocol commit: {key}")
        attested_files[str(key)] = {
            "path": relative,
            "sha256": _sha256_bytes(frozen),
            "bytes": len(frozen),
        }

    attestation = {
        "schema_version": ATTESTATION_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_commit": protocol_commit,
        "repository_head_at_creation": str(_git("rev-parse", "HEAD")),
        "formal_results_absent_at_creation": True,
        "formal_output": config["formal_output"],
        "config_sha256": _sha256_bytes(current_config),
        "attested_files": attested_files,
        "external_input_bindings": _attested_external_inputs(config),
        "held_out_design": {
            "support_camera_ids": [2, 3, 4],
            "support_rotation_degrees": [0, 50, 90],
            "scored_camera_ids": [2, 3, 4],
            "scored_rotation_degrees": 40,
            "held_out_unit": "ROTATION_RUN_NOT_CAMERA",
            "independent_rotation_block_count": 1,
        },
        "claim_boundary": (
            "The protocol, runner, tests, exact support-field provenance, rotation-40 "
            "payload and geometry hashes, same-camera rotation holdout identity, numerical "
            "screen, complexity disclosure, and public-output firewall were committed "
            "before the formal 16-cubed rotation-40 score existed."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(attestation, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return attestation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol-commit", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output or ROOT / config["pre_registration_attestation"]
    result = build_attestation(
        args.config.resolve(),
        protocol_commit=str(args.protocol_commit),
        output_path=output.resolve(),
    )
    print(json.dumps(result, indent=2, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
