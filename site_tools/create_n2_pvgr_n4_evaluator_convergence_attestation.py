#!/usr/bin/env python3
"""Create the one-time attestation for the N2-PVGR N4 evaluator audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "demo_t16_operator/configs/"
    "n2_pvgr_n4_evaluator_convergence_preregistered_v1.json"
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git(*args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True)
    return result.stdout if binary else result.stdout.decode("utf-8").strip()


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
    config_path = config_path.resolve()
    config_relative = _relative(config_path)
    current_config = config_path.read_bytes()
    frozen_config = _git("show", f"{protocol_commit}:{config_relative}", binary=True)
    if current_config != frozen_config:
        raise RuntimeError("current N4 config differs from protocol commit")
    config = json.loads(current_config.decode("utf-8"))
    expected_output = (ROOT / config["pre_registration_attestation"]).resolve()
    if output_path.resolve() != expected_output:
        raise ValueError("output path must match N4 pre_registration_attestation")
    if output_path.exists():
        raise FileExistsError("N4 preregistration attestation already exists")
    formal_output = (ROOT / config["formal_output"]).resolve()
    formal_work = (ROOT / config["formal_work_output"]).resolve()
    if formal_output.exists():
        raise FileExistsError("N4 formal output exists before attestation")
    if formal_work.exists():
        raise FileExistsError("N4 formal checkpoints exist before attestation")

    files: dict[str, dict[str, Any]] = {}
    for key, relative in config["attested_files"].items():
        path = (ROOT / relative).resolve()
        if ROOT not in path.parents:
            raise ValueError(f"N4 attested file escapes repository: {relative}")
        current = path.read_bytes()
        frozen = _git("show", f"{protocol_commit}:{relative}", binary=True)
        if current != frozen:
            raise RuntimeError(f"N4 file differs from protocol commit: {relative}")
        files[str(key)] = {
            "path": relative,
            "sha256": _sha256_bytes(frozen),
            "bytes": len(frozen),
        }

    attestation = {
        "schema": "n2-pvgr-n4-evaluator-convergence-attestation-1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_commit": protocol_commit,
        "repository_head_at_creation": str(_git("rev-parse", "HEAD")),
        "formal_results_absent_at_creation": True,
        "formal_work_output_absent_at_creation": True,
        "formal_output": config["formal_output"],
        "formal_work_output": config["formal_work_output"],
        "config_sha256": _sha256_bytes(current_config),
        "attested_files": files,
        "claim_boundary": (
            "This attests only that N4 selection, thresholds, code, plots, "
            "validation and stopping rules were committed before any N4 checkpoint."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(attestation, indent=2, ensure_ascii=False) + "\n",
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
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
