#!/usr/bin/env python3
"""Attest the opaque N4.1 checkpoint inventory without parsing payloads."""

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
    ROOT / "demo_t16_operator/configs/" "n2_pvgr_n4_1_opaque_artifact_recovery_v1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git(*args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True)
    return result.stdout if binary else result.stdout.decode("utf-8").strip()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _checkpoint_merkle_root(paths: list[Path], work: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(work).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


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
    relative_config = _relative(config_path)
    current_config = config_path.read_bytes()
    frozen_config = _git("show", f"{protocol_commit}:{relative_config}", binary=True)
    if current_config != frozen_config:
        raise RuntimeError("current recovery config differs from protocol commit")
    config = json.loads(current_config.decode("utf-8"))
    expected_output = (ROOT / config["recovery_attestation"]).resolve()
    if output_path.resolve() != expected_output:
        raise ValueError("output must match N4.1 recovery_attestation")
    if output_path.exists():
        raise FileExistsError("N4.1 recovery attestation already exists")
    formal_output = (ROOT / config["formal_output"]).resolve()
    if formal_output.exists():
        raise FileExistsError("N4.1 formal output exists before recovery attestation")
    work = (ROOT / config["formal_work_output"]).resolve()
    checkpoints = sorted(work.glob(config["checkpoint_glob"]))
    expected_count = int(config["expected_opaque_checkpoint_count"])
    expected_h2048 = int(config["expected_opaque_h2048_checkpoint_count"])
    if len(checkpoints) != expected_count:
        raise ValueError("opaque N4.1 checkpoint count drifted")
    h2048_count = sum(path.name == "H2048.json" for path in checkpoints)
    if h2048_count != expected_h2048:
        raise ValueError("opaque N4.1 H2048 checkpoint count drifted")
    log = (ROOT / config["execution_log"]).resolve()
    if _sha256(log) != config["execution_log_sha256"]:
        raise ValueError("N4.1 execution log hash drifted")

    files: dict[str, dict[str, Any]] = {}
    for key, relative in config["attested_files"].items():
        path = (ROOT / relative).resolve()
        current = path.read_bytes()
        frozen = _git("show", f"{protocol_commit}:{relative}", binary=True)
        if current != frozen:
            raise RuntimeError(
                f"recovery file differs from protocol commit: {relative}"
            )
        files[str(key)] = {
            "path": relative,
            "sha256": _sha256_bytes(frozen),
            "bytes": len(frozen),
        }

    attestation = {
        "schema": "n2-pvgr-n4-1-opaque-artifact-recovery-attestation-1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "recovery_protocol_commit": protocol_commit,
        "repository_head_at_creation": str(_git("rev-parse", "HEAD")),
        "formal_output_absent_at_creation": True,
        "checkpoint_payloads_parsed": False,
        "opaque_checkpoint_count": len(checkpoints),
        "opaque_h2048_checkpoint_count": h2048_count,
        "opaque_checkpoint_merkle_root": _checkpoint_merkle_root(checkpoints, work),
        "execution_log_sha256": _sha256(log),
        "recovery_config_sha256": _sha256_bytes(current_config),
        "attested_files": files,
        "claim_boundary": (
            "Inventory paths and bytes were hashed without JSON parsing after aggregate "
            "console statuses were visible; no numerical or decision change is allowed."
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
    output = args.output or ROOT / config["recovery_attestation"]
    result = build_attestation(
        args.config.resolve(),
        protocol_commit=str(args.protocol_commit),
        output_path=output.resolve(),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
