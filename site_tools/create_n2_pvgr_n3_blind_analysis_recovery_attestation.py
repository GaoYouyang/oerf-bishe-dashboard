#!/usr/bin/env python3
"""Attest the blind N3 analysis recovery without parsing checkpoint payloads."""

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
    ROOT / "demo_t16_operator/configs/n2_pvgr_n3_blind_analysis_recovery_v1.json"
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True)
    return result.stdout if binary else result.stdout.decode("utf-8").strip()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _checkpoint_merkle_root(paths: list[Path], base: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(base).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_attestation(
    config_path: Path,
    *,
    recovery_protocol_commit: str,
    output_path: Path,
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", recovery_protocol_commit):
        raise ValueError("recovery_protocol_commit must be a full lowercase Git SHA")
    resolved = str(_git("rev-parse", f"{recovery_protocol_commit}^{{commit}}"))
    if resolved != recovery_protocol_commit:
        raise ValueError("recovery protocol commit did not resolve to itself")

    config_path = config_path.resolve()
    config_relative = _relative(config_path)
    current_config = config_path.read_bytes()
    frozen_config = _git(
        "show", f"{recovery_protocol_commit}:{config_relative}", binary=True
    )
    if current_config != frozen_config:
        raise RuntimeError("current recovery config differs from protocol commit")
    config = json.loads(current_config.decode("utf-8"))

    expected_output = (ROOT / str(config["recovery_attestation"])).resolve()
    if output_path.resolve() != expected_output:
        raise ValueError("output path does not match recovery_attestation")
    if output_path.exists():
        raise FileExistsError("blind-recovery attestation already exists")
    formal_output = (ROOT / str(config["formal_output"])).resolve()
    if formal_output.exists():
        raise FileExistsError("formal output exists before blind recovery freeze")

    work = (ROOT / str(config["formal_work_output"])).resolve()
    checkpoints = list(work.glob(str(config["checkpoint_glob"])))
    expected_count = int(config["expected_opaque_checkpoint_count"])
    if len(checkpoints) != expected_count:
        raise ValueError(
            f"expected {expected_count} opaque checkpoints, found {len(checkpoints)}"
        )
    if any(not path.is_file() for path in checkpoints):
        raise ValueError("checkpoint set contains a non-file entry")

    files: dict[str, dict[str, Any]] = {}
    for key, relative in config["attested_files"].items():
        path = (ROOT / str(relative)).resolve()
        if ROOT not in path.parents:
            raise ValueError(f"attested file escapes repository: {relative}")
        current = path.read_bytes()
        frozen = _git("show", f"{recovery_protocol_commit}:{relative}", binary=True)
        if current != frozen:
            raise RuntimeError(f"current file differs from recovery commit: {relative}")
        files[str(key)] = {
            "path": str(relative),
            "sha256": _sha256_bytes(frozen),
            "bytes": len(frozen),
        }

    attestation = {
        "schema": "n2-pvgr-n3-blind-analysis-recovery-attestation-1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "recovery_protocol_commit": recovery_protocol_commit,
        "repository_head_at_creation": str(_git("rev-parse", "HEAD")),
        "formal_output_absent_at_creation": True,
        "formal_output": str(config["formal_output"]),
        "opaque_checkpoint_count": len(checkpoints),
        "opaque_checkpoint_merkle_root": _checkpoint_merkle_root(checkpoints, work),
        "checkpoint_payloads_parsed": False,
        "recovery_config_sha256": _sha256_bytes(current_config),
        "attested_files": files,
        "failure_boundary": str(config["failure_observed"]),
        "recovery_boundary": (
            "This hashes checkpoint bytes without JSON parsing and freezes only the "
            "query-ledger alias repair before any numerical checkpoint inspection."
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
    parser.add_argument("--recovery-protocol-commit", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = args.output or ROOT / str(config["recovery_attestation"])
    result = build_attestation(
        config_path,
        recovery_protocol_commit=str(args.recovery_protocol_commit),
        output_path=output.resolve(),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
