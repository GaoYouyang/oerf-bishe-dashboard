#!/usr/bin/env python3
"""Create the one-time attestation and frozen inputs for N5-D4."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo_t16_operator.d4_frozen_inputs import build_frozen_inputs  # noqa: E402


DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator/configs/n2_pvgr_n5_d4_tiny_field_derivative_preregistered_v1.json"
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


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _path_occupied(path: Path) -> bool:
    return os.path.lexists(path)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_temp_exclusive(temporary: Path, destination: Path) -> None:
    # A hard link publishes complete same-filesystem bytes without overwriting an
    # artifact that another process may have created after the preflight check.
    os.link(temporary, destination)
    _fsync_directory(destination.parent)
    temporary.unlink()
    _fsync_directory(destination.parent)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    temporary = _temporary_sibling(path)
    if _path_occupied(path) or _path_occupied(temporary):
        raise FileExistsError(f"refusing to replace N5-D4 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _publish_temp_exclusive(temporary, path)


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = _temporary_sibling(path)
    if _path_occupied(path) or _path_occupied(temporary):
        raise FileExistsError(f"refusing to replace N5-D4 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("xb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    _publish_temp_exclusive(temporary, path)


def build_attestation(
    config_path: Path,
    *,
    protocol_commit: str,
    output_path: Path,
    archive_path: Path,
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", protocol_commit):
        raise ValueError("protocol_commit must be a full lowercase Git SHA")
    if str(_git("rev-parse", f"{protocol_commit}^{{commit}}")) != protocol_commit:
        raise ValueError("protocol_commit did not resolve to itself")
    if str(_git("rev-parse", "HEAD")) != protocol_commit:
        raise RuntimeError("N5-D4 attestation must be created at the protocol commit")
    config_path = config_path.resolve()
    current_config = config_path.read_bytes()
    config_relative = _relative(config_path)
    frozen_config = _git("show", f"{protocol_commit}:{config_relative}", binary=True)
    if current_config != frozen_config:
        raise RuntimeError("current N5-D4 config differs from protocol commit")
    config = json.loads(current_config.decode("utf-8"))
    expected_output = (ROOT / config["pre_registration_attestation"]).resolve()
    expected_archive = (ROOT / config["frozen_input_archive"]).resolve()
    if output_path.resolve() != expected_output:
        raise ValueError("output path must match N5-D4 attestation path")
    if archive_path.resolve() != expected_archive:
        raise ValueError("archive path must match N5-D4 frozen-input path")
    artifact_paths = (
        output_path,
        archive_path,
        _temporary_sibling(output_path),
        _temporary_sibling(archive_path),
    )
    if any(_path_occupied(path) for path in artifact_paths):
        raise FileExistsError(
            "N5-D4 attestation, frozen-input archive, or staging file already exists"
        )
    formal_output = (ROOT / config["formal_output"]).resolve()
    formal_work_output = (ROOT / config["formal_work_output"]).resolve()
    if _path_occupied(formal_output) or _path_occupied(formal_work_output):
        raise FileExistsError("N5-D4 formal or work output exists before attestation")

    files: dict[str, dict[str, Any]] = {}
    for key, relative in config["attested_files"].items():
        path = (ROOT / relative).resolve()
        if ROOT not in path.parents:
            raise ValueError(f"N5-D4 attested file escapes repository: {relative}")
        current = path.read_bytes()
        frozen = _git("show", f"{protocol_commit}:{relative}", binary=True)
        if current != frozen:
            raise RuntimeError(f"N5-D4 file differs from protocol commit: {relative}")
        files[str(key)] = {
            "path": relative,
            "sha256": _sha256_bytes(frozen),
            "bytes": len(frozen),
        }

    status_paths = [str(value) for value in config["attested_files"].values()]
    dirty = str(_git("status", "--porcelain", "--", *status_paths))
    if dirty:
        raise RuntimeError("N5-D4 attested files have uncommitted changes")

    metadata, arrays = build_frozen_inputs(config)
    _write_npz_atomic(archive_path, arrays)
    archive_inventory = {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256_float64_le_c_order": _sha256_bytes(value.tobytes(order="C")),
        }
        for key, value in sorted(arrays.items())
    }
    attestation = {
        "schema": "n2-pvgr-n5-d4-tiny-field-derivative-attestation-1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_commit": protocol_commit,
        "repository_head_at_creation": str(_git("rev-parse", "HEAD")),
        "formal_results_absent_at_creation": True,
        "formal_output": config["formal_output"],
        "formal_work_output_absent_at_creation": True,
        "formal_work_output": config["formal_work_output"],
        "config_sha256": _sha256_bytes(current_config),
        "attested_files": files,
        "frozen_input_archive": config["frozen_input_archive"],
        "frozen_input_archive_sha256": _sha256(archive_path),
        "frozen_input_archive_bytes": archive_path.stat().st_size,
        "frozen_input_inventory": archive_inventory,
        "frozen_input_metadata": metadata,
        "frozen_input_metadata_sha256": _sha256_bytes(_canonical_json(metadata)),
        "claim_boundary": (
            "This binds the exact selected grid fields, 256-ray Sobol prefixes, "
            "four-ray rows, directions, cotangents, maps, h-grid, gates, ordered "
            "program-signature implementation, tests and independent validator before "
            "formal D4 output existed. A pass can authorize only a preregistered "
            "32-cell derivative expansion, not reconstruction or model claims."
        ),
    }
    _write_bytes_atomic(
        output_path,
        (json.dumps(attestation, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    return attestation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol-commit", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output or ROOT / config["pre_registration_attestation"]
    archive = args.archive or ROOT / config["frozen_input_archive"]
    result = build_attestation(
        args.config.resolve(),
        protocol_commit=str(args.protocol_commit),
        output_path=output.resolve(),
        archive_path=archive.resolve(),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
