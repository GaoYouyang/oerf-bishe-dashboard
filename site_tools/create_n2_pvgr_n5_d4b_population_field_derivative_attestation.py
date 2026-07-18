#!/usr/bin/env python3
"""Create the one-time attestation and frozen inputs for N5-D4b."""

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

from demo_t16_operator.d4b_frozen_inputs import build_frozen_inputs  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "demo_t16_operator/configs/"
    "n2_pvgr_n5_d4b_population_field_derivative_preregistered_v1.json"
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
    os.link(temporary, destination)
    _fsync_directory(destination.parent)
    temporary.unlink()
    _fsync_directory(destination.parent)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    temporary = _temporary_sibling(path)
    if _path_occupied(path) or _path_occupied(temporary):
        raise FileExistsError(f"refusing to replace N5-D4b artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _publish_temp_exclusive(temporary, path)


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = _temporary_sibling(path)
    if _path_occupied(path) or _path_occupied(temporary):
        raise FileExistsError(f"refusing to replace N5-D4b artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("xb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    _publish_temp_exclusive(temporary, path)


def _bundle_paths(config: dict[str, Any]) -> tuple[Path, Path, Path, Path, Path]:
    bundle = (ROOT / config["pre_registration_bundle"]).resolve()
    attestation = (ROOT / config["pre_registration_attestation"]).resolve()
    archive = (ROOT / config["frozen_input_archive"]).resolve()
    ready = (ROOT / config["pre_registration_ready_marker"]).resolve()
    if {attestation.parent, archive.parent, ready.parent} != {bundle}:
        raise ValueError("N5-D4b preregistration artifacts must share one bundle")
    return bundle, _temporary_sibling(bundle), attestation, archive, ready


def _atomic_rename_bundle(staging: Path, bundle: Path) -> None:
    if _path_occupied(bundle):
        raise FileExistsError(f"refusing to replace N5-D4b bundle: {bundle}")
    _fsync_directory(staging)
    os.rename(staging, bundle)
    _fsync_directory(bundle.parent)


def _verify_complete_staging(
    config_path: Path, config: dict[str, Any], staging: Path
) -> dict[str, Any]:
    _, _, final_attestation, final_archive, final_ready = _bundle_paths(config)
    staged_attestation = staging / final_attestation.name
    staged_archive = staging / final_archive.name
    staged_ready = staging / final_ready.name
    expected_names = {
        final_attestation.name,
        final_archive.name,
        final_ready.name,
    }
    if (
        not staging.is_dir()
        or {path.name for path in staging.iterdir()} != expected_names
    ):
        raise ValueError("N5-D4b staging bundle has missing or unverified extra files")
    for path in (staged_attestation, staged_archive, staged_ready):
        if not path.is_file():
            raise ValueError(f"N5-D4b staging bundle is incomplete: {path.name}")
    ready = json.loads(staged_ready.read_text(encoding="utf-8"))
    attestation = json.loads(staged_attestation.read_text(encoding="utf-8"))
    if attestation.get("schema") != (
        "n2-pvgr-n5-d4b-population-field-derivative-attestation-1.0"
    ):
        raise ValueError("N5-D4b staged attestation schema drifted")
    expected_ready = {
        "schema": "n2-pvgr-n5-d4b-preregistration-ready-1.0",
        "protocol_commit": attestation.get("protocol_commit"),
        "config_sha256": _sha256(config_path),
        "attestation_path": config["pre_registration_attestation"],
        "attestation_sha256": _sha256(staged_attestation),
        "frozen_input_archive_path": config["frozen_input_archive"],
        "frozen_input_archive_sha256": _sha256(staged_archive),
        "publication_rule": "atomically_rename_complete_staging_directory",
    }
    if ready != expected_ready:
        raise ValueError("N5-D4b staged READY marker or hash binding drifted")
    if attestation.get("frozen_input_archive_sha256") != _sha256(staged_archive):
        raise ValueError("N5-D4b attestation/archive binding drifted")
    return attestation


def _publish_staging_bundle(
    config_path: Path,
    config: dict[str, Any],
    staging: Path,
    bundle: Path,
) -> dict[str, Any]:
    attestation = _verify_complete_staging(config_path, config, staging)
    _atomic_rename_bundle(staging, bundle)
    return attestation


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
        raise RuntimeError("N5-D4b attestation must be created at the protocol commit")
    config_path = config_path.resolve()
    current_config = config_path.read_bytes()
    config_relative = _relative(config_path)
    frozen_config = _git("show", f"{protocol_commit}:{config_relative}", binary=True)
    if current_config != frozen_config:
        raise RuntimeError("current N5-D4b config differs from protocol commit")
    config = json.loads(current_config.decode("utf-8"))
    bundle, staging, expected_output, expected_archive, expected_ready = _bundle_paths(
        config
    )
    if output_path.resolve() != expected_output:
        raise ValueError("output path must match N5-D4b attestation path")
    if archive_path.resolve() != expected_archive:
        raise ValueError("archive path must match N5-D4b frozen-input path")
    if _path_occupied(bundle) or _path_occupied(staging):
        raise FileExistsError(
            "N5-D4b preregistration bundle or staging bundle already exists"
        )
    formal_output = (ROOT / config["formal_output"]).resolve()
    formal_work_output = (ROOT / config["formal_work_output"]).resolve()
    if _path_occupied(formal_output) or _path_occupied(formal_work_output):
        raise FileExistsError("N5-D4b formal or work output exists before attestation")

    files: dict[str, dict[str, Any]] = {}
    for key, relative in config["attested_files"].items():
        path = (ROOT / relative).resolve()
        if ROOT not in path.parents:
            raise ValueError(f"N5-D4b attested file escapes repository: {relative}")
        current = path.read_bytes()
        frozen = _git("show", f"{protocol_commit}:{relative}", binary=True)
        if current != frozen:
            raise RuntimeError(f"N5-D4b file differs from protocol commit: {relative}")
        files[str(key)] = {
            "path": relative,
            "sha256": _sha256_bytes(frozen),
            "bytes": len(frozen),
        }

    status_paths = [str(value) for value in config["attested_files"].values()]
    dirty = str(_git("status", "--porcelain", "--", *status_paths))
    if dirty:
        raise RuntimeError("N5-D4b attested files have uncommitted changes")

    staging.mkdir(parents=False)
    staged_output = staging / output_path.name
    staged_archive = staging / archive_path.name
    staged_ready = staging / expected_ready.name
    metadata, arrays = build_frozen_inputs(config)
    _write_npz_atomic(staged_archive, arrays)
    archive_inventory = {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256_float64_le_c_order": _sha256_bytes(value.tobytes(order="C")),
        }
        for key, value in sorted(arrays.items())
    }
    attestation = {
        "schema": "n2-pvgr-n5-d4b-population-field-derivative-attestation-1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_commit": protocol_commit,
        "repository_head_at_creation": str(_git("rev-parse", "HEAD")),
        "formal_results_absent_at_creation": True,
        "formal_output": config["formal_output"],
        "formal_work_output_absent_at_creation": True,
        "formal_work_output": config["formal_work_output"],
        "config_sha256": _sha256_bytes(current_config),
        "attested_files": files,
        "pre_registration_bundle": config["pre_registration_bundle"],
        "frozen_input_archive": config["frozen_input_archive"],
        "frozen_input_archive_sha256": _sha256(staged_archive),
        "frozen_input_archive_bytes": staged_archive.stat().st_size,
        "frozen_input_inventory": archive_inventory,
        "frozen_input_metadata": metadata,
        "frozen_input_metadata_sha256": _sha256_bytes(_canonical_json(metadata)),
        "pre_registration_ready_marker": config["pre_registration_ready_marker"],
        "bundle_publication": "single_atomic_directory_rename_after_READY",
        "claim_boundary": (
            "This binds every cell in the frozen 32-cell D3 pack census, the new "
            "unsearched direction/cotangent seeds, exact arrays, maps, h-grid, unchanged "
            "D4 gates, topology program and validator before formal D4b output existed. "
            "A pass can authorize only preregistration of a decoder-parameter-chain "
            "derivative gate. It cannot authorize reconstruction, neural-operator, "
            "real-data, generalization, superiority or paper claims."
        ),
    }
    _write_bytes_atomic(
        staged_output,
        (json.dumps(attestation, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    ready = {
        "schema": "n2-pvgr-n5-d4b-preregistration-ready-1.0",
        "protocol_commit": protocol_commit,
        "config_sha256": _sha256(config_path),
        "attestation_path": config["pre_registration_attestation"],
        "attestation_sha256": _sha256(staged_output),
        "frozen_input_archive_path": config["frozen_input_archive"],
        "frozen_input_archive_sha256": _sha256(staged_archive),
        "publication_rule": "atomically_rename_complete_staging_directory",
    }
    _write_bytes_atomic(
        staged_ready,
        (json.dumps(ready, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    _publish_staging_bundle(config_path, config, staging, bundle)
    return attestation


def recover_complete_staging(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bundle, staging, _, _, _ = _bundle_paths(config)
    if _path_occupied(bundle):
        raise FileExistsError("N5-D4b final preregistration bundle already exists")
    if not staging.is_dir():
        raise FileNotFoundError("N5-D4b staging bundle is absent")
    attestation = _verify_complete_staging(config_path, config, staging)
    protocol_commit = str(attestation["protocol_commit"])
    if str(_git("rev-parse", "HEAD")) != protocol_commit:
        raise RuntimeError("N5-D4b complete-staging recovery requires protocol HEAD")
    for record in attestation["attested_files"].values():
        path = (ROOT / record["path"]).resolve()
        frozen = _git("show", f"{protocol_commit}:{record['path']}", binary=True)
        if (
            _sha256(path) != record["sha256"]
            or _sha256_bytes(frozen) != record["sha256"]
        ):
            raise RuntimeError("N5-D4b staged dependency drifted before recovery")
    _publish_staging_bundle(config_path, config, staging, bundle)
    return attestation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol-commit")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--recover-complete-staging", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.recover_complete_staging:
        result = recover_complete_staging(args.config.resolve())
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if not args.protocol_commit:
        raise SystemExit("--protocol-commit is required unless recovering staging")
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
