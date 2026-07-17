#!/usr/bin/env python3
"""Open only the preregistered PSU rotation-40 development archive member."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Any, Mapping
from zipfile import BadZipFile, ZipFile


CONFIG_SCHEMA = "psu-rotation40-development-access-config-1.0"
PRIVATE_SCHEMA = "psu-rotation40-development-access-private-1.0"
PUBLIC_SCHEMA = "psu-rotation40-development-access-public-1.0"
CONFIG_STATUS = "FROZEN_BEFORE_FIRST_ROTATION40_EXTRACTION"
PRIVATE_STATUS = "ROTATION40_DEVELOPMENT_MEMBER_EXTRACTED_AND_VERIFIED"
PUBLIC_STATUS = "ROTATION40_DEVELOPMENT_ACCESS_VERIFIED_FINAL_ROTATIONS_REMAIN_SEALED"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CRC32_RE = re.compile(r"^[0-9a-f]{8}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def _require_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _safe_repo_path(repo_root: Path, relative: Any, name: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"{name} must be a nonempty relative path")
    candidate = (repo_root / relative).resolve()
    if not candidate.is_relative_to(repo_root.resolve()):
        raise ValueError(f"{name} escapes the repository root")
    return candidate


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unexpected rotation-40 access config schema")
    if config.get("status") != CONFIG_STATUS:
        raise ValueError("rotation-40 access config is not frozen")
    if config.get("purpose") != (
        "OPEN_ONLY_THE_PREDECLARED_ROTATION40_DEVELOPMENT_RUN_AFTER_SUPPORT_INTERFACE_FREEZE"
    ):
        raise ValueError("rotation-40 access purpose changed")
    dataset = config.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("dataset contract is absent")
    if int(dataset.get("rotation_degrees", -1)) != 40:
        raise ValueError("only rotation 40 is authorized")
    if dataset.get("archive_filename") != (
        "molnar-et-al-open-source-bos-tomography-dataset-2025-05.zip"
    ):
        raise ValueError("unexpected development archive filename")
    member = dataset.get("member")
    if not isinstance(member, str) or PurePosixPath(member).name != "HSOF_DEF_ROT_040.mat":
        raise ValueError("only the rotation-40 MAT member is authorized")
    if PurePosixPath(member).is_absolute() or ".." in PurePosixPath(member).parts:
        raise ValueError("archive member path is unsafe")
    for key in ("archive_bytes", "member_uncompressed_bytes", "member_compressed_bytes"):
        if int(dataset.get(key, 0)) < 1:
            raise ValueError(f"{key} must be positive")
    _require_sha(dataset.get("archive_sha256"), "dataset.archive_sha256")
    crc = dataset.get("member_crc32_hex")
    if not isinstance(crc, str) or CRC32_RE.fullmatch(crc) is None:
        raise ValueError("member CRC32 must be eight lowercase hex characters")
    anchors = config.get("frozen_anchors")
    if not isinstance(anchors, list) or len(anchors) < 3:
        raise ValueError("at least three frozen anchors are required")
    seen: set[str] = set()
    for index, anchor in enumerate(anchors):
        if not isinstance(anchor, Mapping):
            raise ValueError(f"frozen_anchors[{index}] must be an object")
        path = anchor.get("path")
        if not isinstance(path, str) or path in seen:
            raise ValueError("frozen anchor paths must be unique strings")
        seen.add(path)
        _require_sha(anchor.get("sha256"), f"frozen_anchors[{index}].sha256")
    forbidden = set(config.get("forbidden_uses", []))
    required_forbidden = {
        "opening_any_other_rotation_member",
        "calling_rotation40_confirmatory_or_final_evidence",
        "claiming_experimental_field_l2_or_unique_three_dimensional_truth",
        "publishing_raw_or_derived_measurement_arrays",
    }
    if not required_forbidden.issubset(forbidden):
        raise ValueError("rotation-40 claim firewall is incomplete")
    public = config.get("public_export")
    if not isinstance(public, Mapping) or any(
        public.get(key) is not False
        for key in ("include_local_paths", "include_measurement_values", "include_extracted_arrays")
    ):
        raise ValueError("public export must exclude paths, values, and arrays")


def _verify_anchors(repo_root: Path, config: Mapping[str, Any]) -> list[dict[str, str]]:
    verified = []
    for anchor in config["frozen_anchors"]:
        path = _safe_repo_path(repo_root, anchor["path"], "frozen anchor path")
        if not path.is_file():
            raise FileNotFoundError(f"frozen anchor is absent: {anchor['path']}")
        actual = _sha256(path)
        if actual != anchor["sha256"]:
            raise ValueError(f"frozen anchor checksum mismatch: {anchor['path']}")
        verified.append({"path": anchor["path"], "sha256": actual})
    return verified


def open_rotation40_development(
    *,
    repo_root: Path,
    config_path: Path,
    archive_path: Path,
    output_dir: Path,
    public_summary_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    archive_path = archive_path.resolve()
    output_dir = output_dir.resolve()
    config = _read_json(config_path)
    validate_config(config)
    anchors = _verify_anchors(repo_root, config)
    dataset = config["dataset"]
    if archive_path.name != dataset["archive_filename"]:
        raise ValueError("archive filename differs from the frozen contract")
    if archive_path.stat().st_size != int(dataset["archive_bytes"]):
        raise ValueError("archive byte count differs from the frozen contract")
    archive_sha = _sha256(archive_path)
    if archive_sha != dataset["archive_sha256"]:
        raise ValueError("archive checksum differs from the frozen contract")

    try:
        archive = ZipFile(archive_path)
    except BadZipFile as exc:
        raise ValueError("development archive is not a valid ZIP file") from exc
    member_name = str(dataset["member"])
    try:
        info = archive.getinfo(member_name)
    except KeyError as exc:
        archive.close()
        raise ValueError("frozen rotation-40 member is absent") from exc
    expected_crc = str(dataset["member_crc32_hex"])
    if (
        int(info.file_size) != int(dataset["member_uncompressed_bytes"])
        or int(info.compress_size) != int(dataset["member_compressed_bytes"])
        or f"{info.CRC:08x}" != expected_crc
    ):
        archive.close()
        raise ValueError("rotation-40 member metadata differs from the frozen contract")

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_path = output_dir / PurePosixPath(member_name).name
    partial_path = output_dir / f".{extracted_path.name}.partial"
    if extracted_path.exists():
        if extracted_path.stat().st_size != info.file_size:
            archive.close()
            raise ValueError("existing rotation-40 extraction has the wrong size")
        extraction_mode = "VERIFIED_EXISTING"
    else:
        partial_path.unlink(missing_ok=True)
        try:
            with archive.open(info, "r") as source, partial_path.open("xb") as target:
                shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
            if partial_path.stat().st_size != info.file_size:
                raise ValueError("extracted rotation-40 member has the wrong size")
            os.replace(partial_path, extracted_path)
        except Exception:
            partial_path.unlink(missing_ok=True)
            archive.close()
            raise
        extraction_mode = "EXTRACTED_NOW"
    archive.close()
    extracted_sha = _sha256(extracted_path)
    config_sha = _sha256(config_path)
    private_report = {
        "schema_version": PRIVATE_SCHEMA,
        "status": PRIVATE_STATUS,
        "evidence_scope": "DEVELOPMENT_ACCESS_AND_INTEGRITY_ONLY_NO_ARRAY_INTERPRETATION_NO_MODEL_SCORING",
        "dataset": {
            "doi": dataset["doi"],
            "rotation_degrees": 40,
            "archive_filename": archive_path.name,
            "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": archive_sha,
            "member": member_name,
            "member_uncompressed_bytes": info.file_size,
            "member_compressed_bytes": info.compress_size,
            "member_crc32_hex": f"{info.CRC:08x}",
            "extracted_filename": extracted_path.name,
            "extracted_sha256": extracted_sha,
        },
        "config_sha256": config_sha,
        "verified_anchors": anchors,
        "extraction_mode": extraction_mode,
        "claim_boundary": {
            "development_only": True,
            "final_rotations_opened": False,
            "measurement_values_interpreted": False,
            "three_dimensional_reconstruction_run": False,
            "algorithm_superiority": False,
        },
    }
    _write_json_atomic(output_dir / "access_private_report.json", private_report)
    public_summary = {
        "schema_version": PUBLIC_SCHEMA,
        "status": PUBLIC_STATUS,
        "evidence_scope": private_report["evidence_scope"],
        "dataset": {
            "doi": dataset["doi"],
            "rotation_degrees": 40,
            "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": archive_sha,
            "member_uncompressed_bytes": info.file_size,
            "member_compressed_bytes": info.compress_size,
            "member_crc32_hex": f"{info.CRC:08x}",
            "extracted_sha256": extracted_sha,
        },
        "config_sha256": config_sha,
        "verified_anchor_count": len(anchors),
        "extraction_mode": extraction_mode,
        "claim_boundary": dict(private_report["claim_boundary"]),
        "public_export_policy": {
            "contains_local_paths": False,
            "contains_measurement_values": False,
            "contains_extracted_arrays": False,
        },
    }
    if public_summary_path is not None:
        _write_json_atomic(public_summary_path.resolve(), public_summary)
    return private_report, public_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-summary", type=Path)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    _, public = open_rotation40_development(
        repo_root=repo_root,
        config_path=args.config,
        archive_path=args.archive,
        output_dir=args.private_output,
        public_summary_path=args.public_summary,
    )
    print(json.dumps(public, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
