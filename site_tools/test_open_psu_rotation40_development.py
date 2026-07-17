from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import pytest

from site_tools.open_psu_rotation40_development import (
    PUBLIC_STATUS,
    open_rotation40_development,
)


MEMBER = "dataset/data/DEF_PROC/HSOF_DEF_ROT_040.mat"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    anchors = []
    for index in range(3):
        path = repo / f"anchor-{index}.json"
        path.write_text(f"anchor {index}\n", encoding="utf-8")
        anchors.append({"path": path.name, "sha256": _sha(path)})
    archive = tmp_path / "molnar-et-al-open-source-bos-tomography-dataset-2025-05.zip"
    payload = b"rotation-40-fixture" * 17
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as handle:
        handle.writestr(MEMBER, payload)
    with ZipFile(archive) as handle:
        info = handle.getinfo(MEMBER)
    config = {
        "schema_version": "psu-rotation40-development-access-config-1.0",
        "status": "FROZEN_BEFORE_FIRST_ROTATION40_EXTRACTION",
        "purpose": "OPEN_ONLY_THE_PREDECLARED_ROTATION40_DEVELOPMENT_RUN_AFTER_SUPPORT_INTERFACE_FREEZE",
        "dataset": {
            "doi": "10.26208/1VE2-5C19",
            "archive_filename": archive.name,
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": _sha(archive),
            "member": MEMBER,
            "member_uncompressed_bytes": info.file_size,
            "member_compressed_bytes": info.compress_size,
            "member_crc32_hex": f"{info.CRC:08x}",
            "rotation_degrees": 40,
        },
        "frozen_anchors": anchors,
        "authorized_uses": ["schema_and_loader_audit"],
        "forbidden_uses": [
            "opening_any_other_rotation_member",
            "calling_rotation40_confirmatory_or_final_evidence",
            "claiming_experimental_field_l2_or_unique_three_dimensional_truth",
            "publishing_raw_or_derived_measurement_arrays",
        ],
        "public_export": {
            "include_archive_or_member_hashes": True,
            "include_local_paths": False,
            "include_measurement_values": False,
            "include_extracted_arrays": False,
        },
    }
    config_path = repo / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return repo, config_path, archive, tmp_path / "private"


def test_extracts_only_frozen_member_and_reuses_verified_file(tmp_path: Path) -> None:
    repo, config, archive, output = _fixture(tmp_path)
    _, public = open_rotation40_development(
        repo_root=repo,
        config_path=config,
        archive_path=archive,
        output_dir=output,
        public_summary_path=tmp_path / "public.json",
    )
    assert public["status"] == PUBLIC_STATUS
    assert public["extraction_mode"] == "EXTRACTED_NOW"
    assert public["claim_boundary"]["final_rotations_opened"] is False
    assert sorted(path.name for path in output.iterdir()) == [
        "HSOF_DEF_ROT_040.mat",
        "access_private_report.json",
    ]
    _, repeated = open_rotation40_development(
        repo_root=repo,
        config_path=config,
        archive_path=archive,
        output_dir=output,
    )
    assert repeated["extraction_mode"] == "VERIFIED_EXISTING"


def test_rejects_anchor_tamper(tmp_path: Path) -> None:
    repo, config, archive, output = _fixture(tmp_path)
    (repo / "anchor-1.json").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="anchor checksum mismatch"):
        open_rotation40_development(
            repo_root=repo,
            config_path=config,
            archive_path=archive,
            output_dir=output,
        )


def test_rejects_archive_tamper(tmp_path: Path) -> None:
    repo, config, archive, output = _fixture(tmp_path)
    with archive.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="byte count"):
        open_rotation40_development(
            repo_root=repo,
            config_path=config,
            archive_path=archive,
            output_dir=output,
        )


def test_rejects_non_rotation40_member_even_if_metadata_matches(tmp_path: Path) -> None:
    repo, config_path, archive, output = _fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["dataset"]["member"] = "dataset/data/DEF_PROC/HSOF_DEF_ROT_030.mat"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="only the rotation-40 MAT"):
        open_rotation40_development(
            repo_root=repo,
            config_path=config_path,
            archive_path=archive,
            output_dir=output,
        )
