from pathlib import Path

import pytest

from site_tools.psu_bost_selective_fetch import (
    DATASET_SLUG,
    archive_summary,
    is_minimal_entry,
    parse_index,
    safe_target,
)


FIXTURE = f"""
=====================================================
Content of ZIP archive  {DATASET_SLUG}-10.zip
=====================================================
Archive:  {DATASET_SLUG}-10.zip
  Length      Date    Time    Name
---------  ---------- -----   ----
     3335  02-11-2026 17:50   {DATASET_SLUG}/pyscripts/README.txt
  3747566  02-11-2026 19:57   {DATASET_SLUG}/readme.pdf
      123  02-11-2026 19:57   {DATASET_SLUG}/results/raw.bin

=====================================================
Content of ZIP archive  {DATASET_SLUG}-12.zip
=====================================================
      5072  02-11-2026 17:50   {DATASET_SLUG}/scripts/README.txt
      1541  02-09-2026 18:46   {DATASET_SLUG}/tools/Mesh_voxelisation/license.txt
     10032  02-09-2026 18:47   {DATASET_SLUG}/tools/cmap-master/acton.m
"""


def test_parse_index_preserves_archive_and_paths() -> None:
    entries = parse_index(FIXTURE)
    assert len(entries) == 6
    assert entries[0].part == 10
    assert entries[0].relative_path == "pyscripts/README.txt"
    assert entries[-1].part == 12


def test_summary_and_minimal_selector() -> None:
    entries = parse_index(FIXTURE)
    summary = archive_summary(entries)
    assert [row["files"] for row in summary] == [3, 3]
    chosen = [entry.relative_path for entry in entries if is_minimal_entry(entry)]
    assert chosen == [
        "pyscripts/README.txt",
        "readme.pdf",
        "scripts/README.txt",
        "tools/Mesh_voxelisation/license.txt",
    ]


def test_safe_target_rejects_path_traversal(tmp_path: Path) -> None:
    assert safe_target(tmp_path, "pyscripts/NIRT.py") == tmp_path / "pyscripts/NIRT.py"
    with pytest.raises(ValueError, match="unsafe archive path"):
        safe_target(tmp_path, "../outside.txt")


def test_empty_index_is_rejected() -> None:
    with pytest.raises(ValueError, match="no file entries"):
        parse_index("no archives here")
