#!/usr/bin/env python3
"""Audit and selectively fetch the Penn State flight-body BOS dataset.

The default mode downloads only the public text index and prints a derived
summary.  ``--fetch-minimal`` uses HTTP range requests through ``remotezip``
to fetch a small documentation/code subset without downloading all 48.1 GiB.
Raw data remain outside Git because the default destination is private_library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable


DATASET_SLUG = "molnar-et-al-open-source-bos-tomography-dataset-2025"
DOWNLOAD_ROOT = (
    "https://www.datacommons.psu.edu/download/engineering/"
    "molnar-et-al-open-source-bos-tomography-dataset-of-high-speed-flow-over-a-flight-body-2025"
)
INDEX_URL = f"{DOWNLOAD_ROOT}/000-readme-zip-file-content.txt"
ARCHIVE_URL = f"{DOWNLOAD_ROOT}/{DATASET_SLUG}-{{part:02d}}.zip"
ARCHIVE_HEADER = re.compile(r"^Content of ZIP archive\s+(\S+)\s*$")
ENTRY_LINE = re.compile(
    r"^\s*(?P<size>\d+)\s+\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}\s+(?P<path>.+?)\s*$"
)


@dataclass(frozen=True)
class IndexEntry:
    archive: str
    size_bytes: int
    path: str

    @property
    def part(self) -> int:
        match = re.search(r"-(\d{2})\.zip$", self.archive)
        if not match:
            raise ValueError(f"cannot infer archive part from {self.archive!r}")
        return int(match.group(1))

    @property
    def relative_path(self) -> str:
        prefix = f"{DATASET_SLUG}/"
        return self.path[len(prefix) :] if self.path.startswith(prefix) else self.path


def parse_index(text: str) -> list[IndexEntry]:
    """Parse the repository-provided ``unzip -l`` style content index."""

    archive: str | None = None
    entries: list[IndexEntry] = []
    for raw_line in text.splitlines():
        header = ARCHIVE_HEADER.match(raw_line)
        if header:
            archive = header.group(1)
            continue
        match = ENTRY_LINE.match(raw_line)
        if not match or archive is None:
            continue
        path = match.group("path")
        if path.endswith("/"):
            continue
        entries.append(
            IndexEntry(
                archive=archive,
                size_bytes=int(match.group("size")),
                path=path,
            )
        )
    if not entries:
        raise ValueError("the official index contained no file entries")
    return entries


def archive_summary(entries: Iterable[IndexEntry]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, int]] = {}
    for entry in entries:
        row = grouped.setdefault(entry.archive, {"files": 0, "uncompressed_bytes": 0})
        row["files"] += 1
        row["uncompressed_bytes"] += entry.size_bytes
    return [
        {"archive": archive, **grouped[archive]}
        for archive in sorted(grouped)
    ]


def is_minimal_entry(entry: IndexEntry) -> bool:
    """Return the small, directly useful code/documentation subset."""

    rel = entry.relative_path
    if entry.part == 10:
        return (
            rel == "readme.pdf"
            or rel.startswith("pyscripts/")
            or rel.endswith("/Recon/Summary.svg")
            or rel
            in {
                "results/Deflections/Viz_Def_CUT01_CAM03_ROT000.svg",
                "results/Deflections/Viz_Def_CUT02_CAM03_ROT000.svg",
                "results/Deflections/Viz_Def_CUT03_CAM03_ROT000.svg",
            }
        )
    if entry.part != 12:
        return False
    if rel == "results/results_info.txt":
        return True
    if rel.startswith("scripts/") and "/Viz/" not in rel:
        return rel.endswith((".m", ".txt"))
    if rel.startswith("tools/Mesh_voxelisation/"):
        return rel.endswith((".m", ".txt"))
    if rel.startswith("tools/") and rel.count("/") == 1:
        return rel.endswith((".m", ".txt"))
    return False


def safe_target(root: Path, relative_path: str) -> Path:
    root = root.resolve()
    target = (root / relative_path).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"unsafe archive path: {relative_path!r}")
    return target


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_text(url: str, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "oerf-bost-audit/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = response.read()
    destination.write_bytes(payload)
    return payload.decode("utf-8")


def fetch_selected(
    entries: Iterable[IndexEntry],
    destination: Path,
    selector: Callable[[IndexEntry], bool] = is_minimal_entry,
    retries: int = 3,
) -> list[dict[str, object]]:
    try:
        from remotezip import RemoteZip
    except ImportError as exc:  # pragma: no cover - environment-dependent branch
        raise RuntimeError(
            "--fetch-minimal requires remotezip; install it with "
            "`.venv/bin/python -m pip install remotezip`"
        ) from exc

    chosen = [entry for entry in entries if selector(entry)]
    downloaded: list[dict[str, object]] = []
    for number, entry in enumerate(chosen, start=1):
        target = safe_target(destination, entry.relative_path)
        if target.exists() and target.stat().st_size == entry.size_bytes:
            downloaded.append(
                {
                    **asdict(entry),
                    "relative_path": entry.relative_path,
                    "status": "present",
                    "sha256": sha256_file(target),
                }
            )
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f"{target.name}.partial")
        for attempt in range(1, retries + 1):
            try:
                with RemoteZip(ARCHIVE_URL.format(part=entry.part), timeout=90) as archive:
                    payload = archive.read(entry.path)
                if len(payload) != entry.size_bytes:
                    raise OSError(
                        f"size mismatch for {entry.relative_path}: "
                        f"{len(payload)} != {entry.size_bytes}"
                    )
                partial.write_bytes(payload)
                partial.replace(target)
                break
            except Exception:
                if attempt == retries:
                    raise
                time.sleep(2 * attempt)
        downloaded.append(
            {
                **asdict(entry),
                "relative_path": entry.relative_path,
                "status": "downloaded",
                "sha256": sha256_file(target),
                "ordinal": f"{number}/{len(chosen)}",
            }
        )
    return downloaded


def build_report(entries: list[IndexEntry]) -> dict[str, object]:
    minimal = [entry for entry in entries if is_minimal_entry(entry)]
    return {
        "source_index": INDEX_URL,
        "dataset_doi": "10.26208/1VE2-5C19",
        "archive_count": len({entry.archive for entry in entries}),
        "file_count": len(entries),
        "uncompressed_bytes_listed": sum(entry.size_bytes for entry in entries),
        "minimal_file_count": len(minimal),
        "minimal_uncompressed_bytes": sum(entry.size_bytes for entry in minimal),
        "archives": archive_summary(entries),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-cache",
        type=Path,
        default=Path("private_library/external_datasets/psu_bost_flight_body/000-readme-zip-file-content.txt"),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("private_library/external_datasets/psu_bost_flight_body"),
    )
    parser.add_argument("--refresh-index", action="store_true")
    parser.add_argument("--fetch-minimal", action="store_true")
    args = parser.parse_args()

    if args.refresh_index or not args.index_cache.exists():
        text = download_text(INDEX_URL, args.index_cache)
    else:
        text = args.index_cache.read_text(encoding="utf-8")
    entries = parse_index(text)
    report = build_report(entries)

    if args.fetch_minimal:
        report["downloads"] = fetch_selected(entries, args.destination)
        manifest_path = args.destination / "minimal_manifest.json"
        manifest_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        report["local_manifest"] = str(manifest_path)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
