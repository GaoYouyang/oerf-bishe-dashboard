#!/usr/bin/env python3
"""Build a lean GitHub Pages artifact from the tracked public repository."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


DEFAULT_MAX_BYTES = 900 * 1024 * 1024
EXCLUDED_PREFIXES = (
    ".git/",
    ".github/",
    "build/",
    "_site/",
    "private_library/",
    "paper_library/pdfs/",
)
EXCLUDED_SUFFIXES = (".pt", ".pth", ".ckpt")


@dataclass(frozen=True)
class ArtifactStats:
    copied_files: int
    copied_bytes: int
    excluded_pdf_files: int
    excluded_pdf_bytes: int
    excluded_checkpoint_files: int
    excluded_checkpoint_bytes: int


def should_exclude(relative_path: str) -> bool:
    normalized = PurePosixPath(relative_path).as_posix()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return any(
        normalized.startswith(prefix) for prefix in EXCLUDED_PREFIXES
    ) or normalized.lower().endswith(EXCLUDED_SUFFIXES)


def tracked_paths(repo_root: Path) -> list[str]:
    payload = subprocess.check_output(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
    )
    return [
        item.decode("utf-8")
        for item in payload.split(b"\0")
        if item
    ]


def _copy_tracked_files(
    repo_root: Path,
    output_root: Path,
    paths: Iterable[str],
) -> ArtifactStats:
    copied_files = 0
    copied_bytes = 0
    excluded_pdf_files = 0
    excluded_pdf_bytes = 0
    excluded_checkpoint_files = 0
    excluded_checkpoint_bytes = 0

    for relative in paths:
        source = repo_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"Tracked source is missing or not a file: {relative}")

        size = source.stat().st_size
        if relative.startswith("paper_library/pdfs/"):
            excluded_pdf_files += 1
            excluded_pdf_bytes += size
        if relative.lower().endswith(EXCLUDED_SUFFIXES):
            excluded_checkpoint_files += 1
            excluded_checkpoint_bytes += size

        if should_exclude(relative):
            continue

        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied_files += 1
        copied_bytes += size

    return ArtifactStats(
        copied_files=copied_files,
        copied_bytes=copied_bytes,
        excluded_pdf_files=excluded_pdf_files,
        excluded_pdf_bytes=excluded_pdf_bytes,
        excluded_checkpoint_files=excluded_checkpoint_files,
        excluded_checkpoint_bytes=excluded_checkpoint_bytes,
    )


def build_artifact(
    repo_root: Path,
    output_root: Path,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> ArtifactStats:
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    if output_root == repo_root or repo_root not in output_root.parents:
        raise ValueError("The output directory must be inside the repository root.")

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    paths = tracked_paths(repo_root)
    stats = _copy_tracked_files(repo_root, output_root, paths)
    if stats.copied_bytes > max_bytes:
        raise RuntimeError(
            "Pages artifact exceeds the configured safety ceiling: "
            f"{stats.copied_bytes} > {max_bytes} bytes"
        )

    required = (
        "index.html",
        "general_operator_research_lab.html",
        "404.html",
        ".nojekyll",
    )
    missing = [path for path in required if not (output_root / path).is_file()]
    if missing:
        raise RuntimeError(f"Pages artifact is missing required files: {missing}")

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip()
    manifest = {
        "schema_version": "oerf-pages-artifact-1.0",
        "source_commit": commit,
        "copied_files": stats.copied_files,
        "copied_bytes": stats.copied_bytes,
        "excluded_pdf_files": stats.excluded_pdf_files,
        "excluded_pdf_bytes": stats.excluded_pdf_bytes,
        "excluded_checkpoint_files": stats.excluded_checkpoint_files,
        "excluded_checkpoint_bytes": stats.excluded_checkpoint_bytes,
        "pdf_delivery": (
            "Tracked open-access PDFs are omitted from the Pages artifact. "
            "The custom 404 route redirects their stable Pages paths to the "
            "GitHub PDF viewer for the same tracked repository files."
        ),
    }
    (output_root / "pages-build-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="build/pages-site")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_root = repo_root / args.output
    stats = build_artifact(
        repo_root,
        output_root,
        max_bytes=args.max_bytes,
    )
    print(
        json.dumps(
            {
                "output": str(output_root),
                "copied_files": stats.copied_files,
                "copied_bytes": stats.copied_bytes,
                "excluded_pdf_files": stats.excluded_pdf_files,
                "excluded_pdf_bytes": stats.excluded_pdf_bytes,
                "excluded_checkpoint_files": stats.excluded_checkpoint_files,
                "excluded_checkpoint_bytes": stats.excluded_checkpoint_bytes,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
