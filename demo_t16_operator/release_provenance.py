"""Small provenance helpers shared by the V5P-V6A release runners."""

from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def relative_file_hashes(root: str | Path, paths: Iterable[str | Path]) -> dict[str, str]:
    """Hash files and key them by stable paths relative to ``root``."""

    resolved_root = Path(root).resolve()
    output: dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        relative = path.relative_to(resolved_root).as_posix()
        output[relative] = file_sha256(path)
    return dict(sorted(output.items()))


def runtime_environment(*, device: str, torch_version: str) -> dict[str, str]:
    """Return a privacy-safe runtime fingerprint without user or host names."""

    return {
        "device": str(device),
        "machine": platform.machine(),
        "numpy_version": np.__version__,
        "operating_system": platform.system(),
        "operating_system_release": platform.release(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "torch_version": str(torch_version),
        "word_size": str(8 * (sys.maxsize.bit_length() + 1) // 8),
    }
