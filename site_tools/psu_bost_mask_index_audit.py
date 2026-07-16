#!/usr/bin/env python3
"""Audit MATLAB-to-Python index-base semantics for PSU BOST mask lists."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .psu_bost_mat_stream import open_measurement_stream
except ImportError:  # Direct script execution.
    from psu_bost_mat_stream import open_measurement_stream  # type: ignore[no-redef]


MASK_VARIABLES = ("amask_all", "imask_all")


def _line_number(source: str, needle: str) -> int | None:
    offset = source.find(needle)
    return None if offset < 0 else source.count("\n", 0, offset) + 1


def _source_record(path: Path, needles: tuple[str, ...]) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8", errors="replace")
    return {
        "filename": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "evidence": [
            {"needle": needle, "line": _line_number(source, needle)}
            for needle in needles
        ],
    }


def _mask_statistics(
    mat_path: Path,
    variable: str,
    *,
    measurement_count: int,
    chunk_measurements: int,
) -> dict[str, Any]:
    stream = open_measurement_stream(
        mat_path,
        variable,
        chunk_measurements=chunk_measurements,
        cast_dtype="int64",
    )
    minimum = measurement_count + 1
    maximum = -1
    count = 0
    zero_count = 0
    endpoint_count = 0
    invalid_zero_based = 0
    invalid_one_based = 0
    strictly_increasing = True
    previous: int | None = None
    for chunk in stream:
        values = chunk.values[:, 0]
        if values.size == 0:
            continue
        minimum = min(minimum, int(values.min()))
        maximum = max(maximum, int(values.max()))
        count += int(values.size)
        zero_count += int(np.count_nonzero(values == 0))
        endpoint_count += int(np.count_nonzero(values == measurement_count))
        invalid_zero_based += int(
            np.count_nonzero((values < 0) | (values >= measurement_count))
        )
        invalid_one_based += int(
            np.count_nonzero((values < 1) | (values > measurement_count))
        )
        if previous is not None and int(values[0]) <= previous:
            strictly_increasing = False
        if values.size > 1 and np.any(np.diff(values) <= 0):
            strictly_increasing = False
        previous = int(values[-1])
    if not stream.audit.complete:
        raise RuntimeError(f"{variable} stream did not complete integrity verification")
    return {
        "variable": variable,
        "count": count,
        "minimum": minimum,
        "maximum": maximum,
        "zero_count": zero_count,
        "measurement_count_endpoint_count": endpoint_count,
        "strictly_increasing": strictly_increasing,
        "valid_as_zero_based": invalid_zero_based == 0,
        "invalid_as_zero_based_count": invalid_zero_based,
        "valid_as_one_based": invalid_one_based == 0,
        "invalid_as_one_based_count": invalid_one_based,
        "full_source_stream_verified": stream.audit.matrix_stream_verified,
        "source_numeric_sha256": stream.audit.source_numeric_sha256,
    }


def build_mask_index_report(
    *,
    mat_path: Path,
    measurement_count: int,
    producer_source: Path,
    setup_source: Path,
    sample_source: Path,
    chunk_measurements: int = 262_144,
) -> dict[str, Any]:
    if measurement_count < 1:
        raise ValueError("measurement_count must be positive")
    producer_text = producer_source.read_text(encoding="utf-8", errors="replace")
    setup_text = setup_source.read_text(encoding="utf-8", errors="replace")
    sample_text = sample_source.read_text(encoding="utf-8", errors="replace")
    masks = [
        _mask_statistics(
            mat_path,
            variable,
            measurement_count=measurement_count,
            chunk_measurements=chunk_measurements,
        )
        for variable in MASK_VARIABLES
    ]

    producer_find = all(
        token in producer_text
        for token in ("amask_all = find(amask_all)", "imask_all = find(imask_all)")
    )
    setup_loads_indices = all(
        token in setup_text
        for token in ("data['amask_all'].T", "data['imask_all'].T")
    )
    explicit_conversion = bool(
        re.search(
            r"(?:amask|imask)\s*=\s*[^\n]*(?:-\s*1|subtract\s*\([^\n]*,\s*1)",
            setup_text,
        )
    )
    consumer_gathers = "tf.gather(pdict['masks']" in sample_text
    one_based_endpoint_observed = any(
        item["measurement_count_endpoint_count"] > 0 for item in masks
    )
    all_one_based_valid = all(item["valid_as_one_based"] for item in masks)
    mismatch_confirmed = all(
        (
            producer_find,
            setup_loads_indices,
            consumer_gathers,
            not explicit_conversion,
            one_based_endpoint_observed,
            all_one_based_valid,
        )
    )

    checks = [
        {
            "name": "producer_uses_matlab_find",
            "passed": producer_find,
            "meaning": "MATLAB find emits one-based linear indices",
        },
        {
            "name": "real_mask_streams_verified",
            "passed": all(item["full_source_stream_verified"] for item in masks),
            "meaning": "both real mask payloads were fully decoded and hashed",
        },
        {
            "name": "one_based_endpoint_observed",
            "passed": one_based_endpoint_observed,
            "meaning": "at least one stored index equals N and is invalid for zero-based gather",
        },
        {
            "name": "consumer_has_no_base_conversion",
            "passed": not explicit_conversion,
            "meaning": "the official setup path forwards stored indices without subtracting one",
        },
        {
            "name": "tensorflow_gather_consumes_masks",
            "passed": consumer_gathers,
            "meaning": "the lists are used as Python/TensorFlow zero-based gather indices",
        },
    ]
    return {
        "schema_version": "psu-bost-mask-index-audit-1.0",
        "status": (
            "MASK_INDEX_BASE_MISMATCH_CONFIRMED"
            if mismatch_confirmed
            else "MASK_INDEX_BASE_INCONCLUSIVE"
        ),
        "evidence_scope": "REAL_MASK_NUMERIC_STREAMS_PLUS_PRODUCER_CONSUMER_SOURCE_CONTRACT_NO_TENSORFLOW_EXECUTION",
        "configuration": {"measurement_count": measurement_count},
        "masks": masks,
        "source_contract": {
            "producer": _source_record(
                producer_source,
                (
                    "amask_all = find(amask_all)",
                    "imask_all = find(imask_all)",
                ),
            ),
            "setup": _source_record(
                setup_source,
                ("data['amask_all'].T", "data['imask_all'].T"),
            ),
            "sampler": _source_record(
                sample_source, ("tf.gather(pdict['masks']",)
            ),
            "explicit_zero_base_conversion_found": explicit_conversion,
        },
        "checks": checks,
        "decision": {
            "official_mask_indices_safe_for_python_gather": "NO_GO"
            if mismatch_confirmed
            else "UNRESOLVED",
            "required_adapter_action": (
                "CAST_TO_INT64_THEN_SUBTRACT_ONE_BEFORE_SORT_OR_GATHER"
                if mismatch_confirmed
                else "DO_NOT_TRANSFORM_UNTIL_INDEX_BASE_IS_RESOLVED"
            ),
            "author_source_modified": False,
            "next_gate": "TEST_CORRECTED_MASK_SELECTION_AGAINST_REAL_DEFLECTION_ROWS",
        },
        "limitations": [
            "the audit proves an index-base contract mismatch, not its effect on a completed reconstruction",
            "the probability of sampling the single out-of-range endpoint is not a measure of the silent one-pixel shift affecting all indices",
            "the public adapter must preserve the unmodified author source and apply any correction explicitly",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--measurement-count", type=int, required=True)
    parser.add_argument("--producer-source", type=Path, required=True)
    parser.add_argument("--setup-source", type=Path, required=True)
    parser.add_argument("--sample-source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_mask_index_report(
        mat_path=args.mat,
        measurement_count=args.measurement_count,
        producer_source=args.producer_source,
        setup_source=args.setup_source,
        sample_source=args.sample_source,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["status"].endswith("CONFIRMED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
