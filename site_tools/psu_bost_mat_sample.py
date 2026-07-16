#!/usr/bin/env python3
"""Sample selected numeric variables from the 5 GB PSU BOS MAT file.

The official file is MATLAB v5 with one zlib-compressed matrix per top-level
element and an opaque MCOS subsystem at the end.  This reader indexes named
numeric matrices before that subsystem, then streams only requested variables.
Large arrays are never materialized: a deterministic set of scalar values and
the complete numeric-payload SHA-256 are collected while the zlib stream is
validated end to end.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    from site_tools.psu_bost_mat_audit import (
        MATLAB_CLASSES,
        MAT_V5_HEADER_BYTES,
        MI_COMPRESSED,
        MI_MATRIX,
        _compressed_prefix,
        _data_element_tag,
        _padded_bytes,
    )
except ModuleNotFoundError:  # Direct ``python site_tools/...py`` invocation.
    from psu_bost_mat_audit import (  # type: ignore[no-redef]
        MATLAB_CLASSES,
        MAT_V5_HEADER_BYTES,
        MI_COMPRESSED,
        MI_MATRIX,
        _compressed_prefix,
        _data_element_tag,
        _padded_bytes,
    )


MI_INT8 = 1
MI_UINT8 = 2
MI_INT16 = 3
MI_UINT16 = 4
MI_INT32 = 5
MI_UINT32 = 6
MI_SINGLE = 7
MI_DOUBLE = 9
MI_INT64 = 12
MI_UINT64 = 13

NUMERIC_TYPES: dict[int, tuple[str, int, str]] = {
    MI_INT8: ("int8", 1, "b"),
    MI_UINT8: ("uint8", 1, "B"),
    MI_INT16: ("int16", 2, "h"),
    MI_UINT16: ("uint16", 2, "H"),
    MI_INT32: ("int32", 4, "i"),
    MI_UINT32: ("uint32", 4, "I"),
    MI_SINGLE: ("single", 4, "f"),
    MI_DOUBLE: ("double", 8, "d"),
    MI_INT64: ("int64", 8, "q"),
    MI_UINT64: ("uint64", 8, "Q"),
}
DEFAULT_FULL_THRESHOLD_BYTES = 64 * 1024


@dataclass(frozen=True)
class MatrixEntry:
    name: str
    shape: tuple[int, ...]
    matlab_class: str
    endian: str
    top_offset: int
    top_payload_offset: int
    top_payload_bytes: int
    compressed: bool
    matrix_stream_bytes: int
    numeric_type: str
    numeric_type_id: int
    item_bytes: int
    numeric_payload_offset: int
    numeric_payload_bytes: int
    is_complex: bool
    is_logical: bool

    @property
    def elements(self) -> int:
        count = 1
        for dimension in self.shape:
            count *= int(dimension)
        return count


@dataclass(frozen=True)
class MatIndex:
    path: Path
    endian: str
    file_size_bytes: int
    subsystem_offset: int
    entries: tuple[MatrixEntry, ...]

    @property
    def mcos_subsystem_present(self) -> bool:
        return self.subsystem_offset < self.file_size_bytes


def _read_mat_header(path: Path) -> tuple[str, int, int]:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        header = handle.read(MAT_V5_HEADER_BYTES)
    if len(header) != MAT_V5_HEADER_BYTES or not header.startswith(
        b"MATLAB 5.0 MAT-file"
    ):
        raise ValueError("only MATLAB v5 MAT files are supported")
    marker = header[126:128]
    if marker == b"IM":
        endian = "<"
    elif marker == b"MI":
        endian = ">"
    else:
        raise ValueError("invalid MATLAB v5 endian marker")
    subsystem_offset = struct.unpack(f"{endian}Q", header[116:124])[0]
    boundary = (
        int(subsystem_offset)
        if MAT_V5_HEADER_BYTES <= subsystem_offset <= file_size
        else file_size
    )
    return endian, file_size, boundary


def _matrix_layout_from_prefix(
    prefix: bytes,
    *,
    endian: str,
    top_offset: int,
    top_payload_offset: int,
    top_payload_bytes: int,
    compressed: bool,
) -> MatrixEntry | None:
    matrix_type, matrix_bytes, position, matrix_total = _data_element_tag(
        prefix, 0, endian
    )
    if matrix_type != MI_MATRIX:
        raise ValueError(f"matrix stream starts with MAT type {matrix_type}")

    flags_type, flags_bytes, flags_start, flags_total = _data_element_tag(
        prefix, position, endian
    )
    if flags_type != MI_UINT32 or flags_bytes < 8:
        raise ValueError("invalid MAT array-flags element")
    flags = struct.unpack_from(f"{endian}I", prefix, flags_start)[0]
    matlab_class_id = flags & 0xFF
    if matlab_class_id == 17:
        return None
    position += flags_total

    dimensions_type, dimensions_bytes, dimensions_start, dimensions_total = (
        _data_element_tag(prefix, position, endian)
    )
    if dimensions_type != MI_INT32 or dimensions_bytes % 4:
        raise ValueError("invalid MAT dimensions element")
    if dimensions_start + dimensions_bytes > len(prefix):
        raise ValueError("truncated MAT dimensions element")
    dimensions = struct.unpack_from(
        f"{endian}{dimensions_bytes // 4}i", prefix, dimensions_start
    )
    if any(value < 0 for value in dimensions):
        raise ValueError("negative MAT matrix dimension")
    position += dimensions_total

    name_type, name_bytes, name_start, name_total = _data_element_tag(
        prefix, position, endian
    )
    if name_type not in {MI_INT8, MI_UINT8}:
        raise ValueError("invalid MAT variable-name element")
    if name_start + name_bytes > len(prefix):
        raise ValueError("truncated MAT variable name")
    name = prefix[name_start : name_start + name_bytes].decode("latin1")
    if not name:
        return None
    position += name_total

    # Camera calibration objects and the MCOS wrapper are legitimate named
    # matrices in the archive, but they are outside this numeric reader's
    # contract.  Skip them explicitly instead of mistaking nested miMATRIX
    # elements for numeric payloads.
    if matlab_class_id not in range(6, 16):
        return None

    data_type, data_bytes, data_start, _ = _data_element_tag(
        prefix, position, endian
    )
    if data_type not in NUMERIC_TYPES:
        raise ValueError(
            f"variable {name!r} uses unsupported numeric MAT type {data_type}"
        )
    numeric_type, item_bytes, _ = NUMERIC_TYPES[data_type]
    elements = math.prod(int(value) for value in dimensions)
    if data_bytes != elements * item_bytes:
        raise ValueError(
            f"variable {name!r} declares {data_bytes} numeric bytes, "
            f"expected {elements * item_bytes}"
        )
    if flags & 0x800:
        raise ValueError(f"complex MAT variable {name!r} is unsupported")

    return MatrixEntry(
        name=name,
        shape=tuple(int(value) for value in dimensions),
        matlab_class=MATLAB_CLASSES.get(
            matlab_class_id, f"class_{matlab_class_id}"
        ),
        endian=endian,
        top_offset=top_offset,
        top_payload_offset=top_payload_offset,
        top_payload_bytes=int(top_payload_bytes),
        compressed=compressed,
        matrix_stream_bytes=int(matrix_total),
        numeric_type=numeric_type,
        numeric_type_id=int(data_type),
        item_bytes=item_bytes,
        numeric_payload_offset=int(data_start),
        numeric_payload_bytes=int(data_bytes),
        is_complex=bool(flags & 0x800),
        is_logical=bool(flags & 0x200),
    )


def build_index(path: Path) -> MatIndex:
    """Index named numeric matrices without reading their numeric payloads."""

    endian, file_size, boundary = _read_mat_header(path)
    entries: list[MatrixEntry] = []
    with path.open("rb") as handle:
        position = MAT_V5_HEADER_BYTES
        while position < boundary:
            handle.seek(position)
            tag = handle.read(8)
            if len(tag) != 8:
                raise ValueError(f"truncated top-level MAT tag at byte {position}")
            data_type, payload_bytes = struct.unpack(f"{endian}II", tag)
            payload_start = position + 8
            payload_end = payload_start + int(payload_bytes)
            if payload_end > boundary:
                raise ValueError(
                    f"MAT element at byte {position} crosses subsystem boundary"
                )

            if data_type == MI_COMPRESSED:
                prefix = _compressed_prefix(
                    handle,
                    payload_start=payload_start,
                    payload_bytes=payload_bytes,
                )
                entry = _matrix_layout_from_prefix(
                    prefix,
                    endian=endian,
                    top_offset=position,
                    top_payload_offset=payload_start,
                    top_payload_bytes=payload_bytes,
                    compressed=True,
                )
                position = payload_end
            elif data_type == MI_MATRIX:
                handle.seek(position)
                prefix = handle.read(min(8 + int(payload_bytes), 65536))
                entry = _matrix_layout_from_prefix(
                    prefix,
                    endian=endian,
                    top_offset=position,
                    top_payload_offset=payload_start,
                    top_payload_bytes=payload_bytes,
                    compressed=False,
                )
                position = payload_start + _padded_bytes(payload_bytes)
            else:
                raise ValueError(
                    f"unsupported top-level MAT type {data_type} at byte {position}"
                )
            if entry is not None:
                entries.append(entry)
        if position != boundary:
            raise ValueError(f"MAT index ended at {position}, expected {boundary}")

    names = [entry.name for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError("duplicate named matrices in MAT file")
    return MatIndex(
        path=path,
        endian=endian,
        file_size_bytes=file_size,
        subsystem_offset=boundary,
        entries=tuple(entries),
    )


def _iter_matrix_stream(
    handle: Any, entry: MatrixEntry, *, chunk_bytes: int = 1024 * 1024
) -> Iterator[bytes]:
    handle.seek(entry.top_payload_offset if entry.compressed else entry.top_offset)
    if not entry.compressed:
        remaining = entry.matrix_stream_bytes
        while remaining:
            chunk = handle.read(min(remaining, chunk_bytes))
            if not chunk:
                raise ValueError(f"truncated matrix stream for {entry.name!r}")
            remaining -= len(chunk)
            yield chunk
        return

    remaining = entry.top_payload_bytes
    decompressor = zlib.decompressobj()
    total_out = 0
    while remaining:
        pending = handle.read(min(remaining, chunk_bytes))
        if not pending:
            raise ValueError(f"truncated compressed stream for {entry.name!r}")
        remaining -= len(pending)
        while pending:
            output = decompressor.decompress(pending, chunk_bytes)
            pending = decompressor.unconsumed_tail
            if output:
                total_out += len(output)
                yield output
            elif pending:
                raise ValueError(f"zlib stream stalled for {entry.name!r}")
    output = decompressor.flush()
    if output:
        total_out += len(output)
        yield output
    if not decompressor.eof:
        raise ValueError(f"incomplete zlib stream for {entry.name!r}")
    if decompressor.unused_data or decompressor.unconsumed_tail:
        raise ValueError(f"unexpected trailing zlib data for {entry.name!r}")
    if total_out != entry.matrix_stream_bytes:
        raise ValueError(
            f"matrix stream length mismatch for {entry.name!r}: "
            f"{total_out} != {entry.matrix_stream_bytes}"
        )


def _subscripts(flat_index: int, shape: tuple[int, ...], order: str) -> tuple[int, ...]:
    if flat_index < 0 or flat_index >= math.prod(shape):
        raise IndexError(flat_index)
    values = [0] * len(shape)
    remainder = int(flat_index)
    axes = range(len(shape)) if order == "F" else range(len(shape) - 1, -1, -1)
    for axis in axes:
        values[axis] = remainder % shape[axis]
        remainder //= shape[axis]
    return tuple(values)


def _ravel_f(subscripts: Iterable[int], shape: tuple[int, ...]) -> int:
    index = 0
    stride = 1
    for value, dimension in zip(subscripts, shape):
        if value < 0 or value >= dimension:
            raise IndexError(tuple(subscripts))
        index += int(value) * stride
        stride *= dimension
    return index


def _ravel_c(subscripts: Iterable[int], shape: tuple[int, ...]) -> int:
    index = 0
    for value, dimension in zip(subscripts, shape):
        if value < 0 or value >= dimension:
            raise IndexError(tuple(subscripts))
        index = index * dimension + int(value)
    return index


def _sample_requests(
    shape: tuple[int, ...], *, sample_count: int, order: str, strategy: str
) -> list[dict[str, Any]]:
    elements = math.prod(shape)
    if elements == 0:
        return []
    if strategy == "even_linear":
        count = min(max(int(sample_count), 1), elements)
        if count == 1:
            requested_subscripts = [_subscripts(0, shape, "F" if order == "matlab_f" else "C")]
        else:
            requested = sorted(
                {
                    (position * (elements - 1)) // (count - 1)
                    for position in range(count)
                }
            )
            requested_subscripts = [
                _subscripts(index, shape, "F" if order == "matlab_f" else "C")
                for index in requested
            ]
    elif strategy == "grid_landmarks":
        if len(shape) > 3:
            raise ValueError("grid_landmarks supports at most three dimensions")
        center = tuple((dimension - 1) // 2 for dimension in shape)
        landmarks = {center}
        for mask in range(1 << len(shape)):
            landmarks.add(
                tuple(
                    shape[axis] - 1 if mask & (1 << axis) else 0
                    for axis in range(len(shape))
                )
            )
        for axis, dimension in enumerate(shape):
            for value in {0, (dimension - 1) // 2, dimension - 1}:
                point = list(center)
                point[axis] = value
                landmarks.add(tuple(point))
        requested_subscripts = sorted(
            landmarks,
            key=lambda subs: _ravel_f(subs, shape)
            if order == "matlab_f"
            else _ravel_c(subs, shape),
        )
    elif strategy == "measurement_rows":
        if len(shape) != 2 or shape[0] not in {1, 2, 3}:
            raise ValueError(
                "measurement_rows requires a (1|2|3, measurements) matrix"
            )
        measurements = shape[1]
        count = min(max(int(sample_count), 1), measurements)
        if count == 1:
            measurement_indices = [0]
        else:
            measurement_indices = sorted(
                {
                    (position * (measurements - 1)) // (count - 1)
                    for position in range(count)
                }
            )
        requested_subscripts = [
            (component, measurement)
            for measurement in measurement_indices
            for component in range(shape[0])
        ]
    else:
        raise ValueError(f"unsupported sample strategy: {strategy}")

    requests: list[dict[str, Any]] = []
    for subs in requested_subscripts:
        if order == "matlab_f":
            requested_index = _ravel_f(subs, shape)
            matlab_index = requested_index
        elif order == "author_c":
            requested_index = _ravel_c(subs, shape)
            matlab_index = _ravel_f(subs, shape)
        else:
            raise ValueError(f"unsupported sample order: {order}")
        requests.append(
            {
                "requested_flat_index": requested_index,
                "matlab_flat_index": matlab_index,
                "subscripts_zero_based": list(subs),
            }
        )
    return requests


def _json_number(value: int | float) -> int | float | str:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "+Inf" if value > 0 else "-Inf"
    return value


def _statistics(values: list[int | float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    nan_count = sum(math.isnan(float(value)) for value in values)
    inf_count = sum(math.isinf(float(value)) for value in values)
    if not finite:
        return {
            "count": len(values),
            "finite_count": 0,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "min": None,
            "max": None,
            "mean": None,
            "population_std": None,
        }
    mean = math.fsum(finite) / len(finite)
    variance = math.fsum((value - mean) ** 2 for value in finite) / len(finite)
    return {
        "count": len(values),
        "finite_count": len(finite),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "min": min(finite),
        "max": max(finite),
        "mean": mean,
        "population_std": math.sqrt(variance),
    }


def sample_entry(
    path: Path,
    entry: MatrixEntry,
    *,
    sample_count: int = 17,
    order: str = "matlab_f",
    strategy: str = "even_linear",
    full_threshold_bytes: int = DEFAULT_FULL_THRESHOLD_BYTES,
) -> dict[str, Any]:
    """Stream one matrix, validating its full zlib and numeric payload."""

    requests = _sample_requests(
        entry.shape,
        sample_count=sample_count,
        order=order,
        strategy=strategy,
    )
    collect_full = entry.numeric_payload_bytes <= full_threshold_bytes
    if collect_full:
        requested_indices: list[int] = []
        sample_mode = "full"
    else:
        requested_indices = sorted(
            {int(item["matlab_flat_index"]) for item in requests}
        )
        sample_mode = "deterministic_sample"

    full_payload = bytearray() if collect_full else None
    byte_targets = {index: bytearray() for index in requested_indices}
    payload_digest = hashlib.sha256()
    payload_seen = 0
    stream_offset = 0
    data_start = entry.numeric_payload_offset
    data_end = data_start + entry.numeric_payload_bytes

    with path.open("rb") as handle:
        for chunk in _iter_matrix_stream(handle, entry):
            chunk_start = stream_offset
            chunk_end = chunk_start + len(chunk)
            overlap_start = max(chunk_start, data_start)
            overlap_end = min(chunk_end, data_end)
            if overlap_start < overlap_end:
                payload = chunk[
                    overlap_start - chunk_start : overlap_end - chunk_start
                ]
                payload_digest.update(payload)
                payload_seen += len(payload)
                if full_payload is not None:
                    full_payload.extend(payload)
                else:
                    payload_base = overlap_start - data_start
                    payload_limit = overlap_end - data_start
                    first_index = payload_base // entry.item_bytes
                    last_index = (payload_limit - 1) // entry.item_bytes
                    for index in requested_indices:
                        if index < first_index:
                            continue
                        if index > last_index:
                            break
                        item_start = index * entry.item_bytes
                        item_end = item_start + entry.item_bytes
                        part_start = max(item_start, payload_base)
                        part_end = min(item_end, payload_limit)
                        if part_start < part_end:
                            byte_targets[index].extend(
                                payload[
                                    part_start - payload_base : part_end - payload_base
                                ]
                            )
            stream_offset = chunk_end

    if payload_seen != entry.numeric_payload_bytes:
        raise ValueError(
            f"numeric payload length mismatch for {entry.name!r}: "
            f"{payload_seen} != {entry.numeric_payload_bytes}"
        )
    _, _, format_code = NUMERIC_TYPES[entry.numeric_type_id]
    decoded: dict[int, int | float] = {}
    if full_payload is not None:
        if len(full_payload) != entry.numeric_payload_bytes:
            raise ValueError(f"full numeric payload for {entry.name!r} is incomplete")
        decoded = {
            index: value[0]
            for index, value in enumerate(
                struct.iter_unpack(
                    f"{entry.endian}{format_code}", memoryview(full_payload)
                )
            )
        }
    else:
        for index, raw in byte_targets.items():
            if len(raw) != entry.item_bytes:
                raise ValueError(f"sample {index} for {entry.name!r} is incomplete")
            decoded[index] = struct.unpack(f"{entry.endian}{format_code}", raw)[0]

    if sample_mode == "full":
        values = [decoded[index] for index in range(entry.elements)]
        samples = [
            {
                "requested_flat_index": index,
                "matlab_flat_index": index,
                "subscripts_zero_based": list(_subscripts(index, entry.shape, "F")),
                "value": _json_number(value),
            }
            for index, value in enumerate(values)
        ]
        statistics_scope = "exact_full_variable"
    else:
        samples = [
            {**request, "value": _json_number(decoded[request["matlab_flat_index"]])}
            for request in requests
        ]
        values = [decoded[request["matlab_flat_index"]] for request in requests]
        statistics_scope = "deterministic_samples_only"

    return {
        "name": entry.name,
        "shape": list(entry.shape),
        "matlab_class": entry.matlab_class,
        "numeric_type": entry.numeric_type,
        "elements": entry.elements,
        "compressed": entry.compressed,
        "compressed_payload_bytes": (
            entry.top_payload_bytes if entry.compressed else None
        ),
        "numeric_payload_bytes": entry.numeric_payload_bytes,
        "sample_mode": sample_mode,
        "sample_order": order,
        "sample_strategy": strategy,
        "statistics_scope": statistics_scope,
        "statistics": _statistics(values),
        "samples": samples,
        "numeric_payload_sha256": payload_digest.hexdigest(),
        "integrity_status": "FULL_SELECTED_STREAM_VALIDATED",
    }


def build_numeric_report(
    index: MatIndex,
    variable_names: Iterable[str],
    *,
    sample_count: int = 17,
    order: str = "matlab_f",
    strategy: str = "even_linear",
    full_threshold_bytes: int = DEFAULT_FULL_THRESHOLD_BYTES,
) -> dict[str, Any]:
    entries = {entry.name: entry for entry in index.entries}
    requested = list(dict.fromkeys(variable_names))
    missing = sorted(set(requested) - set(entries))
    if missing:
        raise ValueError(f"variables not found before MCOS subsystem: {missing}")
    sampled = [
        sample_entry(
            index.path,
            entries[name],
            sample_count=sample_count,
            order=order,
            strategy=strategy,
            full_threshold_bytes=full_threshold_bytes,
        )
        for name in requested
    ]
    return {
        "schema_version": "psu-hsof-selected-numeric-audit-1.0",
        "status": "SELECTED_NUMERIC_STREAMS_CONFORMANT",
        "evidence_scope": (
            "SELECTED_NUMERIC_VALUES_AND_FULL_SELECTED_STREAM_INTEGRITY_"
            "NO_RECONSTRUCTION"
        ),
        "source_file": index.path.name,
        "source_file_size_bytes": index.file_size_bytes,
        "endian": "little" if index.endian == "<" else "big",
        "subsystem_offset": index.subsystem_offset,
        "mcos_subsystem_present": index.mcos_subsystem_present,
        "indexed_named_numeric_variables": len(index.entries),
        "requested_variables": requested,
        "sample_count": sample_count,
        "sample_order": order,
        "sample_strategy": strategy,
        "variables": sampled,
        "limitations": [
            "large-variable statistics describe deterministic samples, not the full distribution",
            "the MCOS subsystem is intentionally not deserialized",
            "no NIRT training, held-out reprojection, uncertainty calibration or 3-D reconstruction is run",
            "physical units must be established from the official readme and preprocessing code, not inferred from values alone",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mat_path", type=Path)
    parser.add_argument("--variables", nargs="+", required=True)
    parser.add_argument("--sample-count", type=int, default=17)
    parser.add_argument(
        "--order", choices=("matlab_f", "author_c"), default="matlab_f"
    )
    parser.add_argument(
        "--strategy",
        choices=("even_linear", "grid_landmarks", "measurement_rows"),
        default="even_linear",
    )
    parser.add_argument(
        "--full-threshold-bytes", type=int, default=DEFAULT_FULL_THRESHOLD_BYTES
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.mat_path.is_file():
        parser.error(f"MAT file not found: {args.mat_path}")
    if args.sample_count < 1:
        parser.error("--sample-count must be positive")
    if args.full_threshold_bytes < 0:
        parser.error("--full-threshold-bytes cannot be negative")

    report = build_numeric_report(
        build_index(args.mat_path),
        args.variables,
        sample_count=args.sample_count,
        order=args.order,
        strategy=args.strategy,
        full_threshold_bytes=args.full_threshold_bytes,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
