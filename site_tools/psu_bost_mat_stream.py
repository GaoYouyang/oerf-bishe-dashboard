#!/usr/bin/env python3
"""Stream bounded measurement rows from numeric MATLAB v5 variables."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np

try:
    from .psu_bost_mat_sample import (
        MatIndex,
        MatrixEntry,
        _iter_matrix_stream,
        build_index,
    )
except ImportError:  # Direct script execution.
    from psu_bost_mat_sample import (  # type: ignore[no-redef]
        MatIndex,
        MatrixEntry,
        _iter_matrix_stream,
        build_index,
    )


OutputOrder = Literal["measurement_first", "components_first"]
NUMPY_CODES = {
    "int8": "i1",
    "uint8": "u1",
    "int16": "i2",
    "uint16": "u2",
    "int32": "i4",
    "uint32": "u4",
    "single": "f4",
    "double": "f8",
    "int64": "i8",
    "uint64": "u8",
}


@dataclass(frozen=True)
class MeasurementChunk:
    variable: str
    measurement_start: int
    measurement_stop: int
    values: np.ndarray


@dataclass
class StreamAudit:
    complete: bool = False
    matrix_stream_verified: bool = False
    source_numeric_payload_bytes: int = 0
    source_numeric_sha256: str | None = None
    selected_measurements: int = 0
    emitted_measurements: int = 0
    peak_selected_buffer_bytes: int = 0


def _entry_by_name(index: MatIndex, name: str) -> MatrixEntry:
    matches = [entry for entry in index.entries if entry.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected one numeric variable named {name!r}")
    return matches[0]


def _source_dtype(entry: MatrixEntry) -> np.dtype[Any]:
    try:
        code = NUMPY_CODES[entry.numeric_type]
    except KeyError as error:
        raise ValueError(f"unsupported numeric type {entry.numeric_type!r}") from error
    if entry.item_bytes == 1:
        return np.dtype(code)
    return np.dtype(f"{entry.endian}{code}")


def view_measurement_range(
    *, view_id: int, image_height: int, image_width: int, view_count: int
) -> tuple[int, int]:
    if image_height < 1 or image_width < 1 or view_count < 1:
        raise ValueError("image dimensions and view count must be positive")
    if view_id < 0 or view_id >= view_count:
        raise ValueError(f"view_id must be in [0, {view_count})")
    pixels = image_height * image_width
    return view_id * pixels, (view_id + 1) * pixels


class MeasurementStream:
    """Single-use iterator with an integrity audit available after exhaustion."""

    def __init__(
        self,
        *,
        path: Path,
        entry: MatrixEntry,
        measurement_start: int = 0,
        measurement_stop: int | None = None,
        chunk_measurements: int = 65_536,
        output_order: OutputOrder = "measurement_first",
        cast_dtype: str | np.dtype[Any] | None = "float32",
        stream_chunk_bytes: int = 1024 * 1024,
    ) -> None:
        if len(entry.shape) != 2:
            raise ValueError(
                f"measurement streaming requires a rank-2 variable, got {entry.shape}"
            )
        components, measurements = entry.shape
        stop = measurements if measurement_stop is None else int(measurement_stop)
        start = int(measurement_start)
        if start < 0 or stop < start or stop > measurements:
            raise ValueError(
                f"invalid measurement range [{start}, {stop}) for {measurements} rows"
            )
        if chunk_measurements < 1 or stream_chunk_bytes < 1:
            raise ValueError("chunk sizes must be positive")
        if output_order not in {"measurement_first", "components_first"}:
            raise ValueError(f"unsupported output order {output_order!r}")
        self.path = path
        self.entry = entry
        self.measurement_start = start
        self.measurement_stop = stop
        self.chunk_measurements = int(chunk_measurements)
        self.output_order = output_order
        self.cast_dtype = None if cast_dtype is None else np.dtype(cast_dtype)
        self.stream_chunk_bytes = int(stream_chunk_bytes)
        self.audit = StreamAudit(selected_measurements=stop - start)
        self._started = False
        self.components = int(components)

    @property
    def output_shape(self) -> tuple[int, int]:
        rows = self.measurement_stop - self.measurement_start
        if self.output_order == "measurement_first":
            return rows, self.components
        return self.components, rows

    @property
    def output_dtype(self) -> np.dtype[Any]:
        return self.cast_dtype or _source_dtype(self.entry)

    def __iter__(self) -> Iterator[MeasurementChunk]:
        if self._started:
            raise RuntimeError("MeasurementStream is single-use")
        self._started = True
        return self._iterate()

    def _iterate(self) -> Iterator[MeasurementChunk]:
        entry = self.entry
        row_bytes = self.components * entry.item_bytes
        selected_byte_start = entry.numeric_payload_offset + (
            self.measurement_start * row_bytes
        )
        selected_byte_stop = entry.numeric_payload_offset + (
            self.measurement_stop * row_bytes
        )
        payload_byte_start = entry.numeric_payload_offset
        payload_byte_stop = payload_byte_start + entry.numeric_payload_bytes
        source_dtype = _source_dtype(entry)
        selected_buffer = bytearray()
        numeric_hasher = hashlib.sha256()
        numeric_seen = 0
        stream_position = 0
        emitted_position = self.measurement_start

        with self.path.open("rb") as handle:
            for matrix_chunk in _iter_matrix_stream(
                handle, entry, chunk_bytes=self.stream_chunk_bytes
            ):
                chunk_start = stream_position
                chunk_stop = chunk_start + len(matrix_chunk)

                payload_start = max(chunk_start, payload_byte_start)
                payload_stop = min(chunk_stop, payload_byte_stop)
                if payload_start < payload_stop:
                    piece = matrix_chunk[
                        payload_start - chunk_start : payload_stop - chunk_start
                    ]
                    numeric_hasher.update(piece)
                    numeric_seen += len(piece)

                selected_start = max(chunk_start, selected_byte_start)
                selected_stop = min(chunk_stop, selected_byte_stop)
                if selected_start < selected_stop:
                    selected_buffer.extend(
                        matrix_chunk[
                            selected_start - chunk_start : selected_stop - chunk_start
                        ]
                    )
                    self.audit.peak_selected_buffer_bytes = max(
                        self.audit.peak_selected_buffer_bytes, len(selected_buffer)
                    )

                available_rows = len(selected_buffer) // row_bytes
                while available_rows >= self.chunk_measurements:
                    yield self._decode_rows(
                        selected_buffer,
                        row_count=self.chunk_measurements,
                        row_bytes=row_bytes,
                        source_dtype=source_dtype,
                        measurement_start=emitted_position,
                    )
                    emitted_position += self.chunk_measurements
                    available_rows = len(selected_buffer) // row_bytes

                stream_position = chunk_stop

        remaining_rows = len(selected_buffer) // row_bytes
        if remaining_rows:
            yield self._decode_rows(
                selected_buffer,
                row_count=remaining_rows,
                row_bytes=row_bytes,
                source_dtype=source_dtype,
                measurement_start=emitted_position,
            )
            emitted_position += remaining_rows

        if selected_buffer:
            raise ValueError(
                f"selected payload for {entry.name!r} ended on a partial measurement"
            )
        if numeric_seen != entry.numeric_payload_bytes:
            raise ValueError(
                f"numeric payload length mismatch for {entry.name!r}: "
                f"{numeric_seen} != {entry.numeric_payload_bytes}"
            )
        if emitted_position != self.measurement_stop:
            raise ValueError(
                f"emitted range ended at {emitted_position}, expected {self.measurement_stop}"
            )
        self.audit.source_numeric_payload_bytes = numeric_seen
        self.audit.source_numeric_sha256 = numeric_hasher.hexdigest()
        self.audit.emitted_measurements = emitted_position - self.measurement_start
        self.audit.matrix_stream_verified = True
        self.audit.complete = True

    def _decode_rows(
        self,
        selected_buffer: bytearray,
        *,
        row_count: int,
        row_bytes: int,
        source_dtype: np.dtype[Any],
        measurement_start: int,
    ) -> MeasurementChunk:
        byte_count = row_count * row_bytes
        raw = bytes(selected_buffer[:byte_count])
        del selected_buffer[:byte_count]
        values = np.frombuffer(raw, dtype=source_dtype).reshape(
            row_count, self.components
        )
        if self.cast_dtype is not None:
            values = values.astype(self.cast_dtype, copy=False)
        if self.output_order == "components_first":
            values = values.T
        values = np.ascontiguousarray(values)
        stop = measurement_start + row_count
        return MeasurementChunk(
            variable=self.entry.name,
            measurement_start=measurement_start,
            measurement_stop=stop,
            values=values,
        )


def open_measurement_stream(
    path: Path,
    variable: str,
    **kwargs: Any,
) -> MeasurementStream:
    index = build_index(path)
    return MeasurementStream(path=path, entry=_entry_by_name(index, variable), **kwargs)


def _sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def transcode_measurement_range(
    *,
    path: Path,
    variable: str,
    output_path: Path,
    measurement_start: int = 0,
    measurement_stop: int | None = None,
    chunk_measurements: int = 65_536,
    cast_dtype: str = "float32",
    stream_chunk_bytes: int = 1024 * 1024,
) -> dict[str, Any]:
    stream = open_measurement_stream(
        path,
        variable,
        measurement_start=measurement_start,
        measurement_stop=measurement_stop,
        chunk_measurements=chunk_measurements,
        output_order="measurement_first",
        cast_dtype=cast_dtype,
        stream_chunk_bytes=stream_chunk_bytes,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(f".{output_path.name}.partial.npy")
    if partial_path.exists():
        partial_path.unlink()
    mapped = np.lib.format.open_memmap(
        partial_path,
        mode="w+",
        dtype=stream.output_dtype,
        shape=stream.output_shape,
    )
    try:
        for chunk in stream:
            local_start = chunk.measurement_start - stream.measurement_start
            local_stop = chunk.measurement_stop - stream.measurement_start
            mapped[local_start:local_stop] = chunk.values
        mapped.flush()
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise
    del mapped
    if not stream.audit.complete:
        partial_path.unlink(missing_ok=True)
        raise RuntimeError("stream ended without full matrix integrity verification")
    os.replace(partial_path, output_path)
    report = {
        "schema_version": "psu-bost-measurement-shard-1.0",
        "status": "MEASUREMENT_RANGE_TRANSCODED_AND_FULL_SOURCE_STREAM_VERIFIED",
        "evidence_scope": "BYTE_ORDER_SHAPE_RANGE_AND_INTEGRITY_ONLY_NO_PHYSICAL_TRANSFORM_NO_NIRT",
        "source": {
            "filename": path.name,
            "variable": variable,
            "source_shape": list(stream.entry.shape),
            "source_numeric_type": stream.entry.numeric_type,
            "source_numeric_payload_bytes": stream.entry.numeric_payload_bytes,
            "source_numeric_sha256": stream.audit.source_numeric_sha256,
        },
        "selection": {
            "measurement_start": stream.measurement_start,
            "measurement_stop": stream.measurement_stop,
            "measurement_count": stream.measurement_stop
            - stream.measurement_start,
            "components": stream.components,
        },
        "output": {
            "filename": output_path.name,
            "shape": list(stream.output_shape),
            "dtype": str(stream.output_dtype),
            "bytes": output_path.stat().st_size,
            "sha256": _sha256_file(output_path),
        },
        "stream_audit": asdict(stream.audit),
        "limitations": [
            "integrity is guaranteed only after the iterator is fully exhausted",
            "compressed variables still require a sequential decompression pass",
            "the adapter only decodes and reorders numeric rows",
            "mask index bases, physical units, ray intersections, and NIRT remain downstream gates",
        ],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--variable", required=True)
    parser.add_argument("--output-npy", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--measurement-start", type=int, default=0)
    parser.add_argument("--measurement-stop", type=int)
    parser.add_argument("--view-id", type=int)
    parser.add_argument("--image-height", type=int)
    parser.add_argument("--image-width", type=int)
    parser.add_argument("--view-count", type=int)
    parser.add_argument("--chunk-measurements", type=int, default=65_536)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()

    start = args.measurement_start
    stop = args.measurement_stop
    view_fields = (args.image_height, args.image_width, args.view_count)
    if args.view_id is not None:
        if any(value is None for value in view_fields):
            parser.error("--view-id requires image height, image width, and view count")
        start, stop = view_measurement_range(
            view_id=args.view_id,
            image_height=args.image_height,
            image_width=args.image_width,
            view_count=args.view_count,
        )
    elif any(value is not None for value in view_fields):
        parser.error("view dimensions require --view-id")

    report = transcode_measurement_range(
        path=args.mat,
        variable=args.variable,
        output_path=args.output_npy,
        measurement_start=start,
        measurement_stop=stop,
        chunk_measurements=args.chunk_measurements,
        cast_dtype=args.dtype,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
