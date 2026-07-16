#!/usr/bin/env python3
"""Inspect the PSU HSOF_9CAM_RT MAT schema without loading its 5.23 GB arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


EXPECTED_GRID_SHAPE = (400, 350, 350)
REQUIRED_FIELDS = (
    "c",
    "v",
    "epsu_all",
    "epsv_all",
    "siz",
    "Ruvecs",
    "Rvvecs",
    "Csys_all",
    "X",
    "Y",
    "Z",
    "rho_inf",
    "D_inf",
    "c_inf",
    "gamma_inf",
    "T_inf",
    "amask_all",
    "imask_all",
    "bmask_all",
    "Itn_all",
    "Iu_all",
    "Iv_all",
    "Rxvecs",
    "Ryvecs",
    "Rapvec",
    "Dfvec",
)

MATLAB_ITEM_BYTES = {
    "double": 8,
    "single": 4,
    "int64": 8,
    "uint64": 8,
    "int32": 4,
    "uint32": 4,
    "int16": 2,
    "uint16": 2,
    "int8": 1,
    "uint8": 1,
    "logical": 1,
    "char": 2,
}

MAT_V5_HEADER_BYTES = 128
MI_MATRIX = 14
MI_COMPRESSED = 15
MATLAB_CLASSES = {
    1: "cell",
    2: "struct",
    3: "object",
    4: "char",
    5: "sparse",
    6: "double",
    7: "single",
    8: "int8",
    9: "uint8",
    10: "int16",
    11: "uint16",
    12: "int32",
    13: "uint32",
    14: "int64",
    15: "uint64",
    16: "function",
    17: "opaque",
}


@dataclass(frozen=True)
class MatVariable:
    name: str
    shape: tuple[int, ...]
    matlab_class: str

    @property
    def elements(self) -> int:
        value = 1
        for dimension in self.shape:
            value *= int(dimension)
        return value

    @property
    def estimated_bytes(self) -> int | None:
        item_bytes = MATLAB_ITEM_BYTES.get(self.matlab_class)
        return None if item_bytes is None else self.elements * item_bytes


def _padded_bytes(size: int) -> int:
    return ((int(size) + 7) // 8) * 8


def _data_element_tag(
    buffer: bytes, offset: int, endian: str
) -> tuple[int, int, int, int]:
    """Return type, payload bytes, payload offset and padded element bytes."""

    if offset < 0 or offset + 4 > len(buffer):
        raise ValueError("truncated MAT data-element tag")
    small_type, small_bytes = struct.unpack_from(f"{endian}HH", buffer, offset)
    if 1 <= small_type <= 18 and 1 <= small_bytes <= 4:
        if offset + 8 > len(buffer):
            raise ValueError("truncated small MAT data element")
        return small_type, small_bytes, offset + 4, 8
    if offset + 8 > len(buffer):
        raise ValueError("truncated regular MAT data-element tag")
    data_type, payload_bytes = struct.unpack_from(f"{endian}II", buffer, offset)
    return data_type, payload_bytes, offset + 8, 8 + _padded_bytes(payload_bytes)


def _matrix_header_from_prefix(
    prefix: bytes, endian: str
) -> MatVariable | None:
    data_type, _, position, _ = _data_element_tag(prefix, 0, endian)
    if data_type != MI_MATRIX:
        raise ValueError(f"compressed payload starts with MAT type {data_type}, not matrix")

    _, flags_bytes, flags_start, flags_total = _data_element_tag(
        prefix, position, endian
    )
    if flags_bytes < 4 or flags_start + flags_bytes > len(prefix):
        raise ValueError("truncated MAT array flags")
    flags = struct.unpack_from(f"{endian}I", prefix, flags_start)[0]
    matlab_class_id = flags & 0xFF
    if matlab_class_id == 17:
        return None
    position += flags_total

    _, dimensions_bytes, dimensions_start, dimensions_total = _data_element_tag(
        prefix, position, endian
    )
    if dimensions_bytes % 4 or dimensions_start + dimensions_bytes > len(prefix):
        raise ValueError("invalid MAT dimensions element")
    dimensions = struct.unpack_from(
        f"{endian}{dimensions_bytes // 4}i", prefix, dimensions_start
    )
    position += dimensions_total

    _, name_bytes, name_start, _ = _data_element_tag(prefix, position, endian)
    if name_start + name_bytes > len(prefix):
        raise ValueError("truncated MAT variable name")
    name = prefix[name_start : name_start + name_bytes].decode("latin1")
    if not name:
        return None
    return MatVariable(
        name=name,
        shape=tuple(int(value) for value in dimensions),
        matlab_class=MATLAB_CLASSES.get(matlab_class_id, f"class_{matlab_class_id}"),
    )


def _compressed_prefix(
    handle: Any, *, payload_start: int, payload_bytes: int, limit: int = 65536
) -> bytes:
    handle.seek(payload_start)
    decompressor = zlib.decompressobj()
    output = bytearray()
    remaining = int(payload_bytes)
    while remaining > 0 and len(output) < limit:
        chunk = handle.read(min(remaining, 65536))
        if not chunk:
            raise ValueError("truncated compressed MAT payload")
        remaining -= len(chunk)
        output.extend(decompressor.decompress(chunk, limit - len(output)))
    if not output:
        raise ValueError("empty compressed MAT payload")
    return bytes(output)


def inspect_mat(path: Path) -> list[MatVariable]:
    """Stream MATLAB v5 variable headers without loading the large arrays.

    SciPy ``whosmat`` currently fails on the official file's compressed MCOS
    subsystem.  MATLAB stores the subsystem offset in the v5 header, so this
    scanner inventories named arrays before that boundary and ignores opaque
    object metadata.
    """

    file_size = path.stat().st_size
    with path.open("rb") as handle:
        header = handle.read(MAT_V5_HEADER_BYTES)
        if len(header) != MAT_V5_HEADER_BYTES or not header.startswith(
            b"MATLAB 5.0 MAT-file"
        ):
            raise ValueError("only MATLAB v5 MAT files are supported")
        endian_marker = header[126:128]
        if endian_marker == b"IM":
            endian = "<"
        elif endian_marker == b"MI":
            endian = ">"
        else:
            raise ValueError("invalid MATLAB v5 endian marker")
        subsystem_offset = struct.unpack(f"{endian}Q", header[116:124])[0]
        boundary = (
            int(subsystem_offset)
            if MAT_V5_HEADER_BYTES <= subsystem_offset <= file_size
            else file_size
        )

        variables: list[MatVariable] = []
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
                raise ValueError(f"MAT element at byte {position} crosses subsystem boundary")
            if data_type == MI_COMPRESSED:
                prefix = _compressed_prefix(
                    handle,
                    payload_start=payload_start,
                    payload_bytes=payload_bytes,
                )
                variable = _matrix_header_from_prefix(prefix, endian)
                position = payload_end
            elif data_type == MI_MATRIX:
                handle.seek(position)
                prefix = handle.read(min(8 + int(payload_bytes), 65536))
                variable = _matrix_header_from_prefix(prefix, endian)
                position = payload_start + _padded_bytes(payload_bytes)
            else:
                raise ValueError(
                    f"unsupported top-level MAT type {data_type} at byte {position}"
                )
            if variable is not None:
                variables.append(variable)
        if position != boundary:
            raise ValueError(f"MAT scan ended at {position}, expected {boundary}")
    return variables


def _measurement_width(variable: MatVariable) -> int | None:
    if len(variable.shape) != 2:
        return None
    rows, columns = variable.shape
    if rows in {1, 2, 3}:
        return columns
    if columns in {1, 2, 3}:
        return rows
    return None


def build_report(
    variables: Iterable[MatVariable], *, file_size_bytes: int, path_label: str
) -> dict[str, Any]:
    values = list(variables)
    by_name = {variable.name: variable for variable in values}
    missing = sorted(set(REQUIRED_FIELDS) - set(by_name))
    duplicate_names = sorted(
        name for name in {item.name for item in values} if sum(v.name == name for v in values) > 1
    )

    checks: dict[str, Any] = {
        "required_fields_present": not missing,
        "variable_names_unique": not duplicate_names,
        "xyz_shapes_equal": all(
            name in by_name for name in ("X", "Y", "Z")
        )
        and len({by_name[name].shape for name in ("X", "Y", "Z")}) == 1,
        "grid_shape_matches_author_script": all(
            name in by_name and by_name[name].shape == EXPECTED_GRID_SHAPE
            for name in ("X", "Y", "Z")
        ),
    }

    ray_names = (
        "c",
        "v",
        "epsu_all",
        "epsv_all",
        "Ruvecs",
        "Rvvecs",
        "Rxvecs",
        "Ryvecs",
        "Rapvec",
        "Dfvec",
        "Csys_all",
    )
    ray_widths = {
        name: _measurement_width(by_name[name])
        for name in ray_names
        if name in by_name
    }
    known_widths = {value for value in ray_widths.values() if value is not None}
    checks["ray_field_widths_equal"] = (
        len(ray_widths) == len(ray_names) and len(known_widths) == 1
    )
    checks["siz_has_three_entries"] = (
        "siz" in by_name and by_name["siz"].elements == 3
    )

    estimated_known_bytes = sum(
        value.estimated_bytes or 0 for value in values
    )
    unknown_size_fields = sorted(
        value.name for value in values if value.estimated_bytes is None
    )
    status = "SCHEMA_CONFORMANT" if all(checks.values()) else "SCHEMA_REVIEW_REQUIRED"
    return {
        "schema_version": "psu-hsof-9cam-mat-audit-1.0",
        "status": status,
        "evidence_scope": "HEADER_AND_SHAPE_AUDIT_ONLY_NO_RECONSTRUCTION",
        "path_label": path_label,
        "file_size_bytes": int(file_size_bytes),
        "variable_count": len(values),
        "missing_required_fields": missing,
        "duplicate_names": duplicate_names,
        "checks": checks,
        "ray_field_widths": ray_widths,
        "estimated_uncompressed_payload_bytes": estimated_known_bytes,
        "unknown_size_fields": unknown_size_fields,
        "variables": [
            {
                **asdict(variable),
                "elements": variable.elements,
                "estimated_bytes": variable.estimated_bytes,
            }
            for variable in sorted(values, key=lambda item: item.name)
        ],
        "author_loader_conventions": {
            "source": "pyscripts/setup.py",
            "deflection": "epsu_all.T and epsv_all.T",
            "ray_origins": "c",
            "ray_directions": "v",
            "projection_vectors": "Ruvecs.T and Rvvecs.T",
            "aperture_fields": "Rxvecs.T, Ryvecs.T, Rapvec.T, Dfvec.T",
            "grid_flattening": "X/Y/Z flatten(order='C') in the author Python loader",
        },
        "limitations": [
            "the native MATLAB v5 scan reads headers but does not validate numeric values or physical units",
            "no author reconstruction, held-out reprojection or adjoint test is run",
            "the MAT file and this generated report remain in the private local library",
        ],
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mat_path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sha256", action="store_true")
    args = parser.parse_args()
    if not args.mat_path.is_file():
        parser.error(f"MAT file not found: {args.mat_path}")

    report = build_report(
        inspect_mat(args.mat_path),
        file_size_bytes=args.mat_path.stat().st_size,
        path_label=args.mat_path.name,
    )
    report["inventory_parser"] = "native-mat-v5-streaming-header-scan"
    if args.sha256:
        report["sha256"] = sha256_file(args.mat_path)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["status"] == "SCHEMA_CONFORMANT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
