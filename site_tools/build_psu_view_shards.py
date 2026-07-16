#!/usr/bin/env python3
"""Build a bounded-memory, random-access shard bundle for one PSU camera view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

try:
    from .psu_bost_mat_stream import (
        transcode_measurement_range,
        view_measurement_range,
    )
except ImportError:  # Direct script execution.
    from psu_bost_mat_stream import (  # type: ignore[no-redef]
        transcode_measurement_range,
        view_measurement_range,
    )


DEFAULT_VARIABLES = (
    "c",
    "v",
    "epsu_all",
    "epsv_all",
    "Csys_all",
    "Ruvecs",
    "Rvvecs",
    "Rxvecs",
    "Ryvecs",
    "Rapvec",
    "Dfvec",
)


def build_view_bundle(
    *,
    mat_path: Path,
    output_dir: Path,
    view_id: int,
    image_height: int,
    image_width: int,
    view_count: int,
    variables: Sequence[str] = DEFAULT_VARIABLES,
    chunk_measurements: int = 65_536,
    cast_dtype: str = "float32",
) -> dict[str, Any]:
    if not variables or len(set(variables)) != len(variables):
        raise ValueError("variables must be a nonempty unique sequence")
    start, stop = view_measurement_range(
        view_id=view_id,
        image_height=image_height,
        image_width=image_width,
        view_count=view_count,
    )
    total_measurements = image_height * image_width * view_count
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for variable in variables:
        output_path = output_dir / f"{variable}.npy"
        report = transcode_measurement_range(
            path=mat_path,
            variable=variable,
            output_path=output_path,
            measurement_start=start,
            measurement_stop=stop,
            chunk_measurements=chunk_measurements,
            cast_dtype=cast_dtype,
        )
        if report["source"]["source_shape"][1] != total_measurements:
            raise ValueError(
                f"{variable!r} has {report['source']['source_shape'][1]} measurements, "
                f"expected {total_measurements} from the view geometry"
            )
        (output_dir / f"{variable}.summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        reports.append(report)

    manifest = {
        "schema_version": "psu-bost-view-shard-bundle-1.0",
        "status": "VIEW_SHARD_BUNDLE_TRANSCODED_AND_SOURCE_STREAMS_VERIFIED",
        "evidence_scope": "ONE_VIEW_NUMERIC_DECODE_ORDER_AND_INTEGRITY_NO_SETUP_ASSEMBLY_NO_NIRT",
        "source": {"filename": mat_path.name},
        "view": {
            "view_id_zero_based": view_id,
            "image_height": image_height,
            "image_width": image_width,
            "view_count": view_count,
            "measurement_start": start,
            "measurement_stop": stop,
            "measurement_count": stop - start,
        },
        "variables": [
            {
                "name": item["source"]["variable"],
                "source_shape": item["source"]["source_shape"],
                "source_numeric_sha256": item["source"]["source_numeric_sha256"],
                "shard_shape": item["output"]["shape"],
                "shard_dtype": item["output"]["dtype"],
                "shard_bytes": item["output"]["bytes"],
                "shard_sha256": item["output"]["sha256"],
                "peak_selected_buffer_bytes": item["stream_audit"][
                    "peak_selected_buffer_bytes"
                ],
            }
            for item in reports
        ],
        "aggregate": {
            "variable_count": len(reports),
            "shard_bytes": sum(item["output"]["bytes"] for item in reports),
            "all_source_streams_verified": all(
                item["stream_audit"]["matrix_stream_verified"] for item in reports
            ),
            "maximum_selected_buffer_bytes": max(
                item["stream_audit"]["peak_selected_buffer_bytes"]
                for item in reports
            ),
        },
        "limitations": [
            "the bundle contains one view and is not a tomographic reconstruction input by itself",
            "mask lists and X/Y/Z grids are audited separately",
            "ray intersections, aperture radii, cam_data assembly, sampling, and NIRT are not executed",
            "source payload hashes validate bytes, not physical calibration correctness",
        ],
        "decision": {
            "random_access_one_view_fields": "READY",
            "official_setup_equivalence": "NOT_YET_VERIFIED",
            "next_gate": "MASK_INDEX_BASE_AND_STREAMED_CAM_DATA_ASSEMBLY",
        },
    }
    (output_dir / "view_bundle_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view-id", type=int, required=True)
    parser.add_argument("--image-height", type=int, required=True)
    parser.add_argument("--image-width", type=int, required=True)
    parser.add_argument("--view-count", type=int, required=True)
    parser.add_argument("--variables", nargs="+", default=list(DEFAULT_VARIABLES))
    parser.add_argument("--chunk-measurements", type=int, default=65_536)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()
    manifest = build_view_bundle(
        mat_path=args.mat,
        output_dir=args.output_dir,
        view_id=args.view_id,
        image_height=args.image_height,
        image_width=args.image_width,
        view_count=args.view_count,
        variables=args.variables,
        chunk_measurements=args.chunk_measurements,
        cast_dtype=args.dtype,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
