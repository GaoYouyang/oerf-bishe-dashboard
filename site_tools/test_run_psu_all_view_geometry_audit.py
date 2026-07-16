from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from site_tools import run_psu_all_view_geometry_audit as runner
from site_tools.build_psu_view_shards import DEFAULT_VARIABLES
from site_tools.run_psu_all_view_geometry_audit import (
    _load_completed_view,
    _remove_partial_npy_files,
    aggregate_view_records,
)


def _record(view: int, scale: float, status: str) -> dict:
    return {
        "view_id_zero_based": view,
        "measurement_count": 100,
        "bundle_status": "VIEW_SHARD_BUNDLE_TRANSCODED_AND_SOURCE_STREAMS_VERIFIED",
        "mask_status": "CORRECTED_VIEW_MASK_SHARDS_BUILT_MECHANICAL_CONTRACT_PASS",
        "setup_status": status,
        "full_box_zero_count": int(scale > 0),
        "full_box_zero_fraction": 0.01 * scale,
        "box_miss_but_cone_nonzero_count": int(scale > 0),
        "box_miss_but_cone_nonzero_fraction": 0.02 * scale,
        "final_zero_length_count": int(scale > 0),
        "final_zero_length_fraction": 0.03 * scale,
        "cone_outside_ray_count": int(scale > 0),
        "cone_outside_ray_fraction": 0.04 * scale,
        "cone_length_weighted_outside_box_fraction": 0.05 * scale,
        "cone_segment_length_sum_m": 100.0 * scale,
        "cone_box_overlap_length_sum_m": 95.0 * scale,
        "active_unsafe_geometry_count": 0,
        "active_unsafe_geometry_fraction": 0.0,
        "inactive_unsafe_geometry_count": int(scale > 0),
        "inactive_unsafe_geometry_fraction": 0.06 * scale,
        "active_rms_magnitude_pixels": 0.2 * scale,
        "inactive_rms_magnitude_pixels": 0.1 * scale,
        "active_to_inactive_rms_ratio": 2.0,
    }


def test_aggregate_preserves_no_go_and_worst_view() -> None:
    report = aggregate_view_records(
        [
            _record(0, 1.0, "STREAMED_SETUP_DIAGNOSTIC_NO_GO"),
            _record(1, 2.0, "STREAMED_SETUP_DIAGNOSTIC_NO_GO"),
        ]
    )
    assert report["status"] == "ALL_VIEW_GEOMETRY_AUDIT_NO_GO"
    assert report["metric_summary"]["cone_outside_ray_fraction"]["maximum_view_id"] == 1
    assert report["prevalence"]["views_with_inactive_unsafe_geometry"] == 2
    assert report["decision"]["algorithm_success_claim"] == "LOCKED"
    assert report["pooled_geometry"][
        "cone_length_weighted_outside_box_fraction"
    ] == pytest.approx(0.05)


def test_aggregate_rejects_duplicate_views() -> None:
    with pytest.raises(ValueError, match="unique"):
        aggregate_view_records(
            [
                _record(0, 1.0, "STREAMED_SETUP_DIAGNOSTIC_NO_GO"),
                _record(0, 2.0, "STREAMED_SETUP_DIAGNOSTIC_NO_GO"),
            ]
        )


def test_aggregate_requires_exact_expected_view_set() -> None:
    with pytest.raises(ValueError, match="ordered range"):
        aggregate_view_records(
            [
                _record(0, 1.0, "STREAMED_SETUP_DIAGNOSTIC_NO_GO"),
                _record(2, 2.0, "STREAMED_SETUP_DIAGNOSTIC_NO_GO"),
            ],
            expected_view_count=2,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_aggregate_rejects_invalid_fraction_metrics(value: float) -> None:
    record = _record(0, 1.0, "STREAMED_SETUP_MECHANICAL_CONTRACT_PASS")
    record["full_box_zero_fraction"] = value
    with pytest.raises(ValueError, match="fraction|finite"):
        aggregate_view_records([record], expected_view_count=1)


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _npy_record(directory, name: str, shape: tuple[int, ...], dtype: str) -> dict:
    path = directory / f"{name}.npy"
    np.save(path, np.zeros(shape, dtype=dtype))
    return {
        "name": name,
        "filename": path.name,
        "shape": list(shape),
        "dtype": dtype,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def test_resume_validates_complete_view_and_rejects_same_size_corruption(tmp_path) -> None:
    view_dir = tmp_path / "view_00"
    bundle_dir = view_dir / "bundle"
    masks_dir = view_dir / "corrected_masks"
    setup_dir = view_dir / "setup"
    for directory in (bundle_dir, masks_dir, setup_dir):
        directory.mkdir(parents=True)
    mat_path = tmp_path / "fixture.mat"
    mat_path.write_bytes(b"fixture")
    geometry_source = tmp_path / "meas.py"
    geometry_source.write_text("def fixture():\n    return 1\n", encoding="utf-8")

    vector_names = {"c", "v", "Ruvecs", "Rvvecs", "Rxvecs", "Ryvecs"}
    bundle_variables = []
    for name in DEFAULT_VARIABLES:
        columns = 3 if name in vector_names else 1
        output = _npy_record(bundle_dir, name, (2, columns), "float32")
        item = {
            "name": name,
            "source_shape": [columns, 2],
            "source_numeric_sha256": "0" * 64,
            "shard_shape": output["shape"],
            "shard_dtype": output["dtype"],
            "shard_bytes": output["bytes"],
            "shard_sha256": output["sha256"],
            "peak_selected_buffer_bytes": 16,
        }
        bundle_variables.append(item)
        _write_json(
            bundle_dir / f"{name}.summary.json",
            {
                "source": {"variable": name},
                "output": {"sha256": output["sha256"]},
                "stream_audit": {"matrix_stream_verified": True},
            },
        )
    view_contract = {
        "view_id_zero_based": 0,
        "image_height": 1,
        "image_width": 2,
        "measurement_start": 0,
        "measurement_stop": 2,
        "measurement_count": 2,
    }
    bundle_manifest = {
        "schema_version": "psu-bost-view-shard-bundle-1.0",
        "status": "VIEW_SHARD_BUNDLE_TRANSCODED_AND_SOURCE_STREAMS_VERIFIED",
        "source": {"filename": mat_path.name},
        "view": {**view_contract, "view_count": 1},
        "variables": bundle_variables,
        "aggregate": {"all_source_streams_verified": True},
    }
    bundle_manifest_path = bundle_dir / "view_bundle_manifest.json"
    _write_json(bundle_manifest_path, bundle_manifest)

    mask_records = []
    for variable, local_index in (("amask_all", 0), ("imask_all", 1)):
        path = masks_dir / f"{variable}_zero_based.npy"
        np.save(path, np.array([local_index], dtype=np.int64))
        mask_records.append(
            {
                "variable": variable,
                "filename": path.name,
                "count": 1,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    masks_manifest = {
        "schema_version": "psu-bost-corrected-view-masks-1.0",
        "status": "CORRECTED_VIEW_MASK_SHARDS_BUILT_MECHANICAL_CONTRACT_PASS",
        "source": {"filename": mat_path.name},
        "view": view_contract,
        "mask_shards": mask_records,
    }
    _write_json(masks_dir / "corrected_view_masks_manifest.json", masks_manifest)

    setup_outputs = [
        _npy_record(setup_dir, "b_data", (2, 4), "float32"),
        _npy_record(setup_dir, "cam_data", (2, 18), "float32"),
        _npy_record(setup_dir, "ipf", (2, 3), "float32"),
        _npy_record(setup_dir, "epf", (2, 3), "float32"),
        _npy_record(setup_dir, "geometry_flags", (2,), "uint8"),
    ]
    setup_manifest = {
        "schema_version": "psu-bost-streamed-setup-1.0",
        "status": "STREAMED_SETUP_DIAGNOSTIC_NO_GO",
        "configuration": {"view_id_zero_based": 0, "rows": 2},
        "source": {
            "geometry_source_filename": geometry_source.name,
            "geometry_source_sha256": _sha256(geometry_source),
            "view_bundle_manifest_sha256": _sha256(bundle_manifest_path),
        },
        "outputs": setup_outputs,
        "corrected_mask_intersection": {"amask_all": {}, "imask_all": {}},
    }
    _write_json(setup_dir / "streamed_setup_manifest.json", setup_manifest)

    kwargs = {
        "view_dir": view_dir,
        "mat_path": mat_path,
        "geometry_source": geometry_source,
        "view_id": 0,
        "image_height": 1,
        "image_width": 2,
        "view_count": 1,
        "verify_sha256": True,
    }
    assert _load_completed_view(**kwargs) is not None

    with (setup_dir / "b_data.npy").open("r+b") as handle:
        handle.seek(-1, 2)
        final_byte = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([final_byte[0] ^ 1]))
    assert _load_completed_view(**kwargs) is None


def test_partial_cleanup_is_scoped_to_known_directories(tmp_path) -> None:
    view_dir = tmp_path / "view_00"
    partial = view_dir / "bundle" / ".v.npy.partial.npy"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"partial")
    keep = view_dir / "unrelated" / ".keep.partial.npy"
    keep.parent.mkdir()
    keep.write_bytes(b"keep")

    assert _remove_partial_npy_files(view_dir) == ["bundle/.v.npy.partial.npy"]
    assert not partial.exists()
    assert keep.exists()


def test_complete_matching_resume_performs_zero_aggregate_writes(
    tmp_path, monkeypatch
) -> None:
    bundle = {
        "status": "VIEW_SHARD_BUNDLE_TRANSCODED_AND_SOURCE_STREAMS_VERIFIED"
    }
    masks = {
        "status": "CORRECTED_VIEW_MASK_SHARDS_BUILT_MECHANICAL_CONTRACT_PASS",
        "deflection_semantics": {
            "amask_all": {
                "rms_magnitude_pixels": 0.2,
                "uncorrected_plus_one_comparison": {"vector_rmse_pixels": 0.01},
            },
            "imask_all": {
                "rms_magnitude_pixels": 0.1,
                "uncorrected_plus_one_comparison": {"vector_rmse_pixels": 0.02},
            },
        },
        "diagnostic": {"active_to_inactive_rms_magnitude_ratio": 2.0},
    }
    setup = {
        "status": "STREAMED_SETUP_DIAGNOSTIC_NO_GO",
        "diagnostics": {
            "ray_count": 2,
            "full_box_zero_length_count": 1,
            "full_box_miss_but_cone_nonzero_count": 1,
            "final_zero_length_count": 0,
            "cone_segment_partial_outside_box_count": 1,
            "cone_segment_no_box_overlap_count": 0,
            "cone_length_weighted_outside_box_fraction": 0.25,
            "cone_segment_length_sum_m": 4.0,
            "cone_box_overlap_length_sum_m": 3.0,
        },
        "corrected_mask_intersection": {
            "amask_all": {
                "count": 1,
                "unsafe_geometry_union_count": 0,
                "unsafe_geometry_union_fraction": 0.0,
            },
            "imask_all": {
                "count": 1,
                "unsafe_geometry_union_count": 1,
                "unsafe_geometry_union_fraction": 1.0,
            },
        },
    }
    monkeypatch.setattr(
        runner,
        "_load_completed_view",
        lambda **_kwargs: (bundle, masks, setup),
    )
    monkeypatch.setattr(
        runner,
        "_build_run_contract",
        lambda **_kwargs: ({"schema_version": "fixture"}, "contract-sha"),
    )
    record = runner._view_record(
        view_id=0, bundle=bundle, masks=masks, setup=setup
    )
    existing = runner.aggregate_view_records([record], expected_view_count=1)
    existing.update(
        {
            "execution_status": "COMPLETE",
            "scientific_verdict": "NO_GO",
            "run_contract_sha256": "contract-sha",
        }
    )
    output_root = tmp_path / "audit"
    output_root.mkdir()
    report_path = output_root / "all_view_geometry_audit.json"
    report_path.write_text(json.dumps(existing), encoding="utf-8")
    csv_path = output_root / "all_view_geometry_metrics.csv"
    csv_path.write_text("preserve-me\n", encoding="utf-8")
    before = (report_path.read_bytes(), csv_path.read_bytes())

    result = runner.run_all_view_audit(
        mat_path=tmp_path / "fixture.mat",
        geometry_source=tmp_path / "meas.py",
        output_root=output_root,
        image_height=1,
        image_width=2,
        view_count=1,
        resume=True,
    )

    assert result == existing
    assert (report_path.read_bytes(), csv_path.read_bytes()) == before
