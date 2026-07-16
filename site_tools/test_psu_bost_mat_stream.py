from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io

from site_tools.build_psu_view_shards import build_view_bundle
from site_tools.psu_bost_mat_stream import (
    open_measurement_stream,
    transcode_measurement_range,
    view_measurement_range,
)


def _fixture(path: Path, *, compressed: bool = True) -> np.ndarray:
    values = np.vstack(
        [
            np.arange(10, dtype=np.float64),
            100 + np.arange(10, dtype=np.float64),
            200 + np.arange(10, dtype=np.float64),
        ]
    )
    scipy.io.savemat(path, {"v": values}, do_compression=compressed)
    return values


@pytest.mark.parametrize("compressed", [True, False])
def test_streams_measurement_rows_in_author_orientation(
    tmp_path: Path, compressed: bool
) -> None:
    path = tmp_path / "tiny.mat"
    expected = _fixture(path, compressed=compressed)
    stream = open_measurement_stream(
        path,
        "v",
        chunk_measurements=4,
        cast_dtype="float32",
        stream_chunk_bytes=31,
    )
    chunks = list(stream)
    assert [chunk.values.shape for chunk in chunks] == [(4, 3), (4, 3), (2, 3)]
    np.testing.assert_allclose(np.concatenate([chunk.values for chunk in chunks]), expected.T)
    assert stream.audit.complete is True
    assert stream.audit.matrix_stream_verified is True
    assert stream.audit.emitted_measurements == 10
    assert stream.audit.source_numeric_payload_bytes == expected.nbytes
    assert len(stream.audit.source_numeric_sha256 or "") == 64


def test_range_and_components_first_preserve_original_indices(tmp_path: Path) -> None:
    path = tmp_path / "tiny.mat"
    expected = _fixture(path)
    stream = open_measurement_stream(
        path,
        "v",
        measurement_start=2,
        measurement_stop=9,
        chunk_measurements=3,
        output_order="components_first",
        cast_dtype=None,
        stream_chunk_bytes=29,
    )
    chunks = list(stream)
    assert [(item.measurement_start, item.measurement_stop) for item in chunks] == [
        (2, 5),
        (5, 8),
        (8, 9),
    ]
    np.testing.assert_allclose(
        np.concatenate([item.values for item in chunks], axis=1), expected[:, 2:9]
    )
    assert stream.audit.complete is True


def test_stream_is_single_use_and_validates_ranges(tmp_path: Path) -> None:
    path = tmp_path / "tiny.mat"
    _fixture(path)
    stream = open_measurement_stream(path, "v", measurement_stop=1)
    list(stream)
    with pytest.raises(RuntimeError, match="single-use"):
        list(stream)
    with pytest.raises(ValueError, match="invalid measurement range"):
        open_measurement_stream(path, "v", measurement_start=8, measurement_stop=11)


def test_transcode_writes_bounded_npy_shard(tmp_path: Path) -> None:
    path = tmp_path / "tiny.mat"
    expected = _fixture(path)
    output = tmp_path / "view.npy"
    report = transcode_measurement_range(
        path=path,
        variable="v",
        output_path=output,
        measurement_start=1,
        measurement_stop=7,
        chunk_measurements=2,
        cast_dtype="float32",
        stream_chunk_bytes=37,
    )
    saved = np.load(output, mmap_mode="r")
    np.testing.assert_allclose(saved, expected[:, 1:7].T)
    assert report["status"].endswith("FULL_SOURCE_STREAM_VERIFIED")
    assert report["output"]["shape"] == [6, 3]
    assert report["stream_audit"]["complete"] is True
    assert len(report["output"]["sha256"]) == 64
    assert str(tmp_path) not in str(report)


def test_view_ranges_are_contiguous_and_zero_based() -> None:
    assert view_measurement_range(
        view_id=2, image_height=3, image_width=4, view_count=5
    ) == (24, 36)
    with pytest.raises(ValueError, match="view_id"):
        view_measurement_range(
            view_id=5, image_height=3, image_width=4, view_count=5
        )


def test_builds_reproducible_one_view_bundle(tmp_path: Path) -> None:
    mat_path = tmp_path / "bundle.mat"
    vector = np.arange(3 * 8, dtype=np.float32).reshape(3, 8, order="F")
    scalar = np.arange(8, dtype=np.float64).reshape(1, 8)
    scipy.io.savemat(
        mat_path,
        {"c": vector, "epsu_all": scalar},
        do_compression=True,
    )
    output_dir = tmp_path / "view1"
    manifest = build_view_bundle(
        mat_path=mat_path,
        output_dir=output_dir,
        view_id=1,
        image_height=2,
        image_width=2,
        view_count=2,
        variables=("c", "epsu_all"),
        chunk_measurements=3,
    )
    np.testing.assert_allclose(np.load(output_dir / "c.npy"), vector[:, 4:8].T)
    np.testing.assert_allclose(
        np.load(output_dir / "epsu_all.npy"), scalar[:, 4:8].T
    )
    assert manifest["aggregate"]["all_source_streams_verified"] is True
    assert manifest["aggregate"]["variable_count"] == 2
    assert manifest["decision"]["official_setup_equivalence"] == "NOT_YET_VERIFIED"
