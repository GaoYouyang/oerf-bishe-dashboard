import hashlib

import numpy as np
import pytest
from scipy.io import savemat

from site_tools.psu_bost_mat_sample import (
    _sample_requests,
    build_index,
    build_numeric_report,
    sample_entry,
)


def _fixture(tmp_path):
    path = tmp_path / "numeric.mat"
    tiny = np.arange(6, dtype=np.float64).reshape((2, 3), order="F")
    grid = np.arange(24, dtype=np.float32).reshape((2, 3, 4), order="F")
    savemat(path, {"tiny": tiny, "grid": grid}, do_compression=True)
    return path, tiny, grid


def test_index_records_numeric_layout_and_compressed_offsets(tmp_path) -> None:
    path, _, _ = _fixture(tmp_path)
    index = build_index(path)
    entries = {entry.name: entry for entry in index.entries}
    assert index.endian == "<"
    assert entries["tiny"].shape == (2, 3)
    assert entries["tiny"].numeric_type == "double"
    assert entries["tiny"].numeric_payload_bytes == 48
    assert entries["tiny"].compressed is True
    assert entries["tiny"].top_payload_offset > 128


def test_full_small_variable_matches_matlab_linear_order(tmp_path) -> None:
    path, tiny, _ = _fixture(tmp_path)
    entry = {item.name: item for item in build_index(path).entries}["tiny"]
    result = sample_entry(path, entry, full_threshold_bytes=1024)
    values = [item["value"] for item in result["samples"]]
    assert values == tiny.ravel(order="F").tolist()
    assert result["statistics_scope"] == "exact_full_variable"
    assert result["statistics"]["min"] == 0.0
    assert result["statistics"]["max"] == 5.0
    assert result["integrity_status"] == "FULL_SELECTED_STREAM_VALIDATED"
    assert len(result["numeric_payload_sha256"]) == 64


def test_author_c_requests_preserve_subscripts_and_payload_mapping(tmp_path) -> None:
    path, _, grid = _fixture(tmp_path)
    entry = {item.name: item for item in build_index(path).entries}["grid"]
    result = sample_entry(
        path,
        entry,
        sample_count=24,
        order="author_c",
        strategy="even_linear",
        full_threshold_bytes=0,
    )
    values = [item["value"] for item in result["samples"]]
    assert values == grid.ravel(order="C").tolist()
    assert result["statistics_scope"] == "deterministic_samples_only"
    assert result["samples"][1]["subscripts_zero_based"] == [0, 0, 1]
    assert result["samples"][1]["matlab_flat_index"] == 6


def test_payload_digest_covers_exact_numeric_bytes(tmp_path) -> None:
    path, tiny, _ = _fixture(tmp_path)
    entry = {item.name: item for item in build_index(path).entries}["tiny"]
    result = sample_entry(path, entry, full_threshold_bytes=1024)
    expected = hashlib.sha256(tiny.tobytes(order="F")).hexdigest()
    assert result["numeric_payload_sha256"] == expected


def test_report_rejects_missing_variable(tmp_path) -> None:
    path, _, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="variables not found"):
        build_numeric_report(build_index(path), ["not_here"])


def test_even_sample_requests_include_endpoints() -> None:
    requests = _sample_requests(
        (2, 3, 4), sample_count=5, order="matlab_f", strategy="even_linear"
    )
    assert [item["requested_flat_index"] for item in requests] == [0, 5, 11, 17, 23]
    assert requests[0]["subscripts_zero_based"] == [0, 0, 0]
    assert requests[-1]["subscripts_zero_based"] == [1, 2, 3]


def test_grid_landmarks_cover_corners_center_and_each_axis() -> None:
    requests = _sample_requests(
        (5, 7, 9),
        sample_count=1,
        order="author_c",
        strategy="grid_landmarks",
    )
    subscripts = {tuple(item["subscripts_zero_based"]) for item in requests}
    assert (0, 0, 0) in subscripts
    assert (4, 6, 8) in subscripts
    assert (2, 3, 4) in subscripts
    assert (0, 3, 4) in subscripts
    assert (2, 0, 4) in subscripts
    assert (2, 3, 0) in subscripts
    assert len(subscripts) == 15


def test_measurement_rows_keep_vector_components_together() -> None:
    requests = _sample_requests(
        (3, 10),
        sample_count=3,
        order="matlab_f",
        strategy="measurement_rows",
    )
    subscripts = [tuple(item["subscripts_zero_based"]) for item in requests]
    assert subscripts == [
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 4),
        (1, 4),
        (2, 4),
        (0, 9),
        (1, 9),
        (2, 9),
    ]


def test_complex_numeric_variable_is_explicitly_rejected(tmp_path) -> None:
    path = tmp_path / "complex.mat"
    savemat(path, {"z": np.array([[1 + 2j]])}, do_compression=True)
    with pytest.raises(ValueError, match="complex MAT variable"):
        build_index(path)
