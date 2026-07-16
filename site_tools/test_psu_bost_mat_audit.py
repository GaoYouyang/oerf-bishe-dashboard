import numpy as np

from site_tools.psu_bost_mat_audit import (
    EXPECTED_GRID_SHAPE,
    REQUIRED_FIELDS,
    MatVariable,
    build_report,
    inspect_mat,
)


def _conformant_variables() -> list[MatVariable]:
    variables = []
    ray_vectors = {"c", "v", "Ruvecs", "Rvvecs", "Rxvecs", "Ryvecs"}
    ray_scalars = {"epsu_all", "epsv_all", "Rapvec", "Dfvec", "Csys_all"}
    for name in REQUIRED_FIELDS:
        if name in {"X", "Y", "Z"}:
            variables.append(MatVariable(name, EXPECTED_GRID_SHAPE, "single"))
        elif name in ray_vectors:
            variables.append(MatVariable(name, (3, 120), "single"))
        elif name in ray_scalars:
            variables.append(MatVariable(name, (1, 120), "double"))
        elif name == "siz":
            variables.append(MatVariable(name, (1, 3), "double"))
        else:
            variables.append(MatVariable(name, (1, 1), "double"))
    return variables


def test_conformant_header_inventory_passes() -> None:
    report = build_report(
        _conformant_variables(), file_size_bytes=12345, path_label="HSOF_9CAM_RT.mat"
    )
    assert report["status"] == "SCHEMA_CONFORMANT"
    assert all(report["checks"].values())
    assert report["ray_field_widths"]["c"] == 120
    assert report["missing_required_fields"] == []


def test_missing_field_and_wrong_grid_require_review() -> None:
    variables = [
        item
        for item in _conformant_variables()
        if item.name != "epsv_all"
    ]
    variables = [
        MatVariable(item.name, (399, 350, 350), item.matlab_class)
        if item.name == "X"
        else item
        for item in variables
    ]
    report = build_report(
        variables, file_size_bytes=12345, path_label="broken.mat"
    )
    assert report["status"] == "SCHEMA_REVIEW_REQUIRED"
    assert report["missing_required_fields"] == ["epsv_all"]
    assert report["checks"]["xyz_shapes_equal"] is False
    assert report["checks"]["grid_shape_matches_author_script"] is False


def test_payload_estimate_uses_matlab_class_width() -> None:
    variable = MatVariable("sample", (2, 3, 4), "single")
    assert variable.elements == 24
    assert variable.estimated_bytes == 96


def test_streaming_inventory_reads_compressed_mat_v5(tmp_path) -> None:
    from scipy.io import savemat

    path = tmp_path / "compressed.mat"
    savemat(
        path,
        {
            "single_grid": np.zeros((2, 3, 4), dtype=np.float32),
            "ray_values": np.ones((3, 7), dtype=np.float64),
        },
        do_compression=True,
    )
    variables = {item.name: item for item in inspect_mat(path)}
    assert variables["single_grid"] == MatVariable(
        "single_grid", (2, 3, 4), "single"
    )
    assert variables["ray_values"] == MatVariable("ray_values", (3, 7), "double")
