from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from site_tools.psu_bost_streamed_setup import assemble_streamed_setup


def test_assembles_author_layout_without_loading_full_mat(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    rows = 4
    fields = {
        "c": np.zeros((rows, 3), dtype=np.float32),
        "v": np.tile([1.0, 0.0, 0.0], (rows, 1)).astype(np.float32),
        "Ruvecs": np.tile([0.0, 1.0, 0.0], (rows, 1)).astype(np.float32),
        "Rvvecs": np.tile([0.0, 0.0, 1.0], (rows, 1)).astype(np.float32),
        "Rxvecs": np.tile([0.0, 1.0, 0.0], (rows, 1)).astype(np.float32),
        "Ryvecs": np.tile([0.0, 0.0, 1.0], (rows, 1)).astype(np.float32),
        "epsu_all": np.arange(rows, dtype=np.float32)[:, None],
        "epsv_all": (10 + np.arange(rows, dtype=np.float32))[:, None],
        "Csys_all": np.ones((rows, 1), dtype=np.float32),
        "Rapvec": np.full((rows, 1), 0.01, dtype=np.float32),
        "Dfvec": np.full((rows, 1), 10.0, dtype=np.float32),
    }
    for name, values in fields.items():
        np.save(bundle / f"{name}.npy", values)
    (bundle / "view_bundle_manifest.json").write_text(
        json.dumps(
            {
                "status": "VIEW_SHARD_BUNDLE_TRANSCODED_AND_SOURCE_STREAMS_VERIFIED",
                "view": {"view_id_zero_based": 0},
            }
        ),
        encoding="utf-8",
    )
    geometry = tmp_path / "meas.py"
    geometry.write_text(
        """
def rayBoxIntersection(pix_points, vecs, min_bounds, max_bounds):
    n = pix_points.shape[1]
    return np.ones(n), 2*np.ones(n), np.ones(n)

def rayConeIntersection(pix_points, vecs, vertex, axis, angle):
    n = pix_points.shape[1]
    return np.full(n, np.nan), np.full(n, np.nan), np.zeros(n)
""",
        encoding="utf-8",
    )
    output = tmp_path / "setup"
    corrected_masks = tmp_path / "masks"
    corrected_masks.mkdir()
    np.save(corrected_masks / "amask_all_zero_based.npy", np.array([0, 2]))
    np.save(corrected_masks / "imask_all_zero_based.npy", np.array([1, 3]))
    report = assemble_streamed_setup(
        view_bundle_dir=bundle,
        geometry_source=geometry,
        output_dir=output,
        corrected_mask_dir=corrected_masks,
        chunk_rows=2,
    )
    b_data = np.load(output / "b_data.npy")
    cam_data = np.load(output / "cam_data.npy")
    ipf = np.load(output / "ipf.npy")
    epf = np.load(output / "epf.npy")
    assert report["status"] == "STREAMED_SETUP_MECHANICAL_CONTRACT_PASS"
    assert report["corrected_mask_intersection"]["amask_all"]["count"] == 2
    np.testing.assert_allclose(b_data[:, :2], np.column_stack((np.arange(4), 10 + np.arange(4))))
    np.testing.assert_allclose(b_data[:, 2:], 0)
    np.testing.assert_allclose(cam_data[:, 0], 1)
    np.testing.assert_allclose(cam_data[:, 1:4], fields["Ruvecs"])
    np.testing.assert_allclose(cam_data[:, 4:7], fields["Rvvecs"])
    np.testing.assert_allclose(ipf, fields["v"])
    np.testing.assert_allclose(epf, 2 * fields["v"])
    assert str(tmp_path) not in str(report)
