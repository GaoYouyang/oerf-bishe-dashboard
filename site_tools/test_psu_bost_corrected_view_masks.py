from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io

from site_tools.psu_bost_corrected_view_masks import build_corrected_view_masks


def test_builds_zero_based_view_masks_and_uses_real_rows(tmp_path: Path) -> None:
    mat_path = tmp_path / "masks.mat"
    scipy.io.savemat(
        mat_path,
        {
            "amask_all": np.array([[1, 3, 6]], dtype=np.int32),
            "imask_all": np.array([[2, 4, 8]], dtype=np.int32),
        },
        do_compression=True,
    )
    view_bundle = tmp_path / "view0"
    view_bundle.mkdir()
    np.save(view_bundle / "epsu_all.npy", np.array([[1.0], [0.0], [2.0], [0.0]]))
    np.save(view_bundle / "epsv_all.npy", np.zeros((4, 1)))
    output = tmp_path / "corrected"
    report = build_corrected_view_masks(
        mat_path=mat_path,
        view_bundle_dir=view_bundle,
        output_dir=output,
        view_id=0,
        image_height=2,
        image_width=2,
        view_count=2,
        chunk_measurements=1,
    )
    np.testing.assert_array_equal(
        np.load(output / "amask_all_zero_based.npy"), [0, 2]
    )
    np.testing.assert_array_equal(
        np.load(output / "imask_all_zero_based.npy"), [1, 3]
    )
    assert report["decision"]["corrected_indices_mechanically_valid"] is True
    assert report["decision"]["physical_mask_semantics"] == "REVIEW_REQUIRED"
    assert report["diagnostic"]["active_to_inactive_rms_magnitude_ratio"] is None
    assert str(tmp_path) not in str(report)
