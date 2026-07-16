from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io

from site_tools.psu_bost_mask_index_audit import build_mask_index_report


def test_confirms_matlab_find_to_tensorflow_gather_mismatch(tmp_path: Path) -> None:
    mat_path = tmp_path / "masks.mat"
    scipy.io.savemat(
        mat_path,
        {
            "amask_all": np.array([[1, 3]], dtype=np.int32),
            "imask_all": np.array([[2, 4]], dtype=np.int32),
        },
        do_compression=True,
    )
    producer = tmp_path / "producer.m"
    producer.write_text(
        "amask_all = find(amask_all); imask_all = find(imask_all);\n",
        encoding="utf-8",
    )
    setup = tmp_path / "setup.py"
    setup.write_text(
        "amask = data['amask_all'].T\nimask = data['imask_all'].T\n",
        encoding="utf-8",
    )
    sample = tmp_path / "sample.py"
    sample.write_text(
        "index = tf.gather(pdict['masks'][2], draw)[:, 0]\n",
        encoding="utf-8",
    )
    report = build_mask_index_report(
        mat_path=mat_path,
        measurement_count=4,
        producer_source=producer,
        setup_source=setup,
        sample_source=sample,
        chunk_measurements=1,
    )
    assert report["status"] == "MASK_INDEX_BASE_MISMATCH_CONFIRMED"
    assert report["masks"][1]["maximum"] == 4
    assert report["masks"][1]["valid_as_zero_based"] is False
    assert report["decision"]["official_mask_indices_safe_for_python_gather"] == "NO_GO"
    assert str(tmp_path) not in str(report)
