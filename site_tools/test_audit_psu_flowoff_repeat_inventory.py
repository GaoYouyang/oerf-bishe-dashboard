from __future__ import annotations

from site_tools.audit_psu_flowoff_repeat_inventory import (
    summarize_archive_index,
)


def test_archive_inventory_does_not_treat_angles_as_temporal_repeats() -> None:
    text = """
Archive:  sample-01.zip
  100  02-09-2026 18:39   root/def_data/DEF_IMGS/DEF_ROT_000/DEF_ROT_000_CAM_01.tiff
  100  02-09-2026 18:39   root/def_data/DEF_IMGS/DEF_ROT_010/DEF_ROT_010_CAM_01.tiff
  100  02-09-2026 18:35   root/cal_pkg/Data/raw_images/withoutCylinder/Cam1/Cam01_Card_120Deg_0.0mm_pin.tiff
  100  02-09-2026 18:35   root/cal_pkg/Data/raw_images/withoutCylinder/Cam1/Cam01_Card_130Deg_0.0mm_pin.tiff
  100  02-09-2026 18:39   root/def_data/CamAnglesAll.mat
"""
    summary = summarize_archive_index(text)
    assert summary["archive_count"] == 1
    assert summary["deflected_flow_on_tiff"][
        "unique_camera_rotation_conditions"
    ] == 2
    assert summary["deflected_flow_on_tiff"][
        "maximum_files_per_camera_rotation_condition"
    ] == 1
    assert summary["calibration_without_cylinder"][
        "raw_unique_named_conditions"
    ] == 2
    assert not summary["temporal_repeat_assessment"][
        "public_archive_authorizes_temporal_covariance_estimation"
    ]
