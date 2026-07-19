from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from site_tools.run_psu_rotation40_multiresolution_diagnosis import (
    correction_alignment,
    mechanism_decision,
    resize_volume,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "demo_t16_operator/configs/psu_rotation40_multiresolution_diagnosis_development_v1.json"
)


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_frozen_postopen_config_is_valid() -> None:
    validate_config(_config())


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "POSTOPEN_UNFROZEN"),
        (("dataset", "rotation_degrees"), 60),
        (("dataset", "held_out_unit"), "CAMERA"),
        (("source_resolution_protocol", "result_sha256"), "0" * 64),
        (("transforms", "align_corners"), False),
        (("transforms", "fixed_line_alphas"), [0.0, 1.0]),
        (("transforms", "no_alpha_is_selected_as_an_algorithm"), False),
        (("diagnostic_gates", "linearity_max_abs_tolerance"), 1e-3),
        (("forward", "finite_aperture_sample_count"), 8),
        (("formal_output",), "demo_t16_operator/results/changed"),
        (("claim_firewall", "algorithm_superiority"), True),
        (("claim_firewall", "causal_mechanism_proved"), True),
    ],
)
def test_config_drift_fails_closed(path: tuple, value: object) -> None:
    config = copy.deepcopy(_config())
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_protocol_file_set_binds_transitive_operator_dependencies() -> None:
    files = set(_config()["protocol_files"])
    assert {
        "site_tools/psu_rotation40_active_store.py",
        "site_tools/psu_b0_real_support_store.py",
        "site_tools/psu_bost_aperture_domain.py",
        "site_tools/psu_bost_forward_geometry.py",
        "demo_t16_operator/psu_b0_streaming_operator.py",
        "demo_t16_operator/psu_b0_reconstruction_interface.py",
    }.issubset(files)


def test_resize_preserves_shape_dtype_and_zero_boundary() -> None:
    volume = torch.zeros((1, 1, 4, 4, 4), dtype=torch.float64)
    volume[:, :, 1:-1, 1:-1, 1:-1] = 1.0
    resized = resize_volume(volume, (8, 8, 8))
    assert resized.shape == (1, 1, 8, 8, 8)
    assert resized.dtype == torch.float64
    assert torch.all(resized[:, :, 0] == 0)
    assert torch.all(resized[:, :, -1] == 0)
    assert torch.all(resized[:, :, :, 0] == 0)
    assert torch.all(resized[:, :, :, -1] == 0)
    assert torch.all(resized[:, :, :, :, 0] == 0)
    assert torch.all(resized[:, :, :, :, -1] == 0)
    assert float(torch.max(resized)) > 0.0


def test_resize_rejects_noncanonical_volume() -> None:
    with pytest.raises(ValueError, match=r"\[1,1,z,y,x\]"):
        resize_volume(torch.zeros((1, 4, 4, 4)), (8, 8, 8))


def test_correction_alignment_identifies_anti_aligned_update() -> None:
    measured = np.array([[-1.0, 0.0], [-2.0, 0.0]], dtype=np.float64)
    baseline = np.zeros_like(measured)
    corrected = np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    result = correction_alignment(measured, baseline, corrected)
    assert result["residual_correction_cosine"] == pytest.approx(-1.0)
    assert result["unconstrained_least_squares_alpha_diagnostic_only"] == pytest.approx(
        -1.0
    )
    assert result["clipped_zero_one_alpha_diagnostic_only"] == 0.0


def test_correction_alignment_rejects_shape_mismatch() -> None:
    measured = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="share shape"):
        correction_alignment(measured, measured, np.zeros((3, 2), dtype=np.float64))


def test_mechanism_decision_requires_small_grid_gap_harm_and_all_negative() -> None:
    result = mechanism_decision(
        native16_relative_l2=0.84,
        prolonged16_relative_l2=0.845,
        native32_relative_l2=0.96,
        camera_correction_cosines=[-0.2, -0.4, -0.1],
        grid_gap_max=0.01,
        harm_min=0.01,
    )
    assert result["machine_diagnosis"] == (
        "OPENED_BLOCK_FIELD_CORRECTION_ANTI_ALIGNED_GRID_FORWARD_GAP_SMALL"
    )
    assert result["all_three_camera_fine_correction_cosines_negative"]
    assert not result["causal_mechanism_proved"]


def test_mechanism_decision_reports_material_forward_grid_change() -> None:
    result = mechanism_decision(
        native16_relative_l2=0.80,
        prolonged16_relative_l2=0.84,
        native32_relative_l2=0.96,
        camera_correction_cosines=[-0.2, -0.4, -0.1],
        grid_gap_max=0.01,
        harm_min=0.01,
    )
    assert result["machine_diagnosis"] == (
        "OPENED_BLOCK_FORWARD_GRID_CHANGE_MATERIAL_MECHANISM_UNRESOLVED"
    )


def test_mechanism_decision_does_not_hide_mixed_camera_alignment() -> None:
    result = mechanism_decision(
        native16_relative_l2=0.84,
        prolonged16_relative_l2=0.845,
        native32_relative_l2=0.96,
        camera_correction_cosines=[-0.2, 0.1, -0.1],
        grid_gap_max=0.01,
        harm_min=0.01,
    )
    assert result["machine_diagnosis"] == (
        "OPENED_BLOCK_FIELD_DIFFERENCE_HARM_WITH_MIXED_CAMERA_ALIGNMENT"
    )


def test_runner_has_no_final_rotation_path_or_candidate_alpha_search() -> None:
    source = (
        ROOT / "site_tools/run_psu_rotation40_multiresolution_diagnosis.py"
    ).read_text(encoding="utf-8")
    assert "rotation60_development" not in source
    assert "rotation70_development" not in source
    assert "rotation80_development" not in source
    assert "best_alpha" not in source
    assert "optimize.minimize" not in source
    assert "verify_protocol_commit(" in source
    assert "verify_private_inputs(" in source
