from __future__ import annotations

import math

import numpy as np

from demo_m3b.run_m3b_geometry_uq import (
    MODEL_FEATURE_SETS,
    binary_auc,
    circular_gaps,
    geometry_angles,
    geometry_descriptors,
    largest_gap_midpoint,
    percentile_ranks,
    rankdata,
    spearman,
    train_logo_models,
    vote_entropy,
)


GEOMETRIES = [
    "uniform",
    "rotated_uniform",
    "limited_arc",
    "dual_cluster",
    "jittered",
    "calibration_offset_2deg",
]


def test_geometry_angles_are_paired_and_calibration_offset_is_explicit() -> None:
    for geometry in GEOMETRIES:
        true_angles, reconstruction_angles = geometry_angles(geometry, 7)
        assert true_angles.shape == reconstruction_angles.shape == (7,)
        assert np.isfinite(true_angles).all()
        assert np.isfinite(reconstruction_angles).all()
        assert np.all((0.0 <= true_angles) & (true_angles < 180.0))
        assert np.all((0.0 <= reconstruction_angles) & (reconstruction_angles < 180.0))
        if geometry == "calibration_offset_2deg":
            assert np.allclose(reconstruction_angles - true_angles, 2.0)
        else:
            assert np.allclose(reconstruction_angles, true_angles)


def test_circular_gap_and_descriptor_contracts() -> None:
    uniform, _ = geometry_angles("uniform", 5)
    limited, _ = geometry_angles("limited_arc", 5)
    clustered, _ = geometry_angles("dual_cluster", 5)
    assert np.allclose(circular_gaps(uniform), 36.0)
    uniform_features = geometry_descriptors(uniform)
    limited_features = geometry_descriptors(limited)
    clustered_features = geometry_descriptors(clustered)
    assert math.isclose(uniform_features["geometry_gap_entropy"], 1.0, abs_tol=1e-10)
    assert uniform_features["geometry_gap_std_fraction"] < 1e-12
    assert limited_features["geometry_max_gap_fraction"] > uniform_features["geometry_max_gap_fraction"]
    assert clustered_features["geometry_max_gap_fraction"] > uniform_features["geometry_max_gap_fraction"]
    assert limited_features["geometry_coverage_fraction"] < uniform_features["geometry_coverage_fraction"]


def test_heldout_angle_is_midpoint_of_the_largest_gap() -> None:
    for geometry in GEOMETRIES:
        angles, _ = geometry_angles(geometry, 7)
        midpoint = largest_gap_midpoint(angles)
        ordered = np.sort(angles)
        gaps = circular_gaps(ordered)
        index = int(np.argmax(gaps))
        expected = float(np.mod(ordered[index] + 0.5 * gaps[index], 180.0))
        assert math.isclose(midpoint, expected, abs_tol=1e-12)
        circular_distance = np.minimum(np.abs(angles - midpoint), 180.0 - np.abs(angles - midpoint))
        assert float(np.min(circular_distance)) > 0.0


def test_rank_statistics_handle_ties_and_direction() -> None:
    values = [4.0, 1.0, 1.0, 3.0]
    assert np.allclose(rankdata(values), [3.0, 0.5, 0.5, 2.0])
    assert np.allclose(percentile_ranks(values), [1.0, 1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0])
    assert math.isclose(spearman([1, 2, 3], [3, 2, 1]), -1.0)
    assert math.isclose(binary_auc([0.1, 0.2, 0.9, 1.0], [0, 0, 1, 1]), 1.0)
    assert math.isclose(binary_auc([0.9, 1.0, 0.1, 0.2], [0, 0, 1, 1]), 0.0)
    assert math.isclose(binary_auc([0.1, 0.2], [0, 0]), 0.5)


def test_vote_entropy_is_zero_for_agreement_and_positive_for_disagreement() -> None:
    assert abs(vote_entropy([3, 3, 3, 3])) < 1e-10
    assert vote_entropy([2, 3, 5, 8]) > vote_entropy([2, 2, 2, 3]) > 0.0


def test_nested_logo_audit_never_fits_the_outer_heldout_geometry() -> None:
    geometries = ["g0", "g1", "g2", "g3"]
    rows: list[dict[str, object]] = []
    for geometry_index, geometry in enumerate(geometries):
        for cell_index in range(2):
            for rank in [2, 3, 5]:
                row: dict[str, object] = {
                    "cell_id": f"{geometry}-{cell_index}",
                    "geometry": geometry,
                    "rank": rank,
                    "target_log_error_ratio": 0.03 * rank + 0.01 * geometry_index,
                    "oracle_regret_pct": abs(rank - 3) + 0.1 * geometry_index,
                }
                for feature_index, feature in enumerate(
                    sorted({name for features in MODEL_FEATURE_SETS.values() for name in features})
                ):
                    row[feature] = 0.1 * rank + 0.02 * geometry_index + 0.003 * cell_index + 0.0001 * feature_index
                rows.append(row)

    models, audit = train_logo_models(rows, geometries, [0.01, 0.1])
    assert set(models) == set(MODEL_FEATURE_SETS)
    for heldout_geometry, fold in audit.items():
        assert heldout_geometry not in fold["training_geometries"]
        assert set(fold["training_geometries"]) == set(geometries) - {heldout_geometry}
        for mode in MODEL_FEATURE_SETS:
            assert heldout_geometry not in fold[mode]["features"]
            members = fold[mode]["ensemble_members"]
            assert len(members) == len(geometries) - 1
            for member in members:
                omitted = member["omitted_training_geometry"]
                assert heldout_geometry not in member["fit_geometries"]
                assert omitted not in member["fit_geometries"]
                assert set(member["fit_geometries"]) == set(geometries) - {
                    heldout_geometry,
                    omitted,
                }
