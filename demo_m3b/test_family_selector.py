from __future__ import annotations

import numpy as np

from demo_m3b.run_m3b_family_selector import (
    HOLDOUT_FEATURES,
    affine_projection_residual,
    annotate_oracle,
    fit_ridge,
    low_rank_candidates,
    make_family_sequence,
    ridge_predict,
)


def test_phantom_families_are_finite_distinct_sequences() -> None:
    families = ["blobs_sheet", "vortex_ring", "expanding_shell", "jet_filaments"]
    sequences = [make_family_sequence(family, "transient", n=12, nz=6, nt=8) for family in families]
    for sequence in sequences:
        assert sequence.shape == (8, 6, 12, 12)
        assert np.isfinite(sequence).all()
        assert float(sequence.min()) >= 0.0
        assert np.isclose(float(sequence.max()), 1.0)
        assert float(np.linalg.norm(np.diff(sequence, axis=0))) > 0.0
    pairwise = [np.linalg.norm(sequences[i] - sequences[j]) for i in range(4) for j in range(i + 1, 4)]
    assert min(pairwise) > 1.0


def test_full_rank_candidate_reproduces_input() -> None:
    rng = np.random.default_rng(7)
    sequence = rng.normal(size=(8, 4, 6, 6))
    candidates, singular_values = low_rank_candidates(sequence, [2, 8])
    assert singular_values.shape == (8,)
    assert np.allclose(candidates[8], sequence, atol=1e-10)
    assert np.linalg.norm(candidates[2] - sequence) > 0.1


def test_affine_projection_residual_ignores_global_scale_and_offset() -> None:
    rng = np.random.default_rng(11)
    prediction = rng.normal(size=(4, 2, 6, 3))
    observation = 2.4 * prediction - 0.7
    assert affine_projection_residual(prediction, observation) < 1e-10


def test_oracle_annotation_and_ridge_contract() -> None:
    rows = []
    for cell_index in range(4):
        for rank, error in [(2, 0.22 + 0.01 * cell_index), (3, 0.20), (5, 0.24)]:
            row = {
                "cell_id": f"cell-{cell_index}",
                "rank": rank,
                "field_rel_l2": error,
            }
            for feature_index, feature in enumerate(HOLDOUT_FEATURES):
                row[feature] = 0.1 * rank + 0.01 * cell_index + 0.001 * feature_index
            rows.append(row)
    annotate_oracle(rows)
    assert all(int(row["oracle_rank"]) == 3 for row in rows)
    model = fit_ridge(rows, 0.1, HOLDOUT_FEATURES)
    prediction = ridge_predict(model, rows)
    assert prediction.shape == (len(rows),)
    assert np.isfinite(prediction).all()
