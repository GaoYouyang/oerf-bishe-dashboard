from __future__ import annotations

import numpy as np

from site_tools.run_psu_b0_observable_morphology_probe import (
    fit_ridge_classifier,
    fit_ridge_multioutput,
    prediction_gains,
    ridge_scores,
    score_predictions,
    stratified_folds,
)


def _candidate_rows() -> list[dict]:
    rows = []
    for sample, baseline, expert in (
        ("a", 1.0, 0.8),
        ("b", 1.0, 1.2),
        ("c", 1.0, 0.7),
        ("d", 1.0, 0.9),
    ):
        rows.extend(
            [
                {
                    "sample_id": sample,
                    "candidate_id": "base",
                    "field_relative_l2": baseline,
                },
                {
                    "sample_id": sample,
                    "candidate_id": "expert",
                    "field_relative_l2": expert,
                },
            ]
        )
    return rows


def test_stratified_folds_cycle_within_each_class() -> None:
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    folds = stratified_folds(labels, fold_count=4)
    assert folds.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]


def test_ridge_classifier_separates_simple_classes() -> None:
    features = np.asarray(
        [[-2.0, 0.0], [-1.0, 0.1], [1.0, 0.0], [2.0, -0.1]]
    )
    labels = np.asarray([0, 0, 1, 1])
    model = fit_ridge_classifier(
        features,
        labels,
        class_count=2,
        regularization=0.01,
    )
    predicted, margins = score_predictions(ridge_scores(model, features))
    assert predicted.tolist() == labels.tolist()
    assert np.all(margins > 0.0)


def test_ridge_multioutput_fits_linear_targets() -> None:
    features = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
    targets = np.concatenate((features, -features), axis=1)
    model = fit_ridge_multioutput(
        features,
        targets,
        regularization=1e-6,
    )
    predicted = ridge_scores(model, features)
    assert np.allclose(predicted, targets, atol=1e-5, rtol=1e-5)


def test_prediction_gains_falls_back_below_threshold() -> None:
    result = prediction_gains(
        sample_ids=["a", "b", "c", "d"],
        predicted_labels=np.asarray([1, 1, 1, 1]),
        margins=np.asarray([0.9, 0.1, 0.8, 0.2]),
        threshold=0.5,
        class_candidates=["base", "expert"],
        candidate_rows=_candidate_rows(),
        baseline_candidate_id="base",
    )
    assert result["selected_candidate_ids"] == [
        "expert",
        "base",
        "expert",
        "base",
    ]
    assert result["coverage"] == 0.5
    assert result["harm_over_one_percent_rate"] == 0.0
