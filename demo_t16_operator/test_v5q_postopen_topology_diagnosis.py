from __future__ import annotations

from typing import Any

import numpy as np

from demo_t16_operator.run_v5q_postopen_topology_diagnosis import (
    FEATURE_NAMES,
    _finite_spearman,
    compute_source_feature_row,
)


class TargetLabelGuard(dict[str, Any]):
    def __getitem__(self, key: str) -> Any:
        if key in {"target_residual_label", "clean_target_residual"}:
            raise AssertionError("source feature extraction touched a target label")
        return super().__getitem__(key)


def _row() -> TargetLabelGuard:
    return TargetLabelGuard(
        {
            "field_uid": "rig-a:family-a:0",
            "rig_id": "rig-a",
            "family": "family-a",
            "support": np.ones(8, dtype=np.float32),
            "base_field": np.asarray(
                [0.3, 0.5, 0.4, 0.2, 0.35, 0.45, 0.25, 0.15],
                dtype=np.float32,
            ),
            "source_operator": np.asarray(
                [
                    [
                        [1.0, 0.2, 0.0, 0.1, 0.3, 0.0, 0.2, 0.1],
                        [0.0, 0.4, 0.8, 0.2, 0.1, 0.3, 0.0, 0.2],
                    ]
                ],
                dtype=np.float32,
            ),
            "source_residual": np.asarray([[0.15, 0.08]], dtype=np.float32),
            "source_sigma": np.asarray([0.02], dtype=np.float32),
            "target_residual_label": np.zeros(2, dtype=np.float32),
        }
    )


def test_source_feature_extraction_has_target_label_firewall() -> None:
    prior = np.asarray(
        [0.03, -0.02, 0.01, 0.0, 0.02, -0.01, 0.01, 0.0], dtype=np.float32
    )
    candidate = np.asarray(
        [0.04, -0.01, 0.02, 0.0, 0.03, 0.0, 0.02, -0.01], dtype=np.float32
    )
    pbb9 = np.asarray(
        [0.02, 0.0, 0.03, -0.01, 0.02, 0.01, 0.0, -0.01], dtype=np.float32
    )
    members = np.stack([prior * 0.8, prior, prior * 1.2])
    features = compute_source_feature_row(
        _row(),
        members,
        prior,
        candidate,
        pbb9,
        depth=2,
        grid_size=2,
    )
    assert set(FEATURE_NAMES).issubset(features)
    assert all(np.isfinite(float(features[name])) for name in FEATURE_NAMES)


def test_finite_spearman_rejects_constant_or_tiny_inputs() -> None:
    assert _finite_spearman([1.0, 1.0, 1.0], [0.0, 1.0, 2.0]) is None
    assert _finite_spearman([1.0, 2.0], [0.0, 1.0]) is None
    assert np.isclose(
        _finite_spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]), -1.0
    )
