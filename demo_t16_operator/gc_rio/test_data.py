from __future__ import annotations

import numpy as np

from .data import FORBIDDEN_PREDICTOR_KEYS, build_dataset, paired_difference_mad


def config() -> dict:
    rigs = []
    for index, split in enumerate(("train", "validation", "design_lock")):
        rigs.append({"id": f"rig-{index}", "split": split, "angles_degrees": np.arange(7) * (180.0 / 7.0) + index, "source_indices": (0, 1, 2, 3), "target_indices": (4, 5), "reserved_indices": (6,), "cone_u": .07 + .01 * index, "cone_z": .05, "bend": .035, "truth_radius": .10, "model_radius": .07, "truth_aperture_samples": 3, "truth_path_samples": 8, "model_aperture_samples": 3, "model_path_samples": 8})
    return {"seed": 17, "grid_size": 3, "depth": 2, "rigs": rigs, "splits": {"train": {"families": ("expanding_kernel",)}, "validation": {"families": ("jet_shear",)}, "design_lock": {"families": ("shock_cell",)}}, "fields_per_family": 1, "camera_sigma": [.01, .012, .014, .016, .018, .02, .022], "flow_off_replicates": 4, "correlation_fraction": .25, "signal_fraction": .1, "support_threshold": .001, "correction_kappa": .001}


def test_repeated_mad_does_not_read_clean_signal() -> None:
    rng = np.random.default_rng(2)
    repeats = rng.normal(size=(8, 2, 4, 3))
    estimate = paired_difference_mad(repeats)
    np.testing.assert_allclose(estimate, paired_difference_mad(repeats + 1000.0))


def test_target_only_rows_shape_and_poison_firewall() -> None:
    bundle = build_dataset(config())
    assert len(bundle.rows) == 6
    batch = bundle.predictor_batch(range(len(bundle.rows)))
    assert set(batch) == bundle.predictor_keys
    assert batch["source_operator"].ndim == 4 and batch["target_operator"].ndim == 3
    assert batch["source_residual"].ndim == 3 and batch["source_sigma"].shape == (6, 4)
    assert batch["base_field"].shape == batch["support"].shape == (6, 18)
    assert batch["target_sigma"].shape == (6,)
    assert not (FORBIDDEN_PREDICTOR_KEYS & batch.keys())
    before = {key: value.copy() for key, value in batch.items()}
    for row in bundle.rows:
        row["target_observation"] += 9999.0
    after = bundle.predictor_batch(range(len(bundle.rows)))
    for key in before:
        np.testing.assert_equal(before[key], after[key])


def test_split_family_rig_and_field_disjoint_and_target_pair_locked() -> None:
    bundle = build_dataset(config())
    by_field = {row["field_uid"]: {other["split"] for other in bundle.rows if other["field_uid"] == row["field_uid"]} for row in bundle.rows}
    assert all(len(splits) == 1 for splits in by_field.values())
    assert len({row["rig_id"] for row in bundle.rows}) == 3
    assert all(sum(row["target_view"] == view for row in bundle.rows if row["field_uid"] == field) == 1 for field in by_field for view in (4, 5))
    assert len(set(bundle.manifest["operator_hashes"].values())) >= 2


def test_deterministic_small_configuration() -> None:
    left, right = build_dataset(config()), build_dataset(config())
    assert left.manifest == right.manifest
    np.testing.assert_equal(left.predictor_batch([0])["base_field"], right.predictor_batch([0])["base_field"])
    np.testing.assert_equal(left.predictor_batch([0])["analytic_correction"], right.predictor_batch([0])["analytic_correction"])
