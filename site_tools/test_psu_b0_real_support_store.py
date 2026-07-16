from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from site_tools.psu_b0_real_support_store import (
    MASK_STATUS,
    VIEW_BUNDLE_STATUS,
    PSURealSupportRayStore,
    deterministic_quantile_indices,
)


def _write_view(root: Path, view_id: int) -> None:
    view = root / f"view_{view_id:02d}"
    bundle = view / "bundle"
    masks = view / "corrected_masks"
    bundle.mkdir(parents=True)
    masks.mkdir()
    rows = 6
    (bundle / "view_bundle_manifest.json").write_text(
        json.dumps(
            {
                "status": VIEW_BUNDLE_STATUS,
                "view": {
                    "view_id_zero_based": view_id,
                    "measurement_count": rows,
                },
            }
        ),
        encoding="utf-8",
    )
    (masks / "corrected_view_masks_manifest.json").write_text(
        json.dumps({"status": MASK_STATUS}),
        encoding="utf-8",
    )
    active = np.asarray([1, 2, 4, 5], dtype=np.int64)
    np.save(masks / "amask_all_zero_based.npy", active)
    y = np.linspace(-0.04, 0.04, rows)
    origins = np.column_stack((np.full(rows, -0.5), y, np.zeros(rows)))
    directions = np.tile([1.0, 0.0, 0.0], (rows, 1))
    vector_fields = {
        "c": origins,
        "v": directions,
        "Ruvecs": np.tile([0.0, 1.0, 0.0], (rows, 1)),
        "Rvvecs": np.tile([0.0, 0.0, 1.0], (rows, 1)),
        "Rxvecs": np.tile([0.0, 1.0, 0.0], (rows, 1)),
        "Ryvecs": np.tile([0.0, 0.0, 1.0], (rows, 1)),
    }
    scalar_fields = {
        "Rapvec": np.full((rows, 1), 0.001),
        "Dfvec": np.full((rows, 1), 1.0),
        "Csys_all": np.full((rows, 1), 0.8 + 0.01 * view_id),
        "epsu_all": (100 * view_id + np.arange(rows))[:, None],
        "epsv_all": (-100 * view_id - np.arange(rows))[:, None],
    }
    for name, values in {**vector_fields, **scalar_fields}.items():
        np.save(bundle / f"{name}.npy", np.asarray(values, dtype=np.float32))


def test_real_store_quantile_selection_and_contiguous_chunks(tmp_path: Path) -> None:
    for view_id in range(9):
        _write_view(tmp_path, view_id)
    store = PSURealSupportRayStore(
        tmp_path,
        rays_per_view=2,
        sample_count=4,
        chunk_rays=1,
    )
    assert store.ray_count == 18
    assert store.selection_mode.startswith("ordered_active_mask_quantiles")
    assert deterministic_quantile_indices(
        np.asarray([1, 2, 4, 5], dtype=np.int64),
        2,
    ).tolist() == [2, 5]
    chunks = list(store.iter_chunks())
    assert len(chunks) == 18
    assert [chunk.start_index for chunk in chunks] == list(range(18))
    assert [chunk.stop_index for chunk in chunks] == list(range(1, 19))
    assert all(chunk.b0_hit_count == 1 for chunk in chunks)
    assert all(chunk.sample_points_xyz.shape == (1, 4, 3) for chunk in chunks)
    observation = store.load_observations(
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert observation.shape == (1, 18, 2)
    assert observation[0, 0].tolist() == [2.0, -2.0]
    assert observation[0, 2].tolist() == [102.0, -102.0]


def test_real_store_can_stream_all_active_rows(tmp_path: Path) -> None:
    for view_id in range(9):
        _write_view(tmp_path, view_id)
    store = PSURealSupportRayStore(
        tmp_path,
        rays_per_view=None,
        sample_count=4,
        chunk_rays=3,
    )
    assert store.ray_count == 36
    assert store.selection_mode == "all_active_rows"
    assert sum(chunk.ray_count for chunk in store.iter_chunks()) == 36
