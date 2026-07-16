from __future__ import annotations

import json
from pathlib import Path

import torch

from demo_t16_operator.psu_b0_streaming_operator import PSUB0StreamingOperator
from site_tools.psu_b0_compact_cache import (
    CACHE_STATUS,
    PSUCompactCachedRayStore,
    build_compact_cache,
)
from site_tools.psu_b0_real_support_store import PSURealSupportRayStore
from site_tools.test_psu_b0_real_support_store import _write_view


def _stores(tmp_path: Path):
    source_root = tmp_path / "views"
    for view_id in range(9):
        _write_view(source_root, view_id)
    direct = PSURealSupportRayStore(
        source_root,
        rays_per_view=None,
        sample_count=4,
        chunk_rays=3,
        grid_minimum_xyz=(-0.11, -0.11, -0.11),
        grid_maximum_xyz=(0.11, 0.11, 0.11),
    )
    cache_root = tmp_path / "cache"
    manifest = build_compact_cache(
        direct,
        cache_root,
        grid_shape=(5, 5, 5),
        grid_minimum_xyz=(-0.11, -0.11, -0.11),
        grid_maximum_xyz=(0.11, 0.11, 0.11),
        fraction_dtype="float64",
    )
    cached = PSUCompactCachedRayStore(cache_root, verify_hashes=True)
    return direct, cached, cache_root, manifest


def test_compact_cache_matches_direct_store_forward_and_adjoint(
    tmp_path: Path,
) -> None:
    direct, cached, _, manifest = _stores(tmp_path)
    assert manifest["status"] == CACHE_STATUS
    assert cached.ray_count == direct.ray_count
    assert cached.selection_summary()["cache_mode"].startswith("private_compact")
    direct_operator = PSUB0StreamingOperator(
        ray_store=direct,
        grid_shape=(5, 5, 5),
        grid_minimum_xyz=(-0.11, -0.11, -0.11),
        grid_maximum_xyz=(0.11, 0.11, 0.11),
        dtype=torch.float64,
    )
    cached_operator = PSUB0StreamingOperator(
        ray_store=cached,
        grid_shape=(5, 5, 5),
        grid_minimum_xyz=(-0.11, -0.11, -0.11),
        grid_maximum_xyz=(0.11, 0.11, 0.11),
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(20260716)
    volume = torch.randn(
        (2, 1, 5, 5, 5),
        generator=generator,
        dtype=torch.float64,
    )
    residual = torch.randn(
        (2, direct.ray_count, 2),
        generator=generator,
        dtype=torch.float64,
    )
    assert torch.equal(direct_operator(volume), cached_operator(volume))
    assert torch.equal(
        direct_operator.adjoint(residual),
        cached_operator.adjoint(residual),
    )
    assert torch.equal(
        direct_operator.load_observations(),
        cached_operator.load_observations(),
    )


def test_compact_cache_rejects_grid_mismatch_and_incomplete_manifest(
    tmp_path: Path,
) -> None:
    _, cached, cache_root, _ = _stores(tmp_path)
    try:
        PSUB0StreamingOperator(
            ray_store=cached,
            grid_shape=(6, 6, 6),
            grid_minimum_xyz=(-0.11, -0.11, -0.11),
            grid_maximum_xyz=(0.11, 0.11, 0.11),
            dtype=torch.float64,
        )
    except ValueError as exc:
        assert "compact grid" in str(exc)
    else:
        raise AssertionError("grid mismatch should fail")

    manifest_path = cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "BUILDING"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        PSUCompactCachedRayStore(cache_root)
    except ValueError as exc:
        assert "not complete" in str(exc)
    else:
        raise AssertionError("incomplete cache should fail")
