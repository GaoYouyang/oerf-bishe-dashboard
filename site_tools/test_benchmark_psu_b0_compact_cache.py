from __future__ import annotations

from pathlib import Path

from site_tools.benchmark_psu_b0_compact_cache import benchmark_compact_cache
from site_tools.psu_b0_compact_cache import (
    PSUCompactCachedRayStore,
    build_compact_cache,
)
from site_tools.psu_b0_real_support_store import PSURealSupportRayStore
from site_tools.test_psu_b0_real_support_store import _write_view


def test_tiny_compact_cache_benchmark_preserves_operator(tmp_path: Path) -> None:
    view_root = tmp_path / "views"
    for view_id in range(9):
        _write_view(view_root, view_id)
    direct = PSURealSupportRayStore(
        view_root,
        rays_per_view=None,
        sample_count=4,
        chunk_rays=3,
    )
    cache_root = tmp_path / "cache"
    build_compact_cache(
        direct,
        cache_root,
        grid_shape=(5, 5, 5),
        grid_minimum_xyz=(-0.11, -0.11, -0.11),
        grid_maximum_xyz=(0.11, 0.11, 0.11),
    )
    cached = PSUCompactCachedRayStore(cache_root, verify_hashes=True)
    report = benchmark_compact_cache(
        direct_store=direct,
        cached_store=cached,
        grid_shape=(5, 5, 5),
        grid_minimum_xyz=(-0.11, -0.11, -0.11),
        grid_maximum_xyz=(0.11, 0.11, 0.11),
        seed=20260719,
        repeats=1,
        torch_threads=2,
        forward_difference_maximum=1e-14,
        adjoint_difference_maximum=1e-14,
        dot_error_maximum=1e-11,
        speedup_minimum=0.0,
        rss_bytes_maximum=24 * 1024**3,
    )
    assert report["status"].endswith("PASS")
    assert report["numerical_equivalence"]["forward_relative_difference"] == 0
    assert report["numerical_equivalence"]["adjoint_relative_difference"] == 0
    assert report["gates"]["cached_relative_dot_error"] is True
