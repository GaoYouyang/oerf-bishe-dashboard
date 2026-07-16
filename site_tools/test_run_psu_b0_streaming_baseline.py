from __future__ import annotations

from pathlib import Path

from site_tools.run_psu_b0_streaming_baseline import (
    PUBLIC_SCHEMA,
    build_public_summary,
    run_baseline,
)
from site_tools.psu_b0_compact_cache import build_compact_cache
from site_tools.psu_b0_real_support_store import PSURealSupportRayStore
from site_tools.test_psu_b0_real_support_store import _write_view


def test_public_summary_is_independent_and_strips_private_fields() -> None:
    private = {
        "status": "fixture",
        "evidence_scope": "fixture",
        "dataset": {},
        "configuration": {"config_sha256_private_only": "secret"},
        "selection": {},
        "interface_profile": {},
        "optimization": {},
        "evaluation": {"field_diagnostics_private": {"maximum": 1.0}},
        "resource_gate": {},
        "gates": {},
        "claim_boundary": {},
    }
    public = build_public_summary(private)
    public["configuration"].pop("config_sha256_private_only")
    public["evaluation"].pop("field_diagnostics_private")
    assert public["schema_version"] == PUBLIC_SCHEMA
    assert private["configuration"]["config_sha256_private_only"] == "secret"
    assert "field_diagnostics_private" in private["evaluation"]


def test_tiny_real_store_baseline_closes_without_private_export(
    tmp_path: Path,
) -> None:
    for view_id in range(9):
        _write_view(tmp_path, view_id)
    private, public, volume = run_baseline(
        view_root=tmp_path,
        grid_size=5,
        sample_count=4,
        chunk_rays=2,
        rays_per_view=2,
        iterations=2,
        dtype_name="float64",
        device_name="cpu",
        dot_seed=20260716,
        dot_dual="deterministic_random_vector",
        dot_threshold=1e-11,
        recurrence_threshold=1e-10,
        local_pair_seconds_maximum=60.0,
        local_rss_bytes_maximum=24 * 1024**3,
    )
    assert private["status"].endswith("NO_FIELD_TRUTH")
    assert public["status"] == private["status"]
    assert volume.shape == (1, 1, 5, 5, 5)
    assert public["selection"]["total_ray_count"] == 18
    assert public["optimization"]["logical_forward_calls"] == 2
    assert public["optimization"]["logical_adjoint_calls"] == 3
    assert public["evaluation"]["direct_support_relative_measurement_l2"] < 1
    assert "field_diagnostics_private" not in public["evaluation"]
    assert public["claim_boundary"]["algorithm_superiority"] is False


def test_tiny_baseline_can_reuse_compact_cache(tmp_path: Path) -> None:
    view_root = tmp_path / "views"
    for view_id in range(9):
        _write_view(view_root, view_id)
    direct = PSURealSupportRayStore(
        view_root,
        rays_per_view=2,
        sample_count=4,
        chunk_rays=2,
    )
    cache_root = tmp_path / "cache"
    build_compact_cache(
        direct,
        cache_root,
        grid_shape=(5, 5, 5),
        grid_minimum_xyz=(-0.11, -0.11, -0.11),
        grid_maximum_xyz=(0.11, 0.11, 0.11),
    )
    private, public, volume = run_baseline(
        view_root=None,
        cache_root=cache_root,
        grid_size=5,
        sample_count=4,
        chunk_rays=2,
        rays_per_view=2,
        iterations=2,
        dtype_name="float64",
        device_name="cpu",
        dot_seed=20260716,
        dot_dual="deterministic_random_vector",
        dot_threshold=1e-11,
        recurrence_threshold=1e-10,
        local_pair_seconds_maximum=60.0,
        local_rss_bytes_maximum=24 * 1024**3,
        torch_threads=2,
    )
    assert private["status"].endswith("NO_FIELD_TRUTH")
    assert public["configuration"]["ray_store_mode"] == (
        "private_compact_stencil_cache"
    )
    assert public["configuration"]["torch_threads"] == 2
    assert volume.shape == (1, 1, 5, 5, 5)
