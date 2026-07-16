#!/usr/bin/env python3
"""Build the public, root-relative checksum manifest for the V5P-V6A release."""

from __future__ import annotations

from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from demo_t16_operator.release_provenance import file_sha256
else:
    from .release_provenance import file_sha256


OPERATOR_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = OPERATOR_ROOT.parent
MANIFEST_PATH = OPERATOR_ROOT / "results" / "v5p_v6a_release_checksums.sha256"
RESULT_DIRECTORIES = (
    "v5p_fresh_budget_gate",
    "v5q_postopen_topology_diagnosis",
    "v5r_reserved_view_reliability_diagnosis",
    "v5s_dco_low_rank_screening",
    "v5t_camera_local_tangent_diagnosis",
    "v5u_calibrated_renderer_residual_screening",
    "v5v_camera_local_kernel_correction",
    "v5w_clean_aperture_kernel_screening",
    "v5x_ray_conditioned_voxel_kernel",
    "v5y_direct_ray_conditioned_kernel",
    "v5z_stabilized_direct_ray_kernel",
    "v6a_ray_kernel_hypernetwork_development",
)
REPRODUCTION_CHECKPOINTS = tuple(
    OPERATOR_ROOT
    / "results"
    / "v5k_shared_field_work"
    / "sf_rio_adjoint_only"
    / str(seed)
    / "best.pt"
    for seed in (3101, 3102, 3103)
)
EXPLICIT_REPOSITORY_PATHS = (
    "general_operator_research_lab.html",
    "docs/operator_3d_learning_log.md",
    "docs/route_b_dco_trail_research_contract_2026-07-16.md",
    "docs/v5h_v5m_共享场逆算子研究日志_2026-07-16.md",
    "docs/v5p_v6a_release_reproducibility.md",
)
DIRECT_DEPENDENCIES = (
    "gc_rio/data.py",
    "gc_rio/protocol.py",
    "gc_rio/shared_field_model.py",
    "gc_rio/training.py",
    "independent_reaction_bost.py",
    "release_provenance.py",
    "run_v5h_gc_rio_development.py",
    "run_v5n_strong_classical_baselines.py",
    "run_v5o_prior_anchored_frontier.py",
    "configs/v5h_gc_rio_development.json",
    "results/v5k_shared_field_development/report.json",
)
EXCLUDED_PATH_PARTS = {".ruff_cache", ".pytest_cache", "__pycache__"}


def collect_release_paths() -> list[Path]:
    """Return a deterministic, privacy-screened public release file set."""

    paths: set[Path] = {
        REPOSITORY_ROOT / relative for relative in EXPLICIT_REPOSITORY_PATHS
    }
    paths.update(OPERATOR_ROOT / relative for relative in DIRECT_DEPENDENCIES)
    paths.update(REPRODUCTION_CHECKPOINTS)
    paths.update(
        {
            OPERATOR_ROOT / "build_v5p_v6a_release_manifest.py",
            OPERATOR_ROOT / "validate_v5p_v6a_release.py",
            OPERATOR_ROOT / "results" / "operator_structure_funnel_v5s_v6a.png",
        }
    )
    for pattern in (
        "configs/v5[p-z]*.json",
        "configs/v6a*.json",
        "run_v5[p-z]*.py",
        "run_v6a*.py",
        "plot_v5[p-z]*.py",
        "plot_v6a*.py",
        "plot_v5s_v6a*.py",
        "test_v5[p-z]*.py",
        "test_v6a*.py",
    ):
        paths.update(OPERATOR_ROOT.glob(pattern))
    for directory_name in RESULT_DIRECTORIES:
        paths.update(
            path
            for path in (OPERATOR_ROOT / "results" / directory_name).rglob("*")
            if path.is_file() and not (set(path.parts) & EXCLUDED_PATH_PARTS)
        )

    resolved_root = REPOSITORY_ROOT.resolve()
    screened: list[Path] = []
    for path in sorted(paths):
        resolved = path.resolve()
        resolved.relative_to(resolved_root)
        relative = resolved.relative_to(resolved_root)
        if not resolved.is_file():
            raise FileNotFoundError(relative.as_posix())
        if resolved == MANIFEST_PATH.resolve():
            continue
        if set(relative.parts) & EXCLUDED_PATH_PARTS:
            continue
        lowered_parts = {part.lower() for part in relative.parts}
        if {"private_library", "tmp_downloads"} & lowered_parts:
            raise ValueError(f"private path cannot enter release: {relative}")
        if resolved.suffix.lower() == ".pdf":
            raise ValueError(f"PDF cannot enter this public release: {relative}")
        screened.append(resolved)
    return screened


def main() -> None:
    paths = collect_release_paths()
    lines = [
        f"{file_sha256(path)}  {path.relative_to(REPOSITORY_ROOT).as_posix()}"
        for path in paths
    ]
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} release hashes to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
