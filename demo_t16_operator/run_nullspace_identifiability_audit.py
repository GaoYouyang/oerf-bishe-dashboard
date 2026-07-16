#!/usr/bin/env python3
"""Audit whether support-nullspace correction is worth learning for T16.

The oracle correction in this script uses the synthetic field truth. It is an
upper-bound and identifiability diagnostic, not a deployable reconstruction
method. Its purpose is to test the structural premise behind SC-DBNO v2 before
training another neural network.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    from .bost_physics import forward_volume
    from .data import generate_dataset, load_npz, split_indices
    from .run_ablations import SPLIT_ORDER
    from .train_eval import _gradient_relative, _masked_projection_relative, _relative_norm
except ImportError:
    from bost_physics import forward_volume
    from data import generate_dataset, load_npz, split_indices
    from run_ablations import SPLIT_ORDER
    from train_eval import _gradient_relative, _masked_projection_relative, _relative_norm


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "smoke.json"
DEFAULT_DATA = ROOT / "results" / "t16_smoke_v1_dataset.npz"
DEFAULT_RESULTS = ROOT / "results" / "nullspace_identifiability"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--force-data", action="store_true")
    return parser.parse_args()


def support_matrix(operator: np.ndarray, view_mask: np.ndarray) -> np.ndarray:
    selected = np.flatnonzero(np.asarray(view_mask) > 0.5)
    if len(selected) == 0:
        raise ValueError("at least one support view is required")
    return np.asarray(operator[selected], dtype=np.float64).reshape(-1, operator.shape[-1])


def nullspace_decomposition(
    operator: np.ndarray,
    view_mask: np.ndarray,
    relative_tolerance: float = 1e-9,
) -> dict[str, object]:
    matrix = support_matrix(operator, view_mask)
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    threshold = relative_tolerance * max(float(singular_values[0]), 1.0)
    rank = int(np.sum(singular_values > threshold))
    null_basis = vh[rank:]
    return {
        "matrix": matrix,
        "rank": rank,
        "nullity": int(matrix.shape[1] - rank),
        "null_basis": null_basis,
        "threshold": threshold,
        "condition_number_nonzero": float(singular_values[0] / singular_values[rank - 1]) if rank else math.inf,
        "singular_values": singular_values,
    }


def project_to_nullspace(volume: np.ndarray, null_basis: np.ndarray) -> np.ndarray:
    shape = volume.shape
    flat = np.asarray(volume, dtype=np.float64).reshape(shape[0], -1)
    if null_basis.shape[0] == 0:
        return np.zeros_like(volume, dtype=np.float64)
    projected = (flat @ null_basis.T) @ null_basis
    return projected.reshape(shape)


def metric_row(
    sample_index: int,
    split: str,
    data: dict[str, np.ndarray],
    decomposition: dict[str, object],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    target = np.asarray(data["field"][sample_index], dtype=np.float64)
    lift = np.asarray(data["lift"][sample_index], dtype=np.float64)
    mask = np.asarray(data["view_mask"][sample_index], dtype=np.float64)
    clean = np.asarray(data["clean_observation"][sample_index], dtype=np.float64)
    operator = np.asarray(data["forward_matrix"], dtype=np.float64)
    error = target - lift
    null_correction = project_to_nullspace(error, np.asarray(decomposition["null_basis"]))
    oracle_null = lift + null_correction
    remaining = target - oracle_null
    lift_projection = forward_volume(lift, operator)
    null_projection = forward_volume(oracle_null, operator)
    correction_projection = forward_volume(null_correction, operator)
    query_mask = 1.0 - mask

    base_field = _relative_norm(lift - target, target)
    oracle_field = _relative_norm(oracle_null - target, target)
    error_energy = float(np.sum(error**2)) + 1e-12
    correction_energy = float(np.sum(null_correction**2))
    orthogonality = abs(float(np.sum(null_correction * remaining))) / (
        math.sqrt(correction_energy * float(np.sum(remaining**2))) + 1e-12
    )
    support_projection_change = _relative_norm(
        correction_projection * mask[None, :, None],
        clean * mask[None, :, None],
    )
    row = {
        "sample_index": sample_index,
        "sample_seed": int(data["sample_seed"][sample_index]),
        "split": split,
        "family_id": int(data["family_id"][sample_index]),
        "view_count": int(data["view_count"][sample_index]),
        "noise_level": float(data["noise_level"][sample_index]),
        "support_rank": int(decomposition["rank"]),
        "support_nullity": int(decomposition["nullity"]),
        "support_nullity_fraction": float(decomposition["nullity"]) / operator.shape[-1],
        "support_nonzero_condition_number": float(decomposition["condition_number_nonzero"]),
        "base_field_rel_l2": base_field,
        "oracle_null_field_rel_l2": oracle_field,
        "field_improvement_pct": 100.0 * (base_field - oracle_field) / (base_field + 1e-12),
        "error_energy_in_support_nullspace": correction_energy / error_energy,
        "error_energy_remaining_in_support_range": float(np.sum(remaining**2)) / error_energy,
        "decomposition_energy_closure": (correction_energy + float(np.sum(remaining**2))) / error_energy,
        "null_range_orthogonality": orthogonality,
        "support_projection_change_rel_to_clean": support_projection_change,
        "base_support_clean_reprojection": _masked_projection_relative(lift_projection, clean, mask),
        "oracle_null_support_clean_reprojection": _masked_projection_relative(null_projection, clean, mask),
        "base_query_clean_reprojection": _masked_projection_relative(lift_projection, clean, query_mask),
        "oracle_null_query_clean_reprojection": _masked_projection_relative(null_projection, clean, query_mask),
        "query_reprojection_improvement_pct": 100.0
        * (
            _masked_projection_relative(lift_projection, clean, query_mask)
            - _masked_projection_relative(null_projection, clean, query_mask)
        )
        / (_masked_projection_relative(lift_projection, clean, query_mask) + 1e-12),
        "base_gradient_rel_l2": _gradient_relative(lift, target),
        "oracle_null_gradient_rel_l2": _gradient_relative(oracle_null, target),
    }
    return row, {
        "target": target,
        "lift": lift,
        "oracle_null": oracle_null,
        "null_correction": null_correction,
    }


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["split"]), int(row["view_count"]))].append(row)
    output = []
    for (split, view_count), subset in sorted(groups.items()):
        improvement = np.asarray([float(row["field_improvement_pct"]) for row in subset])
        query_improvement = np.asarray(
            [float(row["query_reprojection_improvement_pct"]) for row in subset]
        )
        output.append(
            {
                "split": split,
                "view_count": view_count,
                "sample_count": len(subset),
                "support_rank": int(subset[0]["support_rank"]),
                "support_nullity": int(subset[0]["support_nullity"]),
                "support_nullity_fraction": float(subset[0]["support_nullity_fraction"]),
                "mean_base_field_rel_l2": float(np.mean([float(row["base_field_rel_l2"]) for row in subset])),
                "mean_oracle_null_field_rel_l2": float(
                    np.mean([float(row["oracle_null_field_rel_l2"]) for row in subset])
                ),
                "mean_field_improvement_pct": float(np.mean(improvement)),
                "median_field_improvement_pct": float(np.median(improvement)),
                "p10_field_improvement_pct": float(np.percentile(improvement, 10)),
                "field_improved_fraction": float(np.mean(improvement > 0.0)),
                "mean_error_energy_in_nullspace": float(
                    np.mean([float(row["error_energy_in_support_nullspace"]) for row in subset])
                ),
                "mean_query_reprojection_improvement_pct": float(np.mean(query_improvement)),
                "query_reprojection_improved_fraction": float(np.mean(query_improvement > 0.0)),
                "max_support_projection_change": float(
                    np.max([float(row["support_projection_change_rel_to_clean"]) for row in subset])
                ),
            }
        )
    return output


def plot_layout_capacity(summary_rows: list[dict[str, object]], path: Path) -> None:
    by_view: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        by_view[int(row["view_count"])].append(row)
    views = sorted(by_view)
    nullity = [100.0 * float(by_view[view][0]["support_nullity_fraction"]) for view in views]
    removable = [
        100.0 * float(np.mean([float(row["mean_error_energy_in_nullspace"]) for row in by_view[view]]))
        for view in views
    ]
    field_gain = [
        float(np.mean([float(row["mean_field_improvement_pct"]) for row in by_view[view]]))
        for view in views
    ]
    x = np.arange(len(views))
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), constrained_layout=True)
    axes[0].bar(x - 0.18, nullity, width=0.36, color="#315f8d", label="operator nullity")
    axes[0].bar(x + 0.18, removable, width=0.36, color="#196e63", label="lift-error energy in nullspace")
    axes[0].set_ylabel("fraction (%)")
    axes[0].set_title("How much is invisible to support cameras?")
    axes[0].legend(fontsize=8)
    axes[1].bar(x, field_gain, color="#a65a45")
    axes[1].set_ylabel("oracle-null field improvement (%)")
    axes[1].set_title("Upper-bound value of a learned null correction")
    for ax in axes:
        ax.set_xticks(x, [f"{view} views" for view in views])
        ax.grid(True, axis="y", alpha=0.24)
    fig.suptitle("T16 support-nullspace identifiability audit", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_field_scatter(rows: list[dict[str, object]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6.1), constrained_layout=True)
    colors = {
        "test_iid": "#196e63",
        "test_view_ood": "#315f8d",
        "test_noise_ood": "#bb7a24",
        "test_joint_ood": "#a24d41",
        "test_family_ood": "#70558f",
    }
    for split in SPLIT_ORDER:
        subset = [row for row in rows if row["split"] == split]
        ax.scatter(
            [float(row["base_field_rel_l2"]) for row in subset],
            [float(row["oracle_null_field_rel_l2"]) for row in subset],
            s=28,
            alpha=0.72,
            color=colors.get(split, "#777777"),
            label=split.replace("test_", ""),
        )
    limit = max(
        max(float(row["base_field_rel_l2"]) for row in rows),
        max(float(row["oracle_null_field_rel_l2"]) for row in rows),
    )
    ax.plot([0.0, limit], [0.0, limit], linestyle="--", color="#555555", linewidth=1)
    ax.set_xlabel("physics-lift field relative L2")
    ax.set_ylabel("truth-oracle null-corrected relative L2")
    ax.set_title("Every point below the diagonal is structural headroom")
    ax.grid(True, alpha=0.23)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_preservation(rows: list[dict[str, object]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.3), constrained_layout=True)
    scatter = ax.scatter(
        [float(row["support_projection_change_rel_to_clean"]) for row in rows],
        [float(row["query_reprojection_improvement_pct"]) for row in rows],
        c=[int(row["view_count"]) for row in rows],
        cmap="viridis",
        s=30,
        alpha=0.76,
    )
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=1)
    ax.set_xscale("symlog", linthresh=1e-13)
    ax.set_xlabel("support projection change relative to clean observation")
    ax.set_ylabel("query-view reprojection improvement (%)")
    ax.set_title("Oracle null correction preserves support data by construction")
    ax.grid(True, alpha=0.23)
    fig.colorbar(scatter, ax=ax, label="support view count")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_representative(
    rows: list[dict[str, object]],
    snapshots: dict[int, dict[str, np.ndarray]],
    path: Path,
) -> None:
    eligible = [row for row in rows if int(row["view_count"]) == 3]
    chosen = max(eligible, key=lambda row: float(row["field_improvement_pct"]))
    arrays = snapshots[int(chosen["sample_index"])]
    z_index = arrays["target"].shape[0] // 2
    panels = [
        ("target", arrays["target"][z_index]),
        ("physics lift", arrays["lift"][z_index]),
        ("oracle null corrected", arrays["oracle_null"][z_index]),
        ("null correction", arrays["null_correction"][z_index]),
    ]
    vmax = max(float(np.max(np.abs(panel))) for _, panel in panels[:3])
    correction_limit = float(np.max(np.abs(panels[-1][1]))) + 1e-12
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.7), constrained_layout=True)
    for index, (title, panel) in enumerate(panels):
        if index == 3:
            image = axes[index].imshow(panel, cmap="coolwarm", vmin=-correction_limit, vmax=correction_limit)
        else:
            image = axes[index].imshow(panel, cmap="magma", vmin=0.0, vmax=vmax)
        axes[index].set_title(title)
        axes[index].set_xticks([])
        axes[index].set_yticks([])
        fig.colorbar(image, ax=axes[index], fraction=0.046, pad=0.03)
    fig.suptitle(
        f"Three-view oracle diagnostic: {chosen['split']}, field gain {float(chosen['field_improvement_pct']):.1f}%",
        fontsize=13,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_checksums(output_dir: Path, filenames: list[str]) -> None:
    lines = []
    for filename in filenames:
        digest = hashlib.sha256((output_dir / filename).read_bytes()).hexdigest()
        lines.append(f"{digest}  {filename}")
    (output_dir / "nullspace_checksums.sha256").write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    generate_dataset(config, args.data, force=args.force_data)
    data = load_npz(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    decompositions: dict[tuple[int, ...], dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    snapshots: dict[int, dict[str, np.ndarray]] = {}

    for split, indices in split_indices(data).items():
        if split not in SPLIT_ORDER:
            continue
        for sample_index in indices:
            mask = np.asarray(data["view_mask"][sample_index])
            key = tuple(int(value) for value in mask.tolist())
            if key not in decompositions:
                decompositions[key] = nullspace_decomposition(data["forward_matrix"], mask)
            row, arrays = metric_row(int(sample_index), split, data, decompositions[key])
            rows.append(row)
            snapshots[int(sample_index)] = arrays

    summary_rows = summarize(rows)
    plot_layout_capacity(summary_rows, args.output_dir / "t16_nullspace_capacity.png")
    plot_field_scatter(rows, args.output_dir / "t16_nullspace_field_scatter.png")
    plot_preservation(rows, args.output_dir / "t16_nullspace_support_preservation.png")
    plot_representative(rows, snapshots, args.output_dir / "t16_nullspace_representative.png")
    write_csv(args.output_dir / "nullspace_sample_metrics.csv", rows)
    write_csv(args.output_dir / "nullspace_layout_summary.csv", summary_rows)

    test_rows = [row for row in rows if row["split"] in SPLIT_ORDER]
    report = {
        "status": "completed_oracle_identifiability_audit",
        "experiment": "T16 support-nullspace structural headroom",
        "dataset": {
            "config": str(args.config.relative_to(ROOT.parent)),
            "sample_count": len(rows),
            "test_splits": SPLIT_ORDER,
            "unique_support_masks": len(decompositions),
        },
        "layout_summary": summary_rows,
        "key_findings": {
            "mean_field_improvement_pct": float(
                np.mean([float(row["field_improvement_pct"]) for row in test_rows])
            ),
            "minimum_split_view_mean_improvement_pct": float(
                min(float(row["mean_field_improvement_pct"]) for row in summary_rows)
            ),
            "mean_error_energy_in_support_nullspace": float(
                np.mean([float(row["error_energy_in_support_nullspace"]) for row in test_rows])
            ),
            "maximum_support_projection_change": float(
                max(float(row["support_projection_change_rel_to_clean"]) for row in test_rows)
            ),
            "all_samples_field_improved": bool(
                all(float(row["field_improvement_pct"]) > 0.0 for row in test_rows)
            ),
        },
        "decision_rule": {
            "continue_nullspace_model_if": "substantial oracle field headroom exists across every declared split and support preservation is numerical",
            "stop_or_reframe_if": "headroom vanishes for relevant layouts or query views cannot observe the null correction",
        },
        "claims_boundary": [
            "The null correction is projected from the synthetic ground-truth error and is not available at deployment.",
            "This audit establishes structural headroom only; it does not show that a neural operator can learn the correction.",
            "The projector uses the current linear slice-wise synthetic BOST matrix, not curved rays or OERF calibration.",
            "Nullspace is defined relative to the declared support cameras; adding cameras changes both rank and correction space.",
            "Field truth is used only for retrospective decomposition and upper-bound metrics.",
        ],
    }
    report_path = args.output_dir / "nullspace_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_checksums(
        args.output_dir,
        ["nullspace_sample_metrics.csv", "nullspace_layout_summary.csv", "nullspace_report.json"],
    )
    print(json.dumps(report["key_findings"], indent=2))
    print(f"results: {args.output_dir}")


if __name__ == "__main__":
    main()
