#!/usr/bin/env python3
"""Build and audit the private v3i balanced variable-geometry dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .data import generate_dataset, load_npz
    from .variable_geometry import (
        assign_geometry_partitions,
        build_geometry_manifest,
        build_variable_geometry_operator_data,
        deterministic_noisy_observations,
        geometry_entropy_bits,
        reference_mask_id,
    )
except ImportError:
    from data import generate_dataset, load_npz
    from variable_geometry import (
        assign_geometry_partitions,
        build_geometry_manifest,
        build_variable_geometry_operator_data,
        deterministic_noisy_observations,
        geometry_entropy_bits,
        reference_mask_id,
    )


ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = ROOT / "configs"
DEFAULT_CONFIG = CONFIG_ROOT / "v3i_variable_geometry_dataset.json"
PUBLIC_FILES = [
    "v3i_source_geometry_assignment.csv",
    "v3i_geometry_balance.csv",
    "v3i_input_channel_manifest.csv",
    "v3i_variable_geometry_dashboard.json",
    "v3i_variable_geometry_report.json",
    "t16_v3i_variable_geometry_dataset.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--force-base-data", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def save_dataset(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def build_balance_rows(
    assignments: list[dict[str, object]],
    manifest_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    split_names = sorted({str(row["source_split"]) for row in assignments})
    counts = Counter(
        (str(row["source_split"]), str(row["geometry_id"]))
        for row in assignments
    )
    output = []
    for manifest in manifest_rows:
        identifier = str(manifest["geometry_id"])
        row: dict[str, object] = {
            "geometry_id": identifier,
            "geometry_partition": str(manifest["partition"]),
            "mask_bits": str(manifest["mask_bits"]),
            "active_angles_degrees": str(manifest["active_angles_degrees"]),
            "maximum_gap_degrees": float(manifest["maximum_gap_degrees"]),
            "operator_condition_nonzero": float(
                manifest["operator_condition_nonzero"]
            ),
        }
        for split_name in split_names:
            row[f"count_{split_name}"] = int(counts[(split_name, identifier)])
        row["total_count"] = sum(int(row[f"count_{name}"]) for name in split_names)
        output.append(row)
    return output


def plot_audit(
    output_path: Path,
    assignments: list[dict[str, object]],
    manifest_rows: list[dict[str, object]],
    data: dict[str, np.ndarray],
) -> None:
    geometry_ids = [str(row["geometry_id"]) for row in manifest_rows]
    split_names = [str(value) for value in data["split_names"].tolist()]
    count_matrix = np.zeros((len(split_names), len(geometry_ids)), dtype=int)
    lookup = {identifier: index for index, identifier in enumerate(geometry_ids)}
    for row in assignments:
        count_matrix[split_names.index(str(row["source_split"])), lookup[str(row["geometry_id"])]] += 1
    camera_counts = np.zeros((len(split_names), data["view_mask"].shape[1]), dtype=int)
    for split_index in range(len(split_names)):
        camera_counts[split_index] = data["view_mask"][data["split_id"] == split_index].sum(axis=0)
    entropy = [
        geometry_entropy_bits(data["view_mask"][data["split_id"] == index])
        for index in range(len(split_names))
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.3, 4.8), constrained_layout=True)
    image = axes[0].imshow(count_matrix, cmap="viridis", aspect="auto")
    axes[0].set_title("One field, one assigned geometry")
    axes[0].set_yticks(range(len(split_names)), split_names)
    axes[0].set_xticks(range(len(geometry_ids)), [value[2:] for value in geometry_ids], rotation=90, fontsize=6)
    fig.colorbar(image, ax=axes[0], label="fields")

    x = np.arange(data["view_mask"].shape[1])
    width = 0.12
    for split_index, split_name in enumerate(split_names):
        axes[1].bar(x + (split_index - 2.5) * width, camera_counts[split_index], width, label=split_name)
    axes[1].axvline(3, color="#c94f45", linestyle="--", linewidth=1.5, label="Q_audit 60 deg")
    axes[1].set_title("Active-camera coverage by source split")
    axes[1].set_xlabel("camera index")
    axes[1].set_ylabel("assigned fields")
    axes[1].legend(fontsize=7, ncol=2)

    colors = ["#247ba0", "#2a9d8f", "#6c757d", "#6c757d", "#c9823b", "#c9823b"]
    axes[2].barh(split_names, entropy, color=colors)
    axes[2].set_title("Geometry entropy within each split")
    axes[2].set_xlabel("bits")
    for index, value in enumerate(entropy):
        axes[2].text(value + 0.03, index, f"{value:.2f}", va="center")
    fig.suptitle("v3i balanced variable-acquisition dataset audit", fontsize=14)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    base_config_path = CONFIG_ROOT / str(config["base_dataset_config"])
    base_config = read_json(base_config_path)
    base_path = ROOT / "results" / str(config["base_dataset_npz"])
    generate_dataset(base_config, base_path, force=args.force_base_data)
    base_data = load_npz(base_path)

    manifest_rows, masks = build_geometry_manifest(
        base_data["angles"],
        base_data["forward_matrix"],
        int(config["total_budget"]),
        int(config["audit_query_index"]),
        float(config["geometry_period_degrees"]),
    )
    manifest_rows = assign_geometry_partitions(
        manifest_rows,
        reference_mask_id(config["reference_mask"]),
        int(config["geometry_partition_seed"]),
        {key: int(value) for key, value in config["geometry_partition_counts"].items()},
    )
    data, assignments = build_variable_geometry_operator_data(
        base_data,
        manifest_rows,
        masks,
        {str(key): str(value) for key, value in config["split_partition_map"].items()},
        int(config["assignment_seed"]),
        float(config["ridge_relative"]),
        int(config["total_budget"]),
        int(config["audit_query_index"]),
    )
    private_path = ROOT / "results" / str(config["private_dataset_npz"])
    save_dataset(private_path, data)

    manifest_lookup = {str(row["geometry_id"]): row for row in manifest_rows}
    assignment_rows = []
    for row in assignments:
        manifest = manifest_lookup[str(row["geometry_id"])]
        assignment_rows.append(
            {
                **row,
                "mask_bits": str(manifest["mask_bits"]),
                "active_angles_degrees": str(manifest["active_angles_degrees"]),
                "family_id": int(base_data["family_id"][int(row["source_index"])]),
                "noise_level": float(base_data["noise_level"][int(row["source_index"])]),
            }
        )
    balance_rows = build_balance_rows(assignments, manifest_rows)
    channel_rows = [
        {"channel_index": index, "channel_name": str(name)}
        for index, name in enumerate(data["input_channel_names"].tolist())
    ]

    split_names = [str(value) for value in data["split_names"].tolist()]
    split_audit = {}
    imbalance = []
    for split_id, split_name in enumerate(split_names):
        selected = data["split_id"] == split_id
        identifiers = data["geometry_id"][selected].tolist()
        counts = Counter(str(value) for value in identifiers)
        imbalance.append(max(counts.values()) - min(counts.values()))
        split_audit[split_name] = {
            "source_fields": int(np.sum(selected)),
            "geometry_partition": str(config["split_partition_map"][split_name]),
            "unique_geometries": len(counts),
            "geometry_entropy_bits": geometry_entropy_bits(data["view_mask"][selected]),
            "minimum_fields_per_geometry": min(counts.values()),
            "maximum_fields_per_geometry": max(counts.values()),
        }
    expected_noise = deterministic_noisy_observations(
        base_data, np.arange(len(base_data["field"])), int(config["total_budget"])
    )
    names = [str(value) for value in data["input_channel_names"].tolist()]
    audit_index = int(config["audit_query_index"])
    audit_channels = [
        names.index(f"camera_{audit_index}_active"),
        int(data["ray_view_channel_start"]) + audit_index,
        int(data["ray_angle_sin_channel_start"]) + audit_index,
        int(data["ray_angle_cos_channel_start"]) + audit_index,
    ]
    train_ids = set(
        data["geometry_id"][data["split_id"] == split_names.index("train")].tolist()
    )
    val_ids = set(
        data["geometry_id"][data["split_id"] == split_names.index("val")].tolist()
    )
    gate_checks = {
        "one_geometry_per_source": len(np.unique(data["source_index"])) == len(data["field"]),
        "all_28_geometries_present": len(set(data["geometry_id"].tolist())) == 28,
        "within_split_assignment_exactly_balanced": max(imbalance) <= int(config["gate"]["maximum_within_split_geometry_count_imbalance"]),
        "shared_full_view_noise_exact": bool(np.array_equal(data["observation"], expected_noise)),
        "audit_camera_mask_excluded": bool(np.all(data["view_mask"][:, audit_index] == 0)),
        "audit_camera_input_channels_zero": bool(np.all(data["inputs"][:, audit_channels] == 0)),
        "train_validation_geometries_disjoint": train_ids.isdisjoint(val_ids),
        "inputs_and_targets_finite": bool(np.all(np.isfinite(data["inputs"])) and np.all(np.isfinite(data["field"]))),
        "input_channel_contract_42": data["inputs"].shape[1] == 42,
    }
    dataset_gate_pass = all(gate_checks.values())

    public_dir = ROOT / "results" / str(config["public_output_dir"])
    public_dir.mkdir(parents=True, exist_ok=True)
    write_csv(public_dir / PUBLIC_FILES[0], assignment_rows)
    write_csv(public_dir / PUBLIC_FILES[1], balance_rows)
    write_csv(public_dir / PUBLIC_FILES[2], channel_rows)
    plot_audit(public_dir / PUBLIC_FILES[-1], assignments, manifest_rows, data)

    dashboard = {
        "experiment": config["name"],
        "scientific_status": "VARIABLE_GEOMETRY_DATASET_GATE_PASS_FUNCTIONAL_TRAINING_NOT_RUN" if dataset_gate_pass else "VARIABLE_GEOMETRY_DATASET_GATE_FAIL",
        "dataset_gate_pass": dataset_gate_pass,
        "sample_count": int(len(data["field"])),
        "unique_source_count": int(len(np.unique(data["source_index"]))),
        "unique_geometry_count": len(set(data["geometry_id"].tolist())),
        "input_shape": list(data["inputs"].shape),
        "input_channel_count": int(data["inputs"].shape[1]),
        "split_audit": split_audit,
        "gate_checks": gate_checks,
        "private_dataset": {
            "public": False,
            "relative_path_recorded_publicly": True,
            "sha256": sha256(private_path),
            "compressed_bytes": private_path.stat().st_size,
        },
        "next_decision": {
            "functional_pilot_authorized": dataset_gate_pass,
            "controls": ["locked_fno", "static", "k_cardinality", "shuffled_geometry", "correct_geometry"],
            "seeds": 3,
            "superiority_training_authorized": False,
            "blind_final_opened": False,
        },
        "claims_boundary": config["claims_boundary"],
    }
    report = {
        "status": dashboard["scientific_status"],
        "protocol": {
            "one_field_one_geometry": True,
            "shared_full_view_noise": True,
            "geometry_partition_fixed_before_training": True,
            "source_split_geometry_partition_map": config["split_partition_map"],
            "ridge_relative": config["ridge_relative"],
        },
        "dashboard": dashboard,
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
        "provenance": {
            "config_sha256": sha256(args.config),
            "base_config_sha256": sha256(base_config_path),
            "base_dataset_sha256": sha256(base_path),
            "builder_sha256": sha256(Path(__file__).resolve()),
            "geometry_module_sha256": sha256(ROOT / "variable_geometry.py"),
        },
    }
    (public_dir / PUBLIC_FILES[3]).write_text(json.dumps(dashboard, indent=2), encoding="utf-8")
    (public_dir / PUBLIC_FILES[4]).write_text(json.dumps(report, indent=2), encoding="utf-8")
    checksum_lines = [f"{sha256(public_dir / name)}  {name}" for name in PUBLIC_FILES]
    (public_dir / "v3i_variable_geometry_checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "scientific_status": dashboard["scientific_status"],
        "samples": dashboard["sample_count"],
        "geometries": dashboard["unique_geometry_count"],
        "input_shape": dashboard["input_shape"],
        "private_npz_mib": round(private_path.stat().st_size / 1048576, 2),
        "split_audit": split_audit,
    }, indent=2))


if __name__ == "__main__":
    main()
