#!/usr/bin/env python3
"""Verify the v3c zero-init fallback and commit a disjoint dev2 protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

try:
    from .data import generate_dataset, load_npz, split_indices
    from .direct_operator_data import prepare_direct_operator_data, replace_lift_with_ridge
    from .models import make_model
    from .own_algorithm_data import append_ray_view_channels
    from .own_algorithm_models import ZeroInitializedRaySetAdapter
    from .run_direct_operator_pilot import tune_classical_baselines
    from .train_eval import set_seed
except ImportError:
    from data import generate_dataset, load_npz, split_indices
    from direct_operator_data import prepare_direct_operator_data, replace_lift_with_ridge
    from models import make_model
    from own_algorithm_data import append_ray_view_channels
    from own_algorithm_models import ZeroInitializedRaySetAdapter
    from run_direct_operator_pilot import tune_classical_baselines
    from train_eval import set_seed


ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = ROOT / "configs"
DEFAULT_CONFIG = CONFIG_ROOT / "v3c_protocol_gate.json"
DEFAULT_OUTPUT = ROOT / "results" / "v3c_protocol_gate"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_hashes(values: np.ndarray) -> set[str]:
    return {
        hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        for value in values
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def split_commitments(data: dict[str, np.ndarray]) -> list[dict[str, object]]:
    names = [str(value) for value in data["split_names"].tolist()]
    rows = []
    for split_id, name in enumerate(names):
        selected = np.flatnonzero(data["split_id"] == split_id)
        seed_bytes = np.asarray(data["sample_seed"][selected], dtype="<i8").tobytes()
        rows.append(
            {
                "split": name,
                "field_count": len(selected),
                "sample_seed_sha256": hashlib.sha256(seed_bytes).hexdigest(),
                "family_ids": ",".join(str(value) for value in sorted(set(data["family_id"][selected].tolist()))),
                "noise_levels": ",".join(
                    f"{value:.4f}" for value in sorted(set(data["noise_level"][selected].tolist()))
                ),
            }
        )
    return rows


def build_adapter(
    data: dict[str, np.ndarray],
    dataset_config: dict,
    adapter_config: dict,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    names = [str(value) for value in data["input_channel_names"].tolist()]
    base = make_model(
        "fno",
        dataset_config["models"]["fno"],
        int(data["inputs"].shape[1]),
        residual=True,
    )
    adapter = ZeroInitializedRaySetAdapter(
        base_operator=base,
        view_count=int(data["ray_view_channel_count"]),
        view_channel_start=int(data["ray_view_channel_start"]),
        mask_channel_start=names.index("camera_0_active"),
        angle_sin_channel_start=int(data["ray_angle_sin_channel_start"]),
        angle_cos_channel_start=int(data["ray_angle_cos_channel_start"]),
        coordinate_channels=tuple(names.index(axis) for axis in ("z", "y", "x")),
        view_features=int(adapter_config["view_features"]),
        adapter_hidden=int(adapter_config["adapter_hidden"]),
        gate_hidden=int(adapter_config["gate_hidden"]),
        maximum_correction_scale=float(adapter_config["maximum_correction_scale"]),
        freeze_base=bool(adapter_config["freeze_base"]),
    )
    return base, adapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force-data", action="store_true")
    args = parser.parse_args()

    protocol = read_json(args.config)
    current_config_path = CONFIG_ROOT / protocol["current_development_config"]
    dev2_config_path = CONFIG_ROOT / protocol["dev2_config"]
    blind_config_path = CONFIG_ROOT / protocol["blind_preregistration"]
    current_config = read_json(current_config_path)
    dev2_config = read_json(dev2_config_path)
    blind_config = read_json(blind_config_path)
    current_path = ROOT / "results" / "t16_direct_operator_fields_v1_dataset.npz"
    dev2_path = ROOT / "results" / "t16_v3c_dev2_dataset.npz"
    generate_dataset(current_config, current_path, force=False)
    generate_dataset(dev2_config, dev2_path, force=bool(args.force_data))
    current = load_npz(current_path)
    dev2 = load_npz(dev2_path)

    current_seeds = set(int(value) for value in current["sample_seed"].tolist())
    dev2_seeds = set(int(value) for value in dev2["sample_seed"].tolist())
    seed_overlap = current_seeds & dev2_seeds
    field_hash_overlap = row_hashes(current["field"]) & row_hashes(dev2["field"])
    observation_hash_overlap = row_hashes(current["clean_observation"]) & row_hashes(
        dev2["clean_observation"]
    )
    budgets = [int(value) for value in protocol["reconstruction_budgets"]]
    direct = prepare_direct_operator_data(
        dev2,
        budgets,
        int(protocol["fixed_query_index"]),
        int(protocol["audit_query_index"]),
    )
    selected_ridge, champions, _ = tune_classical_baselines(
        direct,
        budgets,
        [float(value) for value in protocol["ridge_relative_grid"]],
    )
    ridge = replace_lift_with_ridge(direct, selected_ridge)
    ray_data = append_ray_view_channels(ridge)
    exactness_indices = []
    for split_values in split_indices(ray_data).values():
        for budget in budgets:
            candidates = split_values[ray_data["total_budget"][split_values] == budget]
            exactness_indices.extend(int(value) for value in candidates[:2])
    exactness_indices = np.asarray(exactness_indices, dtype=np.int64)
    x = torch.from_numpy(ray_data["inputs"][exactness_indices])
    target = torch.from_numpy(ray_data["field"][exactness_indices, None])

    set_seed(int(protocol["model_seed"]))
    base, adapter = build_adapter(ray_data, dev2_config, protocol["adapter"])
    base.eval()
    adapter.eval()
    with torch.no_grad():
        base_prediction = base(x)
        adapter_prediction = adapter(x)
        zero_correction, gate = adapter.correction(x)
    exact_difference = float(torch.max(torch.abs(adapter_prediction - base_prediction)))
    initial_correction_max = float(torch.max(torch.abs(zero_correction)))
    initial_head_weight_max = float(
        torch.max(torch.abs(adapter.adapter[-1].weight.detach()))
    )
    initial_head_bias_max = float(torch.max(torch.abs(adapter.adapter[-1].bias.detach())))
    gate_min = float(gate.min())
    gate_max = float(gate.max())

    base_state = {
        name: value.detach().clone()
        for name, value in base.state_dict().items()
        if torch.is_tensor(value)
    }
    trainable = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.0)
    adapter.train()
    head_gradient_norm = 0.0
    optimizer_steps_checked = 3
    for step in range(optimizer_steps_checked):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((adapter(x) - target) ** 2)
        loss.backward()
        if step == 0:
            head_gradient_norm = float(
                torch.linalg.vector_norm(adapter.adapter[-1].weight.grad).detach()
            )
        optimizer.step()
    with torch.no_grad():
        trained_correction, _ = adapter.correction(x)
    base_drift = max(
        float(torch.max(torch.abs(value - base_state[name])))
        for name, value in base.state_dict().items()
        if torch.is_tensor(value)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    commitment_rows = split_commitments(dev2)
    write_csv(args.output_dir / "v3c_dev2_split_commitments.csv", commitment_rows)
    trainable_parameters = sum(parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad)
    total_parameters = sum(parameter.numel() for parameter in adapter.parameters())
    base_parameters = sum(parameter.numel() for parameter in base.parameters())
    architecture_gate = (
        exact_difference == 0.0
        and initial_correction_max == 0.0
        and initial_head_weight_max == 0.0
        and initial_head_bias_max == 0.0
        and base_drift == 0.0
        and head_gradient_norm > 0.0
        and float(torch.linalg.vector_norm(trained_correction)) > 0.0
        and not seed_overlap
        and not field_hash_overlap
        and not observation_hash_overlap
    )
    report = {
        "experiment": protocol["name"],
        "scientific_status": "V3C_ARCHITECTURE_GATE_PASS_BLIND_FINAL_NOT_SEALED"
        if architecture_gate
        else "V3C_ARCHITECTURE_GATE_FAIL",
        "architecture_gate_pass": architecture_gate,
        "claims_boundary": "No reconstruction superiority was tested in this protocol gate.",
        "zero_initialization": {
            "maximum_absolute_output_difference_vs_base_fno": exact_difference,
            "maximum_absolute_initial_correction": initial_correction_max,
            "maximum_absolute_initial_head_weight": initial_head_weight_max,
            "maximum_absolute_initial_head_bias": initial_head_bias_max,
            "stratified_exactness_sample_count": int(len(exactness_indices)),
            "gate_min": gate_min,
            "gate_max": gate_max,
            "head_gradient_norm_after_first_backward": head_gradient_norm,
            "optimizer_steps_checked": optimizer_steps_checked,
            "correction_l2_after_checked_steps": float(torch.linalg.vector_norm(trained_correction)),
            "maximum_frozen_base_parameter_drift": base_drift,
        },
        "parameters": {
            "base_fno_total": base_parameters,
            "adapter_trainable": trainable_parameters,
            "combined_total": total_parameters,
            "base_frozen": all(not parameter.requires_grad for parameter in base.parameters()),
        },
        "dev2": {
            "field_count": int(len(dev2["field"])),
            "split_counts": {row["split"]: row["field_count"] for row in commitment_rows},
            "sample_seed_overlap_with_v3b": len(seed_overlap),
            "field_content_hash_overlap_with_v3b": len(field_hash_overlap),
            "clean_observation_hash_overlap_with_v3b": len(observation_hash_overlap),
            "config_sha256": sha256(dev2_config_path),
            "dataset_plaintext_not_committed": True,
        },
        "blind_final": {
            "status": blind_config["status"],
            "preregistration_sha256": sha256(blind_config_path),
            "seed_stored_in_repository": False,
            "confirmatory_claim_allowed": False,
            "unseal_conditions": blind_config["unseal_conditions"],
        },
        "protocol_hashes": {
            "gate_config_sha256": sha256(args.config),
            "current_development_config_sha256": sha256(current_config_path),
            "dev2_config_sha256": sha256(dev2_config_path),
            "blind_preregistration_sha256": sha256(blind_config_path),
        },
        "ridge_validation_champions": {str(key): value for key, value in champions.items()},
        "selected_ridge_relative": {str(key): value for key, value in selected_ridge.items()},
    }
    report_path = args.output_dir / "v3c_protocol_gate.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    checksum_files = ["v3c_dev2_split_commitments.csv", "v3c_protocol_gate.json"]
    checksum_lines = [f"{sha256(args.output_dir / name)}  {name}" for name in checksum_files]
    (args.output_dir / "v3c_protocol_checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="ascii"
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"results: {args.output_dir}")


if __name__ == "__main__":
    main()
