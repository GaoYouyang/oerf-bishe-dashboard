#!/usr/bin/env python3
"""Select v3k-F deterministic stopping controls on V_tune only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from . import run_v3k_d_strong_numerical_controls as v3d
    from .v3k_f_noise_stopping_common import (
        ROOT,
        build_bundle,
        choose_equivalent,
        load_context,
        parameter_grid,
        read_json,
        screen_row,
        validation_roles,
    )
except ImportError:
    import run_v3k_d_strong_numerical_controls as v3d
    from v3k_f_noise_stopping_common import (
        ROOT,
        build_bundle,
        choose_equivalent,
        load_context,
        parameter_grid,
        read_json,
        screen_row,
        validation_roles,
    )


DEFAULT_CONFIG = ROOT / "configs" / "v3k_f_noise_stopping_gate.json"
FAMILIES = [
    "self_discrepancy",
    "camera_discrepancy",
    "ncp",
    "hybrid",
    "generator_discrepancy",
]
PUBLIC_FILES = [
    "v3k_f_validation_assignment.csv",
    "v3k_f_selection_screen.csv",
    "v3k_f_selection_commit.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    context = load_context(config, args.device)
    checkpoint_hash_before = v3d.sha256(context["checkpoint_path"])
    bundle = build_bundle(context, "val", "validation")
    tune_rows, assignment = validation_roles(bundle, config)
    maximum = int(config["frozen_pbb"]["maximum_iterations"])
    tolerance = float(
        config["selection_protocol"]["mean_rel_l2_equivalence_tolerance"]
    )
    screen: list[dict[str, object]] = []
    selected: dict[str, dict[str, object]] = {}
    for family in FAMILIES:
        family_rows = []
        for rank, parameters in enumerate(parameter_grid(config, family)):
            row = screen_row(
                family,
                parameters,
                rank,
                bundle,
                tune_rows,
                context["ncp_thresholds"],
                maximum,
            )
            screen.append(row)
            family_rows.append(row)
        selected[family] = choose_equivalent(family_rows, tolerance)

    output_dir = ROOT / "results" / str(config["selection_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    v3d.write_csv(output_dir / PUBLIC_FILES[0], assignment)
    v3d.write_csv(output_dir / PUBLIC_FILES[1], screen)
    commit = {
        "experiment": config["name"],
        "phase": "selection_only",
        "selection_role": "v_tune_source_fields",
        "selected": selected,
        "ncp_white_noise_reference": context["ncp_thresholds"],
        "tune_field_count": int(config["validation_partition"]["tune_field_count"]),
        "lock_field_count": int(config["validation_partition"]["lock_field_count"]),
        "layout_rows_per_source": int(config["pair_design"]["repeats_per_source"]),
        "assignment_sha256": hashlib.sha256(
            "\n".join(
                f"{row['sample_seed']}:{row['validation_role']}"
                for row in assignment
            ).encode("ascii")
        ).hexdigest(),
        "selection_metric": config["selection_protocol"]["selection_metric"],
        "equivalence_tolerance": tolerance,
        "tie_break": config["selection_protocol"]["tie_break"],
        "frozen_pbb": context["pbb_choice"],
        "test_dataset_constructed": False,
        "test_metric_computed": False,
        "audit_camera_used": False,
        "per_sample_truth_used_by_deployable_rule": False,
        "generator_discrepancy_is_oracle": True,
        "provenance": {
            "config_sha256": v3d.sha256(args.config),
            "pbb_selection_commit_sha256": v3d.sha256(context["pbb_path"]),
            "baseline_selection_commit_sha256": v3d.sha256(context["baseline_path"]),
            "private_dataset_sha256": v3d.sha256(context["private_path"]),
            "base_checkpoint_sha256_before": checkpoint_hash_before,
            "base_checkpoint_sha256_after": v3d.sha256(context["checkpoint_path"]),
        },
        "private_assets": {
            "private_dataset_published": False,
            "checkpoint_weights_published": False,
            "new_checkpoint_count": 0,
        },
    }
    (output_dir / PUBLIC_FILES[2]).write_text(
        json.dumps(commit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    checksum_lines = [
        f"{v3d.sha256(output_dir / name)}  {name}" for name in PUBLIC_FILES
    ]
    (output_dir / "v3k_f_selection_checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "phase": "selection_only",
                "selected": selected,
                "test_dataset_constructed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
