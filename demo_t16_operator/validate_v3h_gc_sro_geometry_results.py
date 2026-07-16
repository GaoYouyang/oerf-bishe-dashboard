#!/usr/bin/env python3
"""Independently validate the v3h geometry-identifiability evidence bundle."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "v3h_gc_sro_geometry_gate"
EXPECTED_STATUS = (
    "CURRENT_FIXED_GEOMETRY_FAIL_VARIABLE_PROTOCOL_READY_GC_SRO_ENGINEERING_PASS"
)


def rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    dashboard = json.loads(
        (RESULTS / "v3h_gc_sro_geometry_dashboard.json").read_text(encoding="utf-8")
    )
    assert dashboard["scientific_status"] == EXPECTED_STATUS
    assert dashboard["development_only"] is True
    assert dashboard["superiority_tested"] is False
    assert dashboard["blind_final_opened"] is False

    manifest = rows("v3h_geometry_manifest.csv")
    field_errors = rows("v3h_geometry_field_errors.csv")
    summaries = rows("v3h_geometry_summary.csv")
    spreads = rows("v3h_field_geometry_spread.csv")
    gates = rows("v3h_geometry_gate.csv")
    controls = rows("v3h_gc_sro_control_contract.csv")
    assert (len(manifest), len(field_errors), len(summaries), len(spreads)) == (
        28,
        1120,
        28,
        40,
    )
    assert len(gates) == 2 and len(controls) == 4
    assert Counter(row["partition"] for row in manifest) == {
        "train": 16,
        "validation": 4,
        "geometry_ood": 4,
        "stress": 4,
    }
    assert all(row["mask_bits"][3] == "0" for row in manifest)
    assert all(row["active_view_count"] == "6" for row in manifest)
    reference = [row for row in manifest if row["reference_fixed_k6_geometry"] == "True"]
    assert len(reference) == 1 and reference[0]["partition"] == "train"
    assert all(
        row["partition_selection_rule"]
        == "largest max-gap then nonzero condition; field errors unused"
        for row in manifest
        if row["partition"] == "stress"
    )
    assert all(
        "field" not in row["partition_selection_rule"]
        for row in manifest
        if row["partition"] != "stress"
    )

    diagnostic = dashboard["geometry_diagnostics"]
    assert diagnostic["current_unique_geometry_masks"] == 1
    assert diagnostic["current_geometry_conditioning_identifiable"] is False
    assert diagnostic["variable_unique_geometry_masks"] == 28
    assert diagnostic["variable_changed_fraction_after_deterministic_shuffle"] == 1.0
    assert diagnostic["mean_ridge_error_range_pct_across_geometries"] > 5.0
    assert diagnostic["median_field_best_worst_spread_pct"] > 10.0
    assert diagnostic["operator_condition_cv"] > 0.05
    assert diagnostic["variable_geometry_protocol_ready"] is True

    engineering = dashboard["gc_sro_engineering_gate"]
    assert engineering["engineering_gate_pass"] is True
    assert engineering["maximum_abs_initial_difference_vs_base_fno"] == 0.0
    assert engineering["maximum_abs_initial_head_weight"] == 0.0
    assert engineering["maximum_abs_initial_head_bias"] == 0.0
    assert engineering["maximum_frozen_base_parameter_drift"] == 0.0
    assert engineering["head_gradient_norm_after_first_backward"] > 0.0
    assert engineering["conditioner_gradient_norm_after_last_backward"] > 0.0
    assert engineering["correction_l2_after_checked_steps"] > 0.0
    assert engineering["current_fixed_protocol_geometry_vs_shuffled_embedding_l2"] == 0.0
    assert engineering["variable_geometry_mean_pairwise_embedding_l2"] > 0.0
    assert engineering["variable_mask_only_mean_pairwise_embedding_l2"] < 1e-7
    assert engineering["variable_static_mean_pairwise_embedding_l2"] < 1e-7
    assert engineering["variable_geometry_vs_shuffled_embedding_l2"] > 0.0
    assert engineering["maximum_joint_camera_permutation_embedding_difference"] < 1e-6
    assert len({row["combined_total_parameters"] for row in controls}) == 1
    assert all(row["parameter_matched_across_descriptor_modes"] == "True" for row in controls)

    decision = dashboard["training_decision"]
    assert decision["train_gc_sro_on_current_fixed_k6"] is False
    assert decision["build_variable_geometry_functional_pilot"] is True
    assert decision["superiority_training_authorized"] is False
    assert decision["blind_final_opened"] is False
    assert dashboard["provenance"]["dataset_npz_public"] is False
    assert dashboard["provenance"]["checkpoint_weights_public"] is False
    assert not list(RESULTS.glob("*.pt"))
    assert not list(RESULTS.glob("*.pth"))
    assert not list(RESULTS.glob("*.npz"))
    assert not list(RESULTS.glob("*.pdf"))
    assert (RESULTS / "t16_v3h_gc_sro_geometry_gate.png").stat().st_size > 20_000

    checksum_lines = (
        RESULTS / "v3h_gc_sro_geometry_checksums.sha256"
    ).read_text(encoding="utf-8").splitlines()
    for line in checksum_lines:
        expected, name = line.split(maxsplit=1)
        assert sha256(RESULTS / name.strip()) == expected
    print(
        json.dumps(
            {
                "status": "PASS",
                "geometry_masks": len(manifest),
                "field_metric_rows": len(field_errors),
                "controls": len(controls),
                "current_geometry_identifiable": False,
                "variable_protocol_ready": True,
                "superiority_authorized": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
