#!/usr/bin/env python3
"""Independently validate the private v3i dataset and its public audit bundle."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "v3i_variable_geometry_dataset.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    public = ROOT / "results" / str(config["public_output_dir"])
    private = ROOT / "results" / str(config["private_dataset_npz"])
    dashboard = json.loads(
        (public / "v3i_variable_geometry_dashboard.json").read_text(encoding="utf-8")
    )
    assignments = csv_rows(public / "v3i_source_geometry_assignment.csv")
    balance = csv_rows(public / "v3i_geometry_balance.csv")
    channels = csv_rows(public / "v3i_input_channel_manifest.csv")

    assert dashboard["scientific_status"] == "VARIABLE_GEOMETRY_DATASET_GATE_PASS_FUNCTIONAL_TRAINING_NOT_RUN"
    assert dashboard["dataset_gate_pass"] is True
    assert dashboard["sample_count"] == dashboard["unique_source_count"] == 328
    assert dashboard["unique_geometry_count"] == 28
    assert dashboard["input_shape"] == [328, 42, 8, 16, 16]
    assert all(dashboard["gate_checks"].values())
    assert dashboard["next_decision"]["functional_pilot_authorized"] is True
    assert dashboard["next_decision"]["superiority_training_authorized"] is False
    assert dashboard["next_decision"]["blind_final_opened"] is False
    assert dashboard["claims_boundary"]["real_bost_geometry_present"] is False
    assert dashboard["private_dataset"]["public"] is False
    assert dashboard["private_dataset"]["relative_path_recorded_publicly"] is True

    assert len(assignments) == 328 and len(balance) == 28 and len(channels) == 42
    assert {int(row["source_index"]) for row in assignments} == set(range(328))
    assert all(row["assignment_rule"] == "sha256 source ordering + balanced cyclic geometry" for row in assignments)
    assert all(row["mask_bits"][3] == "0" for row in assignments)
    expected_unique = {"train": 16, "val": 4, "test_iid": 4, "test_noise_ood": 4, "test_family_ood": 4, "test_joint_ood": 4}
    for split, unique_count in expected_unique.items():
        selected = [row for row in assignments if row["source_split"] == split]
        counts = Counter(row["geometry_id"] for row in selected)
        assert len(counts) == unique_count
        assert max(counts.values()) == min(counts.values())
    train_ids = {row["geometry_id"] for row in assignments if row["source_split"] == "train"}
    val_ids = {row["geometry_id"] for row in assignments if row["source_split"] == "val"}
    assert train_ids.isdisjoint(val_ids)

    assert private.is_file()
    assert sha256(private) == dashboard["private_dataset"]["sha256"]
    with np.load(private, allow_pickle=False) as archive:
        assert archive["inputs"].shape == (328, 42, 8, 16, 16)
        assert archive["field"].shape == (328, 8, 16, 16)
        assert np.array_equal(archive["source_index"], np.arange(328))
        assert len(set(archive["geometry_id"].tolist())) == 28
        assert np.all(archive["view_mask"].sum(axis=1) == 6)
        assert np.all(archive["view_mask"][:, 3] == 0)
        assert np.all(np.isfinite(archive["inputs"]))
        names = archive["input_channel_names"].tolist()
        audit_channels = [names.index("camera_3_active"), int(archive["ray_view_channel_start"]) + 3, int(archive["ray_angle_sin_channel_start"]) + 3, int(archive["ray_angle_cos_channel_start"]) + 3]
        assert np.all(archive["inputs"][:, audit_channels] == 0)
        assert bool(archive["shared_full_view_noise"])

    assert not list(public.glob("*.npz"))
    assert not list(public.glob("*.pt"))
    assert not list(public.glob("*.pth"))
    assert not list(public.glob("*.pdf"))
    assert (public / "t16_v3i_variable_geometry_dataset.png").stat().st_size > 20_000
    for line in (public / "v3i_variable_geometry_checksums.sha256").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        assert sha256(public / name.strip()) == expected
    print(json.dumps({
        "status": "PASS",
        "samples": len(assignments),
        "geometries": len(balance),
        "channels": len(channels),
        "private_npz_sha256_verified": True,
        "functional_pilot_authorized": True,
        "superiority_authorized": False,
    }, indent=2))


if __name__ == "__main__":
    main()
