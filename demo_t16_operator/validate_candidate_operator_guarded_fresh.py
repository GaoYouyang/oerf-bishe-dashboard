#!/usr/bin/env python3
"""Independently validate the guarded fresh operator-mechanism bundle."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "results" / "candidate_operator_guarded_fresh"
UNGARDED = ROOT / "results" / "candidate_operator_smoke"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    unguarded_config = json.loads(
        (UNGARDED / "config_snapshot.json").read_text(encoding="utf-8")
    )
    unguarded_report = json.loads((UNGARDED / "report.json").read_text(encoding="utf-8"))
    assert unguarded_report["status"] == "MECHANISM_SMOKE_ONLY"
    assert unguarded_report["claims_boundary"]["superiority_claim_allowed"] is False
    for line in (UNGARDED / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        assert sha256(UNGARDED / name) == expected
    unguarded_samples = load_csv(UNGARDED / "sample_metrics.csv")
    assert len(unguarded_samples) == (
        int(unguarded_config["validation_fields"])
        + len(unguarded_config["test_domains"])
        * int(unguarded_config["test_fields_per_domain"])
    ) * 3
    unguarded_summary = {
        (row["source_split"], row["method"]): row
        for row in load_csv(UNGARDED / "summary.csv")
    }
    assert float(
        unguarded_summary[("noise_ood", "cg_unrolled4")][
            "mean_gain_vs_prewhitened_pg4_pct"
        ]
    ) < 0.0
    assert float(
        unguarded_summary[("joint_ood", "cg_unrolled4")][
            "mean_gain_vs_prewhitened_pg4_pct"
        ]
    ) < -300.0

    config = json.loads((RESULT / "config_snapshot.json").read_text(encoding="utf-8"))
    report = json.loads((RESULT / "report.json").read_text(encoding="utf-8"))
    selection = json.loads((RESULT / "selection_commit.json").read_text(encoding="utf-8"))
    assert report["status"] == "MECHANISM_SMOKE_ONLY"
    assert config["test_seed_base_offset"] == report["test_seed_base_offset"] == 9_000_000
    assert config["claims_boundary"]["superiority_claim_allowed"] is False
    assert config["claims_boundary"]["deployable_noise_estimator_tested"] is False
    assert selection["selection_metric"].startswith("mean source-field")

    for line in (RESULT / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        assert sha256(RESULT / name) == expected

    samples = load_csv(RESULT / "sample_metrics.csv")
    summaries = load_csv(RESULT / "summary.csv")
    expected_fields = int(config["validation_fields"]) + len(config["test_domains"]) * int(
        config["test_fields_per_domain"]
    )
    assert len(samples) == expected_fields * 4
    assert len(summaries) == (1 + len(config["test_domains"])) * 4

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    seeds_by_split: dict[str, set[int]] = defaultdict(set)
    for row in samples:
        grouped[(row["source_split"], row["method"])].append(float(row["field_relative_l2"]))
        seeds_by_split[row["source_split"]].add(int(row["source_seed"]))
    train_seeds = set(
        range(int(config["seed"]), int(config["seed"]) + int(config["train_fields"]))
    )
    validation_seeds = seeds_by_split["validation"]
    test_seeds = set().union(
        *(seeds for split, seeds in seeds_by_split.items() if split != "validation")
    )
    assert not train_seeds & validation_seeds
    assert not train_seeds & test_seeds
    assert not validation_seeds & test_seeds
    assert min(test_seeds) >= int(config["seed"]) + int(config["test_seed_base_offset"])

    summary_lookup = {(row["source_split"], row["method"]): row for row in summaries}
    for key, values in grouped.items():
        stored = float(summary_lookup[key]["mean_relative_l2"])
        assert np.isclose(np.mean(values), stored, atol=1e-7)

    for split in ["noise_ood", "joint_ood"]:
        control = np.asarray(grouped[(split, "prewhitened_pg4")])
        guarded = np.asarray(grouped[(split, "cg_unrolled4_guarded")])
        unguarded = np.asarray(grouped[(split, "cg_unrolled4")])
        guarded_gain = 100.0 * (control - guarded) / control
        unguarded_gain = 100.0 * (control - unguarded) / control
        assert float(np.mean(guarded_gain)) > 0.0
        assert float(np.mean(unguarded_gain)) < 0.0
        assert float(np.mean(guarded_gain < -1.0)) == 0.0

    print(
        "PASS: 552 unguarded + 736 guarded rows; hashes, fresh seed partition, "
        "arithmetic, claims boundary and guarded OOD reversal verified"
    )


if __name__ == "__main__":
    main()
