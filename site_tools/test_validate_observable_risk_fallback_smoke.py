from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Callable

import pytest

from site_tools.run_observable_risk_fallback_smoke import (
    OUTPUT_PAYLOADS,
    load_config,
    run_smoke,
)
from site_tools.test_run_observable_risk_fallback_smoke import CONFIG_PATH
from site_tools.validate_observable_risk_fallback_smoke import validate_result_bundle


@pytest.fixture(scope="module")
def baseline_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("rccf-baseline") / "result"
    run_smoke(load_config(CONFIG_PATH), output_dir=output)
    return output


def _refresh_checksums(output: Path) -> None:
    (output / "checksums.sha256").write_text(
        "".join(
            f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}\n"
            for name in OUTPUT_PAYLOADS
        ),
        encoding="ascii",
    )


def _rewrite_csv(path: Path, mutate: Callable[[list[dict[str, str]]], None]) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames
    mutate(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _risk_score(output: Path) -> None:
    first_id: list[str] = []
    def selection(rows: list[dict[str, str]]) -> None:
        rows[0]["risk_score"] = "0.999"
        first_id.append(rows[0]["rig_id"])
    _rewrite_csv(output / "selection_rows.csv", selection)
    _rewrite_csv(
        output / "risk_rows.csv",
        lambda rows: [row.__setitem__("risk_score", "0.999") for row in rows if row["rig_id"] == first_id[0]],
    )


def _threshold(output: Path) -> None:
    _rewrite_csv(
        output / "selection_rows.csv",
        lambda rows: [row.__setitem__("acceptance_threshold", "0.999") for row in rows],
    )
    _rewrite_csv(
        output / "risk_rows.csv",
        lambda rows: [row.__setitem__("acceptance_threshold", "0.999") for row in rows],
    )
    path = output / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["risk_calibration"]["acceptance_threshold"] = 0.999
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _fallback_flag(output: Path) -> None:
    rig_id: list[str] = []
    def selection(rows: list[dict[str, str]]) -> None:
        rows[0]["fallback_used"] = "False" if rows[0]["fallback_used"] == "True" else "True"
        rig_id.append(rows[0]["rig_id"])
    _rewrite_csv(output / "selection_rows.csv", selection)
    _rewrite_csv(
        output / "risk_rows.csv",
        lambda rows: [
            row.__setitem__("fallback_used", "False" if row["fallback_used"] == "True" else "True")
            for row in rows
            if row["rig_id"] == rig_id[0]
        ],
    )


def _split_role(output: Path) -> None:
    _rewrite_csv(
        output / "risk_rows.csv",
        lambda rows: rows[0].__setitem__(
            "split_role",
            "risk_calibration"
            if rows[0]["split_role"] == "fresh_geometry_ood"
            else "fresh_geometry_ood",
        ),
    )


def _all_in_one(output: Path) -> None:
    rig_id: list[str] = []
    def selection(rows: list[dict[str, str]]) -> None:
        rows[0]["candidate_partition"] = "all_in_one_exact"
        rig_id.append(rows[0]["rig_id"])
    _rewrite_csv(output / "selection_rows.csv", selection)
    _rewrite_csv(
        output / "risk_rows.csv",
        lambda rows: [row.__setitem__("candidate_partition", "all_in_one_exact") for row in rows if row["rig_id"] == rig_id[0]],
    )


def _feature_hash(output: Path) -> None:
    _rewrite_csv(
        output / "selection_rows.csv",
        lambda rows: rows[0].__setitem__("observable_feature_sha256", "0" * 64),
    )


def _uses_truth(output: Path) -> None:
    _rewrite_csv(
        output / "selection_rows.csv",
        lambda rows: rows[0].__setitem__("uses_truth", "True"),
    )


def _support_gate(output: Path) -> None:
    rig_id: list[str] = []

    def selection(rows: list[dict[str, str]]) -> None:
        rows[0]["support_gate_passed"] = (
            "False" if rows[0]["support_gate_passed"] == "True" else "True"
        )
        rig_id.append(rows[0]["rig_id"])

    _rewrite_csv(output / "selection_rows.csv", selection)
    _rewrite_csv(
        output / "risk_rows.csv",
        lambda rows: [
            row.__setitem__(
                "support_gate_passed",
                "False" if row["support_gate_passed"] == "True" else "True",
            )
            for row in rows
            if row["rig_id"] == rig_id[0]
        ],
    )


def _multiplicity_correction(output: Path) -> None:
    path = output / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["risk_calibration"]["multiplicity_correction"] = "NONE"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _aggregate(output: Path) -> None:
    path = output / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["aggregate"]["fresh_worst_takeover_field_harm"] = 999.0
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    "tamper",
    [
        _risk_score,
        _threshold,
        _fallback_flag,
        _split_role,
        _all_in_one,
        _feature_hash,
        _uses_truth,
        _support_gate,
        _multiplicity_correction,
        _aggregate,
    ],
    ids=[
        "risk-score",
        "threshold",
        "fallback-flag",
        "split-role",
        "all-in-one-injection",
        "feature-hash",
        "uses-truth",
        "support-gate",
        "multiplicity-correction",
        "aggregate",
    ],
)
def test_validator_rejects_synchronized_tamper_with_refreshed_checksums(
    baseline_bundle: Path,
    tmp_path: Path,
    tamper: Callable[[Path], None],
) -> None:
    output = tmp_path / "result"
    shutil.copytree(baseline_bundle, output)
    tamper(output)
    _refresh_checksums(output)
    with pytest.raises(ValueError, match="mismatch"):
        validate_result_bundle(output)


def test_validator_accepts_untampered_rebuilt_bundle(baseline_bundle: Path) -> None:
    report = validate_result_bundle(baseline_bundle)
    assert report["gates"]["research_claim_authorized"] is False
    assert report["observable_only_contract"]["uses_truth"] is False


def test_clean_source_mode_rejects_unbound_source_commit(
    baseline_bundle: Path, tmp_path: Path
) -> None:
    output = tmp_path / "result"
    shutil.copytree(baseline_bundle, output)
    path = output / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["provenance"]["source_commit"] = "0" * 40
    report["provenance"]["source_worktree_dirty"] = False
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _refresh_checksums(output)
    with pytest.raises(ValueError, match="source commit"):
        validate_result_bundle(output, require_clean_source=True)
