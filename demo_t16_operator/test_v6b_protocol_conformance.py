import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "v6b_protocol_conformance.json"
RESULTS = ROOT / "results" / "v6b_protocol_conformance"


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_report_keeps_scientific_gate_closed() -> None:
    report = _load_json(RESULTS / "report.json")
    assert report["decision"] == "PASS_PROTOCOL_CONFORMANCE_ONLY"
    assert report["fresh_v6b_status"] == "UNCONSTRUCTED"
    assert report["scientific_claims_unlocked"] == []
    assert report["evidence_scope"] == "TOY_PROTOCOL_CONFORMANCE_ONLY_NOT_V6B_GO"
    assert report["query_accounting"]["truth_adjoint"] == 0
    assert report["query_accounting"]["all_counts_exact"] is True
    assert (
        report["dot_product"]["max_float64_defect"]
        <= report["dot_product"]["tolerance"]
    )


def test_toy_problem_remains_underdetermined_at_primary_budget() -> None:
    config = _load_json(CONFIG)
    input_size = 1
    for dimension in config["input_shape"]:
        input_size *= int(dimension)
    assert max(config["query_budgets"]) < input_size


def test_metrics_cover_queries_and_both_failure_strata() -> None:
    config = _load_json(CONFIG)
    with (RESULTS / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = 2 * int(config["rigs_per_stratum"]) * len(
        config["query_budgets"]
    ) * 3
    assert len(rows) == expected
    assert all(int(row["query_count"]) == int(row["K"]) for row in rows)
    channel_rows = [row for row in rows if row["method"] == "channel_gate"]
    assert all(
        float(row["dot_defect_max"]) <= float(config["dot_tolerance_float64"])
        for row in channel_rows
    )

    primary = max(config["query_budgets"])
    in_class = [
        float(row["hidden_action_relative_l2"])
        for row in channel_rows
        if row["stratum"] == "in_class" and int(row["K"]) == primary
    ]
    out_of_class = [
        float(row["hidden_action_relative_l2"])
        for row in channel_rows
        if row["stratum"] == "out_of_class" and int(row["K"]) == primary
    ]
    assert max(in_class) < 1e-8
    assert min(out_of_class) > 0.05


def test_predictions_are_hashed_and_artifacts_match_checksums() -> None:
    config = _load_json(CONFIG)
    hashes = _load_json(RESULTS / "prediction_hashes_before_scoring.json")
    assert len(hashes) == 2 * int(config["rigs_per_stratum"]) * len(
        config["query_budgets"]
    )
    for row in hashes:
        assert len(row["candidate_sha256"]) == 64
        assert len(row["secant_sha256"]) == 64

    for line in (RESULTS / "checksums.sha256").read_text(encoding="ascii").splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((RESULTS / name).read_bytes()).hexdigest()
        assert actual == expected
