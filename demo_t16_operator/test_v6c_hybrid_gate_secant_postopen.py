import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "v6c_hybrid_gate_secant_postopen.json"
RESULTS = ROOT / "results" / "v6c_hybrid_gate_secant_postopen"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_v6c_is_recorded_as_non_dominant_post_open_diagnosis() -> None:
    report = _json(RESULTS / "report.json")
    assert report["decision"] == "POST_OPEN_HYBRID_NOT_DOMINANT"
    assert report["fresh_status"] == "NOT_PREREGISTERED"
    assert report["scientific_claims_unlocked"] == []
    assert report["query_accounting_exact"] is True
    assert report["dot_product"]["max_float64_defect"] <= report["dot_product"][
        "tolerance"
    ]


def test_primary_result_preserves_benefit_and_harm() -> None:
    report = _json(RESULTS / "report.json")
    values = report["primary_hidden_action_medians"]
    assert values["out_of_class.gate_plus_secant"] < values["out_of_class.secant"]
    assert values["out_of_class.gate_plus_secant"] < values["out_of_class.channel_gate"]
    assert values["in_class.gate_plus_secant"] > values["in_class.channel_gate"]


def test_v6c_uses_same_query_budget_for_all_methods() -> None:
    config = _json(CONFIG)
    input_size = 1
    for dimension in config["input_shape"]:
        input_size *= int(dimension)
    assert max(config["query_budgets"]) < input_size
    with (RESULTS / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = 2 * config["rigs_per_stratum"] * len(config["query_budgets"]) * 4
    assert len(rows) == expected
    assert all(int(row["query_count"]) == int(row["K"]) for row in rows)


def test_v6c_artifact_checksums_match() -> None:
    for line in (RESULTS / "checksums.sha256").read_text(encoding="ascii").splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() == expected
