import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from demo_t16_operator.run_v6d_df_gated_srco_postopen import _noisy_observations, run


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "v6d_df_gated_srco_postopen.json"
RESULTS = ROOT / "results" / "v6d_df_gated_srco_postopen"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_v6d_keeps_all_scientific_and_preregistration_gates_closed() -> None:
    report = _json(RESULTS / "report.json")
    config = _json(CONFIG)
    assert report["decision"] == "SECOND_STAGE_POST_OPEN_DIAGNOSIS"
    assert report["scientific_claims_unlocked"] == []
    assert report["candidate_preregistration_authorized"] is False
    assert report["query_accounting_exact"] is True
    assert report["method"]["toy_noise_model"] == (
        "probe-dependent diagonal heteroscedastic covariance"
    )
    assert report["method"]["gate_relative_ridge"] == config["gate_relative_ridge"]
    assert report["method"]["secant_relative_ridge"] == config["secant_relative_ridge"]
    assert report["source_provenance"]["runner_sha256"] == hashlib.sha256(
        (ROOT / "run_v6d_df_gated_srco_postopen.py").read_bytes()
    ).hexdigest()
    assert report["source_provenance"]["limited_query_calibration_sha256"] == (
        hashlib.sha256((ROOT / "limited_query_calibration.py").read_bytes()).hexdigest()
    )
    assert report["source_provenance"]["config_file_sha256"] == hashlib.sha256(
        CONFIG.read_bytes()
    ).hexdigest()
    assert report["dot_product"]["max_float64_defect"] <= report["dot_product"][
        "tolerance"
    ]


def test_current_runner_reproduces_archived_report(tmp_path: Path) -> None:
    reproduced = run(CONFIG, tmp_path / "v6d")
    assert reproduced == _json(RESULTS / "report.json")


def test_noise_variances_follow_probe_block_order() -> None:
    clean = np.array([[3.0, 4.0], [0.0, 0.0], [0.0, 0.0]])
    _, total_energy, row_variances = _noisy_observations(
        np.random.default_rng(7), clean, 0.1
    )
    expected_column_variances = np.array([0.03, 0.16 / 3.0])
    expected_rows = np.repeat(expected_column_variances, clean.shape[0])
    np.testing.assert_allclose(row_variances, expected_rows)
    np.testing.assert_allclose(total_energy, np.sum(expected_rows))


def test_df_gate_separates_noise_only_and_misspecified_strata() -> None:
    report = _json(RESULTS / "report.json")
    primary = report["primary_K"]
    activation = {
        row["stratum"]: row
        for row in report["activation_summary"]
        if row["K"] == primary
    }
    assert activation["in_class"]["median"] == 0.0
    assert activation["in_class"]["p90"] < 0.05
    assert activation["out_of_class"]["p10"] > 0.99


def test_primary_result_preserves_gate_and_improves_misspecification() -> None:
    report = _json(RESULTS / "report.json")
    values = report["primary_hidden_action_medians"]
    assert values["in_class.df_gated_srco"] <= 1.01 * values["in_class.channel_gate"]
    assert values["in_class.df_gated_srco"] < values["in_class.full_srco"]
    assert values["out_of_class.df_gated_srco"] < values["out_of_class.channel_gate"]
    assert values["out_of_class.df_gated_srco"] < values["out_of_class.secant"]

    paired = {
        (row["stratum"], row["baseline"]): row
        for row in report["paired_primary"]
    }
    in_class = paired[("in_class", "channel_gate")]
    out_class = paired[("out_of_class", "channel_gate")]
    assert in_class["p90_relative_degradation"] < 0.01
    assert in_class["max_relative_degradation"] < 0.02
    assert out_class["positive_rigs"] == out_class["rigs"]
    assert out_class["max_relative_degradation"] < -0.20


def test_v6d_metrics_and_artifact_checksums_are_complete() -> None:
    config = _json(CONFIG)
    with (RESULTS / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = 2 * config["rigs_per_stratum"] * len(config["query_budgets"]) * 5
    assert len(rows) == expected
    assert all(int(row["query_count"]) == int(row["K"]) for row in rows)
    for line in (RESULTS / "checksums.sha256").read_text(encoding="ascii").splitlines():
        expected_hash, name = line.split("  ", 1)
        assert hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() == expected_hash
