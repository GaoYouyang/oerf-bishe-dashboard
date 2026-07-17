from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from site_tools.run_psu_b0_factor_gate_b import (
    CHECKPOINTS,
    METHODS,
    REACTION_FAMILIES,
    REPOSITORY_ROOT,
    _decision,
    load_gate_b_config,
    load_parent_metric_rows,
    validate_sources,
    write_release,
)


CONFIG = (
    REPOSITORY_ROOT
    / "demo_t16_operator/configs/psu_b0_factor_pdhg_gate_b_v4_a_only_connectivity_amendment.json"
)


def _write_config(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _rows(*, factor_k32: float = 0.60) -> list[dict[str, object]]:
    profiles = {
        "scalar_a_only_pdhg": {4: 1.30, 8: 1.20, 16: 1.10, 32: 1.00},
        "view_block_a_only_pdhg": {4: 1.10, 8: 0.95, 16: 0.85, 32: 0.80},
        "voxel_factor_a_only_pdhg": {4: 0.90, 8: 0.80, 16: 0.70, 32: factor_k32},
        "graph_pcgls": {4: 0.70, 8: 0.65, 16: 0.60, 32: 0.55},
    }
    return [
        {
            "replicate": replicate,
            "sample_index": sample,
            "reaction_family": REACTION_FAMILIES[sample],
            "method": method,
            "iterations": iteration,
            "forward_calls": iteration,
            "adjoint_calls": iteration,
            "field_relative_l2": profiles[method][iteration],
        }
        for replicate in (0, 8)
        for sample in range(len(REACTION_FAMILIES))
        for method in METHODS
        for iteration in CHECKPOINTS
    ]


def _timings() -> list[dict[str, object]]:
    return [
        {
            "replicate": replicate,
            "sample_index": sample,
            "scalar_seconds": 1.5,
            "view_block_seconds": 1.7,
            "voxel_factor_seconds": 2.0,
            "graph_seconds": 1.0,
        }
        for replicate in (0, 8)
        for sample in range(len(REACTION_FAMILIES))
    ]


def test_frozen_gate_b_config_and_all_source_hashes_are_valid() -> None:
    config = load_gate_b_config(CONFIG)
    observed = validate_sources(REPOSITORY_ROOT, config)
    assert observed == config["source_sha256"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["thresholds"].__setitem__(
            "factor_mean_reduction_vs_scalar_percent_min", 24.9
        ),
        lambda value: value["methods"].reverse(),
        lambda value: value["claim_boundary"].__setitem__(
            "algorithm_superiority_claimed", True
        ),
        lambda value: value["source_sha256"].__setitem__(
            "gate_b_runner_source", "0" * 64
        ),
        lambda value: value["test_preflight"].__setitem__(
            "expected_node_manifest_sha256", "0" * 64
        ),
        lambda value: value["timing_order_by_replicate"].__setitem__(
            "0", ["graph", "scalar", "view_block", "voxel_factor"]
        ),
        lambda value: value["amendment"].__setitem__(
            "emitted_metric_row_count", 1
        ),
    ],
)
def test_config_mutations_fail_closed(tmp_path: Path, mutation: object) -> None:
    value = copy.deepcopy(load_gate_b_config(CONFIG))
    mutation(value)
    path = tmp_path / "mutated.json"
    _write_config(path, value)
    with pytest.raises(ValueError):
        parsed = load_gate_b_config(path)
        validate_sources(REPOSITORY_ROOT, parsed)


def test_parent_metric_rows_have_exact_scalar_and_same_k_graph_coverage() -> None:
    config = load_gate_b_config(CONFIG)
    scalar, graph = load_parent_metric_rows(
        REPOSITORY_ROOT / config["source_paths"]["parent_metric_rows"]
    )
    assert len(scalar) == 64
    assert len(graph) == 64
    assert all(key[2] in CHECKPOINTS for key in graph)


def test_gate_b_decision_passes_only_as_oracle_scale_mechanism_signal() -> None:
    decision = _decision(_rows(), _timings(), load_gate_b_config(CONFIG)["thresholds"])
    assert decision["all_gates_passed"] is True
    assert decision["status"] == "GATE_B_E2_ORACLE_SCALE_CONDITIONING_SIGNAL_ONLY"
    assert decision["algorithm_superiority_claim_authorized"] is False
    assert decision["fm_cg_pdno_zero_init_smoke_authorized"] is True


def test_gate_b_decision_no_go_is_not_hidden() -> None:
    decision = _decision(
        _rows(factor_k32=1.10),
        _timings(),
        load_gate_b_config(CONFIG)["thresholds"],
    )
    assert decision["all_gates_passed"] is False
    assert decision["status"] == "GATE_B_E2_MECHANISM_NO_GO"
    assert decision["fm_cg_pdno_zero_init_smoke_authorized"] is False


def test_gate_b_decision_rejects_duplicate_or_missing_rows() -> None:
    rows = _rows()
    with pytest.raises(ValueError, match="duplicate"):
        _decision([*rows, rows[0]], _timings(), load_gate_b_config(CONFIG)["thresholds"])
    with pytest.raises(ValueError, match="coverage"):
        _decision(rows[:-1], _timings(), load_gate_b_config(CONFIG)["thresholds"])


def test_release_writer_hashes_exact_runner_payloads(tmp_path: Path) -> None:
    write_release(
        tmp_path,
        report={"status": "TEST"},
        rows=[{"value": 1}],
        audit={"calls": []},
    )
    lines = (tmp_path / "checksums.sha256").read_text(encoding="ascii").splitlines()
    assert {line.split("  ", 1)[1] for line in lines} == {
        "report.json",
        "metric_rows.csv",
        "audit.json",
    }
