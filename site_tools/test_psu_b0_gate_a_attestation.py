"""Tests for the formal PSU-B0 Gate A attestation tooling."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from demo_t16_operator.psu_b0_gate_a_fixture import (
    DEFAULT_GATE_A_CONFIG_PATH,
    canonical_json_sha256,
    gate_a_input_payload,
    load_gate_a_config,
)
from site_tools.run_psu_b0_gate_a_attestation import (
    CLAIM_BOUNDARY,
    FORMAL_STATUS,
    collect_cpu_numeric_evidence,
    collect_negative_control_evidence,
    evaluate_numeric_gates,
)
from site_tools.psu_b0_gate_a_independent_oracle import (
    build_independent_oracle,
    run_numpy_recurrence,
)
from site_tools.validate_psu_b0_gate_a_attestation import (
    RELEASE_CONTENT_FILES,
    ValidationError,
    Validator,
    _independent_cpu_check,
    _validate_release_bundle,
    file_sha256,
    load_strict_json,
)


def _write_config(path: Path, values: dict[str, object]) -> Path:
    path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    return path


def test_cpu_numeric_evidence_satisfies_all_numeric_gates() -> None:
    config = load_gate_a_config()
    cpu = collect_cpu_numeric_evidence(config)
    mps_placeholder = {
        "available": True,
        "field_relative_difference": 0.0,
    }
    negative_controls = collect_negative_control_evidence(config)
    gates = evaluate_numeric_gates(
        config, cpu, mps_placeholder, negative_controls
    )
    assert all(gates.values())
    assert cpu["adjoint"]["maximum_relative_error"] <= 1e-8
    assert cpu["metric"]["data_dominance_violation_max"] == 0.0
    assert cpu["metric"]["tv_dominance_violation_max"] == 0.0
    assert cpu["metric"]["scaled_operator_norm_squared"] < cpu["metric"]["eta_squared"]
    assert cpu["recurrence"]["iterations"] == 6
    assert cpu["recurrence"]["maximum_state_relative_error"] <= 1e-10
    assert cpu["zero_coupling"]["deleted_data_indices"]
    assert cpu["ledgers"]["setup_logical"] == cpu["ledgers"]["setup_expected"]
    assert cpu["ledgers"]["setup_physical"] == cpu["ledgers"]["setup_expected"]
    assert (
        cpu["ledgers"]["oracle_audit_logical_delta"]
        == cpu["ledgers"]["oracle_audit_expected"]
    )
    assert (
        cpu["ledgers"]["oracle_audit_physical_delta"]
        == cpu["ledgers"]["oracle_audit_expected"]
    )
    assert cpu["ledgers"]["solve_logical_delta"] == cpu["ledgers"]["solve_expected"]
    assert cpu["ledgers"]["solve_physical_delta"] == cpu["ledgers"]["solve_expected"]
    assert cpu["ledgers"]["scorer_logical_delta"] == cpu["ledgers"]["scorer_expected"]
    assert cpu["ledgers"]["scorer_physical_delta"] == cpu["ledgers"]["scorer_expected"]


def test_config_is_truth_free_and_scope_limited() -> None:
    config = load_gate_a_config()
    assert config["evidence_scope"] == "VIEW_LOCAL_SINGLE_FROZEN_SCALE_MECHANICS_ONLY"
    assert config["scientific_claim_boundary"] == "NO_GATE_B_NO_FRESH_NO_REAL_NO_WIN_CLAIM"
    assert config["calibration_provenance"] == {
        "kind": "FROZEN_SYNTHETIC_MECHANICS_FIXTURE",
        "independent_flow_off_calibration": False,
        "truth_available_to_setup": False,
        "morphology_available_to_setup": False,
        "deployment_authorized": False,
    }
    forbidden = {
        "truth",
        "truth_field",
        "morphology",
        "ground_truth",
        "clean_projection_from_truth",
    }
    assert not (forbidden & set(config["fixture"]))
    assert set(config["e1_test_mapping"]) == {
        f"E1-{index:02d}" for index in range(1, 14)
    }


def test_independent_numpy_oracle_has_frozen_matrices_and_trace() -> None:
    config = load_gate_a_config()
    oracle = build_independent_oracle(config)
    trace = run_numpy_recurrence(oracle, config)
    assert oracle.schema_version == "psu-b0-gate-a-independent-numpy-oracle-1.0"
    assert oracle.A.shape == (4, 25)
    assert oracle.D.shape == (81, 25)
    assert float(abs(oracle.A).sum()) == pytest.approx(
        1.33968384, rel=1e-12, abs=1e-12
    )
    assert oracle.scaled_norm_squared == pytest.approx(
        0.47786387837228383, rel=1e-12, abs=1e-12
    )
    assert len(trace) == 6
    assert trace[-1]["x"][:3].tolist() == pytest.approx(
        [0.0065451318352720295, 0.0033012713602288283, -0.0094624631552047],
        rel=1e-12,
        abs=1e-12,
    )
    oracle_source = (
        Path(__file__).with_name("psu_b0_gate_a_independent_oracle.py")
        .read_text(encoding="utf-8")
    )
    assert "from demo_t16_operator" not in oracle_source
    assert "import demo_t16_operator" not in oracle_source


def test_executed_repository_import_closure_is_source_fingerprinted() -> None:
    script = r'''\
import json
from pathlib import Path
import sys

root = Path.cwd().resolve()
import site_tools.run_psu_b0_gate_a_attestation  # noqa: F401,E402
import site_tools.validate_psu_b0_gate_a_attestation  # noqa: F401,E402

loaded = set()
for module in sys.modules.values():
    raw = getattr(module, "__file__", None)
    if not raw:
        continue
    path = Path(raw)
    if not path.is_absolute():
        continue
    try:
        relative = path.resolve().relative_to(root)
    except (OSError, ValueError):
        continue
    if relative.parts[0] in {".venv", "build"}:
        continue
    if relative.suffix == ".py":
        loaded.add(relative.as_posix())
print(json.dumps(sorted(loaded)))
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )
    loaded = set(json.loads(result.stdout))
    configured = set(load_gate_a_config()["source_files"])
    assert loaded <= configured, sorted(loaded - configured)


def test_canonical_input_hash_is_order_stable_and_value_sensitive() -> None:
    config = load_gate_a_config()
    payload = gate_a_input_payload(config)
    reversed_payload = dict(reversed(list(payload.items())))
    assert canonical_json_sha256(payload) == canonical_json_sha256(reversed_payload)
    changed = deepcopy(payload)
    changed["fixture"]["measurement_scale"] = 0.61
    assert canonical_json_sha256(payload) != canonical_json_sha256(changed)


def test_config_rejects_truth_like_fixture_fields(tmp_path: Path) -> None:
    config = load_gate_a_config()
    config["fixture"]["truth"] = [1.0]
    path = _write_config(tmp_path / "truth.json", config)
    with pytest.raises(ValueError, match="truth or morphology"):
        load_gate_a_config(path)


def test_config_rejects_relaxed_claim_boundary(tmp_path: Path) -> None:
    config = load_gate_a_config()
    config["scientific_claim_boundary"] = "WIN"
    path = _write_config(tmp_path / "claim.json", config)
    with pytest.raises(ValueError, match="claim boundary"):
        load_gate_a_config(path)


def test_config_rejects_missing_e1_mapping(tmp_path: Path) -> None:
    config = load_gate_a_config()
    del config["e1_test_mapping"]["E1-13"]
    path = _write_config(tmp_path / "mapping.json", config)
    with pytest.raises(ValueError, match="every E1"):
        load_gate_a_config(path)


def test_config_rejects_test_path_escape(tmp_path: Path) -> None:
    config = load_gate_a_config()
    config["source_files"][0] = "../outside.py"
    path = _write_config(tmp_path / "escape.json", config)
    with pytest.raises(ValueError, match="unsafe repository-relative"):
        load_gate_a_config(path)


def test_config_rejects_unfingerprinted_test_node(tmp_path: Path) -> None:
    config = load_gate_a_config()
    config["source_files"].remove("site_tools/test_psu_b0_factor_interfaces.py")
    path = _write_config(tmp_path / "missing-test-hash.json", config)
    with pytest.raises(ValueError, match="not source-fingerprinted"):
        load_gate_a_config(path)


@pytest.mark.parametrize(
    "payload,message",
    [
        ('{"value": NaN}', "non-finite"),
        ('{"value": 1, "value": 2}', "duplicate"),
    ],
)
def test_runner_config_loader_rejects_nonfinite_and_duplicate_keys(
    tmp_path: Path,
    payload: str,
    message: str,
) -> None:
    path = tmp_path / "bad-runner-config.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_gate_a_config(path)


@pytest.mark.parametrize(
    "attack",
    ["relaxed_threshold", "replaced_test", "collapsed_e1_map"],
)
def test_config_rejects_relaxed_threshold_or_test_manifest(
    tmp_path: Path,
    attack: str,
) -> None:
    config = load_gate_a_config()
    if attack == "relaxed_threshold":
        config["thresholds"]["float64_adjoint_relative_error_max"] = 1e300
    elif attack == "replaced_test":
        config["test_nodes"][0] = (
            "site_tools/test_psu_b0_gate_a_attestation.py::"
            "test_claim_constants_do_not_authorize_performance"
        )
    elif attack == "collapsed_e1_map":
        config["e1_test_mapping"] = {
            gate: [0] for gate in config["e1_test_mapping"]
        }
    else:  # pragma: no cover - parametrization is exhaustive.
        raise AssertionError(attack)
    path = _write_config(tmp_path / f"{attack}.json", config)
    with pytest.raises(ValueError, match="threshold|test-node|E1 mapping"):
        load_gate_a_config(path)


@pytest.mark.parametrize(
    "payload,message",
    [
        ('{"value": NaN}', "non-finite"),
        ('{"value": 1, "value": 2}', "duplicate"),
    ],
)
def test_strict_json_rejects_nonfinite_and_duplicate_keys(
    tmp_path: Path,
    payload: str,
    message: str,
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValidationError, match=message):
        load_strict_json(path)


def test_independent_cpu_validator_recomputes_reported_mechanics() -> None:
    config = load_gate_a_config()
    cpu = collect_cpu_numeric_evidence(config)
    validator = Validator()
    result = _independent_cpu_check(
        validator,
        {"cpu_float64": cpu},
        config,
    )
    assert validator.checks >= 15
    assert result["dominance_violation_max"] == 0.0
    assert result["maximum_six_step_state_relative_error"] <= 1e-10
    assert result["deleted_data_rows"] > 0


def test_independent_cpu_validator_rejects_tampered_norm() -> None:
    config = load_gate_a_config()
    cpu = collect_cpu_numeric_evidence(config)
    cpu["metric"]["scaled_operator_norm_squared"] += 0.1
    with pytest.raises(ValidationError, match="reported scaled norm"):
        _independent_cpu_check(
            Validator(),
            {"cpu_float64": cpu},
            config,
        )


def test_independent_cpu_validator_rejects_tampered_final_state() -> None:
    config = load_gate_a_config()
    cpu = collect_cpu_numeric_evidence(config)
    cpu["recurrence"]["final_reduced_x"][0] += 0.5
    with pytest.raises(ValidationError, match="reported final state"):
        _independent_cpu_check(
            Validator(),
            {"cpu_float64": cpu},
            config,
        )


def test_claim_constants_do_not_authorize_performance() -> None:
    assert FORMAL_STATUS == "FORMAL_GATE_A_ATTESTED_MECHANICS_ONLY"
    assert CLAIM_BOUNDARY == "GATE_B_NOT_RUN_NO_FRESH_REAL_OR_WIN_CLAIM"
    assert DEFAULT_GATE_A_CONFIG_PATH.is_file()


def _write_release_bundle(
    directory: Path,
    validation: dict[str, object],
) -> None:
    for name in RELEASE_CONTENT_FILES:
        path = directory / name
        if name == "validation_report.json":
            path.write_text(
                json.dumps(validation, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text(f"frozen {name}\n", encoding="utf-8")
    (directory / "release_checksums.sha256").write_text(
        "".join(
            f"{file_sha256(directory / name)}  {name}\n"
            for name in sorted(RELEASE_CONTENT_FILES)
        ),
        encoding="ascii",
    )


def test_release_bundle_closes_validation_report_hash_loop(
    tmp_path: Path,
) -> None:
    validation = {"schema_version": "test", "checks": 17}
    _write_release_bundle(tmp_path, validation)
    validator = Validator()
    _validate_release_bundle(validator, tmp_path, validation)
    assert validator.checks > len(RELEASE_CONTENT_FILES)


def test_release_bundle_rejects_rehashed_forged_validation_report(
    tmp_path: Path,
) -> None:
    validation = {"schema_version": "test", "checks": 17}
    _write_release_bundle(tmp_path, validation)
    forged = {"schema_version": "test", "checks": 999999}
    (tmp_path / "validation_report.json").write_text(
        json.dumps(forged, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "release_checksums.sha256").write_text(
        "".join(
            f"{file_sha256(tmp_path / name)}  {name}\n"
            for name in sorted(RELEASE_CONTENT_FILES)
        ),
        encoding="ascii",
    )
    with pytest.raises(ValidationError, match="differs from independent"):
        _validate_release_bundle(Validator(), tmp_path, validation)
