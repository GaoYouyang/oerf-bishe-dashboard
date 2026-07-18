from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from site_tools.n5_d5_private_lab_readiness import (
    build_readiness_report,
    sha256_file as l1_sha256_file,
)
from site_tools.n5_d5_private_replay_foundation import (
    EXPECTED_PRIVATE_PAYLOAD_FILES,
    PLAN_PLACEHOLDER_RELATIVE,
    PUBLIC_FOUNDATION_SOURCES,
    ROOT,
    _safe_private_output,
    build_foundation_report,
    generate_private_probe_bank,
    private_probe_commitment,
    private_probe_reveal,
    private_result_manifest,
    sha256_file,
    validate_closed_world_result_directory,
)


LAB_PLACEHOLDER = ROOT / "data_templates/n5_d5_lab_interface.placeholder.json"
PLAN_PLACEHOLDER = ROOT / PLAN_PLACEHOLDER_RELATIVE


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _relative(repo: Path, path: Path) -> str:
    return path.relative_to(repo).as_posix()


def _inventory_entry(repo: Path, path: Path, role: str) -> dict[str, object]:
    return {
        "relative_path": _relative(repo, path),
        "role": role,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


@pytest.fixture()
def private_foundation(tmp_path: Path) -> dict[str, object]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("private_library/\n", encoding="utf-8")
    for relative in PUBLIC_FOUNDATION_SOURCES:
        source = ROOT / relative
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "n5-d5-l2-test@example.invalid")
    _git(repo, "config", "user.name", "N5 D5 L2 Test")
    _git(repo, "add", ".gitignore", *PUBLIC_FOUNDATION_SOURCES)
    _git(repo, "commit", "-qm", "public L2 foundation fixture")

    bundle = repo / "private_library/n5_d5_bundles/anon_case_v1"
    control = repo / "private_library/n5_d5_control/anon_case_v1"
    bundle.mkdir(parents=True)
    control.mkdir(parents=True)
    adapter = bundle / "adapter.py"
    adapter.write_text(
        "from __future__ import annotations\n"
        "def renderer_entry(request):\n"
        "    return request\n",
        encoding="utf-8",
    )
    base = bundle / "base.npy"
    np.save(base, np.linspace(-0.1, 0.1, 6, dtype=np.float64))
    environment = bundle / "environment-lock.json"
    _write_json(
        environment,
        {
            "schema_version": "n5-d5-private-environment-lock-1.0",
            "lock_reviewed": True,
            "contains_credentials": False,
            "artifacts": [
                {
                    "name": "numpy",
                    "version": "fixture-version",
                    "sha256": hashlib.sha256(b"fixture-wheel").hexdigest(),
                }
            ],
        },
    )
    physical = bundle / "physical-contract.json"
    _write_json(
        physical,
        {
            "schema_version": "n5-d5-private-physical-contract-1.0",
            "reviewed": True,
            "contains_private_identifiers": False,
            "parameterization": "refractive_index_field",
            "tensor_shape": [6],
            "spatial_spacing": [1.0],
            "axis_order": "anonymous_flattened_zyx",
            "field_units": "dimensionless_refractive_index",
            "observation_units": "detector_displacement_pixel",
            "coordinate_handedness": "right_handed",
            "geometry_sha256": hashlib.sha256(b"fixture-geometry").hexdigest(),
            "calibration_sha256": hashlib.sha256(b"fixture-calibration").hexdigest(),
            "optical_wavelength_policy": "monochromatic_fixture",
            "sampling_rule": "reviewed_fixture_sampling",
            "interpolation_rule": "reviewed_fixture_interpolation",
            "boundary_rule": "reviewed_fixture_boundary",
            "termination_rule": "reviewed_fixture_termination",
            "backend_dtype": "float64",
            "wire_dtype": "float64",
            "conversion_policy": "no_conversion_float64",
            "dynamic_sampling": False,
            "dynamic_ray_sample_ledger_required": True,
            "native_direct_residual": True,
            "direct_residual_semantics": "native_same_sample_residual",
            "decoder_checkpoint_sha256": None,
            "noise_floor": {
                "units": "detector_displacement_pixel",
                "absolute_std": 1e-6,
            },
        },
    )

    config = json.loads(LAB_PLACEHOLDER.read_text(encoding="utf-8"))
    config["identity"].update(
        {"bundle_id": "anon_bundle_43f2", "context_id": "anon_context_701a"}
    )
    config["adapter"].update(
        {
            "command": ["{python}", _relative(repo, adapter)],
            "expected_adapter_id": "private_renderer_43f2",
            "expected_adapter_version": "reviewed-v1",
            "expected_implementation_sha256": l1_sha256_file(adapter),
            "source_files": [_relative(repo, adapter)],
        }
    )
    config["field"].update(
        {
            "input_dimension": 6,
            "units": "dimensionless_refractive_index",
            "axis_order": "anonymous_flattened_zyx",
            "base_input": {
                "source": "npy_relative",
                "seed": None,
                "relative_path": _relative(repo, base),
                "sha256": l1_sha256_file(base),
            },
        }
    )
    config["observation"].update(
        {
            "units": "detector_displacement_pixel",
            "component_order": "ray_major_uv",
        }
    )
    for index, path in enumerate(config["paths"]):
        role = path["role"]
        path.update(
            {
                "path_id": f"private_{role}_path_v1",
                "callable_id": f"private_{role}_callable_v1",
                "semantic_digest_sha256": hashlib.sha256(
                    f"private:{index}:{role}".encode()
                ).hexdigest(),
            }
        )
    config_path = bundle / "config.json"
    _write_json(config_path, config)
    l1_report = build_readiness_report(
        config_path,
        repo_root=repo,
        enforce_public_committed=True,
    )
    assert l1_report["status"] == "STATIC_PRIVATE_INTAKE_READY_FORMAL_REPLAY_LOCKED"
    l1_path = control / "l1-readiness.json"
    _write_json(l1_path, l1_report)

    plan = json.loads(PLAN_PLACEHOLDER.read_text(encoding="utf-8"))
    plan["plan_id"] = "anon_plan_43f2"
    plan["l1_report"] = {
        "relative_path": _relative(repo, l1_path),
        "sha256": sha256_file(l1_path),
    }
    plan["private_bundle"] = {
        "root": _relative(repo, bundle),
        "closed_world_required": True,
        "inventory": [
            _inventory_entry(repo, config_path, "config"),
            _inventory_entry(repo, adapter, "adapter_source"),
            _inventory_entry(repo, base, "base_input"),
            _inventory_entry(repo, environment, "environment_lock"),
            _inventory_entry(repo, physical, "physical_contract"),
        ],
    }
    plan["capability_profile"].update(
        {
            "direct_residual_supported": True,
            "direct_residual_semantics": "native_same_sample_residual",
        }
    )
    plan["authorization_budget"].update(
        {
            "primary_requests": 53,
            "validator_base_replay_requests": 53,
            "validator_private_probe_requests": 48,
            "total_requests": 156,
        }
    )
    result_dir = repo / "private_library/n5_d5_runs/anon_case_v1/run-001"
    plan["output_policy"]["result_directory"] = _relative(repo, result_dir)
    plan_path = control / "replay-plan.json"
    _write_json(plan_path, plan)
    return {
        "repo": repo,
        "bundle": bundle,
        "control": control,
        "adapter": adapter,
        "base": base,
        "environment": environment,
        "physical": physical,
        "config_path": config_path,
        "l1_path": l1_path,
        "l1_report": l1_report,
        "plan": plan,
        "plan_path": plan_path,
        "result_dir": result_dir,
    }


def _report(fixture: dict[str, object]) -> dict[str, object]:
    return build_foundation_report(
        fixture["plan_path"],
        repo_root=fixture["repo"],
        enforce_public_committed=True,
    )


def _foundation_codes(report: dict[str, object]) -> set[str]:
    return set(report["foundation_blocker_codes"])


def test_complete_bundle_is_foundation_ready_but_execution_remains_locked(
    private_foundation: dict[str, object],
) -> None:
    report = _report(private_foundation)

    assert report["status"] == "PRIVATE_REPLAY_FOUNDATION_READY_EXECUTION_LOCKED"
    assert report["foundation_ready"] is True
    assert report["execution_policy_declared"] is False
    assert report["private_describe_probe_authorized"] is False
    assert report["formal_53_call_replay_authorized"] is False
    assert report["adapter_executed"] is False
    assert report["foundation_blocker_codes"] == []
    assert set(report["execution_blocker_codes"]) == {
        "NETWORK_ISOLATION_BACKEND_CONFIGURED",
        "INDEPENDENT_COST_INSTRUMENTATION_CONFIGURED",
        "PHYSICAL_TOLERANCES_REVIEWED",
    }
    assert not any(report["claim_authorizations"].values())
    assert len(report["private_inventory"]) == 5
    assert report["authorization_budget"]["computed"]["total_requests"] == 156
    assert report["formal_replay_authorized"] is False
    serialized = json.dumps(report)
    assert str(private_foundation["repo"]) not in serialized
    assert "private_library/" not in serialized


def test_declared_execution_policies_still_do_not_authorize_describe(
    private_foundation: dict[str, object],
) -> None:
    repo = private_foundation["repo"]
    bundle = private_foundation["bundle"]
    network = bundle / "network.sb"
    network.write_text("(version 1)\n(deny network*)\n", encoding="utf-8")
    cost = bundle / "cost-counter.json"
    _write_json(cost, {"schema": "fixture-cost-counter", "reviewed": True})
    plan = copy.deepcopy(private_foundation["plan"])
    plan["private_bundle"]["inventory"].extend(
        [
            _inventory_entry(repo, network, "network_profile"),
            _inventory_entry(repo, cost, "cost_instrumentation"),
        ]
    )
    plan["process_policy"].update(
        {
            "network_isolation_backend": "macos_sandbox_profile",
            "independent_cost_instrumentation": "backend_counter_private_reviewed",
        }
    )
    plan["physical_review"] = {
        "tolerances_reviewed": True,
        "reviewer_pseudonym": "reviewer_43f2",
        "review_digest_sha256": sha256_file(private_foundation["physical"]),
    }
    _write_json(private_foundation["plan_path"], plan)

    report = _report(private_foundation)

    assert report["foundation_ready"] is True
    assert report["execution_policy_declared"] is True
    assert report["execution_blocker_codes"] == []
    assert report["private_describe_probe_authorized"] is False
    assert report["formal_53_call_replay_authorized"] is False


def test_unlisted_bundle_file_breaks_closed_world(
    private_foundation: dict[str, object],
) -> None:
    (private_foundation["bundle"] / "undeclared.cache").write_bytes(b"not declared")

    report = _report(private_foundation)

    assert "BUNDLE_CLOSED_WORLD_EXACT_INVENTORY" in _foundation_codes(report)


def test_bundle_hash_tampering_is_blocked(
    private_foundation: dict[str, object],
) -> None:
    plan = copy.deepcopy(private_foundation["plan"])
    plan["private_bundle"]["inventory"][1]["sha256"] = "f" * 64
    _write_json(private_foundation["plan_path"], plan)

    report = _report(private_foundation)

    assert "BUNDLE_INPUT_1_HASH_MATCHES" in _foundation_codes(report)


def test_l1_report_hash_and_status_tampering_are_blocked(
    private_foundation: dict[str, object],
) -> None:
    l1 = copy.deepcopy(private_foundation["l1_report"])
    l1["status"] = "PRIVATE_INTAKE_READINESS_BLOCKED"
    _write_json(private_foundation["l1_path"], l1)
    plan = copy.deepcopy(private_foundation["plan"])
    plan["l1_report"]["sha256"] = sha256_file(private_foundation["l1_path"])
    _write_json(private_foundation["plan_path"], plan)

    report = _report(private_foundation)

    assert "L1_STATIC_READY_FORMAL_LOCKED" in _foundation_codes(report)


def test_l1_report_digest_mismatch_is_blocked(
    private_foundation: dict[str, object],
) -> None:
    plan = copy.deepcopy(private_foundation["plan"])
    plan["l1_report"]["sha256"] = "f" * 64
    _write_json(private_foundation["plan_path"], plan)

    report = _report(private_foundation)

    assert "L1_REPORT_HASH_MATCHES" in _foundation_codes(report)


def test_symlinked_and_hardlinked_bundle_inputs_are_blocked(
    private_foundation: dict[str, object],
) -> None:
    adapter = private_foundation["adapter"]
    target = private_foundation["bundle"] / "adapter-target.py"
    adapter.rename(target)
    adapter.symlink_to(target.name)

    report = _report(private_foundation)
    assert "BUNDLE_ONLY_REGULAR_NON_SYMLINK_FILES" in _foundation_codes(report)
    assert "BUNDLE_INPUT_1_NO_SYMLINK_COMPONENT" in _foundation_codes(report)

    adapter.unlink()
    target.rename(adapter)
    hardlink = private_foundation["bundle"] / "adapter-hardlink.py"
    hardlink.hardlink_to(adapter)
    report = _report(private_foundation)
    assert "BUNDLE_INPUT_1_NOT_HARDLINKED" in _foundation_codes(report)


def test_tracked_private_input_is_blocked(
    private_foundation: dict[str, object],
) -> None:
    repo = private_foundation["repo"]
    relative = _relative(repo, private_foundation["adapter"])
    _git(repo, "add", "-f", relative)

    report = _report(private_foundation)

    assert "BUNDLE_INPUT_1_NOT_TRACKED" in _foundation_codes(report)


def test_existing_or_public_result_directory_is_blocked(
    private_foundation: dict[str, object],
) -> None:
    private_foundation["result_dir"].mkdir(parents=True)
    report = _report(private_foundation)
    assert "RESULT_DIRECTORY_ABSENT" in _foundation_codes(report)

    plan = copy.deepcopy(private_foundation["plan"])
    plan["output_policy"]["result_directory"] = "public-result"
    _write_json(private_foundation["plan_path"], plan)
    report = _report(private_foundation)
    assert "RESULT_DIRECTORY_UNDER_PRIVATE_ROOT" in _foundation_codes(report)


def test_public_summary_or_payload_allowlist_drift_is_blocked_by_schema(
    private_foundation: dict[str, object],
) -> None:
    plan = copy.deepcopy(private_foundation["plan"])
    plan["output_policy"]["public_summary_write_permitted"] = True
    plan["output_policy"]["expected_payload_files"].append("public_summary.json")
    _write_json(private_foundation["plan_path"], plan)

    report = _report(private_foundation)

    assert "PLAN_SCHEMA_VALID" in _foundation_codes(report)


def test_environment_lock_with_secret_or_absolute_path_is_blocked(
    private_foundation: dict[str, object],
) -> None:
    value = json.loads(private_foundation["environment"].read_text(encoding="utf-8"))
    value["debug_path"] = "/Users/example/private/wheel.whl"
    value["password"] = "example-not-a-real-password"
    _write_json(private_foundation["environment"], value)
    plan = copy.deepcopy(private_foundation["plan"])
    environment_entry = next(
        item
        for item in plan["private_bundle"]["inventory"]
        if item["role"] == "environment_lock"
    )
    environment_entry.update(
        _inventory_entry(
            private_foundation["repo"],
            private_foundation["environment"],
            "environment_lock",
        )
    )
    _write_json(private_foundation["plan_path"], plan)

    report = _report(private_foundation)

    assert "ENVIRONMENT_LOCK_STRUCTURALLY_REVIEWED" in _foundation_codes(report)


def test_dual_profile_with_triple_l1_requires_dual_path_l1_v2(
    private_foundation: dict[str, object],
) -> None:
    physical_value = json.loads(
        private_foundation["physical"].read_text(encoding="utf-8")
    )
    physical_value.update(
        {
            "native_direct_residual": False,
            "direct_residual_semantics": "unavailable",
        }
    )
    _write_json(private_foundation["physical"], physical_value)
    plan = copy.deepcopy(private_foundation["plan"])
    physical_entry = next(
        item
        for item in plan["private_bundle"]["inventory"]
        if item["role"] == "physical_contract"
    )
    physical_entry.update(
        _inventory_entry(
            private_foundation["repo"],
            private_foundation["physical"],
            "physical_contract",
        )
    )
    plan["capability_profile"].update(
        {
            "direct_residual_supported": False,
            "direct_residual_semantics": "unavailable",
        }
    )
    plan["authorization_budget"].update(
        {
            "primary_requests": 36,
            "validator_base_replay_requests": 36,
            "validator_private_probe_requests": 32,
            "total_requests": 106,
        }
    )
    _write_json(private_foundation["plan_path"], plan)

    report = _report(private_foundation)

    assert "L1_CAPABILITY_PROFILE_MATCHES" in _foundation_codes(report)
    assert "AUTHORIZATION_BUDGET_EXACT" in _foundation_codes(report)
    assert report["next_action"] == "BUILD_DUAL_PATH_L1_V2"


def test_authorization_budget_drift_is_blocked(
    private_foundation: dict[str, object],
) -> None:
    plan = copy.deepcopy(private_foundation["plan"])
    plan["authorization_budget"]["total_requests"] = 155
    _write_json(private_foundation["plan_path"], plan)

    report = _report(private_foundation)

    assert "AUTHORIZATION_BUDGET_EXACT" in _foundation_codes(report)
    assert report["authorization_budget"]["computed"]["total_requests"] == 156


def test_physical_contract_invalid_noise_or_semantic_drift_is_blocked(
    private_foundation: dict[str, object],
) -> None:
    physical_value = json.loads(
        private_foundation["physical"].read_text(encoding="utf-8")
    )
    physical_value["noise_floor"]["absolute_std"] = 0.0
    _write_json(private_foundation["physical"], physical_value)
    plan = copy.deepcopy(private_foundation["plan"])
    physical_entry = next(
        item
        for item in plan["private_bundle"]["inventory"]
        if item["role"] == "physical_contract"
    )
    physical_entry.update(
        _inventory_entry(
            private_foundation["repo"],
            private_foundation["physical"],
            "physical_contract",
        )
    )
    _write_json(private_foundation["plan_path"], plan)

    report = _report(private_foundation)

    assert "PHYSICAL_CONTRACT_STRUCTURALLY_REVIEWED" in _foundation_codes(report)


def test_physical_contract_must_match_l1_units_and_shape(
    private_foundation: dict[str, object],
) -> None:
    physical_value = json.loads(
        private_foundation["physical"].read_text(encoding="utf-8")
    )
    physical_value["field_units"] = "incompatible_fixture_units"
    _write_json(private_foundation["physical"], physical_value)
    plan = copy.deepcopy(private_foundation["plan"])
    physical_entry = next(
        item
        for item in plan["private_bundle"]["inventory"]
        if item["role"] == "physical_contract"
    )
    physical_entry.update(
        _inventory_entry(
            private_foundation["repo"],
            private_foundation["physical"],
            "physical_contract",
        )
    )
    _write_json(private_foundation["plan_path"], plan)

    report = _report(private_foundation)

    assert "PHYSICAL_CONTRACT_MATCHES_L1_CONFIG" in _foundation_codes(report)


def test_private_h_intervals_must_be_ordered(
    private_foundation: dict[str, object],
) -> None:
    plan = copy.deepcopy(private_foundation["plan"])
    plan["challenge_policy"]["private_h_intervals"][0] = [2e-5, 8e-6]
    _write_json(private_foundation["plan_path"], plan)

    report = _report(private_foundation)

    assert "PRIVATE_H_INTERVALS_STRICTLY_ORDERED" in _foundation_codes(report)
    assert "PRIVATE_PROBE_SCHEDULE_VALID" in _foundation_codes(report)
    assert "AUTHORIZATION_BUDGET_EXACT" in _foundation_codes(report)


def test_physical_review_digest_must_bind_actual_contract(
    private_foundation: dict[str, object],
) -> None:
    plan = copy.deepcopy(private_foundation["plan"])
    plan["physical_review"] = {
        "tolerances_reviewed": True,
        "reviewer_pseudonym": "reviewer_43f2",
        "review_digest_sha256": hashlib.sha256(b"different-contract").hexdigest(),
    }
    _write_json(private_foundation["plan_path"], plan)

    report = _report(private_foundation)

    assert report["foundation_ready"] is True
    assert "PHYSICAL_TOLERANCES_REVIEWED" in set(report["execution_blocker_codes"])


def test_capability_cannot_declare_precomputed_probe_arrays_sufficient(
    private_foundation: dict[str, object],
) -> None:
    plan = copy.deepcopy(private_foundation["plan"])
    plan["capability_profile"]["precomputed_probe_arrays_are_sufficient"] = True
    _write_json(private_foundation["plan_path"], plan)

    report = _report(private_foundation)

    assert "PLAN_SCHEMA_VALID" in _foundation_codes(report)


def test_dirty_public_foundation_source_is_blocked(
    private_foundation: dict[str, object],
) -> None:
    source = (
        private_foundation["repo"] / "site_tools/n5_d5_private_replay_foundation.py"
    )
    source.write_text("# changed after commit\n", encoding="utf-8")

    report = _report(private_foundation)

    assert any(
        code.startswith("PUBLIC_SOURCE_") and code.endswith("_CLEAN")
        for code in _foundation_codes(report)
    )


def test_private_probe_bank_is_fresh_orthonormal_and_not_pre_persisted() -> None:
    h_intervals = [[8e-6, 2e-5], [8e-5, 2e-4], [8e-4, 2e-3]]
    first = generate_private_probe_bank(
        input_dimension=7,
        output_dimension=5,
        tangent_count=3,
        cotangent_count=2,
        h_intervals=h_intervals,
    )
    second = generate_private_probe_bank(
        input_dimension=7,
        output_dimension=5,
        tangent_count=3,
        cotangent_count=2,
        h_intervals=h_intervals,
    )
    assert first.commitment_sha256 != second.commitment_sha256
    assert first.context_sha256 == second.context_sha256
    assert first.seed != second.seed
    assert first.tangents.shape == (3, 7)
    assert first.cotangents.shape == (2, 5)
    assert first.h_values.shape == (3,)
    assert all(
        low < value < high
        for value, (low, high) in zip(first.h_values, h_intervals, strict=True)
    )
    assert np.allclose(first.tangents @ first.tangents.T, np.eye(3), atol=1e-12)
    assert np.allclose(first.cotangents @ first.cotangents.T, np.eye(2), atol=1e-12)
    commitment = private_probe_commitment(first)
    assert commitment["seed_persisted"] is False
    assert commitment["vectors_persisted"] is False
    assert commitment["h_values_persisted"] is False
    assert commitment["context_sha256"] == first.context_sha256
    assert "seed_hex" not in commitment
    assert first.seed.hex() not in json.dumps(commitment, sort_keys=True)
    reveal = private_probe_reveal(first)
    assert reveal["seed_hex"] == first.seed.hex()
    assert reveal["context_sha256"] == first.context_sha256
    assert reveal["h_values"] == [float(value) for value in first.h_values]
    assert reveal["write_policy"] == "private_result_after_adapter_exit_only"


def test_private_probe_bank_rejects_bad_entropy_or_impossible_count() -> None:
    with pytest.raises(ValueError, match="entropy"):
        generate_private_probe_bank(
            input_dimension=4,
            output_dimension=4,
            tangent_count=2,
            cotangent_count=2,
            h_intervals=[[1e-5, 2e-5]],
            entropy_bytes=16,
        )
    with pytest.raises(ValueError, match="probe count"):
        generate_private_probe_bank(
            input_dimension=2,
            output_dimension=4,
            tangent_count=3,
            cotangent_count=2,
            h_intervals=[[1e-5, 2e-5]],
            entropy_source=lambda size: b"a" * size,
        )
    with pytest.raises(ValueError, match="interval"):
        generate_private_probe_bank(
            input_dimension=4,
            output_dimension=4,
            tangent_count=2,
            cotangent_count=2,
            h_intervals=[[2e-5, 1e-5]],
            entropy_source=lambda size: b"a" * size,
        )


@pytest.fixture()
def closed_world_result(tmp_path: Path) -> dict[str, Path]:
    repo = tmp_path / "repo"
    result = repo / "private_library/runs/run-001"
    result.mkdir(parents=True)
    for index, name in enumerate(EXPECTED_PRIVATE_PAYLOAD_FILES):
        (result / name).write_bytes(f"private-payload-{index}\n".encode())
    _write_json(result / "manifest.json", private_result_manifest(result))
    return {"repo": repo, "result": result}


def test_closed_world_private_result_accepts_exact_manifest(
    closed_world_result: dict[str, Path],
) -> None:
    report = validate_closed_world_result_directory(
        closed_world_result["result"],
        repo_root=closed_world_result["repo"],
    )

    assert report["valid"] is True
    assert report["payload_file_count"] == len(EXPECTED_PRIVATE_PAYLOAD_FILES)
    assert report["public_summary_present"] is False
    assert not any(report["claim_authorizations"].values())


def test_closed_world_private_result_rejects_extra_or_tampered_file(
    closed_world_result: dict[str, Path],
) -> None:
    extra = closed_world_result["result"] / "public_summary.json"
    extra.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="closed-world"):
        validate_closed_world_result_directory(
            closed_world_result["result"], repo_root=closed_world_result["repo"]
        )

    extra.unlink()
    target = closed_world_result["result"] / "result.json"
    target.write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="hash or size mismatch"):
        validate_closed_world_result_directory(
            closed_world_result["result"], repo_root=closed_world_result["repo"]
        )


def test_foundation_report_output_cannot_escape_or_overwrite(
    private_foundation: dict[str, object],
) -> None:
    repo = private_foundation["repo"]
    with pytest.raises(ValueError, match="private_library"):
        _safe_private_output(repo, repo / "public-report.json")

    output = private_foundation["control"] / "foundation-report.json"
    assert _safe_private_output(repo, output) == output
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing"):
        _safe_private_output(repo, output)


def test_foundation_module_does_not_import_runner_protocol_or_adapter() -> None:
    source = (ROOT / "site_tools/n5_d5_private_replay_foundation.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    assert imports.isdisjoint(
        {
            "site_tools.run_n5_d5_minimum_interface_bridge",
            "site_tools.validate_n5_d5_minimum_interface_bridge",
            "demo_t16_operator.n5_d5_adapter_protocol",
            "demo_t16_operator.n5_d5_synthetic_reference_adapter",
        }
    )
    assert "run_adapter" not in source
    assert "Popen" not in source
