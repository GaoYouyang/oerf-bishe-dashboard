from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from jsonschema import Draft202012Validator
import numpy as np
import pytest

from site_tools.n5_d5_private_lab_readiness_dual_v2 import (
    DUAL_PLACEHOLDER_RELATIVE,
    DUAL_SCHEMA_RELATIVE,
    EXPECTED_PRIMARY_COST,
    PUBLIC_DUAL_SOURCES,
    ROOT,
    V1_SCHEMA_RELATIVE,
    build_dual_readiness_report,
    sha256_file,
)
from site_tools.n5_d5_private_replay_foundation import (
    PLAN_PLACEHOLDER_RELATIVE,
    PUBLIC_FOUNDATION_SOURCES,
    build_foundation_report,
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _safe_adapter_source() -> str:
    return (
        "from __future__ import annotations\n"
        "import numpy as np\n\n"
        "def forward_endpoint(path_role: str, x: np.ndarray) -> np.ndarray:\n"
        "    if path_role == 'curved':\n"
        "        return np.sin(x)\n"
        "    if path_role == 'straight':\n"
        "        return np.asarray(x, dtype=np.float64)\n"
        "    raise ValueError('only curved and straight are available')\n\n"
        "def runtime_direction_contract() -> tuple[str, bool]:\n"
        "    return ('arbitrary_runtime_x_v_q', False)\n"
    )


@pytest.fixture()
def dual_private_repo(tmp_path: Path) -> dict[str, object]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("private_library/\n", encoding="utf-8")
    for relative in PUBLIC_DUAL_SOURCES:
        source = ROOT / relative
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "dual-v2-test@example.invalid")
    _git(repo, "config", "user.name", "Dual V2 Test")
    _git(repo, "add", ".gitignore", *PUBLIC_DUAL_SOURCES)
    _git(repo, "commit", "-qm", "public dual-v2 fixture")

    bundle = repo / "private_library/n5_d5_dual_v2/anonymous_case"
    bundle.mkdir(parents=True)
    adapter = bundle / "adapter.py"
    adapter.write_text(_safe_adapter_source(), encoding="utf-8")
    base = bundle / "base.npy"
    np.save(base, np.linspace(-0.2, 0.2, 6, dtype=np.float64))
    config = json.loads(
        (repo / DUAL_PLACEHOLDER_RELATIVE).read_text(encoding="utf-8")
    )
    config["identity"].update(
        {
            "bundle_id": "anonymous_bundle_dual_v2",
            "context_id": "anonymous_context_dual_v2",
        }
    )
    relative_adapter = adapter.relative_to(repo).as_posix()
    relative_base = base.relative_to(repo).as_posix()
    config["adapter"].update(
        {
            "command": ["{python}", relative_adapter],
            "expected_adapter_id": "anonymous_dual_adapter",
            "expected_adapter_version": "dual-v2-fixture",
            "expected_implementation_sha256": sha256_file(adapter),
            "source_files": [relative_adapter],
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
                "relative_path": relative_base,
                "sha256": sha256_file(base),
            },
        }
    )
    config["observation"].update(
        {
            "ray_count": 4,
            "components_per_ray": 2,
            "output_dimension": 8,
            "units": "detector_displacement_pixel",
            "component_order": "ray_major_uv",
        }
    )
    for index, path in enumerate(config["paths"]):
        role = path["role"]
        path.update(
            {
                "path_id": f"anonymous_{role}_path_v2",
                "callable_id": f"anonymous_{role}_callable_v2",
                "semantic_digest_sha256": hashlib.sha256(
                    f"dual-v2:{index}:{role}".encode("utf-8")
                ).hexdigest(),
            }
        )
    config_path = bundle / "config.json"
    _write_json(config_path, config)
    return {
        "repo": repo,
        "bundle": bundle,
        "adapter": adapter,
        "base": base,
        "config": config,
        "config_path": config_path,
    }


def _write_config(fixture: dict[str, object], config: dict[str, object]) -> None:
    _write_json(fixture["config_path"], config)


def _report(fixture: dict[str, object]) -> dict[str, object]:
    return build_dual_readiness_report(
        fixture["config_path"],
        repo_root=fixture["repo"],
        enforce_public_committed=True,
    )


def test_valid_dual_v2_is_static_ready_with_all_execution_and_claims_locked(
    dual_private_repo: dict[str, object],
) -> None:
    report = _report(dual_private_repo)

    assert report["status"] == (
        "STATIC_PRIVATE_DUAL_PATH_INTAKE_READY_FORMAL_REPLAY_LOCKED"
    )
    assert report["protocol_variant"] == "dual_path_no_native_direct_v2"
    assert report["capability_status"] == "NATIVE_DIRECT_RESIDUAL_UNAVAILABLE"
    assert report["path_status"] == "EXACTLY_CURVED_AND_STRAIGHT"
    assert report["path_count"] == 2
    assert report["primary_cost"] == EXPECTED_PRIMARY_COST
    assert next(
        item
        for item in report["checks"]
        if item["code"] == "DUAL_PRIMARY_COST_DERIVED_AND_EXACT_36"
    )["passed"] is True
    assert report["ready_for_private_describe_probe"] is True
    assert report["formal_dual_primary_36_call_replay_authorized"] is False
    assert report["formal_triple_primary_53_call_replay_authorized"] is False
    assert report["formal_replay_authorized"] is False
    assert report["adapter_executed"] is False
    assert report["blocker_codes"] == []
    assert not any(report["claim_authorizations"].values())


def test_dual_schema_is_independent_and_frozen_v1_still_rejects_two_paths(
    dual_private_repo: dict[str, object],
) -> None:
    dual_schema = json.loads(
        (dual_private_repo["repo"] / DUAL_SCHEMA_RELATIVE).read_text(
            encoding="utf-8"
        )
    )
    v1_schema = json.loads(
        (dual_private_repo["repo"] / V1_SCHEMA_RELATIVE).read_text(
            encoding="utf-8"
        )
    )
    config = dual_private_repo["config"]
    assert list(Draft202012Validator(dual_schema).iter_errors(config)) == []
    assert list(Draft202012Validator(v1_schema).iter_errors(config))
    assert dual_schema["$id"] != v1_schema["$id"]


@pytest.mark.parametrize(
    ("mutator", "expected_code"),
    [
        (
            lambda config: config["paths"].append(
                {
                    "role": "direct_residual",
                    "path_id": "forbidden_direct_path",
                    "callable_id": "forbidden_direct_callable",
                    "semantic_digest_sha256": hashlib.sha256(b"direct").hexdigest(),
                    "operation_support": {"forward": True, "jvp": True, "vjp": True},
                }
            ),
            "DUAL_CONFIG_SCHEMA_VALID",
        ),
        (
            lambda config: config["paths"][1].update({"role": "curved"}),
            "DUAL_CONFIG_SCHEMA_VALID",
        ),
        (
            lambda config: config.update({"native_direct_residual_supported": True}),
            "DUAL_CONFIG_SCHEMA_VALID",
        ),
        (
            lambda config: config["cost_contract"].update(
                {"forward_api_calls": 29, "primary_total_requests": 37}
            ),
            "DUAL_CONFIG_SCHEMA_VALID",
        ),
        (
            lambda config: config["callable_contract"].update(
                {"precomputed_probe_arrays_are_sufficient": True}
            ),
            "DUAL_CONFIG_SCHEMA_VALID",
        ),
    ],
)
def test_dual_schema_fails_closed_on_third_path_role_capability_cost_or_lookup_table(
    dual_private_repo: dict[str, object], mutator: object, expected_code: str
) -> None:
    config = copy.deepcopy(dual_private_repo["config"])
    mutator(config)
    _write_config(dual_private_repo, config)
    report = _report(dual_private_repo)
    assert report["ready_for_private_describe_probe"] is False
    assert expected_code in report["blocker_codes"]
    assert not any(report["claim_authorizations"].values())


@pytest.mark.parametrize("field", ["path_id", "callable_id", "semantic_digest_sha256"])
def test_duplicate_path_identity_fields_are_blockers(
    dual_private_repo: dict[str, object], field: str
) -> None:
    config = copy.deepcopy(dual_private_repo["config"])
    config["paths"][1][field] = config["paths"][0][field]
    _write_config(dual_private_repo, config)
    report = _report(dual_private_repo)
    assert f"DUAL_UNIQUE_{field.upper()}" in report["blocker_codes"]
    assert report["ready_for_private_describe_probe"] is False


@pytest.mark.parametrize(
    "source",
    [
        (
            "def endpoint(role, x):\n"
            "    return x\n\n"
            "def wrapped(x):\n"
            "    return endpoint('curved', x) - endpoint('straight', x)\n"
        ),
        (
            "def endpoint(role, x):\n"
            "    return x\n\n"
            "def wrapped(x):\n"
            "    curved_output = endpoint('curved', x)\n"
            "    straight_output = endpoint('straight', x)\n"
            "    return curved_output - straight_output\n"
        ),
        (
            "import numpy as np\n\n"
            "def endpoint(role, x):\n"
            "    return x\n\n"
            "def wrapped(x):\n"
            "    curved_output = endpoint('curved', x)\n"
            "    straight_output = endpoint('straight', x)\n"
            "    return np.subtract(curved_output, straight_output)\n"
        ),
        (
            "import operator\n\n"
            "def endpoint(role, x):\n"
            "    return x\n\n"
            "def wrapped(x):\n"
            "    curved_output = endpoint('curved', x)\n"
            "    straight_output = endpoint('straight', x)\n"
            "    return operator.sub(curved_output, straight_output)\n"
        ),
        (
            "from operator import sub as minus\n\n"
            "def endpoint(role, x):\n"
            "    return x\n\n"
            "def wrapped(x):\n"
            "    curved_output = endpoint('curved', x)\n"
            "    straight_output = endpoint('straight', x)\n"
            "    return minus(curved_output, straight_output)\n"
        ),
        (
            "def endpoint(role, x):\n"
            "    return x\n\n"
            "def wrapped(x):\n"
            "    curved_output, straight_output = (\n"
            "        endpoint('curved', x), endpoint('straight', x)\n"
            "    )\n"
            "    return curved_output - straight_output\n"
        ),
        (
            "def endpoint(role, x):\n"
            "    return x\n\n"
            "def wrapped(x):\n"
            "    outputs = {\n"
            "        'curved': endpoint('curved', x),\n"
            "        'straight': endpoint('straight', x),\n"
            "    }\n"
            "    return outputs['curved'] - outputs['straight']\n"
        ),
    ],
)
def test_endpoint_subtraction_wrapper_is_detected(
    dual_private_repo: dict[str, object], source: str
) -> None:
    adapter = dual_private_repo["adapter"]
    adapter.write_text(source, encoding="utf-8")
    config = copy.deepcopy(dual_private_repo["config"])
    config["adapter"]["expected_implementation_sha256"] = sha256_file(adapter)
    _write_config(dual_private_repo, config)
    report = _report(dual_private_repo)
    assert report["wrapper_subtraction_status"] == "WRAPPER_SUBTRACTION_DETECTED"
    assert any(
        code.endswith("WRAPPER_ENDPOINT_SUBTRACTION_ABSENT")
        for code in report["blocker_codes"]
    )


@pytest.mark.parametrize(
    "source",
    [
        "def native_direct_residual_callable(x):\n    return x\n",
        "native_direct_residual = lambda x: x\n",
        "from helper import native_direct_residual as f\n",
        "setattr(api, 'native_direct_residual', lambda x: x)\n",
    ],
)
def test_direct_residual_callable_marker_is_detected(
    dual_private_repo: dict[str, object], source: str
) -> None:
    adapter = dual_private_repo["adapter"]
    adapter.write_text(source, encoding="utf-8")
    config = copy.deepcopy(dual_private_repo["config"])
    config["adapter"]["expected_implementation_sha256"] = sha256_file(adapter)
    _write_config(dual_private_repo, config)
    report = _report(dual_private_repo)
    assert any(
        code.endswith("NO_DIRECT_RESIDUAL_MARKER")
        for code in report["blocker_codes"]
    )


def test_source_hash_and_base_array_tampering_are_blockers(
    dual_private_repo: dict[str, object]
) -> None:
    dual_private_repo["adapter"].write_text(
        _safe_adapter_source() + "\nTAMPERED = True\n", encoding="utf-8"
    )
    report = _report(dual_private_repo)
    assert "DUAL_IMPLEMENTATION_HASH_MATCHES_ENTRYPOINT" in report["blocker_codes"]

    dual_private_repo["adapter"].write_text(_safe_adapter_source(), encoding="utf-8")
    np.save(dual_private_repo["base"], np.ones(7, dtype=np.float64))
    report = _report(dual_private_repo)
    assert "DUAL_BASE_INPUT_HASH_MATCHES" in report["blocker_codes"]
    assert "DUAL_BASE_INPUT_SIZE_MATCHES" in report["blocker_codes"]


def test_malformed_shared_identity_returns_blocked_report_instead_of_crashing(
    dual_private_repo: dict[str, object]
) -> None:
    config = copy.deepcopy(dual_private_repo["config"])
    config["identity"] = {}
    _write_config(dual_private_repo, config)
    report = _report(dual_private_repo)
    assert "DUAL_SHARED_IDENTITY_VALID" in report["blocker_codes"]
    assert report["ready_for_private_describe_probe"] is False


def test_dual_v2_l1_enters_l2a_with_exact_106_request_budget(
    dual_private_repo: dict[str, object]
) -> None:
    repo = dual_private_repo["repo"]
    for relative in PUBLIC_FOUNDATION_SOURCES:
        target = repo / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
    _git(repo, "add", *PUBLIC_FOUNDATION_SOURCES)
    _git(repo, "commit", "-qm", "add L2-A dual compatibility fixture")

    l1_report = _report(dual_private_repo)
    assert l1_report["ready_for_private_describe_probe"] is True
    bundle = dual_private_repo["bundle"]
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
                    "sha256": hashlib.sha256(b"dual-fixture-wheel").hexdigest(),
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
            "geometry_sha256": hashlib.sha256(b"dual-geometry").hexdigest(),
            "calibration_sha256": hashlib.sha256(b"dual-calibration").hexdigest(),
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
            "native_direct_residual": False,
            "direct_residual_semantics": "unavailable",
            "decoder_checkpoint_sha256": None,
            "noise_floor": {
                "units": "detector_displacement_pixel",
                "absolute_std": 1e-6,
            },
        },
    )
    control = repo / "private_library/n5_d5_control/anonymous_dual_case"
    control.mkdir(parents=True)
    l1_path = control / "l1-dual-v2.json"
    _write_json(l1_path, l1_report)
    plan = json.loads(
        (repo / PLAN_PLACEHOLDER_RELATIVE).read_text(encoding="utf-8")
    )
    plan["plan_id"] = "anonymous_dual_plan_v2"
    plan["l1_report"] = {
        "relative_path": l1_path.relative_to(repo).as_posix(),
        "sha256": sha256_file(l1_path),
    }

    def inventory(path: Path, role: str) -> dict[str, object]:
        return {
            "relative_path": path.relative_to(repo).as_posix(),
            "role": role,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    plan["private_bundle"] = {
        "root": bundle.relative_to(repo).as_posix(),
        "closed_world_required": True,
        "inventory": [
            inventory(dual_private_repo["config_path"], "config"),
            inventory(dual_private_repo["adapter"], "adapter_source"),
            inventory(dual_private_repo["base"], "base_input"),
            inventory(environment, "environment_lock"),
            inventory(physical, "physical_contract"),
        ],
    }
    plan["output_policy"]["result_directory"] = (
        "private_library/n5_d5_runs/anonymous_dual_case/run-001"
    )
    plan_path = control / "replay-plan.json"
    _write_json(plan_path, plan)

    report = build_foundation_report(
        plan_path, repo_root=repo, enforce_public_committed=True
    )
    assert report["status"] == "PRIVATE_REPLAY_FOUNDATION_READY_EXECUTION_LOCKED"
    assert report["foundation_ready"] is True
    assert report["authorization_budget"]["computed"]["path_count"] == 2
    assert report["authorization_budget"]["computed"]["primary_requests"] == 36
    assert report["authorization_budget"]["computed"]["total_requests"] == 106
    assert report["formal_36_call_replay_authorized"] is False
    assert report["formal_53_call_replay_authorized"] is False
    assert report["next_action"] == "IMPLEMENT_L2B_ISOLATED_DESCRIBE_RUNNER"
    assert not any(report["claim_authorizations"].values())
