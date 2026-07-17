#!/usr/bin/env python3
"""Independently validate a PSU-B0 Gate A mechanics attestation.

The validator distrusts reported PASS fields.  It verifies release hashes,
anchors every declared source and test to the recorded git commit, reruns the
declared tests, and independently recomputes the dense-factor, recurrence,
ledger, zero-row, and CPU/MPS parity checks.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys
import tempfile
from typing import Any, Sequence
import xml.etree.ElementTree as ET

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from site_tools.psu_b0_gate_a_independent_oracle import (  # noqa: E402
    build_independent_oracle,
    run_numpy_recurrence,
    run_torch_dense_recurrence,
)


ATTESTATION_SCHEMA = "psu-b0-gate-a-attestation-1.0"
FORMAL_STATUS = "FORMAL_GATE_A_ATTESTED_MECHANICS_ONLY"
VALIDATOR_SCHEMA = "psu-b0-gate-a-independent-validator-1.0"
VALIDATOR_STATUS = "PASS_INDEPENDENT_GATE_A_MECHANICS_VALIDATION"
CLAIM_BOUNDARY = "GATE_B_NOT_RUN_NO_FRESH_REAL_OR_WIN_CLAIM"
GATE_A_THRESHOLD_CEILINGS = {
    "float64_adjoint_relative_error_max": 1e-8,
    "float64_dominance_violation_max": 1e-12,
    "float64_sum_relative_error_max": 1e-10,
    "float64_svd_slack": 1e-12,
    "float64_recurrence_relative_error_max": 1e-10,
    "mps_float32_field_relative_difference_max": 5e-4,
}
EXPECTED_TEST_NODE_MANIFEST_SHA256 = (
    "b638e53bba305318dcbb4242d00d17446929bf4b8af6c6d978d5951301376a5e"
)
EXPECTED_E1_MAPPING_SHA256 = (
    "6b85214c7d1c63fb7ee63dee0a4d5834e23a2a3487b49c9ba1e468747cbfe874"
)
SHA256 = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")
DEFAULT_GATE_A_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "demo_t16_operator"
    / "configs"
    / "psu_b0_gate_a_attestation_v1.json"
)
RUNNER_FILES = {
    "attestation.json",
    "pytest.xml",
    "pytest_output.txt",
    "collected_nodes.txt",
    "attestation_payload.sha256",
}
RELEASE_CONTENT_FILES = RUNNER_FILES | {
    "checksums.sha256",
    "validation_report.json",
}
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


class ValidationError(AssertionError):
    pass


class Validator:
    def __init__(self) -> None:
        self.checks = 0

    def require(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            raise ValidationError(message)

    def close(
        self,
        actual: Any,
        expected: Any,
        message: str,
        *,
        rel_tol: float = 1e-8,
        abs_tol: float = 1e-11,
    ) -> None:
        left = float(actual)
        right = float(expected)
        self.require(
            math.isfinite(left)
            and math.isfinite(right)
            and math.isclose(left, right, rel_tol=rel_tol, abs_tol=abs_tol),
            f"{message}: {left} != {right}",
        )


def _reject_constant(raw: str) -> None:
    raise ValidationError(f"non-finite JSON constant: {raw}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_strict_json(path: Path) -> dict[str, Any]:
    values = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
        object_pairs_hook=_unique_object,
    )
    if not isinstance(values, dict):
        raise ValidationError(f"JSON root must be an object: {path}")
    return values


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _safe_repository_path(raw_path: str) -> Path:
    token = PurePosixPath(raw_path)
    if token.is_absolute() or ".." in token.parts or not token.parts:
        raise ValidationError(f"unsafe repository path: {raw_path}")
    path = (REPOSITORY_ROOT / token).resolve()
    if REPOSITORY_ROOT not in path.parents:
        raise ValidationError(f"repository path escaped root: {raw_path}")
    return path


def _load_independent_config(path: Path) -> dict[str, Any]:
    config = load_strict_json(path)
    if config.get("schema_version") != "psu-b0-gate-a-attestation-config-1.0":
        raise ValidationError("config schema drift")
    if config.get("evidence_scope") != "VIEW_LOCAL_SINGLE_FROZEN_SCALE_MECHANICS_ONLY":
        raise ValidationError("config scope drift")
    if config.get("scientific_claim_boundary") != "NO_GATE_B_NO_FRESH_NO_REAL_NO_WIN_CLAIM":
        raise ValidationError("config claim boundary drift")
    provenance = config.get("calibration_provenance")
    if provenance != {
        "kind": "FROZEN_SYNTHETIC_MECHANICS_FIXTURE",
        "independent_flow_off_calibration": False,
        "truth_available_to_setup": False,
        "morphology_available_to_setup": False,
        "deployment_authorized": False,
    }:
        raise ValidationError("config calibration provenance drift")
    fixture = config.get("fixture")
    selectors = config.get("test_nodes")
    sources = config.get("source_files")
    mapping = config.get("e1_test_mapping")
    if not isinstance(fixture, dict) or not isinstance(selectors, list):
        raise ValidationError("config fixture or test selectors missing")
    if not isinstance(sources, list) or not isinstance(mapping, dict):
        raise ValidationError("config source or E1 map missing")
    thresholds = config.get("thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != set(
        GATE_A_THRESHOLD_CEILINGS
    ):
        raise ValidationError("config threshold key set drift")
    for name, ceiling in GATE_A_THRESHOLD_CEILINGS.items():
        value = thresholds[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 < float(value) <= ceiling
        ):
            raise ValidationError(f"config threshold is invalid or relaxed: {name}")
    forbidden = {"truth", "truth_field", "morphology", "ground_truth"}
    if forbidden & set(fixture):
        raise ValidationError("truth-like field entered mechanics config")
    if fixture.get("iterations") != 6 or fixture.get("penalty") != "huber":
        raise ValidationError("config recurrence drift")
    expected_gates = {f"E1-{index:02d}" for index in range(1, 14)}
    if set(mapping) != expected_gates:
        raise ValidationError("config E1 map drift")
    if len(selectors) != len(set(selectors)) or len(sources) != len(set(sources)):
        raise ValidationError("duplicate test selector or source path")
    if canonical_json_sha256(selectors) != EXPECTED_TEST_NODE_MANIFEST_SHA256:
        raise ValidationError("config test-node manifest drift")
    if canonical_json_sha256(mapping) != EXPECTED_E1_MAPPING_SHA256:
        raise ValidationError("config E1 mapping manifest drift")
    source_set = set(sources)
    for selector in selectors:
        if not isinstance(selector, str) or "::test_" not in selector:
            raise ValidationError("test selector is not explicit")
        if selector.split("::", 1)[0] not in source_set:
            raise ValidationError("test selector file is not source-fingerprinted")
    for raw_path in sources:
        if not isinstance(raw_path, str):
            raise ValidationError("source path is not a string")
        _safe_repository_path(raw_path)
    for gate, indices in mapping.items():
        if not isinstance(indices, list) or not indices:
            raise ValidationError(f"{gate} has no tests")
        if any(
            not isinstance(index, int) or not 0 <= index < len(selectors)
            for index in indices
        ):
            raise ValidationError(f"{gate} test index drift")
    return config


def _input_payload(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "psu-b0-gate-a-fixture-1.0",
        "evidence_scope": config["evidence_scope"],
        "scientific_claim_boundary": config["scientific_claim_boundary"],
        "fixture": config["fixture"],
        "calibration_provenance": config["calibration_provenance"],
        "thresholds": config["thresholds"],
    }


def _source_hashes(config: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_path in config["source_files"]:
        path = _safe_repository_path(raw_path)
        if not path.is_file() or path.is_symlink():
            raise ValidationError(f"source file missing or symlinked: {raw_path}")
        result[raw_path] = file_sha256(path)
    return result


def _all_numbers_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_all_numbers_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_all_numbers_finite(item) for item in value.values())
    return False


def _parse_checksums(validator: Validator, evidence: Path) -> dict[str, str]:
    path = evidence / "checksums.sha256"
    validator.require(path.is_file() and not path.is_symlink(), "missing checksum manifest")
    recorded: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        parts = line.split("  ", 1)
        validator.require(len(parts) == 2, "malformed checksum line")
        digest, raw_name = parts
        token = PurePosixPath(raw_name)
        validator.require(bool(SHA256.fullmatch(digest)), f"invalid checksum: {raw_name}")
        validator.require(
            not token.is_absolute() and ".." not in token.parts and len(token.parts) == 1,
            f"unsafe checksum path: {raw_name}",
        )
        validator.require(raw_name not in recorded, f"duplicate checksum path: {raw_name}")
        target = evidence / raw_name
        validator.require(
            target.is_file() and not target.is_symlink(),
            f"missing or symlinked evidence file: {raw_name}",
        )
        validator.require(file_sha256(target) == digest, f"checksum mismatch: {raw_name}")
        recorded[raw_name] = digest
    validator.require(set(recorded) == RUNNER_FILES, "runner checksum file set drifted")
    return recorded


def _git_bytes(commit: str, raw_path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{raw_path}"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    return result.stdout


def _git_clean() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPOSITORY_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return not bool(result.stdout.strip())


def _git_index_is_plain() -> bool:
    result = subprocess.run(
        ["git", "ls-files", "-v"],
        cwd=REPOSITORY_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return all(line.startswith("H ") for line in result.stdout.splitlines())


def _environment_report() -> dict[str, Any]:
    pip = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        cwd=REPOSITORY_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    packages = sorted(
        line.strip() for line in pip.stdout.splitlines() if line.strip()
    )
    return {
        "python": platform.python_version(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cpu_count": os.cpu_count(),
        "python_executable_sha256": file_sha256(Path(sys.executable).resolve()),
        "pip_freeze_entry_count": len(packages),
        "pip_freeze_sha256": canonical_json_sha256(packages),
        "module_entry_sha256": {
            "torch": file_sha256(Path(torch.__file__).resolve()),
            "numpy": file_sha256(Path(np.__file__).resolve()),
        },
        "distribution_tree_fingerprints": {
            name: _distribution_tree_fingerprint(name)
            for name in ("torch", "numpy", "pytest")
        },
        "environment_isolation": "LOCAL_HOST_NOT_ISOLATED_FINGERPRINT_ONLY",
        "pytest_plugin_autoload_disabled": True,
        "python_user_site_disabled": True,
        "mps_fallback_disabled": True,
    }


def _distribution_tree_fingerprint(project: str) -> dict[str, Any]:
    distribution = metadata.distribution(project)
    roots: dict[str, Path] = {}
    declared = distribution.files or ()
    for raw in declared:
        token = PurePosixPath(str(raw))
        if not token.parts or token.parts[0] in {".", ".."}:
            continue
        root_name = token.parts[0]
        root = Path(distribution.locate_file(root_name)).resolve()
        if root.exists():
            roots[root_name] = root
    manifest: dict[str, str] = {}
    symlink_count = 0
    for root_name, root in sorted(roots.items()):
        candidates = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in candidates:
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            if path.is_symlink():
                symlink_count += 1
            if not path.is_file():
                continue
            relative = (
                root_name
                if path == root
                else f"{root_name}/{path.relative_to(root).as_posix()}"
            )
            manifest[relative] = file_sha256(path)
    if not manifest:
        raise ValidationError(f"empty installed distribution tree: {project}")
    return {
        "version": distribution.version,
        "declared_file_count": len(declared),
        "scanned_file_count": len(manifest),
        "symlink_count": symlink_count,
        "tree_sha256": canonical_json_sha256(manifest),
    }


def _validate_provenance(
    validator: Validator,
    report: dict[str, Any],
    config: dict[str, Any],
    config_path: Path,
) -> None:
    validator.require(report.get("schema_version") == ATTESTATION_SCHEMA, "schema drift")
    validator.require(report.get("status") == FORMAL_STATUS, "formal status drift")
    validator.require(report.get("evidence_scope") == config["evidence_scope"], "scope drift")
    validator.require(report.get("scientific_claim_boundary") == CLAIM_BOUNDARY, "claim boundary drift")
    validator.require(report.get("gate_b_status") == "NOT_RUN", "Gate B must remain not run")
    validator.require(report.get("fresh_data_status") == "NOT_RUN", "fresh data must remain not run")
    validator.require(report.get("real_data_status") == "NOT_RUN", "real data must remain not run")
    validator.require(
        report.get("method_superiority_status") == "NOT_ESTABLISHED",
        "superiority must remain unestablished",
    )
    validator.require(
        report.get("environment_independence_status")
        == "NOT_ESTABLISHED_LOCAL_HOST_FINGERPRINT_ONLY",
        "environment independence must remain unestablished",
    )
    validator.require(report.get("calibration_provenance") == config["calibration_provenance"], "calibration provenance drift")
    validator.require(_all_numbers_finite(report), "attestation contains unsupported or non-finite values")

    source = report.get("source")
    fingerprints = report.get("fingerprints")
    validator.require(isinstance(source, dict), "missing source record")
    validator.require(isinstance(fingerprints, dict), "missing fingerprints")
    commit = source.get("commit")
    validator.require(isinstance(commit, str) and bool(COMMIT.fullmatch(commit)), "invalid source commit")
    validator.require(source.get("worktree_clean_before") is True, "runner did not start clean")
    validator.require(source.get("worktree_clean_after") is True, "runner did not finish clean")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=REPOSITORY_ROOT,
    )
    validator.require(ancestor.returncode == 0, "recorded source commit is not an ancestor of HEAD")

    config_relative = config_path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    config_hash = file_sha256(config_path)
    validator.require(fingerprints.get("config_file_sha256") == config_hash, "config file hash drift")
    validator.require(
        hashlib.sha256(_git_bytes(commit, config_relative)).hexdigest() == config_hash,
        "recorded commit does not contain the attested config",
    )
    input_hash = canonical_json_sha256(_input_payload(config))
    test_hash = canonical_json_sha256(config["test_nodes"])
    source_hashes = _source_hashes(config)
    validator.require(fingerprints.get("input_payload_sha256") == input_hash, "input payload hash drift")
    validator.require(fingerprints.get("test_node_manifest_sha256") == test_hash, "test-node hash drift")
    validator.require(fingerprints.get("source_files_sha256") == source_hashes, "source-file map drift")
    for raw_name, digest in source_hashes.items():
        validator.require(
            hashlib.sha256(_git_bytes(commit, raw_name)).hexdigest() == digest,
            f"recorded commit source mismatch: {raw_name}",
        )
    bundle_hash = canonical_json_sha256(source_hashes)
    validator.require(fingerprints.get("source_bundle_sha256") == bundle_hash, "source bundle hash drift")
    test_attestation = report.get("test_attestation")
    validator.require(isinstance(test_attestation, dict), "test attestation missing")
    collected_nodes = test_attestation.get("collected_node_ids")
    validator.require(isinstance(collected_nodes, list) and collected_nodes, "collected nodes missing")
    collected_hash = canonical_json_sha256(collected_nodes)
    validator.require(
        fingerprints.get("collected_node_manifest_sha256") == collected_hash,
        "collected-node hash drift",
    )
    environment_hash = canonical_json_sha256(report.get("environment"))
    validator.require(
        fingerprints.get("environment_lock_sha256") == environment_hash,
        "environment lock hash drift",
    )
    identity = canonical_json_sha256(
        {
            "source_commit": commit,
            "config_file_sha256": config_hash,
            "input_payload_sha256": input_hash,
            "test_node_manifest_sha256": test_hash,
            "collected_node_manifest_sha256": collected_hash,
            "source_bundle_sha256": bundle_hash,
            "environment_lock_sha256": environment_hash,
        }
    )
    validator.require(fingerprints.get("attestation_identity_sha256") == identity, "attestation identity drift")


def _parse_junit(path: Path) -> dict[str, int | bool]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    totals = {
        field: sum(int(suite.attrib.get(field, "0")) for suite in suites)
        for field in ("tests", "failures", "errors", "skipped")
    }
    totals["passed"] = totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"]
    totals["all_passed_without_skip"] = (
        totals["tests"] > 0
        and totals["failures"] == 0
        and totals["errors"] == 0
        and totals["skipped"] == 0
    )
    return totals


def _clean_test_environment() -> dict[str, str]:
    for name in (
        "PYTEST_ADDOPTS",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTORCH_ENABLE_MPS_FALLBACK",
    ):
        if os.environ.get(name):
            raise ValidationError(f"validator forbids inherited {name}")
    environment = os.environ.copy()
    environment.update(THREAD_ENVIRONMENT)
    environment["PYTEST_ADDOPTS"] = ""
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONSTARTUP", None)
    environment.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    return environment


def _collect_nodes(config: dict[str, Any], environment: dict[str, str]) -> list[str]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "--color=no",
            "--override-ini=addopts=",
            "--noconftest",
            *config["test_nodes"],
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0 or result.stderr.strip():
        raise ValidationError("independent test collection failed or emitted stderr")
    nodes = [
        line.strip()
        for line in result.stdout.splitlines()
        if "::" in line and not line.lstrip().startswith("<")
    ]
    if not nodes or len(nodes) != len(set(nodes)):
        raise ValidationError("independent collection is empty or duplicated")
    return nodes


def _validate_declared_tests(
    validator: Validator,
    report: dict[str, Any],
    config: dict[str, Any],
    evidence: Path,
) -> dict[str, Any]:
    recorded = _parse_junit(evidence / "pytest.xml")
    attested = report.get("test_attestation")
    validator.require(isinstance(attested, dict), "missing test attestation")
    for key in (
        "tests",
        "failures",
        "errors",
        "skipped",
        "passed",
        "all_passed_without_skip",
    ):
        validator.require(
            attested.get(key) == recorded[key],
            f"recorded JUnit summary drift: {key}",
        )
    recorded_nodes = [
        line.strip()
        for line in (evidence / "collected_nodes.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    validator.require(
        attested.get("collected_node_ids") == recorded_nodes,
        "report and collected-node file differ",
    )
    validator.require(
        attested.get("collected_node_manifest_sha256")
        == canonical_json_sha256(recorded_nodes),
        "collected-node manifest drift",
    )
    validator.require(
        recorded["all_passed_without_skip"] is True,
        "recorded tests did not all pass",
    )
    validator.require(
        recorded["tests"] == len(recorded_nodes),
        "recorded testcase count differs from collected nodes",
    )
    validator.require(
        attested.get("declared_node_count") == len(config["test_nodes"]),
        "declared node count drift",
    )
    environment = _clean_test_environment()
    replay_nodes = _collect_nodes(config, environment)
    validator.require(replay_nodes == recorded_nodes, "exact collected node ids drifted")
    with tempfile.TemporaryDirectory(prefix="psu-gate-a-validator-") as directory:
        junit = Path(directory) / "pytest.xml"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--color=no",
                "--override-ini=addopts=",
                "--noconftest",
                f"--junitxml={junit}",
                *config["test_nodes"],
            ],
            cwd=REPOSITORY_ROOT,
            env=environment,
            text=True,
            capture_output=True,
        )
        validator.require(
            result.returncode == 0 and not result.stderr.strip(),
            "independent declared-test replay failed or emitted stderr",
        )
        replay = _parse_junit(junit)
    validator.require(
        replay["all_passed_without_skip"] is True,
        "test replay failed or skipped",
    )
    validator.require(replay["tests"] == len(replay_nodes), "test replay count drift")
    return {
        "recorded": recorded,
        "replayed": replay,
        "declared_node_count": len(config["test_nodes"]),
        "collected_node_count": len(recorded_nodes),
        "collected_node_manifest_sha256": canonical_json_sha256(recorded_nodes),
    }


def _relative_arrays(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    return float(
        np.linalg.norm(a - b)
        / max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1e-30)
    )


def _trace_maximum_error(
    recorded: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    validator: Validator,
) -> float:
    validator.require(len(recorded) == len(reference) == 6, "six-step trace length drift")
    maximum = 0.0
    for step, (observed, expected) in enumerate(zip(recorded, reference), start=1):
        for name in ("x", "x_bar", "data_dual", "tv_dual"):
            error = _relative_arrays(observed[name], expected[name])
            validator.require(math.isfinite(error), f"non-finite trace error at step {step}")
            maximum = max(maximum, error)
        validator.close(
            observed["objective"],
            expected["objective"],
            f"objective step {step}",
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    return maximum


def _independent_cpu_check(
    validator: Validator,
    report: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    oracle = build_independent_oracle(config)
    cpu = report.get("cpu_float64")
    validator.require(isinstance(cpu, dict), "missing CPU report")
    thresholds = config["thresholds"]
    data_violation = float(np.maximum(np.abs(oracle.A) - oracle.M, 0.0).max())
    tv_violation = float(np.maximum(np.abs(oracle.D) - oracle.N, 0.0).max())
    validator.require(
        max(data_violation, tv_violation)
        <= thresholds["float64_dominance_violation_max"],
        "independent NumPy dominance failed",
    )
    validator.close(
        cpu["metric"]["data_dominance_violation_max"],
        data_violation,
        "reported data dominance",
    )
    validator.close(
        cpu["metric"]["tv_dominance_violation_max"],
        tv_violation,
        "reported TV dominance",
    )
    norm_squared = oracle.scaled_norm_squared
    validator.require(
        norm_squared <= oracle.eta**2 + thresholds["float64_svd_slack"],
        "independent NumPy SVD safety failed",
    )
    validator.close(
        cpu["metric"]["scaled_operator_norm_squared"],
        norm_squared,
        "reported scaled norm",
        rel_tol=1e-10,
    )
    validator.require(
        cpu["metric"]["active_data_rows"]
        == int(np.count_nonzero(oracle.data_row_mask)),
        "active data row count drift",
    )
    validator.require(
        cpu["metric"]["active_primal_columns"]
        == int(oracle.active_primal_indices.size),
        "active primal count drift",
    )

    vector = np.sin(np.arange(oracle.A.shape[1]) * 0.31)
    for name, matrix in (
        ("A", oracle.A),
        ("P", oracle.P),
        ("G", oracle.G),
        ("D", oracle.D),
    ):
        x = vector if matrix.shape[1] == vector.size else np.cos(
            np.arange(matrix.shape[1]) * 0.23
        )
        y = np.sin(np.arange(matrix.shape[0]) * 0.17 + 0.4)
        error = abs(float((matrix @ x) @ y - x @ (matrix.T @ y))) / max(
            abs(float((matrix @ x) @ y)),
            abs(float(x @ (matrix.T @ y))),
            1e-30,
        )
        validator.require(
            error <= thresholds["float64_adjoint_relative_error_max"],
            f"independent {name} adjoint failed",
        )

    reference = run_numpy_recurrence(oracle, config)
    recorded_trace = cpu["recurrence"].get("state_trace")
    validator.require(isinstance(recorded_trace, list), "CPU state trace missing")
    maximum_trace_error = _trace_maximum_error(
        recorded_trace,
        reference,
        validator,
    )
    validator.require(
        maximum_trace_error
        <= thresholds["float64_recurrence_relative_error_max"],
        "independent NumPy six-step recurrence failed",
    )
    validator.require(
        _relative_arrays(
            cpu["recurrence"]["final_reduced_x"], reference[-1]["x"]
        )
        <= thresholds["float64_recurrence_relative_error_max"],
        "reported final state drift",
    )
    validator.require(
        _relative_arrays(
            cpu["recurrence"]["final_reduced_x_bar"],
            reference[-1]["x_bar"],
        )
        <= thresholds["float64_recurrence_relative_error_max"],
        "reported final x_bar drift",
    )

    deleted = np.flatnonzero(~oracle.data_row_mask)
    deleted_values = oracle.target[deleted]
    deleted_constant = 0.5 * float(deleted_values @ deleted_values)
    zero = cpu["zero_coupling"]
    validator.require(zero["deleted_data_indices"] == deleted.tolist(), "deleted row indices drift")
    validator.require(
        _relative_arrays(zero["deleted_target_values"], deleted_values) <= 1e-12,
        "deleted target values drift",
    )
    validator.close(zero["objective_constant"], deleted_constant, "deleted constant")

    zero_ledger = {
        "signed_data_forward_calls": 0,
        "signed_data_transpose_calls": 0,
        "absolute_data_forward_calls": 0,
        "absolute_data_transpose_calls": 0,
        "signed_tv_forward_calls": 0,
        "signed_tv_transpose_calls": 0,
        "absolute_tv_forward_calls": 0,
        "absolute_tv_transpose_calls": 0,
    }
    expected_ledgers = {
        "setup": {
            **zero_ledger,
            "absolute_data_forward_calls": 1,
            "absolute_data_transpose_calls": 1,
            "absolute_tv_forward_calls": 1,
            "absolute_tv_transpose_calls": 1,
        },
        "oracle_audit": {
            **zero_ledger,
            "signed_data_forward_calls": 2,
            "signed_data_transpose_calls": 1,
            "absolute_data_forward_calls": 1,
            "signed_tv_forward_calls": 2,
            "signed_tv_transpose_calls": 1,
            "absolute_tv_forward_calls": 1,
        },
        "solve": {
            **zero_ledger,
            "signed_data_forward_calls": 6,
            "signed_data_transpose_calls": 6,
            "signed_tv_forward_calls": 6,
            "signed_tv_transpose_calls": 6,
        },
        "scorer": {
            **zero_ledger,
            "signed_data_forward_calls": 6,
            "signed_tv_forward_calls": 6,
        },
    }
    ledgers = cpu["ledgers"]
    for segment, expected in expected_ledgers.items():
        keys = (
            ("setup_logical", "setup_physical", "setup_expected")
            if segment == "setup"
            else (
                f"{segment}_logical_delta",
                f"{segment}_physical_delta",
                f"{segment}_expected",
            )
        )
        for key in keys:
            validator.require(ledgers.get(key) == expected, f"{key} ledger drift")
    return {
        "oracle_schema": oracle.schema_version,
        "dominance_violation_max": max(data_violation, tv_violation),
        "scaled_operator_norm_squared": norm_squared,
        "eta_squared": float(oracle.eta**2),
        "maximum_six_step_state_relative_error": maximum_trace_error,
        "deleted_data_rows": int(deleted.size),
    }


def _independent_mps_check(
    validator: Validator,
    report: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    validator.require(
        torch.backends.mps.is_available(),
        "MPS unavailable during independent validation",
    )
    oracle = build_independent_oracle(config)
    cpu_trace = run_torch_dense_recurrence(
        oracle, config, device="cpu", dtype=torch.float64
    )
    mps_trace = run_torch_dense_recurrence(
        oracle, config, device="mps", dtype=torch.float32
    )
    torch.mps.synchronize()
    mps_report = report.get("mps_float32_parity")
    validator.require(
        isinstance(mps_report, dict) and mps_report.get("available") is True,
        "missing MPS report",
    )
    recorded_cpu = mps_report.get("cpu_state_trace")
    recorded_mps = mps_report.get("mps_state_trace")
    validator.require(
        isinstance(recorded_cpu, list) and isinstance(recorded_mps, list),
        "MPS trace payload missing",
    )
    validator.require(
        len(recorded_cpu) == len(recorded_mps) == len(cpu_trace) == len(mps_trace) == 6,
        "MPS trace length drift",
    )
    maximum = 0.0
    field_final = 0.0
    by_step: list[dict[str, float]] = []
    for index, (cpu_state, mps_state) in enumerate(zip(cpu_trace, mps_trace)):
        row: dict[str, float] = {}
        for name in ("x", "x_bar", "data_dual", "tv_dual"):
            cpu_array = cpu_state[name].detach().cpu().numpy()
            mps_array = mps_state[name].detach().cpu().numpy()
            validator.require(
                _relative_arrays(recorded_cpu[index][name], cpu_array) <= 1e-10,
                f"reported CPU parity trace drift at step {index + 1}: {name}",
            )
            validator.require(
                _relative_arrays(recorded_mps[index][name], mps_array) <= 1e-6,
                f"reported MPS trace drift at step {index + 1}: {name}",
            )
            row[name] = _relative_arrays(cpu_array, mps_array)
            maximum = max(maximum, row[name])
        by_step.append(row)
        field_final = row["x"]
    threshold = config["thresholds"]["mps_float32_field_relative_difference_max"]
    validator.require(maximum <= threshold, "independent dense CPU/MPS state parity failed")
    validator.close(
        mps_report["field_relative_difference"],
        field_final,
        "reported MPS field difference",
        rel_tol=5e-3,
        abs_tol=1e-8,
    )
    reported_errors = mps_report.get("state_relative_errors_by_step")
    validator.require(
        isinstance(reported_errors, list) and len(reported_errors) == 6,
        "reported per-step MPS errors missing",
    )
    for index, row in enumerate(reported_errors):
        for name in ("x", "x_bar", "data_dual", "tv_dual"):
            recomputed_report_error = _relative_arrays(
                recorded_cpu[index][name], recorded_mps[index][name]
            )
            validator.close(
                row[name],
                recomputed_report_error,
                f"reported production parity step {index + 1}: {name}",
                rel_tol=1e-7,
                abs_tol=1e-12,
            )
    return {
        "field_relative_difference": field_final,
        "maximum_state_relative_difference": maximum,
        "threshold": threshold,
    }


def _validate_e1_records(validator: Validator, report: dict[str, Any], config: dict[str, Any]) -> None:
    records = report.get("e1_records")
    expected = {f"E1-{index:02d}" for index in range(1, 14)}
    validator.require(isinstance(records, dict) and set(records) == expected, "E1 record set drift")
    for gate, indices in config["e1_test_mapping"].items():
        record = records[gate]
        validator.require(record.get("status") == "PASS", f"{gate} is not PASS")
        validator.require(record.get("numeric_contract_passed") is True, f"{gate} numeric contract failed")
        validator.require(record.get("declared_tests_passed_without_skip") is True, f"{gate} tests failed")
        validator.require(record.get("test_nodes") == [config["test_nodes"][index] for index in indices], f"{gate} test mapping drift")


def _validate_payload_marker(
    validator: Validator,
    report: dict[str, Any],
    report_path: Path,
    marker_path: Path,
) -> None:
    lines = marker_path.read_text(encoding="ascii").splitlines()
    validator.require(len(lines) == 2, "payload hash marker line count drift")
    expected = {
        "attestation.json.canonical-json-v1": canonical_json_sha256(report),
        "attestation.json.raw-file": file_sha256(report_path),
    }
    observed: dict[str, str] = {}
    for line in lines:
        parts = line.split("  ", 1)
        validator.require(len(parts) == 2, "malformed payload hash marker")
        digest, name = parts
        validator.require(bool(SHA256.fullmatch(digest)), "invalid payload digest")
        validator.require(name not in observed, "duplicate payload marker name")
        observed[name] = digest
    validator.require(observed == expected, "detached canonical/raw payload hash drift")


def _validate_negative_controls(
    validator: Validator,
    report: dict[str, Any],
) -> None:
    controls = report.get("negative_controls")
    expected_rejections = {
        "stale_signed_kernel",
        "stale_scale_by_view",
        "stale_geometry_weight",
        "stale_support",
        "stale_tau",
        "stale_sigma_data",
        "stale_data_mask",
        "stale_active_primal_indices",
        "stale_pipeline_device_indices",
        "stale_voxel_spacing",
        "multiple_scale_instances",
        "nonzero_zero_tolerance",
        "cross_view_metadata",
    }
    validator.require(
        isinstance(controls, dict)
        and set(controls) == expected_rejections | {"tiny_nonzero_retained"},
        "negative-control set drift",
    )
    for name in expected_rejections:
        control = controls[name]
        validator.require(control.get("rejected") is True, f"{name} did not reject")
        validator.require(
            isinstance(control.get("exception_type"), str)
            and isinstance(control.get("message"), str)
            and bool(control["message"]),
            f"{name} rejection evidence missing",
        )
    tiny = controls["tiny_nonzero_retained"]
    validator.require(tiny.get("rejected") is False, "tiny coupling was rejected")
    validator.require(tiny.get("retained") is True, "tiny coupling was not retained")
    validator.close(tiny.get("coupling"), 1e-12, "tiny coupling value")


def _validate_evidence_directory(validator: Validator, evidence: Path) -> None:
    allowed = RUNNER_FILES | {
        "checksums.sha256",
        "validation_report.json",
        "release_checksums.sha256",
    }
    entries = list(evidence.iterdir())
    validator.require(
        all(path.is_file() or path.is_symlink() for path in entries),
        "evidence directory must be a flat file bundle",
    )
    files = [path for path in entries if path.is_file() or path.is_symlink()]
    validator.require(
        {path.name for path in files} <= allowed,
        "evidence directory contains an unapproved file",
    )
    identities: set[tuple[int, int]] = set()
    for path in files:
        validator.require(not path.is_symlink(), f"symlinked evidence file: {path.name}")
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        validator.require(identity not in identities, f"hardlink alias in evidence: {path.name}")
        identities.add(identity)


def _validate_release_bundle(
    validator: Validator,
    evidence: Path,
    expected_validation: dict[str, Any],
) -> None:
    validation_path = evidence / "validation_report.json"
    manifest_path = evidence / "release_checksums.sha256"
    validator.require(
        validation_path.is_file() and not validation_path.is_symlink(),
        "missing independent validation report",
    )
    validator.require(
        manifest_path.is_file() and not manifest_path.is_symlink(),
        "missing release checksum manifest",
    )
    validator.require(
        load_strict_json(validation_path) == expected_validation,
        "published validation report differs from independent recomputation",
    )
    recorded: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="ascii").splitlines():
        parts = line.split("  ", 1)
        validator.require(len(parts) == 2, "malformed release checksum line")
        digest, raw_name = parts
        token = PurePosixPath(raw_name)
        validator.require(
            bool(SHA256.fullmatch(digest)),
            f"invalid release checksum: {raw_name}",
        )
        validator.require(
            not token.is_absolute()
            and ".." not in token.parts
            and len(token.parts) == 1,
            f"unsafe release checksum path: {raw_name}",
        )
        validator.require(
            raw_name not in recorded,
            f"duplicate release checksum path: {raw_name}",
        )
        target = evidence / raw_name
        validator.require(
            target.is_file() and not target.is_symlink(),
            f"missing or symlinked release file: {raw_name}",
        )
        validator.require(
            file_sha256(target) == digest,
            f"release checksum mismatch: {raw_name}",
        )
        recorded[raw_name] = digest
    validator.require(
        set(recorded) == RELEASE_CONTENT_FILES,
        "release checksum file set drifted",
    )


def validate_attestation(
    *,
    config_path: Path,
    evidence: Path,
    write_report: bool,
) -> dict[str, Any]:
    evidence = evidence.resolve()
    preflight_validator = Validator()
    preflight_validator.require(
        evidence.is_dir() and not evidence.is_symlink(),
        "evidence directory missing or symlinked",
    )
    _validate_evidence_directory(preflight_validator, evidence)
    validator = Validator()
    validator.require(_git_clean(), "independent validator requires a clean worktree")
    validator.require(_git_index_is_plain(), "validator forbids non-plain git index flags")
    checksums = _parse_checksums(validator, evidence)
    report_path = evidence / "attestation.json"
    report = load_strict_json(report_path)
    _validate_payload_marker(
        validator,
        report,
        report_path,
        evidence / "attestation_payload.sha256",
    )
    config = _load_independent_config(config_path.resolve())
    _validate_provenance(validator, report, config, config_path.resolve())
    validator.require(
        report.get("environment") == _environment_report(),
        "runtime environment no longer matches attestation",
    )
    tests = _validate_declared_tests(validator, report, config, evidence)
    _validate_e1_records(validator, report, config)
    _validate_negative_controls(validator, report)
    cpu = _independent_cpu_check(validator, report, config)
    mps = _independent_mps_check(validator, report, config)
    validator.require(_git_clean(), "independent validation changed the worktree")
    validator.require(_git_index_is_plain(), "git index flags changed during validation")
    validation = {
        "schema_version": VALIDATOR_SCHEMA,
        "status": VALIDATOR_STATUS,
        "scientific_claim_boundary": CLAIM_BOUNDARY,
        "gate_b_status": "NOT_RUN",
        "environment_independence_status": (
            "NOT_ESTABLISHED_LOCAL_HOST_FINGERPRINT_ONLY"
        ),
        "attestation_sha256": checksums["attestation.json"],
        "attestation_identity_sha256": report["fingerprints"]["attestation_identity_sha256"],
        "source_commit": report["source"]["commit"],
        "checks": validator.checks,
        "test_replay": tests,
        "independent_cpu_float64": cpu,
        "independent_mps_float32": mps,
    }
    if write_report:
        validation_path = evidence / "validation_report.json"
        validation_path.write_text(
            json.dumps(validation, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        release_files = [
            evidence / name
            for name in sorted(RELEASE_CONTENT_FILES)
        ]
        (evidence / "release_checksums.sha256").write_text(
            "".join(f"{file_sha256(path)}  {path.name}\n" for path in release_files),
            encoding="ascii",
        )
    release_validator = Validator()
    _validate_release_bundle(release_validator, evidence, validation)
    return validation


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_GATE_A_CONFIG_PATH)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = validate_attestation(
        config_path=args.config,
        evidence=args.evidence,
        write_report=not args.no_write,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
