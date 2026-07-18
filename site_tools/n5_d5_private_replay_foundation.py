#!/usr/bin/env python3
"""N5-D5-L2 private replay foundation and fail-closed bundle utilities.

This module never imports or executes a private adapter.  It binds a committed
public protocol to a content-addressed private input bundle, freezes a closed
private result policy, and provides an in-memory CSPRNG probe generator for a
future L2-B runner.  A passing foundation report never authorizes describe,
the 36/53-call base replay, reconstruction, training, or a scientific claim.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
from typing import Any, Callable

from jsonschema import Draft202012Validator
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PLAN_SCHEMA_RELATIVE = "data_templates/n5_d5_private_replay_plan.schema.json"
PLAN_PLACEHOLDER_RELATIVE = "data_templates/n5_d5_private_replay_plan.placeholder.json"
ENVIRONMENT_LOCK_PLACEHOLDER_RELATIVE = (
    "data_templates/n5_d5_private_environment_lock.placeholder.json"
)
PHYSICAL_CONTRACT_PLACEHOLDER_RELATIVE = (
    "data_templates/n5_d5_private_physical_contract.placeholder.json"
)
PUBLIC_FOUNDATION_SOURCES = (
    "data_templates/n5_d5_minimum_bost_interface.schema.json",
    "data_templates/n5_d5_private_adapter_skeleton.py",
    "demo_t16_operator/n5_d5_adapter_protocol.py",
    "site_tools/run_n5_d5_minimum_interface_bridge.py",
    "site_tools/validate_n5_d5_minimum_interface_bridge.py",
    "site_tools/n5_d5_private_lab_readiness.py",
    "site_tools/test_n5_d5_private_lab_readiness.py",
    PLAN_SCHEMA_RELATIVE,
    PLAN_PLACEHOLDER_RELATIVE,
    ENVIRONMENT_LOCK_PLACEHOLDER_RELATIVE,
    PHYSICAL_CONTRACT_PLACEHOLDER_RELATIVE,
    "site_tools/n5_d5_private_replay_foundation.py",
    "site_tools/test_n5_d5_private_replay_foundation.py",
)
EXPECTED_PRIVATE_PAYLOAD_FILES = (
    "requests.jsonl",
    "responses.jsonl",
    "metrics.json",
    "finite_difference_rows.csv",
    "result.json",
    "private_diagnostic.png",
    "process_attestation.json",
    "challenge_reveal.json",
)
FORBIDDEN_RESULT_NAMES = frozenset(
    {
        "public_summary.json",
        "summary.md",
        "interface_bridge.png",
        "validation_report.json",
    }
)
CLAIM_KEYS = (
    "real_bost_interface",
    "physical_forward_correctness",
    "derivative_correctness",
    "reconstruction",
    "algorithm_superiority",
    "generalization",
    "publication",
)
PLACEHOLDER_MARKERS = (
    "REPLACE_ME",
    "REPLACE_WITH",
    "replace_private_plan_id",
)
ABSOLUTE_PATH_LITERAL = re.compile(r"(?:file://|/Users/|/home/|[A-Za-z]:\\\\)")
SECRET_LITERAL = re.compile(
    r"(?i)\b(password|passwd|token|api[_-]?key|secret)\b\s*[:=]\s*['\"][^'\"]+"
)
URL_CREDENTIAL = re.compile(r"(?i)https?://[^\s/@:]+:[^\s/@]+@")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ENVIRONMENT_LOCK_SCHEMA = "n5-d5-private-environment-lock-1.0"
PHYSICAL_CONTRACT_SCHEMA = "n5-d5-private-physical-contract-1.0"
RESULT_MANIFEST_SCHEMA = "n5-d5-private-result-manifest-1.0"


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64).reshape(-1))
    return sha256_bytes(array.tobytes(order="C"))


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )


def _lexical_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return Path(os.path.abspath(path))


def _resolved_under(path: Path, parent: Path) -> bool:
    resolved = path.resolve()
    root = parent.resolve()
    return resolved == root or root in resolved.parents


def _lexically_under(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _contains_symlink(path: Path, parent: Path) -> bool:
    try:
        relative = path.relative_to(parent)
    except ValueError:
        return True
    current = parent
    if current.is_symlink():
        return True
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _relative(repo_root: Path, path: Path) -> str:
    return path.relative_to(repo_root).as_posix()


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(
        self,
        code: str,
        passed: bool,
        detail: str,
        *,
        severity: str = "foundation_blocker",
    ) -> None:
        self.checks.append(
            {
                "code": code,
                "passed": bool(passed),
                "severity": severity,
                "detail": detail,
            }
        )

    def failed(self, severity: str) -> list[dict[str, Any]]:
        return [
            item
            for item in self.checks
            if item["severity"] == severity and not item["passed"]
        ]


def _public_attestation(
    repo_root: Path,
    audit: Audit,
    *,
    enforce_committed: bool,
) -> tuple[str, dict[str, str]]:
    hashes: dict[str, str] = {}
    for index, relative in enumerate(PUBLIC_FOUNDATION_SOURCES):
        path = repo_root / relative
        present = path.is_file() and not path.is_symlink()
        audit.add(
            f"PUBLIC_SOURCE_{index}_PRESENT",
            present,
            f"public L2 foundation source {index} is a regular file",
        )
        if not present:
            continue
        digest = sha256_file(path)
        hashes[relative] = digest
        if enforce_committed:
            tracked = (
                _git(
                    repo_root, "ls-files", "--error-unmatch", "--", relative
                ).returncode
                == 0
            )
            clean = (
                _git(repo_root, "diff", "--quiet", "HEAD", "--", relative).returncode
                == 0
            )
            committed = _git(repo_root, "show", f"HEAD:{relative}")
            commit_match = (
                committed.returncode == 0 and sha256_bytes(committed.stdout) == digest
            )
            audit.add(
                f"PUBLIC_SOURCE_{index}_TRACKED",
                tracked,
                f"public L2 foundation source {index} is Git tracked",
            )
            audit.add(
                f"PUBLIC_SOURCE_{index}_CLEAN",
                clean,
                f"public L2 foundation source {index} matches the working tree HEAD",
            )
            audit.add(
                f"PUBLIC_SOURCE_{index}_COMMIT_MATCH",
                commit_match,
                f"public L2 foundation source {index} bytes match HEAD",
            )
    commit = _git(repo_root, "rev-parse", "HEAD")
    commit_text = (
        commit.stdout.decode("utf-8").strip()
        if commit.returncode == 0
        else "UNAVAILABLE"
    )
    audit.add(
        "PUBLIC_FOUNDATION_COMMIT_AVAILABLE",
        len(commit_text) == 40,
        "a 40-character public L2 foundation commit is available",
    )
    return commit_text, hashes


def _private_file(
    repo_root: Path,
    private_root: Path,
    path: Path,
    label: str,
    audit: Audit,
) -> dict[str, Any] | None:
    inside = _lexically_under(path, private_root) and _resolved_under(
        path, private_root
    )
    audit.add(
        f"{label.upper()}_UNDER_PRIVATE_ROOT",
        inside,
        f"{label} stays under private_library lexically and after resolution",
    )
    if not inside:
        return None
    no_symlink = not _contains_symlink(path, private_root)
    audit.add(
        f"{label.upper()}_NO_SYMLINK_COMPONENT",
        no_symlink,
        f"{label} and its path components are not symlinks",
    )
    regular = path.is_file() and no_symlink
    audit.add(
        f"{label.upper()}_REGULAR_FILE",
        regular,
        f"{label} is a regular file",
    )
    if not regular:
        return None
    single_link = path.stat().st_nlink == 1
    audit.add(
        f"{label.upper()}_NOT_HARDLINKED",
        single_link,
        f"{label} has exactly one filesystem link",
    )
    relative = _relative(repo_root, path)
    ignored = _git(repo_root, "check-ignore", "-q", "--", relative).returncode == 0
    tracked = (
        _git(repo_root, "ls-files", "--error-unmatch", "--", relative).returncode == 0
    )
    audit.add(
        f"{label.upper()}_GIT_IGNORED",
        ignored,
        f"{label} is protected by a Git ignore rule",
    )
    audit.add(
        f"{label.upper()}_NOT_TRACKED",
        not tracked,
        f"{label} is absent from the public Git index",
    )
    return {
        "label": label,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _schema_errors(schema: dict[str, Any], value: object) -> list[str]:
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(value), key=lambda item: list(item.path))
    return [
        f"{'/'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
        for error in errors[:16]
    ]


def _bundle_walk(bundle_root: Path) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    errors: list[str] = []
    for current_text, directories, names in os.walk(bundle_root, followlinks=False):
        current = Path(current_text)
        kept: list[str] = []
        for name in sorted(directories):
            child = current / name
            if child.is_symlink():
                errors.append("symlink directory")
            else:
                kept.append(name)
        directories[:] = kept
        for name in sorted(names):
            child = current / name
            try:
                mode = child.lstat().st_mode
            except OSError:
                errors.append("unreadable file")
                continue
            if stat.S_ISLNK(mode):
                errors.append("symlink file")
            elif not stat.S_ISREG(mode):
                errors.append("non-regular file")
            else:
                files.append(child)
    return sorted(files), errors


def _environment_lock_ok(path: Path) -> tuple[bool, str]:
    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False, "environment lock is not valid UTF-8 JSON"
    if not isinstance(value, dict):
        return False, "environment lock is not an object"
    if value.get("schema_version") != ENVIRONMENT_LOCK_SCHEMA:
        return False, "environment lock schema drifted"
    if value.get("lock_reviewed") is not True:
        return False, "environment lock is not marked reviewed"
    if value.get("contains_credentials") is not False:
        return False, "environment lock does not close credential content"
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return False, "environment lock has no content-addressed artifacts"
    for item in artifacts:
        if not isinstance(item, dict) or set(item) != {"name", "version", "sha256"}:
            return False, "environment lock artifact fields drifted"
        if not isinstance(item["name"], str) or not item["name"].strip():
            return False, "environment lock artifact name is empty"
        if not isinstance(item["version"], str) or not item["version"].strip():
            return False, "environment lock artifact version is empty"
        if not isinstance(item["sha256"], str) or not SHA256_PATTERN.fullmatch(
            item["sha256"]
        ):
            return False, "environment lock artifact digest is invalid"
    if (
        ABSOLUTE_PATH_LITERAL.search(text)
        or SECRET_LITERAL.search(text)
        or URL_CREDENTIAL.search(text)
        or any(marker in text for marker in PLACEHOLDER_MARKERS)
    ):
        return False, "environment lock contains a path, secret, or placeholder marker"
    return True, "environment lock is structurally reviewed and content-addressed"


def _physical_contract_ok(
    path: Path,
    capability: dict[str, Any],
) -> tuple[bool, str]:
    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False, "physical contract is not valid UTF-8 JSON"
    required = {
        "schema_version",
        "reviewed",
        "contains_private_identifiers",
        "parameterization",
        "tensor_shape",
        "spatial_spacing",
        "axis_order",
        "field_units",
        "observation_units",
        "coordinate_handedness",
        "geometry_sha256",
        "calibration_sha256",
        "optical_wavelength_policy",
        "sampling_rule",
        "interpolation_rule",
        "boundary_rule",
        "termination_rule",
        "backend_dtype",
        "wire_dtype",
        "conversion_policy",
        "dynamic_sampling",
        "dynamic_ray_sample_ledger_required",
        "native_direct_residual",
        "direct_residual_semantics",
        "decoder_checkpoint_sha256",
        "noise_floor",
    }
    if not isinstance(value, dict) or set(value) != required:
        return False, "physical contract fields drifted"
    if value["schema_version"] != PHYSICAL_CONTRACT_SCHEMA:
        return False, "physical contract schema drifted"
    if (
        value["reviewed"] is not True
        or value["contains_private_identifiers"] is not False
    ):
        return False, "physical contract is not reviewed or anonymized"
    strings = (
        "parameterization",
        "axis_order",
        "field_units",
        "observation_units",
        "optical_wavelength_policy",
        "sampling_rule",
        "interpolation_rule",
        "boundary_rule",
        "termination_rule",
        "conversion_policy",
    )
    if any(
        not isinstance(value[key], str) or not value[key].strip() for key in strings
    ):
        return False, "physical contract contains an empty semantic field"
    shape = value["tensor_shape"]
    if (
        not isinstance(shape, list)
        or not shape
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 1
            for item in shape
        )
    ):
        return False, "physical contract tensor shape is invalid"
    spacing = value["spatial_spacing"]
    if spacing is not None and (
        not isinstance(spacing, list)
        or len(spacing) != len(shape)
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) <= 0.0
            for item in spacing
        )
    ):
        return False, "physical contract spacing is invalid"
    if value["coordinate_handedness"] not in {"right_handed", "left_handed"}:
        return False, "physical contract coordinate handedness is invalid"
    for key in ("geometry_sha256", "calibration_sha256"):
        if not isinstance(value[key], str) or not SHA256_PATTERN.fullmatch(value[key]):
            return False, f"physical contract {key} is invalid"
    checkpoint = value["decoder_checkpoint_sha256"]
    if checkpoint is not None and (
        not isinstance(checkpoint, str) or not SHA256_PATTERN.fullmatch(checkpoint)
    ):
        return False, "physical contract decoder checkpoint digest is invalid"
    if value["backend_dtype"] not in {"float32", "float64", "mixed_precision"}:
        return False, "physical contract backend dtype is invalid"
    if value["wire_dtype"] != "float64":
        return (
            False,
            "physical contract wire dtype must record the current float64 wire",
        )
    if not isinstance(value["dynamic_sampling"], bool):
        return False, "physical contract dynamic_sampling flag is invalid"
    if value["dynamic_ray_sample_ledger_required"] is not True:
        return False, "physical contract does not require a dynamic ray-sample ledger"
    expected_direct = bool(capability["direct_residual_supported"])
    expected_semantics = (
        "native_same_sample_residual" if expected_direct else "unavailable"
    )
    if (
        value["native_direct_residual"] is not expected_direct
        or value["direct_residual_semantics"] != expected_semantics
    ):
        return False, "physical contract direct-residual capability drifted"
    noise = value["noise_floor"]
    if (
        not isinstance(noise, dict)
        or set(noise) != {"units", "absolute_std"}
        or not isinstance(noise["units"], str)
        or not noise["units"].strip()
        or isinstance(noise["absolute_std"], bool)
        or not isinstance(noise["absolute_std"], (int, float))
        or not math.isfinite(float(noise["absolute_std"]))
        or float(noise["absolute_std"]) <= 0.0
    ):
        return False, "physical contract noise floor is invalid"
    if (
        ABSOLUTE_PATH_LITERAL.search(text)
        or SECRET_LITERAL.search(text)
        or URL_CREDENTIAL.search(text)
        or any(marker in text for marker in PLACEHOLDER_MARKERS)
    ):
        return False, "physical contract contains a path, secret, or placeholder marker"
    return True, "physical metadata and callable capability are structurally reviewed"


def _config_capability_and_budget(
    path: Path,
    capability: dict[str, Any],
    challenge: dict[str, Any],
    audit: Audit,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        audit.add("CONFIG_JSON_OBJECT", False, "private config is not valid UTF-8 JSON")
        return None, None
    if not isinstance(value, dict):
        audit.add("CONFIG_JSON_OBJECT", False, "private config is not an object")
        return None, None
    audit.add("CONFIG_JSON_OBJECT", True, "private config is a JSON object")

    paths = value.get("paths")
    probe_plan = value.get("probe_plan")
    cost_contract = value.get("cost_contract")
    if (
        not isinstance(paths, list)
        or not paths
        or not isinstance(probe_plan, dict)
        or not isinstance(cost_contract, dict)
    ):
        audit.add(
            "L1_CAPABILITY_PROFILE_MATCHES",
            False,
            "config paths, probe plan, or cost contract is missing",
        )
        return value, None

    roles = [str(item.get("role")) for item in paths if isinstance(item, dict)]
    direct_present = roles.count("direct_residual") == 1
    expected_direct = bool(capability["direct_residual_supported"])
    expected_roles = {"curved", "straight"}
    if expected_direct:
        expected_roles.add("direct_residual")
    exact_roles = len(roles) == len(expected_roles) and set(roles) == expected_roles
    semantic_match = capability["direct_residual_semantics"] == (
        "native_same_sample_residual" if expected_direct else "unavailable"
    )
    operations_callable = all(
        isinstance(item, dict)
        and isinstance(item.get("operation_support"), dict)
        and all(
            item["operation_support"].get(name) is True
            for name in ("forward", "jvp", "vjp")
        )
        for item in paths
    )
    capability_match = (
        direct_present is expected_direct
        and exact_roles
        and semantic_match
        and operations_callable
        and capability["arbitrary_runtime_directions_supported"] is True
        and capability["precomputed_probe_arrays_are_sufficient"] is False
    )
    audit.add(
        "L1_CAPABILITY_PROFILE_MATCHES",
        capability_match,
        (
            "L1 config exposes exactly the declared curved/straight/direct capability"
            if capability_match
            else "L1 config path roles or callable capability conflicts with the L2 profile"
        ),
    )

    h_values = probe_plan.get("h_values")
    base_tangent_count = probe_plan.get("tangent_count")
    base_cotangent_count = probe_plan.get("cotangent_count")
    private_h_intervals = challenge["private_h_intervals"]
    intervals_ordered = all(
        isinstance(interval, list)
        and len(interval) == 2
        and float(interval[0]) < float(interval[1])
        for interval in private_h_intervals
    )
    audit.add(
        "PRIVATE_H_INTERVALS_STRICTLY_ORDERED",
        intervals_ordered,
        "each private finite-difference interval has a positive lower bound below its upper bound",
    )
    counts_valid = (
        isinstance(h_values, list)
        and bool(h_values)
        and isinstance(base_tangent_count, int)
        and not isinstance(base_tangent_count, bool)
        and base_tangent_count >= 1
        and isinstance(base_cotangent_count, int)
        and not isinstance(base_cotangent_count, bool)
        and base_cotangent_count >= 1
        and intervals_ordered
    )
    if not counts_valid:
        audit.add(
            "PRIVATE_PROBE_SCHEDULE_VALID",
            False,
            "probe counts or finite-difference schedules are invalid",
        )
        return value, None
    audit.add(
        "PRIVATE_PROBE_SCHEDULE_VALID",
        True,
        "base and private probe schedules have valid counts and ordered h intervals",
    )

    path_count = len(paths)
    base_h_count = len(h_values)
    expected_forward = path_count * (2 + 2 * base_tangent_count * base_h_count)
    expected_jvp = path_count * base_tangent_count
    expected_vjp = path_count * base_cotangent_count
    primary = 2 + expected_forward + expected_jvp + expected_vjp
    private_tangent_count = int(challenge["validator_extra_tangent_count"])
    private_cotangent_count = int(challenge["validator_extra_cotangent_count"])
    private_h_count = len(private_h_intervals)
    validator_private = path_count * (
        private_tangent_count
        + private_cotangent_count
        + 2 * private_tangent_count * private_h_count
    )
    computed = {
        "describe_only_requests": 2,
        "primary_requests": primary,
        "validator_base_replay_requests": primary,
        "validator_private_probe_requests": validator_private,
        "total_requests": 2 + primary + primary + validator_private,
        "separate_authorization_tokens_required": True,
        "path_count": path_count,
        "base_tangent_count": base_tangent_count,
        "base_cotangent_count": base_cotangent_count,
        "base_h_count": base_h_count,
        "private_tangent_count": private_tangent_count,
        "private_cotangent_count": private_cotangent_count,
        "private_h_count": private_h_count,
    }
    # The caller compares this result with the L2 authorization budget.  This
    # local check independently binds the inherited L1 cost contract to the
    # same operation count.
    cost_match = (
        cost_contract.get("describe_calls") == 2
        and cost_contract.get("forward_api_calls") == expected_forward
        and cost_contract.get("jvp_api_calls") == expected_jvp
        and cost_contract.get("vjp_api_calls") == expected_vjp
    )
    audit.add(
        "L1_COST_CONTRACT_MATCHES_CAPABILITY",
        cost_match,
        "L1 cost contract matches the declared path and base-probe schedule",
    )
    return value, computed


def _physical_contract_matches_config(
    physical: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    field = config.get("field")
    observation = config.get("observation")
    if not isinstance(field, dict) or not isinstance(observation, dict):
        return False
    shape = physical.get("tensor_shape")
    if not isinstance(shape, list):
        return False
    input_dimension = field.get("input_dimension")
    if not isinstance(input_dimension, int) or isinstance(input_dimension, bool):
        return False
    return (
        math.prod(shape) == input_dimension
        and physical.get("parameterization") == field.get("parameterization")
        and physical.get("axis_order") == field.get("axis_order")
        and physical.get("field_units") == field.get("units")
        and physical.get("observation_units") == observation.get("units")
        and physical.get("wire_dtype") == field.get("dtype") == observation.get("dtype")
    )


def _verify_l1_public_attestation(
    repo_root: Path,
    report: dict[str, Any],
    audit: Audit,
) -> None:
    commit = str(report.get("protocol_commit", ""))
    hashes = report.get("public_protocol_source_hashes")
    audit.add(
        "L1_PUBLIC_COMMIT_AVAILABLE",
        len(commit) == 40,
        "L1 report carries a 40-character public commit",
    )
    audit.add(
        "L1_PUBLIC_SOURCE_HASH_MAP",
        isinstance(hashes, dict) and bool(hashes),
        "L1 report carries public protocol source hashes",
    )
    if len(commit) != 40 or not isinstance(hashes, dict):
        return
    for index, (relative, expected) in enumerate(sorted(hashes.items())):
        path = repo_root / relative
        current_match = path.is_file() and sha256_file(path) == expected
        committed = _git(repo_root, "show", f"{commit}:{relative}")
        commit_match = (
            committed.returncode == 0 and sha256_bytes(committed.stdout) == expected
        )
        audit.add(
            f"L1_PUBLIC_SOURCE_{index}_CURRENT_MATCH",
            current_match,
            f"L1 public source {index} still matches its recorded hash",
        )
        audit.add(
            f"L1_PUBLIC_SOURCE_{index}_COMMIT_MATCH",
            commit_match,
            f"L1 public source {index} matches the recorded commit",
        )


def build_foundation_report(
    plan_path: Path,
    *,
    repo_root: Path = ROOT,
    enforce_public_committed: bool = True,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    private_root = (repo_root / "private_library").resolve()
    plan_path = _lexical_path(repo_root, plan_path)
    audit = Audit()
    public_commit, public_hashes = _public_attestation(
        repo_root,
        audit,
        enforce_committed=enforce_public_committed,
    )
    plan_entry = _private_file(
        repo_root,
        private_root,
        plan_path,
        "private_plan",
        audit,
    )
    plan: dict[str, Any] | None = None
    if plan_entry is not None:
        try:
            value = json.loads(plan_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                plan = value
            else:
                audit.add("PLAN_JSON_OBJECT", False, "private plan is not an object")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            audit.add("PLAN_JSON_PARSE", False, "private plan is not valid UTF-8 JSON")
    if plan is None:
        return _final_report(
            audit,
            public_commit=public_commit,
            public_hashes=public_hashes,
            plan=None,
            private_inventory=[],
            bundle_digest=None,
        )

    schema = json.loads((repo_root / PLAN_SCHEMA_RELATIVE).read_text(encoding="utf-8"))
    errors = _schema_errors(schema, plan)
    audit.add(
        "PLAN_SCHEMA_VALID",
        not errors,
        "private replay plan satisfies the L2 JSON Schema"
        if not errors
        else "; ".join(errors),
    )
    serialized = canonical_json(plan)
    audit.add(
        "PLAN_PLACEHOLDERS_REPLACED",
        not any(marker in serialized for marker in PLACEHOLDER_MARKERS),
        "private replay plan contains no placeholder marker",
    )
    audit.add(
        "PLAN_NO_ABSOLUTE_PATH_OR_SECRET",
        ABSOLUTE_PATH_LITERAL.search(serialized) is None
        and SECRET_LITERAL.search(serialized) is None
        and URL_CREDENTIAL.search(serialized) is None,
        "private replay plan contains no local absolute path or obvious secret",
    )
    if errors:
        return _final_report(
            audit,
            public_commit=public_commit,
            public_hashes=public_hashes,
            plan=plan,
            private_inventory=[],
            bundle_digest=None,
        )

    audit.add(
        "ALL_SCIENTIFIC_CLAIMS_CLOSED",
        set(plan["claim_authorizations"]) == set(CLAIM_KEYS)
        and not any(plan["claim_authorizations"].values()),
        "all seven scientific claim authorizations remain false",
    )
    l1_path = _lexical_path(repo_root, plan["l1_report"]["relative_path"])
    l1_entry = _private_file(
        repo_root,
        private_root,
        l1_path,
        "l1_report",
        audit,
    )
    l1_report: dict[str, Any] | None = None
    if l1_entry is not None:
        audit.add(
            "L1_REPORT_HASH_MATCHES",
            l1_entry["sha256"] == plan["l1_report"]["sha256"],
            "L1 report bytes match the private replay plan",
        )
        try:
            value = json.loads(l1_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                l1_report = value
            else:
                audit.add("L1_REPORT_OBJECT", False, "L1 report is not an object")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            audit.add("L1_REPORT_PARSE", False, "L1 report is not valid JSON")
    if l1_report is not None:
        audit.add(
            "L1_STATIC_READY_FORMAL_LOCKED",
            l1_report.get("status")
            == "STATIC_PRIVATE_INTAKE_READY_FORMAL_REPLAY_LOCKED"
            and l1_report.get("ready_for_private_describe_probe") is True
            and l1_report.get("formal_53_call_replay_authorized") is False
            and int(l1_report.get("blocker_count", -1)) == 0,
            "L1 report is static-ready while formal replay remains locked",
        )
        audit.add(
            "L1_CLAIMS_CLOSED",
            isinstance(l1_report.get("claim_authorizations"), dict)
            and not any(l1_report["claim_authorizations"].values()),
            "L1 report keeps every scientific claim closed",
        )
        _verify_l1_public_attestation(repo_root, l1_report, audit)

    bundle_root = _lexical_path(repo_root, plan["private_bundle"]["root"])
    bundle_inside = _lexically_under(bundle_root, private_root) and _resolved_under(
        bundle_root, private_root
    )
    audit.add(
        "BUNDLE_ROOT_UNDER_PRIVATE_ROOT",
        bundle_inside,
        "private bundle root stays under private_library",
    )
    bundle_regular = (
        bundle_inside
        and bundle_root.is_dir()
        and not _contains_symlink(bundle_root, private_root)
    )
    audit.add(
        "BUNDLE_ROOT_REGULAR_DIRECTORY",
        bundle_regular,
        "private bundle root is a non-symlink directory",
    )
    control_disjoint = (
        bundle_regular
        and not _lexically_under(plan_path, bundle_root)
        and not _lexically_under(l1_path, bundle_root)
    )
    audit.add(
        "CONTROL_FILES_OUTSIDE_IMMUTABLE_BUNDLE",
        control_disjoint,
        "plan and L1 report stay outside the immutable input bundle",
    )

    actual_files: list[Path] = []
    walk_errors: list[str] = []
    if bundle_regular:
        actual_files, walk_errors = _bundle_walk(bundle_root)
    audit.add(
        "BUNDLE_ONLY_REGULAR_NON_SYMLINK_FILES",
        not walk_errors,
        "private bundle contains only regular non-symlink files",
    )
    inventory = plan["private_bundle"]["inventory"]
    inventory_paths = [
        _lexical_path(repo_root, item["relative_path"]) for item in inventory
    ]
    unique_inventory = len(inventory_paths) == len(set(inventory_paths))
    audit.add(
        "BUNDLE_INVENTORY_PATHS_UNIQUE",
        unique_inventory,
        "private bundle inventory paths are unique",
    )
    inventory_inside = all(
        _lexically_under(path, bundle_root) and _resolved_under(path, bundle_root)
        for path in inventory_paths
    )
    audit.add(
        "BUNDLE_INVENTORY_STAYS_UNDER_ROOT",
        inventory_inside,
        "every inventory path stays under the immutable bundle root",
    )
    exact_inventory = set(inventory_paths) == set(actual_files)
    audit.add(
        "BUNDLE_CLOSED_WORLD_EXACT_INVENTORY",
        unique_inventory and inventory_inside and exact_inventory,
        "actual private bundle files exactly equal the declared inventory",
    )
    private_inventory: list[dict[str, Any]] = []
    role_counts = Counter(str(item["role"]) for item in inventory)
    role_seen: Counter[str] = Counter()
    for index, (item, path) in enumerate(zip(inventory, inventory_paths, strict=True)):
        entry = _private_file(
            repo_root,
            private_root,
            path,
            f"bundle_input_{index}",
            audit,
        )
        if entry is None:
            continue
        hash_match = entry["sha256"] == item["sha256"]
        size_match = entry["bytes"] == int(item["bytes"])
        audit.add(
            f"BUNDLE_INPUT_{index}_HASH_MATCHES",
            hash_match,
            f"private bundle input {index} matches its SHA-256",
        )
        audit.add(
            f"BUNDLE_INPUT_{index}_SIZE_MATCHES",
            size_match,
            f"private bundle input {index} matches its byte count",
        )
        role_seen[str(item["role"])] += 1
        private_inventory.append(
            {
                "label": f"{item['role']}:{role_seen[str(item['role'])]}",
                "role": item["role"],
                "sha256": entry["sha256"],
                "bytes": entry["bytes"],
            }
        )
    audit.add(
        "BUNDLE_REQUIRED_ROLES_PRESENT",
        role_counts["config"] == 1
        and role_counts["adapter_source"] >= 1
        and role_counts["base_input"] == 1
        and role_counts["environment_lock"] == 1
        and role_counts["physical_contract"] == 1,
        (
            "bundle has one config/base/environment/physical contract and at least "
            "one adapter source"
        ),
    )
    if l1_report is not None and isinstance(l1_report.get("private_inventory"), list):
        l1_counter = Counter(
            (str(item.get("sha256")), int(item.get("bytes", -1)))
            for item in l1_report["private_inventory"]
        )
        plan_counter = Counter(
            (str(item["sha256"]), int(item["bytes"]))
            for item in inventory
            if item["role"] in {"config", "adapter_source", "base_input"}
        )
        audit.add(
            "L1_PRIVATE_INPUTS_BOUND_TO_BUNDLE",
            l1_counter == plan_counter,
            "L1 config/source/base digests exactly match the L2 private bundle",
        )

    environment_entries = [
        path
        for item, path in zip(inventory, inventory_paths, strict=True)
        if item["role"] == "environment_lock"
    ]
    environment_ok = False
    environment_detail = "environment lock role is not unique"
    if len(environment_entries) == 1 and environment_entries[0].is_file():
        environment_ok, environment_detail = _environment_lock_ok(
            environment_entries[0]
        )
    audit.add(
        "ENVIRONMENT_LOCK_STRUCTURALLY_REVIEWED",
        environment_ok,
        environment_detail,
    )

    config_entries = [
        path
        for item, path in zip(inventory, inventory_paths, strict=True)
        if item["role"] == "config"
    ]
    config_value: dict[str, Any] | None = None
    computed_budget: dict[str, Any] | None = None
    if len(config_entries) == 1 and config_entries[0].is_file():
        config_value, computed_budget = _config_capability_and_budget(
            config_entries[0],
            plan["capability_profile"],
            plan["challenge_policy"],
            audit,
        )
    else:
        audit.add(
            "L1_CAPABILITY_PROFILE_MATCHES",
            False,
            "config role is not unique",
        )
    budget_keys = (
        "describe_only_requests",
        "primary_requests",
        "validator_base_replay_requests",
        "validator_private_probe_requests",
        "total_requests",
        "separate_authorization_tokens_required",
    )
    budget_exact = computed_budget is not None and plan["authorization_budget"] == {
        key: computed_budget[key] for key in budget_keys
    }
    audit.add(
        "AUTHORIZATION_BUDGET_EXACT",
        budget_exact,
        (
            "declared authorization budget exactly counts describe, base replays, and private probes"
            if budget_exact
            else "declared authorization budget does not equal the callable-derived request count"
        ),
    )

    physical_entries = [
        path
        for item, path in zip(inventory, inventory_paths, strict=True)
        if item["role"] == "physical_contract"
    ]
    physical_contract_ok = False
    physical_contract_detail = "physical contract role is not unique"
    physical_contract_value: dict[str, Any] | None = None
    physical_contract_digest: str | None = None
    if len(physical_entries) == 1 and physical_entries[0].is_file():
        physical_contract_ok, physical_contract_detail = _physical_contract_ok(
            physical_entries[0], plan["capability_profile"]
        )
        physical_contract_digest = sha256_file(physical_entries[0])
        if physical_contract_ok:
            parsed_physical = json.loads(
                physical_entries[0].read_text(encoding="utf-8")
            )
            if isinstance(parsed_physical, dict):
                physical_contract_value = parsed_physical
    audit.add(
        "PHYSICAL_CONTRACT_STRUCTURALLY_REVIEWED",
        physical_contract_ok,
        physical_contract_detail,
    )
    physical_config_match = (
        physical_contract_value is not None
        and config_value is not None
        and _physical_contract_matches_config(physical_contract_value, config_value)
    )
    audit.add(
        "PHYSICAL_CONTRACT_MATCHES_L1_CONFIG",
        physical_config_match,
        "physical shape, units, axis order, parameterization, and wire dtype match L1",
    )
    audit.add(
        "DYNAMIC_COST_LEDGER_CONTRACT_FROZEN",
        physical_contract_value is not None
        and physical_contract_value["dynamic_ray_sample_ledger_required"] is True
        and plan["capability_profile"]["dynamic_ray_sample_ledger_required"] is True
        and plan["capability_profile"]["fixed_zero_ray_sample_ledger_forbidden"]
        is True,
        "dynamic ray/sample accounting is required and a fixed zero ledger is forbidden",
    )

    output = _lexical_path(repo_root, plan["output_policy"]["result_directory"])
    output_inside = (
        _lexically_under(output, private_root)
        and _resolved_under(output.parent, private_root)
        and not _contains_symlink(output.parent, private_root)
    )
    audit.add(
        "RESULT_DIRECTORY_UNDER_PRIVATE_ROOT",
        output_inside,
        "future result directory stays under private_library",
    )
    output_disjoint = (
        output_inside
        and not _lexically_under(output, bundle_root)
        and not _lexically_under(bundle_root, output)
    )
    audit.add(
        "RESULT_DIRECTORY_DISJOINT_FROM_INPUT_BUNDLE",
        output_disjoint,
        "future result directory is disjoint from immutable inputs",
    )
    audit.add(
        "RESULT_DIRECTORY_ABSENT",
        not output.exists(),
        "future result directory does not already exist",
    )
    expected_names = tuple(plan["output_policy"]["expected_payload_files"])
    audit.add(
        "RESULT_PAYLOAD_ALLOWLIST_FROZEN",
        expected_names == EXPECTED_PRIVATE_PAYLOAD_FILES,
        "future private result payload matches the hard-coded allowlist",
    )
    audit.add(
        "PUBLIC_SUMMARY_HARD_WRITE_POLICY_CLOSED",
        plan["output_policy"]["public_summary_write_permitted"] is False
        and plan["output_policy"]["raw_trace_publication_permitted"] is False
        and not (set(expected_names) & FORBIDDEN_RESULT_NAMES),
        "private result policy cannot write a public summary or public plot",
    )

    process = plan["process_policy"]
    network_configured = process["network_isolation_backend"] != "not_configured"
    network_profile_present = role_counts["network_profile"] == 1
    audit.add(
        "NETWORK_ISOLATION_BACKEND_CONFIGURED",
        network_configured and network_profile_present,
        "an OS isolation backend and one private profile are declared",
        severity="execution_blocker",
    )
    cost_configured = (
        process["independent_cost_instrumentation"] != "not_configured"
        and role_counts["cost_instrumentation"] == 1
    )
    audit.add(
        "INDEPENDENT_COST_INSTRUMENTATION_CONFIGURED",
        cost_configured,
        "an independent cost mechanism and one private instrumentation file are declared",
        severity="execution_blocker",
    )
    physical = plan["physical_review"]
    physical_reviewed = (
        physical["tolerances_reviewed"] is True
        and isinstance(physical["reviewer_pseudonym"], str)
        and physical_contract_digest is not None
        and physical["review_digest_sha256"] == physical_contract_digest
    )
    audit.add(
        "PHYSICAL_TOLERANCES_REVIEWED",
        physical_reviewed,
        "laboratory precision and noise-floor tolerances have a private review digest",
        severity="execution_blocker",
    )
    audit.add(
        "IN_MEMORY_PRIVATE_PROBE_GENERATOR_AVAILABLE",
        True,
        "CSPRNG probes can be generated after attestation without pre-run persistence",
    )
    audit.add(
        "L2B_ISOLATED_DESCRIBE_RUNNER_IMPLEMENTED",
        False,
        "L2-A does not execute the adapter; an isolated describe runner is still required",
        severity="warning",
    )
    audit.add(
        "RUNTIME_NETWORK_ISOLATION_OBSERVED",
        False,
        "a declared isolation profile is not evidence that a subprocess was contained",
        severity="warning",
    )
    audit.add(
        "RUNTIME_COST_INSTRUMENTATION_OBSERVED",
        False,
        "a declared cost mechanism is not evidence that hidden calls were measured",
        severity="warning",
    )
    audit.add(
        "ARBITRARY_RUNTIME_DIRECTIONS_OBSERVED",
        False,
        "static support flags do not prove the adapter accepts fresh runtime v and q arrays",
        severity="warning",
    )
    audit.add(
        "PHYSICAL_RENDERER_CORRECTNESS_REVIEWED",
        False,
        "content addressing and source review cannot prove the optical renderer is physical",
        severity="warning",
    )

    bundle_digest = None
    if private_inventory:
        digest_rows = sorted(
            (
                {
                    "role": item["role"],
                    "sha256": item["sha256"],
                    "bytes": item["bytes"],
                }
                for item in private_inventory
            ),
            key=lambda item: (item["role"], item["sha256"], item["bytes"]),
        )
        bundle_digest = sha256_bytes(canonical_json(digest_rows).encode("utf-8"))
    return _final_report(
        audit,
        public_commit=public_commit,
        public_hashes=public_hashes,
        plan=plan,
        private_inventory=private_inventory,
        bundle_digest=bundle_digest,
        computed_budget=computed_budget,
    )


def _final_report(
    audit: Audit,
    *,
    public_commit: str,
    public_hashes: dict[str, str],
    plan: dict[str, Any] | None,
    private_inventory: list[dict[str, Any]],
    bundle_digest: str | None,
    computed_budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    foundation_blockers = audit.failed("foundation_blocker")
    execution_blockers = audit.failed("execution_blocker")
    warnings = audit.failed("warning")
    foundation_ready = not foundation_blockers
    plan_digest = None
    if plan is not None:
        plan_digest = sha256_bytes(canonical_json(plan).encode("utf-8"))
    capability_mismatch = "L1_CAPABILITY_PROFILE_MATCHES" in {
        item["code"] for item in foundation_blockers
    }
    needs_dual_l1 = (
        capability_mismatch
        and isinstance(plan, dict)
        and isinstance(plan.get("capability_profile"), dict)
        and plan["capability_profile"].get("direct_residual_supported") is False
    )
    return {
        "schema_version": "n5-d5-private-replay-foundation-report-1.0",
        "status": (
            "PRIVATE_REPLAY_FOUNDATION_READY_EXECUTION_LOCKED"
            if foundation_ready
            else "PRIVATE_REPLAY_FOUNDATION_BLOCKED"
        ),
        "evidence_scope": (
            "PUBLIC_PRIVATE_CONTENT_ATTESTATION_INPUT_CLOSED_WORLD_OUTPUT_POLICY_"
            "AND_IN_MEMORY_PROBE_MECHANISM_NO_ADAPTER_EXECUTION"
        ),
        "foundation_ready": foundation_ready,
        "execution_policy_declared": foundation_ready and not execution_blockers,
        "private_describe_probe_authorized": False,
        "formal_replay_authorized": False,
        "formal_53_call_replay_authorized": False,
        "adapter_executed": False,
        "public_foundation_commit": public_commit,
        "public_foundation_source_hashes": public_hashes,
        "private_plan_sha256": plan_digest,
        "private_bundle_sha256": bundle_digest,
        "private_inventory": private_inventory,
        "declared_capability_profile": (
            plan.get("capability_profile") if isinstance(plan, dict) else None
        ),
        "authorization_budget": {
            "declared": (
                plan.get("authorization_budget") if isinstance(plan, dict) else None
            ),
            "computed": computed_budget,
        },
        "result_payload_allowlist": list(EXPECTED_PRIVATE_PAYLOAD_FILES),
        "checks": audit.checks,
        "foundation_blocker_codes": [item["code"] for item in foundation_blockers],
        "execution_blocker_codes": [item["code"] for item in execution_blockers],
        "warning_codes": [item["code"] for item in warnings],
        "claim_authorizations": {key: False for key in CLAIM_KEYS},
        "next_action": (
            "BUILD_DUAL_PATH_L1_V2"
            if needs_dual_l1
            else (
                "IMPLEMENT_L2B_ISOLATED_DESCRIBE_RUNNER"
                if foundation_ready
                else "FIX_PRIVATE_REPLAY_FOUNDATION_BLOCKERS"
            )
        ),
        "limitations": [
            "No private adapter is imported, launched, or queried.",
            "The private bundle digest is content-addressed but not externally signed.",
            "Declared OS isolation and cost instrumentation are not runtime observations.",
            "Static operation flags do not prove arbitrary fresh runtime directions are accepted.",
            "CSPRNG probe code reduces public lookup risk only after an L2-B runner uses it in memory.",
            "No BOST physics, derivative, reconstruction, superiority, generalization, or publication claim is authorized.",
        ],
    }


@dataclass(frozen=True)
class PrivateProbeBank:
    seed: bytes
    commitment_sha256: str
    context_sha256: str
    tangents: np.ndarray
    cotangents: np.ndarray
    h_values: np.ndarray


def _orthonormal_vectors(
    rng: np.random.Generator,
    *,
    count: int,
    dimension: int,
) -> np.ndarray:
    if count < 1 or count > dimension:
        raise ValueError("probe count must be between one and its vector dimension")
    matrix = rng.normal(size=(dimension, count))
    q, _ = np.linalg.qr(matrix, mode="reduced")
    vectors = np.ascontiguousarray(q.T, dtype=np.float64)
    if not bool(np.all(np.isfinite(vectors))):
        raise ValueError("generated probe vector is non-finite")
    return vectors


def generate_private_probe_bank(
    *,
    input_dimension: int,
    output_dimension: int,
    tangent_count: int,
    cotangent_count: int,
    h_intervals: list[list[float]] | tuple[tuple[float, float], ...],
    entropy_bytes: int = 32,
    entropy_source: Callable[[int], bytes] = secrets.token_bytes,
) -> PrivateProbeBank:
    if entropy_bytes < 32 or entropy_bytes > 64:
        raise ValueError("private probe entropy must be between 32 and 64 bytes")
    seed = entropy_source(entropy_bytes)
    if not isinstance(seed, bytes) or len(seed) != entropy_bytes:
        raise ValueError("entropy source returned the wrong number of bytes")
    normalized_intervals: list[tuple[float, float]] = []
    for interval in h_intervals:
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError("each private h interval must contain two bounds")
        low, high = float(interval[0]), float(interval[1])
        if not (
            math.isfinite(low) and math.isfinite(high) and low > 0.0 and low < high
        ):
            raise ValueError(
                "private h interval bounds must be finite, positive, and ordered"
            )
        normalized_intervals.append((low, high))
    if not normalized_intervals:
        raise ValueError("at least one private h interval is required")
    context = {
        "schema": "n5-d5-private-probe-context-1.0",
        "input_dimension": int(input_dimension),
        "output_dimension": int(output_dimension),
        "tangent_count": int(tangent_count),
        "cotangent_count": int(cotangent_count),
        "h_intervals": [list(interval) for interval in normalized_intervals],
    }
    context_bytes = canonical_json(context).encode("utf-8")
    context_sha256 = sha256_bytes(context_bytes)
    commitment = sha256_bytes(b"n5-d5-private-probe|" + seed + context_bytes)
    tangent_seed = hashlib.sha256(seed + b"|tangent|" + context_bytes).digest()
    cotangent_seed = hashlib.sha256(seed + b"|cotangent|" + context_bytes).digest()
    h_seed = hashlib.sha256(seed + b"|finite-difference-h|" + context_bytes).digest()
    tangent_rng = np.random.default_rng(
        int.from_bytes(tangent_seed[:16], "big", signed=False)
    )
    cotangent_rng = np.random.default_rng(
        int.from_bytes(cotangent_seed[:16], "big", signed=False)
    )
    h_rng = np.random.default_rng(int.from_bytes(h_seed[:16], "big", signed=False))
    tangents = _orthonormal_vectors(
        tangent_rng,
        count=int(tangent_count),
        dimension=int(input_dimension),
    )
    cotangents = _orthonormal_vectors(
        cotangent_rng,
        count=int(cotangent_count),
        dimension=int(output_dimension),
    )
    h_values = np.ascontiguousarray(
        [
            math.exp(h_rng.uniform(math.log(low), math.log(high)))
            for low, high in normalized_intervals
        ],
        dtype=np.float64,
    )
    return PrivateProbeBank(
        seed=seed,
        commitment_sha256=commitment,
        context_sha256=context_sha256,
        tangents=tangents,
        cotangents=cotangents,
        h_values=h_values,
    )


def private_probe_commitment(bank: PrivateProbeBank) -> dict[str, Any]:
    return {
        "schema": "n5-d5-private-probe-commitment-1.0",
        "commitment_sha256": bank.commitment_sha256,
        "context_sha256": bank.context_sha256,
        "tangent_count": int(bank.tangents.shape[0]),
        "cotangent_count": int(bank.cotangents.shape[0]),
        "h_count": int(bank.h_values.shape[0]),
        "seed_persisted": False,
        "vectors_persisted": False,
        "h_values_persisted": False,
    }


def private_probe_reveal(bank: PrivateProbeBank) -> dict[str, Any]:
    return {
        "schema": "n5-d5-private-probe-reveal-1.0",
        "commitment_sha256": bank.commitment_sha256,
        "context_sha256": bank.context_sha256,
        "seed_hex": bank.seed.hex(),
        "tangent_sha256": [array_sha256(value) for value in bank.tangents],
        "cotangent_sha256": [array_sha256(value) for value in bank.cotangents],
        "h_values": [float(value) for value in bank.h_values],
        "h_values_sha256": array_sha256(bank.h_values),
        "write_policy": "private_result_after_adapter_exit_only",
    }


def private_result_manifest(result_dir: Path) -> dict[str, Any]:
    files = {
        name: {
            "sha256": sha256_file(result_dir / name),
            "bytes": (result_dir / name).stat().st_size,
        }
        for name in EXPECTED_PRIVATE_PAYLOAD_FILES
    }
    return {"schema": RESULT_MANIFEST_SCHEMA, "files": files}


def validate_closed_world_result_directory(
    result_dir: Path,
    *,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    private_root = (repo_root / "private_library").resolve()
    result_dir = _lexical_path(repo_root, result_dir)
    if not (
        _lexically_under(result_dir, private_root)
        and _resolved_under(result_dir, private_root)
        and result_dir.is_dir()
        and not _contains_symlink(result_dir, private_root)
    ):
        raise ValueError("private result directory boundary failed")
    children = sorted(result_dir.iterdir())
    if any(child.is_dir() for child in children):
        raise ValueError("private result directory contains a nested directory")
    if any(child.is_symlink() or not child.is_file() for child in children):
        raise ValueError("private result directory contains a non-regular file")
    if any(child.stat().st_nlink != 1 for child in children):
        raise ValueError("private result directory contains a hardlinked file")
    actual = {child.name for child in children}
    expected = set(EXPECTED_PRIVATE_PAYLOAD_FILES) | {"manifest.json"}
    if actual != expected:
        extra = sorted(actual - expected)
        missing = sorted(expected - actual)
        raise ValueError(
            f"closed-world result inventory drift; extra={extra}, missing={missing}"
        )
    if actual & FORBIDDEN_RESULT_NAMES:
        raise ValueError("private result contains a forbidden public artifact")
    manifest_path = result_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("private result manifest is invalid") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != RESULT_MANIFEST_SCHEMA
    ):
        raise ValueError("private result manifest schema drifted")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(EXPECTED_PRIVATE_PAYLOAD_FILES):
        raise ValueError("private result manifest inventory drifted")
    for name in EXPECTED_PRIVATE_PAYLOAD_FILES:
        entry = files[name]
        path = result_dir / name
        if not isinstance(entry, dict) or set(entry) != {"sha256", "bytes"}:
            raise ValueError(f"private result manifest entry drifted: {name}")
        if (
            entry["sha256"] != sha256_file(path)
            or entry["bytes"] != path.stat().st_size
        ):
            raise ValueError(f"private result payload hash or size mismatch: {name}")
    return {
        "schema": "n5-d5-private-result-closed-world-check-1.0",
        "valid": True,
        "payload_file_count": len(EXPECTED_PRIVATE_PAYLOAD_FILES),
        "manifest_sha256": sha256_file(manifest_path),
        "public_summary_present": False,
        "raw_trace_publication_authorized": False,
        "claim_authorizations": {key: False for key in CLAIM_KEYS},
    }


def _safe_private_output(repo_root: Path, output: Path) -> Path:
    repo_root = repo_root.resolve()
    private_root = (repo_root / "private_library").resolve()
    output = _lexical_path(repo_root, output)
    if not (
        _lexically_under(output, private_root)
        and _resolved_under(output.parent, private_root)
        and not _contains_symlink(output.parent, private_root)
    ):
        raise ValueError("L2 foundation reports must stay in private_library")
    if output.exists():
        raise FileExistsError(
            f"refusing to replace existing private report: {output.name}"
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-uncommitted-public-sources",
        action="store_true",
        help="tests only; real L2 foundation reports require committed public code",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_foundation_report(
        args.plan,
        repo_root=ROOT,
        enforce_public_committed=not args.allow_uncommitted_public_sources,
    )
    payload = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = _safe_private_output(ROOT, args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["foundation_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
