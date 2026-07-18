#!/usr/bin/env python3
"""Static fail-closed readiness audit for the independent N5-D5 dual-path v2.

The v2 protocol accepts exactly the curved and straight callables when a
laboratory backend has no native direct-residual callable.  It never creates a
third path by subtracting endpoint outputs.  This audit parses private source
without importing or executing it, preserves the frozen three-path v1 files,
and never authorizes the 36-call replay, reconstruction, training, or a claim.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from jsonschema import Draft202012Validator
import numpy as np

from site_tools.n5_d5_private_lab_readiness import (
    ABSOLUTE_PATH_LITERAL,
    Audit,
    CLAIM_KEYS,
    NETWORK_IMPORTS,
    PLACEHOLDER_MARKERS,
    SECRET_LITERAL,
    SHELL_META,
    _canonical_json,
    _contains_symlink,
    _git,
    _imported_modules,
    _lexical_path,
    _private_file_check,
    _schema_errors,
    _under,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
DUAL_SCHEMA_RELATIVE = (
    "data_templates/n5_d5_minimum_bost_interface_dual_v2.schema.json"
)
DUAL_PLACEHOLDER_RELATIVE = (
    "data_templates/n5_d5_lab_interface_dual_v2.placeholder.json"
)
V1_SCHEMA_RELATIVE = "data_templates/n5_d5_minimum_bost_interface.schema.json"
PUBLIC_DUAL_SOURCES = (
    V1_SCHEMA_RELATIVE,
    "site_tools/n5_d5_private_lab_readiness.py",
    "site_tools/test_n5_d5_private_lab_readiness.py",
    DUAL_SCHEMA_RELATIVE,
    DUAL_PLACEHOLDER_RELATIVE,
    "site_tools/n5_d5_private_lab_readiness_dual_v2.py",
    "site_tools/test_n5_d5_private_lab_readiness_dual_v2.py",
)
EXPECTED_PRIMARY_COST = {
    "describe": 2,
    "forward": 28,
    "jvp": 4,
    "vjp": 2,
    "total": 36,
}
DUAL_ROLES = ("curved", "straight")
PLACEHOLDER_DIGESTS = {f"{value:064x}" for value in range(32)}


def _public_dual_attestation(
    repo_root: Path,
    audit: Audit,
    *,
    enforce_committed: bool,
) -> tuple[str, dict[str, str]]:
    hashes: dict[str, str] = {}
    for index, relative in enumerate(PUBLIC_DUAL_SOURCES):
        path = repo_root / relative
        exists = path.is_file() and not path.is_symlink()
        audit.add(
            f"DUAL_PUBLIC_SOURCE_{index}_PRESENT",
            exists,
            f"dual-path public source {index} is a regular file",
        )
        if not exists:
            continue
        digest = sha256_file(path)
        hashes[relative] = digest
        if enforce_committed:
            tracked = (
                _git(repo_root, "ls-files", "--error-unmatch", "--", relative).returncode
                == 0
            )
            clean = _git(repo_root, "diff", "--quiet", "HEAD", "--", relative).returncode == 0
            committed = _git(repo_root, "show", f"HEAD:{relative}")
            commit_match = (
                committed.returncode == 0
                and hashlib.sha256(committed.stdout).hexdigest() == digest
            )
            audit.add(
                f"DUAL_PUBLIC_SOURCE_{index}_TRACKED",
                tracked,
                f"dual-path public source {index} is Git tracked",
            )
            audit.add(
                f"DUAL_PUBLIC_SOURCE_{index}_CLEAN",
                clean,
                f"dual-path public source {index} matches HEAD",
            )
            audit.add(
                f"DUAL_PUBLIC_SOURCE_{index}_COMMIT_MATCH",
                commit_match,
                f"dual-path public source {index} bytes match HEAD",
            )
    commit = _git(repo_root, "rev-parse", "HEAD")
    commit_text = (
        commit.stdout.decode("utf-8").strip()
        if commit.returncode == 0
        else "UNAVAILABLE"
    )
    audit.add(
        "DUAL_PUBLIC_COMMIT_AVAILABLE",
        len(commit_text) == 40,
        "a 40-character dual-path public protocol commit is available",
    )
    return commit_text, hashes


def _shared_definition_errors(
    v1_schema: dict[str, Any], definition: str, value: object
) -> list[str]:
    wrapper = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": v1_schema["$defs"],
        "$ref": f"#/$defs/{definition}",
    }
    errors = sorted(
        Draft202012Validator(wrapper).iter_errors(value),
        key=lambda item: list(item.path),
    )
    return [
        f"{'/'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
        for error in errors[:12]
    ]


def _subscript_key(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int)):
        return repr(node.value)
    return None


def _role_hint(
    node: ast.AST,
    aliases: dict[str, str],
    container_roles: dict[str, dict[str, str]] | None = None,
) -> str | None:
    if isinstance(node, ast.Name):
        if node.id in aliases:
            return aliases[node.id]
        lowered = node.id.lower()
        for role in DUAL_ROLES:
            if role in lowered:
                return role
    if isinstance(node, ast.Attribute):
        lowered = node.attr.lower()
        for role in DUAL_ROLES:
            if role in lowered:
                return role
    if isinstance(node, ast.Subscript):
        key = _subscript_key(node.slice)
        if key is not None and isinstance(node.value, ast.Name):
            role = (container_roles or {}).get(node.value.id, {}).get(key)
            if role is not None:
                return role
        slice_role = _role_hint_leaf(node.slice, aliases)
        if slice_role is not None:
            return slice_role
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        lowered = node.value.lower()
        for role in DUAL_ROLES:
            if lowered == role or role in lowered:
                return role
    if isinstance(node, ast.Call):
        hints = {
            hint
            for child in ast.walk(node)
            if (hint := _role_hint_leaf(child, aliases)) is not None
        }
        if len(hints) == 1:
            return next(iter(hints))
    return None


def _bind_assignment_roles(
    target: ast.AST,
    value: ast.AST,
    aliases: dict[str, str],
    container_roles: dict[str, dict[str, str]],
) -> None:
    if isinstance(target, (ast.Tuple, ast.List)) and isinstance(
        value, (ast.Tuple, ast.List)
    ):
        if len(target.elts) == len(value.elts):
            for nested_target, nested_value in zip(target.elts, value.elts):
                _bind_assignment_roles(
                    nested_target, nested_value, aliases, container_roles
                )
        return
    if not isinstance(target, ast.Name):
        return
    role = _role_hint(value, aliases, container_roles)
    if role is not None:
        aliases[target.id] = role
    mapping: dict[str, str] = {}
    if isinstance(value, (ast.Tuple, ast.List)):
        for index, item in enumerate(value.elts):
            item_role = _role_hint(item, aliases, container_roles)
            if item_role is not None:
                mapping[repr(index)] = item_role
    elif isinstance(value, ast.Dict):
        for key_node, item in zip(value.keys, value.values):
            if key_node is None:
                continue
            key = _subscript_key(key_node)
            item_role = _role_hint(item, aliases, container_roles)
            if key is not None and item_role is not None:
                mapping[key] = item_role
    if mapping:
        container_roles[target.id] = mapping


def _role_hint_leaf(node: ast.AST, aliases: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        if node.id in aliases:
            return aliases[node.id]
        lowered = node.id.lower()
    elif isinstance(node, ast.Attribute):
        lowered = node.attr.lower()
    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
        lowered = node.value.lower()
    else:
        return None
    for role in DUAL_ROLES:
        if lowered == role or role in lowered:
            return role
    return None


def _wrapper_subtraction_lines(tree: ast.AST) -> list[int]:
    aliases: dict[str, str] = {}
    container_roles: dict[str, dict[str, str]] = {}
    subtraction_call_names = {"sub", "subtract"}
    subtraction_module_aliases = {"operator", "np", "numpy"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name in {"operator", "numpy"}:
                    subtraction_module_aliases.add(
                        (imported.asname or imported.name.split(".")[0]).lower()
                    )
        elif isinstance(node, ast.ImportFrom) and node.module in {
            "operator",
            "numpy",
        }:
            for imported in node.names:
                if imported.name in {"sub", "subtract"}:
                    subtraction_call_names.add(
                        (imported.asname or imported.name).lower()
                    )
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if value is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                _bind_assignment_roles(target, value, aliases, container_roles)
            subtraction_alias = False
            if isinstance(value, ast.Name):
                subtraction_alias = value.id.lower() in subtraction_call_names
            elif isinstance(value, ast.Attribute):
                subtraction_alias = (
                    value.attr.lower() in {"sub", "subtract"}
                    and isinstance(value.value, ast.Name)
                    and value.value.id.lower() in subtraction_module_aliases
                )
            if subtraction_alias:
                for target in targets:
                    if isinstance(target, ast.Name):
                        subtraction_call_names.add(target.id.lower())
    lines: list[int] = []
    for node in ast.walk(tree):
        left: str | None = None
        right: str | None = None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
            left = _role_hint(node.left, aliases, container_roles)
            right = _role_hint(node.right, aliases, container_roles)
        elif isinstance(node, ast.Call) and len(node.args) >= 2:
            function_name = ""
            if isinstance(node.func, ast.Name):
                function_name = node.func.id.lower()
            elif isinstance(node.func, ast.Attribute):
                function_name = node.func.attr.lower()
            if function_name in subtraction_call_names:
                left = _role_hint(node.args[0], aliases, container_roles)
                right = _role_hint(node.args[1], aliases, container_roles)
        if {left, right} == set(DUAL_ROLES):
            lines.append(int(getattr(node, "lineno", -1)))
    return sorted(set(lines))


def _direct_residual_markers(tree: ast.AST) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            callable_like = isinstance(
                value, (ast.Lambda, ast.Call, ast.Name, ast.Attribute)
            )
            if callable_like:
                for target in targets:
                    target_text = ""
                    if isinstance(target, ast.Name):
                        target_text = target.id
                    elif isinstance(target, ast.Attribute):
                        target_text = target.attr
                    if "direct_residual" in target_text.lower():
                        lines.append(int(getattr(node, "lineno", -1)))
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for imported in node.names:
                exposed = imported.asname or imported.name
                if (
                    "direct_residual" in exposed.lower()
                    or "direct_residual" in imported.name.lower()
                ):
                    lines.append(int(getattr(node, "lineno", -1)))
        text: str | None = None
        callable_marker = False
        if isinstance(node, ast.Call):
            callable_marker = True
            if isinstance(node.func, ast.Name):
                text = node.func.id
            elif isinstance(node.func, ast.Attribute):
                text = node.func.attr
            function_name = text.lower() if text is not None else ""
            if function_name in {"setattr", "getattr"} and len(node.args) >= 2:
                attribute_name = node.args[1]
                if (
                    isinstance(attribute_name, ast.Constant)
                    and isinstance(attribute_name.value, str)
                    and "direct_residual" in attribute_name.value.lower()
                ):
                    lines.append(int(getattr(node, "lineno", -1)))
        elif isinstance(node, ast.Name):
            text = node.id
        elif isinstance(node, ast.Attribute):
            text = node.attr
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            text = node.name
            callable_marker = True
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
        lowered = text.lower() if text is not None else ""
        metadata_only = (
            not callable_marker
            and lowered
            in {
                "native_direct_residual",
                "native_direct_residual_supported",
                "native_direct_residual_available",
            }
        )
        if (
            "direct_residual" in lowered
            and not metadata_only
            and lowered not in {"direct_residual_semantics", "direct_residual_supported"}
        ):
            lines.append(int(getattr(node, "lineno", -1)))
    return sorted(set(lines))


def _final_report(
    audit: Audit,
    *,
    protocol_commit: str,
    public_hashes: dict[str, str],
    inventory: list[dict[str, Any]],
    config: dict[str, Any] | None,
    derived_primary_cost: dict[str, int] | None = None,
) -> dict[str, Any]:
    ready = not audit.blockers
    blocker_codes = [item["code"] for item in audit.blockers]
    wrapper_blocked = any(
        code.endswith("WRAPPER_ENDPOINT_SUBTRACTION_ABSENT")
        for code in blocker_codes
    )
    direct_marker_blocked = any(
        code.endswith("NO_DIRECT_RESIDUAL_MARKER") for code in blocker_codes
    )
    identity_digest = None
    if config is not None and isinstance(config.get("identity"), dict):
        identity_digest = hashlib.sha256(
            _canonical_json(config["identity"]).encode("utf-8")
        ).hexdigest()
    return {
        "schema_version": "n5-d5-private-lab-readiness-dual-v2-1.0",
        "protocol_variant": "dual_path_no_native_direct_v2",
        "status": (
            "STATIC_PRIVATE_DUAL_PATH_INTAKE_READY_FORMAL_REPLAY_LOCKED"
            if ready
            else "PRIVATE_DUAL_PATH_INTAKE_READINESS_BLOCKED"
        ),
        "evidence_scope": (
            "STATIC_PRIVATE_DUAL_PATH_SCHEMA_HASH_ARRAY_SOURCE_AND_WRAPPER_PREFLIGHT_"
            "NO_ADAPTER_EXECUTION_NO_RENDERER_CALLS"
        ),
        "capability_status": "NATIVE_DIRECT_RESIDUAL_UNAVAILABLE",
        "path_status": "EXACTLY_CURVED_AND_STRAIGHT",
        "wrapper_subtraction_status": (
            "WRAPPER_SUBTRACTION_DETECTED"
            if wrapper_blocked
            else "FORBIDDEN_AND_NOT_DETECTED_BY_STATIC_HEURISTIC"
        ),
        "direct_residual_marker_status": (
            "DIRECT_RESIDUAL_MARKER_DETECTED"
            if direct_marker_blocked
            else "ABSENT_BY_STATIC_HEURISTIC"
        ),
        "ready_for_private_describe_probe": ready,
        "formal_dual_primary_36_call_replay_authorized": False,
        "formal_triple_primary_53_call_replay_authorized": False,
        "formal_53_call_replay_authorized": False,
        "formal_replay_authorized": False,
        "adapter_executed": False,
        "primary_cost": derived_primary_cost,
        "expected_primary_cost": EXPECTED_PRIMARY_COST,
        "path_count": (
            len(config.get("paths", []))
            if isinstance(config, dict) and isinstance(config.get("paths"), list)
            else None
        ),
        "protocol_commit": protocol_commit,
        "public_protocol_source_hashes": public_hashes,
        "private_identity_sha256": identity_digest,
        "private_inventory": inventory,
        "checks": audit.checks,
        "blocker_count": len(audit.blockers),
        "warning_count": len(audit.warnings),
        "blocker_codes": blocker_codes,
        "warning_codes": [item["code"] for item in audit.warnings],
        "claim_authorizations": {key: False for key in CLAIM_KEYS},
        "next_action": (
            "BUILD_SEPARATELY_AUTHORIZED_L2A_L2B_DUAL_PATH_CHAIN"
            if ready
            else "FIX_STATIC_DUAL_PATH_INTAKE_BLOCKERS"
        ),
        "limitations": [
            "The private adapter is parsed but never imported, launched, or queried.",
            "Static wrapper heuristics catch declared endpoint subtraction patterns but cannot prove arbitrary runtime behavior.",
            "The 36-request budget is an interface-audit cost, not training or reconstruction performance.",
            "No native direct-residual callable or endpoint subtraction was detected by the current static heuristics; arbitrary runtime absence is not proven.",
            "No BOST physics, derivative, reconstruction, superiority, generalization, or publication claim is authorized.",
        ],
    }


def build_dual_readiness_report(
    config_path: Path,
    *,
    repo_root: Path = ROOT,
    enforce_public_committed: bool = True,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    private_root = (repo_root / "private_library").resolve()
    config_path = _lexical_path(repo_root, config_path)
    audit = Audit()
    protocol_commit, public_hashes = _public_dual_attestation(
        repo_root, audit, enforce_committed=enforce_public_committed
    )
    inventory: list[dict[str, Any]] = []
    config_entry = _private_file_check(
        repo_root, private_root, config_path, "dual_private_config", audit
    )
    if config_entry is not None:
        inventory.append(config_entry)
    config: dict[str, Any] | None = None
    if config_entry is not None:
        try:
            value = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                config = value
            else:
                audit.add("DUAL_CONFIG_JSON_OBJECT", False, "dual config is not an object")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            audit.add("DUAL_CONFIG_JSON_PARSE", False, "dual config is not valid UTF-8 JSON")
    if config is None:
        return _final_report(
            audit,
            protocol_commit=protocol_commit,
            public_hashes=public_hashes,
            inventory=inventory,
            config=None,
        )

    dual_schema = json.loads(
        (repo_root / DUAL_SCHEMA_RELATIVE).read_text(encoding="utf-8")
    )
    dual_errors = _schema_errors(dual_schema, config)
    audit.add(
        "DUAL_CONFIG_SCHEMA_VALID",
        not dual_errors,
        "dual config satisfies its independent JSON Schema"
        if not dual_errors
        else "; ".join(dual_errors),
    )
    if dual_errors:
        return _final_report(
            audit,
            protocol_commit=protocol_commit,
            public_hashes=public_hashes,
            inventory=inventory,
            config=config,
        )

    v1_schema = json.loads(
        (repo_root / V1_SCHEMA_RELATIVE).read_text(encoding="utf-8")
    )
    shared_definitions = {
        "identity": "identity",
        "adapter": "adapter",
        "field": "field",
        "observation": "observation",
        "probe_plan": "probePlan",
        "state_contract": "stateContract",
        "engineering_tolerances": "tolerances",
        "privacy": "privacy",
        "claim_authorizations": "claimAuthorizations",
    }
    shared_valid = True
    for field, definition in shared_definitions.items():
        errors = _shared_definition_errors(v1_schema, definition, config[field])
        shared_valid = shared_valid and not errors
        audit.add(
            f"DUAL_SHARED_{field.upper()}_VALID",
            not errors,
            f"dual {field} preserves the frozen v1 shared contract"
            if not errors
            else "; ".join(errors),
        )
    if not shared_valid:
        return _final_report(
            audit,
            protocol_commit=protocol_commit,
            public_hashes=public_hashes,
            inventory=inventory,
            config=config,
        )

    identity = config["identity"]
    audit.add(
        "DUAL_ANONYMIZED_IDENTITY",
        identity["anonymized"] is True
        and identity["contains_private_identifiers"] is False,
        "dual identity explicitly excludes private identifiers",
    )
    audit.add(
        "DUAL_REAL_DATA_ACKNOWLEDGED",
        identity["real_lab_data_present"] is True,
        "dual private record acknowledges real lab material",
    )
    claims = config["claim_authorizations"]
    audit.add(
        "DUAL_ALL_SCIENTIFIC_CLAIMS_CLOSED",
        set(claims) == set(CLAIM_KEYS) and not any(claims.values()),
        "all seven dual-path scientific authorizations remain false",
    )
    serialized = _canonical_json(config)
    audit.add(
        "DUAL_PLACEHOLDERS_REPLACED",
        not any(marker in serialized for marker in PLACEHOLDER_MARKERS),
        "all dual-path REPLACE/TODO markers have been removed",
    )
    audit.add(
        "DUAL_NO_ABSOLUTE_OR_FILE_URL_IN_CONFIG",
        ABSOLUTE_PATH_LITERAL.search(serialized) is None,
        "dual private config contains no absolute path or file URL",
    )

    paths = config["paths"]
    roles = [str(path["role"]) for path in paths]
    audit.add(
        "DUAL_EXACT_PATH_ROLES",
        len(paths) == 2 and set(roles) == set(DUAL_ROLES) and len(set(roles)) == 2,
        "dual protocol contains exactly one curved and one straight path",
    )
    for field in ("path_id", "callable_id", "semantic_digest_sha256"):
        values = [str(path[field]) for path in paths]
        audit.add(
            f"DUAL_UNIQUE_{field.upper()}",
            len(values) == len(set(values)) == 2,
            f"dual path {field} values are unique",
        )
    semantic_digests = {
        str(path["semantic_digest_sha256"]) for path in paths
    }
    audit.add(
        "DUAL_PATH_DIGESTS_NOT_PLACEHOLDERS",
        not bool(semantic_digests & PLACEHOLDER_DIGESTS),
        "dual path semantic digests are non-placeholder values",
    )
    audit.add(
        "DUAL_NATIVE_DIRECT_FALSE",
        config["native_direct_residual_supported"] is False
        and config["residual_contract"]["native_direct_residual"] is False,
        "config and residual contract both declare native direct unavailable",
    )
    cost = config["cost_contract"]
    declared_cost = {
        "describe": int(cost["describe_calls"]),
        "forward": int(cost["forward_api_calls"]),
        "jvp": int(cost["jvp_api_calls"]),
        "vjp": int(cost["vjp_api_calls"]),
        "total": int(cost["primary_total_requests"]),
    }
    probe_plan = config["probe_plan"]
    path_count = len(paths)
    tangent_count = int(probe_plan["tangent_count"])
    cotangent_count = int(probe_plan["cotangent_count"])
    h_count = len(probe_plan["h_values"])
    derived_cost = {
        "describe": 2,
        "forward": path_count * (2 + 2 * tangent_count * h_count),
        "jvp": path_count * tangent_count,
        "vjp": path_count * cotangent_count,
    }
    derived_cost["total"] = sum(derived_cost.values())
    audit.add(
        "DUAL_PRIMARY_COST_DERIVED_AND_EXACT_36",
        declared_cost == derived_cost == EXPECTED_PRIMARY_COST,
        (
            "dual primary cost independently derives from 2 paths, 2 tangents, "
            "1 cotangent, and 3 finite-difference steps as 2 + 28 + 4 + 2 = 36"
        ),
    )

    adapter = config["adapter"]
    command = [str(token) for token in adapter["command"]]
    audit.add(
        "DUAL_COMMAND_NO_SHELL_META",
        not any(SHELL_META.search(token) for token in command),
        "dual adapter command contains no shell metacharacters",
    )
    audit.add(
        "DUAL_COMMAND_USES_PYTHON_TOKEN",
        bool(command) and command[0] == "{python}",
        "dual adapter command uses the portable {python} token",
    )
    command_scripts = [token for token in command[1:] if token.endswith(".py")]
    audit.add(
        "DUAL_COMMAND_HAS_ONE_PRIVATE_SCRIPT",
        len(command_scripts) == 1,
        "dual adapter command identifies exactly one Python entry script",
    )
    command_path = (
        _lexical_path(repo_root, command_scripts[0])
        if len(command_scripts) == 1
        else None
    )
    source_paths = [_lexical_path(repo_root, item) for item in adapter["source_files"]]
    audit.add(
        "DUAL_SOURCE_REVIEW_STATUS_PRIVATE",
        adapter["source_review_status"] == "private_source_reviewed",
        "dual adapter source is marked private_source_reviewed",
    )
    audit.add(
        "DUAL_COMMAND_SCRIPT_IN_SOURCE_INVENTORY",
        command_path is not None and command_path in source_paths,
        "dual command entrypoint is listed in source_files",
    )
    source_entries: dict[Path, dict[str, Any]] = {}
    for index, source_path in enumerate(source_paths):
        entry = _private_file_check(
            repo_root,
            private_root,
            source_path,
            f"dual_adapter_source_{index}",
            audit,
        )
        if entry is None:
            continue
        inventory.append(entry)
        source_entries[source_path] = entry
        try:
            source = source_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            syntax_ok = True
        except (OSError, UnicodeDecodeError, SyntaxError):
            source = ""
            tree = ast.Module(body=[], type_ignores=[])
            syntax_ok = False
        audit.add(
            f"DUAL_SOURCE_{index}_AST_VALID",
            syntax_ok,
            f"dual adapter source {index} is valid UTF-8 Python",
        )
        audit.add(
            f"DUAL_SOURCE_{index}_IMPLEMENTED",
            syntax_ok
            and not any(marker in source for marker in PLACEHOLDER_MARKERS)
            and "NotImplementedError" not in source,
            f"dual adapter source {index} contains no teaching placeholder",
        )
        imports = _imported_modules(source) if syntax_ok else set()
        unsafe_imports = sorted(
            module
            for module in imports
            if any(
                module == blocked or module.startswith(f"{blocked}.")
                for blocked in NETWORK_IMPORTS
            )
        )
        audit.add(
            f"DUAL_SOURCE_{index}_NO_NETWORK_IMPORT",
            not unsafe_imports,
            f"dual adapter source {index} has no network-egress import",
        )
        audit.add(
            f"DUAL_SOURCE_{index}_NO_SECRET_LITERAL",
            SECRET_LITERAL.search(source) is None,
            f"dual adapter source {index} has no obvious embedded credential",
        )
        audit.add(
            f"DUAL_SOURCE_{index}_NO_ABSOLUTE_PATH_LITERAL",
            ABSOLUTE_PATH_LITERAL.search(source) is None,
            f"dual adapter source {index} has no absolute local path literal",
        )
        subtraction_lines = _wrapper_subtraction_lines(tree) if syntax_ok else []
        audit.add(
            f"DUAL_SOURCE_{index}_WRAPPER_ENDPOINT_SUBTRACTION_ABSENT",
            not subtraction_lines,
            (
                f"dual adapter source {index} does not subtract curved and straight endpoints"
                if not subtraction_lines
                else f"endpoint subtraction detected at lines {subtraction_lines}"
            ),
        )
        direct_lines = _direct_residual_markers(tree) if syntax_ok else []
        audit.add(
            f"DUAL_SOURCE_{index}_NO_DIRECT_RESIDUAL_MARKER",
            not direct_lines,
            (
                f"dual adapter source {index} contains no direct_residual identifier"
                if not direct_lines
                else f"direct_residual marker detected at lines {direct_lines}"
            ),
        )
    actual_implementation = (
        source_entries.get(command_path, {}).get("sha256")
        if command_path is not None
        else None
    )
    audit.add(
        "DUAL_IMPLEMENTATION_HASH_MATCHES_ENTRYPOINT",
        actual_implementation == adapter["expected_implementation_sha256"],
        "dual expected implementation hash matches the command entrypoint",
    )

    base_spec = config["field"]["base_input"]
    base_path = _lexical_path(repo_root, str(base_spec["relative_path"]))
    base_entry = _private_file_check(
        repo_root, private_root, base_path, "dual_base_input", audit
    )
    if base_entry is not None:
        inventory.append(base_entry)
        audit.add(
            "DUAL_BASE_INPUT_HASH_MATCHES",
            base_entry["sha256"] == base_spec["sha256"],
            "dual base input bytes match the declared hash",
        )
        try:
            base = np.load(base_path, allow_pickle=False)
            load_ok = True
        except (OSError, ValueError):
            base = np.asarray([], dtype=np.float64)
            load_ok = False
        audit.add(
            "DUAL_BASE_INPUT_NUMPY_LOADABLE",
            load_ok,
            "dual base input is loadable without pickle",
        )
        if load_ok:
            audit.add(
                "DUAL_BASE_INPUT_SIZE_MATCHES",
                base.size == int(config["field"]["input_dimension"]),
                "dual base input size matches input_dimension",
            )
            audit.add(
                "DUAL_BASE_INPUT_DTYPE_MATCHES",
                base.dtype == np.dtype(config["field"]["dtype"]),
                "dual base input dtype matches field contract",
            )
            audit.add(
                "DUAL_BASE_INPUT_FINITE",
                bool(np.all(np.isfinite(base))),
                "dual base input contains only finite values",
            )

    audit.add(
        "DUAL_L2A_L2B_CHAIN_EXECUTED",
        False,
        "static dual readiness does not execute L2-A or L2-B",
        severity="warning",
    )
    audit.add(
        "DUAL_RUNTIME_ARBITRARY_X_V_Q_OBSERVED",
        False,
        "static callable declarations do not prove arbitrary runtime x/v/q",
        severity="warning",
    )
    audit.add(
        "DUAL_PHYSICAL_CORRECTNESS_OBSERVED",
        False,
        "static source review cannot prove BOST optical correctness",
        severity="warning",
    )
    audit.add(
        "DUAL_DEPENDENCY_CLOSURE_REVIEWED",
        False,
        "private imports, weights, libraries, and calibration still need a closed-world manifest",
        severity="warning",
    )
    return _final_report(
        audit,
        protocol_commit=protocol_commit,
        public_hashes=public_hashes,
        inventory=inventory,
        config=config,
        derived_primary_cost=derived_cost,
    )


def _safe_private_output(repo_root: Path, output: Path) -> Path:
    private_root = (repo_root / "private_library").resolve()
    output = _lexical_path(repo_root, output)
    if not (
        (output == private_root or private_root in output.parents)
        and _under(output.parent, private_root)
        and not _contains_symlink(output.parent, private_root)
    ):
        raise ValueError("dual readiness report must stay under private_library")
    if output.exists():
        raise FileExistsError("refusing to replace a dual readiness report")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-uncommitted-public-sources",
        action="store_true",
        help="tests only; a real readiness audit requires committed public sources",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_dual_readiness_report(
        args.config,
        repo_root=ROOT,
        enforce_public_committed=not args.allow_uncommitted_public_sources,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = _safe_private_output(ROOT, args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["ready_for_private_describe_probe"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
