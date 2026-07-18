#!/usr/bin/env python3
"""Fail-closed static readiness audit for a private N5-D5 lab adapter.

This tool does not import or execute the private adapter and does not call a
renderer.  A green report authorizes only a local describe-probe preparation;
it never authorizes the 53-call scientific replay, reconstruction, training,
algorithm comparison, generalization, or publication claims.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from jsonschema import Draft202012Validator
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_RELATIVE = "data_templates/n5_d5_minimum_bost_interface.schema.json"
PUBLIC_READINESS_SOURCES = (
    SCHEMA_RELATIVE,
    "data_templates/n5_d5_private_adapter_skeleton.py",
    "demo_t16_operator/n5_d5_adapter_protocol.py",
    "site_tools/run_n5_d5_minimum_interface_bridge.py",
    "site_tools/validate_n5_d5_minimum_interface_bridge.py",
    "site_tools/n5_d5_private_lab_readiness.py",
    "site_tools/test_n5_d5_private_lab_readiness.py",
)
PLACEHOLDER_MARKERS = (
    "REPLACE_ME",
    "REPLACE_WITH",
    "N5_D5_PRIVATE_ADAPTER_NOT_IMPLEMENTED",
)
PLACEHOLDER_DIGESTS = {f"{value:064x}" for value in range(4)}
NETWORK_IMPORTS = {
    "aiohttp",
    "ftplib",
    "http.client",
    "httpx",
    "requests",
    "socket",
    "urllib",
}
SECRET_LITERAL = re.compile(
    r"(?i)\b(password|passwd|token|api[_-]?key|secret)\b\s*=\s*['\"][^'\"]+"
)
ABSOLUTE_PATH_LITERAL = re.compile(r"(?:file://|/Users/|/home/|[A-Za-z]:\\\\)")
SHELL_META = re.compile(r"[;&|><`\n\x00]")
CLAIM_KEYS = (
    "real_bost_interface",
    "physical_forward_correctness",
    "derivative_correctness",
    "reconstruction",
    "algorithm_superiority",
    "generalization",
    "publication",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _under(path: Path, parent: Path) -> bool:
    path = path.resolve()
    parent = parent.resolve()
    return path == parent or parent in path.parents


def _lexical_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return Path(os.path.abspath(path))


def _relative(repo_root: Path, path: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def _contains_symlink(path: Path, parent: Path) -> bool:
    """Inspect every lexical component below parent without following links."""
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


def _git(
    repo_root: Path,
    *args: str,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )


def _imported_modules(source: str) -> set[str]:
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(
        self,
        code: str,
        passed: bool,
        detail: str,
        *,
        severity: str = "blocker",
    ) -> None:
        self.checks.append(
            {
                "code": code,
                "passed": bool(passed),
                "severity": severity,
                "detail": detail,
            }
        )

    @property
    def blockers(self) -> list[dict[str, Any]]:
        return [
            item
            for item in self.checks
            if item["severity"] == "blocker" and not item["passed"]
        ]

    @property
    def warnings(self) -> list[dict[str, Any]]:
        return [
            item
            for item in self.checks
            if item["severity"] == "warning" and not item["passed"]
        ]


def _public_protocol_attestation(
    repo_root: Path,
    audit: Audit,
    *,
    enforce_committed: bool,
) -> tuple[str, dict[str, str]]:
    hashes: dict[str, str] = {}
    for relative in PUBLIC_READINESS_SOURCES:
        path = repo_root / relative
        exists = path.is_file() and not path.is_symlink()
        audit.add(
            f"PUBLIC_SOURCE_PRESENT_{len(hashes)}",
            exists,
            f"public protocol source {len(hashes)} is a regular file",
        )
        if not exists:
            continue
        hashes[relative] = sha256_file(path)
        if enforce_committed:
            tracked = _git(
                repo_root,
                "ls-files",
                "--error-unmatch",
                "--",
                relative,
            ).returncode == 0
            clean = _git(
                repo_root,
                "diff",
                "--quiet",
                "HEAD",
                "--",
                relative,
            ).returncode == 0
            audit.add(
                f"PUBLIC_SOURCE_TRACKED_{len(hashes) - 1}",
                tracked,
                f"public protocol source {len(hashes) - 1} is Git tracked",
            )
            audit.add(
                f"PUBLIC_SOURCE_CLEAN_{len(hashes) - 1}",
                clean,
                f"public protocol source {len(hashes) - 1} matches HEAD",
            )
    commit = _git(repo_root, "rev-parse", "HEAD")
    commit_text = (
        commit.stdout.decode("utf-8").strip()
        if commit.returncode == 0
        else "UNAVAILABLE"
    )
    audit.add(
        "PUBLIC_PROTOCOL_COMMIT_AVAILABLE",
        len(commit_text) == 40,
        "a 40-character public protocol commit is available",
    )
    return commit_text, hashes


def _private_file_check(
    repo_root: Path,
    private_root: Path,
    path: Path,
    label: str,
    audit: Audit,
) -> dict[str, Any] | None:
    lexical_inside = path == private_root or private_root in path.parents
    resolved_inside = _under(path, private_root)
    inside = lexical_inside and resolved_inside
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
        f"{label} and its path components below private_library are not symlinks",
    )
    regular = path.is_file() and no_symlink
    audit.add(
        f"{label.upper()}_REGULAR_FILE",
        regular,
        f"{label} is a non-symlink regular file",
    )
    if not regular:
        return None
    audit.add(
        f"{label.upper()}_NOT_HARDLINKED",
        path.stat().st_nlink == 1,
        f"{label} has exactly one filesystem link",
    )
    relative = _relative(repo_root, path)
    ignored = _git(repo_root, "check-ignore", "-q", "--", relative).returncode == 0
    tracked = (
        _git(
            repo_root,
            "ls-files",
            "--error-unmatch",
            "--",
            relative,
        ).returncode
        == 0
    )
    audit.add(
        f"{label.upper()}_GIT_IGNORED",
        ignored,
        f"{label} is protected by a Git ignore rule",
    )
    audit.add(
        f"{label.upper()}_NOT_TRACKED",
        not tracked,
        f"{label} is not present in the public Git index",
    )
    return {
        "label": label,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _schema_errors(schema: dict[str, Any], config: object) -> list[str]:
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(config), key=lambda item: list(item.path))
    return [
        f"{'/'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
        for error in errors[:12]
    ]


def build_readiness_report(
    config_path: Path,
    *,
    repo_root: Path = ROOT,
    enforce_public_committed: bool = True,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    private_root = (repo_root / "private_library").resolve()
    config_path = _lexical_path(repo_root, config_path)
    audit = Audit()
    protocol_commit, public_hashes = _public_protocol_attestation(
        repo_root,
        audit,
        enforce_committed=enforce_public_committed,
    )
    inventory: list[dict[str, Any]] = []
    config_entry = _private_file_check(
        repo_root,
        private_root,
        config_path,
        "private_config",
        audit,
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
                audit.add("CONFIG_JSON_OBJECT", False, "private config is not an object")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            audit.add("CONFIG_JSON_PARSE", False, "private config is not valid UTF-8 JSON")
    if config is None:
        return _final_report(
            audit,
            protocol_commit=protocol_commit,
            public_hashes=public_hashes,
            inventory=inventory,
            config=None,
        )

    schema = json.loads((repo_root / SCHEMA_RELATIVE).read_text(encoding="utf-8"))
    errors = _schema_errors(schema, config)
    audit.add(
        "CONFIG_SCHEMA_VALID",
        not errors,
        "private config satisfies the N5-D5 JSON Schema"
        if not errors
        else "; ".join(errors),
    )
    if errors:
        return _final_report(
            audit,
            protocol_commit=protocol_commit,
            public_hashes=public_hashes,
            inventory=inventory,
            config=config,
        )

    audit.add(
        "LAB_RECORD_KIND",
        config["record_kind"] == "LAB_ANONYMOUS_INTERFACE",
        "record_kind is LAB_ANONYMOUS_INTERFACE",
    )
    identity = config["identity"]
    audit.add(
        "ANONYMIZED_IDENTITY",
        identity["anonymized"] is True
        and identity["contains_private_identifiers"] is False,
        "identity explicitly excludes private identifiers",
    )
    audit.add(
        "REAL_DATA_ACKNOWLEDGED",
        identity["real_lab_data_present"] is True,
        "the private record explicitly acknowledges real lab material",
    )
    privacy = config["privacy"]
    audit.add(
        "RAW_TRACE_PUBLICATION_CLOSED",
        privacy["raw_trace_publication_permitted"] is False
        and privacy["public_summary_permitted"] is False,
        "raw traces and summaries are not authorized for public publication",
    )
    audit.add(
        "PRIVATE_ADAPTER_REQUIRED",
        privacy["private_adapter_required_for_lab"] is True,
        "the lab adapter is required to remain private",
    )
    claims = config["claim_authorizations"]
    audit.add(
        "ALL_SCIENTIFIC_CLAIMS_CLOSED",
        set(claims) == set(CLAIM_KEYS) and not any(claims.values()),
        "all seven scientific claim authorizations remain false",
    )
    serialized = _canonical_json(config)
    audit.add(
        "PLACEHOLDERS_REPLACED",
        not any(marker in serialized for marker in PLACEHOLDER_MARKERS),
        "all REPLACE/TODO markers have been removed",
    )
    audit.add(
        "NO_ABSOLUTE_OR_FILE_URL_IN_CONFIG",
        ABSOLUTE_PATH_LITERAL.search(serialized) is None,
        "the private config contains no absolute local path or file URL",
    )
    adapter = config["adapter"]
    command = [str(token) for token in adapter["command"]]
    audit.add(
        "COMMAND_NO_SHELL_META",
        not any(SHELL_META.search(token) for token in command),
        "adapter command contains no shell metacharacters or control bytes",
    )
    audit.add(
        "COMMAND_USES_PYTHON_TOKEN",
        bool(command) and command[0] == "{python}",
        "adapter command uses the portable {python} token",
    )
    command_scripts = [token for token in command[1:] if token.endswith(".py")]
    audit.add(
        "COMMAND_HAS_ONE_PRIVATE_SCRIPT",
        len(command_scripts) == 1,
        "adapter command identifies exactly one Python entry script",
    )
    command_path = (
        _lexical_path(repo_root, command_scripts[0])
        if len(command_scripts) == 1
        else None
    )
    source_paths = [_lexical_path(repo_root, item) for item in adapter["source_files"]]
    audit.add(
        "SOURCE_REVIEW_STATUS_PRIVATE",
        adapter["source_review_status"] == "private_source_reviewed",
        "adapter source is marked private_source_reviewed",
    )
    audit.add(
        "COMMAND_SCRIPT_IN_SOURCE_INVENTORY",
        command_path is not None and command_path in source_paths,
        "the command entry script is included in source_files",
    )
    source_entries: dict[Path, dict[str, Any]] = {}
    for index, source_path in enumerate(source_paths):
        entry = _private_file_check(
            repo_root,
            private_root,
            source_path,
            f"adapter_source_{index}",
            audit,
        )
        if entry is None:
            continue
        inventory.append(entry)
        source_entries[source_path] = entry
        try:
            source = source_path.read_text(encoding="utf-8")
            ast.parse(source)
            syntax_ok = True
        except (OSError, UnicodeDecodeError, SyntaxError):
            source = ""
            syntax_ok = False
        audit.add(
            f"ADAPTER_SOURCE_{index}_AST_VALID",
            syntax_ok,
            f"adapter source {index} is valid UTF-8 Python syntax",
        )
        audit.add(
            f"ADAPTER_SOURCE_{index}_IMPLEMENTED",
            syntax_ok
            and not any(marker in source for marker in PLACEHOLDER_MARKERS)
            and "NotImplementedError" not in source,
            f"adapter source {index} contains no teaching placeholder",
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
            f"ADAPTER_SOURCE_{index}_NO_NETWORK_IMPORT",
            not unsafe_imports,
            f"adapter source {index} has no network-egress import",
        )
        audit.add(
            f"ADAPTER_SOURCE_{index}_NO_SECRET_LITERAL",
            SECRET_LITERAL.search(source) is None,
            f"adapter source {index} has no obvious embedded credential literal",
        )
        audit.add(
            f"ADAPTER_SOURCE_{index}_NO_ABSOLUTE_PATH_LITERAL",
            ABSOLUTE_PATH_LITERAL.search(source) is None,
            f"adapter source {index} has no absolute local path literal",
        )
    expected_implementation = str(adapter["expected_implementation_sha256"])
    actual_implementation = (
        source_entries.get(command_path, {}).get("sha256")
        if command_path is not None
        else None
    )
    audit.add(
        "IMPLEMENTATION_HASH_MATCHES_ENTRY_SCRIPT",
        actual_implementation == expected_implementation,
        "expected implementation hash matches the private command script",
    )

    base_spec = config["field"]["base_input"]
    base_path = _lexical_path(repo_root, str(base_spec["relative_path"]))
    base_entry = _private_file_check(
        repo_root,
        private_root,
        base_path,
        "base_input",
        audit,
    )
    if base_entry is not None:
        inventory.append(base_entry)
        audit.add(
            "BASE_INPUT_HASH_MATCHES",
            base_entry["sha256"] == base_spec["sha256"],
            "base input bytes match the declared SHA-256",
        )
        try:
            base = np.load(base_path, allow_pickle=False)
            load_ok = True
        except (OSError, ValueError):
            base = np.asarray([], dtype=np.float64)
            load_ok = False
        audit.add(
            "BASE_INPUT_NUMPY_LOADABLE",
            load_ok,
            "base input is a NumPy array loadable without pickle",
        )
        if load_ok:
            audit.add(
                "BASE_INPUT_SIZE_MATCHES",
                base.size == int(config["field"]["input_dimension"]),
                "base input flattened size matches input_dimension",
            )
            audit.add(
                "BASE_INPUT_DTYPE_MATCHES",
                base.dtype == np.dtype(config["field"]["dtype"]),
                "base input dtype matches the field contract",
            )
            audit.add(
                "BASE_INPUT_FINITE",
                bool(np.all(np.isfinite(base))),
                "base input contains only finite values",
            )
    semantic_digests = {
        str(item["semantic_digest_sha256"]) for item in config["paths"]
    }
    audit.add(
        "PATH_DIGESTS_NOT_PLACEHOLDERS",
        not bool(semantic_digests & PLACEHOLDER_DIGESTS),
        "all path semantic digests are non-placeholder values",
    )
    audit.add(
        "FORMAL_REPLAY_PRIVATE_ATTESTATION_AVAILABLE",
        False,
        (
            "the frozen synthetic runner still requires tracked sources; a separate "
            "private attestation/replay layer must be committed before 53 calls"
        ),
        severity="warning",
    )
    audit.add(
        "PHYSICAL_TOLERANCES_REVIEWED",
        False,
        (
            "engineering tolerances are configuration declarations and still require "
            "laboratory precision/noise-floor review before formal replay"
        ),
        severity="warning",
    )
    audit.add(
        "PRIVATE_DEPENDENCY_CLOSURE_REVIEWED",
        False,
        (
            "static source_files do not yet prove that every imported module, dynamic "
            "library, calibration file, weight, and cache is content-addressed"
        ),
        severity="warning",
    )
    audit.add(
        "FORMAL_REPLAY_CLOSED_WORLD_MANIFEST_AVAILABLE",
        False,
        (
            "the frozen validator verifies listed artifacts but a private formal layer "
            "must also reject every unlisted file in the result directory"
        ),
        severity="warning",
    )
    audit.add(
        "LAB_PUBLIC_SUMMARY_HARD_GUARD_AVAILABLE",
        False,
        (
            "public_summary_permitted is false, but a private formal runner must still "
            "turn that declaration into a hard no-public-summary write rule"
        ),
        severity="warning",
    )
    audit.add(
        "UNPREDICTABLE_PRIVATE_PROBES_AVAILABLE",
        False,
        (
            "the public two-tangent/one-cotangent plan is replayable but does not yet "
            "include validator-created private probes against lookup-table behavior"
        ),
        severity="warning",
    )
    return _final_report(
        audit,
        protocol_commit=protocol_commit,
        public_hashes=public_hashes,
        inventory=inventory,
        config=config,
    )


def _final_report(
    audit: Audit,
    *,
    protocol_commit: str,
    public_hashes: dict[str, str],
    inventory: list[dict[str, Any]],
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    ready = not audit.blockers
    identity_digest = None
    if config is not None and isinstance(config.get("identity"), dict):
        identity_digest = hashlib.sha256(
            _canonical_json(config["identity"]).encode("utf-8")
        ).hexdigest()
    return {
        "schema_version": "n5-d5-private-lab-readiness-1.0",
        "status": (
            "STATIC_PRIVATE_INTAKE_READY_FORMAL_REPLAY_LOCKED"
            if ready
            else "PRIVATE_INTAKE_READINESS_BLOCKED"
        ),
        "evidence_scope": (
            "STATIC_PRIVATE_FILE_SCHEMA_HASH_ARRAY_AND_SOURCE_PREFLIGHT_"
            "NO_ADAPTER_EXECUTION_NO_RENDERER_CALLS"
        ),
        "ready_for_private_describe_probe": ready,
        "formal_53_call_replay_authorized": False,
        "protocol_commit": protocol_commit,
        "public_protocol_source_hashes": public_hashes,
        "private_identity_sha256": identity_digest,
        "private_inventory": inventory,
        "checks": audit.checks,
        "blocker_count": len(audit.blockers),
        "warning_count": len(audit.warnings),
        "blocker_codes": [item["code"] for item in audit.blockers],
        "warning_codes": [item["code"] for item in audit.warnings],
        "claim_authorizations": {key: False for key in CLAIM_KEYS},
        "next_action": (
            "REVIEW_PRIVATE_ADAPTER_THEN_BUILD_PRIVATE_ATTESTED_REPLAY"
            if ready
            else "FIX_STATIC_PRIVATE_INTAKE_BLOCKERS"
        ),
        "limitations": [
            "The private adapter is parsed but never imported or executed.",
            "No renderer output, JVP, VJP, branch state, or cost ledger is observed.",
            "Git-ignore and hash checks reduce publication risk but do not prove physical correctness.",
            "A green report does not authorize the frozen 53-call replay or any model training.",
        ],
    }


def _safe_private_output(repo_root: Path, output: Path) -> Path:
    private_root = (repo_root / "private_library").resolve()
    output = output.resolve()
    if not _under(output, private_root):
        raise ValueError("readiness reports containing private hashes must stay in private_library")
    if output.exists():
        raise FileExistsError(f"refusing to replace existing private report: {output.name}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-uncommitted-public-sources",
        action="store_true",
        help="tests only; real readiness requires committed public audit code",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_readiness_report(
        args.config,
        repo_root=ROOT,
        enforce_public_committed=not args.allow_uncommitted_public_sources,
    )
    payload = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = _safe_private_output(ROOT, args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["ready_for_private_describe_probe"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
