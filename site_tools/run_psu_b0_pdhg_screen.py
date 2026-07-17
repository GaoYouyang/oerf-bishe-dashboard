#!/usr/bin/env python3
"""Run the preregistered two-replicate covariance-weighted PDHG screen.

The command is deliberately fail-closed.  It reads one frozen JSON file,
checks the exact 32-candidate grid and 17 controls/references, and refuses to
write a result unless all 49 methods contribute the expected 16 paired field
rows.  The two opened replicates are development evidence only.

Each ``(penalty, alpha)`` combination runs one maximum-depth trajectory and
reuses the frozen ``K=4/8/16/32`` checkpoint volumes.  Every candidate still
reports the logical calls of an independent deployment.  Shared norm setup,
physical prefix reuse, and terminal projections used only for objective
scoring are kept in separate ledgers.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch

from demo_t16_operator.detector_covariance_whitening import (
    DetectorCovarianceWhitening,
    WhitenedMeasurementOperator,
)
from demo_t16_operator.detector_graph_covariance import (
    detector_graph_spectral_basis,
)
from demo_t16_operator.psu_b0_classical_baselines import (
    GeneralizedSobolevDirection,
    preconditioned_cgls_reconstruction,
    preconditioned_cgls_trajectory,
)
from demo_t16_operator.psu_b0_detector_graph_features import (
    build_detector_knn_graph,
)
from demo_t16_operator.psu_b0_primal_dual import (
    BlockOperatorNormEstimate,
    estimate_block_operator_norms,
    isotropic_edge_penalty,
    primal_dual_reconstruction,
    regularization_site_mask,
)
from demo_t16_operator.psu_b0_reaction_phantoms import (
    reaction_morphology_batch,
)
from demo_t16_operator.psu_b0_streaming_operator import (
    zero_outer_boundary_support,
)
from site_tools.run_psu_b0_covariance_conditioned_pcgls_diagnosis import (
    _derive_smoke_config,
)
from site_tools.run_psu_b0_dg_wpcgls_smoke import (
    _calibrate,
    _flowon_unit_noise,
    _load_json,
    _synchronize,
)
from site_tools.run_psu_b0_real_interface_audit import (
    _make_operator,
    load_real_support_geometry,
)
from site_tools.run_psu_b0_spectral_preconditioner_pilot import (
    _field_metrics,
)
from site_tools.psu_b0_pdhg_evidence import (
    geometry_operator_anchor,
    verify_exact_anchor,
    whitener_anchor,
)


SCHEMA = "psu-b0-covariance-pdhg-screen-report-1.0"
V1_CONFIG_SCHEMA = "psu-b0-pdhg-scale-smoke-config-1.0"
V2_CONFIG_SCHEMA = "psu-b0-pdhg-scale-smoke-config-2.0-infrastructure-amendment"
V2_AMENDMENT_EXPECTED = {
    "amendment_type": "INFRASTRUCTURE_ONLY",
    "parent_git_commit": "04f0f8877dee500ad77bd184453166e3b1e325d1",
    "parent_config_path": (
        "demo_t16_operator/configs/psu_b0_pdhg_scale_smoke_v1.json"
    ),
    "parent_config_sha256": (
        "57a172e74cbab96173e463651d4420ef4306e4c245b3eb6bb5e4c8b03699fb01"
    ),
    "parent_protocol_path": (
        "docs/psu_b0_pdhg_preregistered_protocol_2026-07-17.md"
    ),
    "parent_protocol_sha256": (
        "73e98b9f8af615c86cdaa94579d537e6b8d05642d8041cd7f5dfe16ce6fd93e3"
    ),
    "parent_result_status": "PDHG_PREFLIGHT_INVALID",
    "parent_result_private_storage": True,
    "parent_preflight_invalid_sha256": (
        "12af605565568c744183858d52e93fe7095ac4773e465ef8e84c5238030f338b"
    ),
    "parent_public_summary_path": (
        "docs/psu_b0_pdhg_v1_preflight_invalid_public_summary_2026-07-17.json"
    ),
    "parent_public_summary_sha256": (
        "99b0b84b23f894a53c873077c60e3dfa32dcd95958edcfd206818ff653e26601"
    ),
    "parent_performance_metric_row_count": 0,
    "parent_performance_ranking_generated": False,
    "allowed_numeric_change": (
        "move_mps_float32_tensor_to_cpu_before_cpu_float64_audit_cast"
    ),
    "solver_math_unchanged": True,
    "scientific_protocol_unchanged": True,
    "reuse_parent_intermediate_state": False,
}
OPENED_REPLICATES = (0, 8)
REACTION_FAMILIES = (
    "plume",
    "wavy_front",
    "thin_front",
    "double_front",
    "annular_kernel",
    "oblique_shock",
    "vortex_pair",
    "multi_plume",
)
FRONT_CRITICAL_FAMILIES = frozenset(
    {
        "wavy_front",
        "thin_front",
        "double_front",
        "annular_kernel",
        "oblique_shock",
    }
)
PDHG_PENALTIES = ("tv", "huber")
PDHG_ALPHA_SPECS = (
    ("1/256", "1of256", 1.0 / 256.0),
    ("1/64", "1of64", 1.0 / 64.0),
    ("1/16", "1of16", 1.0 / 16.0),
    ("1/4", "1of4", 1.0 / 4.0),
)
PDHG_ITERATIONS = (4, 8, 16, 32)
GRAPH_STAGES = (2, 3, 4, 5, 6, 8, 12, 16, 24, 32)
COMPONENT_ID = "component_s3_k4"
HISTORICAL_SUP_IDS = (
    "sup_huber_k3_n1_g0p32_a0p7",
    "sup_huber_k6_n1_g2p56_a0p7",
)
EXPECTED_FORMAL_CANDIDATES = 32
EXPECTED_CONTROLS_AND_REFERENCES = 17
EXPECTED_METHODS = 49
EXPECTED_ROWS_PER_METHOD = len(OPENED_REPLICATES) * len(REACTION_FAMILIES)
EXPECTED_METRIC_ROWS = EXPECTED_METHODS * EXPECTED_ROWS_PER_METHOD
REQUIRED_SOURCE_PATH_KEYS = (
    "source_multiseed_config",
    "source_smoke_config",
    "source_conditioning_report",
    "source_runtime_anchor_manifest",
    "source_superiorization_rows",
    "source_superiorization_tail_rows",
)
STRESS_TRAJECTORY_SPECS = {
    "pdhg_data_only_k32": ("tv", 0.0),
    "pdhg_tv_a1of4_k32": ("tv", 1.0 / 4.0),
    "pdhg_huber_a1of4_k32": ("huber", 1.0 / 4.0),
}
STRESS_DUAL_FEASIBILITY_TOLERANCE = 1e-7
E1_TEST_PATHS = (
    "demo_t16_operator/test_psu_b0_primal_dual.py",
    "site_tools/test_psu_b0_pdhg_evidence.py",
    "site_tools/test_run_psu_b0_pdhg_screen.py",
)
E1_IMPLEMENTATION_PATHS = (
    "demo_t16_operator/psu_b0_primal_dual.py",
    "site_tools/psu_b0_pdhg_evidence.py",
    "site_tools/run_psu_b0_pdhg_screen.py",
)


class PDHGPreflightInvalid(RuntimeError):
    """Stop the formal screen before any performance rows are generated."""

    status = "PDHG_PREFLIGHT_INVALID"

    def __init__(
        self,
        audit: Mapping[str, Any],
        *,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        self.audit = dict(audit)
        self.evidence = dict(evidence or {})
        reasons = "; ".join(str(value) for value in audit.get("failures", ()))
        super().__init__(f"{self.status}: {reasons or 'stability gate failed'}")


@dataclass(frozen=True, slots=True)
class PDHGStressContext:
    """Truth-inaccessible inputs for the observation-only stability gate."""

    replicate: int
    operator: Any
    observation_b0: torch.Tensor
    sigma_by_view: torch.Tensor
    view_mask: torch.Tensor
    rays_per_view: int
    measurement_scale: float
    regularization_vector_count: int
    norm_estimate: BlockOperatorNormEstimate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_source_path(root: Path, raw_path: str) -> tuple[str, Path]:
    root = root.resolve()
    relative = Path(raw_path)
    if not raw_path or relative.is_absolute():
        raise ValueError("frozen source inputs must use relative paths")
    resolved = (root / relative).resolve()
    try:
        canonical = resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(
            f"config.source_sha256 path escapes repository root: {raw_path}"
        ) from error
    if canonical != relative.as_posix():
        raise ValueError(
            f"config.source_sha256 path must be canonical: {raw_path}"
        )
    return canonical, resolved


def _validated_amendment_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    schema = str(config.get("schema_version", ""))
    amendment = config.get("amendment")
    if schema == V1_CONFIG_SCHEMA:
        if amendment is not None:
            raise ValueError("v1 config must not carry v2 amendment metadata")
        return {"active": False, "schema_version": schema}
    if schema != V2_CONFIG_SCHEMA:
        raise ValueError("frozen JSON has an unsupported schema_version")
    if not isinstance(amendment, Mapping):
        raise ValueError("v2 config must contain infrastructure amendment metadata")
    if dict(amendment) != V2_AMENDMENT_EXPECTED:
        raise ValueError("v2 infrastructure amendment metadata is not exact")
    return {
        "active": True,
        "schema_version": schema,
        **dict(amendment),
    }


def validate_infrastructure_amendment(
    *,
    root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a v2 rerun to the immutable v1 failure and parent protocol."""

    audit = _validated_amendment_metadata(config)
    if not audit["active"]:
        return audit
    verified: dict[str, dict[str, str]] = {}
    for label, path_key, sha_key in (
        ("parent_config", "parent_config_path", "parent_config_sha256"),
        ("parent_protocol", "parent_protocol_path", "parent_protocol_sha256"),
        (
            "parent_public_summary",
            "parent_public_summary_path",
            "parent_public_summary_sha256",
        ),
    ):
        canonical, path = _canonical_source_path(root, str(audit[path_key]))
        if not path.is_file():
            raise ValueError(f"v2 amendment input is missing: {canonical}")
        actual = _sha256(path)
        expected = str(audit[sha_key])
        if actual != expected:
            raise ValueError(f"v2 amendment SHA-256 mismatch: {canonical}")
        verified[label] = {"path": canonical, "sha256": actual}
    audit["verified_parent_files"] = verified
    return audit


def validate_source_sha256(
    *,
    root: Path,
    config: Mapping[str, Any],
) -> dict[str, str]:
    """Verify the frozen repository-relative input manifest fail-closed."""

    raw_digests = config.get("source_sha256")
    if not isinstance(raw_digests, Mapping) or not raw_digests:
        raise ValueError(
            "config.source_sha256 must be a non-empty source-key-to-SHA256 object"
        )
    expected_keys = set(REQUIRED_SOURCE_PATH_KEYS)
    if set(raw_digests) != expected_keys:
        raise ValueError(
            "config.source_sha256 must exactly cover the frozen source path keys"
        )

    verified: dict[str, str] = {}
    for source_key in REQUIRED_SOURCE_PATH_KEYS:
        raw_path = config.get(source_key)
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"frozen JSON must contain path-valued {source_key}")
        canonical, source_path = _canonical_source_path(root, raw_path)
        digest = str(raw_digests[source_key])
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(
                f"config.source_sha256 has an invalid digest for {source_key}"
            )
        if not source_path.is_file():
            raise ValueError(f"frozen source input is missing: {canonical}")
        actual = _sha256(source_path)
        if actual != digest:
            raise ValueError(f"frozen source SHA-256 mismatch: {canonical}")
        verified[canonical] = actual

    return dict(sorted(verified.items()))


def validate_geometry_audit_manifest(
    *,
    root: Path,
    view_root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash only the frozen geometry audit manifest before opening view shards."""

    manifest = config.get("geometry_audit_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("config.geometry_audit_manifest must be an object")
    filename = manifest.get("filename")
    if not isinstance(filename, str) or not filename:
        raise ValueError("config.geometry_audit_manifest.filename is required")
    relative = Path(filename)
    if relative.is_absolute() or relative.name != filename:
        raise ValueError("geometry audit manifest filename must be a plain filename")
    configured_path = manifest.get("path")
    if not isinstance(configured_path, str) or not configured_path:
        raise ValueError("config.geometry_audit_manifest.path is required")
    canonical_path, resolved_configured_path = _canonical_source_path(
        root, configured_path
    )
    path = (view_root.resolve() / filename).resolve()
    if resolved_configured_path != path:
        raise ValueError(
            "geometry audit manifest path does not match view_root/filename"
        )
    verification_scope = str(manifest.get("verification_scope", ""))
    if verification_scope != (
        "locks_completed_source_stream_verification_audit_manifest"
    ):
        raise ValueError("geometry audit manifest verification_scope is invalid")
    digest = str(manifest.get("sha256", ""))
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("config.geometry_audit_manifest.sha256 is invalid")
    if not path.is_file():
        raise ValueError(f"geometry audit manifest is missing: {filename}")
    manifest_bytes = path.read_bytes()
    actual = hashlib.sha256(manifest_bytes).hexdigest()
    if actual != digest:
        raise ValueError(f"geometry audit manifest SHA-256 mismatch: {filename}")
    try:
        content = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("geometry audit manifest is not valid UTF-8 JSON") from error
    if not isinstance(content, Mapping):
        raise ValueError("geometry audit manifest must contain a JSON object")
    required_view_count = int(manifest.get("required_view_count", -1))
    if int(content.get("view_count", -1)) != required_view_count:
        raise ValueError("geometry audit manifest view_count mismatch")
    resume_audit = content.get("resume_audit")
    if (
        not isinstance(resume_audit, Mapping)
        or resume_audit.get("sha256_verified") is not True
    ):
        raise ValueError(
            "geometry audit manifest resume_audit.sha256_verified must be true"
        )
    views = content.get("views")
    required_status = str(manifest.get("required_view_bundle_status", ""))
    if not isinstance(views, list) or len(views) != required_view_count:
        raise ValueError("geometry audit manifest views ledger is incomplete")
    if any(
        not isinstance(view, Mapping)
        or view.get("bundle_status") != required_status
        for view in views
    ):
        raise ValueError("geometry audit manifest contains an invalid bundle_status")
    return {
        "filename": filename,
        "path": canonical_path,
        "expected_sha256": digest,
        "actual_sha256": actual,
        "view_count": required_view_count,
        "view_count_valid": True,
        "resume_sha256_verified": True,
        "view_bundle_status": required_status,
        "all_view_bundle_statuses_valid": True,
        "verification_scope": verification_scope,
        "rehash_verified_view_shards_each_run": False,
    }


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _git_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def validate_clean_worktree(root: Path) -> dict[str, Any]:
    """Bind a formal run to HEAD by rejecting every tracked/untracked change."""

    commit = _git_commit(root)
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip():
        raise ValueError(
            "formal execution requires a clean git worktree before data open"
        )
    return {"git_commit": commit, "worktree_clean": True}


def require_repository_root(root: Path) -> Path:
    """Reject alternate clean clones whose imports still come from this checkout."""

    resolved = root.resolve()
    expected = REPOSITORY_ROOT.resolve()
    if resolved != expected:
        raise ValueError(
            f"formal --root must be the imported repository root: {expected}"
        )
    return resolved


def run_e1_test_attestation(
    root: Path,
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute the frozen E1 tests before geometry or synthetic truth is opened."""

    root = require_repository_root(root)
    paths = [root / value for value in (*E1_TEST_PATHS, *E1_IMPLEMENTATION_PATHS)]
    missing = [str(path.relative_to(root)) for path in paths if not path.is_file()]
    if missing:
        raise PDHGPreflightInvalid(
            {
                "status": PDHGPreflightInvalid.status,
                "valid": False,
                "failures": [f"missing E1 source: {value}" for value in missing],
            }
        )
    preflight = config.get("preflight")
    if not isinstance(preflight, Mapping):
        raise ValueError("frozen JSON must contain preflight")
    relative_python = str(preflight.get("e1_python_executable_relative", ""))
    if relative_python != ".venv/bin/python":
        raise ValueError("E1 Python executable must be frozen as .venv/bin/python")
    python_path = root / relative_python
    if not python_path.is_file():
        raise PDHGPreflightInvalid(
            {
                "status": PDHGPreflightInvalid.status,
                "valid": False,
                "failures": ["frozen E1 Python executable is missing"],
            }
        )
    if Path(sys.executable).resolve() != python_path.resolve():
        raise PDHGPreflightInvalid(
            {
                "status": PDHGPreflightInvalid.status,
                "valid": False,
                "failures": [
                    "formal runner and E1 tests must use the same frozen Python"
                ],
            }
        )
    expected_case_count = int(preflight.get("e1_expected_test_case_count", -1))
    expected_node_sha = str(preflight.get("e1_expected_node_sha256", ""))
    allow_skipped = preflight.get("e1_allow_skipped")
    with tempfile.TemporaryDirectory(prefix="psu-b0-pdhg-e1-") as temporary:
        junit_path = Path(temporary) / "e1_pytest.xml"
        command = [
            str(python_path),
            "-m",
            "pytest",
            "-q",
            *E1_TEST_PATHS,
            f"--junitxml={junit_path}",
        ]
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        junit_bytes = junit_path.read_bytes() if junit_path.is_file() else b""
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    node_names: list[str] = []
    junit_error: str | None = None
    try:
        junit_root = ET.fromstring(junit_bytes)
        suites = (
            [junit_root]
            if junit_root.tag == "testsuite"
            else list(junit_root.findall("testsuite"))
        )
        for suite in suites:
            for key in counts:
                counts[key] += int(suite.attrib.get(key, 0))
        node_names = sorted(
            f"{case.attrib.get('classname', '')}::{case.attrib.get('name', '')}"
            for case in junit_root.iter("testcase")
        )
    except (ET.ParseError, ValueError) as error:
        junit_error = f"{type(error).__name__}: {error}"
    node_payload = json.dumps(
        node_names,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    node_sha = hashlib.sha256(node_payload).hexdigest()
    valid = (
        completed.returncode == 0
        and junit_error is None
        and counts["tests"] == expected_case_count
        and len(node_names) == expected_case_count
        and counts["failures"] == 0
        and counts["errors"] == 0
        and (bool(allow_skipped) or counts["skipped"] == 0)
        and node_sha == expected_node_sha
    )
    audit = {
        "schema_version": "psu-b0-pdhg-e1-attestation-1.0",
        "valid": valid,
        "command": command,
        "python_executable": str(python_path),
        "return_code": int(completed.returncode),
        "elapsed_seconds": float(time.perf_counter() - started),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "junit_xml": junit_bytes.decode("utf-8", errors="replace"),
        "junit_xml_sha256": hashlib.sha256(junit_bytes).hexdigest(),
        "junit_parse_error": junit_error,
        "counts": counts,
        "expected_test_case_count": expected_case_count,
        "observed_test_case_count": len(node_names),
        "allow_skipped": bool(allow_skipped),
        "node_names": node_names,
        "node_sha256": node_sha,
        "expected_node_sha256": expected_node_sha,
        "test_files": {
            value: _sha256(root / value) for value in E1_TEST_PATHS
        },
        "implementation_files": {
            value: _sha256(root / value) for value in E1_IMPLEMENTATION_PATHS
        },
        "git_commit": _git_commit(root),
        "truth_data_opened": False,
        "geometry_opened": False,
    }
    if not audit["valid"]:
        raise PDHGPreflightInvalid(
            {
                "status": PDHGPreflightInvalid.status,
                "valid": False,
                "failures": ["frozen E1 pytest attestation failed"],
                "e1_attestation": audit,
            }
        )
    return audit


def capture_repository_snapshot(
    *,
    root: Path,
    config_path: Path,
    protocol_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash every frozen repository input for an end-of-run equality check."""

    root = require_repository_root(root)
    rows: dict[str, Path] = {
        "config": config_path.resolve(),
        "protocol": protocol_path.resolve(),
    }
    for source_key in REQUIRED_SOURCE_PATH_KEYS:
        rows[f"source:{source_key}"] = (root / str(config[source_key])).resolve()
    geometry_section = config.get("geometry_audit_manifest")
    if not isinstance(geometry_section, Mapping):
        raise ValueError("geometry_audit_manifest must be present in snapshot")
    rows["geometry_audit_manifest"] = (
        root / str(geometry_section["path"])
    ).resolve()
    amendment = _validated_amendment_metadata(config)
    if amendment["active"]:
        rows["amendment:parent_config"] = (
            root / str(amendment["parent_config_path"])
        ).resolve()
        rows["amendment:parent_protocol"] = (
            root / str(amendment["parent_protocol_path"])
        ).resolve()
        rows["amendment:parent_public_summary"] = (
            root / str(amendment["parent_public_summary_path"])
        ).resolve()
    files: dict[str, dict[str, str]] = {}
    for label, path in rows.items():
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"snapshot path escapes repository root: {label}") from error
        if not path.is_file():
            raise ValueError(f"snapshot input is missing: {relative}")
        files[label] = {"path": relative, "sha256": _sha256(path)}
    return {
        "git": validate_clean_worktree(root),
        "files": files,
    }


def require_unchanged_repository_snapshot(
    *,
    root: Path,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Recheck HEAD, worktree, and each frozen byte stream before publication."""

    root = require_repository_root(root)
    expected_files = expected.get("files")
    if not isinstance(expected_files, Mapping):
        raise ValueError("initial repository snapshot has no file ledger")
    observed_files: dict[str, dict[str, str]] = {}
    for label, raw in expected_files.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"invalid initial snapshot row: {label}")
        relative = str(raw.get("path", ""))
        path = (root / relative).resolve()
        try:
            canonical = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"snapshot path escaped repository root: {label}") from error
        if canonical != relative or not path.is_file():
            raise ValueError(f"snapshot path changed or disappeared: {label}")
        observed_files[str(label)] = {"path": relative, "sha256": _sha256(path)}
    observed = {"git": validate_clean_worktree(root), "files": observed_files}
    if observed != expected:
        raise ValueError("repository inputs or HEAD changed during formal execution")
    return observed


def _pdhg_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    section = config.get("pdhg", config.get("screen"))
    if not isinstance(section, Mapping):
        raise ValueError("frozen JSON must contain a 'pdhg' object")
    return section


def _alpha_rows(config: Mapping[str, Any]) -> list[tuple[str, str, float]]:
    raw = _pdhg_section(config).get("alphas")
    if not isinstance(raw, list):
        raise ValueError("pdhg.alphas must be a list of exact fractions")
    parsed: list[tuple[str, str, float]] = []
    for item in raw:
        if isinstance(item, str):
            fraction_text = item
            decimal = float(Fraction(item))
        elif isinstance(item, Mapping):
            fraction_text = str(item.get("fraction"))
            if "decimal" in item:
                decimal = float(item["decimal"])
            elif "value" in item:
                decimal = float(item["value"])
            else:
                raise ValueError("each alpha needs fraction and decimal")
        else:
            raise ValueError("each alpha must be a string or object")
        exact = Fraction(fraction_text)
        if not math.isclose(decimal, float(exact), rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"alpha decimal does not match {fraction_text}")
        slug = fraction_text.replace("/", "of")
        parsed.append((fraction_text, slug, decimal))
    return parsed


def _eta_stress_gate(config: Mapping[str, Any]) -> Mapping[str, Any]:
    preflight = config.get("preflight")
    if not isinstance(preflight, Mapping):
        raise ValueError("frozen JSON must contain preflight")
    gate = preflight.get("eta_stress_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("preflight.eta_stress_gate must be an object")
    return gate


def validate_frozen_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the machine-readable preregistration without opening data."""

    amendment_audit = _validated_amendment_metadata(config)
    if str(config.get("status")) != "PREREGISTERED_NOT_RUN":
        raise ValueError("frozen JSON status must be PREREGISTERED_NOT_RUN")
    replicates = tuple(int(value) for value in config.get("replicate_indices", ()))
    if replicates != OPENED_REPLICATES:
        raise ValueError(
            f"replicate_indices must be exactly {list(OPENED_REPLICATES)}"
        )
    families = tuple(str(value) for value in config.get("reaction_families", ()))
    if families and families != REACTION_FAMILIES:
        raise ValueError("reaction_families do not match the frozen protocol")

    section = _pdhg_section(config)
    penalties = tuple(str(value) for value in section.get("penalties", ()))
    iterations = tuple(
        int(value)
        for value in section.get("iterations", section.get("stages", ()))
    )
    if penalties != PDHG_PENALTIES:
        raise ValueError(f"pdhg.penalties must be {list(PDHG_PENALTIES)}")
    if iterations != PDHG_ITERATIONS:
        raise ValueError(f"pdhg.iterations must be {list(PDHG_ITERATIONS)}")
    if _alpha_rows(config) != list(PDHG_ALPHA_SPECS):
        raise ValueError("pdhg.alphas do not match the four frozen scales")
    solver = config.get("solver")
    if not isinstance(solver, Mapping):
        raise ValueError("frozen JSON must contain solver")
    if not math.isclose(float(solver.get("extrapolation", -1.0)), 1.0):
        raise ValueError("pdhg.theta must be 1")
    if not math.isclose(float(solver.get("step_safety_eta", -1.0)), 0.90):
        raise ValueError("pdhg.eta must be 0.90")
    if not math.isclose(float(solver.get("huber_delta", -1.0)), 0.5):
        raise ValueError("pdhg.huber_delta must be 0.5")
    if int(solver.get("norm_power_iterations", -1)) != 16:
        raise ValueError("norm power iterations must be 16")
    if not math.isclose(float(solver.get("norm_safety_factor", -1.0)), 1.50):
        raise ValueError("norm safety factor must be 1.50")
    if not math.isclose(float(solver.get("conservative_stress_eta", -1.0)), 0.50):
        raise ValueError("conservative stress eta must be 0.50")

    source_policy = config.get("source_sha256_policy")
    if not isinstance(source_policy, Mapping):
        raise ValueError("frozen JSON must contain source_sha256_policy")
    expected_source_policy = {
        "algorithm": "sha256_raw_file_bytes",
        "mode": "fail_closed",
        "require_exact_key_set": True,
        "verify_before_data_open": True,
        "geometry_audit_manifest_required": True,
    }
    for key, expected in expected_source_policy.items():
        if source_policy.get(key) != expected:
            raise ValueError(f"source_sha256_policy.{key} must be {expected!r}")

    geometry_manifest = config.get("geometry_audit_manifest")
    if not isinstance(geometry_manifest, Mapping):
        raise ValueError("frozen JSON must contain geometry_audit_manifest")
    filename = str(geometry_manifest.get("filename", ""))
    configured_geometry_path = str(geometry_manifest.get("path", ""))
    geometry_sha = str(geometry_manifest.get("sha256", ""))
    if not filename or Path(filename).name != filename:
        raise ValueError("geometry_audit_manifest.filename must be a plain filename")
    if not configured_geometry_path or Path(configured_geometry_path).is_absolute():
        raise ValueError("geometry_audit_manifest.path must be relative")
    if geometry_manifest.get("verification_scope") != (
        "locks_completed_source_stream_verification_audit_manifest"
    ):
        raise ValueError("geometry_audit_manifest.verification_scope is invalid")
    if len(geometry_sha) != 64 or any(
        character not in "0123456789abcdef" for character in geometry_sha
    ):
        raise ValueError("geometry_audit_manifest.sha256 must be a SHA-256 digest")
    expected_geometry_contract = {
        "required_view_count": 9,
        "required_resume_sha256_verified": True,
        "required_view_bundle_status": (
            "VIEW_SHARD_BUNDLE_TRANSCODED_AND_SOURCE_STREAMS_VERIFIED"
        ),
        "rehash_verified_view_shards_each_run": False,
    }
    for key, expected in expected_geometry_contract.items():
        if geometry_manifest.get(key) != expected:
            raise ValueError(f"geometry_audit_manifest.{key} must be {expected!r}")

    anchor_policy = config.get("runtime_anchor_policy")
    expected_anchor_policy = {
        "whitener_mode": "deterministic_refit_verified_against_prerun_anchor",
        "historical_in_memory_state_reuse_proven": False,
        "synthetic_scale_uses_clean_truth": True,
        "verify_before_observation_only_stress": True,
    }
    if not isinstance(anchor_policy, Mapping):
        raise ValueError("frozen JSON must contain runtime_anchor_policy")
    for key, expected in expected_anchor_policy.items():
        if anchor_policy.get(key) != expected:
            raise ValueError(f"runtime_anchor_policy.{key} must be {expected!r}")

    stress = _eta_stress_gate(config)
    if stress.get("truth_access_allowed") is not False:
        raise ValueError("eta stress preflight must not access truth")
    if tuple(int(value) for value in stress.get("replicate_indices", ())) != (
        OPENED_REPLICATES
    ):
        raise ValueError("eta stress replicates must be the opened pair")
    if int(stress.get("iterations", -1)) != 32:
        raise ValueError("eta stress trajectories must use 32 iterations")
    if tuple(str(value) for value in stress.get("trajectory_ids", ())) != tuple(
        STRESS_TRAJECTORY_SPECS
    ):
        raise ValueError("eta stress trajectory IDs differ from protocol")
    stress_etas = tuple(float(value) for value in stress.get("etas", ()))
    if stress_etas != (
        float(solver["step_safety_eta"]),
        float(solver["conservative_stress_eta"]),
    ):
        raise ValueError("eta stress values must consume both solver etas")
    exact_stress_values = {
        "head_iteration_first": 1,
        "head_iteration_last": 8,
        "tail_iteration_first": 25,
        "tail_iteration_last": 32,
        "window_reducer": "per_field_median",
        "fail_if_any_field_fails": True,
    }
    for key, expected in exact_stress_values.items():
        if stress.get(key) != expected:
            raise ValueError(f"eta_stress_gate.{key} must be {expected!r}")
    absolute = stress.get("absolute_explosion")
    comparative = stress.get("comparative_instability")
    if not isinstance(absolute, Mapping) or not isinstance(comparative, Mapping):
        raise ValueError("eta stress thresholds must be objects")
    if (
        float(absolute.get("maximum_relative_data_objective", math.nan)) != 100.0
        or float(
            absolute.get("maximum_tail_to_head_fixed_point_ratio", math.nan)
        )
        != 100.0
        or absolute.get("failure_operator") != "strict_greater_than"
    ):
        raise ValueError("eta stress absolute thresholds differ from protocol")
    if (
        float(
            comparative.get(
                "minimum_eta_0p90_to_0p50_tail_fixed_point_ratio",
                math.nan,
            )
        )
        != 10.0
        or float(
            comparative.get(
                "minimum_eta_0p90_to_0p50_tail_data_objective_ratio",
                math.nan,
            )
        )
        != 2.0
        or comparative.get("condition_operator") != "strict_greater_than"
        or comparative.get("require_both_conditions") is not True
    ):
        raise ValueError("eta stress comparative thresholds differ from protocol")
    preflight = config["preflight"]
    if preflight.get("failure_status") != PDHGPreflightInvalid.status:
        raise ValueError("preflight failure status must be PDHG_PREFLIGHT_INVALID")
    if tuple(str(value) for value in preflight.get("e1_pytest_files", ())) != (
        E1_TEST_PATHS
    ):
        raise ValueError("preflight.e1_pytest_files differ from the frozen suite")
    if preflight.get("e1_python_executable_relative") != ".venv/bin/python":
        raise ValueError("preflight E1 Python executable differs from protocol")
    if tuple(
        str(value) for value in preflight.get("e1_implementation_files", ())
    ) != E1_IMPLEMENTATION_PATHS:
        raise ValueError(
            "preflight.e1_implementation_files differ from the frozen suite"
        )
    if int(preflight.get("e1_expected_test_case_count", -1)) < 1:
        raise ValueError("preflight.e1_expected_test_case_count must be positive")
    e1_node_sha = str(preflight.get("e1_expected_node_sha256", ""))
    if len(e1_node_sha) != 64 or any(
        character not in "0123456789abcdef" for character in e1_node_sha
    ):
        raise ValueError("preflight.e1_expected_node_sha256 must be a SHA-256 digest")
    if preflight.get("e1_allow_skipped") is not False:
        raise ValueError("preflight.e1_allow_skipped must remain false")

    controls = config.get("controls")
    if not isinstance(controls, Mapping):
        raise ValueError("frozen JSON must contain controls")
    zero_iterations = tuple(
        int(value)
        for value in controls.get("unregularized_pdhg_iterations", ())
    )
    if zero_iterations != PDHG_ITERATIONS:
        raise ValueError("alpha=0 controls must use K=4/8/16/32")

    graph_stages = tuple(
        int(value) for value in controls.get("graph_pcgls_stages", ())
    )
    component_stages = int(controls.get("component_pcgls_stages", -1))
    if graph_stages != GRAPH_STAGES:
        raise ValueError(f"graph stages must be {list(GRAPH_STAGES)}")
    if component_stages != 4:
        raise ValueError("component baseline must be strength-3, K=4")

    timing_protocol = config.get("timing_protocol")
    expected_timing_protocol = {
        "warmup_runs_per_solver_family": 1,
        "warmup_excluded_from_ratios": True,
        "process_cold_start_claimed": False,
        "candidate_pair_order_replicate_0": "candidate_then_baseline",
        "candidate_pair_order_replicate_8": "baseline_then_candidate",
        "candidate_traversal_replicate_0": "forward",
        "candidate_traversal_replicate_8": "reverse",
        "screen_ratio_reducer": "median_of_two_adjacent_paired_ratios",
        "winner_repeats_per_replicate": 5,
        "winner_total_ab_count": 5,
        "winner_total_ba_count": 5,
        "winner_gate_ratio_reducer": "median_of_ten_adjacent_paired_ratios",
    }
    if not isinstance(timing_protocol, Mapping):
        raise ValueError("frozen JSON must contain timing_protocol")
    if dict(timing_protocol) != expected_timing_protocol:
        raise ValueError("timing_protocol differs from the frozen AB/BA design")

    historical = historical_method_specs(config)
    if tuple(row["candidate_id"] for row in historical) != HISTORICAL_SUP_IDS:
        raise ValueError("historical SupPCG references are not the frozen pair")

    exact_counts = {
        "screen.candidate_count_expected": (
            section.get("candidate_count_expected"),
            EXPECTED_FORMAL_CANDIDATES,
        ),
        "screen.metric_row_count_expected": (
            section.get("metric_row_count_expected"),
            EXPECTED_FORMAL_CANDIDATES * EXPECTED_ROWS_PER_METHOD,
        ),
        "controls.control_count_expected": (
            controls.get("control_count_expected"),
            EXPECTED_CONTROLS_AND_REFERENCES,
        ),
        "controls.method_count_expected": (
            controls.get("method_count_expected"),
            EXPECTED_METHODS,
        ),
        "controls.all_metric_row_count_expected": (
            controls.get("all_metric_row_count_expected"),
            EXPECTED_METRIC_ROWS,
        ),
    }
    for name, (observed, expected) in exact_counts.items():
        if int(observed if observed is not None else -1) != expected:
            raise ValueError(f"{name} must be {expected}")

    gates = config.get("decision_gates")
    if not isinstance(gates, Mapping):
        raise ValueError("frozen JSON must contain decision_gates")
    exact_gates = {
        "minimum_mean_field_gain_percent": 1.0,
        "minimum_field_p10_percent": -1.0,
        "maximum_harm_rate": 1.0 / 16.0,
        "minimum_worst_field_gain_percent": -3.0,
        "minimum_mean_gradient_gain_percent": 0.0,
        "minimum_mean_front_gain": 0.0,
        "minimum_front_critical_mean_gain": 0.0,
        "maximum_median_wall_time_ratio": 3.0,
    }
    for key, expected in exact_gates.items():
        if not math.isclose(
            float(gates.get(key, math.nan)),
            expected,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"decision_gates.{key} must be {expected}")
    if gates.get("require_positive_mean_in_each_replicate") is not True:
        raise ValueError("both replicate-specific means must be positive")

    claim = config.get("claim_boundary")
    if not isinstance(claim, Mapping):
        raise ValueError("frozen JSON must contain claim_boundary")
    if claim.get("synthetic_scale_by_view_uses_clean_truth") is not True:
        raise ValueError(
            "claim_boundary.synthetic_scale_by_view_uses_clean_truth must be true"
        )
    if claim.get("experimental_deployment_scale_available") is not False:
        raise ValueError(
            "claim_boundary.experimental_deployment_scale_available must be false"
        )
    for key in (
        "algorithm_superiority_claimed",
        "broader_opened_grid_authorized",
        "fresh_seed_set_opened",
        "neural_operator_training_authorized",
    ):
        if claim.get(key) is not False:
            raise ValueError(f"claim_boundary.{key} must remain false")

    return {
        "amendment": amendment_audit,
        "replicate_indices": list(replicates),
        "reaction_families": list(REACTION_FAMILIES),
        "formal_candidate_count": EXPECTED_FORMAL_CANDIDATES,
        "control_reference_count": EXPECTED_CONTROLS_AND_REFERENCES,
        "method_count": EXPECTED_METHODS,
        "metric_row_count": EXPECTED_METRIC_ROWS,
    }


def validate_runtime_anchor_manifest(
    manifest: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the pre-performance geometry and deterministic-refit anchors."""

    if manifest.get("schema_version") != (
        "psu-b0-pdhg-runtime-anchor-manifest-1.0"
    ):
        raise ValueError("runtime anchor manifest schema is invalid")
    if manifest.get("status") != "PRERUN_ANCHORS_FROZEN_BEFORE_FORMAL_SCREEN":
        raise ValueError("runtime anchor manifest status is invalid")
    boundary = manifest.get("generation_boundary")
    expected_boundary = {
        "candidate_solvers_run": False,
        "truth_performance_metrics_computed": False,
        "synthetic_scale_by_view_uses_clean_truth": True,
        "deterministic_covariance_refit_performed": True,
        "historical_in_memory_state_reuse_proven": False,
    }
    if not isinstance(boundary, Mapping):
        raise ValueError("runtime anchor generation_boundary is missing")
    for key, expected in expected_boundary.items():
        if boundary.get(key) != expected:
            raise ValueError(f"runtime anchor generation_boundary.{key} is invalid")

    geometry_anchor = manifest.get("runtime_geometry_anchor")
    if not isinstance(geometry_anchor, Mapping) or geometry_anchor.get("schema") != (
        "psu-b0-geometry-operator-anchor-1.0"
    ):
        raise ValueError("runtime geometry anchor schema is invalid")
    geometry_digest = str(geometry_anchor.get("sha256", ""))
    if len(geometry_digest) != 64 or any(
        character not in "0123456789abcdef" for character in geometry_digest
    ):
        raise ValueError("runtime geometry anchor SHA-256 is invalid")

    whitener_anchors = manifest.get("runtime_whitener_anchors")
    expected_replicates = {str(value) for value in OPENED_REPLICATES}
    if not isinstance(whitener_anchors, Mapping) or set(whitener_anchors) != (
        expected_replicates
    ):
        raise ValueError("runtime whitener anchors must exactly cover 0 and 8")
    for replicate, anchor in whitener_anchors.items():
        if not isinstance(anchor, Mapping) or anchor.get("schema") != (
            "psu-b0-whitener-anchor-1.0"
        ):
            raise ValueError(f"runtime whitener anchor {replicate} schema is invalid")
        digest = str(anchor.get("sha256", ""))
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"runtime whitener anchor {replicate} SHA-256 is invalid")

    source_rows = manifest.get("source_sha256")
    if not isinstance(source_rows, Mapping):
        raise ValueError("runtime anchor source_sha256 is missing")
    expected_sources = {
        "source_multiseed_config": str(
            config["source_sha256"]["source_multiseed_config"]
        ),
        "source_smoke_config": str(config["source_sha256"]["source_smoke_config"]),
        "source_conditioning_report": str(
            config["source_sha256"]["source_conditioning_report"]
        ),
        "geometry_audit_manifest": str(
            config["geometry_audit_manifest"]["sha256"]
        ),
    }
    if dict(source_rows) != expected_sources:
        raise ValueError("runtime anchor source hashes differ from frozen inputs")
    return {
        "valid": True,
        "geometry_sha256": geometry_digest,
        "whitener_sha256": {
            str(replicate): str(anchor["sha256"])
            for replicate, anchor in whitener_anchors.items()
        },
        "generation_boundary": dict(boundary),
    }


def pdhg_candidate_grid(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the exact 2 x 4 x 4 preregistered formal grid."""

    section = _pdhg_section(config)
    penalties = tuple(str(value) for value in section["penalties"])
    iterations = tuple(
        int(value)
        for value in section.get("iterations", section.get("stages", ()))
    )
    candidates: list[dict[str, Any]] = []
    for penalty, alpha_spec, count in itertools.product(
        penalties,
        _alpha_rows(config),
        iterations,
    ):
        fraction, slug, alpha = alpha_spec
        candidates.append(
            {
                "candidate_id": f"pdhg_{penalty}_a{slug}_k{count}",
                "method": "pdhg",
                "covariance_mode": "full_graph",
                "penalty": penalty,
                "alpha_fraction": fraction,
                "alpha": float(alpha),
                "iterations": int(count),
                "forward_calls": int(count),
                "adjoint_calls": int(count),
                "total_operator_calls": 2 * int(count),
                "is_formal_candidate": True,
            }
        )
    identifiers = [str(row["candidate_id"]) for row in candidates]
    if len(candidates) != EXPECTED_FORMAL_CANDIDATES:
        raise ValueError(
            f"PDHG grid has {len(candidates)} candidates, expected 32"
        )
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("PDHG candidate identifiers must be unique")
    expected_ids = {
        f"pdhg_{penalty}_a{slug}_k{count}"
        for penalty, (_, slug, _), count in itertools.product(
            PDHG_PENALTIES,
            PDHG_ALPHA_SPECS,
            PDHG_ITERATIONS,
        )
    }
    if set(identifiers) != expected_ids:
        raise ValueError("PDHG candidate identifiers differ from protocol")
    return candidates


def alpha_zero_controls(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    controls = config["controls"]
    iterations = controls.get("unregularized_pdhg_iterations", ())
    rows = []
    for raw in iterations:
        count = int(raw)
        rows.append(
            {
                "candidate_id": f"pdhg_data_only_k{count}",
                "method": "pdhg_data_only",
                "covariance_mode": "full_graph",
                "penalty": "tv",
                "alpha_fraction": "0",
                "alpha": 0.0,
                "iterations": count,
                "forward_calls": count,
                "adjoint_calls": count,
                "total_operator_calls": 2 * count,
                "is_formal_candidate": False,
            }
        )
    if tuple(row["iterations"] for row in rows) != PDHG_ITERATIONS:
        raise ValueError("data-only controls differ from frozen K grid")
    return rows


def graph_pcgls_candidates(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    stages = [
        int(value) for value in config["controls"]["graph_pcgls_stages"]
    ]
    rows = [
        {
            "candidate_id": f"graph_s3_k{count}",
            "method": "graph_pcgls",
            "covariance_mode": "full_graph",
            "penalty": "none",
            "alpha_fraction": "",
            "alpha": 0.0,
            "iterations": count,
            "forward_calls": count,
            "adjoint_calls": count,
            "total_operator_calls": 2 * count,
            "is_formal_candidate": False,
        }
        for count in stages
    ]
    if tuple(stages) != GRAPH_STAGES:
        raise ValueError("graph-PCGLS checkpoints differ from protocol")
    return rows


def component_pcgls_candidate(config: Mapping[str, Any]) -> dict[str, Any]:
    count = int(config["controls"]["component_pcgls_stages"])
    if count != 4:
        raise ValueError("component anchor must use K=4")
    return {
        "candidate_id": COMPONENT_ID,
        "method": "component_pcgls",
        "covariance_mode": "component_iid",
        "penalty": "none",
        "alpha_fraction": "",
        "alpha": 0.0,
        "iterations": count,
        "forward_calls": count,
        "adjoint_calls": count,
        "total_operator_calls": 2 * count,
        "is_formal_candidate": False,
    }


def historical_method_specs(config: Mapping[str, Any]) -> list[dict[str, str]]:
    controls = config.get("controls")
    if not isinstance(controls, Mapping):
        raise ValueError("frozen JSON must contain controls")
    identifiers = tuple(
        str(value)
        for value in controls.get(
            "historical_superiorization_candidate_ids", ()
        )
    )
    source_rows = (
        (
            "source_superiorization_rows",
            str(config.get("source_superiorization_rows", "")),
        ),
        (
            "source_superiorization_tail_rows",
            str(config.get("source_superiorization_tail_rows", "")),
        ),
    )
    source_sha256 = config.get("source_sha256", {})
    if not isinstance(source_sha256, Mapping):
        source_sha256 = {}
    rows = [
        {
            "candidate_id": identifier,
            "metric_rows_csv": source,
            "sha256": str(source_sha256.get(source_key, "")),
        }
        for identifier, (source_key, source) in zip(
            identifiers, source_rows, strict=True
        )
    ]
    if any(not row["metric_rows_csv"] for row in rows):
        raise ValueError("frozen JSON must name both SupPCG source CSVs")
    if tuple(row["candidate_id"] for row in rows) != HISTORICAL_SUP_IDS:
        raise ValueError("historical SupPCG entries must keep protocol order")
    return rows


def all_method_ids(config: Mapping[str, Any]) -> list[str]:
    methods = [row["candidate_id"] for row in pdhg_candidate_grid(config)]
    methods.extend(row["candidate_id"] for row in alpha_zero_controls(config))
    methods.extend(row["candidate_id"] for row in graph_pcgls_candidates(config))
    methods.append(component_pcgls_candidate(config)["candidate_id"])
    methods.extend(row["candidate_id"] for row in historical_method_specs(config))
    if len(methods) != EXPECTED_METHODS or len(set(methods)) != EXPECTED_METHODS:
        raise ValueError("method ledger must contain 49 unique identifiers")
    return methods


def audit_metric_row_counts(
    rows: Sequence[Mapping[str, Any]],
    *,
    method_ids: Sequence[str],
    replicate_indices: Sequence[int] = OPENED_REPLICATES,
    reaction_families: Sequence[str] = REACTION_FAMILIES,
) -> dict[str, Any]:
    """Return an explicit completeness audit; no metric values are imputed."""

    expected = {
        (str(method), int(replicate), str(family))
        for method, replicate, family in itertools.product(
            method_ids,
            replicate_indices,
            reaction_families,
        )
    }
    observed_list = [
        (
            str(row.get("candidate_id", "")),
            int(row.get("replicate", -1)),
            str(row.get("reaction_family", "")),
        )
        for row in rows
    ]
    counts = Counter(observed_list)
    observed = set(observed_list)
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    per_method = Counter(key[0] for key in observed_list)
    wrong_method_counts = {
        method: int(per_method.get(method, 0))
        for method in method_ids
        if int(per_method.get(method, 0)) != (
            len(replicate_indices) * len(reaction_families)
        )
    }
    valid = not duplicates and not missing and not unexpected
    valid = valid and not wrong_method_counts and len(rows) == len(expected)
    return {
        "valid": bool(valid),
        "expected_method_count": len(method_ids),
        "observed_method_count": len({key[0] for key in observed_list}),
        "expected_row_count": len(expected),
        "observed_row_count": len(rows),
        "missing_row_count": len(missing),
        "unexpected_row_count": len(unexpected),
        "duplicate_key_count": len(duplicates),
        "wrong_method_row_counts": wrong_method_counts,
        "missing_examples": [list(key) for key in missing[:8]],
        "unexpected_examples": [list(key) for key in unexpected[:8]],
        "duplicate_examples": [list(key) for key in duplicates[:8]],
    }


def validate_metric_row_counts(
    rows: Sequence[Mapping[str, Any]],
    *,
    method_ids: Sequence[str],
    replicate_indices: Sequence[int] = OPENED_REPLICATES,
    reaction_families: Sequence[str] = REACTION_FAMILIES,
) -> dict[str, Any]:
    audit = audit_metric_row_counts(
        rows,
        method_ids=method_ids,
        replicate_indices=replicate_indices,
        reaction_families=reaction_families,
    )
    if not audit["valid"]:
        raise ValueError(f"metric row ledger is incomplete: {audit}")
    return audit


def audit_pdhg_performance_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit every formal row before any comparative gain or ranking is computed."""

    selected = [row for row in rows if str(row.get("method")) == "pdhg"]
    failures: list[dict[str, Any]] = []
    finite_fields = (
        "field_relative_l2",
        "gradient_relative_l2",
        "front_top10_f1",
        "whitened_data_objective",
        "regularization_penalty",
        "total_objective",
        "primal_update_norm",
        "dual_feasibility_violation",
        "solver_elapsed_seconds",
        "output_amplitude_scale",
    )
    for row in selected:
        key = {
            "candidate_id": str(row.get("candidate_id", "")),
            "replicate": int(row.get("replicate", -1)),
            "reaction_family": str(row.get("reaction_family", "")),
        }
        reasons: list[str] = []
        values: dict[str, float] = {}
        for field in finite_fields:
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError):
                reasons.append(f"{field}_missing_or_non_numeric")
                continue
            if not math.isfinite(value):
                reasons.append(f"{field}_non_finite")
            values[field] = value
        if str(row.get("execution_status")) != "ok":
            reasons.append("execution_status_not_ok")
        if row.get("is_finite") is not True:
            reasons.append("is_finite_not_true")
        if row.get("support_gauge_ok") is not True:
            reasons.append("support_gauge_not_true")
        if int(row.get("evaluation_forward_calls", -1)) != 1:
            reasons.append("checkpoint_scorer_forward_count_not_one")
        iterations = int(row.get("iterations", -1))
        if (
            int(row.get("forward_calls", -1)) != iterations
            or int(row.get("adjoint_calls", -1)) != iterations
            or int(row.get("total_operator_calls", -1)) != 2 * iterations
        ):
            reasons.append("logical_call_budget_mismatch")
        if values.get("field_relative_l2", -1.0) < 0.0:
            reasons.append("negative_field_error")
        if values.get("gradient_relative_l2", -1.0) < 0.0:
            reasons.append("negative_gradient_error")
        front = values.get("front_top10_f1", math.nan)
        if math.isfinite(front) and not 0.0 <= front <= 1.0:
            reasons.append("front_f1_out_of_range")
        for field in (
            "whitened_data_objective",
            "regularization_penalty",
            "total_objective",
            "primal_update_norm",
        ):
            if values.get(field, -1.0) < 0.0:
                reasons.append(f"{field}_negative")
        dual = values.get("dual_feasibility_violation", math.inf)
        if dual < 0.0 or dual > STRESS_DUAL_FEASIBILITY_TOLERANCE:
            reasons.append("dual_feasibility_violation_exceeds_tolerance")
        if values.get("solver_elapsed_seconds", 0.0) <= 0.0:
            reasons.append("solver_elapsed_not_positive")
        if values.get("output_amplitude_scale", 0.0) <= 0.0:
            reasons.append("output_amplitude_scale_not_positive")
        if all(
            field in values
            for field in (
                "whitened_data_objective",
                "regularization_penalty",
                "total_objective",
            )
        ) and not math.isclose(
            values["total_objective"],
            values["whitened_data_objective"]
            + values["regularization_penalty"],
            rel_tol=1e-5,
            abs_tol=1e-7,
        ):
            reasons.append("total_objective_identity_mismatch")
        if reasons:
            failures.append({**key, "reasons": reasons})
    expected = EXPECTED_FORMAL_CANDIDATES * EXPECTED_ROWS_PER_METHOD
    if len(selected) != expected:
        failures.append(
            {
                "candidate_id": "__ledger__",
                "replicate": -1,
                "reaction_family": "",
                "reasons": [
                    f"formal_row_count_{len(selected)}_does_not_equal_{expected}"
                ],
            }
        )
    return {
        "valid": not failures,
        "expected_row_count": expected,
        "observed_row_count": len(selected),
        "failure_count": len(failures),
        "failure_examples": failures[:16],
    }


def validate_pdhg_performance_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    audit = audit_pdhg_performance_rows(rows)
    if not audit["valid"]:
        raise ValueError(f"formal PDHG row validity audit failed: {audit}")
    return audit


def graph_budget_frontier(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_iterations: Sequence[int] = PDHG_ITERATIONS,
    graph_stages: Sequence[int] = GRAPH_STAGES,
    expected_fields_per_method: int | None = EXPECTED_ROWS_PER_METHOD,
) -> dict[int, dict[str, Any]]:
    """Choose one pooled graph stage per PDHG budget, never a field oracle."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        identifier = str(row.get("candidate_id", ""))
        if identifier.startswith("graph_s3_k"):
            grouped.setdefault(identifier, []).append(row)
    summaries: list[dict[str, Any]] = []
    for raw_stage in graph_stages:
        stage = int(raw_stage)
        identifier = f"graph_s3_k{stage}"
        selected = grouped.get(identifier, [])
        if not selected:
            raise ValueError(f"missing graph checkpoint {identifier}")
        if (
            expected_fields_per_method is not None
            and len(selected) != int(expected_fields_per_method)
        ):
            raise ValueError(
                f"{identifier} has {len(selected)} fields, expected "
                f"{expected_fields_per_method}"
            )
        values = np.asarray(
            [float(row["field_relative_l2"]) for row in selected],
            dtype=np.float64,
        )
        if not bool(np.all(np.isfinite(values))):
            raise ValueError(f"{identifier} has non-finite field error")
        summaries.append(
            {
                "candidate_id": identifier,
                "stage": stage,
                "pooled_mean_field_relative_l2": float(np.mean(values)),
                "field_count": len(values),
            }
        )
    frontier: dict[int, dict[str, Any]] = {}
    for raw_count in candidate_iterations:
        count = int(raw_count)
        eligible = [row for row in summaries if int(row["stage"]) <= count]
        if not eligible:
            raise ValueError(f"no graph checkpoint fits PDHG K={count}")
        winner = min(
            eligible,
            key=lambda row: (
                float(row["pooled_mean_field_relative_l2"]),
                int(row["stage"]),
                str(row["candidate_id"]),
            ),
        )
        frontier[count] = dict(winner)
    return frontier


def _gain_percent(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / max(reference, 1e-12)


def add_pdhg_comparative_gains(
    rows: Sequence[Mapping[str, Any]],
    *,
    frontier: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Attach paired gains to PDHG candidates using globally fixed references."""

    by_key = {
        (
            int(row["replicate"]),
            str(row["reaction_family"]),
            str(row["candidate_id"]),
        ): row
        for row in rows
    }
    output: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        if str(row.get("method")) != "pdhg":
            output.append(row)
            continue
        replicate = int(row["replicate"])
        family = str(row["reaction_family"])
        count = int(row["iterations"])
        frontier_id = str(frontier[count]["candidate_id"])
        same_stage_id = f"graph_s3_k{count}"
        data_only_id = f"pdhg_data_only_k{count}"
        reference_ids = {
            "graph_budget_frontier": frontier_id,
            "graph_same_stage": same_stage_id,
            "component": COMPONENT_ID,
            "data_only": data_only_id,
        }
        row["graph_budget_frontier_id"] = frontier_id
        row["graph_same_stage_id"] = same_stage_id
        row["data_only_control_id"] = data_only_id
        if (
            str(row.get("execution_status", "ok")) != "ok"
            or row.get("field_relative_l2") is None
            or row.get("gradient_relative_l2") is None
            or row.get("front_top10_f1") is None
        ):
            for label in reference_ids:
                row[f"field_gain_vs_{label}_percent"] = None
                row[f"gradient_gain_vs_{label}_percent"] = None
                row[f"front_gain_vs_{label}"] = None
            output.append(row)
            continue
        references: dict[str, Mapping[str, Any]] = {}
        for label, identifier in reference_ids.items():
            key = (replicate, family, identifier)
            if key not in by_key:
                raise ValueError(f"missing paired reference row {key}")
            references[label] = by_key[key]
        if any(
            reference.get(metric) is None
            for reference in references.values()
            for metric in (
                "field_relative_l2",
                "gradient_relative_l2",
                "front_top10_f1",
            )
        ):
            row["execution_status"] = "paired_reference_failed"
            for label in reference_ids:
                row[f"field_gain_vs_{label}_percent"] = None
                row[f"gradient_gain_vs_{label}_percent"] = None
                row[f"front_gain_vs_{label}"] = None
            output.append(row)
            continue
        for label, reference in references.items():
            row[f"field_gain_vs_{label}_percent"] = _gain_percent(
                float(reference["field_relative_l2"]),
                float(row["field_relative_l2"]),
            )
            row[f"gradient_gain_vs_{label}_percent"] = _gain_percent(
                float(reference["gradient_relative_l2"]),
                float(row["gradient_relative_l2"]),
            )
            row[f"front_gain_vs_{label}"] = (
                float(row["front_top10_f1"])
                - float(reference["front_top10_f1"])
            )
        output.append(row)
    return output


def _linear_p10(values: Sequence[float]) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), 0.10, method="linear"))


def summarize_pdhg_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    wall_time_ratios: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if str(row.get("method")) == "pdhg":
            grouped.setdefault(str(row["candidate_id"]), []).append(row)
    if len(grouped) != EXPECTED_FORMAL_CANDIDATES:
        raise ValueError(
            f"formal PDHG summary saw {len(grouped)} candidates, expected 32"
        )
    ratios = wall_time_ratios or {}
    summaries: list[dict[str, Any]] = []
    required_gain_keys = (
        "field_gain_vs_graph_budget_frontier_percent",
        "gradient_gain_vs_graph_budget_frontier_percent",
        "front_gain_vs_graph_budget_frontier",
    )
    for identifier, selected in sorted(grouped.items()):
        if len(selected) != EXPECTED_ROWS_PER_METHOD:
            raise ValueError(
                f"{identifier} has {len(selected)} rows, expected 16"
            )
        invalid_rows = 0
        for row in selected:
            if str(row.get("execution_status", "ok")) != "ok":
                invalid_rows += 1
                continue
            try:
                values = [float(row[key]) for key in required_gain_keys]
            except (KeyError, TypeError, ValueError):
                invalid_rows += 1
                continue
            if not all(math.isfinite(value) for value in values):
                invalid_rows += 1
        first = selected[0]
        raw_ratio = ratios.get(identifier)
        timing_valid = raw_ratio is not None and math.isfinite(float(raw_ratio))
        summary: dict[str, Any] = {
            "candidate_id": identifier,
            "method": "pdhg",
            "penalty": str(first["penalty"]),
            "alpha_fraction": str(first["alpha_fraction"]),
            "alpha": float(first["alpha"]),
            "iterations": int(first["iterations"]),
            "forward_calls": int(first["forward_calls"]),
            "adjoint_calls": int(first["adjoint_calls"]),
            "total_operator_calls": int(first["total_operator_calls"]),
            "field_count": len(selected),
            "replicate_count": len({int(row["replicate"]) for row in selected}),
            "invalid_row_count": invalid_rows,
            "ranking_valid": invalid_rows == 0 and timing_valid,
            "median_wall_time_ratio_vs_graph_frontier": (
                float(raw_ratio) if timing_valid else None
            ),
        }
        if invalid_rows or not timing_valid:
            summaries.append(summary)
            continue
        field = [
            float(row["field_gain_vs_graph_budget_frontier_percent"])
            for row in selected
        ]
        gradient = [
            float(row["gradient_gain_vs_graph_budget_frontier_percent"])
            for row in selected
        ]
        front = [
            float(row["front_gain_vs_graph_budget_frontier"])
            for row in selected
        ]
        critical = [
            float(row["front_gain_vs_graph_budget_frontier"])
            for row in selected
            if str(row["reaction_family"]) in FRONT_CRITICAL_FAMILIES
        ]
        replicate_means = {
            replicate: float(
                np.mean(
                    [
                        float(
                            row[
                                "field_gain_vs_graph_budget_frontier_percent"
                            ]
                        )
                        for row in selected
                        if int(row["replicate"]) == replicate
                    ]
                )
            )
            for replicate in OPENED_REPLICATES
        }
        replicate_worst = {
            replicate: float(
                np.min(
                    [
                        float(
                            row[
                                "field_gain_vs_graph_budget_frontier_percent"
                            ]
                        )
                        for row in selected
                        if int(row["replicate"]) == replicate
                    ]
                )
            )
            for replicate in OPENED_REPLICATES
        }
        morphology = {}
        for family in REACTION_FAMILIES:
            family_rows = [
                row for row in selected if str(row["reaction_family"]) == family
            ]
            morphology[family] = {
                "mean_field_gain_percent": float(
                    np.mean(
                        [
                            float(
                                row[
                                    "field_gain_vs_graph_budget_frontier_percent"
                                ]
                            )
                            for row in family_rows
                        ]
                    )
                ),
                "mean_gradient_gain_percent": float(
                    np.mean(
                        [
                            float(
                                row[
                                    "gradient_gain_vs_graph_budget_frontier_percent"
                                ]
                            )
                            for row in family_rows
                        ]
                    )
                ),
                "mean_front_gain": float(
                    np.mean(
                        [
                            float(row["front_gain_vs_graph_budget_frontier"])
                            for row in family_rows
                        ]
                    )
                ),
            }
        harm_count = int(np.sum(np.asarray(field) < -1.0))
        summary.update(
            {
                "graph_budget_frontier_id": str(
                    first["graph_budget_frontier_id"]
                ),
                "mean_field_gain_vs_graph_budget_frontier_percent": float(
                    np.mean(field)
                ),
                "field_gain_vs_graph_budget_frontier_p10_percent": _linear_p10(
                    field
                ),
                "field_harm_vs_graph_budget_frontier_over_one_percent_count": harm_count,
                "field_harm_vs_graph_budget_frontier_over_one_percent_rate": float(
                    harm_count / len(field)
                ),
                "worst_field_gain_vs_graph_budget_frontier_percent": float(
                    np.min(field)
                ),
                "mean_gradient_gain_vs_graph_budget_frontier_percent": float(
                    np.mean(gradient)
                ),
                "gradient_gain_vs_graph_budget_frontier_p10_percent": _linear_p10(
                    gradient
                ),
                "worst_gradient_gain_vs_graph_budget_frontier_percent": float(
                    np.min(gradient)
                ),
                "mean_front_gain_vs_graph_budget_frontier": float(np.mean(front)),
                "front_gain_vs_graph_budget_frontier_p10": _linear_p10(front),
                "worst_front_gain_vs_graph_budget_frontier": float(np.min(front)),
                "front_critical_mean_front_gain_vs_graph_budget_frontier": float(
                    np.mean(critical)
                ),
                "replicate_0_mean_field_gain_percent": replicate_means[0],
                "replicate_0_worst_field_gain_percent": replicate_worst[0],
                "replicate_8_mean_field_gain_percent": replicate_means[8],
                "replicate_8_worst_field_gain_percent": replicate_worst[8],
                "mean_field_gain_vs_graph_same_stage_percent": float(
                    np.mean(
                        [
                            float(row["field_gain_vs_graph_same_stage_percent"])
                            for row in selected
                        ]
                    )
                ),
                "mean_field_gain_vs_component_percent": float(
                    np.mean(
                        [
                            float(row["field_gain_vs_component_percent"])
                            for row in selected
                        ]
                    )
                ),
                "mean_field_gain_vs_data_only_percent": float(
                    np.mean(
                        [
                            float(row["field_gain_vs_data_only_percent"])
                            for row in selected
                        ]
                    )
                ),
                "morphology_gain_json": json.dumps(
                    morphology,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
        summaries.append(summary)
    return summaries


def rank_pdhg_candidates(
    summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(summaries) != EXPECTED_FORMAL_CANDIDATES:
        raise ValueError("ranking requires all 32 PDHG candidates")
    identifiers = [str(row["candidate_id"]) for row in summaries]
    if len(set(identifiers)) != EXPECTED_FORMAL_CANDIDATES:
        raise ValueError("ranking candidate IDs must be unique")

    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        if not bool(row.get("ranking_valid", False)):
            return (1, str(row["candidate_id"]))
        return (
            0,
            -float(row["mean_field_gain_vs_graph_budget_frontier_percent"]),
            -float(row["field_gain_vs_graph_budget_frontier_p10_percent"]),
            float(
                row[
                    "field_harm_vs_graph_budget_frontier_over_one_percent_rate"
                ]
            ),
            -float(row["worst_field_gain_vs_graph_budget_frontier_percent"]),
            -float(row["mean_gradient_gain_vs_graph_budget_frontier_percent"]),
            -float(row["mean_front_gain_vs_graph_budget_frontier"]),
            float(row["median_wall_time_ratio_vs_graph_frontier"]),
            float(row["alpha"]),
            str(row["candidate_id"]),
        )

    ranked = [dict(row) for row in sorted(summaries, key=key)]
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return ranked


def pdhg_screen_decision(
    ranking: Sequence[Mapping[str, Any]],
    *,
    count_audit: Mapping[str, Any] | None = None,
    winner_timing_ratio: float | None = None,
) -> dict[str, Any]:
    """Apply the preregistered one-winner gate without fallback selection."""

    if len(ranking) != EXPECTED_FORMAL_CANDIDATES:
        raise ValueError("decision requires the complete 32-candidate ranking")
    invalid_candidates = sum(
        not bool(row.get("ranking_valid", False)) for row in ranking
    )
    counts_valid = count_audit is None or bool(count_audit.get("valid", False))
    if not counts_valid or invalid_candidates:
        return {
            "status": "PDHG_PREFLIGHT_INVALID",
            "winner_candidate_id": None,
            "invalid_candidate_count": int(invalid_candidates),
            "count_audit_valid": bool(counts_valid),
            "all_accuracy_gates_pass": False,
            "all_structure_gates_pass": False,
            "runtime_gate_pass": False,
            "fresh_authorized": False,
            "neural_training_authorized": False,
        }

    winner = ranking[0]
    ratio = (
        float(winner_timing_ratio)
        if winner_timing_ratio is not None
        else float(winner["median_wall_time_ratio_vs_graph_frontier"])
    )
    gates = {
        "mean_field_gain_at_least_one_percent": float(
            winner["mean_field_gain_vs_graph_budget_frontier_percent"]
        )
        >= 1.0,
        "field_p10_at_least_minus_one_percent": float(
            winner["field_gain_vs_graph_budget_frontier_p10_percent"]
        )
        >= -1.0,
        "field_harm_rate_at_most_one_of_sixteen": float(
            winner[
                "field_harm_vs_graph_budget_frontier_over_one_percent_rate"
            ]
        )
        <= 1.0 / 16.0,
        "worst_field_gain_at_least_minus_three_percent": float(
            winner["worst_field_gain_vs_graph_budget_frontier_percent"]
        )
        >= -3.0,
        "mean_gradient_gain_nonnegative": float(
            winner["mean_gradient_gain_vs_graph_budget_frontier_percent"]
        )
        >= 0.0,
        "mean_front_gain_nonnegative": float(
            winner["mean_front_gain_vs_graph_budget_frontier"]
        )
        >= 0.0,
        "front_critical_mean_nonnegative": float(
            winner[
                "front_critical_mean_front_gain_vs_graph_budget_frontier"
            ]
        )
        >= 0.0,
        "median_wall_time_ratio_at_most_three": math.isfinite(ratio)
        and ratio <= 3.0,
        "replicate_0_mean_positive": float(
            winner["replicate_0_mean_field_gain_percent"]
        )
        > 0.0,
        "replicate_8_mean_positive": float(
            winner["replicate_8_mean_field_gain_percent"]
        )
        > 0.0,
    }
    accuracy_names = (
        "mean_field_gain_at_least_one_percent",
        "field_p10_at_least_minus_one_percent",
        "field_harm_rate_at_most_one_of_sixteen",
        "worst_field_gain_at_least_minus_three_percent",
        "replicate_0_mean_positive",
        "replicate_8_mean_positive",
    )
    structure_names = (
        "mean_gradient_gain_nonnegative",
        "mean_front_gain_nonnegative",
        "front_critical_mean_nonnegative",
    )
    accuracy_pass = all(gates[name] for name in accuracy_names)
    structure_pass = all(gates[name] for name in structure_names)
    runtime_pass = gates["median_wall_time_ratio_at_most_three"]
    if not accuracy_pass:
        status = "POSTOPEN_PDHG_SCALE_NO_GO"
    elif not structure_pass:
        status = "POSTOPEN_PDHG_STRUCTURE_NO_GO"
    elif not runtime_pass:
        status = "POSTOPEN_PDHG_RUNTIME_NO_GO"
    else:
        status = "POSTOPEN_PDHG_SCALE_SIGNAL_ONLY"
    return {
        "status": status,
        "winner_candidate_id": str(winner["candidate_id"]),
        "winner_rank": int(winner.get("rank", 1)),
        "winner_median_wall_time_ratio_vs_graph_frontier": ratio,
        "gates": gates,
        "all_accuracy_gates_pass": bool(accuracy_pass),
        "all_structure_gates_pass": bool(structure_pass),
        "runtime_gate_pass": bool(runtime_pass),
        "invalid_candidate_count": 0,
        "count_audit_valid": bool(counts_valid),
        "evidence_level": "E2_POSTOPEN_TWO_REPLICATE_SCALE_SMOKE",
        "fresh_authorized": False,
        "neural_training_authorized": False,
    }


def _support_ok(volume: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    outside = (1.0 - support)[None, None].to(volume)
    maximum = torch.amax(torch.abs(volume * outside).flatten(1), dim=1)
    return maximum <= 1e-6


def regularization_vector_count(support: torch.Tensor) -> int:
    """Count potential forward-Neumann vectors, including support interfaces."""

    count = int(torch.count_nonzero(regularization_site_mask(support)).item())
    if count <= 0:
        raise ValueError("regularization vector count must be positive")
    return count


def _base_metric_rows(
    *,
    replicate: int,
    candidate: Mapping[str, Any],
    families: Sequence[str],
    prediction: torch.Tensor,
    truth: torch.Tensor,
    data_objective: torch.Tensor | None,
    regularization_penalty: torch.Tensor | None,
    total_objective: torch.Tensor | None,
    primal_update_norm: torch.Tensor | None,
    dual_feasibility_violation: torch.Tensor | None,
    solver_elapsed_seconds: float,
    support_gauge_ok: torch.Tensor | None,
    execution_status: str = "ok",
    source_evidence: str = "recomputed",
    evaluation_forward_calls: int = 0,
    output_amplitude_scale: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    metrics = _field_metrics(prediction, truth)
    rows = []
    for index, family in enumerate(families):
        finite = bool(
            torch.isfinite(metrics["field_relative_l2"][index])
            and torch.isfinite(metrics["gradient_relative_l2"][index])
            and torch.isfinite(metrics["front_top10_f1"][index])
        )
        row = {
            "replicate": int(replicate),
            "sample_index": int(index),
            "reaction_family": str(family),
            **dict(candidate),
            "field_relative_l2": float(metrics["field_relative_l2"][index]),
            "gradient_relative_l2": float(
                metrics["gradient_relative_l2"][index]
            ),
            "front_top10_f1": float(metrics["front_top10_f1"][index]),
            "whitened_data_objective": (
                None
                if data_objective is None
                else float(data_objective[index])
            ),
            "regularization_penalty": (
                None
                if regularization_penalty is None
                else float(regularization_penalty[index])
            ),
            "total_objective": (
                None
                if total_objective is None
                else float(total_objective[index])
            ),
            "primal_update_norm": (
                None
                if primal_update_norm is None
                else float(primal_update_norm[index])
            ),
            "dual_feasibility_violation": (
                None
                if dual_feasibility_violation is None
                else float(dual_feasibility_violation[index])
            ),
            "solver_elapsed_seconds": float(solver_elapsed_seconds),
            "evaluation_forward_calls": int(evaluation_forward_calls),
            "is_finite": finite,
            "support_gauge_ok": (
                None
                if support_gauge_ok is None
                else bool(support_gauge_ok[index])
            ),
            "execution_status": execution_status if finite else "non_finite",
            "source_evidence": source_evidence,
            "output_amplitude_scale": (
                None
                if output_amplitude_scale is None
                else float(output_amplitude_scale[index])
            ),
        }
        rows.append(row)
    return rows


def _failed_metric_rows(
    *,
    replicate: int,
    candidate: Mapping[str, Any],
    families: Sequence[str],
    error: BaseException,
) -> list[dict[str, Any]]:
    return [
        {
            "replicate": int(replicate),
            "sample_index": int(index),
            "reaction_family": str(family),
            **dict(candidate),
            "field_relative_l2": None,
            "gradient_relative_l2": None,
            "front_top10_f1": None,
            "whitened_data_objective": None,
            "regularization_penalty": None,
            "total_objective": None,
            "primal_update_norm": None,
            "dual_feasibility_violation": None,
            "solver_elapsed_seconds": None,
            "evaluation_forward_calls": 0,
            "is_finite": False,
            "support_gauge_ok": False,
            "execution_status": "solver_failed",
            "failure_type": type(error).__name__,
            "failure_message": str(error),
            "source_evidence": "recomputed_failure_retained",
            "output_amplitude_scale": None,
        }
        for index, family in enumerate(families)
    ]


def _read_historical_sup_rows(
    *,
    root: Path,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    output: list[dict[str, Any]] = []
    manifest: list[dict[str, str]] = []
    for spec in historical_method_specs(config):
        path = root / spec["metric_rows_csv"]
        if not path.is_file():
            raise ValueError(f"historical SupPCG CSV is missing: {path}")
        actual_sha = _sha256(path)
        if spec["sha256"] and actual_sha != spec["sha256"]:
            raise ValueError(f"historical SupPCG checksum mismatch: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            selected = [
                row
                for row in csv.DictReader(handle)
                if row["candidate_id"] == spec["candidate_id"]
            ]
        if len(selected) != EXPECTED_ROWS_PER_METHOD:
            raise ValueError(
                f"{spec['candidate_id']} has {len(selected)} historical rows"
            )
        for raw in selected:
            output.append(
                {
                    "replicate": int(raw["replicate"]),
                    "sample_index": int(raw["sample_index"]),
                    "reaction_family": str(raw["reaction_family"]),
                    "candidate_id": str(raw["candidate_id"]),
                    "method": "superiorized_pcgls",
                    "covariance_mode": "full_graph",
                    "penalty": str(raw["penalty"]),
                    "alpha_fraction": "",
                    "alpha": 0.0,
                    "iterations": int(raw["stages"]),
                    "forward_calls": int(raw["forward_calls"]),
                    "adjoint_calls": int(raw["adjoint_calls"]),
                    "total_operator_calls": int(raw["total_operator_calls"]),
                    "is_formal_candidate": False,
                    "field_relative_l2": float(raw["field_relative_l2"]),
                    "gradient_relative_l2": float(
                        raw["gradient_relative_l2"]
                    ),
                    "front_top10_f1": float(raw["front_top10_f1"]),
                    "whitened_data_objective": None,
                    "regularization_penalty": None,
                    "total_objective": None,
                    "primal_update_norm": None,
                    "dual_feasibility_violation": None,
                    "solver_elapsed_seconds": float(
                        raw["solver_elapsed_seconds"]
                    ),
                    "evaluation_forward_calls": 0,
                    "is_finite": True,
                    "support_gauge_ok": None,
                    "execution_status": "historical_metrics_only",
                    "source_evidence": "frozen_sup_pcg_csv",
                    "output_amplitude_scale": None,
                }
            )
        manifest.append(
            {
                "candidate_id": spec["candidate_id"],
                "path": str(path.relative_to(root)),
                "sha256": actual_sha,
            }
        )
    return output, manifest


def _method_summaries(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["candidate_id"]), []).append(row)
    output = []
    for identifier, selected in sorted(grouped.items()):
        valid = [
            row
            for row in selected
            if row.get("field_relative_l2") is not None
            and math.isfinite(float(row["field_relative_l2"]))
        ]
        first = selected[0]
        output.append(
            {
                "candidate_id": identifier,
                "method": str(first["method"]),
                "penalty": str(first["penalty"]),
                "alpha_fraction": str(first["alpha_fraction"]),
                "alpha": float(first["alpha"]),
                "iterations": int(first["iterations"]),
                "forward_calls": int(first["forward_calls"]),
                "adjoint_calls": int(first["adjoint_calls"]),
                "total_operator_calls": int(first["total_operator_calls"]),
                "row_count": len(selected),
                "valid_metric_row_count": len(valid),
                "failed_metric_row_count": len(selected) - len(valid),
                "mean_field_relative_l2": (
                    None
                    if not valid
                    else float(
                        np.mean(
                            [float(row["field_relative_l2"]) for row in valid]
                        )
                    )
                ),
                "mean_gradient_relative_l2": (
                    None
                    if not valid
                    else float(
                        np.mean(
                            [
                                float(row["gradient_relative_l2"])
                                for row in valid
                            ]
                        )
                    )
                ),
                "mean_front_top10_f1": (
                    None
                    if not valid
                    else float(
                        np.mean([float(row["front_top10_f1"]) for row in valid])
                    )
                ),
                "source_evidence": str(first["source_evidence"]),
            }
        )
    if len(output) != EXPECTED_METHODS:
        raise ValueError("method summary must contain exactly 49 methods")
    return output


def _median_iqr(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median_seconds": float(np.median(array)),
        "q25_seconds": float(np.quantile(array, 0.25, method="linear")),
        "q75_seconds": float(np.quantile(array, 0.75, method="linear")),
        "iqr_seconds": float(
            np.quantile(array, 0.75, method="linear")
            - np.quantile(array, 0.25, method="linear")
        ),
    }


def _cpu_float64_numpy(value: torch.Tensor) -> np.ndarray:
    """Move off accelerators before casting to CPU-only float64."""

    return value.detach().to(device="cpu").to(dtype=torch.float64).numpy()


def _run_pdhg_stress_trajectory(
    *,
    replicate: int,
    trajectory_id: str,
    penalty: str,
    alpha: float,
    eta_label: str,
    eta: float,
    iterations: int,
    head_iteration_first: int,
    head_iteration_last: int,
    tail_iteration_first: int,
    tail_iteration_last: int,
    operator: WhitenedMeasurementOperator,
    observation_b0: torch.Tensor,
    ones_sigma: torch.Tensor,
    ones_mask: torch.Tensor,
    rays_per_view: int,
    measurement_scale: float,
    regularization_vector_count: int,
    norm_estimate: BlockOperatorNormEstimate,
    theta: float,
    huber_delta: float,
    device: torch.device,
) -> dict[str, Any]:
    """Run one truth-free observable stress trajectory and audit every call."""

    root_data = math.sqrt(float(norm_estimate.data_norm_squared_upper))
    root_edge = math.sqrt(float(norm_estimate.gradient_norm_squared_upper))
    sigma_data = float(eta) / root_data
    sigma_edge = float(eta) / root_edge
    tau = float(eta) / (root_data + root_edge)
    regularization_weight = float(alpha) / float(regularization_vector_count)
    expected_calls = {
        "forward_calls": int(iterations),
        "adjoint_calls": int(iterations),
    }

    operator.reset_call_counts()
    _synchronize(device)
    started = time.perf_counter()
    try:
        result = primal_dual_reconstruction(
            operator,
            observation_b0,
            sigma_by_view=ones_sigma,
            view_mask=ones_mask,
            rays_per_view=rays_per_view,
            iterations=int(iterations),
            regularization_weight=regularization_weight,
            penalty=penalty,
            primal_step=tau,
            data_dual_step=sigma_data,
            edge_dual_step=sigma_edge,
            norm_estimate=norm_estimate,
            huber_delta=float(huber_delta),
            measurement_scale=float(measurement_scale),
            extrapolation=float(theta),
        )
        data_history = torch.stack(
            [
                row["extrapolated_relative_data_objective"]
                for row in result.history
            ]
        )
        fixed_point_history = torch.stack(
            [row["normalized_joint_fixed_point_residual"] for row in result.history]
        )
        edge_dual_history = torch.stack(
            [row["edge_dual_maximum_norm"] for row in result.history]
        )
        observables = torch.cat(
            (
                data_history.flatten(),
                fixed_point_history.flatten(),
                edge_dual_history.flatten(),
                result.volume.flatten(),
                result.data_dual.flatten(),
                result.edge_dual.flatten(),
            )
        )
        is_finite = bool(torch.all(torch.isfinite(observables)))
        data_values = _cpu_float64_numpy(data_history)
        fixed_values = _cpu_float64_numpy(fixed_point_history)
        dual_violation = torch.amax(
            torch.clamp(
                edge_dual_history - regularization_weight,
                min=0.0,
            ),
            dim=0,
        )
        dual_violation = _cpu_float64_numpy(dual_violation)
        support_gauge_ok = _support_ok(
            result.volume, operator.support
        ).detach().to("cpu").numpy()
        alpha_zero_edge_dual_exact_zero = torch.all(
            (result.edge_dual == 0.0).flatten(1),
            dim=1,
        ) & torch.all(edge_dual_history == 0.0, dim=0)
        alpha_zero_edge_dual_exact_zero = (
            alpha_zero_edge_dual_exact_zero.detach().to("cpu").numpy()
        )
        head = fixed_values[
            int(head_iteration_first) - 1 : int(head_iteration_last)
        ]
        tail_fixed = fixed_values[
            int(tail_iteration_first) - 1 : int(tail_iteration_last)
        ]
        tail_data = data_values[
            int(tail_iteration_first) - 1 : int(tail_iteration_last)
        ]
        head_fixed_median = np.median(head, axis=0)
        tail_fixed_median = np.median(tail_fixed, axis=0)
        tail_data_median = np.median(tail_data, axis=0)
        tail_to_head = tail_fixed_median / np.maximum(
            head_fixed_median, 1e-12
        )
        field_observations = [
            {
                "sample_index": int(index),
                "maximum_relative_data_objective": float(
                    np.max(data_values[:, index])
                ),
                "head_fixed_point_median": float(head_fixed_median[index]),
                "tail_fixed_point_median": float(tail_fixed_median[index]),
                "tail_data_objective_median": float(tail_data_median[index]),
                "tail_to_head_fixed_point_ratio": float(tail_to_head[index]),
                "maximum_dual_feasibility_violation": float(
                    dual_violation[index]
                ),
                "support_gauge_ok": bool(support_gauge_ok[index]),
                "alpha_zero_edge_dual_exact_zero": (
                    bool(alpha_zero_edge_dual_exact_zero[index])
                    if float(alpha) == 0.0
                    else None
                ),
            }
            for index in range(len(observation_b0))
        ]
        is_finite = is_finite and all(
            math.isfinite(float(value))
            for row in field_observations
            for key, value in row.items()
            if key not in {
                "sample_index",
                "support_gauge_ok",
                "alpha_zero_edge_dual_exact_zero",
            }
        )
        step_contract_value = float(result.step_contract_value)
        observed_gradient_calls = int(result.gradient_calls)
        observed_gradient_adjoint_calls = int(result.gradient_adjoint_calls)
        failure_type = None
        failure_message = None
        execution_status = "ok" if is_finite else "non_finite"
    except (FloatingPointError, RuntimeError, TypeError, ValueError) as error:
        is_finite = False
        field_observations = []
        step_contract_value = None
        observed_gradient_calls = 0
        observed_gradient_adjoint_calls = 0
        failure_type = type(error).__name__
        failure_message = str(error)
        execution_status = "solver_failed"
    _synchronize(device)
    elapsed = time.perf_counter() - started
    observed_calls = operator.call_report()
    return {
        "replicate": int(replicate),
        "trajectory_id": str(trajectory_id),
        "penalty": str(penalty),
        "eta_label": str(eta_label),
        "eta": float(eta),
        "iterations": int(iterations),
        "alpha": float(alpha),
        "truth_used": False,
        "sample_count": int(len(observation_b0)),
        "is_finite": is_finite,
        "execution_status": execution_status,
        "field_observations": field_observations,
        "step_contract_value": step_contract_value,
        "solver_forward_calls": int(iterations),
        "solver_adjoint_calls": int(iterations),
        "solver_gradient_calls": int(iterations),
        "solver_gradient_adjoint_calls": int(iterations),
        "history_scorer_forward_calls": 0,
        "observed_physical_forward_calls": int(
            observed_calls["forward_calls"]
        ),
        "observed_physical_adjoint_calls": int(
            observed_calls["adjoint_calls"]
        ),
        "observed_gradient_calls": observed_gradient_calls,
        "observed_gradient_adjoint_calls": observed_gradient_adjoint_calls,
        "call_ledger_valid": (
            observed_calls == expected_calls
            and observed_gradient_calls == int(iterations)
            and observed_gradient_adjoint_calls == int(iterations)
        ),
        "elapsed_seconds": float(elapsed),
        "failure_type": failure_type,
        "failure_message": failure_message,
    }


def audit_pdhg_stability_preflight(
    rows: Sequence[Mapping[str, Any]],
    *,
    gate: Mapping[str, Any],
    primary_eta: float,
    conservative_eta: float,
    replicate_indices: Sequence[int] = OPENED_REPLICATES,
) -> dict[str, Any]:
    """Apply the frozen truth-free finite/call/relative-instability gate."""

    trajectory_ids = tuple(str(value) for value in gate["trajectory_ids"])
    iterations = int(gate["iterations"])
    absolute = gate["absolute_explosion"]
    comparative = gate["comparative_instability"]
    maximum_data = float(absolute["maximum_relative_data_objective"])
    maximum_tail_to_head = float(
        absolute["maximum_tail_to_head_fixed_point_ratio"]
    )
    minimum_fixed_ratio = float(
        comparative["minimum_eta_0p90_to_0p50_tail_fixed_point_ratio"]
    )
    minimum_data_ratio = float(
        comparative["minimum_eta_0p90_to_0p50_tail_data_objective_ratio"]
    )
    expected_keys = {
        (int(replicate), trajectory_id, eta_label)
        for replicate in replicate_indices
        for trajectory_id in trajectory_ids
        for eta_label in ("conservative", "primary")
    }
    counts = Counter(
        (
            int(row.get("replicate", -1)),
            str(row.get("trajectory_id", "")),
            str(row.get("eta_label", "")),
        )
        for row in rows
    )
    observed_keys = set(counts)
    failures: list[str] = []
    missing = sorted(expected_keys - observed_keys)
    unexpected = sorted(observed_keys - expected_keys)
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    if missing:
        failures.append(f"missing stress trajectories: {missing}")
    if unexpected:
        failures.append(f"unexpected stress trajectories: {unexpected}")
    if duplicates:
        failures.append(f"duplicate stress trajectories: {duplicates}")

    indexed = {
        (
            int(row["replicate"]),
            str(row["trajectory_id"]),
            str(row["eta_label"]),
        ): row
        for row in rows
    }
    for key in sorted(expected_keys & observed_keys):
        row = indexed[key]
        expected_eta = (
            float(conservative_eta)
            if key[2] == "conservative"
            else float(primary_eta)
        )
        if not math.isclose(
            float(row.get("eta", math.nan)),
            expected_eta,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            failures.append(f"stress eta mismatch for {key}")
        if row.get("is_finite") is not True:
            failures.append(f"non-finite or failed stress trajectory: {key}")
        if row.get("truth_used") is not False:
            failures.append(f"stress trajectory accessed truth: {key}")
        if row.get("call_ledger_valid") is not True:
            failures.append(f"stress call ledger mismatch: {key}")
        if int(row.get("iterations", -1)) != iterations:
            failures.append(f"stress iteration mismatch: {key}")
        expected_penalty, expected_alpha = STRESS_TRAJECTORY_SPECS[key[1]]
        if str(row.get("penalty")) != expected_penalty or not math.isclose(
            float(row.get("alpha", math.nan)),
            expected_alpha,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            failures.append(f"stress trajectory specification mismatch: {key}")
        field_rows = row.get("field_observations")
        if not isinstance(field_rows, list) or len(field_rows) != int(
            row.get("sample_count", -1)
        ):
            failures.append(f"stress field ledger is incomplete: {key}")
            continue
        sample_indices = [int(field["sample_index"]) for field in field_rows]
        if sample_indices != list(range(len(field_rows))):
            failures.append(f"stress field indices are not exact: {key}")
        for field in field_rows:
            sample = int(field["sample_index"])
            metric_values = {
                name: field.get(name)
                for name in (
                    "maximum_relative_data_objective",
                    "head_fixed_point_median",
                    "tail_fixed_point_median",
                    "tail_data_objective_median",
                    "tail_to_head_fixed_point_ratio",
                    "maximum_dual_feasibility_violation",
                )
            }
            if any(
                value is None or not math.isfinite(float(value))
                for value in metric_values.values()
            ):
                failures.append(f"non-finite stress field metrics: {key + (sample,)}")
                continue
            if (
                float(metric_values["maximum_relative_data_objective"])
                > maximum_data
            ):
                failures.append(
                    "absolute relative-data-objective explosion: "
                    f"{key + (sample,)}"
                )
            if (
                float(metric_values["tail_to_head_fixed_point_ratio"])
                > maximum_tail_to_head
            ):
                failures.append(
                    "absolute fixed-point tail/head explosion: "
                    f"{key + (sample,)}"
                )
            if field.get("support_gauge_ok") is not True:
                failures.append(f"support/gauge violation: {key + (sample,)}")
            if (
                float(metric_values["maximum_dual_feasibility_violation"])
                > STRESS_DUAL_FEASIBILITY_TOLERANCE
            ):
                failures.append(f"dual feasibility violation: {key + (sample,)}")
            if (
                str(row.get("trajectory_id")) == "pdhg_data_only_k32"
                and field.get("alpha_zero_edge_dual_exact_zero") is not True
            ):
                failures.append(
                    f"alpha=0 edge dual is not exactly zero: {key + (sample,)}"
                )

    comparisons: list[dict[str, Any]] = []
    for replicate in replicate_indices:
        for trajectory_id in trajectory_ids:
            primary = indexed.get((int(replicate), trajectory_id, "primary"))
            conservative = indexed.get(
                (int(replicate), trajectory_id, "conservative")
            )
            if primary is None or conservative is None:
                continue
            primary_fields = {
                int(row["sample_index"]): row
                for row in primary.get("field_observations", ())
            }
            conservative_fields = {
                int(row["sample_index"]): row
                for row in conservative.get("field_observations", ())
            }
            for sample in sorted(set(primary_fields) & set(conservative_fields)):
                primary_field = primary_fields[sample]
                conservative_field = conservative_fields[sample]
                fixed_ratio = float(
                    primary_field["tail_fixed_point_median"]
                ) / max(
                    float(conservative_field["tail_fixed_point_median"]),
                    1e-12,
                )
                data_ratio = float(
                    primary_field["tail_data_objective_median"]
                ) / max(
                    float(conservative_field["tail_data_objective_median"]),
                    1e-12,
                )
                unstable = (
                    fixed_ratio > minimum_fixed_ratio
                    and data_ratio > minimum_data_ratio
                )
                comparison = {
                    "replicate": int(replicate),
                    "trajectory_id": trajectory_id,
                    "sample_index": sample,
                    "primary_eta": float(primary_eta),
                    "conservative_eta": float(conservative_eta),
                    "tail_fixed_point_ratio": fixed_ratio,
                    "tail_data_objective_ratio": data_ratio,
                    "minimum_fixed_point_ratio_for_failure": minimum_fixed_ratio,
                    "minimum_data_objective_ratio_for_failure": minimum_data_ratio,
                    "both_failure_conditions_required": True,
                    "valid": not unstable,
                }
                comparisons.append(comparison)
                if unstable:
                    failures.append(
                        "primary eta is relatively unstable for "
                        f"replicate={replicate}, trajectory={trajectory_id}, "
                        f"sample={sample}"
                    )

    failures = list(dict.fromkeys(failures))
    expected_run_count = len(expected_keys)
    call_ledger = {
        "run_count": len(rows),
        "expected_run_count": expected_run_count,
        "solver_forward_calls": sum(
            int(row.get("solver_forward_calls", 0)) for row in rows
        ),
        "solver_adjoint_calls": sum(
            int(row.get("solver_adjoint_calls", 0)) for row in rows
        ),
        "history_scorer_forward_calls": sum(
            int(row.get("history_scorer_forward_calls", 0)) for row in rows
        ),
        "observed_physical_forward_calls": sum(
            int(row.get("observed_physical_forward_calls", 0)) for row in rows
        ),
        "observed_physical_adjoint_calls": sum(
            int(row.get("observed_physical_adjoint_calls", 0)) for row in rows
        ),
        "observed_gradient_calls": sum(
            int(row.get("observed_gradient_calls", 0)) for row in rows
        ),
        "observed_gradient_adjoint_calls": sum(
            int(row.get("observed_gradient_adjoint_calls", 0)) for row in rows
        ),
        "expected_solver_forward_calls": expected_run_count * iterations,
        "expected_solver_adjoint_calls": expected_run_count * iterations,
        "expected_gradient_calls": expected_run_count * iterations,
        "expected_gradient_adjoint_calls": expected_run_count * iterations,
        "expected_history_scorer_forward_calls": 0,
        "norm_setup_calls_included": False,
    }
    valid = not failures
    return {
        "status": (
            "PDHG_PREFLIGHT_VALID" if valid else PDHGPreflightInvalid.status
        ),
        "valid": valid,
        "truth_used": False,
        "criteria": {
            **dict(gate),
            "primary_eta": float(primary_eta),
            "conservative_eta": float(conservative_eta),
            "all_observables_must_be_finite": True,
            "solver_calls_per_run": {
                "forward_calls": iterations,
                "adjoint_calls": iterations,
                "gradient_calls": iterations,
                "gradient_adjoint_calls": iterations,
            },
            "history_extra_forward_calls_per_run": 0,
            "fixed_point_history_key": (
                "normalized_joint_fixed_point_residual"
            ),
            "ratio_denominator_floor": 1e-12,
            "support_and_gauge_required": True,
            "alpha_zero_edge_dual_exact_zero_required": True,
            "dual_feasibility_tolerance": (
                STRESS_DUAL_FEASIBILITY_TOLERANCE
            ),
        },
        "failures": failures,
        "comparisons": comparisons,
        "call_ledger": call_ledger,
        "runs": [dict(row) for row in rows],
    }


def run_pdhg_stability_preflight(
    *,
    contexts: Sequence[PDHGStressContext],
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Consume both configured etas before any truth-based performance solve."""

    solver = config.get("solver")
    if not isinstance(solver, Mapping):
        raise ValueError("frozen JSON must contain solver")
    gate = _eta_stress_gate(config)
    primary_eta = float(solver["step_safety_eta"])
    conservative_eta = float(solver["conservative_stress_eta"])
    rows: list[dict[str, Any]] = []
    for context in contexts:
        if type(context) is not PDHGStressContext:
            raise TypeError(
                "stability preflight accepts only truth-inaccessible "
                "PDHGStressContext instances"
            )
        for trajectory_id, (penalty, alpha) in STRESS_TRAJECTORY_SPECS.items():
            for eta_label, eta in (
                ("conservative", conservative_eta),
                ("primary", primary_eta),
            ):
                rows.append(
                    _run_pdhg_stress_trajectory(
                        replicate=int(context.replicate),
                        trajectory_id=str(trajectory_id),
                        penalty=penalty,
                        alpha=alpha,
                        eta_label=eta_label,
                        eta=eta,
                        iterations=int(gate["iterations"]),
                        head_iteration_first=int(gate["head_iteration_first"]),
                        head_iteration_last=int(gate["head_iteration_last"]),
                        tail_iteration_first=int(gate["tail_iteration_first"]),
                        tail_iteration_last=int(gate["tail_iteration_last"]),
                        operator=context.operator,
                        observation_b0=context.observation_b0,
                        ones_sigma=context.sigma_by_view,
                        ones_mask=context.view_mask,
                        rays_per_view=int(context.rays_per_view),
                        measurement_scale=float(context.measurement_scale),
                        regularization_vector_count=int(
                            context.regularization_vector_count
                        ),
                        norm_estimate=context.norm_estimate,
                        theta=float(solver["extrapolation"]),
                        huber_delta=float(solver["huber_delta"]),
                        device=device,
                    )
                )
    return audit_pdhg_stability_preflight(
        rows,
        gate=gate,
        primary_eta=primary_eta,
        conservative_eta=conservative_eta,
        replicate_indices=[int(context.replicate) for context in contexts],
    )


def require_valid_pdhg_stability_preflight(audit: Mapping[str, Any]) -> None:
    if audit.get("valid") is not True:
        raise PDHGPreflightInvalid(audit)


def _run_pdhg_trajectory(
    *,
    operator: WhitenedMeasurementOperator,
    observation_b0: torch.Tensor,
    truth: torch.Tensor,
    output_scale: torch.Tensor,
    trajectory_candidates: Sequence[Mapping[str, Any]],
    ones_sigma: torch.Tensor,
    ones_mask: torch.Tensor,
    rays_per_view: int,
    measurement_scale: float,
    regularization_vector_count: int,
    norm_estimate: BlockOperatorNormEstimate,
    eta: float,
    theta: float,
    huber_delta: float,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any], Any]:
    if not trajectory_candidates:
        raise ValueError("PDHG trajectory needs at least one checkpoint candidate")
    penalties = {str(row["penalty"]) for row in trajectory_candidates}
    alphas = {float(row["alpha"]) for row in trajectory_candidates}
    if len(penalties) != 1 or len(alphas) != 1:
        raise ValueError("one PDHG trajectory must share penalty and alpha")
    checkpoints = tuple(
        sorted({int(row["iterations"]) for row in trajectory_candidates})
    )
    if checkpoints[-1] != max(checkpoints):
        raise ValueError("invalid checkpoint sequence")
    maximum_iterations = checkpoints[-1]
    penalty = next(iter(penalties))
    alpha = next(iter(alphas))
    root_data = math.sqrt(float(norm_estimate.data_norm_squared_upper))
    root_edge = math.sqrt(float(norm_estimate.gradient_norm_squared_upper))
    sigma_data = float(eta) / root_data
    sigma_edge = float(eta) / root_edge
    tau = float(eta) / (root_data + root_edge)
    regularization_weight = alpha / float(
        regularization_vector_count
    )
    operator.reset_call_counts()
    _synchronize(device)
    started = time.perf_counter()
    result = primal_dual_reconstruction(
        operator,
        observation_b0,
        sigma_by_view=ones_sigma,
        view_mask=ones_mask,
        rays_per_view=rays_per_view,
        iterations=maximum_iterations,
        regularization_weight=regularization_weight,
        penalty=penalty,
        primal_step=tau,
        data_dual_step=sigma_data,
        edge_dual_step=sigma_edge,
        norm_estimate=norm_estimate,
        huber_delta=float(huber_delta),
        measurement_scale=float(measurement_scale),
        extrapolation=float(theta),
        checkpoint_iterations=checkpoints,
    )
    _synchronize(device)
    solver_elapsed = time.perf_counter() - started
    expected_calls = {
        "forward_calls": maximum_iterations,
        "adjoint_calls": maximum_iterations,
    }
    if operator.call_report() != expected_calls:
        raise ValueError(
            f"PDHG trajectory {penalty}/{alpha:g} violated max-K call budget"
        )
    if (
        result.gradient_calls != maximum_iterations
        or result.gradient_adjoint_calls != maximum_iterations
    ):
        raise ValueError(
            f"PDHG trajectory {penalty}/{alpha:g} violated local D/D^T budget"
        )
    if set(result.checkpoint_volumes) != set(checkpoints):
        raise ValueError("PDHG core did not return every frozen checkpoint")

    rows: list[dict[str, Any]] = []
    for candidate in sorted(
        trajectory_candidates,
        key=lambda row: int(row["iterations"]),
    ):
        count = int(candidate["iterations"])
        volume = result.checkpoint_volumes[count]
        with torch.no_grad():
            projected = measurement_scale * operator(volume)
        residual = projected - observation_b0
        data_objective = 0.5 * torch.sum(residual.square(), dim=(1, 2))
        edge_penalty = isotropic_edge_penalty(
            volume,
            spacing_xyz=tuple(float(value) for value in operator.spacing_xyz),
            penalty=str(candidate["penalty"]),
            huber_delta=float(huber_delta),
        )
        regularization_penalty = regularization_weight * edge_penalty
        total_objective = data_objective + regularization_penalty
        dual_violation = torch.clamp(
            result.history[count - 1]["edge_dual_maximum_norm"]
            - regularization_weight,
            min=0.0,
        )
        prediction = volume * output_scale[:, None, None, None, None]
        checkpoint_rows = _base_metric_rows(
            replicate=-1,
            candidate=candidate,
            families=REACTION_FAMILIES,
            prediction=prediction,
            truth=truth,
            data_objective=data_objective,
            regularization_penalty=regularization_penalty,
            total_objective=total_objective,
            primal_update_norm=result.history[count - 1]["primal_update_norm"],
            dual_feasibility_violation=dual_violation,
            solver_elapsed_seconds=solver_elapsed,
            support_gauge_ok=_support_ok(volume, operator.support),
            evaluation_forward_calls=1,
            output_amplitude_scale=output_scale,
        )
        for row in checkpoint_rows:
            row["shared_trajectory_max_iterations"] = maximum_iterations
            row["shared_trajectory_elapsed_seconds"] = float(solver_elapsed)
            row["solver_elapsed_source"] = (
                "shared_prefix_trajectory_total_not_candidate_timing"
            )
            row["physical_solver_forward_calls_for_trajectory"] = (
                maximum_iterations
            )
            row["physical_solver_adjoint_calls_for_trajectory"] = (
                maximum_iterations
            )
        rows.extend(checkpoint_rows)
    evaluation_report = operator.call_report()
    if evaluation_report != {
        "forward_calls": expected_calls["forward_calls"] + len(checkpoints),
        "adjoint_calls": expected_calls["adjoint_calls"],
    }:
        raise ValueError("checkpoint scoring forwards were not separately counted")
    logical_forward = sum(int(row["iterations"]) for row in trajectory_candidates)
    logical_adjoint = logical_forward
    ledger = {
        "trajectory_id": (
            f"pdhg_{penalty}_a"
            f"{str(trajectory_candidates[0]['alpha_fraction']).replace('/', 'of')}"
        ),
        "candidate_ids": [
            str(row["candidate_id"]) for row in trajectory_candidates
        ],
        "checkpoint_iterations": list(checkpoints),
        "logical_solver_forward_calls_if_independent": logical_forward,
        "logical_solver_adjoint_calls_if_independent": logical_adjoint,
        "logical_solver_gradient_calls_if_independent": logical_forward,
        "logical_solver_gradient_adjoint_calls_if_independent": logical_adjoint,
        "physical_solver_forward_calls": maximum_iterations,
        "physical_solver_adjoint_calls": maximum_iterations,
        "physical_solver_gradient_calls": maximum_iterations,
        "physical_solver_gradient_adjoint_calls": maximum_iterations,
        "prefix_forward_calls_saved": logical_forward - maximum_iterations,
        "prefix_adjoint_calls_saved": logical_adjoint - maximum_iterations,
        "prefix_gradient_calls_saved": logical_forward - maximum_iterations,
        "prefix_gradient_adjoint_calls_saved": logical_adjoint - maximum_iterations,
        "checkpoint_metric_forward_calls": len(checkpoints),
        "checkpoint_metric_adjoint_calls": 0,
        "checkpoint_metric_gradient_calls": len(checkpoints),
        "checkpoint_metric_gradient_adjoint_calls": 0,
        "solver_elapsed_seconds": float(solver_elapsed),
        "regularization_vector_count": int(regularization_vector_count),
        "regularization_site_count": int(regularization_vector_count),
        "regularization_count_definition": (
            "count_nonzero(regularization_site_mask(support))"
        ),
        "regularization_weight": float(regularization_weight),
        "step_contract_value": float(result.step_contract_value),
        "trajectory_prefix_reused": len(checkpoints) > 1,
    }
    return rows, ledger, result


def _timed_pdhg_solve(
    *,
    operator: WhitenedMeasurementOperator,
    observation_b0: torch.Tensor,
    candidate: Mapping[str, Any],
    ones_sigma: torch.Tensor,
    ones_mask: torch.Tensor,
    rays_per_view: int,
    measurement_scale: float,
    regularization_vector_count: int,
    norm_estimate: BlockOperatorNormEstimate,
    eta: float,
    theta: float,
    huber_delta: float,
    device: torch.device,
) -> tuple[float, Any, dict[str, Any]]:
    """Time one independent K deployment without truth or metric scoring."""

    count = int(candidate["iterations"])
    alpha = float(candidate["alpha"])
    root_data = math.sqrt(float(norm_estimate.data_norm_squared_upper))
    root_edge = math.sqrt(float(norm_estimate.gradient_norm_squared_upper))
    sigma_data = float(eta) / root_data
    sigma_edge = float(eta) / root_edge
    tau = float(eta) / (root_data + root_edge)
    operator.reset_call_counts()
    _synchronize(device)
    started = time.perf_counter()
    result = primal_dual_reconstruction(
        operator,
        observation_b0,
        sigma_by_view=ones_sigma,
        view_mask=ones_mask,
        rays_per_view=rays_per_view,
        iterations=count,
        regularization_weight=(
            alpha / float(regularization_vector_count)
        ),
        penalty=str(candidate["penalty"]),
        primal_step=tau,
        data_dual_step=sigma_data,
        edge_dual_step=sigma_edge,
        norm_estimate=norm_estimate,
        huber_delta=float(huber_delta),
        measurement_scale=float(measurement_scale),
        extrapolation=float(theta),
    )
    _synchronize(device)
    elapsed = time.perf_counter() - started
    expected_calls = {"forward_calls": count, "adjoint_calls": count}
    if operator.call_report() != expected_calls:
        raise ValueError("independent PDHG timing violated physical call budget")
    if result.gradient_calls != count or result.gradient_adjoint_calls != count:
        raise ValueError("independent PDHG timing violated local D/D^T budget")
    ledger = {
        "candidate_id": str(candidate["candidate_id"]),
        "iterations": count,
        "solver_forward_calls": count,
        "solver_adjoint_calls": count,
        "solver_gradient_calls": count,
        "solver_gradient_adjoint_calls": count,
        "metric_forward_calls": 0,
        "metric_adjoint_calls": 0,
        "metric_gradient_calls": 0,
        "metric_gradient_adjoint_calls": 0,
        "truth_used": False,
        "precision_metrics_used": False,
        "elapsed_seconds": float(elapsed),
    }
    return elapsed, result, ledger


def _timed_graph_solve(
    *,
    operator: WhitenedMeasurementOperator,
    prepared: torch.Tensor,
    ones_sigma: torch.Tensor,
    ones_mask: torch.Tensor,
    rays_per_view: int,
    stages: int,
    direction: GeneralizedSobolevDirection,
    device: torch.device,
) -> tuple[float, Any]:
    operator.reset_call_counts()
    _synchronize(device)
    started = time.perf_counter()
    result = preconditioned_cgls_reconstruction(
        operator,
        prepared,
        sigma_by_view=ones_sigma,
        view_mask=ones_mask,
        rays_per_view=rays_per_view,
        stages=int(stages),
        preconditioner=direction,
    )
    _synchronize(device)
    elapsed = time.perf_counter() - started
    if operator.call_report() != {
        "forward_calls": int(stages),
        "adjoint_calls": int(stages),
    }:
        raise ValueError("independent graph timing violated call budget")
    return elapsed, result


def independent_pdhg_wall_time_ratios(
    candidates: Sequence[Mapping[str, Any]],
    *,
    frontier: Mapping[int, Mapping[str, Any]],
    pdhg_timing: Mapping[tuple[int, str], float],
    graph_timing: Mapping[tuple[int, str], float],
    replicate_indices: Sequence[int] = OPENED_REPLICATES,
) -> dict[str, float]:
    """Pair each candidate's own K timing with its independent graph frontier."""

    output: dict[str, float] = {}
    for candidate in candidates:
        identifier = str(candidate["candidate_id"])
        frontier_id = str(frontier[int(candidate["iterations"])]["candidate_id"])
        ratios: list[float] = []
        for replicate in replicate_indices:
            candidate_key = (int(replicate), identifier)
            graph_key = (int(replicate), frontier_id)
            if candidate_key not in pdhg_timing or graph_key not in graph_timing:
                raise ValueError(
                    f"independent timing ledger is incomplete for {identifier}"
                )
            candidate_seconds = float(pdhg_timing[candidate_key])
            graph_seconds = float(graph_timing[graph_key])
            if (
                not math.isfinite(candidate_seconds)
                or not math.isfinite(graph_seconds)
                or candidate_seconds <= 0.0
                or graph_seconds <= 0.0
            ):
                raise ValueError(
                    f"independent timing ledger is invalid for {identifier}"
                )
            ratios.append(candidate_seconds / graph_seconds)
        output[identifier] = float(statistics.median(ratios))
    return output


def paired_pdhg_wall_time_ratios(
    candidates: Sequence[Mapping[str, Any]],
    *,
    paired_timing: Mapping[tuple[int, str], Mapping[str, Any]],
    replicate_indices: Sequence[int] = OPENED_REPLICATES,
) -> dict[str, float]:
    """Use the median of adjacent candidate/baseline ratios, not ratio of medians."""

    output: dict[str, float] = {}
    for candidate in candidates:
        identifier = str(candidate["candidate_id"])
        ratios: list[float] = []
        orders: set[str] = set()
        for replicate in replicate_indices:
            key = (int(replicate), identifier)
            if key not in paired_timing:
                raise ValueError(f"paired timing ledger is incomplete for {identifier}")
            row = paired_timing[key]
            candidate_seconds = float(row["candidate_seconds"])
            baseline_seconds = float(row["baseline_seconds"])
            order = str(row["order"])
            if (
                not math.isfinite(candidate_seconds)
                or not math.isfinite(baseline_seconds)
                or candidate_seconds <= 0.0
                or baseline_seconds <= 0.0
            ):
                raise ValueError(f"paired timing ledger is invalid for {identifier}")
            if order not in {"candidate_then_baseline", "baseline_then_candidate"}:
                raise ValueError(f"paired timing order is invalid for {identifier}")
            ratios.append(candidate_seconds / baseline_seconds)
            orders.add(order)
        if len(replicate_indices) == 2 and orders != {
            "candidate_then_baseline",
            "baseline_then_candidate",
        }:
            raise ValueError(
                f"paired timing must counterbalance AB/BA for {identifier}"
            )
        output[identifier] = float(statistics.median(ratios))
    return output


def _build_replicate_context(
    *,
    root: Path,
    config: Mapping[str, Any],
    runtime_anchor_manifest: Mapping[str, Any],
    smoke: Mapping[str, Any],
    multiseed: Mapping[str, Any],
    replicate: int,
    operator: Any,
    geometry: Mapping[str, Any],
    eigenpairs: Sequence[tuple[torch.Tensor, torch.Tensor]],
    graph: Any,
    graph_family_index: int,
    device: torch.device,
) -> dict[str, Any]:
    derived = _derive_smoke_config(smoke, multiseed, replicate=replicate)
    geometry_config = smoke["geometry"]
    view_count = int(geometry_config["view_count"])
    rays_per_view = int(geometry_config["rays_per_view"])
    grid_size = int(geometry_config["grid_size"])
    families = [str(value) for value in smoke["data"]["reaction_families"]]
    if tuple(families) != REACTION_FAMILIES:
        raise ValueError("source smoke reaction families differ from protocol")
    field_seeds = [
        int(derived["data"]["field_seed_start"]) + index
        for index in range(len(families))
    ]
    truth = reaction_morphology_batch(
        grid_size=grid_size,
        families=families,
        seeds=field_seeds,
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        clean = operator(truth).detach()
    clean_by_view = clean.reshape(
        len(truth),
        view_count,
        rays_per_view,
        2,
    )
    view_rms = torch.sqrt(
        torch.mean(clean_by_view.square(), dim=(2, 3)).clamp_min(1e-20)
    )
    floor = (
        float(derived["data"]["minimum_view_signal_fraction"])
        * torch.mean(view_rms, dim=1, keepdim=True)
    )
    levels = tuple(
        float(value) for value in derived["data"]["relative_noise_levels"]
    )
    relative_noise = torch.as_tensor(
        [levels[index % len(levels)] for index in range(len(truth))],
        dtype=torch.float32,
        device=device,
    )
    view_factors = torch.as_tensor(
        derived["data"]["view_noise_factors"],
        dtype=torch.float32,
        device=device,
    )
    scale_by_view = (
        relative_noise[:, None]
        * torch.maximum(view_rms, floor)
        * view_factors[None]
    ).clamp_min(1e-8)
    fits, gate_rows = _calibrate(
        family="graph_heat",
        family_index=graph_family_index,
        config=derived,
        graph=graph,
        eigenpairs=eigenpairs,
    )
    unit_noise = _flowon_unit_noise(
        family="graph_heat",
        family_index=graph_family_index,
        sample_count=len(truth),
        config=derived,
        graph=graph,
        eigenpairs=eigenpairs,
    ).to(device)
    observation = clean + unit_noise * scale_by_view.repeat_interleave(
        rays_per_view,
        dim=1,
    )[:, :, None]
    eigenvectors = [vectors for _, vectors in eigenpairs]
    scale_by_view_cpu = scale_by_view.detach().to("cpu")
    component_whitening = DetectorCovarianceWhitening(
        fits["component_iid"],
        eigenvectors_by_view=eigenvectors,
        scale_by_view=scale_by_view_cpu,
        predictive_mean_correction=True,
        dtype=torch.float32,
    )
    graph_whitening = DetectorCovarianceWhitening(
        fits["dg_covgate"],
        eigenvectors_by_view=eigenvectors,
        scale_by_view=scale_by_view_cpu,
        predictive_mean_correction=True,
        dtype=torch.float32,
    )
    observed_whitener_anchor = whitener_anchor(
        scale_by_view_cpu,
        component_whitening,
        graph_whitening,
        gate_rows,
        {
            "replicate": int(replicate),
            "calibration_seed": int(
                derived["covariance_calibration"]["base_seed"]
            ),
            "field_seed_start": int(derived["data"]["field_seed_start"]),
            "field_seeds": field_seeds,
            "flowon_noise_seed": int(derived["data"]["flowon_noise_seed"]),
            "graph_family_index": int(graph_family_index),
            "noise_family": "graph_heat",
        },
    )
    expected_whitener_anchor = runtime_anchor_manifest[
        "runtime_whitener_anchors"
    ][str(replicate)]
    verify_exact_anchor(
        expected_whitener_anchor,
        observed_whitener_anchor,
        f"replicate {replicate} whitener",
    )
    component_whitening = component_whitening.to(device)
    graph_whitening = graph_whitening.to(device)
    component_operator = WhitenedMeasurementOperator(
        operator,
        component_whitening,
    ).to(device)
    graph_operator = WhitenedMeasurementOperator(
        operator,
        graph_whitening,
    ).to(device)
    component_prepared = component_operator.prepare_observation(observation)
    graph_prepared = graph_operator.prepare_observation(observation)
    ones_sigma = torch.ones(
        (len(truth), view_count),
        dtype=torch.float32,
        device=device,
    )
    ones_mask = torch.ones_like(ones_sigma)
    scalar_measurement_count = torch.full(
        (len(truth),),
        float(view_count * rays_per_view * 2),
        dtype=torch.float32,
        device=device,
    )
    root_m = torch.sqrt(scalar_measurement_count)
    amplitude_scale = torch.linalg.vector_norm(
        graph_prepared.flatten(1), dim=1
    ) / root_m
    amplitude_scale = amplitude_scale.clamp_min(1e-12)
    b0 = graph_prepared / (
        amplitude_scale[:, None, None] * root_m[:, None, None]
    )
    measurement_scale = 1.0 / float(root_m[0])
    solver = config.get("solver", {})
    if not isinstance(solver, Mapping):
        raise ValueError("solver must be an object")
    graph_operator.reset_call_counts()
    _synchronize(device)
    norm_started = time.perf_counter()
    norm = estimate_block_operator_norms(
        graph_operator,
        batch_size=len(truth),
        sigma_by_view=ones_sigma,
        view_mask=ones_mask,
        rays_per_view=rays_per_view,
        power_iterations=int(solver["norm_power_iterations"]),
        safety_factor=float(solver["norm_safety_factor"]),
        measurement_scale=measurement_scale,
        seed=int(solver["norm_seed"]) + int(replicate),
    )
    _synchronize(device)
    norm_elapsed = time.perf_counter() - norm_started
    if graph_operator.call_report() != {
        "forward_calls": int(norm.forward_calls),
        "adjoint_calls": int(norm.adjoint_calls),
    }:
        raise ValueError("norm setup call ledger is inconsistent")
    regularization_count = regularization_vector_count(operator.support)
    return {
        "replicate": int(replicate),
        "families": families,
        "truth": truth,
        "component_operator": component_operator,
        "graph_operator": graph_operator,
        "component_prepared": component_prepared,
        "graph_prepared": graph_prepared,
        "b0": b0,
        "amplitude_scale": amplitude_scale,
        "measurement_scale": measurement_scale,
        "scalar_measurement_count": int(scalar_measurement_count[0]),
        "ones_sigma": ones_sigma,
        "ones_mask": ones_mask,
        "rays_per_view": rays_per_view,
        "regularization_vector_count": regularization_count,
        "norm": norm,
        "norm_elapsed_seconds": float(norm_elapsed),
        "whitener_anchor": observed_whitener_anchor,
        "graph_activated_view_count": int(
            sum(bool(row["graph_activated"]) for row in gate_rows)
        ),
    }


def run_screen(
    *,
    root: Path,
    config_path: Path,
    view_root: Path,
    device_name: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    """Execute the frozen opened screen.  Call only after explicit authorization."""

    root = require_repository_root(root)
    config_path = config_path.resolve()
    try:
        config_path.relative_to(root)
    except ValueError as error:
        raise ValueError("formal config must be inside the imported repository") from error
    config = _load_json(config_path)
    config_audit = validate_frozen_config(config)
    config_audit["amendment"] = validate_infrastructure_amendment(
        root=root,
        config=config,
    )
    git_audit = validate_clean_worktree(root)
    source_sha256_audit = validate_source_sha256(root=root, config=config)
    geometry_manifest_audit = validate_geometry_audit_manifest(
        root=root,
        view_root=view_root,
        config=config,
    )
    config_audit["source_sha256"] = source_sha256_audit
    config_audit["geometry_audit_manifest"] = geometry_manifest_audit
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("frozen JSON must contain protocol path and SHA-256")
    protocol_path = root / str(protocol.get("path", ""))
    if not protocol_path.is_file():
        raise ValueError("source_protocol is missing from the frozen JSON")
    protocol_sha = _sha256(protocol_path)
    expected_protocol_sha = str(protocol.get("sha256", ""))
    if not expected_protocol_sha or expected_protocol_sha != protocol_sha:
        raise ValueError("frozen protocol SHA-256 is missing or mismatched")
    repository_snapshot = capture_repository_snapshot(
        root=root,
        config_path=config_path,
        protocol_path=protocol_path,
        config=config,
    )
    try:
        e1_attestation = run_e1_test_attestation(root, config=config)
        require_unchanged_repository_snapshot(
            root=root,
            expected=repository_snapshot,
        )
    except PDHGPreflightInvalid as error:
        error.evidence.update(
            {
                "repository_snapshot": repository_snapshot,
                "git": git_audit,
                "truth_based_performance_rows_generated": False,
                "performance_ranking_generated": False,
            }
        )
        raise
    config_audit["e1_attestation"] = e1_attestation
    config_audit["initial_repository_snapshot"] = repository_snapshot
    runtime_anchor_path = root / str(config["source_runtime_anchor_manifest"])
    runtime_anchor_manifest = _load_json(runtime_anchor_path)
    config_audit["runtime_anchor_manifest"] = validate_runtime_anchor_manifest(
        runtime_anchor_manifest,
        config=config,
    )
    multiseed_path = root / str(config["source_multiseed_config"])
    smoke_path = root / str(config["source_smoke_config"])
    multiseed = _load_json(multiseed_path)
    smoke = _load_json(smoke_path)
    if max(OPENED_REPLICATES) >= int(multiseed["replicates"]["count"]):
        raise ValueError("opened replicate index exceeds source configuration")
    if device_name != "mps":
        raise ValueError("formal protocol fixes device='mps'")
    device = torch.device(device_name)
    if not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    torch.set_num_threads(8)

    candidates = pdhg_candidate_grid(config)
    controls = alpha_zero_controls(config)
    graph_candidates = graph_pcgls_candidates(config)
    component = component_pcgls_candidate(config)
    method_ids = all_method_ids(config)
    started = time.perf_counter()

    geometry_config = smoke["geometry"]
    view_count = int(geometry_config["view_count"])
    rays_per_view = int(geometry_config["rays_per_view"])
    grid_size = int(geometry_config["grid_size"])
    finite_aperture_sample_count = int(
        geometry_config["finite_aperture_sample_count"]
    )
    geometry, geometry_provenance = load_real_support_geometry(
        view_root,
        rays_per_view=rays_per_view,
        sample_count=finite_aperture_sample_count,
    )
    view_ids = tuple(
        int(row.get("view_id_zero_based", -1)) for row in geometry_provenance
    )
    if view_ids != tuple(range(view_count)):
        raise ValueError(
            "loaded geometry provenance must contain ordered view IDs 0..8"
        )
    operator = _make_operator(
        geometry,
        grid_size=grid_size,
        dtype=torch.float32,
    )
    operator.support.copy_(zero_outer_boundary_support((grid_size,) * 3))
    observed_geometry_anchor = geometry_operator_anchor(
        geometry,
        geometry_provenance,
        operator.support,
        {
            "grid_shape_zyx": [grid_size, grid_size, grid_size],
            "grid_minimum_xyz_m": [-0.11, -0.11, -0.11],
            "grid_maximum_xyz_m": [0.11, 0.11, 0.11],
            "view_count": view_count,
            "rays_per_view": rays_per_view,
            "finite_aperture_sample_count": finite_aperture_sample_count,
            "operator_dtype": "torch.float32",
            "support_boundary": "zero_outer_boundary",
        },
    )
    verify_exact_anchor(
        runtime_anchor_manifest["runtime_geometry_anchor"],
        observed_geometry_anchor,
        "runtime geometry/operator inputs",
    )
    config_audit["runtime_geometry_anchor"] = observed_geometry_anchor
    operator = operator.to(device)
    detector_config = _load_json(root / str(smoke["source_detector_config"]))
    graph_config = detector_config["detector_graph"]
    graph = build_detector_knn_graph(
        geometry["detector_xy"],
        view_count=view_count,
        rays_per_view=rays_per_view,
        neighbor_count=int(graph_config["neighbor_count"]),
        least_squares_ridge=float(graph_config["least_squares_ridge"]),
    )
    eigenpairs = [
        detector_graph_spectral_basis(graph, view_index=view)
        for view in range(view_count)
    ]
    graph_family_index = [
        str(value) for value in smoke["data"]["noise_families"]
    ].index("graph_heat")
    solver = config["solver"]
    direction = GeneralizedSobolevDirection(
        (grid_size,) * 3,
        strength=float(solver["sobolev_strength"]),
        epsilon=float(solver["sobolev_epsilon"]),
    ).to(device)
    eta = float(solver["step_safety_eta"])
    theta = float(solver["extrapolation"])
    huber_delta = float(solver["huber_delta"])

    metric_rows: list[dict[str, Any]] = []
    call_ledger: dict[str, Any] = {
        "normalization": (
            "A0=Ag/sqrt(m), b0=bg/(s_i*sqrt(m)), output=s_i*z"
        ),
        "whitened_solver_sigma": "ones (whitening already contains scale_by_view)",
        "replicates": [],
    }
    contexts: list[dict[str, Any]] = []

    for replicate in OPENED_REPLICATES:
        context = _build_replicate_context(
            root=root,
            config=config,
            runtime_anchor_manifest=runtime_anchor_manifest,
            smoke=smoke,
            multiseed=multiseed,
            replicate=replicate,
            operator=operator,
            geometry=geometry,
            eigenpairs=eigenpairs,
            graph=graph,
            graph_family_index=graph_family_index,
            device=device,
        )
        contexts.append(context)

    config_audit["runtime_whitener_anchors"] = {
        str(context["replicate"]): context["whitener_anchor"]
        for context in contexts
    }

    stress_contexts = [
        PDHGStressContext(
            replicate=int(context["replicate"]),
            operator=context["graph_operator"],
            observation_b0=context["b0"],
            sigma_by_view=context["ones_sigma"],
            view_mask=context["ones_mask"],
            rays_per_view=int(context["rays_per_view"]),
            measurement_scale=float(context["measurement_scale"]),
            regularization_vector_count=int(
                context["regularization_vector_count"]
            ),
            norm_estimate=context["norm"],
        )
        for context in contexts
    ]
    stability_preflight = run_pdhg_stability_preflight(
        contexts=stress_contexts,
        config=config,
        device=device,
    )
    try:
        require_valid_pdhg_stability_preflight(stability_preflight)
    except PDHGPreflightInvalid as error:
        try:
            relative_config_path = str(config_path.resolve().relative_to(root))
        except ValueError:
            relative_config_path = str(config_path.resolve())
        error.evidence.update(
            {
                "config": {
                    "path": relative_config_path,
                    "sha256": _sha256(config_path),
                    "schema_version": config.get("schema_version"),
                    "status": config.get("status"),
                    "amendment": config.get("amendment"),
                    "source_sha256": source_sha256_audit,
                    "geometry_audit_manifest": geometry_manifest_audit,
                },
                "protocol": {
                    "path": str(protocol_path.relative_to(root)),
                    "expected_sha256": expected_protocol_sha,
                    "actual_sha256": protocol_sha,
                },
                "git": git_audit,
                "repository_snapshot": repository_snapshot,
                "environment": {
                    "python_version": sys.version,
                    "torch_version": torch.__version__,
                    "numpy_version": np.__version__,
                    "platform": platform.platform(),
                    "device": device_name,
                    "dtype": "float32",
                    "synthetic_scale_by_view_uses_clean_truth": True,
                    "experimental_deployment_scale_available": False,
                },
                "truth_based_performance_rows_generated": False,
                "performance_ranking_generated": False,
            }
        )
        raise
    call_ledger["pdhg_stability_preflight"] = stability_preflight[
        "call_ledger"
    ]

    for context in contexts:
        replicate = int(context["replicate"])
        rep_ledger: dict[str, Any] = {
            "replicate": replicate,
            "data_generation_forward_calls": 1,
            "shared_norm_setup": {
                "forward_calls": int(context["norm"].forward_calls),
                "adjoint_calls": int(context["norm"].adjoint_calls),
                "elapsed_seconds": float(context["norm_elapsed_seconds"]),
                "power_iterations": int(context["norm"].power_iterations),
                "safety_factor": float(context["norm"].safety_factor),
                "data_norm_squared_upper": float(
                    context["norm"].data_norm_squared_upper
                ),
                "gradient_norm_squared_upper": float(
                    context["norm"].gradient_norm_squared_upper
                ),
                "ordinary_power_estimate_is_not_a_proven_bound": True,
            },
            "component": {},
            "graph_trajectory": {},
            "pdhg_runs": [],
        }

        component_operator = context["component_operator"]
        component_operator.reset_call_counts()
        _synchronize(device)
        solve_started = time.perf_counter()
        component_result = preconditioned_cgls_reconstruction(
            component_operator,
            context["component_prepared"],
            sigma_by_view=context["ones_sigma"],
            view_mask=context["ones_mask"],
            rays_per_view=context["rays_per_view"],
            stages=4,
            preconditioner=direction,
        )
        _synchronize(device)
        component_elapsed = time.perf_counter() - solve_started
        if component_operator.call_report() != {
            "forward_calls": 4,
            "adjoint_calls": 4,
        }:
            raise ValueError("component baseline violated 4 F + 4 A^T")
        component_data = 0.5 * torch.sum(
            component_result.residual_uv.square(), dim=(1, 2)
        )
        component_rows = _base_metric_rows(
            replicate=replicate,
            candidate=component,
            families=context["families"],
            prediction=component_result.volume,
            truth=context["truth"],
            data_objective=component_data,
            regularization_penalty=torch.zeros_like(component_data),
            total_objective=component_data,
            primal_update_norm=None,
            dual_feasibility_violation=None,
            solver_elapsed_seconds=component_elapsed,
            support_gauge_ok=_support_ok(component_result.volume, operator.support),
        )
        metric_rows.extend(component_rows)
        rep_ledger["component"] = {
            "physical_forward_calls": 4,
            "physical_adjoint_calls": 4,
            "logical_forward_calls": 4,
            "logical_adjoint_calls": 4,
            "elapsed_seconds": component_elapsed,
        }

        graph_operator = context["graph_operator"]
        graph_operator.reset_call_counts()
        _synchronize(device)
        graph_started = time.perf_counter()
        graph_trajectory = preconditioned_cgls_trajectory(
            graph_operator,
            context["graph_prepared"],
            sigma_by_view=context["ones_sigma"],
            view_mask=context["ones_mask"],
            rays_per_view=context["rays_per_view"],
            checkpoint_stages=GRAPH_STAGES,
            preconditioner=direction,
        )
        _synchronize(device)
        graph_elapsed = time.perf_counter() - graph_started
        if graph_operator.call_report() != {
            "forward_calls": max(GRAPH_STAGES),
            "adjoint_calls": max(GRAPH_STAGES),
        }:
            raise ValueError("graph trajectory violated prefix call budget")
        for graph_candidate in graph_candidates:
            result = graph_trajectory[int(graph_candidate["iterations"])]
            data_objective = 0.5 * torch.sum(
                result.residual_uv.square(), dim=(1, 2)
            )
            metric_rows.extend(
                _base_metric_rows(
                    replicate=replicate,
                    candidate=graph_candidate,
                    families=context["families"],
                    prediction=result.volume,
                    truth=context["truth"],
                    data_objective=data_objective,
                    regularization_penalty=torch.zeros_like(data_objective),
                    total_objective=data_objective,
                    primal_update_norm=None,
                    dual_feasibility_violation=None,
                    solver_elapsed_seconds=graph_elapsed,
                    support_gauge_ok=_support_ok(result.volume, operator.support),
                )
            )
        rep_ledger["graph_trajectory"] = {
            "logical_forward_calls_if_independent": int(sum(GRAPH_STAGES)),
            "logical_adjoint_calls_if_independent": int(sum(GRAPH_STAGES)),
            "physical_forward_calls": max(GRAPH_STAGES),
            "physical_adjoint_calls": max(GRAPH_STAGES),
            "prefix_forward_calls_saved": int(sum(GRAPH_STAGES) - max(GRAPH_STAGES)),
            "prefix_adjoint_calls_saved": int(sum(GRAPH_STAGES) - max(GRAPH_STAGES)),
            "elapsed_seconds": graph_elapsed,
        }

        trajectory_groups: list[list[dict[str, Any]]] = [list(controls)]
        for penalty in PDHG_PENALTIES:
            for fraction, _, _ in PDHG_ALPHA_SPECS:
                trajectory_groups.append(
                    [
                        row
                        for row in candidates
                        if str(row["penalty"]) == penalty
                        and str(row["alpha_fraction"]) == fraction
                    ]
                )
        if len(trajectory_groups) != 9 or any(
            len(group) != 4 for group in trajectory_groups
        ):
            raise ValueError("PDHG grid must form nine four-checkpoint trajectories")
        for trajectory_candidates in trajectory_groups:
            try:
                rows, ledger, _ = _run_pdhg_trajectory(
                    operator=graph_operator,
                    observation_b0=context["b0"],
                    truth=context["truth"],
                    output_scale=context["amplitude_scale"],
                    trajectory_candidates=trajectory_candidates,
                    ones_sigma=context["ones_sigma"],
                    ones_mask=context["ones_mask"],
                    rays_per_view=context["rays_per_view"],
                    measurement_scale=context["measurement_scale"],
                    regularization_vector_count=context[
                        "regularization_vector_count"
                    ],
                    norm_estimate=context["norm"],
                    eta=eta,
                    theta=theta,
                    huber_delta=huber_delta,
                    device=device,
                )
                for row in rows:
                    row["replicate"] = replicate
                metric_rows.extend(rows)
                ledger["replicate"] = replicate
                rep_ledger["pdhg_runs"].append(ledger)
            except (FloatingPointError, RuntimeError, ValueError) as error:
                for candidate in trajectory_candidates:
                    metric_rows.extend(
                        _failed_metric_rows(
                            replicate=replicate,
                            candidate=candidate,
                            families=context["families"],
                            error=error,
                        )
                    )
                rep_ledger["pdhg_runs"].append(
                    {
                        "replicate": replicate,
                        "candidate_ids": [
                            str(row["candidate_id"])
                            for row in trajectory_candidates
                        ],
                        "status": "solver_failed",
                        "failure_type": type(error).__name__,
                        "failure_message": str(error),
                    }
                )
        call_ledger["replicates"].append(rep_ledger)

    historical_rows, historical_manifest = _read_historical_sup_rows(
        root=root,
        config=config,
    )
    metric_rows.extend(historical_rows)
    pdhg_logical_per_replicate = sum(
        int(row["iterations"]) for row in [*controls, *candidates]
    )
    pdhg_physical_per_replicate = 9 * max(PDHG_ITERATIONS)
    graph_logical_per_replicate = sum(GRAPH_STAGES)
    graph_physical_per_replicate = max(GRAPH_STAGES)
    historical_first_rows = {
        identifier: next(
            row for row in historical_rows if row["candidate_id"] == identifier
        )
        for identifier in HISTORICAL_SUP_IDS
    }
    call_ledger["summary_excluding_timing_reruns"] = {
        "replicate_count": len(OPENED_REPLICATES),
        "pdhg_stability_preflight_physical_forward_calls": (
            stability_preflight["call_ledger"][
                "observed_physical_forward_calls"
            ]
        ),
        "pdhg_stability_preflight_physical_adjoint_calls": (
            stability_preflight["call_ledger"][
                "observed_physical_adjoint_calls"
            ]
        ),
        "pdhg_stability_preflight_history_extra_forward_calls": (
            stability_preflight["call_ledger"][
                "history_scorer_forward_calls"
            ]
        ),
        "pdhg_candidate_and_control_logical_forward_calls": (
            len(OPENED_REPLICATES) * pdhg_logical_per_replicate
        ),
        "pdhg_candidate_and_control_logical_adjoint_calls": (
            len(OPENED_REPLICATES) * pdhg_logical_per_replicate
        ),
        "pdhg_trajectory_physical_forward_calls": (
            len(OPENED_REPLICATES) * pdhg_physical_per_replicate
        ),
        "pdhg_trajectory_physical_adjoint_calls": (
            len(OPENED_REPLICATES) * pdhg_physical_per_replicate
        ),
        "pdhg_trajectory_physical_gradient_calls": (
            len(OPENED_REPLICATES) * pdhg_physical_per_replicate
        ),
        "pdhg_trajectory_physical_gradient_adjoint_calls": (
            len(OPENED_REPLICATES) * pdhg_physical_per_replicate
        ),
        "pdhg_prefix_forward_calls_saved": len(OPENED_REPLICATES)
        * (pdhg_logical_per_replicate - pdhg_physical_per_replicate),
        "pdhg_prefix_adjoint_calls_saved": len(OPENED_REPLICATES)
        * (pdhg_logical_per_replicate - pdhg_physical_per_replicate),
        "pdhg_prefix_gradient_calls_saved": len(OPENED_REPLICATES)
        * (pdhg_logical_per_replicate - pdhg_physical_per_replicate),
        "pdhg_prefix_gradient_adjoint_calls_saved": len(OPENED_REPLICATES)
        * (pdhg_logical_per_replicate - pdhg_physical_per_replicate),
        "pdhg_checkpoint_metric_forward_calls": (
            len(OPENED_REPLICATES) * 9 * len(PDHG_ITERATIONS)
        ),
        "pdhg_checkpoint_metric_gradient_calls": (
            len(OPENED_REPLICATES) * 9 * len(PDHG_ITERATIONS)
        ),
        "pdhg_checkpoint_metric_adjoint_calls": 0,
        "pdhg_checkpoint_metric_gradient_adjoint_calls": 0,
        "graph_logical_forward_calls": (
            len(OPENED_REPLICATES) * graph_logical_per_replicate
        ),
        "graph_logical_adjoint_calls": (
            len(OPENED_REPLICATES) * graph_logical_per_replicate
        ),
        "graph_trajectory_physical_forward_calls": (
            len(OPENED_REPLICATES) * graph_physical_per_replicate
        ),
        "graph_trajectory_physical_adjoint_calls": (
            len(OPENED_REPLICATES) * graph_physical_per_replicate
        ),
        "graph_prefix_forward_calls_saved": len(OPENED_REPLICATES)
        * (graph_logical_per_replicate - graph_physical_per_replicate),
        "graph_prefix_adjoint_calls_saved": len(OPENED_REPLICATES)
        * (graph_logical_per_replicate - graph_physical_per_replicate),
        "component_physical_forward_calls": len(OPENED_REPLICATES) * 4,
        "component_physical_adjoint_calls": len(OPENED_REPLICATES) * 4,
        "norm_setup_forward_calls": sum(
            int(context["norm"].forward_calls) for context in contexts
        ),
        "norm_setup_adjoint_calls": sum(
            int(context["norm"].adjoint_calls) for context in contexts
        ),
        "data_generation_forward_calls": len(OPENED_REPLICATES),
        "historical_sup_logical_forward_calls_per_replicate": sum(
            int(row["forward_calls"])
            for row in historical_first_rows.values()
        ),
        "historical_sup_logical_adjoint_calls_per_replicate": sum(
            int(row["adjoint_calls"])
            for row in historical_first_rows.values()
        ),
        "historical_sup_physical_calls_in_this_run": 0,
    }
    preliminary_count_audit = validate_metric_row_counts(
        metric_rows,
        method_ids=method_ids,
    )
    pdhg_row_validity_audit = validate_pdhg_performance_rows(metric_rows)
    frontier = graph_budget_frontier(metric_rows)

    warmup_context = contexts[0]
    warmup_candidate = candidates[0]
    warmup_frontier_id = str(
        frontier[int(warmup_candidate["iterations"])]["candidate_id"]
    )
    warmup_frontier_stage = int(warmup_frontier_id.rsplit("k", 1)[1])
    warmup_pdhg_seconds, _, warmup_pdhg_ledger = _timed_pdhg_solve(
        operator=warmup_context["graph_operator"],
        observation_b0=warmup_context["b0"],
        candidate=warmup_candidate,
        ones_sigma=warmup_context["ones_sigma"],
        ones_mask=warmup_context["ones_mask"],
        rays_per_view=warmup_context["rays_per_view"],
        measurement_scale=warmup_context["measurement_scale"],
        regularization_vector_count=warmup_context[
            "regularization_vector_count"
        ],
        norm_estimate=warmup_context["norm"],
        eta=eta,
        theta=theta,
        huber_delta=huber_delta,
        device=device,
    )
    warmup_graph_seconds, _ = _timed_graph_solve(
        operator=warmup_context["graph_operator"],
        prepared=warmup_context["graph_prepared"],
        ones_sigma=warmup_context["ones_sigma"],
        ones_mask=warmup_context["ones_mask"],
        rays_per_view=warmup_context["rays_per_view"],
        stages=warmup_frontier_stage,
        direction=direction,
        device=device,
    )
    timing_warmup = {
        "excluded_from_all_ratios": True,
        "process_cold_start_claimed": False,
        "note": (
            "first explicit timing-family conditioning runs; required numerical "
            "preflight already used MPS, so these are not process cold-start probes"
        ),
        "pdhg": {
            **warmup_pdhg_ledger,
            "replicate": int(warmup_context["replicate"]),
            "elapsed_seconds": float(warmup_pdhg_seconds),
        },
        "graph": {
            "candidate_id": warmup_frontier_id,
            "replicate": int(warmup_context["replicate"]),
            "stages": warmup_frontier_stage,
            "solver_forward_calls": warmup_frontier_stage,
            "solver_adjoint_calls": warmup_frontier_stage,
            "elapsed_seconds": float(warmup_graph_seconds),
            "truth_used": False,
            "precision_metrics_used": False,
        },
    }

    independent_pdhg_timing: dict[tuple[int, str], float] = {}
    independent_pdhg_timing_runs: list[dict[str, Any]] = []
    independent_graph_timing_runs: list[dict[str, Any]] = []
    paired_timing: dict[tuple[int, str], dict[str, Any]] = {}
    for context_index, context in enumerate(contexts):
        ordered_candidates = (
            list(candidates)
            if context_index % 2 == 0
            else list(reversed(candidates))
        )
        pair_order = (
            "candidate_then_baseline"
            if context_index % 2 == 0
            else "baseline_then_candidate"
        )
        for pair_index, candidate in enumerate(ordered_candidates):
            identifier = str(candidate["candidate_id"])
            frontier_id = str(
                frontier[int(candidate["iterations"])]["candidate_id"]
            )
            frontier_stage = int(frontier_id.rsplit("k", 1)[1])

            def time_candidate() -> tuple[float, dict[str, Any]]:
                elapsed, _, ledger = _timed_pdhg_solve(
                    operator=context["graph_operator"],
                    observation_b0=context["b0"],
                    candidate=candidate,
                    ones_sigma=context["ones_sigma"],
                    ones_mask=context["ones_mask"],
                    rays_per_view=context["rays_per_view"],
                    measurement_scale=context["measurement_scale"],
                    regularization_vector_count=context[
                        "regularization_vector_count"
                    ],
                    norm_estimate=context["norm"],
                    eta=eta,
                    theta=theta,
                    huber_delta=huber_delta,
                    device=device,
                )
                return float(elapsed), ledger

            def time_baseline() -> float:
                elapsed, _ = _timed_graph_solve(
                    operator=context["graph_operator"],
                    prepared=context["graph_prepared"],
                    ones_sigma=context["ones_sigma"],
                    ones_mask=context["ones_mask"],
                    rays_per_view=context["rays_per_view"],
                    stages=frontier_stage,
                    direction=direction,
                    device=device,
                )
                return float(elapsed)

            if pair_order == "candidate_then_baseline":
                candidate_seconds, timing_ledger = time_candidate()
                baseline_seconds = time_baseline()
            else:
                baseline_seconds = time_baseline()
                candidate_seconds, timing_ledger = time_candidate()
            replicate = int(context["replicate"])
            pair_row = {
                "replicate": replicate,
                "candidate_id": identifier,
                "graph_frontier_id": frontier_id,
                "candidate_iterations": int(candidate["iterations"]),
                "baseline_stages": frontier_stage,
                "order": pair_order,
                "pair_index_within_replicate": pair_index,
                "candidate_seconds": candidate_seconds,
                "baseline_seconds": baseline_seconds,
                "paired_ratio": candidate_seconds / baseline_seconds,
            }
            paired_timing[(replicate, identifier)] = pair_row
            independent_pdhg_timing[(replicate, identifier)] = candidate_seconds
            independent_pdhg_timing_runs.append(
                {**timing_ledger, **pair_row}
            )
            independent_graph_timing_runs.append(
                {
                    **pair_row,
                    "solver_forward_calls": frontier_stage,
                    "solver_adjoint_calls": frontier_stage,
                    "truth_used": False,
                    "precision_metrics_used": False,
                }
            )
    independent_timing_expected_calls = sum(
        int(candidate["iterations"]) for candidate in candidates
    ) * len(OPENED_REPLICATES)
    call_ledger["independent_pdhg_candidate_timing"] = {
        "run_count": len(independent_pdhg_timing_runs),
        "expected_run_count": (
            EXPECTED_FORMAL_CANDIDATES * len(OPENED_REPLICATES)
        ),
        "solver_forward_calls": independent_timing_expected_calls,
        "solver_adjoint_calls": independent_timing_expected_calls,
        "solver_gradient_calls": independent_timing_expected_calls,
        "solver_gradient_adjoint_calls": independent_timing_expected_calls,
        "metric_forward_calls": 0,
        "metric_adjoint_calls": 0,
        "metric_gradient_calls": 0,
        "metric_gradient_adjoint_calls": 0,
        "truth_used": False,
        "precision_metrics_used": False,
        "runs": independent_pdhg_timing_runs,
    }
    graph_timing_expected_calls = sum(
        int(frontier[int(candidate["iterations"])]["stage"])
        for candidate in candidates
    ) * len(OPENED_REPLICATES)
    call_ledger["independent_graph_frontier_timing"] = {
        "run_count": len(independent_graph_timing_runs),
        "expected_run_count": (
            EXPECTED_FORMAL_CANDIDATES * len(OPENED_REPLICATES)
        ),
        "solver_forward_calls": graph_timing_expected_calls,
        "solver_adjoint_calls": graph_timing_expected_calls,
        "truth_used": False,
        "precision_metrics_used": False,
        "runs": independent_graph_timing_runs,
    }
    call_ledger["timing_warmup_excluded"] = timing_warmup
    for row in metric_rows:
        if str(row.get("method")) != "pdhg":
            continue
        timing_key = (int(row["replicate"]), str(row["candidate_id"]))
        row["solver_elapsed_seconds"] = float(
            independent_pdhg_timing[timing_key]
        )
        row["solver_elapsed_source"] = "independent_k_timing_only"

    wall_ratios = paired_pdhg_wall_time_ratios(
        candidates,
        paired_timing=paired_timing,
    )

    comparative_rows = add_pdhg_comparative_gains(
        metric_rows,
        frontier=frontier,
    )
    count_audit = validate_metric_row_counts(
        comparative_rows,
        method_ids=method_ids,
    )
    pdhg_row_validity_after_timing = validate_pdhg_performance_rows(
        comparative_rows
    )
    summaries = summarize_pdhg_candidates(
        comparative_rows,
        wall_time_ratios=wall_ratios,
    )
    ranking = rank_pdhg_candidates(summaries)

    timing_audit: dict[str, Any] = {
        "screen_ratio_basis": (
            "median of two adjacent candidate/frontier paired ratios; replicate 0 "
            "uses candidate-then-baseline and replicate 8 uses the reverse order; "
            "shared max-K prefix elapsed and explicit warm-ups are excluded"
        ),
        "warmup": timing_warmup,
        "paired_runs": [
            paired_timing[key] for key in sorted(paired_timing)
        ],
        "candidate_independent_seconds": {
            f"replicate_{replicate}:{identifier}": float(value)
            for (replicate, identifier), value in sorted(
                independent_pdhg_timing.items()
            )
        },
        "paired_frontier_seconds": {
            f"replicate_{replicate}:{identifier}": float(
                row["baseline_seconds"]
            )
            for (replicate, identifier), row in sorted(paired_timing.items())
        },
        "winner_five_repeat_audit": None,
    }
    valid_winner = (
        ranking[0]
        if all(bool(row.get("ranking_valid", False)) for row in ranking)
        else None
    )
    winner_ratio: float | None = None
    if valid_winner is not None:
        winner_id = str(valid_winner["candidate_id"])
        winner_candidate = next(
            row for row in candidates if row["candidate_id"] == winner_id
        )
        frontier_id = str(
            frontier[int(winner_candidate["iterations"])]["candidate_id"]
        )
        frontier_stage = int(frontier_id.rsplit("k", 1)[1])
        candidate_times: list[float] = []
        baseline_times: list[float] = []
        winner_pairs: list[dict[str, Any]] = []
        for context_index, context in enumerate(contexts):
            for repeat_index in range(5):
                pair_order = (
                    "candidate_then_baseline"
                    if (context_index + repeat_index) % 2 == 0
                    else "baseline_then_candidate"
                )

                def time_winner() -> float:
                    elapsed, _, _ = _timed_pdhg_solve(
                        operator=context["graph_operator"],
                        observation_b0=context["b0"],
                        candidate=winner_candidate,
                        ones_sigma=context["ones_sigma"],
                        ones_mask=context["ones_mask"],
                        rays_per_view=context["rays_per_view"],
                        measurement_scale=context["measurement_scale"],
                        regularization_vector_count=context[
                            "regularization_vector_count"
                        ],
                        norm_estimate=context["norm"],
                        eta=eta,
                        theta=theta,
                        huber_delta=huber_delta,
                        device=device,
                    )
                    return float(elapsed)

                def time_winner_baseline() -> float:
                    elapsed, _ = _timed_graph_solve(
                        operator=context["graph_operator"],
                        prepared=context["graph_prepared"],
                        ones_sigma=context["ones_sigma"],
                        ones_mask=context["ones_mask"],
                        rays_per_view=context["rays_per_view"],
                        stages=frontier_stage,
                        direction=direction,
                        device=device,
                    )
                    return float(elapsed)

                if pair_order == "candidate_then_baseline":
                    candidate_elapsed = time_winner()
                    baseline_elapsed = time_winner_baseline()
                else:
                    baseline_elapsed = time_winner_baseline()
                    candidate_elapsed = time_winner()
                candidate_times.append(candidate_elapsed)
                baseline_times.append(baseline_elapsed)
                winner_pairs.append(
                    {
                        "replicate": int(context["replicate"]),
                        "repeat_index": repeat_index,
                        "order": pair_order,
                        "candidate_seconds": candidate_elapsed,
                        "baseline_seconds": baseline_elapsed,
                        "paired_ratio": candidate_elapsed / baseline_elapsed,
                    }
                )
        candidate_stats = _median_iqr(candidate_times)
        baseline_stats = _median_iqr(baseline_times)
        winner_ratio = float(
            statistics.median(
                float(row["paired_ratio"]) for row in winner_pairs
            )
        )
        timing_audit["winner_five_repeat_audit"] = {
            "winner_candidate_id": winner_id,
            "graph_frontier_id": frontier_id,
            "runs_per_replicate_per_method": 5,
            "candidate": candidate_stats,
            "baseline": baseline_stats,
            "paired_runs": winner_pairs,
            "median_of_paired_ratios": winner_ratio,
            "ratio_of_global_medians_descriptive_only": (
                candidate_stats["median_seconds"]
                / baseline_stats["median_seconds"]
            ),
            "order_counts": dict(Counter(row["order"] for row in winner_pairs)),
            "call_ledger": {
                "candidate_solver_forward_calls": len(candidate_times)
                * int(winner_candidate["iterations"]),
                "candidate_solver_adjoint_calls": len(candidate_times)
                * int(winner_candidate["iterations"]),
                "candidate_solver_gradient_calls": len(candidate_times)
                * int(winner_candidate["iterations"]),
                "candidate_solver_gradient_adjoint_calls": len(candidate_times)
                * int(winner_candidate["iterations"]),
                "candidate_metric_forward_calls": 0,
                "candidate_metric_gradient_calls": 0,
                "candidate_metric_adjoint_calls": 0,
                "candidate_metric_gradient_adjoint_calls": 0,
                "baseline_solver_forward_calls": len(baseline_times)
                * frontier_stage,
                "baseline_solver_adjoint_calls": len(baseline_times)
                * frontier_stage,
            },
        }

    call_ledger["winner_five_repeat_timing"] = (
        None
        if timing_audit["winner_five_repeat_audit"] is None
        else {
            "truth_used": False,
            "precision_metrics_used": False,
            "paired_runs": timing_audit["winner_five_repeat_audit"][
                "paired_runs"
            ],
            **timing_audit["winner_five_repeat_audit"]["call_ledger"],
        }
    )

    decision = pdhg_screen_decision(
        ranking,
        count_audit=count_audit,
        winner_timing_ratio=winner_ratio,
    )
    methods = _method_summaries(comparative_rows)
    environment = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "device": device_name,
        "dtype": "float32",
        "git_commit": git_audit["git_commit"],
        "worktree_clean": git_audit["worktree_clean"],
        "config_path": str(config_path.relative_to(root)),
        "config_sha256": _sha256(config_path),
        "protocol_path": str(protocol_path.relative_to(root)),
        "protocol_sha256": protocol_sha,
        "source_multiseed_config_sha256": _sha256(multiseed_path),
        "source_smoke_config_sha256": _sha256(smoke_path),
        "source_sha256": source_sha256_audit,
        "geometry_audit_manifest_filename": geometry_manifest_audit[
            "filename"
        ],
        "geometry_audit_manifest_sha256": geometry_manifest_audit[
            "actual_sha256"
        ],
        "runtime_geometry_anchor_sha256": observed_geometry_anchor["sha256"],
        "runtime_whitener_anchor_sha256": {
            str(context["replicate"]): context["whitener_anchor"]["sha256"]
            for context in contexts
        },
        "whitener_mode": (
            "deterministic_refit_verified_against_prerun_anchor"
        ),
        "historical_in_memory_whitener_state_reuse_proven": False,
        "e1_attestation_valid": bool(e1_attestation["valid"]),
        "synthetic_scale_by_view_uses_clean_truth": True,
        "experimental_deployment_scale_available": False,
        "historical_sup_manifest": historical_manifest,
    }
    report = {
        "schema_version": SCHEMA,
        "status": decision["status"],
        "evidence_scope": "POSTOPEN_TWO_REPLICATE_SCALE_SMOKE",
        "configuration": config,
        "configuration_audit": config_audit,
        "pdhg_stability_preflight": stability_preflight,
        "preliminary_count_audit": preliminary_count_audit,
        "pdhg_row_validity_audit": pdhg_row_validity_audit,
        "pdhg_row_validity_after_timing": pdhg_row_validity_after_timing,
        "fail_closed_count_audit": count_audit,
        "formal_candidate_count": len(candidates),
        "control_reference_count": EXPECTED_CONTROLS_AND_REFERENCES,
        "method_count": len(methods),
        "metric_row_count": len(comparative_rows),
        "graph_budget_frontier": {
            str(key): value for key, value in frontier.items()
        },
        "ranked_pdhg_candidates": ranking,
        "decision": decision,
        "runtime": {
            "wall_seconds": float(time.perf_counter() - started),
            "maximum_rss_bytes": _max_rss_bytes(),
        },
        "claim_boundary": {
            "both_replicates_already_opened": True,
            "real_detector_geometry_used": True,
            "experimental_flow_field_truth_used": False,
            "synthetic_scale_by_view_uses_clean_truth": True,
            "experimental_deployment_scale_available": False,
            "whitener_deterministic_refit_verified_against_prerun_anchor": True,
            "historical_in_memory_whitener_state_reuse_proven": False,
            "ordinary_power_norm_estimate_is_not_a_proven_upper_bound": True,
            "fresh_seed_set_opened": False,
            "heldout_camera_opened": False,
            "neural_operator_training_authorized": False,
            "algorithm_superiority_claimed": False,
        },
    }
    return (
        report,
        comparative_rows,
        ranking,
        methods,
        call_ledger,
        {**timing_audit, "environment": environment},
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_checksums(output_dir: Path, paths: Iterable[Path]) -> None:
    rows = [f"{_sha256(path)}  {path.name}" for path in sorted(paths)]
    (output_dir / "checksums.sha256").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def _write_readme(output_dir: Path, report: Mapping[str, Any]) -> Path:
    decision = report["decision"]
    winner = decision.get("winner_candidate_id") or "none (invalid screen)"
    amendment_lines: list[str] = []
    configuration = report.get("configuration")
    if isinstance(configuration, Mapping) and isinstance(
        configuration.get("amendment"), Mapping
    ):
        amendment_lines = [
            "- v1 failed at the infrastructure/export stage.",
            "- v2 is a fresh rerun after an infrastructure-only amendment.",
            "- v1 performance metric rows: `0`; no v1 state was reused.",
        ]
    path = output_dir / "README.md"
    path.write_text(
        "\n".join(
            [
                "# PSU B0 covariance-weighted PDHG opened scale smoke",
                "",
                f"- Decision: `{decision['status']}`",
                f"- One preregistered winner: `{winner}`",
                f"- Formal candidates: `{report['formal_candidate_count']}`",
                f"- Methods: `{report['method_count']}`",
                f"- Paired metric rows: `{report['metric_row_count']}`",
                "- Evidence: two already-opened replicates only (E2).",
                "- Fresh data and all neural training remain sealed.",
                "- The ordinary power norm estimate is recorded as an estimate, "
                "not a proven bound.",
                *amendment_lines,
                "",
                "All numerical statements must be regenerated from the CSV/JSON "
                "files in this directory.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_preflight_invalid_bundle(
    output_dir: Path,
    error: PDHGPreflightInvalid,
) -> dict[str, Any]:
    """Atomically preserve a failed stress audit without performance outputs."""

    if output_dir.exists():
        raise ValueError("output directory must not exist before invalid bundle")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(
        f".{output_dir.name}.preflight-invalid-{time.time_ns()}"
    )
    staging.mkdir(parents=False, exist_ok=False)
    payload = {
        "schema_version": SCHEMA,
        "status": PDHGPreflightInvalid.status,
        "audit": error.audit,
        "evidence": error.evidence,
        "truth_based_performance_rows_generated": False,
        "performance_ranking_generated": False,
        "performance_metric_row_count": 0,
    }
    amendment_lines: list[str] = []
    config_evidence = error.evidence.get("config")
    if isinstance(config_evidence, Mapping) and isinstance(
        config_evidence.get("amendment"), Mapping
    ):
        amendment_lines = [
            "- v1 remains an infrastructure-invalid run with zero performance rows.",
            "- v2 was rerun from scratch after an infrastructure-only amendment.",
        ]
    try:
        json_path = staging / "preflight_invalid.json"
        readme_path = staging / "README.md"
        _write_json(json_path, payload)
        readme_path.write_text(
            "\n".join(
                [
                    "# PDHG preflight invalid",
                    "",
                    "- Status: `PDHG_PREFLIGHT_INVALID`",
                    "- No truth-based performance rows were generated.",
                    "- No candidate ranking was generated.",
                    *amendment_lines,
                    "- See `preflight_invalid.json` for the stress audit.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        _write_checksums(staging, (json_path, readme_path))
        staging.replace(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return payload


def _write_success_bundle(
    output_dir: Path,
    *,
    report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    ranking: Sequence[Mapping[str, Any]],
    methods: Sequence[Mapping[str, Any]],
    call_ledger: Mapping[str, Any],
    timing_bundle: Mapping[str, Any],
) -> None:
    """Atomically publish a complete successful-screen artifact bundle."""

    if output_dir.exists():
        raise ValueError("output directory must not exist before success bundle")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(
        f".{output_dir.name}.success-staging-{time.time_ns()}"
    )
    staging.mkdir(parents=False, exist_ok=False)
    try:
        environment = timing_bundle.get("environment")
        if not isinstance(environment, Mapping):
            raise ValueError("timing bundle must contain an environment object")
        timing_audit = {
            key: value for key, value in timing_bundle.items() if key != "environment"
        }
        paths = [
            staging / "report.json",
            staging / "metric_rows.csv",
            staging / "candidate_summaries.csv",
            staging / "method_summaries.csv",
            staging / "operator_call_ledger.json",
            staging / "timing_audit.json",
            staging / "environment.json",
            staging / "e1_attestation.json",
            staging / "e1_pytest.xml",
        ]
        _write_json(paths[0], report)
        _write_csv(paths[1], rows)
        _write_csv(paths[2], ranking)
        _write_csv(paths[3], methods)
        _write_json(paths[4], call_ledger)
        _write_json(paths[5], timing_audit)
        _write_json(paths[6], environment)
        configuration_audit = report.get("configuration_audit")
        if not isinstance(configuration_audit, Mapping):
            raise ValueError("report must contain configuration_audit")
        e1_attestation = configuration_audit.get("e1_attestation")
        if not isinstance(e1_attestation, Mapping):
            raise ValueError("report must contain E1 attestation")
        e1_output = dict(e1_attestation)
        e1_xml = e1_output.pop("junit_xml", None)
        if not isinstance(e1_xml, str) or not e1_xml:
            raise ValueError("E1 attestation must contain non-empty JUnit XML")
        _write_json(paths[7], e1_output)
        paths[8].write_text(e1_xml, encoding="utf-8")
        paths.append(_write_readme(staging, report))
        _write_checksums(staging, paths)
        staging.replace(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--execute-opened-screen",
        action="store_true",
        help="required acknowledgement that replicates 0 and 8 will be opened",
    )
    args = parser.parse_args()
    if not args.execute_opened_screen:
        parser.error("formal execution requires --execute-opened-screen")
    root = require_repository_root(args.root.resolve())
    config_path = args.config if args.config.is_absolute() else root / args.config
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise ValueError("output directory must not exist before formal execution")
    try:
        (
            report,
            rows,
            ranking,
            methods,
            call_ledger,
            timing_bundle,
        ) = run_screen(
            root=root,
            config_path=config_path,
            view_root=args.view_root.resolve(),
            device_name=str(args.device),
        )
    except PDHGPreflightInvalid as error:
        snapshot = error.evidence.get("repository_snapshot")
        if isinstance(snapshot, Mapping):
            error.evidence["final_repository_snapshot"] = (
                require_unchanged_repository_snapshot(
                    root=root,
                    expected=snapshot,
                )
            )
        _write_preflight_invalid_bundle(output_dir, error)
        print(
            json.dumps(
                {
                    "status": PDHGPreflightInvalid.status,
                    "performance_ranking_generated": False,
                    "output_dir": str(output_dir),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    initial_snapshot = report["configuration_audit"][
        "initial_repository_snapshot"
    ]
    final_snapshot = require_unchanged_repository_snapshot(
        root=root,
        expected=initial_snapshot,
    )
    report["configuration_audit"]["final_repository_snapshot"] = final_snapshot
    _write_success_bundle(
        output_dir,
        report=report,
        rows=rows,
        ranking=ranking,
        methods=methods,
        call_ledger=call_ledger,
        timing_bundle=timing_bundle,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "winner": report["decision"]["winner_candidate_id"],
                "method_count": report["method_count"],
                "metric_row_count": report["metric_row_count"],
                "output_dir": str(output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
