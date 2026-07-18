#!/usr/bin/env python3
"""Recover the frozen N3 analysis from an attested query-ledger key mismatch."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

try:
    from . import run_n2_pvgr_n3_grouped_factorial as grouped
except ImportError:
    import run_n2_pvgr_n3_grouped_factorial as grouped


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECOVERY_CONFIG = (
    ROOT / "demo_t16_operator/configs/n2_pvgr_n3_blind_analysis_recovery_v1.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str) -> Path:
    return (ROOT / value).resolve()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=check,
        capture_output=True,
    )


def _checkpoint_merkle_root(paths: list[Path], base: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(base).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_recovery_contract(config: dict[str, Any]) -> None:
    if config.get("schema") != "n2-pvgr-n3-blind-analysis-recovery-1.0":
        raise ValueError("recovery schema drifted")
    if config.get("status") != (
        "blind_analysis_code_recovery_no_threshold_or_cell_change"
    ):
        raise ValueError("recovery status drifted")
    if config.get("numerical_checkpoint_values_inspected_before_recovery_freeze"):
        raise ValueError("blind recovery declaration is not closed")
    if config.get("original_protocol_commit") != (
        "676c23d1962c93f17e8c2d0c0a81332146268790"
    ):
        raise ValueError("original protocol commit drifted")
    actions = config["frozen_recovery_actions"]
    required_true = {
        "reuse_all_hash_validated_cell_checkpoints",
        "rerun_full_interleaved_timing_bundle",
        "publish_recovery_metadata_in_result_and_manifest",
    }
    required_false = {
        "rerun_physical_cells",
        "change_thresholds",
        "change_seeds_or_factorial",
        "exclude_cells",
        "change_bootstrap_or_machine_decisions",
    }
    if any(actions.get(key) is not True for key in required_true):
        raise ValueError("required recovery action drifted")
    if any(actions.get(key) is not False for key in required_false):
        raise ValueError("forbidden recovery action was opened")
    aliases = config["query_count_contract"]["allowed_alias_by_method"]
    if aliases != {
        "picard_1": "total_field_point_queries",
        "picard_2": "total_field_point_queries",
    }:
        raise ValueError("query-count alias contract drifted")
    if int(config["expected_opaque_checkpoint_count"]) != 96:
        raise ValueError("opaque checkpoint count drifted")
    if not all(value is False for value in config["claim_authorizations"].values()):
        raise ValueError("broad claim authorization was opened")


def _validate_recovery_attestation(
    config: dict[str, Any], config_path: Path
) -> dict[str, Any]:
    attestation_path = _resolve(str(config["recovery_attestation"]))
    if not attestation_path.is_file():
        raise FileNotFoundError("committed blind-recovery attestation is missing")
    attestation = _read_json(attestation_path)
    if attestation.get("formal_output_absent_at_creation") is not True:
        raise ValueError("recovery attestation did not preserve blinded publication")
    if attestation.get("checkpoint_payloads_parsed") is not False:
        raise ValueError("recovery attestation parsed checkpoint payloads")
    if attestation.get("recovery_config_sha256") != _sha256(config_path):
        raise ValueError("recovery config hash drifted")
    if int(attestation.get("opaque_checkpoint_count", -1)) != int(
        config["expected_opaque_checkpoint_count"]
    ):
        raise ValueError("opaque checkpoint count no longer matches")

    work = _resolve(str(config["formal_work_output"]))
    checkpoints = list(work.glob(str(config["checkpoint_glob"])))
    if len(checkpoints) != int(config["expected_opaque_checkpoint_count"]):
        raise ValueError("checkpoint set changed after blind recovery attestation")
    if _checkpoint_merkle_root(checkpoints, work) != attestation.get(
        "opaque_checkpoint_merkle_root"
    ):
        raise ValueError("checkpoint bytes changed after blind recovery attestation")

    protocol_commit = str(attestation["recovery_protocol_commit"])
    if _git(
        "merge-base", "--is-ancestor", protocol_commit, "HEAD", check=False
    ).returncode:
        raise ValueError("recovery protocol commit is not an ancestor of HEAD")
    if set(attestation["attested_files"]) != set(config["attested_files"]):
        raise ValueError("recovery attested file set drifted")
    for key, entry in attestation["attested_files"].items():
        expected = str(config["attested_files"][key])
        if entry["path"] != expected:
            raise ValueError(f"recovery attested path drifted for {key}")
        current = _resolve(expected)
        if _sha256(current) != entry["sha256"]:
            raise ValueError(f"recovery attested hash drifted for {key}")
        frozen = _git("show", f"{protocol_commit}:{expected}").stdout
        if hashlib.sha256(frozen).hexdigest() != entry["sha256"]:
            raise ValueError(f"recovery protocol commit hash drifted for {key}")

    relevant = [
        _relative(attestation_path),
        *(str(value) for value in config["attested_files"].values()),
    ]
    tracked = _git(
        "ls-files", "--error-unmatch", _relative(attestation_path), check=False
    )
    if tracked.returncode:
        raise ValueError("recovery attestation is not committed")
    if _git("status", "--porcelain", "--", *relevant).stdout.strip():
        raise ValueError("recovery protocol files have uncommitted changes")
    return attestation


def logical_query_count(
    accounting: dict[str, Any], method: str, recovery_config: dict[str, Any]
) -> int:
    contract = recovery_config["query_count_contract"]
    canonical = str(contract["canonical_field"])
    alias = contract["allowed_alias_by_method"].get(method)
    canonical_present = canonical in accounting
    alias_present = alias is not None and alias in accounting
    if canonical_present and alias_present:
        canonical_value = int(accounting[canonical])
        alias_value = int(accounting[str(alias)])
        if canonical_value != alias_value:
            raise ValueError(f"conflicting query counts for {method}")
        return canonical_value
    if canonical_present:
        return int(accounting[canonical])
    if alias_present:
        return int(accounting[str(alias)])
    raise KeyError(f"no frozen logical query-count field for {method}")


def _query_ratio(
    query_rows: list[dict[str, Any]],
    method: str,
    recovery_config: dict[str, Any],
) -> float:
    return max(
        logical_query_count(row["query_accounting"][method], method, recovery_config)
        / logical_query_count(
            row["query_accounting"]["high128"], "high128", recovery_config
        )
        for row in query_rows
    )


def _query_ratio_vs_ocbh(
    query_rows: list[dict[str, Any]],
    method: str,
    recovery_config: dict[str, Any],
) -> float:
    return max(
        logical_query_count(row["query_accounting"][method], method, recovery_config)
        / logical_query_count(
            row["query_accounting"]["operator_consistent_homotopy"],
            "operator_consistent_homotopy",
            recovery_config,
        )
        for row in query_rows
    )


def _flatten_query_rows(
    query_rows: list[dict[str, Any]], recovery_config: dict[str, Any]
) -> list[dict[str, Any]]:
    flat = []
    for row in query_rows:
        metadata = {
            key: value for key, value in row.items() if key != "query_accounting"
        }
        for method, accounting in row["query_accounting"].items():
            source_field = (
                recovery_config["query_count_contract"]["canonical_field"]
                if recovery_config["query_count_contract"]["canonical_field"]
                in accounting
                else recovery_config["query_count_contract"][
                    "allowed_alias_by_method"
                ].get(method)
            )
            flat.append(
                {
                    **metadata,
                    "method_id": method,
                    **accounting,
                    "logical_scalar_grid_point_queries": logical_query_count(
                        accounting, method, recovery_config
                    ),
                    "logical_query_count_source_field": source_field,
                }
            )
    return flat


def _recovery_metadata(
    recovery_config: dict[str, Any], attestation: dict[str, Any], attestation_path: Path
) -> dict[str, Any]:
    return {
        "schema": str(recovery_config["schema"]),
        "failure_observed": str(recovery_config["failure_observed"]),
        "recovery_protocol_commit": str(attestation["recovery_protocol_commit"]),
        "recovery_attestation": _relative(attestation_path),
        "recovery_attestation_sha256": _sha256(attestation_path),
        "opaque_checkpoint_count": int(attestation["opaque_checkpoint_count"]),
        "opaque_checkpoint_merkle_root": str(
            attestation["opaque_checkpoint_merkle_root"]
        ),
        "query_count_aliases": copy.deepcopy(
            recovery_config["query_count_contract"]["allowed_alias_by_method"]
        ),
        "threshold_seed_cell_bootstrap_changes": False,
        "physical_cells_rerun": False,
        "timing_bundle_rerun": True,
        "claim_boundary": (
            "Blind analysis-code recovery only; no numerical checkpoint payload was "
            "inspected before the recovery protocol was frozen."
        ),
    }


def run_recovery(
    recovery_config_path: Path = DEFAULT_RECOVERY_CONFIG,
) -> dict[str, Any]:
    recovery_config_path = recovery_config_path.resolve()
    recovery_config = _read_json(recovery_config_path)
    _validate_recovery_contract(recovery_config)
    attestation = _validate_recovery_attestation(recovery_config, recovery_config_path)
    formal_output = _resolve(str(recovery_config["formal_output"]))
    if formal_output.exists():
        raise FileExistsError("formal output already exists; recovery is single-use")
    original_config = _resolve(str(recovery_config["original_config"]))
    work = _resolve(str(recovery_config["formal_work_output"]))
    recovery_attestation_path = _resolve(str(recovery_config["recovery_attestation"]))
    recovery_metadata = _recovery_metadata(
        recovery_config, attestation, recovery_attestation_path
    )

    original_query_ratio = grouped._query_ratio
    original_query_ratio_vs_ocbh = grouped._query_ratio_vs_ocbh
    original_flatten = grouped._flatten_query_rows
    original_atomic_json = grouped._atomic_json
    original_write_summary = grouped._write_summary

    def patched_atomic_json(path: Path, payload: dict[str, Any]) -> None:
        enriched = payload
        if path.name == "result.json":
            enriched = copy.deepcopy(payload)
            enriched["analysis_recovery"] = copy.deepcopy(recovery_metadata)
        elif path.name == "manifest.json":
            enriched = copy.deepcopy(payload)
            enriched["analysis_recovery"] = copy.deepcopy(recovery_metadata)
            for key, relative in recovery_config["attested_files"].items():
                source = _resolve(str(relative))
                enriched["files"][f"recovery_attested_{key}"] = {
                    "path": str(relative),
                    "sha256": _sha256(source),
                    "bytes": source.stat().st_size,
                }
            enriched["files"]["recovery_attestation"] = {
                "path": _relative(recovery_attestation_path),
                "sha256": _sha256(recovery_attestation_path),
                "bytes": recovery_attestation_path.stat().st_size,
            }
        original_atomic_json(path, enriched)

    def patched_write_summary(path: Path, result: dict[str, Any]) -> None:
        original_write_summary(path, result)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                "- Analysis recovery: blind query-ledger alias repair; no thresholds, "
                "seeds, cells, bootstrap, or machine decisions changed.\n"
            )

    grouped._query_ratio = lambda rows, method: _query_ratio(
        rows, method, recovery_config
    )
    grouped._query_ratio_vs_ocbh = lambda rows, method: _query_ratio_vs_ocbh(
        rows, method, recovery_config
    )
    grouped._flatten_query_rows = lambda rows: _flatten_query_rows(
        rows, recovery_config
    )
    grouped._atomic_json = patched_atomic_json
    grouped._write_summary = patched_write_summary
    try:
        return grouped.run(
            original_config,
            formal_output,
            work,
            resume=True,
            enforce_formal_output=True,
        )
    finally:
        grouped._query_ratio = original_query_ratio
        grouped._query_ratio_vs_ocbh = original_query_ratio_vs_ocbh
        grouped._flatten_query_rows = original_flatten
        grouped._atomic_json = original_atomic_json
        grouped._write_summary = original_write_summary


def main() -> int:
    result = run_recovery()
    print(json.dumps({"machine_decision": result["machine_decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
