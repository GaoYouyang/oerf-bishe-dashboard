#!/usr/bin/env python3
"""Recover N4.1 artifacts from an attested opaque checkpoint inventory."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from . import run_n2_pvgr_n4_1_execution_amendment as amendment
except ImportError:
    import run_n2_pvgr_n4_1_execution_amendment as amendment


base = amendment.base
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "demo_t16_operator/configs/" "n2_pvgr_n4_1_opaque_artifact_recovery_v1.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=ROOT, check=check, capture_output=True)


def _checkpoint_paths(config: dict[str, Any]) -> list[Path]:
    work = _resolve(str(config["formal_work_output"]))
    return sorted(work.glob(str(config["checkpoint_glob"])))


def _checkpoint_merkle_root(paths: list[Path], work: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(work).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_contract(config: dict[str, Any]) -> None:
    if config.get("schema") != "n2-pvgr-n4-1-opaque-artifact-recovery-1.0":
        raise ValueError("N4.1 recovery schema drifted")
    if config.get("status") != (
        "opaque_checkpoint_artifact_recovery_no_scientific_or_machine_decision_change"
    ):
        raise ValueError("N4.1 recovery status drifted")
    disclosure = config["disclosure"]
    if disclosure.get(
        "checkpoint_payloads_parsed_by_recovery_author_before_protocol_freeze"
    ):
        raise ValueError("N4.1 recovery payload disclosure is not closed")
    if disclosure.get("aggregate_console_gate_statuses_observed") is not True:
        raise ValueError("N4.1 aggregate-status disclosure drifted")
    if disclosure.get("all_32_cell_status_lines_observed") is not True:
        raise ValueError("N4.1 cell-status disclosure drifted")
    if disclosure.get("formal_result_bundle_exists_before_recovery") is not False:
        raise ValueError("N4.1 formal-result disclosure drifted")
    actions = config["frozen_recovery_actions"]
    required_true = {
        "reuse_all_hash_validated_N4_1_level_checkpoints",
        "replace_only_plot_bar_x_input_with_list_of_count_keys",
        "publish_recovery_metadata_in_result_and_manifest",
    }
    required_false = {
        "rerun_any_numerical_level",
        "change_samples_or_thresholds",
        "change_metrics_or_gate_logic",
        "change_per_cell_or_machine_decisions",
        "exclude_cells",
    }
    if any(actions.get(key) is not True for key in required_true):
        raise ValueError("N4.1 required recovery action drifted")
    if any(actions.get(key) is not False for key in required_false):
        raise ValueError("N4.1 forbidden recovery action was opened")
    if int(config["expected_opaque_checkpoint_count"]) != 105:
        raise ValueError("N4.1 recovery checkpoint count drifted")
    if int(config["expected_opaque_h2048_checkpoint_count"]) != 9:
        raise ValueError("N4.1 recovery H2048 count drifted")
    if not all(value is False for value in config["claim_authorizations"].values()):
        raise ValueError("N4.1 recovery broad claim authorization was opened")
    amendment_config = _resolve(str(config["original_amendment_config"]))
    original = _read_json(amendment_config)
    if config["formal_output"] != original["formal_output"]:
        raise ValueError("N4.1 recovery formal output drifted")
    if config["formal_work_output"] != original["formal_work_output"]:
        raise ValueError("N4.1 recovery formal work drifted")
    if _resolve(str(config["original_amendment_attestation"])) != _resolve(
        str(original["pre_registration_attestation"])
    ):
        raise ValueError("N4.1 recovery amendment attestation drifted")
    log = _resolve(str(config["execution_log"]))
    if not log.is_file() or _sha256(log) != config["execution_log_sha256"]:
        raise ValueError("N4.1 recovery execution log hash drifted")


def _validate_attestation(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    path = _resolve(str(config["recovery_attestation"]))
    if not path.is_file():
        raise FileNotFoundError("committed N4.1 recovery attestation is missing")
    attestation = _read_json(path)
    if attestation.get("formal_output_absent_at_creation") is not True:
        raise ValueError("N4.1 recovery attestation did not preserve output absence")
    if attestation.get("checkpoint_payloads_parsed") is not False:
        raise ValueError("N4.1 recovery attestation parsed checkpoint payloads")
    if attestation.get("recovery_config_sha256") != _sha256(config_path):
        raise ValueError("N4.1 recovery config hash drifted")
    paths = _checkpoint_paths(config)
    work = _resolve(str(config["formal_work_output"]))
    if len(paths) != int(config["expected_opaque_checkpoint_count"]):
        raise ValueError("N4.1 recovery checkpoint set count changed")
    h2048_count = sum(path.name == "H2048.json" for path in paths)
    if h2048_count != int(config["expected_opaque_h2048_checkpoint_count"]):
        raise ValueError("N4.1 recovery H2048 set count changed")
    if _checkpoint_merkle_root(paths, work) != attestation.get(
        "opaque_checkpoint_merkle_root"
    ):
        raise ValueError("N4.1 recovery checkpoint bytes changed")
    protocol_commit = str(attestation["recovery_protocol_commit"])
    if _git(
        "merge-base", "--is-ancestor", protocol_commit, "HEAD", check=False
    ).returncode:
        raise ValueError("N4.1 recovery protocol commit is not an ancestor")
    if set(attestation["attested_files"]) != set(config["attested_files"]):
        raise ValueError("N4.1 recovery attested file set drifted")
    for key, entry in attestation["attested_files"].items():
        expected = str(config["attested_files"][key])
        if entry["path"] != expected:
            raise ValueError(f"N4.1 recovery attested path drifted: {key}")
        current = _resolve(expected)
        if _sha256(current) != entry["sha256"]:
            raise ValueError(f"N4.1 recovery current hash drifted: {key}")
        frozen = _git("show", f"{protocol_commit}:{expected}").stdout
        if hashlib.sha256(frozen).hexdigest() != entry["sha256"]:
            raise ValueError(f"N4.1 recovery protocol hash drifted: {key}")
    relevant = [
        _relative(path),
        *(str(value) for value in config["attested_files"].values()),
    ]
    if _git("ls-files", "--error-unmatch", _relative(path), check=False).returncode:
        raise ValueError("N4.1 recovery attestation is not committed")
    if _git("status", "--porcelain", "--", *relevant).stdout.strip():
        raise ValueError("N4.1 recovery protocol files have uncommitted changes")
    return attestation


def _fixed_plot(
    path: Path,
    decisions: list[dict[str, Any]],
    cost_rows: list[dict[str, Any]],
) -> None:
    failure = [item for item in decisions if item["role"] == "n3_failure"]
    controls = [item for item in decisions if item["role"] == "matched_control"]
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    for values, label, color, marker in (
        (failure, "N3 failure", "#b33f40", "o"),
        (controls, "matched control", "#286f6b", "s"),
    ):
        axes[0].scatter(
            [item["h1024_output_contraction"]["contraction_ratio"] for item in values],
            [
                item["h1024_matched_residual_contraction"]["contraction_ratio"]
                for item in values
            ],
            label=label,
            color=color,
            marker=marker,
            alpha=0.85,
        )
    axes[0].axvline(0.5, color="#555", linestyle="--", linewidth=1)
    axes[0].axhline(0.5, color="#555", linestyle="--", linewidth=1)
    axes[0].set(
        xlabel="output contraction",
        ylabel="matched-residual contraction",
        title="H512 to H1024",
    )
    axes[0].legend(frameon=False)
    counts = {
        "H1024 pass": sum(item["h1024_all_gates_pass"] for item in decisions),
        "H2048 escalated": sum(item["requires_h2048_escalation"] for item in decisions),
        "final authorized": sum(
            item["final_cellwise_reference_authorized"] for item in decisions
        ),
    }
    axes[1].bar(
        list(counts.keys()),
        list(counts.values()),
        color=["#286f6b", "#c58b2a", "#4267a8"],
    )
    axes[1].set_ylim(0, 32)
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set(title="Fail-closed cell counts", ylabel="cells")
    by_step: dict[int, list[float]] = {}
    for row in cost_rows:
        by_step.setdefault(int(row["step_count"]), []).append(
            float(row["wall_seconds"])
        )
    steps = sorted(by_step)
    axes[2].plot(
        steps,
        [np.median(by_step[step]) for step in steps],
        marker="o",
        color="#5c4b8a",
    )
    axes[2].set(
        xlabel="RK4 steps H",
        ylabel="median wall seconds",
        title="Observed evaluator cost",
    )
    axes[2].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _recovery_metadata(
    config: dict[str, Any], attestation: dict[str, Any], attestation_path: Path
) -> dict[str, Any]:
    return {
        "schema": config["schema"],
        "failure_observed": config["failure_observed"],
        "recovery_protocol_commit": attestation["recovery_protocol_commit"],
        "recovery_attestation": _relative(attestation_path),
        "recovery_attestation_sha256": _sha256(attestation_path),
        "opaque_checkpoint_count": attestation["opaque_checkpoint_count"],
        "opaque_h2048_checkpoint_count": attestation["opaque_h2048_checkpoint_count"],
        "opaque_checkpoint_merkle_root": attestation["opaque_checkpoint_merkle_root"],
        "checkpoint_payloads_parsed_before_recovery_freeze": False,
        "aggregate_console_gate_statuses_observed": True,
        "numerical_levels_rerun": False,
        "scientific_or_machine_decision_change": False,
        "plot_change": "bar_x_dict_to_list_of_keys_only",
        "claim_boundary": (
            "Opaque checkpoint artifact recovery after aggregate statuses were visible; "
            "no numerical level, gate, threshold, cell, or decision changed."
        ),
    }


def run_recovery(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = _read_json(config_path)
    _validate_contract(config)
    attestation = _validate_attestation(config, config_path)
    formal_output = _resolve(str(config["formal_output"]))
    if formal_output.exists():
        raise FileExistsError(
            "N4.1 formal output already exists; recovery is single-use"
        )
    work = _resolve(str(config["formal_work_output"]))
    amendment_config = _resolve(str(config["original_amendment_config"]))
    attestation_path = _resolve(str(config["recovery_attestation"]))
    metadata = _recovery_metadata(config, attestation, attestation_path)

    original_plot = base._plot
    original_load = base._load_or_run_level
    original_atomic = base._atomic_json
    original_summary = amendment._summary_markdown

    def checkpoint_only_load(*args: Any, **kwargs: Any) -> dict[str, Any]:
        cell = args[0]
        step_count = int(kwargs["step_count"])
        checkpoint = base._level_path(work, cell["cell_id"], step_count)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"recovery refuses to run missing numerical level: {checkpoint}"
            )
        return original_load(*args, **kwargs)

    def enriched_atomic(path: Path, payload: dict[str, Any]) -> None:
        enriched = payload
        if path.name == "result.json":
            enriched = copy.deepcopy(payload)
            enriched["artifact_recovery"] = copy.deepcopy(metadata)
        elif path.name == "manifest.json":
            enriched = copy.deepcopy(payload)
            enriched["artifact_recovery"] = copy.deepcopy(metadata)
            for key, relative in config["attested_files"].items():
                source = _resolve(str(relative))
                enriched["files"][f"recovery_attested_{key}"] = {
                    "path": str(relative),
                    "sha256": _sha256(source),
                    "bytes": source.stat().st_size,
                }
            enriched["files"]["recovery_attestation"] = {
                "path": _relative(attestation_path),
                "sha256": _sha256(attestation_path),
                "bytes": attestation_path.stat().st_size,
            }
        original_atomic(path, enriched)

    def enriched_summary(result: dict[str, Any]) -> str:
        return original_summary(result) + (
            "\n- Artifact recovery: attested opaque-checkpoint reuse; only the "
            "Matplotlib bar x input changed from a dict to its list of keys.\n"
        )

    base._plot = _fixed_plot
    base._load_or_run_level = checkpoint_only_load
    base._atomic_json = enriched_atomic
    amendment._summary_markdown = enriched_summary
    try:
        return amendment.run(
            amendment_config,
            formal_output,
            work,
            resume=True,
            enforce_formal_output=True,
        )
    finally:
        base._plot = original_plot
        base._load_or_run_level = original_load
        base._atomic_json = original_atomic
        amendment._summary_markdown = original_summary


def main() -> int:
    result = run_recovery()
    print(json.dumps({"machine_decision": result["machine_decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
