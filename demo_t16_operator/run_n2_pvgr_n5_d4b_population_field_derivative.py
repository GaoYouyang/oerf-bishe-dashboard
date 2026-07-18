#!/usr/bin/env python3
"""Run the preregistered N2-PVGR N5-D4b 32-cell derivative census."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .d4b_frozen_inputs import array_sha256
    from .field_jvp_vjp_gate import ClosureDerivativeAudit, audit_tensor_closure
    from .run_n2_pvgr_n5_d4_tiny_field_derivative import (
        _closures,
        _peak_rss_bytes,
        _rig,
        _serialize_audit,
        _tensor_metrics,
        _topology_rows,
    )
except ImportError:
    from d4b_frozen_inputs import array_sha256
    from field_jvp_vjp_gate import ClosureDerivativeAudit, audit_tensor_closure
    from run_n2_pvgr_n5_d4_tiny_field_derivative import (
        _closures,
        _peak_rss_bytes,
        _rig,
        _serialize_audit,
        _tensor_metrics,
        _topology_rows,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "demo_t16_operator/configs/"
    "n2_pvgr_n5_d4b_population_field_derivative_preregistered_v1.json"
)

DERIVATIVE_GATE_KEYS = (
    "maximum_dot_relative_defect",
    "minimum_dot_signal_absolute",
    "maximum_best_finite_difference_relative_l2",
    "maximum_required_h_finite_difference_relative_l2",
    "maximum_repeat_output_relative_l2",
    "maximum_raw_structural_output_relative_l2",
    "maximum_raw_structural_jvp_relative_l2",
    "maximum_raw_structural_vjp_relative_l2",
    "maximum_raw_paired_output_relative_l2",
    "maximum_raw_paired_jvp_relative_l2",
    "maximum_raw_paired_vjp_relative_l2",
    "structural_absolute_tolerance_floor",
    "required_finite_fraction",
    "minimum_domain_margin",
    "minimum_stencil_margin",
    "maximum_direction_norm_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _path_occupied(path: Path) -> bool:
    return os.path.lexists(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=ROOT, check=check, capture_output=True)


def recompute_budget(config: dict[str, Any]) -> dict[str, int]:
    rays = int(config["ray_count"])
    steps = int(config["step_count"])
    cells = int(config["population_contract"]["expected_cell_count"])
    directions = int(config["direction_count_per_cell"])
    maps = len(config["map_order"])
    h_count = len(config["finite_difference"]["h_values"])
    closure_invocations = 2 + 2 * h_count
    direction_contexts = cells * directions
    map_query_coefficient = 35 + 7 + 42 + 42
    derivative_queries = (
        direction_contexts * closure_invocations * map_query_coefficient * rays * steps
    )
    curved_dispatches = 28 * steps + 7
    straight_dispatches = 7
    raw_dispatches = curved_dispatches + straight_dispatches
    paired_dispatches = 28 * steps + 1
    derivative_dispatches = (
        direction_contexts
        * closure_invocations
        * (curved_dispatches + straight_dispatches + raw_dispatches + paired_dispatches)
    )
    signatures = direction_contexts * (1 + 2 * h_count)
    topology_queries = signatures * 70 * rays * steps
    topology_dispatches = signatures * (28 * steps + 4 * steps + 2)
    return {
        "derivative_map_logical_queries": derivative_queries,
        "derivative_map_interpolation_dispatches": derivative_dispatches,
        "topology_logical_queries": topology_queries,
        "topology_interpolation_dispatches": topology_dispatches,
        "d4b_total_logical_queries": derivative_queries + topology_queries,
        "d4b_total_interpolation_dispatches": derivative_dispatches
        + topology_dispatches,
        "map_closure_invocations": direction_contexts * maps * closure_invocations,
        "jvp_sweeps": direction_contexts * maps,
        "vjp_sweeps": direction_contexts * maps,
        "finite_difference_forward_calls": direction_contexts * maps * 2 * h_count,
        "topology_signature_invocations": signatures,
    }


def _validate_contract(config: dict[str, Any]) -> None:
    if (
        config.get("schema")
        != "n2-pvgr-n5-d4b-population-field-derivative-preregistered-1.0"
    ):
        raise ValueError("N5-D4b schema drifted")
    if config.get("candidate_id") != (
        "N2-PVGR-N5-D4B-CLOSED-POPULATION-FIELD-DERIVATIVE"
    ):
        raise ValueError("N5-D4b candidate identifier drifted")
    if (config.get("device"), config.get("dtype")) != ("cpu", "float64"):
        raise ValueError("N5-D4b requires CPU float64")
    if [
        int(config[key])
        for key in (
            "ray_count",
            "source_population_count",
            "step_count",
            "direction_count_per_cell",
        )
    ] != [4, 256, 16, 2]:
        raise ValueError("N5-D4b ray/step/direction contract drifted")
    population = config["population_contract"]
    if int(population.get("expected_cell_count", 0)) != 32:
        raise ValueError("N5-D4b census must contain 32 cells")
    if population.get("expected_pair_ids") != [
        f"p{index:02d}" for index in range(1, 17)
    ]:
        raise ValueError("N5-D4b pair census drifted")
    if population.get("expected_role_order_within_each_pair") != [
        "n3_failure",
        "matched_control",
    ]:
        raise ValueError("N5-D4b pair-role order drifted")
    if population.get("required_reporting_levels") != [
        "cell",
        "pair_cluster",
        "field_unit_cluster",
    ]:
        raise ValueError("N5-D4b reporting levels drifted")
    if population.get("closed_development_population_not_iid_sample") is not True:
        raise ValueError("N5-D4b cannot claim an iid population")
    if population.get("no_cell_or_pair_exclusion_after_preregistration") is not True:
        raise ValueError("N5-D4b cell exclusion boundary drifted")
    if config.get("map_order") != [
        "curved_detector",
        "straight_detector",
        "raw_curved_minus_straight",
        "paired_neumaier_residual",
    ]:
        raise ValueError("N5-D4b map order drifted")
    if config["direction_contract"].get("field_direction_modes_in_order") != [
        "smooth_avg_pool3d_kernel3_stride1_padding1",
        "raw_torch_randn",
    ]:
        raise ValueError("N5-D4b direction modes drifted")
    if [
        int(config["direction_contract"][key])
        for key in ("seed_base", "cotangent_seed_base")
    ] != [66001, 66101]:
        raise ValueError("N5-D4b frozen seed bases drifted")
    if config["finite_difference"].get("h_values_float64_hex") != [
        float(value).hex() for value in config["finite_difference"]["h_values"]
    ]:
        raise ValueError("N5-D4b float64 h bytes drifted")

    parent_d4_config = _read_json(_resolve(str(config["parent_d4_config"])))
    for key in (
        "device",
        "dtype",
        "ray_count",
        "source_population_count",
        "step_count",
        "direction_count_per_cell",
        "map_order",
        "observable_contract",
        "finite_difference",
        "topology_contract",
    ):
        if config[key] != parent_d4_config[key]:
            raise ValueError(f"N5-D4b changed frozen D4 contract: {key}")
    if [
        config["direction_contract"]["seed_base"],
        config["direction_contract"]["cotangent_seed_base"],
    ] == [
        parent_d4_config["direction_contract"]["seed_base"],
        parent_d4_config["direction_contract"]["cotangent_seed_base"],
    ]:
        raise ValueError("N5-D4b must use fresh preregistered direction seed bases")
    for key in DERIVATIVE_GATE_KEYS:
        if config["gates"].get(key) != parent_d4_config["gates"].get(key):
            raise ValueError(f"N5-D4b changed D4 gate threshold: {key}")
    if config["gates"].get("threshold_source") != (
        "copied_unchanged_from_attested_n5_d4_before_d4b_results"
    ):
        raise ValueError("N5-D4b threshold provenance drifted")

    budget = recompute_budget(config)
    contract = config["budget_contract"]
    expected_mapping = {
        "expected_derivative_map_logical_queries": "derivative_map_logical_queries",
        "expected_derivative_map_interpolation_dispatches": "derivative_map_interpolation_dispatches",
        "expected_topology_logical_queries": "topology_logical_queries",
        "expected_topology_interpolation_dispatches": "topology_interpolation_dispatches",
        "expected_total_logical_queries": "d4b_total_logical_queries",
        "expected_total_interpolation_dispatches": "d4b_total_interpolation_dispatches",
        "expected_map_closure_invocations": "map_closure_invocations",
        "expected_jvp_sweeps": "jvp_sweeps",
        "expected_vjp_sweeps": "vjp_sweeps",
        "expected_finite_difference_forward_calls": "finite_difference_forward_calls",
        "expected_topology_signature_invocations": "topology_signature_invocations",
    }
    for config_key, budget_key in expected_mapping.items():
        if int(contract[config_key]) != budget[budget_key]:
            raise ValueError(f"N5-D4b budget drifted: {config_key}")
    if (
        int(
            contract[
                "lineage_total_logical_queries_including_once_only_parent_toy_control"
            ]
        )
        != budget["d4b_total_logical_queries"] + 3360
    ):
        raise ValueError("N5-D4b lineage logical-query accounting drifted")
    if (
        int(
            contract[
                "lineage_total_interpolation_dispatches_including_once_only_parent_toy_control"
            ]
        )
        != budget["d4b_total_interpolation_dispatches"] + 686
    ):
        raise ValueError("N5-D4b lineage dispatch accounting drifted")
    if (
        contract.get("parent_d4_strong_toy_control_is_frozen_and_not_repeated")
        is not True
    ):
        raise ValueError("N5-D4b must not repeat the frozen D4 toy control")
    if any(config["claim_authorizations"].values()):
        raise ValueError("N5-D4b preregistered claims must all remain false")
    bundle = _resolve(str(config["pre_registration_bundle"]))
    if {
        _resolve(str(config["pre_registration_attestation"])).parent,
        _resolve(str(config["frozen_input_archive"])).parent,
        _resolve(str(config["pre_registration_ready_marker"])).parent,
    } != {bundle}:
        raise ValueError("N5-D4b preregistration bundle path contract drifted")


def _validate_preregistration(
    config: dict[str, Any], config_path: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    bundle_path = _resolve(str(config["pre_registration_bundle"]))
    attestation_path = _resolve(str(config["pre_registration_attestation"]))
    archive_path = _resolve(str(config["frozen_input_archive"]))
    ready_path = _resolve(str(config["pre_registration_ready_marker"]))
    if (
        not bundle_path.is_dir()
        or {attestation_path.parent, archive_path.parent, ready_path.parent}
        != {bundle_path}
        or not attestation_path.is_file()
        or not archive_path.is_file()
        or not ready_path.is_file()
    ):
        raise FileNotFoundError(
            "committed complete N5-D4b preregistration bundle is missing"
        )
    attestation = _read_json(attestation_path)
    ready = _read_json(ready_path)
    if attestation.get("schema") != (
        "n2-pvgr-n5-d4b-population-field-derivative-attestation-1.0"
    ):
        raise ValueError("N5-D4b attestation schema drifted")
    if attestation.get("formal_results_absent_at_creation") is not True:
        raise ValueError("N5-D4b attestation does not prove output absence")
    if attestation.get("formal_work_output_absent_at_creation") is not True:
        raise ValueError("N5-D4b attestation does not prove work-output absence")
    if attestation.get("formal_output") != config["formal_output"]:
        raise ValueError("N5-D4b attestation output path drifted")
    if attestation.get("formal_work_output") != config["formal_work_output"]:
        raise ValueError("N5-D4b attestation work-output path drifted")
    if attestation.get("pre_registration_bundle") != config["pre_registration_bundle"]:
        raise ValueError("N5-D4b attestation bundle path drifted")
    if (
        attestation.get("pre_registration_ready_marker")
        != config["pre_registration_ready_marker"]
    ):
        raise ValueError("N5-D4b attestation READY path drifted")
    if attestation.get("bundle_publication") != (
        "single_atomic_directory_rename_after_READY"
    ):
        raise ValueError("N5-D4b bundle publication contract drifted")
    if _sha256_bytes(config_path.read_bytes()) != attestation.get("config_sha256"):
        raise ValueError("N5-D4b config differs from attestation")
    protocol_commit = str(attestation.get("protocol_commit", ""))
    if len(protocol_commit) != 40:
        raise ValueError("N5-D4b protocol commit drifted")
    ancestor = _git("merge-base", "--is-ancestor", protocol_commit, "HEAD", check=False)
    if ancestor.returncode != 0:
        raise ValueError("N5-D4b protocol commit is not an ancestor of HEAD")
    if ready != {
        "schema": "n2-pvgr-n5-d4b-preregistration-ready-1.0",
        "protocol_commit": protocol_commit,
        "config_sha256": _sha256(config_path),
        "attestation_path": config["pre_registration_attestation"],
        "attestation_sha256": _sha256(attestation_path),
        "frozen_input_archive_path": config["frozen_input_archive"],
        "frozen_input_archive_sha256": _sha256(archive_path),
        "publication_rule": "atomically_rename_complete_staging_directory",
    }:
        raise ValueError("N5-D4b READY marker or atomic bundle binding drifted")
    if set(attestation.get("attested_files", {})) != set(config["attested_files"]):
        raise ValueError("N5-D4b attested file set drifted")
    for key, relative in config["attested_files"].items():
        path = _resolve(str(relative))
        record = attestation["attested_files"][key]
        if record.get("path") != relative or _sha256(path) != record.get("sha256"):
            raise ValueError(f"N5-D4b attested file drifted: {key}")
        frozen = _git("show", f"{protocol_commit}:{relative}")
        if frozen.returncode != 0 or _sha256_bytes(frozen.stdout) != record.get(
            "sha256"
        ):
            raise ValueError(f"N5-D4b protocol file hash drifted: {key}")
    if attestation.get("frozen_input_archive") != config["frozen_input_archive"]:
        raise ValueError("N5-D4b frozen-input path drifted")
    if _sha256(archive_path) != attestation.get("frozen_input_archive_sha256"):
        raise ValueError("N5-D4b frozen-input archive hash drifted")
    status_paths = [str(value) for value in config["attested_files"].values()]
    status_paths.extend((str(attestation_path), str(archive_path), str(ready_path)))
    dirty = _git("status", "--porcelain", "--", *status_paths).stdout.decode().strip()
    if dirty:
        raise ValueError("N5-D4b preregistered files have uncommitted changes")

    with np.load(archive_path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    inventory = attestation["frozen_input_inventory"]
    if set(arrays) != set(inventory):
        raise ValueError("N5-D4b frozen-input inventory drifted")
    for key, value in arrays.items():
        record = inventory[key]
        if list(value.shape) != record["shape"] or str(value.dtype) != record["dtype"]:
            raise ValueError(f"N5-D4b frozen-input shape/dtype drifted: {key}")
        if array_sha256(value) != record["sha256_float64_le_c_order"]:
            raise ValueError(f"N5-D4b frozen-input array hash drifted: {key}")
    metadata = attestation["frozen_input_metadata"]
    if _sha256_bytes(_canonical_json(metadata)) != attestation.get(
        "frozen_input_metadata_sha256"
    ):
        raise ValueError("N5-D4b frozen-input metadata hash drifted")
    return attestation, arrays


def _structural_controls(
    audits: dict[str, ClosureDerivativeAudit], gates: dict[str, Any]
) -> tuple[dict[str, Any], bool, bool]:
    curved = audits["curved_detector"].autodiff
    straight = audits["straight_detector"].autodiff
    raw = audits["raw_curved_minus_straight"].autodiff
    paired = audits["paired_neumaier_residual"].autodiff
    raw_structure = {
        "primal": _tensor_metrics(
            raw.primal,
            curved.primal - straight.primal,
            scale_tensors=(curved.primal, straight.primal),
        ),
        "jvp": _tensor_metrics(
            raw.jvp,
            curved.jvp - straight.jvp,
            scale_tensors=(curved.jvp, straight.jvp),
        ),
        "vjp": _tensor_metrics(
            raw.vjp,
            curved.vjp - straight.vjp,
            scale_tensors=(curved.vjp, straight.vjp),
        ),
        "evidence_role": "wiring_invariant_not_independent_physics_evidence",
    }
    paired_structure = {
        "primal": _tensor_metrics(
            paired.primal,
            raw.primal,
            scale_tensors=(curved.primal, straight.primal),
        ),
        "jvp": _tensor_metrics(
            paired.jvp, raw.jvp, scale_tensors=(curved.jvp, straight.jvp)
        ),
        "vjp": _tensor_metrics(
            paired.vjp, raw.vjp, scale_tensors=(curved.vjp, straight.vjp)
        ),
        "evidence_role": "accumulation_order_control_not_new_physics_model",
    }
    floor = float(gates["structural_absolute_tolerance_floor"])

    def passes(metric: dict[str, float | bool], rtol: float) -> bool:
        return bool(
            metric["finite"]
            and metric["absolute_l2"] <= floor + rtol * float(metric["scale"])
        )

    raw_gate = bool(
        passes(
            raw_structure["primal"],
            float(gates["maximum_raw_structural_output_relative_l2"]),
        )
        and passes(
            raw_structure["jvp"], float(gates["maximum_raw_structural_jvp_relative_l2"])
        )
        and passes(
            raw_structure["vjp"], float(gates["maximum_raw_structural_vjp_relative_l2"])
        )
    )
    paired_gate = bool(
        passes(
            paired_structure["primal"],
            float(gates["maximum_raw_paired_output_relative_l2"]),
        )
        and passes(
            paired_structure["jvp"], float(gates["maximum_raw_paired_jvp_relative_l2"])
        )
        and passes(
            paired_structure["vjp"], float(gates["maximum_raw_paired_vjp_relative_l2"])
        )
    )
    return (
        {
            "raw_equals_curved_minus_straight": raw_structure | {"gate": raw_gate},
            "paired_equals_raw_control": paired_structure | {"gate": paired_gate},
        },
        raw_gate,
        paired_gate,
    )


def _cluster_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    summary = []
    for cluster_id, members in sorted(groups.items()):
        map_gates = [
            bool(payload["map_gate"])
            for row in members
            for payload in row["maps"].values()
        ]
        structure_gates = [
            bool(payload["gate"])
            for row in members
            for payload in row["structure"].values()
        ]
        topology_gates = [bool(row["topology_gate"]) for row in members]
        summary.append(
            {
                "cluster_id": cluster_id,
                "direction_context_count": len(members),
                "map_context_count": len(map_gates),
                "passing_map_context_count": sum(map_gates),
                "structural_control_count": len(structure_gates),
                "passing_structural_control_count": sum(structure_gates),
                "topology_context_count": len(topology_gates),
                "passing_topology_context_count": sum(topology_gates),
                "all_gates_pass": bool(
                    all(map_gates) and all(structure_gates) and all(topology_gates)
                ),
            }
        )
    return summary


def _extrema(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    maps = [payload for row in rows for payload in row["maps"].values()]
    required = [
        level for payload in maps for level in payload["levels"] if level["required"]
    ]
    best_per_map = [
        min(level["relative_error"] for level in payload["levels"]) for payload in maps
    ]
    topology = [row["topology"] for row in rows]
    all_signatures = [item["base"] for item in topology] + [
        item for context in topology for item in context["perturbations"]
    ]
    floor = float(config["gates"]["minimum_dot_signal_absolute"])
    return {
        "worst_dot_relative_defect": max(
            float(item["dot_relative_defect"]) for item in maps
        ),
        "minimum_dot_signal_absolute": min(float(item["dot_signal"]) for item in maps),
        "minimum_dot_signal_over_floor": min(
            float(item["dot_signal"]) / floor for item in maps
        ),
        "worst_best_h_finite_difference_relative_l2": max(best_per_map),
        "worst_required_h_finite_difference_relative_l2": max(
            float(item["relative_error"]) for item in required
        ),
        "minimum_domain_margin": min(
            float(item["minimum_domain_margin"]) for item in all_signatures
        ),
        "minimum_stencil_margin": min(
            float(item["minimum_stencil_margin"]) for item in all_signatures
        ),
        "minimum_frustum_margin": min(
            float(item["minimum_frustum_margin"]) for item in all_signatures
        ),
        "maximum_direction_norm_error": max(
            float(item["maximum_direction_norm_error"]) for item in all_signatures
        ),
    }


def _figure(rows: list[dict[str, Any]], result: dict[str, Any], path: Path) -> None:
    by_cell: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[int(row["cell_index"])].append(row)
    indices = sorted(by_cell)
    labels = [by_cell[index][0]["pair_id"] for index in indices]
    dot = []
    fd = []
    margins = []
    pass_fraction = []
    for index in indices:
        members = by_cell[index]
        maps = [payload for row in members for payload in row["maps"].values()]
        dot.append(max(float(payload["dot_relative_defect"]) for payload in maps))
        fd.append(
            max(
                min(float(level["relative_error"]) for level in payload["levels"])
                for payload in maps
            )
        )
        margins.append(
            min(
                float(row["topology"]["base"]["minimum_stencil_margin"])
                for row in members
            )
        )
        gates = [bool(payload["map_gate"]) for payload in maps]
        gates.extend(bool(row["topology_gate"]) for row in members)
        gates.extend(
            bool(payload["gate"])
            for row in members
            for payload in row["structure"].values()
        )
        pass_fraction.append(sum(gates) / len(gates))

    x = np.arange(len(indices))
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    axes[0, 0].semilogy(x, np.maximum(dot, 1e-30), "o-", color="#006d77", ms=4)
    axes[0, 0].axhline(1e-10, color="#b23a48", ls="--", lw=1.2, label="gate")
    axes[0, 0].set_title("Worst dot-product defect per cell")
    axes[0, 0].set_ylabel("relative defect")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].semilogy(x, np.maximum(fd, 1e-30), "o-", color="#355070", ms=4)
    axes[0, 1].axhline(1e-6, color="#b23a48", ls="--", lw=1.2, label="gate")
    axes[0, 1].set_title("Worst best-h finite-difference error per cell")
    axes[0, 1].set_ylabel("relative L2")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].plot(x, margins, "o-", color="#e29578", ms=4)
    axes[1, 0].axhline(0.0, color="#b23a48", ls="--", lw=1.2)
    axes[1, 0].set_title("Minimum base stencil margin per cell")
    axes[1, 0].set_ylabel("normalized domain margin")
    axes[1, 1].bar(x, pass_fraction, color="#5b8e7d", width=0.75)
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].set_title("Strict gate pass fraction per cell")
    axes[1, 1].set_ylabel("fraction")
    for axis in axes.flat:
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=90, fontsize=7)
        axis.grid(alpha=0.18)
    fig.suptitle(
        "N5-D4b closed 32-cell field-derivative census\n"
        f"Decision: {result['machine_decision']}",
        fontsize=15,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_manifest(
    output: Path,
    config_path: Path,
    attestation_path: Path,
    archive_path: Path,
    ready_path: Path,
    *,
    published_output: Path,
) -> None:
    files: dict[str, dict[str, str]] = {}
    for label, source in (
        ("config", config_path),
        ("attestation", attestation_path),
        ("frozen_input_archive", archive_path),
        ("pre_registration_ready_marker", ready_path),
    ):
        files[label] = {"path": _relative(source), "sha256": _sha256(source)}
    for source in sorted(output.iterdir()):
        if (
            source.name in {"manifest.json", "validation_report.json"}
            or not source.is_file()
        ):
            continue
        files[source.name] = {
            "path": _relative(published_output / source.name),
            "sha256": _sha256(source),
        }
    _write_json(
        output / "manifest.json",
        {
            "schema": "n2-pvgr-n5-d4b-population-field-derivative-manifest-1.0",
            "files": files,
            "post_validation_artifacts_excluded_from_manifest": [
                "validation_report.json"
            ],
        },
    )


def run(config_path: Path, output: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = _read_json(config_path)
    _validate_contract(config)
    attestation, frozen_arrays = _validate_preregistration(config, config_path)
    metadata = attestation["frozen_input_metadata"]
    formal_output = output.resolve()
    work_output = _resolve(str(config["formal_work_output"]))
    if formal_output != _resolve(str(config["formal_output"])):
        raise ValueError(
            "N5-D4b output path must match the preregistered formal output"
        )
    if _path_occupied(formal_output) or _path_occupied(work_output):
        raise FileExistsError(
            "N5-D4b formal/work output already exists; formal reruns are forbidden"
        )
    work_output.mkdir(parents=True)
    output = work_output
    start = time.perf_counter()
    peak_before = _peak_rss_bytes()
    result_arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    all_map_gates: list[bool] = []
    all_structure_gates: list[bool] = []
    topology_gates: list[bool] = []
    h_values = [float(value) for value in config["finite_difference"]["h_values"]]
    gates = config["gates"]

    for context in metadata["contexts"]:
        cell_index = int(context["cell_index"])
        field_array = frozen_arrays[str(context["field_key"])]
        rays_array = frozen_arrays[str(context["rays_key"])]
        if array_sha256(field_array) != context["field_sha256"]:
            raise ValueError("N5-D4b frozen field metadata hash drifted")
        if array_sha256(rays_array) != context["rays_sha256"]:
            raise ValueError("N5-D4b frozen ray metadata hash drifted")
        field = torch.from_numpy(field_array.copy()).to(torch.float64)
        rays = torch.from_numpy(rays_array.copy()).to(torch.float64)
        rig = _rig(context["rig"])
        closures = _closures(
            rays,
            rig,
            difference_step=float(context["difference_step"]),
            refractivity_scale=float(context["refractivity_scale"]),
            step_count=int(config["step_count"]),
        )
        for direction_entry in context["directions"]:
            direction_index = int(direction_entry["direction_index"])
            direction = torch.from_numpy(
                frozen_arrays[str(direction_entry["direction_key"])].copy()
            ).to(torch.float64)
            cotangent = torch.from_numpy(
                frozen_arrays[str(direction_entry["cotangent_key"])].copy()
            ).to(torch.float64)
            if array_sha256(direction) != direction_entry["direction_sha256"]:
                raise ValueError("N5-D4b frozen direction hash drifted")
            if array_sha256(cotangent) != direction_entry["cotangent_sha256"]:
                raise ValueError("N5-D4b frozen cotangent hash drifted")
            topology, topology_gate = _topology_rows(
                field, direction, rays, rig, context, config
            )
            topology_gates.append(topology_gate)
            map_rows: dict[str, Any] = {}
            audits: dict[str, ClosureDerivativeAudit] = {}
            context_start = time.perf_counter()
            for map_id in config["map_order"]:
                audit = audit_tensor_closure(
                    closures[map_id],
                    field,
                    direction,
                    cotangent,
                    h_values,
                    denominator_floor=1e-30,
                    nondegenerate_floor=float(gates["minimum_dot_signal_absolute"]),
                )
                prefix = f"c{cell_index:02d}_d{direction_index:02d}_{map_id}"
                payload, map_gate = _serialize_audit(
                    audit, prefix=prefix, arrays=result_arrays, config=config
                )
                map_rows[map_id] = payload
                audits[map_id] = audit
                all_map_gates.append(map_gate)
            structure, raw_gate, paired_gate = _structural_controls(audits, gates)
            all_structure_gates.extend((raw_gate, paired_gate))
            rows.append(
                {
                    "cell_index": cell_index,
                    "cell_id": context["cell_id"],
                    "pair_id": context["pair_id"],
                    "role": context["role"],
                    "field_unit_id": context["field_unit_id"],
                    "family_id": context["family_id"],
                    "dimensionless_stress_multiplier": context[
                        "dimensionless_stress_multiplier"
                    ],
                    "d3_identity_sha256": context["d3_identity_sha256"],
                    "d3_reference_method": context["d3_reference_method"],
                    "direction_index": direction_index,
                    "field_sha256": context["field_sha256"],
                    "rays_sha256": context["rays_sha256"],
                    "direction_sha256": direction_entry["direction_sha256"],
                    "cotangent_sha256": direction_entry["cotangent_sha256"],
                    "topology": topology,
                    "topology_gate": topology_gate,
                    "maps": map_rows,
                    "structure": structure,
                    "wall_seconds": time.perf_counter() - context_start,
                }
            )

    expected_cells = int(config["population_contract"]["expected_cell_count"])
    expected_direction_contexts = expected_cells * int(
        config["direction_count_per_cell"]
    )
    expected_map_contexts = expected_direction_contexts * len(config["map_order"])
    expected_structure_gates = expected_direction_contexts * 2
    context_changed = not all(topology_gates)
    ordinary_pass = bool(
        len(rows) == expected_direction_contexts
        and len(all_map_gates) == expected_map_contexts
        and len(all_structure_gates) == expected_structure_gates
        and len(topology_gates) == expected_direction_contexts
        and all(all_map_gates)
        and all(all_structure_gates)
        and all(topology_gates)
    )
    if context_changed:
        decision = config["decision_contract"]["context_change_decision"]
    elif ordinary_pass:
        decision = config["decision_contract"]["pass_decision"]
    else:
        decision = config["decision_contract"]["failure_decision"]
    wall_seconds = time.perf_counter() - start
    peak_after = _peak_rss_bytes()
    authorizations = dict(config["claim_authorizations"])
    authorizations["closed_n4_32_cell_field_derivative_contract"] = bool(
        decision == config["decision_contract"]["pass_decision"]
    )
    budget = recompute_budget(config)
    parent_d4 = _read_json(_resolve(str(config["parent_d4_result"])))
    parent_d4_validation = _read_json(_resolve(str(config["parent_d4_validation"])))
    result = {
        "schema": "n2-pvgr-n5-d4b-population-field-derivative-result-1.0",
        "candidate_id": config["candidate_id"],
        "protocol_commit": attestation["protocol_commit"],
        "machine_decision": decision,
        "closed_development_population_only": True,
        "iid_population_claim": False,
        "grid_field_derivative_core_only": True,
        "decoder_parameterization_tested": False,
        "counts": {
            "cell_count": expected_cells,
            "pair_count": int(metadata["pair_count"]),
            "field_unit_count": int(metadata["field_unit_count"]),
            "direction_context_count": len(rows),
            "map_context_count": len(all_map_gates),
            "passing_map_context_count": sum(all_map_gates),
            "structural_control_count": len(all_structure_gates),
            "passing_structural_control_count": sum(all_structure_gates),
            "combined_structure_context_count": len(rows),
            "passing_combined_structure_context_count": sum(
                all(bool(payload["gate"]) for payload in row["structure"].values())
                for row in rows
            ),
            "topology_context_count": len(topology_gates),
            "stable_topology_context_count": sum(topology_gates),
        },
        "all_map_gates_pass": all(all_map_gates),
        "all_structure_gates_pass": all(all_structure_gates),
        "all_topology_gates_pass": all(topology_gates),
        "cluster_accounting": {
            "statistical_independence_claimed": False,
            "cell": _cluster_summary(rows, "cell_id"),
            "pair_cluster": _cluster_summary(rows, "pair_id"),
            "field_unit_cluster": _cluster_summary(rows, "field_unit_id"),
            "family": _cluster_summary(rows, "family_id"),
        },
        "extrema": _extrema(rows, config),
        "parent_d4_evidence_reused_not_recomputed": {
            "result_sha256": _sha256(_resolve(str(config["parent_d4_result"]))),
            "validation_sha256": _sha256(_resolve(str(config["parent_d4_validation"]))),
            "machine_decision": parent_d4["machine_decision"],
            "validation_valid": parent_d4_validation["valid"],
            "strong_toy_detach_control_gate": parent_d4["strong_toy_detach_control"][
                "gate"
            ],
        },
        "budget": {
            **budget,
            "lineage_total_logical_queries_including_once_only_parent_toy_control": budget[
                "d4b_total_logical_queries"
            ]
            + 3360,
            "lineage_total_interpolation_dispatches_including_once_only_parent_toy_control": budget[
                "d4b_total_interpolation_dispatches"
            ]
            + 686,
            "failed_or_retried_calls": 0,
            "host_sync_count": "not_instrumented_not_claimed_zero",
            "saved_tensor_bytes": "not_instrumented_not_claimed_zero",
        },
        "timing": {
            "wall_seconds": wall_seconds,
            "peak_rss_before_bytes": peak_before,
            "peak_rss_bytes": peak_after,
            "peak_rss_increment_bytes": max(peak_after - peak_before, 0),
            "fresh_subprocess_peak_rss": False,
        },
        "authorizations": authorizations,
        "limitations": config["known_limitations"],
        "next_protocol_boundary": (
            "preregister_decoder_parameter_chain_derivative_gate"
            if authorizations["closed_n4_32_cell_field_derivative_contract"]
            else "do_not_enter_decoder_chain_reconstruction_or_training"
        ),
    }
    _write_json(output / "config_snapshot.json", config)
    _write_json(output / "rows.json", rows)
    _write_json(output / "result.json", result)
    np.savez(output / "derivative_arrays.npz", **result_arrays)
    _write_json(
        output / "array_inventory.json",
        {
            "schema": "n2-pvgr-n5-d4b-derivative-array-inventory-1.0",
            "encoding": "float64_little_endian_c_order",
            "array_count": len(result_arrays),
            "arrays": {
                key: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "dtype_str": value.dtype.str,
                    "c_contiguous": bool(value.flags.c_contiguous),
                    "sha256_float64_le_c_order": array_sha256(value),
                }
                for key, value in sorted(result_arrays.items())
            },
        },
    )
    _figure(rows, result, output / "n2_pvgr_n5_d4b_population_field_derivative.png")
    (output / "summary.md").write_text(
        "\n".join(
            [
                "# N2-PVGR N5-D4b closed 32-cell field derivative result",
                "",
                f"Machine decision: `{decision}`",
                "",
                f"- Cells / pairs / field units: {expected_cells} / {metadata['pair_count']} / {metadata['field_unit_count']}",
                f"- Passing map contexts: {sum(all_map_gates)}/{len(all_map_gates)}",
                f"- Passing structural controls: {sum(all_structure_gates)}/{len(all_structure_gates)}",
                f"- Stable ordered topology contexts: {sum(topology_gates)}/{len(topology_gates)}",
                f"- D4b logical field-point queries: {budget['d4b_total_logical_queries']:,}",
                f"- Wall time: {wall_seconds:.3f} s",
                f"- Peak RSS: {peak_after:,} bytes",
                "",
                "This is a complete closed synthetic development-census derivative gate.",
                "The 32 cells are clustered in 16 pairs and five field units; they are not 32 iid samples.",
                "It is not decoder-chain, reconstruction, neural-operator, real-data, generalization or paper evidence.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_manifest(
        output,
        config_path,
        _resolve(str(config["pre_registration_attestation"])),
        _resolve(str(config["frozen_input_archive"])),
        _resolve(str(config["pre_registration_ready_marker"])),
        published_output=formal_output,
    )
    if _path_occupied(formal_output):
        raise FileExistsError("N5-D4b formal output became occupied before publication")
    output.replace(formal_output)
    return result


def main() -> int:
    args = parse_args()
    config = _read_json(args.config)
    output = args.output or _resolve(str(config["formal_output"]))
    result = run(args.config.resolve(), output.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
