#!/usr/bin/env python3
"""Run the preregistered N2-PVGR N5-D4 tiny field-derivative gate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import time
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .automatic_discrete_multifidelity import SyntheticRayRig
    from .cancellation_aware_residual import evaluate_paired_residual
    from .d4_frozen_inputs import array_sha256, canonical_float64_array
    from .field_dependent_ray import (
        path_integrated_deflection,
        sample_pupil_sobol,
        straight_ray_deflection,
        trace_field_dependent_rays,
    )
    from .field_jvp_vjp_gate import ClosureDerivativeAudit, audit_tensor_closure
    from .field_program_signature import build_field_program_signature
except ImportError:
    from automatic_discrete_multifidelity import SyntheticRayRig
    from cancellation_aware_residual import evaluate_paired_residual
    from d4_frozen_inputs import array_sha256, canonical_float64_array
    from field_dependent_ray import (
        path_integrated_deflection,
        sample_pupil_sobol,
        straight_ray_deflection,
        trace_field_dependent_rays,
    )
    from field_jvp_vjp_gate import ClosureDerivativeAudit, audit_tensor_closure
    from field_program_signature import build_field_program_signature


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator/configs/n2_pvgr_n5_d4_tiny_field_derivative_preregistered_v1.json"
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


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=ROOT, check=check, capture_output=True)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _validate_contract(config: dict[str, Any]) -> None:
    if config.get("schema") != "n2-pvgr-n5-d4-tiny-field-derivative-preregistered-1.0":
        raise ValueError("N5-D4 schema drifted")
    if config.get("candidate_id") != "N2-PVGR-N5-D4-TINY-FIELD-DERIVATIVE":
        raise ValueError("N5-D4 candidate identifier drifted")
    if (config.get("device"), config.get("dtype")) != ("cpu", "float64"):
        raise ValueError("N5-D4 requires CPU float64")
    if (int(config.get("ray_count", 0)), int(config.get("step_count", 0))) != (4, 16):
        raise ValueError("N5-D4 ray/step contract drifted")
    if int(config.get("source_population_count", 0)) != 256:
        raise ValueError("N5-D4 source population must remain 256")
    if int(config.get("direction_count_per_cell", 0)) != 2:
        raise ValueError("N5-D4 direction count drifted")
    if config["direction_contract"].get("field_direction_modes_in_order") != [
        "smooth_avg_pool3d_kernel3_stride1_padding1",
        "raw_torch_randn",
    ]:
        raise ValueError("N5-D4 field-direction modes drifted")
    if len(config.get("selected_cells", [])) != 4:
        raise ValueError("N5-D4 must contain four frozen cells")
    expected_maps = [
        "curved_detector",
        "straight_detector",
        "raw_curved_minus_straight",
        "paired_neumaier_residual",
    ]
    if config.get("map_order") != expected_maps:
        raise ValueError("N5-D4 map order drifted")
    expected_h = [0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]
    if [
        float(value) for value in config["finite_difference"]["h_values"]
    ] != expected_h:
        raise ValueError("N5-D4 h-grid drifted")
    if config["finite_difference"].get("h_values_float64_hex") != [
        value.hex() for value in expected_h
    ]:
        raise ValueError("N5-D4 float64 h bytes drifted")
    if [
        float(value) for value in config["finite_difference"]["required_h_values"]
    ] != expected_h[:3]:
        raise ValueError("N5-D4 required h-grid drifted")
    budget = config["budget_contract"]
    expected_budget = {
        "expected_derivative_map_logical_queries": 1_032_192,
        "expected_derivative_map_interpolation_dispatches": 175_744,
        "expected_topology_logical_queries": 537_600,
        "expected_topology_interpolation_dispatches": 61_680,
        "expected_total_logical_queries": 1_569_792,
        "expected_total_interpolation_dispatches": 237_424,
        "expected_strong_toy_detach_control_logical_queries": 3_360,
        "expected_strong_toy_detach_control_interpolation_dispatches": 686,
        "expected_grand_total_logical_queries": 1_573_152,
        "expected_grand_total_interpolation_dispatches": 238_110,
    }
    if any(int(budget.get(key, -1)) != value for key, value in expected_budget.items()):
        raise ValueError("N5-D4 query budget drifted")
    if (
        not all(
            value is False
            for key, value in config["claim_authorizations"].items()
            if key != "tiny_selected_field_derivative_contract"
        )
        or config["claim_authorizations"]["tiny_selected_field_derivative_contract"]
        is not False
    ):
        raise ValueError("N5-D4 preregistered claims must all remain false")


def _validate_preregistration(
    config: dict[str, Any], config_path: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    attestation_path = _resolve(str(config["pre_registration_attestation"]))
    archive_path = _resolve(str(config["frozen_input_archive"]))
    if not attestation_path.is_file() or not archive_path.is_file():
        raise FileNotFoundError("committed N5-D4 attestation/frozen inputs are missing")
    attestation = _read_json(attestation_path)
    if (
        attestation.get("schema")
        != "n2-pvgr-n5-d4-tiny-field-derivative-attestation-1.0"
    ):
        raise ValueError("N5-D4 attestation schema drifted")
    if attestation.get("formal_results_absent_at_creation") is not True:
        raise ValueError("N5-D4 attestation does not prove output absence")
    if attestation.get("formal_work_output_absent_at_creation") is not True:
        raise ValueError("N5-D4 attestation does not prove work-output absence")
    if attestation.get("formal_output") != config["formal_output"]:
        raise ValueError("N5-D4 attestation output path drifted")
    if attestation.get("formal_work_output") != config["formal_work_output"]:
        raise ValueError("N5-D4 attestation work-output path drifted")
    if attestation.get("config_sha256") != _sha256(config_path):
        raise ValueError("N5-D4 config differs from attestation")
    protocol_commit = str(attestation["protocol_commit"])
    if attestation.get("repository_head_at_creation") != protocol_commit:
        raise ValueError("N5-D4 attestation was not created at protocol commit")
    if _git(
        "merge-base", "--is-ancestor", protocol_commit, "HEAD", check=False
    ).returncode:
        raise ValueError("N5-D4 protocol commit is not an ancestor of HEAD")
    if set(attestation.get("attested_files", {})) != set(config["attested_files"]):
        raise ValueError("N5-D4 attested file set drifted")
    for key, entry in attestation["attested_files"].items():
        expected = str(config["attested_files"][key])
        if entry["path"] != expected or _sha256(_resolve(expected)) != entry["sha256"]:
            raise ValueError(f"N5-D4 attested file drifted: {key}")
        frozen = _git("show", f"{protocol_commit}:{expected}").stdout
        if _sha256_bytes(frozen) != entry["sha256"]:
            raise ValueError(f"N5-D4 protocol file hash drifted: {key}")
    if attestation.get("frozen_input_archive") != config["frozen_input_archive"]:
        raise ValueError("N5-D4 frozen-input path drifted")
    if _sha256(archive_path) != attestation.get("frozen_input_archive_sha256"):
        raise ValueError("N5-D4 frozen-input archive hash drifted")
    watched = [
        _relative(attestation_path),
        _relative(archive_path),
        *(str(value) for value in config["attested_files"].values()),
    ]
    for path in watched[:2]:
        if _git("ls-files", "--error-unmatch", path, check=False).returncode:
            raise ValueError(f"N5-D4 prerequisite is not committed: {path}")
    if _git("status", "--porcelain", "--", *watched).stdout.strip():
        raise ValueError("N5-D4 preregistered files have uncommitted changes")

    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {key: canonical_float64_array(archive[key]) for key in archive.files}
    inventory = attestation["frozen_input_inventory"]
    if set(arrays) != set(inventory):
        raise ValueError("N5-D4 frozen-input inventory drifted")
    for key, array in arrays.items():
        entry = inventory[key]
        if list(array.shape) != entry["shape"] or str(array.dtype) != entry["dtype"]:
            raise ValueError(f"N5-D4 frozen-input shape/dtype drifted: {key}")
        if array_sha256(array) != entry["sha256_float64_le_c_order"]:
            raise ValueError(f"N5-D4 frozen-input array hash drifted: {key}")
    metadata = attestation["frozen_input_metadata"]
    if (
        _sha256_bytes(_canonical_json(metadata))
        != attestation["frozen_input_metadata_sha256"]
    ):
        raise ValueError("N5-D4 frozen-input metadata hash drifted")
    return attestation, arrays


def _rig(value: dict[str, Any]) -> SyntheticRayRig:
    return SyntheticRayRig(**value)


def _closures(
    rays: torch.Tensor,
    rig: SyntheticRayRig,
    *,
    difference_step: float,
    refractivity_scale: float,
    step_count: int,
) -> dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    def curved(field: torch.Tensor) -> torch.Tensor:
        trace = trace_field_dependent_rays(
            field,
            rays,
            rig,
            gradient_mode="central",
            difference_step=difference_step,
            refractivity_scale=refractivity_scale,
            step_count=step_count,
            create_graph=True,
        )
        return path_integrated_deflection(
            field,
            trace,
            gradient_mode="central",
            difference_step=difference_step,
            refractivity_scale=refractivity_scale,
            create_graph=True,
            detach_path=False,
        )

    def straight(field: torch.Tensor) -> torch.Tensor:
        return straight_ray_deflection(
            field,
            rays,
            rig,
            gradient_mode="central",
            difference_step=difference_step,
            refractivity_scale=refractivity_scale,
            step_count=step_count,
            create_graph=True,
        )

    def raw(field: torch.Tensor) -> torch.Tensor:
        return curved(field) - straight(field)

    def paired(field: torch.Tensor) -> torch.Tensor:
        return evaluate_paired_residual(
            field,
            rays,
            rig,
            difference_step=difference_step,
            refractivity_scale=refractivity_scale,
            step_count=step_count,
        ).paired_neumaier

    return {
        "curved_detector": curved,
        "straight_detector": straight,
        "raw_curved_minus_straight": raw,
        "paired_neumaier_residual": paired,
    }


def _tensor_metrics(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    scale_tensors: tuple[torch.Tensor, ...] = (),
) -> dict[str, float | bool]:
    finite = bool(torch.all(torch.isfinite(candidate))) and bool(
        torch.all(torch.isfinite(reference))
    )
    if not finite:
        return {
            "finite": False,
            "absolute_l2": math.inf,
            "relative_l2": math.inf,
            "maximum_absolute": math.inf,
            "scale": math.inf,
        }
    delta = candidate - reference
    absolute = float(torch.linalg.vector_norm(delta))
    scales = [
        float(torch.linalg.vector_norm(candidate)),
        float(torch.linalg.vector_norm(reference)),
    ]
    scales.extend(float(torch.linalg.vector_norm(value)) for value in scale_tensors)
    scale = max(*scales, 1e-30)
    return {
        "finite": True,
        "absolute_l2": absolute,
        "relative_l2": absolute / scale,
        "maximum_absolute": float(torch.max(torch.abs(delta))),
        "scale": scale,
    }


def _store_array(
    arrays: dict[str, np.ndarray], key: str, value: torch.Tensor
) -> dict[str, Any]:
    array = canonical_float64_array(value)
    arrays[key] = array
    return {
        "key": key,
        "shape": list(array.shape),
        "sha256_float64_le_c_order": array_sha256(array),
    }


def _serialize_audit(
    audit: ClosureDerivativeAudit,
    *,
    prefix: str,
    arrays: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    gates = config["gates"]
    required_h = {
        float(value) for value in config["finite_difference"]["required_h_values"]
    }
    eps = torch.finfo(torch.float64).eps
    autodiff = audit.autodiff
    levels: list[dict[str, Any]] = []
    best_gate = False
    required_gates: list[bool] = []
    all_signal = True
    for index, level in enumerate(audit.finite_difference.levels):
        derivative_scale = max(
            float(level.estimate_norm),
            float(audit.finite_difference.reference_jvp_norm),
            1e-30,
        )
        roundoff_atol = 128.0 * eps * float(level.output_scale) / float(level.h)
        minimum_signal = 1024.0 * eps * float(level.output_scale)
        informative = bool(level.symmetric_difference_norm > minimum_signal)
        all_signal = all_signal and informative
        best_level_gate = bool(
            level.finite
            and informative
            and level.relative_error
            <= float(gates["maximum_best_finite_difference_relative_l2"])
            and level.absolute_error
            <= roundoff_atol
            + float(gates["maximum_best_finite_difference_relative_l2"])
            * derivative_scale
        )
        best_gate = best_gate or best_level_gate
        required = float(level.h) in required_h
        required_gate = bool(
            level.finite
            and informative
            and level.relative_error
            <= float(gates["maximum_required_h_finite_difference_relative_l2"])
            and level.absolute_error
            <= roundoff_atol
            + float(gates["maximum_required_h_finite_difference_relative_l2"])
            * derivative_scale
        )
        if required:
            required_gates.append(required_gate)
        levels.append(
            {
                "index": index,
                "h": float(level.h),
                "required": required,
                "finite": bool(level.finite),
                "nondegenerate": bool(level.nondegenerate),
                "estimate_norm": float(level.estimate_norm),
                "symmetric_difference_norm": float(level.symmetric_difference_norm),
                "output_scale": float(level.output_scale),
                "minimum_signal": minimum_signal,
                "informative": informative,
                "absolute_error": float(level.absolute_error),
                "relative_error": float(level.relative_error),
                "roundoff_atol": roundoff_atol,
                "best_accuracy_gate": best_level_gate,
                "required_accuracy_gate": required_gate if required else None,
                "plus": _store_array(arrays, f"{prefix}_h{index:02d}_plus", level.plus),
                "minus": _store_array(
                    arrays, f"{prefix}_h{index:02d}_minus", level.minus
                ),
                "estimate": _store_array(
                    arrays, f"{prefix}_h{index:02d}_estimate", level.estimate
                ),
            }
        )
    if len(required_gates) != len(required_h):
        raise ValueError("N5-D4 required h values are absent from audit")
    dot_gate = bool(
        autodiff.finite
        and autodiff.nondegenerate
        and autodiff.dot_signal >= float(gates["minimum_dot_signal_absolute"])
        and autodiff.dot_relative_defect <= float(gates["maximum_dot_relative_defect"])
    )
    repeat_gate = bool(
        autodiff.repeated_primal_relative_defect
        <= float(gates["maximum_repeat_output_relative_l2"])
    )
    map_gate = bool(
        audit.finite
        and audit.nondegenerate
        and dot_gate
        and repeat_gate
        and best_gate
        and all(required_gates)
        and all_signal
    )
    return (
        {
            "finite": bool(audit.finite),
            "nondegenerate": bool(audit.nondegenerate),
            "primal": _store_array(arrays, f"{prefix}_primal", autodiff.primal),
            "jvp": _store_array(arrays, f"{prefix}_jvp", autodiff.jvp),
            "vjp": _store_array(arrays, f"{prefix}_vjp", autodiff.vjp),
            "dot_jvp_cotangent": float(autodiff.dot_jvp_cotangent),
            "dot_tangent_vjp": float(autodiff.dot_tangent_vjp),
            "dot_absolute_defect": float(autodiff.dot_absolute_defect),
            "dot_relative_defect": float(autodiff.dot_relative_defect),
            "dot_signal": float(autodiff.dot_signal),
            "jvp_norm": float(autodiff.jvp_norm),
            "vjp_norm": float(autodiff.vjp_norm),
            "repeated_primal_relative_defect": float(
                autodiff.repeated_primal_relative_defect
            ),
            "dot_gate": dot_gate,
            "repeat_output_gate": repeat_gate,
            "best_fd_gate": best_gate,
            "required_h_gates": required_gates,
            "all_h_informative": all_signal,
            "levels": levels,
            "map_gate": map_gate,
        },
        map_gate,
    )


def _topology_rows(
    field: torch.Tensor,
    direction: torch.Tensor,
    rays: torch.Tensor,
    rig: SyntheticRayRig,
    context: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    kwargs = {
        "difference_step": float(context["difference_step"]),
        "refractivity_scale": float(context["refractivity_scale"]),
        "step_count": int(config["step_count"]),
        "support_threshold": float(context["support_threshold"]),
        "frustum_half_width_u": float(context["frustum_half_width_u"]),
        "frustum_half_width_v": float(context["frustum_half_width_v"]),
    }
    base = build_field_program_signature(field, rays, rig, **kwargs)
    rows: list[dict[str, Any]] = []
    unchanged = True
    margins_ok = bool(
        base.minimum_domain_margin >= float(config["gates"]["minimum_domain_margin"])
        and base.minimum_stencil_margin
        >= float(config["gates"]["minimum_stencil_margin"])
        and base.maximum_direction_norm_error
        <= float(config["gates"]["maximum_direction_norm_error"])
    )
    for h in (float(value) for value in config["finite_difference"]["h_values"]):
        for sign, label in ((1.0, "plus"), (-1.0, "minus")):
            candidate = build_field_program_signature(
                field + sign * h * direction, rays, rig, **kwargs
            )
            same = candidate.digest_sha256 == base.digest_sha256
            candidate_margins_ok = bool(
                candidate.minimum_domain_margin
                >= float(config["gates"]["minimum_domain_margin"])
                and candidate.minimum_stencil_margin
                >= float(config["gates"]["minimum_stencil_margin"])
                and candidate.maximum_direction_norm_error
                <= float(config["gates"]["maximum_direction_norm_error"])
            )
            unchanged = unchanged and same
            margins_ok = margins_ok and candidate_margins_ok
            rows.append(
                {
                    "h": h,
                    "side": label,
                    "digest_sha256": candidate.digest_sha256,
                    "interpolation_cells_sha256": candidate.interpolation_cells_sha256,
                    "support_bits_sha256": candidate.support_bits_sha256,
                    "frustum_margin_signs_sha256": candidate.frustum_margin_signs_sha256,
                    "matches_base": same,
                    "minimum_domain_margin": candidate.minimum_domain_margin,
                    "minimum_stencil_margin": candidate.minimum_stencil_margin,
                    "minimum_frustum_margin": candidate.minimum_frustum_margin,
                    "maximum_direction_norm_error": candidate.maximum_direction_norm_error,
                    "margins_gate": candidate_margins_ok,
                }
            )
    return (
        {
            "base": asdict(base),
            "perturbations": rows,
            "ordered_signature_unchanged": unchanged,
            "margins_gate": margins_ok,
        },
        bool(unchanged and margins_ok),
    )


def _strong_toy_detach_control(config: dict[str, Any]) -> dict[str, Any]:
    axis = torch.linspace(-1.0, 1.0, 9, dtype=torch.float64)
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    field = -torch.exp(-((xx / 0.42) ** 2 + (yy / 0.34) ** 2 + (zz / 0.38) ** 2))
    rays = sample_pupil_sobol(4, seed=9107)
    rig = SyntheticRayRig(
        rig_id="n5-d4-strong-detach-control",
        view_angle_degrees=27.0,
        detector_u=0.04,
        detector_z=-0.03,
        aperture_radius=0.025,
        path_half_length=0.62,
        cone_u=0.02,
        cone_z=0.015,
        bend=0.0,
    )
    generator = torch.Generator(device="cpu").manual_seed(9109)
    direction = torch.randn(field.shape, generator=generator, dtype=torch.float64)
    direction = direction / torch.linalg.vector_norm(direction)

    def closure(value: torch.Tensor, *, detach_path: bool) -> torch.Tensor:
        trace = trace_field_dependent_rays(
            value,
            rays,
            rig,
            gradient_mode="central",
            difference_step=0.002,
            refractivity_scale=0.03,
            step_count=12,
            create_graph=True,
        )
        return path_integrated_deflection(
            value,
            trace,
            gradient_mode="central",
            difference_step=0.002,
            refractivity_scale=0.03,
            create_graph=True,
            detach_path=detach_path,
        )

    _, full = torch.func.jvp(
        lambda value: closure(value, detach_path=False), (field,), (direction,)
    )
    _, frozen = torch.func.jvp(
        lambda value: closure(value, detach_path=True), (field,), (direction,)
    )
    metrics = _tensor_metrics(full, frozen)
    threshold = float(
        config["gates"][
            "minimum_full_vs_frozen_path_jvp_relative_difference_in_strong_toy_control"
        ]
    )
    passed = bool(metrics["finite"] and metrics["relative_l2"] >= threshold)
    return {
        "grid_shape": list(field.shape),
        "ray_count": 4,
        "step_count": 12,
        "refractivity_scale": 0.03,
        "field_sha256": array_sha256(field),
        "direction_sha256": array_sha256(direction),
        "full_jvp_sha256": array_sha256(full),
        "frozen_path_jvp_sha256": array_sha256(frozen),
        "comparison": metrics,
        "required_minimum_relative_difference": threshold,
        "gate": passed,
        "logical_field_point_queries": 3360,
    }


def _figure(rows: list[dict[str, Any]], result: dict[str, Any], path: Path) -> None:
    colors = {
        "curved_detector": "#006d77",
        "straight_detector": "#e29578",
        "raw_curved_minus_straight": "#355070",
        "paired_neumaier_residual": "#9b5de5",
    }
    fig, axes = plt.subplots(2, 2, figsize=(14.3, 8.7), constrained_layout=True)
    ax = axes[0, 0]
    for row in rows:
        for map_id, audit in row["maps"].items():
            ax.loglog(
                [level["h"] for level in audit["levels"]],
                np.maximum(
                    [level["relative_error"] for level in audit["levels"]],
                    1e-30,
                ),
                color=colors[map_id],
                alpha=0.22,
                linewidth=1.0,
            )
    for map_id, color in colors.items():
        ax.plot([], [], color=color, label=map_id.replace("_", " "))
    ax.axhline(1e-6, color="#111111", linestyle="--", linewidth=1, label="best gate")
    ax.axhline(
        1e-5, color="#666666", linestyle=":", linewidth=1, label="required-h gate"
    )
    ax.set_title("Fixed multi-h central finite difference")
    ax.set_xlabel("h")
    ax.set_ylabel("relative L2(JVP, FD)")
    ax.grid(True, which="both", alpha=0.18)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[0, 1]
    for map_index, map_id in enumerate(colors):
        values = np.maximum(
            [row["maps"][map_id]["dot_relative_defect"] for row in rows],
            1e-30,
        )
        ax.scatter(
            np.full(len(values), map_index) + np.linspace(-0.1, 0.1, len(values)),
            values,
            color=colors[map_id],
            s=22,
            alpha=0.8,
        )
    ax.axhline(1e-10, color="#111111", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_xticks(
        range(len(colors)), [key.replace("_", "\n") for key in colors], fontsize=7
    )
    ax.set_ylabel("relative dot defect")
    ax.set_title("JVP/VJP same-cotangent closure")
    ax.grid(True, axis="y", which="both", alpha=0.18)

    ax = axes[1, 0]
    x = np.arange(len(rows))
    raw_jvp = [
        row["structure"]["raw_equals_curved_minus_straight"]["jvp"]["relative_l2"]
        for row in rows
    ]
    paired_jvp = [
        row["structure"]["paired_equals_raw_control"]["jvp"]["relative_l2"]
        for row in rows
    ]
    ax.semilogy(x, np.maximum(raw_jvp, 1e-30), "o-", label="raw = curved - straight")
    ax.semilogy(x, np.maximum(paired_jvp, 1e-30), "s-", label="paired vs raw")
    ax.set_xticks(
        x,
        [f"c{row['cell_index']}/d{row['direction_index']}" for row in rows],
        rotation=35,
        ha="right",
    )
    ax.set_ylabel("component-scaled relative JVP error")
    ax.set_title("Residual wiring and accumulation controls")
    ax.grid(True, which="both", alpha=0.18)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.axis("off")
    lines = [
        "N5-D4 TINY FIELD DERIVATIVE GATE",
        "",
        f"Decision: {result['machine_decision']}",
        f"Passing map contexts: {result['counts']['passing_map_context_count']} / {result['counts']['map_context_count']}",
        f"Stable topology contexts: {result['counts']['stable_topology_context_count']} / {result['counts']['direction_context_count']}",
        f"Strong detach control: {result['strong_toy_detach_control']['gate']}",
        f"Logical queries: {result['budget']['grand_total_logical_queries']:,}",
        f"Wall time: {result['timing']['wall_seconds']:.2f} s",
        f"Peak RSS: {result['timing']['peak_rss_bytes'] / (1024**2):.1f} MiB",
        "",
        "A pass opens only a preregistered 32-cell derivative expansion.",
        "No reconstruction, neural-operator, real-data, or paper claim.",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )
    fig.suptitle(
        "N2-PVGR N5-D4: result-before-preregistered field derivative evidence",
        fontsize=14,
    )
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _write_manifest(
    output: Path,
    config_path: Path,
    attestation_path: Path,
    archive_path: Path,
    *,
    published_output: Path | None = None,
) -> None:
    published = published_output or output
    files: dict[str, dict[str, str]] = {}
    for label, path in (
        ("config", config_path),
        ("attestation", attestation_path),
        ("frozen_input_archive", archive_path),
    ):
        files[label] = {"path": _relative(path), "sha256": _sha256(path)}
    for path in sorted(output.iterdir()):
        if (
            path.name in {"manifest.json", "validation_report.json"}
            or not path.is_file()
        ):
            continue
        files[path.name] = {
            "path": _relative(published / path.name),
            "sha256": _sha256(path),
        }
    _write_json(
        output / "manifest.json",
        {
            "schema": "n2-pvgr-n5-d4-tiny-field-derivative-manifest-1.0",
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
        raise ValueError("N5-D4 output path must match the preregistered formal output")
    if _path_occupied(formal_output) or _path_occupied(work_output):
        raise FileExistsError(
            "N5-D4 formal/work output already exists; formal reruns are forbidden"
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
            raise ValueError("N5-D4 frozen field metadata hash drifted")
        if array_sha256(rays_array) != context["rays_sha256"]:
            raise ValueError("N5-D4 frozen ray metadata hash drifted")
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
                raise ValueError("N5-D4 frozen direction hash drifted")
            if array_sha256(cotangent) != direction_entry["cotangent_sha256"]:
                raise ValueError("N5-D4 frozen cotangent hash drifted")
            topology, topology_gate = _topology_rows(
                field, direction, rays, rig, context, config
            )
            topology_gates.append(topology_gate)
            map_rows: dict[str, Any] = {}
            map_audits: dict[str, ClosureDerivativeAudit] = {}
            map_start = time.perf_counter()
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
                map_audits[map_id] = audit
                all_map_gates.append(map_gate)

            curved = map_audits["curved_detector"].autodiff
            straight = map_audits["straight_detector"].autodiff
            raw = map_audits["raw_curved_minus_straight"].autodiff
            paired = map_audits["paired_neumaier_residual"].autodiff
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
            absolute_floor = float(gates["structural_absolute_tolerance_floor"])

            def structural_gate(metric: dict[str, float | bool], rtol: float) -> bool:
                return bool(
                    metric["finite"]
                    and metric["absolute_l2"]
                    <= absolute_floor + float(rtol) * float(metric["scale"])
                )

            raw_gate = bool(
                structural_gate(
                    raw_structure["primal"],
                    float(gates["maximum_raw_structural_output_relative_l2"]),
                )
                and structural_gate(
                    raw_structure["jvp"],
                    float(gates["maximum_raw_structural_jvp_relative_l2"]),
                )
                and structural_gate(
                    raw_structure["vjp"],
                    float(gates["maximum_raw_structural_vjp_relative_l2"]),
                )
            )
            paired_gate = bool(
                structural_gate(
                    paired_structure["primal"],
                    float(gates["maximum_raw_paired_output_relative_l2"]),
                )
                and structural_gate(
                    paired_structure["jvp"],
                    float(gates["maximum_raw_paired_jvp_relative_l2"]),
                )
                and structural_gate(
                    paired_structure["vjp"],
                    float(gates["maximum_raw_paired_vjp_relative_l2"]),
                )
            )
            all_structure_gates.extend((raw_gate, paired_gate))
            rows.append(
                {
                    "cell_index": cell_index,
                    "cell_id": context["cell_id"],
                    "pair_id": context["pair_id"],
                    "role": context["role"],
                    "direction_index": direction_index,
                    "field_sha256": context["field_sha256"],
                    "rays_sha256": context["rays_sha256"],
                    "direction_sha256": direction_entry["direction_sha256"],
                    "cotangent_sha256": direction_entry["cotangent_sha256"],
                    "topology": topology,
                    "topology_gate": topology_gate,
                    "maps": map_rows,
                    "structure": {
                        "raw_equals_curved_minus_straight": raw_structure
                        | {"gate": raw_gate},
                        "paired_equals_raw_control": paired_structure
                        | {"gate": paired_gate},
                    },
                    "wall_seconds": time.perf_counter() - map_start,
                }
            )

    strong_control = _strong_toy_detach_control(config)
    context_changed = not all(topology_gates)
    ordinary_pass = bool(
        all(all_map_gates)
        and all(all_structure_gates)
        and strong_control["gate"]
        and len(all_map_gates) == 32
        and len(all_structure_gates) == 16
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
    authorizations["tiny_selected_field_derivative_contract"] = bool(
        decision == config["decision_contract"]["pass_decision"]
    )
    result = {
        "schema": "n2-pvgr-n5-d4-tiny-field-derivative-result-1.0",
        "candidate_id": config["candidate_id"],
        "protocol_commit": attestation["protocol_commit"],
        "machine_decision": decision,
        "selected_context_only": True,
        "grid_field_derivative_core_only": True,
        "decoder_parameterization_tested": False,
        "counts": {
            "cell_count": 4,
            "direction_context_count": 8,
            "map_context_count": 32,
            "passing_map_context_count": sum(all_map_gates),
            "structure_gate_count": 16,
            "passing_structure_gate_count": sum(all_structure_gates),
            "stable_topology_context_count": sum(topology_gates),
        },
        "all_map_gates_pass": all(all_map_gates),
        "all_structure_gates_pass": all(all_structure_gates),
        "all_topology_gates_pass": all(topology_gates),
        "strong_toy_detach_control": strong_control,
        "budget": {
            "derivative_map_logical_queries": 1_032_192,
            "derivative_map_interpolation_dispatches": 175_744,
            "topology_logical_queries": 537_600,
            "topology_interpolation_dispatches": 61_680,
            "selected_context_logical_queries": 1_569_792,
            "selected_context_interpolation_dispatches": 237_424,
            "strong_toy_control_logical_queries": 3_360,
            "strong_toy_control_interpolation_dispatches": 686,
            "grand_total_logical_queries": 1_573_152,
            "grand_total_interpolation_dispatches": 238_110,
            "map_closure_invocations": 512,
            "jvp_sweeps": 32,
            "vjp_sweeps": 32,
            "finite_difference_forward_calls": 448,
            "topology_signature_invocations": 120,
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
            "preregister_32_cell_derivative_expansion"
            if authorizations["tiny_selected_field_derivative_contract"]
            else "do_not_enter_32_cell_derivative_expansion_or_reconstruction"
        ),
    }
    _write_json(output / "config_snapshot.json", config)
    _write_json(output / "rows.json", rows)
    _write_json(output / "result.json", result)
    np.savez(output / "derivative_arrays.npz", **result_arrays)
    _write_json(
        output / "array_inventory.json",
        {
            "schema": "n2-pvgr-n5-d4-derivative-array-inventory-1.0",
            "encoding": "float64_little_endian_c_order",
            "array_count": len(result_arrays),
            "arrays": {
                key: {
                    "shape": list(value.shape),
                    "sha256_float64_le_c_order": array_sha256(value),
                }
                for key, value in sorted(result_arrays.items())
            },
        },
    )
    _figure(rows, result, output / "n2_pvgr_n5_d4_tiny_field_derivative.png")
    (output / "summary.md").write_text(
        "\n".join(
            [
                "# N2-PVGR N5-D4 tiny field derivative result",
                "",
                f"Machine decision: `{decision}`",
                "",
                f"- Passing map contexts: {sum(all_map_gates)}/32",
                f"- Passing structural controls: {sum(all_structure_gates)}/16",
                f"- Stable ordered topology contexts: {sum(topology_gates)}/8",
                f"- Strong full-vs-frozen-path control: {strong_control['gate']}",
                f"- Logical field-point queries: {1_573_152:,}",
                f"- Wall time: {wall_seconds:.3f} s",
                f"- Peak RSS: {peak_after:,} bytes",
                "",
                "This is a selected synthetic grid-field derivative implementation gate only.",
                "It is not reconstruction, neural-operator, real-data, generalization or paper evidence.",
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
        published_output=formal_output,
    )
    if _path_occupied(formal_output):
        raise FileExistsError("N5-D4 formal output became occupied before publication")
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
