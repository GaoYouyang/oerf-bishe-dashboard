#!/usr/bin/env python3
"""Audit convergence of the small curved-minus-straight BOST residual.

The full high-fidelity output can look converged while the much smaller
``H-M`` routing target is still contaminated by quadrature error.  This
development audit therefore checks the residual itself across matched medium
and high step counts.  It never opens reserved families and does not authorize
real BOST, reconstruction, generalization, or paper claims.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .analytic_bost_phantoms import analytic_phantom_grid, make_analytic_phantom
    from .automatic_discrete_multifidelity import trace_sample_variance
    from .field_dependent_ray import relative_l2, sample_pupil_sobol
    from .run_n2_pvgr_n0_trifidelity_development import (
        _high_route,
        _rig_from_case,
        _straight_route,
        stable_seed,
    )
except ImportError:
    from analytic_bost_phantoms import analytic_phantom_grid, make_analytic_phantom
    from automatic_discrete_multifidelity import trace_sample_variance
    from field_dependent_ray import relative_l2, sample_pupil_sobol
    from run_n2_pvgr_n0_trifidelity_development import (
        _high_route,
        _rig_from_case,
        _straight_route,
        stable_seed,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator"
    / "configs"
    / "n2_pvgr_residual_convergence_audit_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "demo_t16_operator"
    / "results"
    / "n2_pvgr_residual_convergence_audit_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_variance_ratio(
    candidate: torch.Tensor,
    reference: torch.Tensor,
) -> float:
    reference_variance = trace_sample_variance(reference)
    if reference_variance <= 1e-30:
        return 1.0 if trace_sample_variance(candidate) <= 1e-30 else float("inf")
    return trace_sample_variance(candidate) / reference_variance


def _validate_contracts(
    config: dict[str, Any],
    source: dict[str, Any],
) -> tuple[list[int], int, int, tuple[str, ...]]:
    steps = sorted({int(value) for value in config["step_counts"]})
    if len(steps) < 3 or steps[0] < 2:
        raise ValueError("step_counts must contain at least three values >=2")
    reference = int(config["reference_step_count"])
    accepted = int(config["accepted_execution_step_count"])
    if reference not in steps or accepted not in steps or accepted >= reference:
        raise ValueError("accepted/reference step counts are inconsistent")
    reserved = tuple(str(value) for value in config["reserved_audit_families_not_opened"])
    if set(reserved) != set(source["reserved_audit_families_not_opened"]):
        raise ValueError("reserved-family contract differs from the source experiment")
    development = {str(case["phantom_family"]) for case in source["development_cases"]}
    if development & set(reserved):
        raise RuntimeError("a reserved audit family appears in development cases")
    return steps, reference, accepted, reserved


def _case_scale_rows(
    case: dict[str, Any],
    source: dict[str, Any],
    *,
    steps: list[int],
    reference_step: int,
    accepted_step: int,
    maximum_residual_relative_l2: float,
    maximum_variance_ratio_deviation: float,
) -> list[dict[str, Any]]:
    spec = make_analytic_phantom(
        family=str(case["phantom_family"]),
        seed=int(case["phantom_seed"]),
    )
    values = analytic_phantom_grid(
        spec,
        grid_shape=tuple(int(value) for value in source["grid_shape_zyx"]),
        dtype=torch.float64,
        device="cpu",
    ).field
    states = sample_pupil_sobol(
        int(source["population_count"]),
        seed=stable_seed(
            int(source["seed_roles"]["population_state_base"]),
            case["id"],
        ),
    )
    rig = _rig_from_case(case)
    difference_step = float(source["difference_step"])
    rows: list[dict[str, Any]] = []
    for stress in source["dimensionless_stress_scale_multipliers"]:
        scale = float(source["base_refractivity_scale"]) * float(stress)
        outputs: dict[int, dict[str, torch.Tensor]] = {}
        for step_count in steps:
            medium = _straight_route(
                values,
                states,
                rig,
                gradient_mode="central",
                difference_step=difference_step,
                refractivity_scale=scale,
                step_count=step_count,
                create_graph=False,
            )
            high, _ = _high_route(
                values,
                states,
                rig,
                difference_step=difference_step,
                refractivity_scale=scale,
                step_count=step_count,
                create_graph=False,
            )
            outputs[step_count] = {
                "medium": medium,
                "high": high,
                "residual": high - medium,
            }
        reference = outputs[reference_step]
        convergence: list[dict[str, Any]] = []
        for step_count in steps:
            current = outputs[step_count]
            convergence.append(
                {
                    "step_count": step_count,
                    "high_relative_l2_to_reference": relative_l2(
                        current["high"],
                        reference["high"],
                    ),
                    "medium_relative_l2_to_reference": relative_l2(
                        current["medium"],
                        reference["medium"],
                    ),
                    "residual_relative_l2_to_reference": relative_l2(
                        current["residual"],
                        reference["residual"],
                    ),
                    "residual_variance_ratio_to_reference": _relative_variance_ratio(
                        current["residual"],
                        reference["residual"],
                    ),
                }
            )
        accepted_metrics = next(
            item for item in convergence if item["step_count"] == accepted_step
        )
        residual_norm_fraction = float(
            torch.linalg.vector_norm(reference["residual"])
            / torch.linalg.vector_norm(reference["high"]).clamp_min(1e-30)
        )
        residual_gate = (
            accepted_metrics["residual_relative_l2_to_reference"]
            <= maximum_residual_relative_l2
        )
        variance_gate = (
            abs(accepted_metrics["residual_variance_ratio_to_reference"] - 1.0)
            <= maximum_variance_ratio_deviation
        )
        rows.append(
            {
                "case_id": str(case["id"]),
                "phantom_family": str(case["phantom_family"]),
                "phantom_seed": int(case["phantom_seed"]),
                "dimensionless_stress_multiplier": float(stress),
                "refractivity_scale": scale,
                "reference_step_count": reference_step,
                "accepted_execution_step_count": accepted_step,
                "convergence": convergence,
                "reference_residual_norm_to_high_output_norm": residual_norm_fraction,
                "accepted_residual_relative_l2_gate_met": residual_gate,
                "accepted_residual_variance_gate_met": variance_gate,
                "accepted_execution_step_gate_met": residual_gate and variance_gate,
            }
        )
    return rows


def _write_figure(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    accepted_step: int,
    residual_threshold: float,
    variance_threshold: float,
) -> None:
    colors = {
        "smooth_narrow_aperture": "#176b67",
        "wrinkled_wide_aperture": "#a34e3f",
        "smooth_wide_aperture": "#405a8a",
    }
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    base_rows = [row for row in rows if row["dimensionless_stress_multiplier"] == 1.0]
    for row in base_rows:
        steps = [item["step_count"] for item in row["convergence"]]
        errors = [
            max(item["residual_relative_l2_to_reference"], 1e-8)
            for item in row["convergence"]
        ]
        axes[0, 0].plot(
            steps,
            errors,
            marker="o",
            label=row["case_id"].replace("_", " "),
            color=colors.get(row["case_id"], "#4c5e65"),
        )
    axes[0, 0].axhline(residual_threshold, color="#20282c", linestyle="--")
    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("matched medium/high step count")
    axes[0, 0].set_ylabel("residual relative-L2 to reference")
    axes[0, 0].set_title("small residual needs its own convergence gate")
    axes[0, 0].legend(fontsize=8)

    case_ids = sorted({row["case_id"] for row in rows})
    markers = {1.0: "o", 3.0: "s", 10.0: "^"}
    for row in rows:
        accepted = next(
            item for item in row["convergence"] if item["step_count"] == accepted_step
        )
        axes[0, 1].scatter(
            case_ids.index(row["case_id"]),
            accepted["residual_relative_l2_to_reference"],
            marker=markers[row["dimensionless_stress_multiplier"]],
            s=60,
            color=colors.get(row["case_id"], "#4c5e65"),
            label=f"{row['dimensionless_stress_multiplier']:g}x",
        )
        axes[1, 0].scatter(
            case_ids.index(row["case_id"]),
            abs(accepted["residual_variance_ratio_to_reference"] - 1.0),
            marker=markers[row["dimensionless_stress_multiplier"]],
            s=60,
            color=colors.get(row["case_id"], "#4c5e65"),
        )
        axes[1, 1].scatter(
            row["dimensionless_stress_multiplier"],
            row["reference_residual_norm_to_high_output_norm"],
            marker="o",
            s=55,
            color=colors.get(row["case_id"], "#4c5e65"),
        )
    axes[0, 1].axhline(residual_threshold, color="#20282c", linestyle="--")
    axes[0, 1].set_xticks(range(len(case_ids)), [value.replace("_", "\n") for value in case_ids])
    axes[0, 1].set_ylabel(f"residual relative-L2 at {accepted_step} steps")
    axes[0, 1].set_title("accepted execution target across stress")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    axes[0, 1].legend(unique.values(), unique.keys(), title="stress", fontsize=8)

    axes[1, 0].axhline(variance_threshold, color="#20282c", linestyle="--")
    axes[1, 0].set_xticks(range(len(case_ids)), [value.replace("_", "\n") for value in case_ids])
    axes[1, 0].set_ylabel("abs(residual variance ratio - 1)")
    axes[1, 0].set_title("variance can look stable before the vector residual")

    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("dimensionless stress multiplier")
    axes[1, 1].set_ylabel("reference residual norm / high-output norm")
    axes[1, 1].set_title("the learned target is orders smaller than full output")
    figure.suptitle(
        "N2-PVGR residual convergence audit: full-output convergence is not enough",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    source_path = ROOT / str(config["source_config"])
    source = _read_json(source_path)
    steps, reference_step, accepted_step, reserved = _validate_contracts(config, source)
    maximum_residual = float(config["maximum_residual_relative_l2"])
    maximum_variance_deviation = float(
        config["maximum_residual_variance_ratio_deviation"]
    )
    if not 0.0 < maximum_residual < 1.0:
        raise ValueError("maximum_residual_relative_l2 must lie in (0,1)")
    if not 0.0 < maximum_variance_deviation < 1.0:
        raise ValueError("maximum variance-ratio deviation must lie in (0,1)")

    rows = [
        row
        for case in source["development_cases"]
        for row in _case_scale_rows(
            case,
            source,
            steps=steps,
            reference_step=reference_step,
            accepted_step=accepted_step,
            maximum_residual_relative_l2=maximum_residual,
            maximum_variance_ratio_deviation=maximum_variance_deviation,
        )
    ]
    pass_count = sum(row["accepted_execution_step_gate_met"] for row in rows)
    required_fraction = float(config["required_case_scale_pass_fraction"])
    required_count = int(np.ceil(required_fraction * len(rows) - 1e-12))
    accepted = pass_count >= required_count
    machine_decision = (
        str(config["hard_conclusion"])
        if accepted
        else "RESIDUAL_TARGET_128_REJECTED_DEVELOPMENT_ONLY"
    )
    result = {
        "schema": str(config["schema"]),
        "audit_id": str(config["audit_id"]),
        "machine_decision": machine_decision,
        "claim_boundary": (
            "synthetic matched-discretization convergence only; no real BOST, "
            "reconstruction, generalization, or paper authorization"
        ),
        "step_counts": steps,
        "reference_step_count": reference_step,
        "accepted_execution_step_count": accepted_step,
        "case_scale_count": len(rows),
        "accepted_execution_step_screen_count": pass_count,
        "accepted_execution_step_screen_required_count": required_count,
        "reserved_audit_families_not_opened": list(reserved),
        "rows": rows,
        "figures": ["n2_pvgr_residual_convergence_audit.png"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    snapshot_path = output_dir / "config_snapshot.json"
    csv_path = output_dir / "metrics.csv"
    summary_path = output_dir / "summary.md"
    figure_path = output_dir / result["figures"][0]
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    snapshot_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "case_id",
            "stress_multiplier",
            "step_count",
            "high_relative_l2_to_reference",
            "medium_relative_l2_to_reference",
            "residual_relative_l2_to_reference",
            "residual_variance_ratio_to_reference",
            "reference_residual_norm_to_high_output_norm",
            "accepted_execution_step_gate_met",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            for item in row["convergence"]:
                writer.writerow(
                    {
                        "case_id": row["case_id"],
                        "stress_multiplier": row["dimensionless_stress_multiplier"],
                        "step_count": item["step_count"],
                        "high_relative_l2_to_reference": item[
                            "high_relative_l2_to_reference"
                        ],
                        "medium_relative_l2_to_reference": item[
                            "medium_relative_l2_to_reference"
                        ],
                        "residual_relative_l2_to_reference": item[
                            "residual_relative_l2_to_reference"
                        ],
                        "residual_variance_ratio_to_reference": item[
                            "residual_variance_ratio_to_reference"
                        ],
                        "reference_residual_norm_to_high_output_norm": row[
                            "reference_residual_norm_to_high_output_norm"
                        ],
                        "accepted_execution_step_gate_met": row[
                            "accepted_execution_step_gate_met"
                        ],
                    }
                )
    _write_figure(
        rows,
        figure_path,
        accepted_step=accepted_step,
        residual_threshold=maximum_residual,
        variance_threshold=maximum_variance_deviation,
    )
    summary_path.write_text(
        "\n".join(
            (
                "# N2-PVGR residual convergence audit",
                "",
                f"- Machine decision: `{machine_decision}`.",
                f"- Accepted step: {accepted_step}; reference step: {reference_step}.",
                f"- Screen: {pass_count}/{len(rows)} case x stress rows.",
                "- The small H-M residual, not only the full high output, is checked.",
                "- This is synthetic development evidence and no algorithm-success claim.",
                "",
            )
        ),
        encoding="utf-8",
    )
    generated = (result_path, snapshot_path, csv_path, summary_path, figure_path)
    manifest = {
        "schema": str(config["schema"]),
        "source_sha256": {
            "runner": _sha256(Path(__file__)),
            "config": _sha256(config_path),
            "source_config": _sha256(source_path),
            "field_dependent_ray": _sha256(
                ROOT / "demo_t16_operator/field_dependent_ray.py"
            ),
            "trifidelity_runner": _sha256(
                ROOT
                / "demo_t16_operator/run_n2_pvgr_n0_trifidelity_development.py"
            ),
            "analytic_phantoms": _sha256(
                ROOT / "demo_t16_operator/analytic_bost_phantoms.py"
            ),
        },
        "files": {path.name: _sha256(path) for path in generated},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    args = parse_args()
    result = run(args.config.resolve(), args.output.resolve())
    print(
        json.dumps(
            {
                "machine_decision": result["machine_decision"],
                "screen": (
                    f"{result['accepted_execution_step_screen_count']}/"
                    f"{result['case_scale_count']}"
                ),
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
