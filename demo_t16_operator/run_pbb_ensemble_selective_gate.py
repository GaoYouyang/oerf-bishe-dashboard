#!/usr/bin/env python3
"""Fresh-lock gate for a shared-physics PBB correction ensemble.

The ensemble pays for one projected-BB physical trajectory.  Multiple small
correction heads reuse that trajectory, and their disagreement is calibrated
on an independent selection generator.  The lock generator is constructed
only after the uncertainty threshold has been written to disk.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .bost_physics import build_forward_matrix
    from .cg_pdno import PBBBaseCorrectionCGPDNO, descent_certificate
    from .independent_reaction_bost import array_sha256, build_curved_cone_operator
    from .measurement_contract import DenseVolumeLinearBOST, DepthSeparableLinearBOST
    from .run_base_correction_independent_gate import (
        analytic_batch,
        fixed_projected_gradient,
        front_f1,
        gradient_relative_l2,
        independent_batch,
        projected_bb,
        projected_fista,
        relative_l2,
        train_seed,
    )
except ImportError:
    from bost_physics import build_forward_matrix
    from cg_pdno import PBBBaseCorrectionCGPDNO, descent_certificate
    from independent_reaction_bost import array_sha256, build_curved_cone_operator
    from measurement_contract import DenseVolumeLinearBOST, DepthSeparableLinearBOST
    from run_base_correction_independent_gate import (
        analytic_batch,
        fixed_projected_gradient,
        front_f1,
        gradient_relative_l2,
        independent_batch,
        projected_bb,
        projected_fista,
        relative_l2,
        train_seed,
    )


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "pbb_ensemble_selective_gate.json"
DEFAULT_OUTPUT = ROOT / "results" / "pbb_ensemble_selective_gate"
METHODS = ("candidate", "raw_ensemble", "pbb_fallback", "fixed_pg", "projected_bb", "fista")


class WeightedGradientCounter:
    """Count physical gradient calls while delegating to one operator."""

    def __init__(self, operator):
        self.operator = operator
        self.calls = 0

    def weighted_gradient(self, values, batch):
        self.calls += 1
        return self.operator.weighted_gradient(values, batch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def shared_ensemble_state(
    models: list[PBBBaseCorrectionCGPDNO],
    batch,
    operator,
    lipschitz: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if len(models) < 2:
        raise ValueError("uncertainty ensemble requires at least two members")
    reference = models[0]
    for model in models[1:]:
        if (
            model.stages != reference.stages
            or model.bb_normalized_step_min != reference.bb_normalized_step_min
            or model.bb_normalized_step_max != reference.bb_normalized_step_max
        ):
            raise ValueError("ensemble members disagree on the physical PBB path")
    warm = torch.zeros_like(batch.truth)
    state = reference.prepare_shared_state(batch, operator, warm, lipschitz)
    member_predictions = []
    member_acceptance = []
    member_ratios = []
    for model in models:
        model.eval()
        output = model.correct_from_shared_state(state, lipschitz)
        member_predictions.append(output["prediction"])
        member_acceptance.append(output["acceptance_gate"])
        member_ratios.append(output["raw_correction_ratio"])
    members = torch.stack(member_predictions, dim=0)
    fallback = state["fallback"]
    corrections = members - fallback.unsqueeze(0)
    mean_prediction = torch.mean(members, dim=0)

    # The mean of safe member fields is convex, but this second certificate
    # also guards floating-point and member-gate edge cases without a new A/A^T call.
    mean_delta = mean_prediction - state["base"]
    alpha, upper_bound, descent = descent_certificate(
        state["gradient"],
        mean_delta,
        lipschitz,
        safety=reference.certificate_safety,
    )
    certified_mean = state["base"] + alpha[:, None, None, None, None] * mean_delta
    certified_mean = torch.clamp(certified_mean, min=0.0) * state["support"]

    disagreement_rms = torch.sqrt(torch.mean(torch.var(corrections, dim=0, unbiased=False), dim=(1, 2, 3, 4)))
    physical_scale = torch.sqrt(torch.mean(state["pbb_direction"].square(), dim=(1, 2, 3, 4))).clamp_min(1e-8)
    uncertainty = disagreement_rms / physical_scale
    correction_rms = torch.sqrt(torch.mean((certified_mean - fallback).square(), dim=(1, 2, 3, 4)))
    return {
        **state,
        "raw_ensemble": certified_mean,
        "uncertainty_score": uncertainty,
        "ensemble_correction_ratio": correction_rms / physical_scale,
        "ensemble_alpha": alpha,
        "ensemble_bound": upper_bound,
        "ensemble_descent": descent,
        "member_acceptance_rate": torch.stack(member_acceptance).mean(dim=0),
        "member_raw_ratio_mean": torch.stack(member_ratios).mean(dim=0),
    }


def apply_uncertainty_gate(
    state: dict[str, torch.Tensor], threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    accept = state["uncertainty_score"] <= float(threshold)
    prediction = torch.where(
        accept[:, None, None, None, None], state["raw_ensemble"], state["fallback"]
    )
    return prediction, accept


def field_metrics(field: torch.Tensor, batch, operator) -> dict[str, np.ndarray]:
    residual = batch.observation - operator.forward(field, batch)
    white = batch.whitened(residual)
    active_count = batch.active_observation_mask().flatten(1).sum(dim=1).clamp_min(1)
    return {
        "relative_l2": relative_l2(field, batch.truth).cpu().numpy(),
        "gradient_relative_l2": gradient_relative_l2(field, batch.truth).cpu().numpy(),
        "front_f1": front_f1(field, batch.truth, batch.expanded_support()).cpu().numpy(),
        "whitened_residual_rms": torch.sqrt(
            torch.sum(white.square(), dim=(1, 2, 3)) / active_count
        ).cpu().numpy(),
    }


@torch.no_grad()
def evaluate_ensemble(
    models: list[PBBBaseCorrectionCGPDNO],
    batch,
    operator,
    lipschitz: torch.Tensor,
    baseline_config: dict,
    *,
    threshold: float,
) -> dict[str, object]:
    counted_operator = WeightedGradientCounter(operator)
    state = shared_ensemble_state(models, batch, counted_operator, lipschitz)
    if counted_operator.calls != models[0].stages:
        raise RuntimeError("shared candidate physical-call count disagrees with stages")
    candidate, accepted = apply_uncertainty_gate(state, threshold)
    stages = models[0].stages
    fixed = fixed_projected_gradient(batch, operator, lipschitz, stages)
    pbb = projected_bb(
        batch,
        operator,
        lipschitz,
        stages,
        float(baseline_config["bb_normalized_step_min"]),
        float(baseline_config["bb_normalized_step_max"]),
    )
    if not torch.allclose(state["fallback"], pbb, rtol=2e-5, atol=2e-6):
        raise RuntimeError("shared fallback and declared projected-BB baseline disagree")
    fields = {
        "candidate": candidate,
        "raw_ensemble": state["raw_ensemble"],
        "pbb_fallback": state["fallback"],
        "fixed_pg": fixed,
        "projected_bb": pbb,
        "fista": projected_fista(batch, operator, lipschitz, stages),
    }
    metrics = {name: field_metrics(field, batch, operator) for name, field in fields.items()}
    base_residual = batch.observation - operator.forward(state["base"], batch)
    base_objective = 0.5 * torch.sum(batch.whitened(base_residual).square(), dim=(1, 2, 3))
    candidate_residual = batch.observation - operator.forward(candidate, batch)
    candidate_objective = 0.5 * torch.sum(
        batch.whitened(candidate_residual).square(), dim=(1, 2, 3)
    )
    return {
        "metrics": metrics,
        "accepted": accepted.cpu().numpy(),
        "uncertainty": state["uncertainty_score"].cpu().numpy(),
        "ensemble_correction_ratio": state["ensemble_correction_ratio"].cpu().numpy(),
        "ensemble_alpha": state["ensemble_alpha"].cpu().numpy(),
        "member_acceptance_rate": state["member_acceptance_rate"].cpu().numpy(),
        "member_raw_ratio_mean": state["member_raw_ratio_mean"].cpu().numpy(),
        "certificate_violation": (
            candidate_objective > base_objective * (1.0 + 1e-6) + 1e-6
        ).cpu().numpy(),
        "forward_calls": counted_operator.calls,
        "adjoint_calls": counted_operator.calls,
        "correction_head_passes": len(models),
    }


def threshold_candidates(values: np.ndarray, quantile_count: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, int(quantile_count))
    candidates = np.unique(np.quantile(values, quantiles))
    upper = float(np.max(values)) + max(1e-9, 1e-6 * float(np.max(values)))
    return np.concatenate((np.array([-1.0]), candidates, np.array([upper])))


def lipschitz_values(config: dict, operator, batch) -> torch.Tensor:
    method = config.get("lipschitz_method", "power_iteration")
    if method == "exact_small_matrix":
        estimate = operator.weighted_lipschitz_exact(batch)
    elif method == "power_iteration":
        estimate = operator.weighted_lipschitz(
            batch, power_iterations=int(config["power_iterations"])
        )
    else:
        raise ValueError(f"unknown lipschitz_method: {method}")
    return float(config["lipschitz_safety_factor"]) * estimate


def gated_gain(
    raw_error: np.ndarray,
    fallback_error: np.ndarray,
    baseline_error: np.ndarray,
    uncertainty: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    accepted = uncertainty <= float(threshold)
    error = np.where(accepted, raw_error, fallback_error)
    gain = 100.0 * (baseline_error - error) / np.maximum(baseline_error, 1e-12)
    return gain, accepted


def calibrate_threshold(
    result: dict[str, object],
    selected_baseline: str,
    config: dict,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    metrics = result["metrics"]
    uncertainty = np.asarray(result["uncertainty"])
    rows: list[dict[str, object]] = []
    constraints = config["selection_gate"]
    for threshold in threshold_candidates(uncertainty, int(constraints["quantile_count"])):
        gain, accepted = gated_gain(
            metrics["raw_ensemble"]["relative_l2"],
            metrics["pbb_fallback"]["relative_l2"],
            metrics[selected_baseline]["relative_l2"],
            uncertainty,
            float(threshold),
        )
        row = {
            "threshold": float(threshold),
            "coverage": float(np.mean(accepted)),
            "mean_gain_percent": float(np.mean(gain)),
            "p10_gain_percent": float(np.quantile(gain, 0.10)),
            "harm_rate_over_1_percent": float(np.mean(gain < -1.0)),
        }
        row["feasible"] = bool(
            row["coverage"] >= float(constraints["minimum_coverage"])
            and row["p10_gain_percent"] >= float(constraints["minimum_p10_gain_percent"])
            and row["harm_rate_over_1_percent"]
            <= float(constraints["maximum_harm_rate_over_1_percent"])
        )
        rows.append(row)
    feasible = [row for row in rows if row["feasible"]]
    if feasible:
        chosen = max(
            feasible,
            key=lambda row: (
                row["mean_gain_percent"],
                row["coverage"],
                -row["threshold"],
            ),
        )
        reason = "best_select_mean_gain_subject_to_predeclared_tail_and_coverage_constraints"
    else:
        chosen = rows[0]
        reason = "no_feasible_select_threshold_abstain_all"
    return {**chosen, "selection_reason": reason}, rows


def summary_from_result(result: dict[str, object], baseline: str) -> dict[str, float]:
    metrics = result["metrics"]
    gain = 100.0 * (
        metrics[baseline]["relative_l2"] - metrics["candidate"]["relative_l2"]
    ) / np.maximum(metrics[baseline]["relative_l2"], 1e-12)
    return {
        "candidate_mean_relative_l2": float(np.mean(metrics["candidate"]["relative_l2"])),
        "baseline_mean_relative_l2": float(np.mean(metrics[baseline]["relative_l2"])),
        "mean_gain_percent": float(np.mean(gain)),
        "p10_gain_percent": float(np.quantile(gain, 0.10)),
        "harm_rate_over_1_percent": float(np.mean(gain < -1.0)),
        "coverage": float(np.mean(result["accepted"])),
        "certificate_violation_rate": float(np.mean(result["certificate_violation"])),
        "candidate_mean_gradient_relative_l2": float(
            np.mean(metrics["candidate"]["gradient_relative_l2"])
        ),
        "baseline_mean_gradient_relative_l2": float(
            np.mean(metrics[baseline]["gradient_relative_l2"])
        ),
        "candidate_mean_front_f1": float(np.mean(metrics["candidate"]["front_f1"])),
        "baseline_mean_front_f1": float(np.mean(metrics[baseline]["front_f1"])),
    }


def sample_rows(split: str, batch, result: dict[str, object], selected_baseline: str) -> list[dict[str, object]]:
    metrics = result["metrics"]
    rows = []
    for index, geometry_id in enumerate(batch.geometry_ids):
        row: dict[str, object] = {
            "split": split,
            "sample_index": index,
            "geometry_id": geometry_id,
            "active_views": int(batch.view_mask[index].sum()),
            "selected_baseline": selected_baseline,
            "accepted": bool(result["accepted"][index]),
            "uncertainty_score": float(result["uncertainty"][index]),
            "ensemble_correction_ratio": float(result["ensemble_correction_ratio"][index]),
            "ensemble_alpha": float(result["ensemble_alpha"][index]),
            "member_acceptance_rate": float(result["member_acceptance_rate"][index]),
            "member_raw_ratio_mean": float(result["member_raw_ratio_mean"][index]),
            "certificate_violation": bool(result["certificate_violation"][index]),
        }
        for method in METHODS:
            for metric, values in metrics[method].items():
                row[f"{method}_{metric}"] = float(values[index])
        for baseline in ("fixed_pg", "projected_bb", "fista"):
            row[f"gain_vs_{baseline}_percent"] = float(
                100.0
                * (metrics[baseline]["relative_l2"][index] - metrics["candidate"]["relative_l2"][index])
                / max(metrics[baseline]["relative_l2"][index], 1e-12)
            )
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config_snapshot.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    n, depth = int(config["grid_size"]), int(config["depth"])
    train_angles = np.asarray(config["train_view_angles_degrees"], dtype=np.float32)
    train_matrix = build_forward_matrix(n, train_angles)
    train_operator = DepthSeparableLinearBOST(torch.from_numpy(train_matrix), (depth, n, n))
    train = analytic_batch(config, "train", train_matrix, set())
    train_ids = set(train.geometry_ids)
    validation = analytic_batch(config, "validation", train_matrix, train_ids)
    iterations = int(config["power_iterations"])
    train_lipschitz = lipschitz_values(config, train_operator, train)
    validation_lipschitz = lipschitz_values(config, train_operator, validation)

    select_rig = config["independent_rigs"]["independent_select"]
    select_matrix = build_curved_cone_operator(
        n,
        depth,
        np.asarray(select_rig["angles_degrees"], dtype=np.float32),
        path_samples=int(select_rig["path_samples"]),
        cone_u=float(select_rig["cone_u"]),
        cone_z=float(select_rig["cone_z"]),
        bend=float(select_rig["bend"]),
    )
    select_operator = DenseVolumeLinearBOST(torch.from_numpy(select_matrix), (depth, n, n))
    select = independent_batch(config, "independent_select", select_matrix, set())
    select_lipschitz = lipschitz_values(config, select_operator, select)

    models: list[PBBBaseCorrectionCGPDNO] = []
    history_rows: list[dict[str, object]] = []
    best_epochs: dict[str, int] = {}
    started = time.perf_counter()
    for seed_value in config["training"]["seeds"]:
        seed = int(seed_value)
        model, history, best_epoch = train_seed(
            config,
            seed,
            train,
            validation,
            train_operator,
            train_lipschitz,
            validation_lipschitz,
        )
        if not isinstance(model, PBBBaseCorrectionCGPDNO):
            raise TypeError("ensemble gate requires model_family=projected_bb_base")
        models.append(model)
        history_rows.extend(history)
        best_epochs[str(seed)] = int(best_epoch)

    raw_select = evaluate_ensemble(
        models,
        select,
        select_operator,
        select_lipschitz,
        config["numerical_baselines"],
        threshold=float("inf"),
    )
    baseline_means = {
        name: float(np.mean(raw_select["metrics"][name]["relative_l2"]))
        for name in ("fixed_pg", "projected_bb", "fista")
    }
    selected_baseline = min(baseline_means, key=baseline_means.get)
    threshold, calibration_rows = calibrate_threshold(raw_select, selected_baseline, config)
    select_result = evaluate_ensemble(
        models,
        select,
        select_operator,
        select_lipschitz,
        config["numerical_baselines"],
        threshold=float(threshold["threshold"]),
    )
    selection_path = output / "selection_commit.json"
    selection = {
        "created_before_independent_lock": True,
        "selection_role": "independent_select_ground_truth_used_only_for_threshold_calibration",
        "selected_deterministic_baseline": selected_baseline,
        "baseline_mean_relative_l2": baseline_means,
        "uncertainty_definition": "RMS ensemble correction disagreement divided by RMS shared PBB direction",
        "threshold_selection": threshold,
        "best_epoch_by_seed": best_epochs,
        "config_sha256": sha256(config_path),
        "train_operator_sha256": array_sha256(train_matrix),
        "select_operator_sha256": array_sha256(select_matrix),
    }
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # The lock rig and fields do not exist until the selection commit is durable.
    lock_rig = config["independent_rigs"]["independent_lock"]
    lock_matrix = build_curved_cone_operator(
        n,
        depth,
        np.asarray(lock_rig["angles_degrees"], dtype=np.float32),
        path_samples=int(lock_rig["path_samples"]),
        cone_u=float(lock_rig["cone_u"]),
        cone_z=float(lock_rig["cone_z"]),
        bend=float(lock_rig["bend"]),
    )
    lock_operator = DenseVolumeLinearBOST(torch.from_numpy(lock_matrix), (depth, n, n))
    lock = independent_batch(config, "independent_lock", lock_matrix, set(select.geometry_ids))
    lock_lipschitz = lipschitz_values(config, lock_operator, lock)
    lock_result = evaluate_ensemble(
        models,
        lock,
        lock_operator,
        lock_lipschitz,
        config["numerical_baselines"],
        threshold=float(threshold["threshold"]),
    )
    elapsed = time.perf_counter() - started

    select_summary = summary_from_result(select_result, selected_baseline)
    lock_summary = summary_from_result(lock_result, selected_baseline)
    gate = config["claim_gate"]
    gate_checks = {
        "mean_gain": lock_summary["mean_gain_percent"] >= float(gate["minimum_mean_gain_percent"]),
        "p10_gain": lock_summary["p10_gain_percent"] >= float(gate["minimum_p10_gain_percent"]),
        "harm_rate": lock_summary["harm_rate_over_1_percent"]
        <= float(gate["maximum_harm_rate_over_1_percent"]),
        "certificate": lock_summary["certificate_violation_rate"]
        <= float(gate["maximum_certificate_violation_rate"]),
    }
    claim_status = (
        "INDEPENDENT_LINEAR_SELECTIVE_ENSEMBLE_GATE_PASSED_REAL_DATA_AND_STRONG_LEARNED_BASELINES_REQUIRED"
        if all(gate_checks.values())
        else "INDEPENDENT_LINEAR_SELECTIVE_ENSEMBLE_GATE_FAILED_OR_INCOMPLETE"
    )

    history_path = output / "history.csv"
    calibration_path = output / "threshold_calibration.csv"
    samples_path = output / "sample_metrics.csv"
    summary_path = output / "summary.csv"
    write_csv(history_path, history_rows)
    write_csv(calibration_path, calibration_rows)
    rows = sample_rows("independent_select", select, select_result, selected_baseline)
    rows.extend(sample_rows("independent_lock", lock, lock_result, selected_baseline))
    write_csv(samples_path, rows)
    write_csv(
        summary_path,
        [
            {"split": "independent_select", "baseline": selected_baseline, **select_summary},
            {"split": "independent_lock", "baseline": selected_baseline, **lock_summary},
        ],
    )

    figure_path = output / "pbb_ensemble_selective_gate.png"
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    axes[0].scatter(
        raw_select["uncertainty"],
        100.0
        * (
            raw_select["metrics"][selected_baseline]["relative_l2"]
            - raw_select["metrics"]["raw_ensemble"]["relative_l2"]
        )
        / raw_select["metrics"][selected_baseline]["relative_l2"],
        s=24,
        alpha=0.75,
    )
    axes[0].axvline(float(threshold["threshold"]), color="tab:red", ls="--")
    axes[0].axhline(0.0, color="black", lw=1)
    axes[0].set(title="Select: disagreement vs gain", xlabel="uncertainty score", ylabel="raw gain (%)")
    for axis, split, result in zip(
        axes[1:],
        ("select", "fresh lock"),
        (select_result, lock_result),
    ):
        gains = 100.0 * (
            result["metrics"][selected_baseline]["relative_l2"]
            - result["metrics"]["candidate"]["relative_l2"]
        ) / result["metrics"][selected_baseline]["relative_l2"]
        axis.hist(gains, bins=14, color="tab:blue", alpha=0.78)
        axis.axvline(0.0, color="black", lw=1)
        axis.set(title=f"{split}: selective gain", xlabel="gain (%)", ylabel="count")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    report_path = output / "report.json"
    report = {
        "evidence_label": config["evidence_label"],
        "claim_status": claim_status,
        "claim_boundary": "small prescribed linear weak-deflection curved/cone generator only; no nonlinear ray tracing, CFD, OpenBOST/OERF, DeepONet, FNO/FFNO or Learned Primal-Dual comparison",
        "lock_status": "FIRST_OPEN; threshold and architecture were committed before lock construction",
        "selected_deterministic_baseline": selected_baseline,
        "threshold_selection": threshold,
        "gate_checks": gate_checks,
        "select_summary": select_summary,
        "lock_summary": lock_summary,
        "operator_audit": {
            "train_kind": "depth-separable straight weak-deflection",
            "select_kind": "fully-3D prescribed curved/cone weak-deflection",
            "lock_kind": "stronger fully-3D prescribed curved/cone weak-deflection",
            "train_sha256": array_sha256(train_matrix),
            "select_sha256": array_sha256(select_matrix),
            "lock_sha256": array_sha256(lock_matrix),
            "select_lock_equal": bool(np.array_equal(select_matrix, lock_matrix)),
            "select_adjoint_relative_error": select_operator.adjoint_relative_error(select, 81),
            "lock_adjoint_relative_error": lock_operator.adjoint_relative_error(lock, 82),
        },
        "geometry_overlap": {
            "train_validation": sorted(train_ids & set(validation.geometry_ids)),
            "select_lock": sorted(set(select.geometry_ids) & set(lock.geometry_ids)),
        },
        "call_accounting": {
            "candidate_shared_forward": models[0].stages,
            "candidate_shared_adjoint": models[0].stages,
            "correction_head_passes": len(models),
            "physical_trajectory_repeated_per_head": False,
            "fixed_pg_forward": models[0].stages,
            "fixed_pg_adjoint": models[0].stages,
            "projected_bb_forward": models[0].stages,
            "projected_bb_adjoint": models[0].stages,
            "fista_forward": models[0].stages,
            "fista_adjoint": models[0].stages,
            "lipschitz_power_iterations_precomputation": (
                iterations if config.get("lipschitz_method", "power_iteration") == "power_iteration" else 0
            ),
            "lipschitz_method": config.get("lipschitz_method", "power_iteration"),
            "exact_spectral_decompositions": (
                len(train.geometry_ids) * train.observation.shape[1]
                + len(validation.geometry_ids) * validation.observation.shape[1]
                + len(select.geometry_ids)
                + len(lock.geometry_ids)
                if config.get("lipschitz_method") == "exact_small_matrix"
                else 0
            ),
            "metric_only_forward_per_method": 1,
        },
        "training": {
            "best_epoch_by_seed": best_epochs,
            "elapsed_seconds": elapsed,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "member_count": len(models),
            "parameter_count_per_member": sum(p.numel() for p in models[0].parameters()),
            "parameter_count_total": sum(sum(p.numel() for p in model.parameters()) for model in models),
        },
        "selection_commit_sha256": sha256(selection_path),
        "source_sha256": {
            "runner": sha256(Path(__file__)),
            "cg_pdno": sha256(ROOT / "cg_pdno.py"),
            "independent_generator": sha256(ROOT / "independent_reaction_bost.py"),
            "measurement_contract": sha256(ROOT / "measurement_contract.py"),
        },
        "config": config,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files = [
        config_path,
        selection_path,
        history_path,
        calibration_path,
        samples_path,
        summary_path,
        figure_path,
        report_path,
    ]
    checksum_path = output / "checksums.sha256"
    checksum_path.write_text(
        "\n".join(f"{sha256(path)}  {path.name}" for path in files) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "claim_status": claim_status,
                "selected_baseline": selected_baseline,
                "threshold": threshold,
                "lock": lock_summary,
                "gate_checks": gate_checks,
                "elapsed_seconds": elapsed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
