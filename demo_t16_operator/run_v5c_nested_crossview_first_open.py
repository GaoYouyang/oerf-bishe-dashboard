#!/usr/bin/env python3
"""Run the precommitted v5c nested cross-view new-generator-family first open.

Radius and a dimensionless ridge ratio are selected only inside the declared
inner-camera set.  Two outer cameras form a transparent deployment gate and the
final audit camera is read only after every decision has been constructed.
The same process still owns truth and audit data, so this is a development
first-open rather than a confirmatory or deployment guarantee.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .nested_crossview import (
        CrossViewSelection,
        refit_scaled_selection,
        select_radius_kappa_crossview,
        whitened_per_view_rms,
    )
    from .rig_shared_profile import profile_shared_radius
    from .run_v5b_rig_shared_profile_pilot import (
        DevelopmentBlock,
        build_development_blocks,
        nearest_index,
        oracle_field_scores,
        read_json,
        relative_l2,
        safe_percent_change,
        support_mask_from_config,
        validate_forward_model_gate,
    )
except ImportError:
    from nested_crossview import (
        CrossViewSelection,
        refit_scaled_selection,
        select_radius_kappa_crossview,
        whitened_per_view_rms,
    )
    from rig_shared_profile import profile_shared_radius
    from run_v5b_rig_shared_profile_pilot import (
        DevelopmentBlock,
        build_development_blocks,
        nearest_index,
        oracle_field_scores,
        read_json,
        relative_l2,
        safe_percent_change,
        support_mask_from_config,
        validate_forward_model_gate,
    )


ROOT = Path(__file__).resolve().parent
SITE_ROOT = ROOT.parent
DEFAULT_CONFIG = ROOT / "configs" / "v5c_nested_crossview_first_open.json"
DEFAULT_OUTPUT = ROOT / "results" / "v5c_nested_crossview_first_open"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def preopen_git_commit(paths: Sequence[Path]) -> str:
    """Require every decision dependency to be tracked and clean before opening."""

    relative = [str(path.resolve().relative_to(SITE_ROOT)) for path in paths]
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", *relative],
        cwd=SITE_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--", *relative],
        cwd=SITE_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(f"first-open dependency is not committed and clean: {status}")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=SITE_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _percent_improvement(candidate: float, baseline: float) -> float:
    return float(100.0 * (baseline - candidate) / max(abs(baseline), 1e-12))


def outer_view_block_metrics(improvements: np.ndarray) -> dict[str, tuple[float, ...]]:
    """Summarize every outer camera separately; never pool camera observations."""

    values = np.asarray(improvements, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("outer improvements must be a nonempty sample-by-view matrix")
    if np.any(~np.isfinite(values)):
        raise ValueError("outer improvements must be finite")
    return {
        "outer_view_median_improvements_percent": tuple(
            float(value) for value in np.median(values, axis=0)
        ),
        "outer_view_positive_fractions": tuple(
            float(value) for value in np.mean(values > 0.0, axis=0)
        ),
        "outer_view_worst_improvements_percent": tuple(
            float(value) for value in np.min(values, axis=0)
        ),
    }


def block_gate_reasons(
    metrics: dict[str, Any], acceptance: dict[str, Any]
) -> tuple[bool, tuple[str, ...]]:
    """Evaluate the frozen block gate without reading any audit quantity."""

    reasons: list[str] = []
    if float(metrics["camera_deletion_radius_stability_fraction"]) < float(
        acceptance["minimum_camera_deletion_radius_stability_fraction"]
    ):
        reasons.append("radius_unstable")
    if float(metrics["relative_radius_margin"]) < float(
        acceptance["minimum_relative_radius_margin"]
    ):
        reasons.append("radius_margin_small")
    medians = tuple(float(value) for value in metrics["outer_view_median_improvements_percent"])
    positive = tuple(float(value) for value in metrics["outer_view_positive_fractions"])
    worst = tuple(float(value) for value in metrics["outer_view_worst_improvements_percent"])
    if not medians or len(medians) != len(positive) or len(medians) != len(worst):
        raise ValueError("outer block metrics must contain equally sized nonempty view vectors")
    for view_offset, (median, fraction, minimum) in enumerate(
        zip(medians, positive, worst, strict=True)
    ):
        if median < float(
            acceptance["minimum_block_outer_view_median_improvement_percent"]
        ):
            reasons.append(f"outer_view_{view_offset}_median_below_threshold")
        if fraction < float(acceptance["minimum_block_outer_view_positive_fraction"]):
            reasons.append(f"outer_view_{view_offset}_sign_inconsistent")
        if minimum < float(
            acceptance["minimum_block_outer_view_worst_improvement_percent"]
        ):
            reasons.append(f"outer_view_{view_offset}_tail_harm")
    if float(metrics["metadata_z"]) > float(acceptance["maximum_metadata_z"]):
        reasons.append("metadata_conflict")
    if bool(metrics["boundary"]):
        reasons.append("radius_boundary")
    if bool(acceptance.get("require_radius_change", False)) and not bool(
        metrics["radius_changed"]
    ):
        reasons.append("no_optical_change")
    return not reasons, tuple(reasons)


def sample_outer_gate(
    improvements: Sequence[float], acceptance: dict[str, Any]
) -> tuple[bool, tuple[str, ...]]:
    """Evaluate one target using outer cameras only, never the audit camera."""

    values = np.asarray(tuple(float(value) for value in improvements), dtype=float)
    if values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("outer improvements must be finite and nonempty")
    threshold = float(acceptance["minimum_sample_outer_improvement_percent"])
    reasons: list[str] = []
    if bool(acceptance.get("require_all_outer_views_improve", False)):
        if np.any(values < threshold):
            reasons.append("outer_camera_not_improved")
    elif float(np.mean(values)) < threshold:
        reasons.append("outer_mean_not_improved")
    return not reasons, tuple(reasons)


def _selection_rows(
    block: DevelopmentBlock,
    selection: CrossViewSelection,
    method: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected = selection.selected_candidate_index
    fold_score_deletion = set(selection.fold_score_deletion_candidate_indices)
    for candidate_index, candidate in enumerate(selection.candidates):
        row: dict[str, Any] = {
            "rig_id": block.rig_id,
            "block_id": block.block_id,
            "method": method,
            "true_aperture_radius": block.true_radius,
            "metadata_aperture_radius": block.metadata_radius,
            "candidate_aperture_radius": candidate.radius,
            "kappa": candidate.kappa,
            "mean_validation_mse": candidate.mean_validation_mse,
            "median_effective_lambda": candidate.median_effective_lambda,
            "selected": candidate_index == selected,
            "selected_by_any_fold_score_deletion": candidate_index
            in fold_score_deletion,
            "validation_views": "|".join(str(view) for view in selection.validation_views),
        }
        for fold, value in enumerate(candidate.fold_validation_mse):
            row[f"fold_{fold}_validation_mse"] = value
        rows.append(row)
    return rows


def _safe_correlation(x: Sequence[float], y: Sequence[float]) -> float | None:
    first = np.asarray(x, dtype=float)
    second = np.asarray(y, dtype=float)
    if len(first) < 2 or np.std(first) <= 1e-15 or np.std(second) <= 1e-15:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def evaluate(
    config: dict[str, Any], blocks: Sequence[DevelopmentBlock]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Construct frozen inner/outer decisions, then open truth and audit diagnostics."""

    radii = np.asarray(config["candidate_aperture_radii"], dtype=float)
    kappas = tuple(float(value) for value in config["crossview_kappas"])
    support = support_mask_from_config(config)
    if np.asarray(support).dtype != bool:
        raise ValueError("configured support must be an explicit boolean mask")
    acceptance = dict(config["acceptance"])
    fixed_ridge = float(config["ridge_lambda"])
    metadata_sigma = float(config["metadata_sigma"])

    candidate_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    postdecision_blocks: list[dict[str, Any]] = []
    postdecision_samples: list[dict[str, Any]] = []

    for block in blocks:
        if len(block.outer_views) != 2:
            raise ValueError("v5c two-way transfer requires exactly two outer views")
        if len(block.audit_views) != 1:
            raise ValueError("v5c requires exactly one sealed audit view")

        selection = select_radius_kappa_crossview(
            block.reconstruction_bank,
            radii,
            block.observations,
            block.noise_std,
            block.inner_views,
            support,
            kappas,
        )
        camera_deletions: list[tuple[int, CrossViewSelection]] = []
        for omitted_view in block.inner_views:
            reduced_views = tuple(
                view for view in block.inner_views if view != omitted_view
            )
            deleted = select_radius_kappa_crossview(
                block.reconstruction_bank,
                radii,
                block.observations,
                block.noise_std,
                reduced_views,
                support,
                kappas,
            )
            camera_deletions.append((omitted_view, deleted))
            candidate_rows.extend(
                _selection_rows(block, deleted, f"camera_delete_{omitted_view}")
            )

        candidate_refit = refit_scaled_selection(
            selection,
            block.reconstruction_bank,
            block.observations,
            block.noise_std,
            block.inner_views,
            support,
        )
        metadata_index = nearest_index(radii, block.metadata_radius)
        fallback_selection = select_radius_kappa_crossview(
            block.reconstruction_bank[metadata_index : metadata_index + 1],
            radii[metadata_index : metadata_index + 1],
            block.observations,
            block.noise_std,
            block.inner_views,
            support,
            kappas,
        )
        fallback_refit = refit_scaled_selection(
            fallback_selection,
            block.reconstruction_bank[metadata_index : metadata_index + 1],
            block.observations,
            block.noise_std,
            block.inner_views,
            support,
        )

        inner_bank = np.take(
            block.reconstruction_bank,
            np.asarray(block.inner_views, dtype=int),
            axis=2,
        )
        inner_observations = tuple(
            np.take(observation, np.asarray(block.inner_views, dtype=int), axis=1)
            for observation in block.observations
        )
        inner_noise = tuple(
            np.take(np.asarray(sigma), np.asarray(block.inner_views, dtype=int))
            for sigma in block.noise_std
        )
        local_inner_views = tuple(range(len(block.inner_views)))
        fixed_profile = profile_shared_radius(
            inner_bank,
            radii,
            inner_observations,
            inner_noise,
            local_inner_views,
            support,
            fixed_ridge,
        )

        candidate_rows.extend(_selection_rows(block, selection, "radius_kappa"))
        candidate_rows.extend(
            _selection_rows(block, fallback_selection, "metadata_kappa")
        )

        selected_index = int(selection.selected.radius_index)
        candidate_operator = block.reconstruction_bank[selected_index]
        fallback_operator = block.reconstruction_bank[metadata_index]
        pending_samples: list[dict[str, Any]] = []
        for sample, (
            family,
            truth,
            observation,
            sigma,
            candidate_fit,
            fallback_fit,
        ) in enumerate(
            zip(
                block.families,
                block.fields,
                block.observations,
                block.noise_std,
                candidate_refit.fits,
                fallback_refit.fits,
                strict=True,
            )
        ):
            candidate_outer = whitened_per_view_rms(
                candidate_operator,
                candidate_fit.field,
                observation,
                sigma,
                block.outer_views,
            )
            fallback_outer = whitened_per_view_rms(
                fallback_operator,
                fallback_fit.field,
                observation,
                sigma,
                block.outer_views,
            )
            outer_improvements = tuple(
                _percent_improvement(candidate, fallback)
                for candidate, fallback in zip(
                    candidate_outer, fallback_outer, strict=True
                )
            )
            pending_samples.append(
                {
                    "sample": sample,
                    "family": family,
                    "truth": truth,
                    "observation": observation,
                    "sigma": sigma,
                    "candidate_fit": candidate_fit,
                    "fallback_fit": fallback_fit,
                    "candidate_outer": candidate_outer,
                    "fallback_outer": fallback_outer,
                    "outer_improvements": outer_improvements,
                }
            )

        outer_matrix = np.asarray(
            [pending["outer_improvements"] for pending in pending_samples],
            dtype=float,
        )
        full_outer_metrics = outer_view_block_metrics(outer_matrix)
        camera_deletion_radius_stability = float(
            np.mean(
                [
                    deleted.selected.radius_index == selected_index
                    for _, deleted in camera_deletions
                ]
            )
        )
        camera_deletion_kappa_stability = float(
            np.mean(
                [
                    deleted.selected.kappa == selection.selected.kappa
                    for _, deleted in camera_deletions
                ]
            )
        )
        candidate_cv = float(selection.selected.mean_validation_mse)
        fallback_cv = float(fallback_selection.selected.mean_validation_mse)
        base_block_metrics = {
            "camera_deletion_radius_stability_fraction": (
                camera_deletion_radius_stability
            ),
            "camera_deletion_kappa_stability_fraction": (
                camera_deletion_kappa_stability
            ),
            "fold_score_deletion_radius_stability_fraction": (
                selection.fold_score_deletion_radius_stability_fraction
            ),
            "fold_score_deletion_kappa_stability_fraction": (
                selection.fold_score_deletion_kappa_stability_fraction
            ),
            "relative_score_margin": selection.relative_score_margin,
            "relative_radius_margin": selection.relative_radius_margin,
            "candidate_cv_advantage_percent_asymmetric_diagnostic": (
                _percent_improvement(candidate_cv, fallback_cv)
            ),
            "metadata_z": float(
                abs(float(radii[selected_index]) - block.metadata_radius)
                / metadata_sigma
            ),
            "boundary": selected_index in (0, len(radii) - 1),
            "radius_changed": selected_index != metadata_index,
        }
        full_block_metrics = {**base_block_metrics, **full_outer_metrics}
        full_block_gate, full_block_reasons = block_gate_reasons(
            full_block_metrics, acceptance
        )

        block_row: dict[str, Any] = {
            "rig_id": block.rig_id,
            "block_id": block.block_id,
            "true_aperture_radius": block.true_radius,
            "metadata_aperture_radius": block.metadata_radius,
            "metadata_nearest_radius": float(radii[metadata_index]),
            "selected_radius": float(radii[selected_index]),
            "selected_kappa": selection.selected.kappa,
            "fallback_kappa": fallback_selection.selected.kappa,
            "fixed_ridge_profile_radius": float(
                radii[fixed_profile.selected_index]
            ),
            "camera_deletion_omitted_views": "|".join(
                str(view) for view, _ in camera_deletions
            ),
            "camera_deletion_selected_radii": "|".join(
                f"{deleted.selected.radius:.12g}" for _, deleted in camera_deletions
            ),
            "camera_deletion_selected_kappas": "|".join(
                f"{deleted.selected.kappa:.12g}" for _, deleted in camera_deletions
            ),
            "candidate_cv_mse": candidate_cv,
            "fallback_cv_mse": fallback_cv,
            **full_block_metrics,
            "block_gate": full_block_gate,
            "block_gate_reasons": "|".join(full_block_reasons),
            "block_gate_scope": (
                "full_block_descriptive_not_used_for_sample_decision"
            ),
        }
        for view_offset, camera_index in enumerate(block.outer_views):
            block_row[f"outer_{view_offset}_camera_index"] = camera_index
            block_row[
                f"outer_{view_offset}_median_improvement_percent"
            ] = full_outer_metrics[
                "outer_view_median_improvements_percent"
            ][view_offset]
            block_row[
                f"outer_{view_offset}_positive_fraction"
            ] = full_outer_metrics["outer_view_positive_fractions"][view_offset]
            block_row[
                f"outer_{view_offset}_worst_improvement_percent"
            ] = full_outer_metrics[
                "outer_view_worst_improvements_percent"
            ][view_offset]
        block_rows.append(block_row)
        postdecision_blocks.append(
            {
                "row": block_row,
                "block": block,
                "selected_index": selected_index,
            }
        )

        for sample_offset, pending in enumerate(pending_samples):
            reduced_outer = np.delete(outer_matrix, sample_offset, axis=0)
            leave_one_out_metrics = outer_view_block_metrics(reduced_outer)
            sample_block_metrics = {
                **base_block_metrics,
                **leave_one_out_metrics,
            }
            sample_block_gate, sample_block_reasons = block_gate_reasons(
                sample_block_metrics, acceptance
            )
            sample_gate, sample_reasons = sample_outer_gate(
                pending["outer_improvements"], acceptance
            )

            directions: list[dict[str, Any]] = []
            for gate_offset in range(2):
                evaluation_offset = 1 - gate_offset
                directional_metrics = outer_view_block_metrics(
                    reduced_outer[:, gate_offset : gate_offset + 1]
                )
                direction_block_gate, direction_block_reasons = block_gate_reasons(
                    {**base_block_metrics, **directional_metrics},
                    acceptance,
                )
                direction_sample_gate, direction_sample_reasons = sample_outer_gate(
                    [pending["outer_improvements"][gate_offset]],
                    acceptance,
                )
                directions.append(
                    {
                        "gate_offset": gate_offset,
                        "evaluation_offset": evaluation_offset,
                        "block_gate": direction_block_gate,
                        "block_reasons": direction_block_reasons,
                        "sample_gate": direction_sample_gate,
                        "sample_reasons": direction_sample_reasons,
                        "accepted": bool(
                            direction_block_gate and direction_sample_gate
                        ),
                        "heldout_improvement": float(
                            pending["outer_improvements"][evaluation_offset]
                        ),
                    }
                )

            accepted = bool(all(record["accepted"] for record in directions))
            if accepted != bool(sample_block_gate and sample_gate):
                raise AssertionError(
                    "two-way outer intersection and joint gate disagree"
                )
            if not bool(base_block_metrics["radius_changed"]):
                outcome = "NO_ACTION_FALLBACK"
            elif not all(record["block_gate"] for record in directions):
                outcome = "REJECT_BLOCK"
            elif not all(record["sample_gate"] for record in directions):
                outcome = "REJECT_SAMPLE_OUTER"
            else:
                outcome = "ACCEPT"

            row: dict[str, Any] = {
                "rig_id": block.rig_id,
                "block_id": block.block_id,
                "sample_index_in_block": pending["sample"],
                "family": pending["family"],
                "true_aperture_radius": block.true_radius,
                "metadata_nearest_radius": float(radii[metadata_index]),
                "selected_radius": float(radii[selected_index]),
                "selected_kappa": selection.selected.kappa,
                "fallback_kappa": fallback_selection.selected.kappa,
                "camera_deletion_radius_stability_fraction": (
                    camera_deletion_radius_stability
                ),
                "camera_deletion_kappa_stability_fraction": (
                    camera_deletion_kappa_stability
                ),
                "fold_score_deletion_radius_stability_fraction": (
                    selection.fold_score_deletion_radius_stability_fraction
                ),
                "fold_score_deletion_kappa_stability_fraction": (
                    selection.fold_score_deletion_kappa_stability_fraction
                ),
                "relative_score_margin": selection.relative_score_margin,
                "relative_radius_margin": selection.relative_radius_margin,
                "candidate_cv_advantage_percent_asymmetric_diagnostic": (
                    base_block_metrics[
                        "candidate_cv_advantage_percent_asymmetric_diagnostic"
                    ]
                ),
                "minimum_outer_improvement_percent": float(
                    np.min(pending["outer_improvements"])
                ),
                "mean_outer_improvement_percent": float(
                    np.mean(pending["outer_improvements"])
                ),
                "block_gate_minus_sample": sample_block_gate,
                "block_gate_minus_sample_reasons": "|".join(
                    sample_block_reasons
                ),
                "sample_outer_gate": sample_gate,
                "sample_outer_gate_reasons": "|".join(sample_reasons),
                "accepted": accepted,
                "outcome_code": outcome,
                "decision_uses_truth": False,
                "decision_uses_audit": False,
            }
            for view_offset, (view, candidate, fallback, improvement) in enumerate(
                zip(
                    block.outer_views,
                    pending["candidate_outer"],
                    pending["fallback_outer"],
                    pending["outer_improvements"],
                    strict=True,
                )
            ):
                row[f"outer_{view_offset}_camera_index"] = view
                row[f"outer_{view_offset}_candidate_rms"] = candidate
                row[f"outer_{view_offset}_fallback_rms"] = fallback
                row[f"outer_{view_offset}_improvement_percent"] = improvement
                row[
                    f"outer_{view_offset}_loo_block_median_improvement_percent"
                ] = leave_one_out_metrics[
                    "outer_view_median_improvements_percent"
                ][view_offset]
                row[
                    f"outer_{view_offset}_loo_block_positive_fraction"
                ] = leave_one_out_metrics[
                    "outer_view_positive_fractions"
                ][view_offset]
                row[
                    f"outer_{view_offset}_loo_block_worst_improvement_percent"
                ] = leave_one_out_metrics[
                    "outer_view_worst_improvements_percent"
                ][view_offset]

            for direction, record in enumerate(directions):
                row[
                    f"direction_{direction}_gate_camera_index"
                ] = block.outer_views[record["gate_offset"]]
                row[
                    f"direction_{direction}_evaluation_camera_index"
                ] = block.outer_views[record["evaluation_offset"]]
                row[f"direction_{direction}_block_gate"] = record["block_gate"]
                row[
                    f"direction_{direction}_block_gate_reasons"
                ] = "|".join(record["block_reasons"])
                row[f"direction_{direction}_sample_gate"] = record["sample_gate"]
                row[
                    f"direction_{direction}_sample_gate_reasons"
                ] = "|".join(record["sample_reasons"])
                row[f"direction_{direction}_accepted"] = record["accepted"]
                row[
                    f"direction_{direction}_heldout_improvement_percent"
                ] = record["heldout_improvement"]

            sample_rows.append(row)
            postdecision_samples.append(
                {
                    "row": row,
                    "block": block,
                    "truth": pending["truth"],
                    "observation": pending["observation"],
                    "sigma": pending["sigma"],
                    "candidate_fit": pending["candidate_fit"],
                    "fallback_fit": pending["fallback_fit"],
                    "candidate_operator": candidate_operator,
                    "fallback_operator": fallback_operator,
                }
            )

    routing_excluded_report_fields = {
        "block_id",
        "family",
        "true_aperture_radius",
    }
    routing_commit_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in routing_excluded_report_fields
        }
        for row in sample_rows
    ]
    decision_payload = json.dumps(
        routing_commit_rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    decision_sha256 = hashlib.sha256(decision_payload).hexdigest()

    # These loops are deliberately after every sample route has been fixed.
    for pending in postdecision_blocks:
        row = pending["row"]
        block = pending["block"]
        selected_index = int(pending["selected_index"])
        oracle_nearest_index = nearest_index(radii, block.true_radius)
        clean_oracle_scores, operator_distances = oracle_field_scores(
            block, clean=True
        )
        noisy_oracle_scores, _ = oracle_field_scores(block, clean=False)
        row.update(
            {
                "oracle_nearest_radius": float(radii[oracle_nearest_index]),
                "clean_truth_field_radius": float(
                    radii[int(np.argmin(clean_oracle_scores))]
                ),
                "noisy_truth_field_radius": float(
                    radii[int(np.argmin(noisy_oracle_scores))]
                ),
                "closest_operator_matrix_radius": float(
                    radii[int(np.argmin(operator_distances))]
                ),
                "nearest_bank_match": selected_index == oracle_nearest_index,
            }
        )

    for pending in postdecision_samples:
        row = pending["row"]
        candidate_fit = pending["candidate_fit"]
        fallback_fit = pending["fallback_fit"]
        candidate_l2 = relative_l2(candidate_fit.field, pending["truth"])
        fallback_l2 = relative_l2(fallback_fit.field, pending["truth"])
        raw_gain = _percent_improvement(candidate_l2, fallback_l2)
        candidate_audit = whitened_per_view_rms(
            pending["candidate_operator"],
            candidate_fit.field,
            pending["observation"],
            pending["sigma"],
            pending["block"].audit_views,
        )[0]
        fallback_audit = whitened_per_view_rms(
            pending["fallback_operator"],
            fallback_fit.field,
            pending["observation"],
            pending["sigma"],
            pending["block"].audit_views,
        )[0]
        raw_audit_change = safe_percent_change(
            candidate_audit, fallback_audit
        )
        accepted = bool(row["accepted"])
        row.update(
            {
                "fallback_relative_l2": fallback_l2,
                "candidate_relative_l2": candidate_l2,
                "selected_relative_l2": (
                    candidate_l2 if accepted else fallback_l2
                ),
                "raw_field_gain_percent": raw_gain,
                "selected_field_gain_percent": raw_gain if accepted else 0.0,
                "fallback_audit_rms": fallback_audit,
                "candidate_audit_rms": candidate_audit,
                "selected_audit_rms": (
                    candidate_audit if accepted else fallback_audit
                ),
                "raw_audit_change_percent": raw_audit_change,
                "selected_audit_change_percent": (
                    raw_audit_change if accepted else 0.0
                ),
            }
        )

    accepted = np.asarray(
        [bool(row["accepted"]) for row in sample_rows], dtype=bool
    )
    selected_gains = np.asarray(
        [float(row["selected_field_gain_percent"]) for row in sample_rows],
        dtype=float,
    )
    accepted_gains = selected_gains[accepted]
    selected_audit = np.asarray(
        [float(row["selected_audit_change_percent"]) for row in sample_rows],
        dtype=float,
    )
    raw_audit = np.asarray(
        [float(row["raw_audit_change_percent"]) for row in sample_rows],
        dtype=float,
    )
    min_outer = np.asarray(
        [float(row["minimum_outer_improvement_percent"]) for row in sample_rows],
        dtype=float,
    )
    nearest_match = float(
        np.mean([bool(row["nearest_bank_match"]) for row in block_rows])
    )

    safety = {
        "nominal_row_coverage": float(np.mean(accepted)),
        "accepted_count": int(np.sum(accepted)),
        "accepted_mean_field_gain_percent": (
            None if not np.any(accepted) else float(np.mean(accepted_gains))
        ),
        "accepted_p10_field_gain_percent": (
            None
            if not np.any(accepted)
            else float(np.percentile(accepted_gains, 10))
        ),
        "accepted_field_harm_rate_over_1_percent": (
            None
            if not np.any(accepted)
            else float(np.mean(accepted_gains < -1.0))
        ),
        "mean_selected_audit_change_percent": float(
            np.mean(selected_audit)
        ),
        "accepted_audit_increase_rate": (
            None
            if not np.any(accepted)
            else float(np.mean(selected_audit[accepted] > 0.0))
        ),
        "maximum_accepted_audit_increase_percent": (
            None
            if not np.any(accepted)
            else float(np.max(selected_audit[accepted]))
        ),
    }

    directional_transfer: dict[str, Any] = {}
    for direction in range(2):
        direction_accepted = np.asarray(
            [
                bool(row[f"direction_{direction}_accepted"])
                for row in sample_rows
            ],
            dtype=bool,
        )
        heldout = np.asarray(
            [
                float(
                    row[
                        f"direction_{direction}_heldout_improvement_percent"
                    ]
                )
                for row in sample_rows
            ],
            dtype=float,
        )
        accepted_heldout = heldout[direction_accepted]
        accepted_block_means: list[float] = []
        for block_id in sorted(
            {str(row["block_id"]) for row in sample_rows}
        ):
            block_mask = np.asarray(
                [
                    str(row["block_id"]) == block_id
                    for row in sample_rows
                ],
                dtype=bool,
            )
            selected = block_mask & direction_accepted
            if np.any(selected):
                accepted_block_means.append(
                    float(np.mean(heldout[selected]))
                )
        name = (
            f"outer_{direction}_gate_to_outer_{1 - direction}_evaluation"
        )
        directional_transfer[name] = {
            "nominal_row_coverage": float(np.mean(direction_accepted)),
            "accepted_count": int(np.sum(direction_accepted)),
            "accepted_heldout_mean_improvement_percent": (
                None
                if not np.any(direction_accepted)
                else float(np.mean(accepted_heldout))
            ),
            "accepted_heldout_p10_improvement_percent": (
                None
                if not np.any(direction_accepted)
                else float(np.percentile(accepted_heldout, 10))
            ),
            "accepted_heldout_harm_rate_below_minus_1_percent": (
                None
                if not np.any(direction_accepted)
                else float(np.mean(accepted_heldout < -1.0))
            ),
            "minimum_accepted_block_mean_improvement_percent": (
                None
                if not accepted_block_means
                else float(np.min(accepted_block_means))
            ),
        }

    evaluation = config["pilot_evaluation"]
    directional_gates: dict[str, bool] = {}
    for name, metrics in directional_transfer.items():
        directional_gates[name] = bool(
            float(metrics["nominal_row_coverage"])
            >= float(evaluation["minimum_coverage"])
            and metrics[
                "accepted_heldout_mean_improvement_percent"
            ]
            is not None
            and float(
                metrics["accepted_heldout_mean_improvement_percent"]
            )
            >= float(
                evaluation["minimum_accepted_mean_field_gain_percent"]
            )
            and metrics[
                "accepted_heldout_p10_improvement_percent"
            ]
            is not None
            and float(
                metrics["accepted_heldout_p10_improvement_percent"]
            )
            >= float(evaluation["minimum_accepted_p10_field_gain_percent"])
            and metrics[
                "accepted_heldout_harm_rate_below_minus_1_percent"
            ]
            is not None
            and float(
                metrics[
                    "accepted_heldout_harm_rate_below_minus_1_percent"
                ]
            )
            <= float(
                evaluation[
                    "maximum_accepted_field_harm_rate_over_1_percent"
                ]
            )
            and metrics[
                "minimum_accepted_block_mean_improvement_percent"
            ]
            is not None
            and float(
                metrics[
                    "minimum_accepted_block_mean_improvement_percent"
                ]
            )
            >= 0.0
        )

    gates = {
        "nearest_bank_match_rate": nearest_match
        >= float(evaluation["minimum_nearest_bank_match_rate"]),
        "coverage": safety["nominal_row_coverage"]
        >= float(evaluation["minimum_coverage"]),
        "accepted_mean_field_gain": safety[
            "accepted_mean_field_gain_percent"
        ]
        is not None
        and safety["accepted_mean_field_gain_percent"]
        >= float(evaluation["minimum_accepted_mean_field_gain_percent"]),
        "accepted_p10_field_gain": safety[
            "accepted_p10_field_gain_percent"
        ]
        is not None
        and safety["accepted_p10_field_gain_percent"]
        >= float(evaluation["minimum_accepted_p10_field_gain_percent"]),
        "accepted_field_harm": safety[
            "accepted_field_harm_rate_over_1_percent"
        ]
        is not None
        and safety["accepted_field_harm_rate_over_1_percent"]
        <= float(
            evaluation["maximum_accepted_field_harm_rate_over_1_percent"]
        ),
        "accepted_audit_increase": safety[
            "accepted_audit_increase_rate"
        ]
        is not None
        and safety["accepted_audit_increase_rate"]
        <= float(evaluation["maximum_accepted_audit_increase_rate"]),
        "mean_selected_audit_change": safety[
            "mean_selected_audit_change_percent"
        ]
        <= float(evaluation["maximum_mean_selected_audit_change_percent"]),
        **directional_gates,
    }

    summary = {
        "nominal_sample_rows": len(sample_rows),
        "paired_radius_blocks": len(block_rows),
        "field_rig_bundles": len(
            {
                (
                    str(row["rig_id"]),
                    int(row["sample_index_in_block"]),
                )
                for row in sample_rows
            }
        ),
        "independent_rigs": len(
            {str(row["rig_id"]) for row in sample_rows}
        ),
        "independence_boundary": {
            "radius_blocks_reuse_fields_and_paired_noise_within_rig": True,
            "views_are_repeated_measurements_not_independent_units": True,
            "nominal_rows_are_not_iid_samples": True,
        },
        "families": sorted(
            {str(row["family"]) for row in sample_rows}
        ),
        "support_audit": {
            "active_voxels": int(np.sum(support)),
            "total_voxels": int(support.size),
            "threshold": float(config["support_threshold"]),
        },
        "calibration": {
            "nearest_bank_match_rate": nearest_match,
            "mean_aperture_absolute_error": float(
                np.mean(
                    [
                        abs(
                            float(row["selected_radius"])
                            - float(row["true_aperture_radius"])
                        )
                        for row in block_rows
                    ]
                )
            ),
            "fixed_ridge_nearest_bank_match_rate": float(
                np.mean(
                    [
                        float(row["fixed_ridge_profile_radius"])
                        == float(row["oracle_nearest_radius"])
                        for row in block_rows
                    ]
                )
            ),
            "clean_truth_field_match_rate": float(
                np.mean(
                    [
                        float(row["clean_truth_field_radius"])
                        == float(row["oracle_nearest_radius"])
                        for row in block_rows
                    ]
                )
            ),
            "noisy_truth_field_match_rate": float(
                np.mean(
                    [
                        float(row["noisy_truth_field_radius"])
                        == float(row["oracle_nearest_radius"])
                        for row in block_rows
                    ]
                )
            ),
            "operator_matrix_match_rate": float(
                np.mean(
                    [
                        float(row["closest_operator_matrix_radius"])
                        == float(row["oracle_nearest_radius"])
                        for row in block_rows
                    ]
                )
            ),
        },
        "selection": {
            "mean_camera_deletion_radius_stability_fraction": float(
                np.mean(
                    [
                        float(
                            row[
                                "camera_deletion_radius_stability_fraction"
                            ]
                        )
                        for row in block_rows
                    ]
                )
            ),
            "mean_camera_deletion_kappa_stability_fraction": float(
                np.mean(
                    [
                        float(
                            row[
                                "camera_deletion_kappa_stability_fraction"
                            ]
                        )
                        for row in block_rows
                    ]
                )
            ),
            "full_block_descriptive_gate_pass_rate": float(
                np.mean(
                    [bool(row["block_gate"]) for row in block_rows]
                )
            ),
            "sample_excluded_block_gate_pass_rate": float(
                np.mean(
                    [
                        bool(row["block_gate_minus_sample"])
                        for row in sample_rows
                    ]
                )
            ),
            "selected_kappas": [
                float(row["selected_kappa"]) for row in block_rows
            ],
            "asymmetric_candidate_cv_advantage_is_gate": False,
        },
        "decision_construction": {
            "in_memory_routing_sha256_before_truth_field_and_audit_evaluation": (
                decision_sha256
            ),
            "routing_commit_excluded_report_fields": sorted(
                routing_excluded_report_fields
            ),
            "all_decisions_constructed_before_truth_and_audit_evaluation": (
                True
            ),
            "same_process_not_cryptographic_escrow": True,
        },
        "two_way_outer_transfer": {
            "directions": directional_transfer,
            "fixed_two_views_not_iid_and_not_a_risk_bound": True,
            "both_directions_required_for_final_acceptance": True,
        },
        "selective_safety": safety,
        "outer_audit_diagnostic": {
            "pearson_min_outer_improvement_vs_raw_audit_change": (
                _safe_correlation(min_outer, raw_audit)
            ),
            "uses_audit_for_sample_routing": False,
            "audit_used_for_post_open_scientific_verdict": True,
            "descriptive_only_not_a_calibrated_risk_bound": True,
            "two_outer_views_cannot_support_arbitrary_view_safety": True,
        },
        "pilot_gates": gates,
        "overall_pilot_pass": all(gates.values()),
    }
    return sample_rows, block_rows, candidate_rows, summary

def write_figure(
    path: Path,
    sample_rows: Sequence[dict[str, Any]],
    block_rows: Sequence[dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), constrained_layout=True)
    for gate, marker, color, label in (
        (True, "o", "tab:green", "block gate passed"),
        (False, "x", "tab:red", "block gate failed"),
    ):
        rows = [row for row in block_rows if bool(row["block_gate"]) == gate]
        axes[0, 0].scatter(
            [row["true_aperture_radius"] for row in rows],
            [row["selected_radius"] for row in rows],
            marker=marker,
            color=color,
            s=70,
            label=label,
        )
    axes[0, 0].plot([0.04, 0.15], [0.04, 0.15], "k--", lw=1)
    axes[0, 0].set(
        title="Nested cross-view aperture selection",
        xlabel="true radius (report only)",
        ylabel="selected radius",
    )
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].scatter(
        [
            row["candidate_cv_advantage_percent_asymmetric_diagnostic"]
            for row in block_rows
        ],
        [
            row["camera_deletion_radius_stability_fraction"]
            for row in block_rows
        ],
        c=[row["true_aperture_radius"] for row in block_rows],
        cmap="viridis",
        s=70,
    )
    axes[0, 1].set(
        title="Inner-only selection diagnostics",
        xlabel="asymmetric CV diagnostic vs metadata (%)",
        ylabel="true camera-deletion radius stability",
    )

    accepted = np.asarray([bool(row["accepted"]) for row in sample_rows], dtype=bool)
    outer = np.asarray(
        [float(row["minimum_outer_improvement_percent"]) for row in sample_rows]
    )
    audit = np.asarray([float(row["raw_audit_change_percent"]) for row in sample_rows])
    axes[1, 0].scatter(outer[~accepted], audit[~accepted], color="tab:gray", alpha=0.55, label="rejected")
    axes[1, 0].scatter(outer[accepted], audit[accepted], color="tab:purple", alpha=0.85, label="accepted")
    axes[1, 0].axhline(0.0, color="black", lw=1)
    axes[1, 0].axvline(0.0, color="black", lw=1)
    axes[1, 0].set(
        title="Does the worst outer camera predict audit?",
        xlabel="minimum outer improvement (%)",
        ylabel="raw audit RMS change (%)",
    )
    axes[1, 0].legend(fontsize=8)

    gains = np.asarray([float(row["selected_field_gain_percent"]) for row in sample_rows])
    selected_audit = np.asarray(
        [float(row["selected_audit_change_percent"]) for row in sample_rows]
    )
    axes[1, 1].scatter(
        gains[~accepted], selected_audit[~accepted], color="tab:gray", alpha=0.55
    )
    axes[1, 1].scatter(
        gains[accepted], selected_audit[accepted], color="tab:purple", alpha=0.85
    )
    axes[1, 1].axhline(0.0, color="black", lw=1)
    axes[1, 1].axvline(0.0, color="black", lw=1)
    axes[1, 1].set(
        title="Selected field gain and held-out audit camera",
        xlabel="selected field L2 gain (%)",
        ylabel="selected audit RMS change (%)",
    )
    figure.suptitle(
        "v5c nested cross-view new-family first open - development only",
        fontsize=14,
    )
    figure.savefig(path, dpi=190)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    output = args.output_dir.resolve()
    config = read_json(config_path)
    if output.exists():
        raise FileExistsError(f"first-open output already exists: {output}")
    gate_relative = Path(str(config["forward_model_gate_report"]))
    gate_report_path = (
        gate_relative if gate_relative.is_absolute() else ROOT / gate_relative
    ).resolve()
    dependencies = [
        Path(__file__).resolve(),
        (ROOT / "nested_crossview.py").resolve(),
        (ROOT / "rig_shared_profile.py").resolve(),
        (ROOT / "run_v5b_rig_shared_profile_pilot.py").resolve(),
        (ROOT / "finite_aperture_bost.py").resolve(),
        (ROOT / "independent_reaction_bost.py").resolve(),
        (ROOT / "v5c_nested_crossview_protocol.md").resolve(),
        gate_report_path,
        config_path,
    ]
    preopen_commit = preopen_git_commit(dependencies)
    forward_model_gate = validate_forward_model_gate(config)
    blocks, manifest_rows = build_development_blocks(config)
    sample_rows, block_rows, candidate_rows, summary = evaluate(config, blocks)

    output.mkdir(parents=True, exist_ok=False)
    config_snapshot = output / "config_snapshot.json"
    config_snapshot.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest_path = output / "paired_factorial_manifest.csv"
    sample_path = output / "sample_metrics.csv"
    block_path = output / "block_metrics.csv"
    candidates_path = output / "crossview_candidates.csv"
    write_csv(manifest_path, manifest_rows)
    write_csv(sample_path, sample_rows)
    write_csv(block_path, block_rows)
    write_csv(candidates_path, candidate_rows)
    figure_path = output / "v5c_nested_crossview_first_open.png"
    write_figure(figure_path, sample_rows, block_rows)

    report = {
        "claim_status": config["claim_status"],
        "claim_boundary": (
            "precommitted new-generator-family first open on an 8x8x5 prescribed "
            "linear weak-deflection surrogate; same two Stage-0-audited rigs and new "
            "jet_shear/shock_cell fields, seeds and noise levels; the new families "
            "participate in current inner selection, so only cameras are held out; "
            "no separate audit process, "
            "real BOS, nonlinear ray tracing, CFD, NeRIF/TDBOST, FNO/FFNO or DeepONet"
        ),
        "scientific_question": (
            "Can inner-only cross-view selection of aperture and dimensionless ridge, "
            "followed by sample-excluded per-camera and two-way outer transfer gates, "
            "retain aperture recovery while reducing the held-out audit-camera tail "
            "failure seen in v5b M1?"
        ),
        "preopen_git_commit": preopen_commit,
        "data_separation": {
            "inner_views_select_radius_and_kappa": True,
            "outer_views_used_only_after_inner_selection": True,
            "outer_cameras_scored_separately": True,
            "audit_not_used_for_sample_routing": True,
            "audit_used_for_post_open_scientific_verdict": True,
            "all_sample_routes_constructed_before_truth_and_audit_evaluation": True,
            "audit_view_opened_in_same_process_after_decisions": True,
            "audit_escrowed_in_separate_process": False,
            "direct_truth_fields_and_true_radius_used_for_routing": False,
            "family_labels_used_for_routing": False,
            "metadata_radius_is_truth_derived_synthetic_proxy": True,
            "new_generator_families_relative_to_v5b_m1": [
                "jet_shear",
                "shock_cell",
            ],
            "family_holdout_against_current_inner_selection": False,
            "camera_holdout_against_current_inner_selection": True,
            "noise_sigma_scaled_from_clean_inner_truth": True,
            "noise_whitening_is_oracle_nominal_not_experimental_calibration": True,
            "support_mask_uses_generator_matched_reaction_support": True,
            "same_rigs_as_stage0_and_v5b_m1": True,
            "confirmatory_lock_constructed": False,
        },
        "forward_model_gate": forward_model_gate,
        "summary": summary,
        "source_hashes": {
            "runner": sha256(Path(__file__).resolve()),
            "nested_crossview": sha256((ROOT / "nested_crossview.py").resolve()),
            "profile_module": sha256((ROOT / "rig_shared_profile.py").resolve()),
            "data_builder": sha256(
                (ROOT / "run_v5b_rig_shared_profile_pilot.py").resolve()
            ),
            "finite_aperture_operator": sha256(
                (ROOT / "finite_aperture_bost.py").resolve()
            ),
            "reaction_generator": sha256(
                (ROOT / "independent_reaction_bost.py").resolve()
            ),
            "frozen_protocol": sha256(
                (ROOT / "v5c_nested_crossview_protocol.md").resolve()
            ),
            "config": sha256(config_path),
            "forward_model_gate_report": forward_model_gate["report_sha256"],
        },
    }
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    targets = [
        config_snapshot,
        manifest_path,
        sample_path,
        block_path,
        candidates_path,
        figure_path,
        report_path,
    ]
    (output / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in targets),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
