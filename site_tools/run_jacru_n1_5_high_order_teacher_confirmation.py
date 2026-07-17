#!/usr/bin/env python3
"""Open the frozen N1.5 high-order-teacher synthetic confirmation once."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from site_tools import (  # noqa: E402
    run_jacru_n1_5_approximation_error_headroom as n15a,
)
from site_tools import (  # noqa: E402
    run_jacru_n1_5_reconstruction_aware_postopen as n15b,
)


DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator/configs/"
    "jacru_n1_5_high_order_teacher_confirmation_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "demo_t16_operator/results/"
    "jacru_n1_5_high_order_teacher_confirmation_once"
)
REPORT_SCHEMA = "jacru-n1-5-high-order-teacher-confirmation-report-1.0"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed-limit", type=int)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_checksums(output: Path) -> None:
    files = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "checksums.sha256"
    )
    (output / "checksums.sha256").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )


def _derive_confirmation_seeds(config: Mapping[str, Any]) -> list[int]:
    derivation = config["confirmation"]["seed_derivation"]
    if derivation["hash"] != "sha256":
        raise ValueError("only the frozen SHA-256 seed derivation is accepted")
    output: list[int] = []
    for index in derivation["indices"]:
        token = str(derivation["token_pattern"]).format(index=int(index)).encode("ascii")
        digest = hashlib.sha256(token).digest()
        value = 2200 + int.from_bytes(digest[:4], "big") % 700
        while value in output:
            value += 1
        output.append(value)
    return output


def _source_manifest(config_path: Path, config: Mapping[str, Any]) -> dict[str, str]:
    paths = [
        config_path,
        ROOT / str(config["source_t0_config"]),
        ROOT / str(config["source_n1_5_a_config"]),
        ROOT / str(config["source_n1_5_a_results"]) / "summary.json",
        ROOT / str(config["source_postopen_config"]),
        ROOT / str(config["source_postopen_results"]) / "summary.json",
        ROOT / "demo_t16_operator/jacru_n1_5_high_order_correction.py",
        ROOT / "site_tools/run_jacru_n1_5_reconstruction_aware_postopen.py",
        Path(__file__).resolve(),
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"source manifest is incomplete: {missing}")
    return {str(path.relative_to(ROOT)): _sha256(path) for path in paths}


def _validate_config(
    config: Mapping[str, Any],
    source_t0: Mapping[str, Any],
    postopen_summary: Mapping[str, Any],
    seed_limit: int | None,
) -> None:
    if config.get("status") != "FROZEN_BEFORE_FIRST_CONFIRMATION_EXECUTION":
        raise RuntimeError("confirmation config is not frozen")
    if config.get("may_construct_or_evaluate_ood") is not False:
        raise RuntimeError("OOD construction must remain disabled")
    if seed_limit is not None and seed_limit < 1:
        raise ValueError("seed-limit must be positive")
    derived = _derive_confirmation_seeds(config)
    declared = [int(value) for value in config["confirmation"]["base_seeds"]]
    if declared != derived or len(set(declared)) != len(declared):
        raise ValueError("declared confirmation seeds do not match frozen derivation")
    prior = {
        int(value)
        for split in source_t0["splits"].values()
        for value in split["base_seeds"]
    }
    if prior & set(declared):
        raise ValueError("confirmation seeds overlap a prior T0 split")
    selection = postopen_summary["future_confirmation_hypothesis"]
    candidate = config["selected_candidate"]
    if candidate["candidate_id"] != selection["candidate_id"] or float(
        candidate["beta"]
    ) != float(selection["beta"]):
        raise ValueError("selected confirmation candidate drifted from post-open freeze")
    if candidate.get("may_be_changed_after_confirmation_open") is not False:
        raise ValueError("candidate must be immutable after confirmation open")
    budget = config["budget"]
    if int(budget["warm_cgls_iterations"]) + int(
        budget["visible_low_projection_forward_calls"]
    ) + int(budget["corrected_warm_cgls_iterations"]) != int(
        budget["corrected_total_low_forward_calls"]
    ):
        raise ValueError("confirmation forward-call budget drifted")
    if int(budget["warm_cgls_iterations"]) + int(
        budget["corrected_warm_cgls_iterations"]
    ) != int(budget["corrected_total_low_adjoint_calls"]):
        raise ValueError("confirmation adjoint-call budget drifted")
    if config["claim_boundary"].get("may_claim_general_algorithm_superiority") is not False:
        raise ValueError("synthetic confirmation cannot claim general superiority")


def _execution_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "evaluated_partitions": ["development"],
        "budget": dict(config["budget"]),
        "correction_candidates": {
            "include_component_damping": True,
            "include_visible_curvature_ridge": False,
            "include_damping_curvature_half_interpolation": False,
            "damping_to_high_order_betas": [float(config["selected_candidate"]["beta"])],
            "include_exact_mismatch_oracle": True,
        },
    }


def _cluster_gains(
    rows: list[dict[str, Any]], candidate_id: str
) -> list[dict[str, Any]]:
    baseline = {
        str(row["case_id"]): row for row in rows if row["candidate_id"] == "low_cgls25"
    }
    damping = {
        str(row["case_id"]): row
        for row in rows if row["candidate_id"] == "component_damping_correction"
    }
    candidate = [row for row in rows if row["candidate_id"] == candidate_id]
    output: list[dict[str, Any]] = []
    for seed in sorted({int(row["base_seed"]) for row in candidate}):
        selected = [row for row in candidate if int(row["base_seed"]) == seed]
        field_gains = []
        h1_gains = []
        damping_gains = []
        for row in selected:
            base = baseline[str(row["case_id"])]
            simple = damping[str(row["case_id"])]
            field_gains.append(
                1.0 - float(row["field_relative_l2"]) / float(base["field_relative_l2"])
            )
            h1_gains.append(
                1.0
                - float(row["h1_seminorm_relative_error"])
                / float(base["h1_seminorm_relative_error"])
            )
            damping_gains.append(
                1.0 - float(row["field_relative_l2"]) / float(simple["field_relative_l2"])
            )
        output.append(
            {
                "base_seed": seed,
                "geometry_digest": selected[0]["geometry_digest"],
                "family_count": len(selected),
                "mean_field_gain_over_low_cgls25": float(np.mean(field_gains)),
                "mean_h1_gain_over_low_cgls25": float(np.mean(h1_gains)),
                "mean_field_gain_over_component_damping": float(np.mean(damping_gains)),
                "worst_family_field_gain_over_low_cgls25": float(min(field_gains)),
            }
        )
    return output


def _bootstrap_interval(values: list[float], config: Mapping[str, Any]) -> dict[str, float]:
    policy = config["bootstrap"]
    generator = np.random.default_rng(int(policy["seed"]))
    samples = np.asarray(values, dtype=np.float64)
    replicates = int(policy["replicates"])
    means = np.mean(
        samples[generator.integers(0, len(samples), size=(replicates, len(samples)))],
        axis=1,
    )
    alpha = 1.0 - float(policy["confidence_level"])
    return {
        "lower": float(np.quantile(means, alpha / 2.0)),
        "upper": float(np.quantile(means, 1.0 - alpha / 2.0)),
        "participates_in_gate": False,
    }


def _gate(
    rows: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    candidate_id: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = config["confirmation_gates"]
    baseline = {
        str(row["case_id"]): row for row in rows if row["candidate_id"] == "low_cgls25"
    }
    damping = {
        str(row["case_id"]): row
        for row in rows if row["candidate_id"] == "component_damping_correction"
    }
    candidate = [row for row in rows if row["candidate_id"] == candidate_id]
    field_gains = [
        1.0 - float(row["field_relative_l2"]) / float(baseline[str(row["case_id"])]["field_relative_l2"])
        for row in candidate
    ]
    h1_gains = [
        1.0
        - float(row["h1_seminorm_relative_error"])
        / float(baseline[str(row["case_id"])]["h1_seminorm_relative_error"])
        for row in candidate
    ]
    damping_gains = [
        1.0
        - float(row["field_relative_l2"])
        / float(damping[str(row["case_id"])]["field_relative_l2"])
        for row in candidate
    ]
    family_means = {
        family: float(np.mean([
            field_gains[index]
            for index, row in enumerate(candidate)
            if row["family"] == family
        ]))
        for family in config["confirmation"]["families"]
    }
    checks = {
        "mean_field": float(np.mean(field_gains))
        >= float(gates["mean_field_gain_over_low_cgls25_minimum"]),
        "mean_h1": float(np.mean(h1_gains))
        >= float(gates["mean_h1_gain_over_low_cgls25_minimum"]),
        "worst_geometry": min(row["mean_field_gain_over_low_cgls25"] for row in clusters)
        >= float(gates["worst_geometry_cluster_field_gain_minimum"]),
        "case_harm": float(np.mean(np.asarray(field_gains) < -0.01))
        <= float(gates["case_field_harm_over_one_percent_rate_maximum"]),
        "beats_damping": float(np.mean(damping_gains))
        >= float(gates["mean_field_gain_over_component_damping_minimum"]),
        "each_family": min(family_means.values())
        >= float(gates["each_family_mean_field_gain_over_low_cgls25_minimum"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "mean_field_gain_over_low_cgls25": float(np.mean(field_gains)),
        "mean_h1_gain_over_low_cgls25": float(np.mean(h1_gains)),
        "worst_geometry_cluster_field_gain": float(
            min(row["mean_field_gain_over_low_cgls25"] for row in clusters)
        ),
        "worst_case_field_gain": float(min(field_gains)),
        "case_field_harm_over_one_percent_rate": float(np.mean(np.asarray(field_gains) < -0.01)),
        "mean_field_gain_over_component_damping": float(np.mean(damping_gains)),
        "family_mean_field_gains": family_means,
        "geometry_cluster_bootstrap_95_interval": _bootstrap_interval(
            [row["mean_field_gain_over_low_cgls25"] for row in clusters], config
        ),
    }


def _plot(rows: list[dict[str, Any]], clusters: list[dict[str, Any]], candidate_id: str, output: Path) -> None:
    aggregates = n15b._aggregate(rows, {"evaluated_partitions": ["development"]})
    display = [row for row in aggregates if not row["evaluator_only"]]
    display = sorted(display, key=lambda row: float(row["mean_field_gain_over_low_cgls25"]), reverse=True)
    labels = [str(row["candidate_id"]) for row in display]
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), constrained_layout=True)
    axes[0].barh(y, [row["mean_field_gain_over_low_cgls25"] for row in display], color="#1d6f78")
    axes[0].axvline(0.05, color="#202020", linestyle="--", linewidth=1)
    axes[0].axvline(0.0, color="#202020", linewidth=1)
    axes[0].set_yticks(y, labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("mean field gain vs low CGLS-25")
    axes[1].bar(
        [str(row["base_seed"]) for row in clusters],
        [row["mean_field_gain_over_low_cgls25"] for row in clusters],
        color="#517a3a",
    )
    axes[1].axhline(0.0, color="#202020", linewidth=1)
    axes[1].set_xlabel("confirmation geometry seed")
    axes[1].set_ylabel("cluster mean field gain")
    selected = [row for row in rows if row["candidate_id"] == candidate_id]
    family_names = sorted({str(row["family"]) for row in selected})
    base = {str(row["case_id"]): row for row in rows if row["candidate_id"] == "low_cgls25"}
    axes[2].bar(
        family_names,
        [
            np.mean([
                1.0 - float(row["field_relative_l2"]) / float(base[str(row["case_id"])]["field_relative_l2"])
                for row in selected if row["family"] == family
            ])
            for family in family_names
        ],
        color=("#c05640", "#8a5a20"),
    )
    axes[2].axhline(0.0, color="#202020", linewidth=1)
    axes[2].set_ylabel("family mean field gain")
    fig.suptitle("N1.5 frozen high-order-teacher synthetic confirmation", fontsize=15)
    fig.savefig(output / "diagnostic.png", dpi=180)
    fig.savefig(output / "diagnostic.pdf")
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    config = _read_json(config_path)
    source_t0 = _read_json(ROOT / str(config["source_t0_config"]))
    postopen_results = ROOT / str(config["source_postopen_results"])
    postopen_summary = _read_json(postopen_results / "summary.json")
    _validate_config(config, source_t0, postopen_summary, args.seed_limit)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(
            f"confirmation output already exists and is immutable: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    started = time.perf_counter()
    n15a_config = _read_json(ROOT / str(config["source_n1_5_a_config"]))
    confirmation_config = json.loads(json.dumps(n15a_config))
    seeds = [int(value) for value in config["confirmation"]["base_seeds"]]
    if args.seed_limit is not None:
        seeds = seeds[: args.seed_limit]
    confirmation_config["development"]["base_seeds"] = seeds
    records, manifest = n15a._prepare_records(
        confirmation_config, source_t0, seed_limit=None
    )
    fixed = n15a._fixed_predictors(records)
    curvature_model = n15b._load_curvature_model(
        ROOT / str(config["source_n1_5_a_results"]) / "selected_ridge_models.json"
    )
    execution = _execution_config(config)
    rows = n15b._run_cases(records, fixed, curvature_model, execution)
    candidate_id = str(config["selected_candidate"]["candidate_id"])
    clusters = _cluster_gains(rows, candidate_id)
    gate = _gate(rows, clusters, candidate_id, config)
    status = (
        "SYNTHETIC_CONFIRMATION_GATE_PASS_LIMITED_SCOPE"
        if gate["passed"]
        else "SYNTHETIC_CONFIRMATION_NO_GO"
    )
    _write_csv(output / "case_metrics.csv", rows)
    _write_csv(output / "geometry_cluster_metrics.csv", clusters)
    _write_csv(output / "case_manifest.csv", manifest)
    summary = {
        "schema": REPORT_SCHEMA,
        "status": status,
        "candidate_id": candidate_id,
        "candidate_beta": float(config["selected_candidate"]["beta"]),
        "confirmation_geometry_cluster_count": len(clusters),
        "confirmation_case_count": len([
            row for row in rows if row["candidate_id"] == candidate_id
        ]),
        "independent_unit": "base_seed_geometry_cluster",
        "confirmation_gate": gate,
        "candidate_was_frozen_before_open": True,
        "candidate_changed_after_open": False,
        "opens_ood_or_final": False,
        "seed_limit": args.seed_limit,
        "runtime_seconds": time.perf_counter() - started,
        "claim_boundary": config["claim_boundary"],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    provenance = {
        "schema": "jacru-n1-5-high-order-teacher-confirmation-provenance-1.0",
        "git_commit_at_start": _git_commit(),
        "source_sha256": _source_manifest(config_path, config),
        "config": config,
        "exact_cli": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    _plot(rows, clusters, candidate_id, output)
    readme = f"""# N1.5 high-order-teacher synthetic confirmation\n\nStatus: **{status}**.\n\nThe candidate `{candidate_id}` (beta={config['selected_candidate']['beta']}) and all gates were frozen before these six new development-geometry seeds were constructed. The candidate was not changed after opening.\n\n- Mean field gain vs low CGLS-25: {gate['mean_field_gain_over_low_cgls25']:.6f}.\n- Mean H1 gain: {gate['mean_h1_gain_over_low_cgls25']:.6f}.\n- Worst geometry-cluster field gain: {gate['worst_geometry_cluster_field_gain']:.6f}.\n- Mean field gain over component damping: {gate['mean_field_gain_over_component_damping']:.6f}.\n- Geometry-cluster bootstrap interval is descriptive and does not participate in the gate.\n- This confirms or rejects only a small synthetic continuous-renderer/voxel-discretization hypothesis. It is not real BOST, finite-aperture, ray-bending, calibration-drift, OOD, publication, or general-superiority evidence.\n"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    _write_checksums(output)
    print(json.dumps({
        "status": status,
        "passed": gate["passed"],
        "mean_field_gain": gate["mean_field_gain_over_low_cgls25"],
        "mean_h1_gain": gate["mean_h1_gain_over_low_cgls25"],
        "output": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
