#!/usr/bin/env python3
"""Render the opened PSU active-view support-envelope diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIGURE_SCHEMA = "psu-b0-support-envelope-postopen-figure-1.0"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("summary must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_svg(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(
        "\n".join(line.rstrip() for line in lines) + "\n",
        encoding="utf-8",
    )


def render_figure(
    summary: dict[str, Any],
    output_stem: Path,
) -> dict[str, Any]:
    if summary["status"] != "POSTOPEN_SUPPORT_ENVELOPE_DIAGNOSIS_COMPLETE_NOT_FRESH":
        raise ValueError("this figure expects the frozen post-open diagnosis")
    rows = {
        (str(row["split"]), str(row["method"])): row
        for row in summary["aggregates"]
    }
    seeds = sorted(
        {
            int(method.rsplit("_", 1)[1])
            for _, method in rows
            if method.startswith("raw_seed_")
        }
    )
    splits = (
        "test_iid",
        "test_family_ood",
        "test_noise_ood",
        "test_geometry_ood",
        "test_joint_ood",
        "test_exact_operator_control",
    )
    labels = (
        "IID",
        "Family\nOOD",
        "Noise\nOOD",
        "View\nOOD",
        "Joint\nOOD",
        "Exact-op\ncontrol",
    )
    coverage_by_split = {
        row["split"]: float(row["support_envelope_coverage"])
        for row in summary["dataset"]["split_support_coverage"]
    }
    x = np.arange(len(splits))

    def values(prefix: str, key: str) -> np.ndarray:
        return np.asarray(
            [
                np.mean(
                    [
                        float(rows[(split, f"{prefix}_{seed}")][key])
                        for seed in seeds
                    ]
                )
                for split in splits
            ]
        )

    raw_gain = values("raw_seed", "field_gain_vs_sobolev_mean_percent")
    envelope_gain = values(
        "enveloped_seed",
        "field_gain_vs_sobolev_mean_percent",
    )
    raw_p10 = values("raw_seed", "field_gain_vs_sobolev_p10_percent")
    envelope_p10 = values(
        "enveloped_seed",
        "field_gain_vs_sobolev_p10_percent",
    )
    raw_harm = 100.0 * values(
        "raw_seed",
        "field_harm_over_one_percent_rate",
    )
    envelope_harm = 100.0 * values(
        "enveloped_seed",
        "field_harm_over_one_percent_rate",
    )
    coverage = 100.0 * np.asarray(
        [coverage_by_split[split] for split in splits]
    )

    figure, axes = plt.subplots(2, 2, figsize=(12.4, 8.4))
    figure.subplots_adjust(
        left=0.075,
        right=0.97,
        bottom=0.13,
        top=0.89,
        hspace=0.42,
        wspace=0.26,
    )
    width = 0.36
    raw_color = "#8d99ae"
    envelope_color = "#00798c"
    risk_color = "#d1495b"

    axes[0, 0].bar(
        x - width / 2,
        raw_gain,
        width,
        label="Raw learned",
        color=raw_color,
    )
    axes[0, 0].bar(
        x + width / 2,
        envelope_gain,
        width,
        label="View envelope",
        color=envelope_color,
    )
    axes[0, 0].axhline(0.0, color="#272727", linewidth=1)
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_ylabel("Mean field gain vs Sobolev (%)")
    axes[0, 0].set_title("A  View OOD harm is neutralized, not improved", loc="left")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(axis="y", alpha=0.25)

    axes[0, 1].bar(
        x - width / 2,
        raw_p10,
        width,
        label="Raw learned",
        color=raw_color,
    )
    axes[0, 1].bar(
        x + width / 2,
        envelope_p10,
        width,
        label="View envelope",
        color=envelope_color,
    )
    axes[0, 1].axhline(0.0, color="#272727", linewidth=1)
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].set_ylabel("p10 paired field gain (%)")
    axes[0, 1].set_title("B  Family tails remain inside the view support", loc="left")
    axes[0, 1].grid(axis="y", alpha=0.25)

    axes[1, 0].bar(
        x - width / 2,
        raw_harm,
        width,
        label="Raw learned",
        color="#ef8354",
    )
    axes[1, 0].bar(
        x + width / 2,
        envelope_harm,
        width,
        label="View envelope",
        color=risk_color,
    )
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set_ylabel(">1% field harm rate (%)")
    axes[1, 0].set_title("C  Joint harm falls to zero; family harm does not", loc="left")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(axis="y", alpha=0.25)

    axes[1, 1].bar(x, coverage, color="#3066be", width=0.62)
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].set_ylim(0.0, 108.0)
    axes[1, 1].set_ylabel("Learned correction coverage (%)")
    axes[1, 1].set_title("D  Safety comes from zero coverage outside 6–9 views", loc="left")
    axes[1, 1].grid(axis="y", alpha=0.25)
    axes[1, 1].annotate(
        "Not a positive OOD gain",
        xy=(4, 0),
        xytext=(3.15, 45),
        fontsize=9,
        color=risk_color,
        arrowprops={"arrowstyle": "->", "color": risk_color},
    )

    figure.suptitle(
        "PSU view-support envelope: exact fallback fixes one failure mode only",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.075,
        0.035,
        (
            "Post-open mechanism diagnosis on previously opened analytic fields "
            "with real PSU support geometry; 3 seeds, 4F + 4A^T.\n"
            "No fresh candidate gate, experimental 3D truth, rotation-40, final "
            "audit, or superiority claim."
        ),
        fontsize=8.5,
        color="#4f5d66",
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for suffix, options in (
        (".png", {"dpi": 220}),
        (".pdf", {}),
        (".svg", {}),
    ):
        path = output_stem.with_suffix(suffix)
        figure.savefig(path, facecolor="white", **options)
        if suffix == ".svg":
            _normalize_svg(path)
        outputs[path.name] = {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    plt.close(figure)
    return {
        "schema_version": FIGURE_SCHEMA,
        "status": "POSTOPEN_SUPPORT_ENVELOPE_FIGURE_COMPLETE",
        "source_status": summary["status"],
        "output_files": outputs,
        "claim_boundary": {
            "fresh_candidate_gate": False,
            "view_ood_positive_gain": False,
            "family_risk_resolved": False,
            "algorithm_superiority": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    manifest = render_figure(
        _load_json(args.summary),
        args.output_stem,
    )
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
