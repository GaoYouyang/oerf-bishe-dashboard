#!/usr/bin/env python3
"""Plot the frozen PSU residual-risk fresh audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SPLIT_LABELS = {
    "fresh_iid_support": "Support IID",
    "fresh_family_ood": "Held-out\nmorphology",
    "fresh_correlated_noise_ood": "Strong camera\ncorrelation",
    "fresh_family_noise_ood": "Morphology +\nnoise",
    "fresh_exact_operator_control": "Exact-operator\ncontrol",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("summary must be a JSON object")
    return value


def _method_rows(summary: dict, prefix: str) -> list[dict]:
    return [
        row
        for row in summary["aggregates"]
        if str(row["method"]).startswith(prefix)
        and row["split"] in SPLIT_LABELS
    ]


def plot(
    *,
    summary_path: Path,
    output_prefix: Path,
) -> dict:
    summary = _load(summary_path)
    raw = _method_rows(summary, "raw_seed_")
    gated = _method_rows(summary, "gated_seed_")
    if len(raw) != 15 or len(gated) != 15:
        raise ValueError("expected three seeds across five support splits")
    order = list(SPLIT_LABELS)

    def matrix(rows: list[dict], key: str) -> np.ndarray:
        by_key = {
            (row["split"], row["method"].rsplit("_", 1)[-1]): float(row[key])
            for row in rows
        }
        seeds = sorted({method for _, method in by_key})
        return np.asarray(
            [[by_key[(split, seed)] for seed in seeds] for split in order],
            dtype=np.float64,
        )

    raw_harm = matrix(raw, "field_harm_over_one_percent_rate") * 100.0
    gated_harm = matrix(gated, "field_harm_over_one_percent_rate") * 100.0
    gated_gain = matrix(gated, "field_gain_vs_sobolev_mean_percent")
    gated_coverage = matrix(gated, "candidate_coverage") * 100.0
    raw_p10 = matrix(raw, "field_gain_vs_sobolev_p10_percent")
    gated_p10 = matrix(gated, "field_gain_vs_sobolev_p10_percent")

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.facecolor": "#f7f8f5",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#c9d0cc",
            "axes.grid": True,
            "grid.color": "#e7ebe8",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.8))
    x = np.arange(len(order))
    width = 0.34

    axis = axes[0, 0]
    axis.bar(
        x - width / 2,
        raw_harm.mean(axis=1),
        width,
        color="#d65f4b",
        label="raw learned",
    )
    axis.bar(
        x + width / 2,
        gated_harm.mean(axis=1),
        width,
        color="#1f8a70",
        label="risk-gated",
    )
    axis.errorbar(
        x - width / 2,
        raw_harm.mean(axis=1),
        yerr=raw_harm.std(axis=1),
        fmt="none",
        ecolor="#6f3128",
        capsize=3,
    )
    axis.errorbar(
        x + width / 2,
        gated_harm.mean(axis=1),
        yerr=gated_harm.std(axis=1),
        fmt="none",
        ecolor="#145c4b",
        capsize=3,
    )
    axis.axhline(5.0, color="#8f6d20", linestyle="--", linewidth=1.2)
    axis.set_ylabel("Harm >1% rate (%)")
    axis.set_title("A  Selective gate suppresses harmful reconstructions")
    axis.legend(frameon=False, loc="upper right")

    axis = axes[0, 1]
    colors = ["#226f9b", "#3a9d78", "#d28b32", "#a55f8a", "#626c78"]
    for index, split in enumerate(order):
        axis.scatter(
            gated_coverage[index],
            gated_gain[index],
            s=75,
            color=colors[index],
            edgecolor="white",
            linewidth=0.8,
            label=SPLIT_LABELS[split].replace("\n", " "),
        )
    axis.axhline(0.0, color="#5d6460", linewidth=1.0)
    axis.set_xlabel("Candidate coverage (%)")
    axis.set_ylabel("Mean field gain vs Sobolev (%)")
    axis.set_title("B  Nonzero coverage is retained on every support split")
    axis.legend(frameon=False, fontsize=8, ncols=2, loc="lower right")

    axis = axes[1, 0]
    axis.bar(
        x - width / 2,
        raw_p10.mean(axis=1),
        width,
        color="#ba7a6e",
        label="raw p10",
    )
    axis.bar(
        x + width / 2,
        gated_p10.mean(axis=1),
        width,
        color="#51a38d",
        label="gated p10",
    )
    axis.axhline(0.0, color="#5d6460", linewidth=1.0)
    axis.set_ylabel("10th-percentile field gain (%)")
    axis.set_title("C  Lower tail improves, but does not become a proof")
    axis.legend(frameon=False)

    axis = axes[1, 1]
    coverage_image = np.vstack(
        (
            gated_coverage,
            np.zeros((2, gated_coverage.shape[1])),
        )
    )
    image = axis.imshow(
        coverage_image,
        aspect="auto",
        vmin=0.0,
        vmax=60.0,
        cmap="YlGnBu",
    )
    ylabels = [
        *[SPLIT_LABELS[split].replace("\n", " ") for split in order],
        "3-5 view geometry OOD",
        "3-5 view joint OOD",
    ]
    axis.set_yticks(np.arange(len(ylabels)), labels=ylabels)
    axis.set_xticks(np.arange(3), labels=("seed 1741", "seed 1742", "seed 1743"))
    for row in range(coverage_image.shape[0]):
        for column in range(coverage_image.shape[1]):
            axis.text(
                column,
                row,
                f"{coverage_image[row, column]:.0f}%",
                ha="center",
                va="center",
                color=(
                    "white"
                    if coverage_image[row, column] >= 35.0
                    else "#20302a"
                ),
                fontsize=8,
            )
    axis.grid(False)
    axis.set_title("D  Hard support envelope gives exact 0% outside support")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Coverage (%)")

    for axis in (axes[0, 0], axes[1, 0]):
        axis.set_xticks(x, labels=[SPLIT_LABELS[value] for value in order])
        axis.tick_params(axis="x", labelsize=8)
    figure.suptitle(
        "Observable conformal residual-risk gate | frozen fresh synthetic audit",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.012,
        (
            "Real PSU support geometry; 32^3 analytic morphology proxies; "
            "no experimental field truth. Three frozen spectral checkpoints, "
            "24 fields/split, 4F+4Aᵀ. Four accepted >1% harm rows remain."
        ),
        ha="center",
        fontsize=9,
        color="#4c5551",
    )
    figure.tight_layout(rect=(0.02, 0.04, 0.98, 0.95))

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for suffix in ("png", "pdf", "svg"):
        path = output_prefix.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=220 if suffix == "png" else None)
        if suffix == "svg":
            normalized = "\n".join(
                line.rstrip()
                for line in path.read_text(encoding="utf-8").splitlines()
            )
            path.write_text(normalized + "\n", encoding="utf-8")
        outputs[suffix] = {
            "filename": path.name,
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    plt.close(figure)
    manifest = {
        "schema_version": "psu-b0-residual-risk-fresh-figure-manifest-1.0",
        "source_summary": summary_path.name,
        "source_summary_sha256": _sha256(summary_path),
        "status": summary["status"],
        "outputs": outputs,
        "claim_boundary": summary["claim_boundary"],
    }
    manifest_path = output_prefix.with_name(
        f"{output_prefix.name}_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    result = plot(
        summary_path=args.summary,
        output_prefix=args.output_prefix,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
