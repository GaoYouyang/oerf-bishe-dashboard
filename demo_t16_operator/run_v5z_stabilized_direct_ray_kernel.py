#!/usr/bin/env python3
"""Run the explicitly post-v5y stabilized direct ray-kernel development screen."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from demo_t16_operator.run_v5y_direct_ray_conditioned_kernel import (
        run as run_base,
    )
else:
    from .run_v5y_direct_ray_conditioned_kernel import run as run_base


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "v5z_stabilized_direct_ray_kernel.json"
OUTPUT_DIR = ROOT / "results" / "v5z_stabilized_direct_ray_kernel"


def run() -> dict[str, Any]:
    return run_base(CONFIG_PATH, OUTPUT_DIR, Path(__file__).resolve())


def main() -> None:
    report = run()
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "training": report["training_protocol"],
                "model_size": report["model_size"],
                "improvement": report[
                    "prediction_improvement_over_full_matrix_ridge"
                ],
                "worst_ratio": report[
                    "prediction_worst_rig_ratio_to_full_matrix_ridge"
                ],
                "development_summary": report["development_summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
