from __future__ import annotations

from site_tools.run_psu_b0_exact_absolute_diagnostic import classify_root_cause


METHODS = (
    "scalar_a_only_pdhg",
    "formal_factor_view_a_only_pdhg",
    "factor_row_hybrid_a_only_pdhg",
    "exact_abs_view_a_only_pdhg",
    "exact_abs_row_a_only_pdhg",
    "graph_pcgls",
)


def _rows(errors: dict[str, tuple[float, float]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for replicate in (0, 8):
        for sample_index in range(8):
            for method in METHODS:
                for iteration, value in zip((32, 128), errors[method], strict=True):
                    rows.append(
                        {
                            "replicate": replicate,
                            "sample_index": sample_index,
                            "method": method,
                            "iterations": iteration,
                            "field_relative_l2": value,
                            "normalized_data_residual_l2": (
                                None if method == "graph_pcgls" else value
                            ),
                        }
                    )
    return rows


def _tightness(value: float = 0.4) -> list[dict[str, float]]:
    return [
        {
            "row_ratio_median": value,
            "column_ratio_median": value,
            "row_ratio_p05": value,
            "column_ratio_p05": value,
        }
        for _ in range(16)
    ]


THRESHOLDS = {
    "descriptive_endpoint_k": 128,
    "material_residual_gain_percent_min": 10.0,
    "material_paired_sample_count_min": 12,
    "material_high_quantile_slack_min": 0.1,
    "nonbinding_exact_to_graph_field_ratio_min": 1.5,
}


def test_classification_separates_factor_cancellation_from_view_aggregation() -> None:
    cancellation = {
        "scalar_a_only_pdhg": (1.1, 0.9),
        "formal_factor_view_a_only_pdhg": (1.0, 0.8),
        "factor_row_hybrid_a_only_pdhg": (0.95, 0.75),
        "exact_abs_view_a_only_pdhg": (0.65, 0.55),
        "exact_abs_row_a_only_pdhg": (0.6, 0.5),
        "graph_pcgls": (0.4, 0.3),
    }
    observed = classify_root_cause(
        _rows(cancellation),
        _tightness(),
        THRESHOLDS,
    )
    assert observed["status"] == "FACTOR_MAJORIZER_CANCELLATION_MATERIAL_DESCRIPTIVE"
    assert observed["formal_gate_b_reopened"] is False

    aggregation = dict(cancellation)
    aggregation["factor_row_hybrid_a_only_pdhg"] = (0.65, 0.55)
    aggregation["exact_abs_view_a_only_pdhg"] = (0.95, 0.75)
    observed = classify_root_cause(
        _rows(aggregation),
        _tightness(0.8),
        THRESHOLDS,
    )
    assert observed["status"] == "VIEW_AGGREGATION_MATERIAL_DESCRIPTIVE"


def test_classification_preserves_krylov_headroom_without_causal_overclaim() -> None:
    errors = {
        "scalar_a_only_pdhg": (1.1, 1.0),
        "formal_factor_view_a_only_pdhg": (1.0, 0.9),
        "factor_row_hybrid_a_only_pdhg": (0.98, 0.85),
        "exact_abs_view_a_only_pdhg": (0.97, 0.83),
        "exact_abs_row_a_only_pdhg": (0.95, 0.82),
        "graph_pcgls": (0.4, 0.3),
    }
    observed = classify_root_cause(_rows(errors), _tightness(0.8), THRESHOLDS)
    assert observed["status"] == "STATIC_DIAGONAL_GAIN_SMALL_GRAPH_HEADROOM_NONBINDING"
    assert observed["exact_abs_row_to_graph_field_error_ratio_nonbinding"] > 1.5
