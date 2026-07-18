from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from site_tools.run_n5_d5_minimum_interface_bridge import (
    DEFAULT_SCHEMA,
    ROOT,
    build_requests,
    run,
    validate_config,
)


CONFIG = (
    ROOT
    / "demo_t16_operator/configs/"
    "n5_d5_minimum_interface_bridge_preregistered_v1.json"
)


def _config() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_config_and_request_plan_freeze_exact_probe_budget() -> None:
    config = _config()
    validate_config(config, DEFAULT_SCHEMA)
    requests, vectors = build_requests(config, protocol_commit="a" * 40)

    assert len(requests) == 53
    assert sum(row["operation"] == "describe" for row in requests) == 2
    assert sum(row["operation"] == "forward" for row in requests) == 42
    assert sum(row["operation"] == "jvp" for row in requests) == 6
    assert sum(row["operation"] == "vjp" for row in requests) == 3
    assert vectors["tangents"].shape == (2, 23)
    assert vectors["cotangents"].shape == (1, 8)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["field"].pop("units"), "units"),
        (
            lambda value: value["observation"].update({"output_dimension": 7}),
            "output dimension",
        ),
        (
            lambda value: value["paths"][1].update(
                {"path_id": value["paths"][0]["path_id"]}
            ),
            "path_id",
        ),
        (
            lambda value: value["claim_authorizations"].update(
                {"derivative_correctness": True}
            ),
            "claim_authorizations",
        ),
    ],
)
def test_config_rejects_missing_or_false_authority(
    mutation: object, message: str
) -> None:
    config = copy.deepcopy(_config())
    mutation(config)

    with pytest.raises(ValueError, match=message):
        validate_config(config, DEFAULT_SCHEMA)


def test_lab_record_forbids_public_raw_traces() -> None:
    config = copy.deepcopy(_config())
    config["record_kind"] = "LAB_ANONYMOUS_INTERFACE"
    config["identity"]["real_lab_data_present"] = True
    config["identity"]["evidence_class"] = "private_lab_callable"
    config["adapter"]["source_review_status"] = "private_source_reviewed"
    config["privacy"]["raw_trace_publication_permitted"] = True

    with pytest.raises(ValueError, match="raw traces"):
        validate_config(config, DEFAULT_SCHEMA)


def test_synthetic_run_emits_replayable_raw_trace_and_closed_claims(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result"
    result = run(
        config_path=CONFIG,
        output_dir=output,
        enforce_committed=False,
    )

    assert result["machine_decision"] == (
        "SYNTHETIC_PROTOCOL_PASS_NO_LAB_AUTHORIZATION"
    )
    assert result["counts"]["request_count"] == 53
    assert result["counts"]["response_count"] == 53
    assert result["counts"]["forward_api_calls"] == 42
    assert result["counts"]["jvp_api_calls"] == 6
    assert result["counts"]["vjp_api_calls"] == 3
    assert result["decision"]["claim_authorized"] is False
    assert not any(result["claim_authorizations"].values())
    assert result["process"]["command"] == [
        "{python}",
        "-m",
        "demo_t16_operator.n5_d5_synthetic_reference_adapter",
    ]
    assert "/Users/" not in json.dumps(result)
    assert result["metrics_summary"]["all_path_alias_checks_passed"] is True
    assert result["metrics_summary"]["any_actual_forward_branch_change"] is False

    requests = [
        json.loads(line)
        for line in (output / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    responses = [
        json.loads(line)
        for line in (output / "responses.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(requests) == len(responses) == 53
    assert all(
        request["audit_nonce"] == response["audit_nonce"]
        for request, response in zip(requests, responses, strict=True)
    )
    assert (output / "interface_bridge.png").stat().st_size > 10_000
    assert (output / "manifest.json").is_file()
