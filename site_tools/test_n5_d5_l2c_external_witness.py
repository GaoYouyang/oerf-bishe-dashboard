from __future__ import annotations

import base64
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import inspect
import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from jsonschema import Draft202012Validator, FormatChecker
import pytest

import site_tools.n5_d5_l2c_external_witness as witness


def _write_json(path: Path, value: object) -> str:
    payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _public_key_base64(private_key: Ed25519PrivateKey) -> str:
    raw = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return base64.b64encode(raw).decode("ascii")


def _claims() -> dict[str, bool]:
    return {key: False for key in witness.CLAIM_KEYS}


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _policy(
    host_key: Ed25519PrivateKey,
    cost_key: Ed25519PrivateKey,
    *,
    now: datetime,
) -> dict[str, object]:
    return {
        "schema_version": "n5-d5-l2c-trust-policy-1.0",
        "record_kind": "OUT_OF_BAND_PINNED_WITNESS_TRUST_POLICY",
        "policy_id": "independent_witness_policy_v1",
        "policy_epoch": 1,
        "valid_from_utc": _utc(now - timedelta(hours=1)),
        "valid_until_utc": _utc(now + timedelta(hours=1)),
        "expected_predicate_type": (
            "https://gaoyouyang.github.io/oerf-bishe-dashboard/"
            "attestations/n5-d5-l2c-external-witness/v1"
        ),
        "required_signature_roles": list(witness.REQUIRED_ROLES),
        "signature_threshold": 2,
        "trusted_keys": [
            {
                "key_id": "host_witness_key_v1",
                "role": "host_capability_witness",
                "algorithm": "ed25519",
                "public_key_base64": _public_key_base64(host_key),
                "operational_independence": "OUT_OF_PROCESS_FROM_RUNNER",
            },
            {
                "key_id": "cost_observer_key_v1",
                "role": "run_cost_observer",
                "algorithm": "ed25519",
                "public_key_base64": _public_key_base64(cost_key),
                "operational_independence": "OUT_OF_PROCESS_FROM_RUNNER",
            },
        ],
        "claim_authorizations": _claims(),
    }


def _subject_files(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    root = tmp_path / "subject"
    root.mkdir()
    for key in witness.SUBJECT_KEYS:
        path = root / f"{key}.bin"
        path.write_bytes(("fixture-bytes:" + key).encode("utf-8"))
        paths[key] = path
    return paths


def _subject(policy_sha256: str, subject_files: dict[str, Path]) -> dict[str, str]:
    value = {
        key: hashlib.sha256(subject_files[key].read_bytes()).hexdigest()
        for key in witness.SUBJECT_KEYS
    }
    value["trust_policy_sha256"] = policy_sha256
    return value


def _evidence_materials(
    tmp_path: Path,
) -> tuple[list[Path], dict[str, tuple[str, str]], str]:
    paths: list[Path] = []
    references: dict[str, tuple[str, str]] = {}
    root = tmp_path / "evidence"
    root.mkdir()
    for index, code in enumerate(witness.CAPABILITY_CODES):
        artifact = root / f"capability_{index}.json"
        tool = root / f"probe_tool_{index}.bin"
        artifact.write_text("private probe result: " + code, encoding="utf-8")
        tool.write_bytes(("probe-tool:" + code).encode("utf-8"))
        paths.extend((artifact, tool))
        references[code] = (
            hashlib.sha256(artifact.read_bytes()).hexdigest(),
            hashlib.sha256(tool.read_bytes()).hexdigest(),
        )
    observer = root / "cost_observer.bin"
    observer.write_bytes(b"independent-cost-observer-fixture")
    paths.append(observer)
    return paths, references, hashlib.sha256(observer.read_bytes()).hexdigest()


def _capability_statement(
    references: dict[str, tuple[str, str]],
) -> dict[str, object]:
    evidence = []
    for code in witness.CAPABILITY_CODES:
        evidence.append(
            {
                "capability_code": code,
                "result": "PASS",
                "artifact_sha256": references[code][0],
                "measurement_tool_sha256": references[code][1],
                "measurement_method": "independent out-of-process adversarial probe",
                "private_artifact_published": False,
            }
        )
    return {
        "scope": (
            "SIGNED_WITNESS_ASSERTION_REQUIRES_EVIDENCE_AND_BACKEND_REVIEW"
        ),
        "capabilities": {
            "process_exec_replacement_denied": True,
            "private_input_root_external_mutation_denied": True,
            "durable_nonce_ledger_root_protected": True,
            "output_root_external_mutation_denied": True,
        },
        "evidence": evidence,
    }


def _cost(observer_tool_sha256: str) -> dict[str, object]:
    return {
        "phase": "DESCRIBE_ONLY",
        "observation_source": "EXTERNAL_OUT_OF_PROCESS",
        "clock_kind": "MONOTONIC",
        "describe_only_wall_time_ns": 1_000_000,
        "describe_only_cpu_user_ns": 120_000,
        "describe_only_cpu_system_ns": 80_000,
        "describe_only_peak_rss_bytes": None,
        "process_launches": 1,
        "failed_prior_attempts": 0,
        "describe_event_measurement_complete": True,
        "unobserved_processes_possible": False,
        "processes_observed": 1,
        "descendants_observed": 0,
        "external_operation_counts": {
            "describe": 2,
            "forward": 0,
            "jvp": 0,
            "vjp": 0,
        },
        "describe_only_adapter_physics_work": {
            "status": "NOT_OBSERVED_IN_DESCRIBE_ONLY_RUN",
            "ray_segments": None,
            "integration_samples": None,
            "kernel_evaluations": None,
        },
        "memory_observation_status": "NOT_RELIABLY_OBSERVED",
        "observer_tool_sha256": observer_tool_sha256,
    }


def _event_chain(
    subject: dict[str, str],
    capability: dict[str, object],
    cost: dict[str, object],
) -> dict[str, object]:
    capability_sha256 = witness.sha256_bytes(
        witness.canonical_json(capability).encode("utf-8")
    )
    cost_sha256 = witness.sha256_bytes(
        witness.canonical_json(cost).encode("utf-8")
    )
    payloads = [_digest(f"event-payload-{index}") for index in range(14)]
    payloads[0] = witness.sha256_bytes(
        witness.canonical_json(subject).encode("utf-8")
    )
    payloads[2] = capability_sha256
    payloads[3] = subject["authorization_sha256"]
    payloads[4] = subject["authorization_sha256"]
    payloads[5] = subject["adapter_entrypoint_sha256"]
    payloads[11] = subject["output_manifest_sha256"]
    payloads[12] = cost_sha256

    events: list[dict[str, object]] = []
    previous = witness.ZERO_SHA256
    for sequence, event_type in enumerate(witness.EXPECTED_EVENT_TYPES):
        request_index, operation = witness._event_expected_request(sequence)
        if sequence == 13:
            seal_material = {
                "subject_sha256": witness.sha256_bytes(
                    witness.canonical_json(subject).encode("utf-8")
                ),
                "capability_statement_sha256": capability_sha256,
                "observed_cost_sha256": cost_sha256,
                "previous_event_sha256": events[12]["event_sha256"],
            }
            payloads[13] = witness.sha256_bytes(
                witness.canonical_json(seal_material).encode("utf-8")
            )
        event: dict[str, object] = {
            "sequence": sequence,
            "event_type": event_type,
            "request_index": request_index,
            "operation": operation,
            "monotonic_ns": (sequence + 1) * 1_000,
            "payload_sha256": payloads[sequence],
            "previous_event_sha256": previous,
        }
        event["event_sha256"] = witness.event_sha256(event)
        previous = str(event["event_sha256"])
        events.append(event)
    return {
        "hash_algorithm": "sha256",
        "domain": "OERF:N5-D5:L2-C:EVENT:V1",
        "event_count": 14,
        "final_event_sha256": previous,
        "events": events,
    }


def _resign(
    bundle: dict[str, object],
    host_key: Ed25519PrivateKey,
    cost_key: Ed25519PrivateKey,
) -> None:
    bundle["signatures"] = []
    signatures = []
    for key_id, role, private_key in (
        ("host_witness_key_v1", "host_capability_witness", host_key),
        ("cost_observer_key_v1", "run_cost_observer", cost_key),
    ):
        message = witness.role_signed_payload_bytes(bundle, role)
        payload_sha256 = witness.sha256_bytes(message)
        payload_scope = (
            "HOST_CAPABILITY_SUBJECT_AND_FINAL_SEAL"
            if role == "host_capability_witness"
            else "RUN_EVENT_CHAIN_AND_DESCRIBE_TELEMETRY"
        )
        signatures.append(
            {
                "key_id": key_id,
                "role": role,
                "algorithm": "ed25519",
                "payload_scope": payload_scope,
                "signed_payload_sha256": payload_sha256,
                "signature_base64": base64.b64encode(
                    private_key.sign(message)
                ).decode("ascii"),
            }
        )
    bundle["signatures"] = signatures


def _bundle(
    subject: dict[str, str],
    capability: dict[str, object],
    cost: dict[str, object],
    host_key: Ed25519PrivateKey,
    cost_key: Ed25519PrivateKey,
    *,
    now: datetime,
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "n5-d5-l2c-external-witness-bundle-1.0",
        "record_kind": "EXTERNALLY_WITNESSED_DESCRIBE_ONLY_CAPSULE",
        "predicate_type": (
            "https://gaoyouyang.github.io/oerf-bishe-dashboard/"
            "attestations/n5-d5-l2c-external-witness/v1"
        ),
        "run_id": "anonymous_witnessed_describe_run_v1",
        "issued_at_utc": _utc(now - timedelta(seconds=10)),
        "expires_at_utc": _utc(now + timedelta(minutes=5)),
        "subject": subject,
        "host_capability_statement": capability,
        "event_chain": _event_chain(subject, capability, cost),
        "observed_cost": cost,
        "terminal_status": "DESCRIBE_ONLY_COMPLETE_NO_SCIENCE_AUTHORIZATION",
        "claim_authorizations": _claims(),
        "limitations": list(witness.REQUIRED_LIMITATIONS),
        "signatures": [],
    }
    _resign(value, host_key, cost_key)
    return value


def _registry(policy: dict[str, object], policy_sha256: str) -> dict[str, object]:
    return {
        "schema_version": "n5-d5-l2c-trust-anchor-registry-1.0",
        "record_kind": "VERIFIER_CONTROLLED_TRUST_ANCHOR_REGISTRY",
        "registry_id": "temporary_development_registry_v1",
        "production_trust_anchor_enrolled": True,
        "active_policy_pins": [
            {
                "policy_id": policy["policy_id"],
                "policy_epoch": policy["policy_epoch"],
                "policy_sha256": policy_sha256,
                "predicate_type": policy["expected_predicate_type"],
                "required_signature_roles": list(witness.REQUIRED_ROLES),
                "enrollment_review_sha256": _digest("independent-review-record"),
            }
        ],
        "claim_authorizations": _claims(),
    }


def _fixture(tmp_path: Path) -> dict[str, object]:
    now = datetime.now(timezone.utc)
    host_key = Ed25519PrivateKey.generate()
    cost_key = Ed25519PrivateKey.generate()
    policy = _policy(host_key, cost_key, now=now)
    policy_path = tmp_path / "policy.json"
    policy_sha256 = _write_json(policy_path, policy)
    registry = _registry(policy, policy_sha256)
    registry_path = tmp_path / "registry.json"
    _write_json(registry_path, registry)
    subject_files = _subject_files(tmp_path)
    subject = _subject(policy_sha256, subject_files)
    evidence_files, evidence_references, cost_observer_sha256 = (
        _evidence_materials(tmp_path)
    )
    capability = _capability_statement(evidence_references)
    cost = _cost(cost_observer_sha256)
    bundle = _bundle(
        subject, capability, cost, host_key, cost_key, now=now
    )
    bundle_path = tmp_path / "bundle.json"
    _write_json(bundle_path, bundle)
    return {
        "host_key": host_key,
        "cost_key": cost_key,
        "policy": policy,
        "policy_path": policy_path,
        "policy_sha256": policy_sha256,
        "registry": registry,
        "registry_path": registry_path,
        "bundle": bundle,
        "bundle_path": bundle_path,
        "subject_files": subject_files,
        "evidence_files": evidence_files,
    }


def _verify(fixture: dict[str, object]) -> dict[str, object]:
    return witness._verify_external_witness_bundle_with_registry(
        fixture["bundle_path"],
        fixture["policy_path"],
        subject_files=fixture["subject_files"],
        evidence_files=fixture["evidence_files"],
        trust_anchor_registry_path=fixture["registry_path"],
    )


def _rewrite_bundle(fixture: dict[str, object], *, resign: bool = False) -> None:
    bundle = fixture["bundle"]
    if resign:
        _resign(bundle, fixture["host_key"], fixture["cost_key"])
    _write_json(fixture["bundle_path"], bundle)


def test_placeholder_schemas_are_closed_and_keep_all_claims_false() -> None:
    format_checker = FormatChecker()
    for schema_relative, placeholder_relative in (
        (witness.POLICY_SCHEMA_RELATIVE, witness.POLICY_PLACEHOLDER_RELATIVE),
        (witness.BUNDLE_SCHEMA_RELATIVE, witness.BUNDLE_PLACEHOLDER_RELATIVE),
        (
            witness.TRUST_ANCHOR_REGISTRY_SCHEMA_RELATIVE,
            witness.TRUST_ANCHOR_REGISTRY_RELATIVE,
        ),
    ):
        schema = json.loads((witness.ROOT / schema_relative).read_text())
        placeholder = json.loads((witness.ROOT / placeholder_relative).read_text())
        errors = list(
            Draft202012Validator(
                schema, format_checker=format_checker
            ).iter_errors(placeholder)
        )
        assert errors == []
        changed = copy.deepcopy(placeholder)
        changed["claim_authorizations"]["publication"] = True
        assert list(
            Draft202012Validator(
                schema, format_checker=format_checker
            ).iter_errors(changed)
        )


def test_verifier_has_no_signer_or_production_bypass_surface() -> None:
    parameters = inspect.signature(
        witness.verify_external_witness_bundle
    ).parameters
    assert not any(
        marker in name.lower()
        for name in parameters
        for marker in ("fixture", "bypass", "override", "unsafe", "insecure")
    )
    source = inspect.getsource(witness)
    assert "Ed25519PrivateKey" not in source
    assert "production_backend_authorized\": True" not in source
    assert "expected_policy_sha256" not in parameters
    assert "trust_anchor_registry_path" not in parameters


def test_valid_two_role_bundle_verifies_transport_but_not_capability_truth(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    report = _verify(fixture)
    assert report["trusted_signature_count"] == 2
    assert report["event_chain_verified"] is True
    assert report["subject_artifact_bytes_rehashed"] is True
    assert report["private_evidence_bytes_rehashed"] is True
    assert report[
        "external_describe_telemetry_statement_structurally_verified"
    ] is True
    assert report[
        "external_describe_telemetry_truth_independently_proven"
    ] is False
    assert report["eligible_for_reconstruction_or_model_cost_comparison"] is False
    assert report["host_capability_claims_signed"] is True
    assert report["host_capability_truth_independently_proven"] is False
    assert report["role_key_separation_verified"] is True
    assert report["role_operational_independence_proven"] is False
    assert report["replay_ledger_checked"] is False
    assert report["one_time_acceptance_proven"] is False
    assert report["backend_capability_attestation_blocker_automatically_cleared"] is False
    assert report["production_backend_authorized"] is False
    assert report["formal_replay_authorized"] is False
    assert report["model_training_authorized"] is False
    assert all(value is False for value in report["claim_authorizations"].values())
    repeated = _verify(fixture)
    assert repeated["one_time_acceptance_proven"] is False
    removed = fixture["evidence_files"].pop()
    with pytest.raises(witness.L2CWitnessError, match="observer tool bytes"):
        _verify(fixture)
    fixture["evidence_files"].append(removed)


def test_unsigned_tamper_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["bundle"]["run_id"] = "tampered_run_id"
    _rewrite_bundle(fixture)
    with pytest.raises(witness.L2CWitnessError, match="signature"):
        _verify(fixture)


def test_authorized_resign_cannot_hide_broken_event_hash(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["bundle"]["event_chain"]["events"][6]["payload_sha256"] = _digest(
        "tampered-request"
    )
    _rewrite_bundle(fixture, resign=True)
    with pytest.raises(witness.L2CWitnessError, match="event hash"):
        _verify(fixture)


def test_event_reordering_is_rejected_even_when_resigned(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    events = fixture["bundle"]["event_chain"]["events"]
    events[6], events[7] = events[7], events[6]
    _rewrite_bundle(fixture, resign=True)
    with pytest.raises(witness.L2CWitnessError, match="sequence"):
        _verify(fixture)


def test_output_manifest_event_must_match_subject(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    event = fixture["bundle"]["event_chain"]["events"][11]
    event["payload_sha256"] = _digest("wrong-output")
    unsigned = dict(event)
    unsigned.pop("event_sha256")
    event["event_sha256"] = witness.event_sha256(unsigned)
    for later in fixture["bundle"]["event_chain"]["events"][12:]:
        later["previous_event_sha256"] = fixture["bundle"]["event_chain"]["events"][
            later["sequence"] - 1
        ]["event_sha256"]
        unsigned = dict(later)
        unsigned.pop("event_sha256")
        later["event_sha256"] = witness.event_sha256(unsigned)
    fixture["bundle"]["event_chain"]["final_event_sha256"] = fixture["bundle"][
        "event_chain"
    ]["events"][-1]["event_sha256"]
    _rewrite_bundle(fixture, resign=True)
    with pytest.raises(witness.L2CWitnessError, match="output-bound"):
        _verify(fixture)


def test_duplicate_signer_key_cannot_satisfy_threshold(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["bundle"]["signatures"][1] = copy.deepcopy(
        fixture["bundle"]["signatures"][0]
    )
    _rewrite_bundle(fixture)
    with pytest.raises(witness.L2CWitnessError, match="duplicate signer"):
        _verify(fixture)


def test_untrusted_signer_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    attacker = Ed25519PrivateKey.generate()
    bundle = fixture["bundle"]
    _resign(bundle, fixture["host_key"], fixture["cost_key"])
    host_message = witness.role_signed_payload_bytes(
        bundle, "host_capability_witness"
    )
    bundle["signatures"][0] = {
        "key_id": "attacker_host_key",
        "role": "host_capability_witness",
        "algorithm": "ed25519",
        "payload_scope": "HOST_CAPABILITY_SUBJECT_AND_FINAL_SEAL",
        "signed_payload_sha256": witness.sha256_bytes(host_message),
        "signature_base64": base64.b64encode(attacker.sign(host_message)).decode(),
    }
    _rewrite_bundle(fixture)
    with pytest.raises(witness.L2CWitnessError, match="untrusted signer"):
        _verify(fixture)


def test_public_api_cannot_accept_a_caller_controlled_trust_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    public_registry = witness.ROOT / witness.TRUST_ANCHOR_REGISTRY_RELATIVE
    assert hashlib.sha256(public_registry.read_bytes()).hexdigest() == (
        witness.COMPILED_TRUST_ANCHOR_REGISTRY_SHA256
    )
    with pytest.raises(
        witness.L2CWitnessError, match="NO_PRODUCTION_TRUST_ANCHOR_ENROLLED"
    ):
        witness.verify_external_witness_bundle(
            fixture["bundle_path"],
            fixture["policy_path"],
            subject_files=fixture["subject_files"],
            evidence_files=fixture["evidence_files"],
        )
    fake_install_root = tmp_path / "tampered-fixed-install"
    tampered_registry = fake_install_root / witness.TRUST_ANCHOR_REGISTRY_RELATIVE
    tampered_registry.parent.mkdir(parents=True)
    _write_json(tampered_registry, fixture["registry"])
    monkeypatch.setattr(witness, "ROOT", fake_install_root)
    with pytest.raises(
        witness.L2CWitnessError, match="VERIFIER_CONTROLLED_REGISTRY_BYTES_CHANGED"
    ):
        witness.verify_external_witness_bundle(
            fixture["bundle_path"],
            fixture["policy_path"],
            subject_files=fixture["subject_files"],
            evidence_files=fixture["evidence_files"],
        )


def test_subject_mismatch_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["subject_files"]["adapter_entrypoint_sha256"].write_bytes(
        b"different-adapter-bytes"
    )
    with pytest.raises(witness.L2CWitnessError, match="subject mismatch"):
        _verify(fixture)


def test_expired_bundle_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    now = datetime.now(timezone.utc)
    fixture["bundle"]["issued_at_utc"] = _utc(now - timedelta(minutes=10))
    fixture["bundle"]["expires_at_utc"] = _utc(now - timedelta(minutes=5))
    _rewrite_bundle(fixture, resign=True)
    with pytest.raises(witness.L2CWitnessError, match="expired"):
        _verify(fixture)


def test_bundle_lifetime_over_fifteen_minutes_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    now = datetime.now(timezone.utc)
    fixture["bundle"]["issued_at_utc"] = _utc(now - timedelta(minutes=1))
    fixture["bundle"]["expires_at_utc"] = _utc(now + timedelta(minutes=15))
    _rewrite_bundle(fixture, resign=True)
    with pytest.raises(witness.L2CWitnessError, match="fifteen"):
        _verify(fixture)


def test_duplicate_capability_evidence_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    evidence = fixture["bundle"]["host_capability_statement"]["evidence"]
    evidence[1]["capability_code"] = evidence[0]["capability_code"]
    _rewrite_bundle(fixture, resign=True)
    with pytest.raises(witness.L2CWitnessError, match="duplicate codes"):
        _verify(fixture)


def test_describe_cost_cannot_hide_a_forward_call(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["bundle"]["observed_cost"]["external_operation_counts"]["forward"] = 1
    _rewrite_bundle(fixture, resign=True)
    with pytest.raises(witness.L2CWitnessError, match="schema failed"):
        _verify(fixture)


def test_unreliable_memory_observation_cannot_publish_pseudo_precision(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["bundle"]["observed_cost"][
        "describe_only_peak_rss_bytes"
    ] = 123456
    _rewrite_bundle(fixture, resign=True)
    with pytest.raises(witness.L2CWitnessError, match="pseudo-precision"):
        _verify(fixture)


def test_required_limitations_cannot_be_softened(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["bundle"]["limitations"][0] = (
        "Ed25519 signatures prove all backend capability statements are true."
    )
    _rewrite_bundle(fixture, resign=True)
    with pytest.raises(witness.L2CWitnessError, match="weakens"):
        _verify(fixture)


def test_duplicate_json_keys_are_rejected_before_signature_work(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    raw = fixture["bundle_path"].read_text(encoding="utf-8")
    raw = raw.replace(
        '  "schema_version": "n5-d5-l2c-external-witness-bundle-1.0",',
        '  "schema_version": "n5-d5-l2c-external-witness-bundle-1.0",\n'
        '  "schema_version": "n5-d5-l2c-external-witness-bundle-1.0",',
        1,
    )
    fixture["bundle_path"].write_text(raw, encoding="utf-8")
    with pytest.raises(witness.L2CWitnessError, match="duplicate JSON key"):
        _verify(fixture)


def test_nonfinite_json_is_rejected_before_schema_work(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    raw = fixture["bundle_path"].read_text(encoding="utf-8")
    raw = raw.replace(
        '"describe_only_wall_time_ns": 1000000',
        '"describe_only_wall_time_ns": NaN',
        1,
    )
    fixture["bundle_path"].write_text(raw, encoding="utf-8")
    with pytest.raises(witness.L2CWitnessError, match="non-finite"):
        _verify(fixture)


def test_all_zero_trust_key_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["policy"]["trusted_keys"][0]["public_key_base64"] = (
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
    )
    fixture["policy_sha256"] = _write_json(
        fixture["policy_path"], fixture["policy"]
    )
    fixture["registry"]["active_policy_pins"][0]["policy_sha256"] = fixture[
        "policy_sha256"
    ]
    _write_json(fixture["registry_path"], fixture["registry"])
    with pytest.raises(witness.L2CWitnessError, match="all-zero"):
        _verify(fixture)


def test_placeholder_bundle_is_schema_valid_but_operationally_rejected() -> None:
    placeholder = witness.ROOT / witness.BUNDLE_PLACEHOLDER_RELATIVE
    policy = witness.ROOT / witness.POLICY_PLACEHOLDER_RELATIVE
    with pytest.raises(
        witness.L2CWitnessError, match="NO_PRODUCTION_TRUST_ANCHOR_ENROLLED"
    ):
        witness.verify_external_witness_bundle(
            placeholder,
            policy,
            subject_files={key: placeholder for key in witness.SUBJECT_KEYS},
            evidence_files=[placeholder],
        )
