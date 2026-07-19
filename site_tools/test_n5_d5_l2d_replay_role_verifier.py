from __future__ import annotations

import base64
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from jsonschema import Draft202012Validator, FormatChecker
import pytest

import site_tools.n5_d5_l2d_replay_role_verifier as replay


def _write_json(path: Path, value: object) -> str:
    payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _claims() -> dict[str, bool]:
    return {key: False for key in replay.CLAIM_KEYS}


def _public_key_base64(private_key: Ed25519PrivateKey) -> str:
    raw = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return base64.b64encode(raw).decode("ascii")


def _historical_entry(now: datetime) -> dict[str, object]:
    body = {
        "entry_kind": "ONE_TIME_AUTHORIZATION_ACCEPTANCE",
        "ledger_namespace": "oerf_n5_d5_describe_authorizations_v1",
        "acceptance_id": "historical_acceptance_entry_v1",
        "run_id": "historical_anonymous_run_v1",
        "authorization_id": "historical_authorization_id_v1",
        "nonce_sha256": _digest("historical-nonce"),
        "authorization_sha256": _digest("historical-authorization"),
        "l2c_bundle_sha256": _digest("historical-l2c-bundle"),
        "l2c_verification_report_sha256": _digest("historical-l2c-report"),
        "l2c_final_event_sha256": _digest("historical-final-event"),
        "decision": "FIRST_ACCEPT",
        "recorded_at_utc": _utc(now - timedelta(minutes=20)),
    }
    return {
        "index": 0,
        "body": body,
        "leaf_sha256": replay.merkle_leaf_sha256(body),
    }


def _policy(
    keys: dict[str, Ed25519PrivateKey],
    *,
    now: datetime,
    anchor_root: str,
) -> dict[str, object]:
    metadata = {
        "acceptance_sequencer": (
            "sequencer_key_v1",
            "ledger_authority_domain_v1",
            "https://ledger.example.invalid/sequencer",
            "https://iam-a.example.invalid/issuer",
        ),
        "checkpoint_monitor_a": (
            "monitor_a_key_v1",
            "checkpoint_monitor_domain_a_v1",
            "https://monitor-a.example.invalid/service",
            "https://iam-b.example.invalid/issuer",
        ),
        "checkpoint_monitor_b": (
            "monitor_b_key_v1",
            "checkpoint_monitor_domain_b_v1",
            "https://monitor-b.example.invalid/service",
            "https://iam-c.example.invalid/issuer",
        ),
    }
    trusted_keys = []
    for role in replay.REQUIRED_ROLES:
        key_id, operator_domain, service_identity, issuer = metadata[role]
        trusted_keys.append(
            {
                "key_id": key_id,
                "role": role,
                "algorithm": "ed25519",
                "public_key_base64": _public_key_base64(keys[role]),
                "operator_domain_id": operator_domain,
                "service_identity_uri": service_identity,
                "credential_issuer_uri": issuer,
            }
        )
    return {
        "schema_version": "n5-d5-l2d-replay-role-policy-1.0",
        "record_kind": "OUT_OF_BAND_PINNED_REPLAY_AND_ROLE_POLICY",
        "policy_id": "external_replay_and_role_policy_v1",
        "policy_epoch": 1,
        "valid_from_utc": _utc(now - timedelta(hours=1)),
        "valid_until_utc": _utc(now + timedelta(hours=1)),
        "expected_predicate_type": replay.PREDICATE_TYPE,
        "ledger_namespace": "oerf_n5_d5_describe_authorizations_v1",
        "log_id": "external_replay_log_v1",
        "log_epoch": 1,
        "required_signature_roles": list(replay.REQUIRED_ROLES),
        "signature_threshold": 3,
        "max_witnessed_prefix_entries": 256,
        "max_bundle_lifetime_seconds": 900,
        "max_checkpoint_age_seconds": 900,
        "max_merge_delay_seconds": 120,
        "floor_checkpoint": {"tree_size": 1, "root_sha256": anchor_root},
        "trusted_keys": trusted_keys,
        "claim_authorizations": _claims(),
    }


def _registry(policy: dict[str, object], policy_sha256: str) -> dict[str, object]:
    return {
        "schema_version": "n5-d5-l2d-trust-anchor-registry-1.0",
        "record_kind": "VERIFIER_CONTROLLED_REPLAY_TRUST_ANCHOR_REGISTRY",
        "registry_id": "temporary_l2d_development_registry_v1",
        "production_trust_anchor_enrolled": True,
        "active_policy_pins": [
            {
                "policy_id": policy["policy_id"],
                "policy_epoch": policy["policy_epoch"],
                "policy_sha256": policy_sha256,
                "predicate_type": policy["expected_predicate_type"],
                "required_signature_roles": list(replay.REQUIRED_ROLES),
                "log_epoch": policy["log_epoch"],
                "floor_checkpoint": policy["floor_checkpoint"],
                "enrollment_review_sha256": _digest("independent-l2d-review"),
            }
        ],
        "claim_authorizations": _claims(),
    }


def _authorization_fixture() -> dict[str, object]:
    value = json.loads(
        (
            replay.ROOT
            / "data_templates/n5_d5_l2b_describe_authorization.placeholder.json"
        ).read_text(encoding="utf-8")
    )
    value["authorization_id"] = "anonymous_describe_authorization_v1"
    value["one_time_nonce"] = _digest("semantic-one-time-nonce")
    value["foundation_report"] = {
        "relative_path": "private/foundation.json",
        "sha256": _digest("foundation"),
    }
    value["replay_plan"] = {
        "relative_path": "private/replay-plan.json",
        "sha256": _digest("replay-plan"),
    }
    value["describe_entrypoint"]["inventory_relative_path"] = (
        "private/describe_adapter.py"
    )
    value["expected_descriptor"]["adapter_id"] = "anonymous_adapter_v1"
    value["expected_descriptor"]["adapter_version"] = "fixture-v1"
    value["expected_descriptor"]["implementation_sha256"] = _digest("adapter")
    value["expected_descriptor"]["context_id"] = "anonymous_context_v1"
    value["output_directory"] = "private/runs/describe-001"
    return value


def _subject_files(
    tmp_path: Path,
) -> tuple[dict[str, Path], str, dict[str, object]]:
    root = tmp_path / "subject"
    root.mkdir()
    paths: dict[str, Path] = {}
    authorization = _authorization_fixture()
    authorization_path = root / "authorization.json"
    _write_json(authorization_path, authorization)
    paths["authorization_sha256"] = authorization_path

    challenge_path = root / "challenge_commitment.bin"
    challenge_path.write_bytes(b"bound-fixture:challenge-commitment")
    paths["challenge_commitment_sha256"] = challenge_path

    nonce_commitment = {
        "schema_version": "n5-d5-l2d-semantic-nonce-commitment-1.0",
        "authorization_id": authorization["authorization_id"],
        "domain": replay.NONCE_COMMITMENT_DOMAIN_LABEL,
        "semantic_nonce_sha256": replay.semantic_nonce_sha256(
            str(authorization["one_time_nonce"])
        ),
    }
    nonce_commitment_path = root / "nonce_commitment.json"
    _write_json(nonce_commitment_path, nonce_commitment)
    paths["nonce_commitment_sha256"] = nonce_commitment_path

    final_event = _digest("l2c-final-event")
    l2c_bundle = {
        "schema_version": "n5-d5-l2c-external-witness-bundle-1.0",
        "record_kind": "EXTERNALLY_WITNESSED_DESCRIBE_ONLY_CAPSULE",
        "event_chain": {"final_event_sha256": final_event},
        "claim_authorizations": _claims(),
    }
    l2c_bundle_path = root / "l2c_bundle.json"
    l2c_bundle_sha256 = _write_json(l2c_bundle_path, l2c_bundle)
    paths["l2c_bundle_sha256"] = l2c_bundle_path

    l2c_report = {
        "schema_version": "n5-d5-l2c-external-witness-verification-report-1.0",
        "status": (
            "L2C_SIGNED_STATEMENTS_STRUCTURALLY_VERIFIED_"
            "REPLAY_AND_EVIDENCE_REVIEW_REQUIRED"
        ),
        "bundle_file_sha256": l2c_bundle_sha256,
        "verifier_controlled_registry_used": True,
        "enrolled_policy_digest_matched": True,
        "event_chain_verified": True,
        "final_event_sha256": final_event,
        "role_key_separation_verified": True,
        "role_operational_independence_proven": False,
        "replay_ledger_checked": False,
        "one_time_acceptance_proven": False,
        "production_backend_authorized": False,
        "formal_replay_authorized": False,
        "model_training_authorized": False,
        "claim_authorizations": _claims(),
    }
    l2c_report_path = root / "l2c_report.json"
    _write_json(l2c_report_path, l2c_report)
    paths["l2c_verification_report_sha256"] = l2c_report_path
    return paths, final_event, authorization


def _role_evidence_files(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "role-evidence"
    root.mkdir()
    result = {}
    for role in replay.REQUIRED_ROLES:
        path = root / f"{role}.json"
        path.write_text("private operational evidence for " + role, encoding="utf-8")
        result[role] = path
    return result


def _subject(policy_sha256: str, subject_files: dict[str, Path]) -> dict[str, str]:
    return {
        **{
            key: hashlib.sha256(subject_files[key].read_bytes()).hexdigest()
            for key in replay.SUBJECT_KEYS
        },
        "trust_policy_sha256": policy_sha256,
    }


def _role_evidence(
    policy: dict[str, object], role_evidence_files: dict[str, Path]
) -> list[dict[str, object]]:
    by_role = {entry["role"]: entry for entry in policy["trusted_keys"]}
    result = []
    for role in replay.REQUIRED_ROLES:
        trusted = by_role[role]
        result.append(
            {
                "role": role,
                "key_id": trusted["key_id"],
                "operator_domain_id": trusted["operator_domain_id"],
                "service_identity_uri": trusted["service_identity_uri"],
                "credential_issuer_uri": trusted["credential_issuer_uri"],
                "evidence_kind": "SERVICE_IDENTITY_AND_DEPLOYMENT_MEASUREMENT",
                "evidence_artifact_sha256": hashlib.sha256(
                    role_evidence_files[role].read_bytes()
                ).hexdigest(),
                "private_artifact_published": False,
            }
        )
    return result


def _resign(
    bundle: dict[str, object], keys: dict[str, Ed25519PrivateKey], policy: dict[str, object]
) -> None:
    key_ids = {entry["role"]: entry["key_id"] for entry in policy["trusted_keys"]}
    signatures = []
    for role in replay.REQUIRED_ROLES:
        message = replay.role_signed_payload_bytes(bundle, role)
        signatures.append(
            {
                "key_id": key_ids[role],
                "role": role,
                "algorithm": "ed25519",
                "payload_scope": (
                    "ACCEPTANCE_SUBJECT_AND_CHECKPOINT"
                    if role == "acceptance_sequencer"
                    else "CHECKPOINT_PREFIX_AND_ROLE_EVIDENCE"
                ),
                "signed_payload_sha256": replay.sha256_bytes(message),
                "signature_base64": base64.b64encode(
                    keys[role].sign(message)
                ).decode("ascii"),
            }
        )
    bundle["signatures"] = signatures


def _bundle(
    *,
    now: datetime,
    subject: dict[str, str],
    final_event: str,
    historical: dict[str, object],
    policy: dict[str, object],
    policy_sha256: str,
    registry: dict[str, object],
    registry_sha256: str,
    authorization: dict[str, object],
    role_evidence: list[dict[str, object]],
    keys: dict[str, Ed25519PrivateKey],
) -> dict[str, object]:
    acceptance = {
        "entry_kind": "ONE_TIME_AUTHORIZATION_ACCEPTANCE",
        "ledger_namespace": policy["ledger_namespace"],
        "acceptance_id": "target_first_acceptance_v1",
        "run_id": "anonymous_l2d_describe_run_v1",
        "authorization_id": authorization["authorization_id"],
        "nonce_sha256": replay.semantic_nonce_sha256(
            str(authorization["one_time_nonce"])
        ),
        "authorization_sha256": subject["authorization_sha256"],
        "l2c_bundle_sha256": subject["l2c_bundle_sha256"],
        "l2c_verification_report_sha256": subject[
            "l2c_verification_report_sha256"
        ],
        "l2c_final_event_sha256": final_event,
        "decision": "FIRST_ACCEPT",
        "recorded_at_utc": _utc(now - timedelta(seconds=15)),
    }
    target_entry = {
        "index": 1,
        "body": acceptance,
        "leaf_sha256": replay.merkle_leaf_sha256(acceptance),
    }
    entries = [historical, target_entry]
    current_root = replay.merkle_root_from_leaf_hashes(
        [entry["leaf_sha256"] for entry in entries]
    )
    value = {
        "schema_version": "n5-d5-l2d-replay-role-bundle-1.0",
        "record_kind": "EXTERNALLY_WITNESSED_REPLAY_PREFIX_AND_ROLE_EVIDENCE_CAPSULE",
        "predicate_type": replay.PREDICATE_TYPE,
        "run_id": acceptance["run_id"],
        "issued_at_utc": _utc(now - timedelta(seconds=30)),
        "expires_at_utc": _utc(now + timedelta(minutes=5)),
        "trust_context": replay._expected_trust_context(
            registry=registry,
            registry_sha256=registry_sha256,
            policy=policy,
            policy_sha256=policy_sha256,
            challenge_commitment_sha256=subject[
                "challenge_commitment_sha256"
            ],
        ),
        "subject": subject,
        "acceptance": acceptance,
        "witnessed_prefix": {
            "hash_scheme": "RFC9162_SHA256_MERKLE_PREFIX_RECOMPUTED_FROM_GENESIS",
            "complete_from_genesis": True,
            "entries": entries,
        },
        "previous_checkpoint": {
            "log_id": policy["log_id"],
            "tree_size": 1,
            "root_sha256": historical["leaf_sha256"],
            "issued_at_utc": _utc(now - timedelta(minutes=10)),
        },
        "checkpoint": {
            "log_id": policy["log_id"],
            "tree_size": 2,
            "root_sha256": current_root,
            "issued_at_utc": _utc(now - timedelta(seconds=5)),
            "maximum_merge_delay_seconds": 120,
        },
        "role_evidence": role_evidence,
        "terminal_status": "PREFIX_AND_ROLE_EVIDENCE_VERIFIED_NO_PRODUCTION_AUTHORIZATION",
        "claim_authorizations": _claims(),
        "limitations": list(replay.REQUIRED_LIMITATIONS),
        "signatures": [],
    }
    _resign(value, keys, policy)
    return value


def _fixture(tmp_path: Path) -> dict[str, object]:
    now = datetime.now(timezone.utc)
    historical = _historical_entry(now)
    keys = {role: Ed25519PrivateKey.generate() for role in replay.REQUIRED_ROLES}
    policy = _policy(
        keys, now=now, anchor_root=str(historical["leaf_sha256"])
    )
    policy_path = tmp_path / "policy.json"
    policy_sha256 = _write_json(policy_path, policy)
    registry = _registry(policy, policy_sha256)
    registry_path = tmp_path / "registry.json"
    registry_sha256 = _write_json(registry_path, registry)
    subject_files, final_event, authorization = _subject_files(tmp_path)
    subject = _subject(policy_sha256, subject_files)
    role_evidence_files = _role_evidence_files(tmp_path)
    role_evidence = _role_evidence(policy, role_evidence_files)
    bundle = _bundle(
        now=now,
        subject=subject,
        final_event=final_event,
        historical=historical,
        policy=policy,
        policy_sha256=policy_sha256,
        registry=registry,
        registry_sha256=registry_sha256,
        authorization=authorization,
        role_evidence=role_evidence,
        keys=keys,
    )
    bundle_path = tmp_path / "bundle.json"
    _write_json(bundle_path, bundle)
    return {
        "now": now,
        "keys": keys,
        "policy": policy,
        "policy_path": policy_path,
        "registry": registry,
        "registry_path": registry_path,
        "authorization": authorization,
        "subject_files": subject_files,
        "role_evidence_files": role_evidence_files,
        "bundle": bundle,
        "bundle_path": bundle_path,
    }


def _verify(fixture: dict[str, object]) -> dict[str, object]:
    return replay._verify_with_registry(
        fixture["bundle_path"],
        fixture["policy_path"],
        subject_files=fixture["subject_files"],
        role_evidence_files=fixture["role_evidence_files"],
        trust_anchor_registry_path=fixture["registry_path"],
    )


def _rewrite_bundle(fixture: dict[str, object], *, resign: bool = False) -> None:
    if resign:
        _resign(fixture["bundle"], fixture["keys"], fixture["policy"])
    _write_json(fixture["bundle_path"], fixture["bundle"])


def _rebuild_current_checkpoint(fixture: dict[str, object]) -> None:
    entries = fixture["bundle"]["witnessed_prefix"]["entries"]
    for index, entry in enumerate(entries):
        entry["index"] = index
        entry["leaf_sha256"] = replay.merkle_leaf_sha256(entry["body"])
    fixture["bundle"]["checkpoint"]["tree_size"] = len(entries)
    fixture["bundle"]["checkpoint"]["root_sha256"] = (
        replay.merkle_root_from_leaf_hashes(
            [entry["leaf_sha256"] for entry in entries]
        )
    )


def _sync_subject_and_target(fixture: dict[str, object]) -> None:
    policy_sha256 = hashlib.sha256(fixture["policy_path"].read_bytes()).hexdigest()
    subject = _subject(policy_sha256, fixture["subject_files"])
    fixture["bundle"]["subject"] = subject
    target = fixture["bundle"]["acceptance"]
    target["authorization_sha256"] = subject["authorization_sha256"]
    target["l2c_bundle_sha256"] = subject["l2c_bundle_sha256"]
    target["l2c_verification_report_sha256"] = subject[
        "l2c_verification_report_sha256"
    ]
    fixture["bundle"]["witnessed_prefix"]["entries"][-1]["body"] = copy.deepcopy(
        target
    )
    _rebuild_current_checkpoint(fixture)


def _repin_policy(fixture: dict[str, object]) -> None:
    policy_sha256 = _write_json(fixture["policy_path"], fixture["policy"])
    fixture["registry"] = _registry(fixture["policy"], policy_sha256)
    _write_json(fixture["registry_path"], fixture["registry"])


def test_placeholder_schemas_are_closed_and_all_claims_remain_false() -> None:
    checker = FormatChecker()
    pairs = (
        (replay.REGISTRY_SCHEMA_RELATIVE, replay.REGISTRY_RELATIVE),
        (replay.POLICY_SCHEMA_RELATIVE, replay.POLICY_PLACEHOLDER_RELATIVE),
        (replay.BUNDLE_SCHEMA_RELATIVE, replay.BUNDLE_PLACEHOLDER_RELATIVE),
    )
    for schema_relative, value_relative in pairs:
        schema = json.loads((replay.ROOT / schema_relative).read_text())
        value = json.loads((replay.ROOT / value_relative).read_text())
        Draft202012Validator(schema, format_checker=checker).validate(value)
        assert schema["additionalProperties"] is False
        assert _claims() == value["claim_authorizations"]


def test_production_api_has_no_registry_or_signer_surface() -> None:
    parameters = inspect.signature(replay.verify_replay_role_bundle).parameters
    source = inspect.getsource(replay)
    assert "trust_anchor_registry_path" not in parameters
    assert replay._production_api_has_no_registry_parameter() is True
    assert "Ed25519PrivateKey" not in source
    assert ".sign(" not in source


def test_rfc9162_style_merkle_root_uses_leaf_and_node_domain_separation() -> None:
    a = replay.merkle_leaf_sha256({"x": 1})
    b = replay.merkle_leaf_sha256({"x": 2})
    assert a == hashlib.sha256(b"\x00" + b'{"x":1}').hexdigest()
    assert replay.merkle_root_from_leaf_hashes([]) == hashlib.sha256(b"").hexdigest()
    assert replay.merkle_root_from_leaf_hashes([a]) == a
    expected = hashlib.sha256(b"\x01" + bytes.fromhex(a) + bytes.fromhex(b)).hexdigest()
    assert replay.merkle_root_from_leaf_hashes([a, b]) == expected


def test_valid_prefix_verifies_limited_scope_but_not_global_one_time(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    report = _verify(fixture)
    assert report["target_semantic_nonce_unique_in_supplied_signed_prefix"] is True
    assert report["static_registry_floor_checkpoint_matched"] is True
    assert report["anti_rollback_protection_proven"] is False
    assert report["prefix_completeness_external_to_bundle_proven"] is False
    assert report["semantic_nonce_derived_from_parsed_authorization"] is True
    assert report["l2c_report_authenticity_proven"] is False
    assert report["l2c_report_status_interpreted"] is False
    assert report["trusted_signature_count"] == 3
    assert report["role_key_separation_verified"] is True
    assert report["role_evidence_files_rehashed"] is True
    assert report["online_atomic_consume_observed"] is False
    assert report["cross_observer_gossip_performed"] is False
    assert report["log_fork_freedom_proven"] is False
    assert report["role_operational_independence_proven"] is False
    assert report["one_time_acceptance_proven"] is False
    assert report["production_backend_authorized"] is False
    assert report["model_training_authorized"] is False
    assert report["claim_authorizations"] == _claims()


def test_repeated_offline_verification_does_not_create_one_time_proof(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    first = _verify(fixture)
    second = _verify(fixture)
    assert first["bundle_file_sha256"] == second["bundle_file_sha256"]
    assert first["one_time_acceptance_proven"] is False
    assert second["one_time_acceptance_proven"] is False


def test_unsigned_acceptance_tamper_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["bundle"]["acceptance"]["acceptance_id"] = "tampered_acceptance_v1"
    _rewrite_bundle(fixture)
    with pytest.raises(replay.L2DReplayError, match="signature|acceptance|leaf"):
        _verify(fixture)


def test_duplicate_nonce_is_rejected_even_when_merkle_root_and_signatures_are_rebuilt(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    duplicate = copy.deepcopy(fixture["bundle"]["witnessed_prefix"]["entries"][1])
    duplicate["index"] = 2
    duplicate["body"]["acceptance_id"] = "second_acceptance_same_nonce_v1"
    duplicate["body"]["run_id"] = "second_run_same_nonce_v1"
    duplicate["body"]["authorization_id"] = "second_authorization_same_nonce_v1"
    duplicate["body"]["authorization_sha256"] = _digest(
        "second-authorization-same-semantic-nonce"
    )
    duplicate["leaf_sha256"] = replay.merkle_leaf_sha256(duplicate["body"])
    fixture["bundle"]["witnessed_prefix"]["entries"].append(duplicate)
    leaves = [
        entry["leaf_sha256"]
        for entry in fixture["bundle"]["witnessed_prefix"]["entries"]
    ]
    fixture["bundle"]["checkpoint"]["tree_size"] = 3
    fixture["bundle"]["checkpoint"]["root_sha256"] = replay.merkle_root_from_leaf_hashes(leaves)
    _rewrite_bundle(fixture, resign=True)
    with pytest.raises(replay.L2DReplayError, match="duplicate claimed nonce"):
        _verify(fixture)


def test_previous_checkpoint_must_match_registry_pinned_anchor(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["bundle"]["previous_checkpoint"]["root_sha256"] = _digest("fork")
    _rewrite_bundle(fixture, resign=True)
    with pytest.raises(replay.L2DReplayError, match="registry floor"):
        _verify(fixture)


def test_current_checkpoint_root_must_match_full_prefix(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["bundle"]["checkpoint"]["root_sha256"] = _digest("wrong-root")
    _rewrite_bundle(fixture, resign=True)
    with pytest.raises(replay.L2DReplayError, match="current checkpoint"):
        _verify(fixture)


def test_prefix_indices_must_be_contiguous_from_genesis(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["bundle"]["witnessed_prefix"]["entries"][1]["index"] = 7
    _rewrite_bundle(fixture, resign=True)
    with pytest.raises(replay.L2DReplayError, match="indices"):
        _verify(fixture)


def test_l2c_report_must_bind_the_actual_l2c_bundle(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    report_path = fixture["subject_files"]["l2c_verification_report_sha256"]
    report = json.loads(report_path.read_text())
    report["bundle_file_sha256"] = _digest("different-bundle")
    _write_json(report_path, report)
    fixture["bundle"]["subject"]["l2c_verification_report_sha256"] = hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    fixture["bundle"]["acceptance"]["l2c_verification_report_sha256"] = fixture[
        "bundle"
    ]["subject"]["l2c_verification_report_sha256"]
    fixture["bundle"]["witnessed_prefix"]["entries"][1]["body"] = copy.deepcopy(
        fixture["bundle"]["acceptance"]
    )
    fixture["bundle"]["witnessed_prefix"]["entries"][1]["leaf_sha256"] = replay.merkle_leaf_sha256(
        fixture["bundle"]["acceptance"]
    )
    leaves = [entry["leaf_sha256"] for entry in fixture["bundle"]["witnessed_prefix"]["entries"]]
    fixture["bundle"]["checkpoint"]["root_sha256"] = replay.merkle_root_from_leaf_hashes(leaves)
    _rewrite_bundle(fixture, resign=True)
    with pytest.raises(replay.L2DReplayError, match="different bundle bytes"):
        _verify(fixture)


def test_l2c_report_cannot_overstate_scientific_authorization(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    report_path = fixture["subject_files"]["l2c_verification_report_sha256"]
    report = json.loads(report_path.read_text())
    report["claim_authorizations"]["reconstruction"] = True
    _write_json(report_path, report)
    with pytest.raises(replay.L2DReplayError, match="subject digests|scientific claim"):
        _verify(fixture)


def test_forged_l2c_success_status_is_bound_but_never_authenticated(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    report_path = fixture["subject_files"]["l2c_verification_report_sha256"]
    report = json.loads(report_path.read_text())
    report["status"] = "FORGED_ALL_GATES_PASSED"
    report["verifier_controlled_registry_used"] = True
    report["event_chain_verified"] = True
    _write_json(report_path, report)
    _sync_subject_and_target(fixture)
    _rewrite_bundle(fixture, resign=True)

    result = _verify(fixture)
    assert result["l2c_report_bytes_bound_without_authentication"] is True
    assert result["l2c_report_authenticity_proven"] is False
    assert result["l2c_report_status_interpreted"] is False
    assert result["l2c_verifier_execution_reperformed"] is False


def test_nonce_is_derived_from_authorization_semantics_not_file_serialization(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    authorization_path = fixture["subject_files"]["authorization_sha256"]
    authorization = json.loads(authorization_path.read_text())
    semantic_digest = replay.semantic_nonce_sha256(authorization["one_time_nonce"])
    authorization_path.write_text(
        json.dumps(authorization, ensure_ascii=True, separators=(",", ":")),
        encoding="utf-8",
    )
    assert hashlib.sha256(authorization_path.read_bytes()).hexdigest() != fixture[
        "bundle"
    ]["acceptance"]["authorization_sha256"]
    _sync_subject_and_target(fixture)
    _rewrite_bundle(fixture, resign=True)

    result = _verify(fixture)
    assert result["target_semantic_nonce_sha256"] == semantic_digest
    assert result["semantic_nonce_derived_from_parsed_authorization"] is True


def test_duplicate_nonce_between_non_target_history_entries_is_rejected(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    historical = fixture["bundle"]["witnessed_prefix"]["entries"][0]
    duplicate = copy.deepcopy(historical)
    duplicate["body"]["acceptance_id"] = "historical_acceptance_entry_v2"
    duplicate["body"]["run_id"] = "historical_anonymous_run_v2"
    duplicate["body"]["authorization_id"] = "historical_authorization_id_v2"
    duplicate["body"]["authorization_sha256"] = _digest(
        "different-historical-authorization"
    )
    fixture["bundle"]["witnessed_prefix"]["entries"].insert(1, duplicate)
    _rebuild_current_checkpoint(fixture)
    _rewrite_bundle(fixture, resign=True)

    with pytest.raises(replay.L2DReplayError, match="duplicate claimed nonce"):
        _verify(fixture)


def test_two_branches_from_same_static_floor_can_each_verify_offline(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    first = _verify(fixture)
    first_root = first["current_root_sha256"]

    fixture["bundle"]["acceptance"]["acceptance_id"] = (
        "alternative_target_acceptance_v1"
    )
    fixture["bundle"]["witnessed_prefix"]["entries"][-1]["body"] = copy.deepcopy(
        fixture["bundle"]["acceptance"]
    )
    _rebuild_current_checkpoint(fixture)
    _rewrite_bundle(fixture, resign=True)
    second = _verify(fixture)

    assert second["current_root_sha256"] != first_root
    assert first["static_registry_floor_checkpoint_matched"] is True
    assert second["static_registry_floor_checkpoint_matched"] is True
    assert first["log_fork_freedom_proven"] is False
    assert second["log_fork_freedom_proven"] is False


def test_bundle_validity_must_be_fully_inside_policy_window(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["policy"]["valid_from_utc"] = _utc(
        fixture["now"] - timedelta(seconds=10)
    )
    _repin_policy(fixture)
    with pytest.raises(replay.L2DReplayError, match="outside the policy window"):
        _verify(fixture)


def test_actual_acceptance_to_checkpoint_delay_must_meet_mmd(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["bundle"]["issued_at_utc"] = _utc(
        fixture["now"] - timedelta(seconds=300)
    )
    recorded = _utc(fixture["now"] - timedelta(seconds=200))
    fixture["bundle"]["acceptance"]["recorded_at_utc"] = recorded
    fixture["bundle"]["witnessed_prefix"]["entries"][-1]["body"] = copy.deepcopy(
        fixture["bundle"]["acceptance"]
    )
    _rebuild_current_checkpoint(fixture)
    _rewrite_bundle(fixture, resign=True)
    with pytest.raises(
        replay.L2DReplayError,
        match="actual acceptance-to-checkpoint delay exceeds signed checkpoint MMD",
    ):
        _verify(fixture)


def test_actual_delay_must_meet_checkpoint_declared_mmd_not_only_policy(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["bundle"]["checkpoint"]["maximum_merge_delay_seconds"] = 1
    _rewrite_bundle(fixture, resign=True)
    with pytest.raises(
        replay.L2DReplayError,
        match="actual acceptance-to-checkpoint delay exceeds signed checkpoint MMD",
    ):
        _verify(fixture)


def test_monitor_signatures_cannot_be_replayed_across_policy_context(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["policy"]["policy_epoch"] = 2
    _repin_policy(fixture)
    policy_sha256 = hashlib.sha256(fixture["policy_path"].read_bytes()).hexdigest()
    registry_sha256 = hashlib.sha256(fixture["registry_path"].read_bytes()).hexdigest()
    fixture["bundle"]["subject"] = _subject(
        policy_sha256, fixture["subject_files"]
    )
    fixture["bundle"]["trust_context"] = replay._expected_trust_context(
        registry=fixture["registry"],
        registry_sha256=registry_sha256,
        policy=fixture["policy"],
        policy_sha256=policy_sha256,
        challenge_commitment_sha256=fixture["bundle"]["subject"][
            "challenge_commitment_sha256"
        ],
    )
    _write_json(fixture["bundle_path"], fixture["bundle"])

    with pytest.raises(replay.L2DReplayError, match="signed-payload|signature"):
        _verify(fixture)


def test_all_zero_enrollment_review_digest_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["registry"]["active_policy_pins"][0][
        "enrollment_review_sha256"
    ] = "0" * 64
    _write_json(fixture["registry_path"], fixture["registry"])
    with pytest.raises(replay.L2DReplayError, match="schema|enrollment review"):
        _verify(fixture)


def test_role_evidence_bytes_are_rehashed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["role_evidence_files"]["checkpoint_monitor_a"].write_text(
        "tampered evidence", encoding="utf-8"
    )
    with pytest.raises(replay.L2DReplayError, match="evidence digest"):
        _verify(fixture)


def test_one_role_evidence_artifact_cannot_satisfy_two_roles(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["bundle"]["role_evidence"][2]["evidence_artifact_sha256"] = fixture[
        "bundle"
    ]["role_evidence"][1]["evidence_artifact_sha256"]
    fixture["role_evidence_files"]["checkpoint_monitor_b"] = fixture[
        "role_evidence_files"
    ]["checkpoint_monitor_a"]
    _rewrite_bundle(fixture, resign=True)
    with pytest.raises(replay.L2DReplayError, match="reused across roles"):
        _verify(fixture)


def test_policy_cannot_reuse_operator_domain_labels(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["policy"]["trusted_keys"][2]["operator_domain_id"] = fixture[
        "policy"
    ]["trusted_keys"][1]["operator_domain_id"]
    _repin_policy(fixture)
    with pytest.raises(replay.L2DReplayError, match="operator-domain"):
        _verify(fixture)


def test_policy_cannot_reuse_one_public_key_across_roles(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["policy"]["trusted_keys"][2]["public_key_base64"] = fixture[
        "policy"
    ]["trusted_keys"][1]["public_key_base64"]
    _repin_policy(fixture)
    with pytest.raises(replay.L2DReplayError, match="reused across replay roles"):
        _verify(fixture)


def test_missing_checkpoint_monitor_signature_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["bundle"]["signatures"] = fixture["bundle"]["signatures"][:2]
    _rewrite_bundle(fixture)
    with pytest.raises(replay.L2DReplayError, match="schema|signatures"):
        _verify(fixture)


def test_stale_checkpoint_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["policy"]["max_checkpoint_age_seconds"] = 1
    _repin_policy(fixture)
    fixture["bundle"]["subject"]["trust_policy_sha256"] = hashlib.sha256(
        fixture["policy_path"].read_bytes()
    ).hexdigest()
    _rewrite_bundle(fixture, resign=True)
    with pytest.raises(replay.L2DReplayError, match="older than"):
        _verify(fixture)


def test_required_limitations_cannot_be_softened(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["bundle"]["limitations"][0] = "This prefix proves global uniqueness forever."
    _rewrite_bundle(fixture, resign=True)
    with pytest.raises(replay.L2DReplayError, match="weakens"):
        _verify(fixture)


def test_public_registry_bytes_are_compiled_and_caller_cannot_replace_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    public_registry = replay.ROOT / replay.REGISTRY_RELATIVE
    assert hashlib.sha256(public_registry.read_bytes()).hexdigest() == replay.COMPILED_REGISTRY_SHA256
    fixture = _fixture(tmp_path)
    fake_install = tmp_path / "fake-install"
    fake_registry = fake_install / replay.REGISTRY_RELATIVE
    fake_registry.parent.mkdir(parents=True)
    _write_json(fake_registry, fixture["registry"])
    monkeypatch.setattr(replay, "ROOT", fake_install)
    with pytest.raises(
        replay.L2DReplayError, match="L2D_VERIFIER_CONTROLLED_REGISTRY_BYTES_CHANGED"
    ):
        replay.verify_replay_role_bundle(
            fixture["bundle_path"],
            fixture["policy_path"],
            subject_files=fixture["subject_files"],
            role_evidence_files=fixture["role_evidence_files"],
        )


def test_public_cli_fails_before_reading_private_inputs_when_no_anchor_exists(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "does-not-exist"
    command = [
        sys.executable,
        str(replay.ROOT / "site_tools/n5_d5_l2d_replay_role_verifier.py"),
        "--bundle",
        str(missing / "bundle.json"),
        "--trust-policy",
        str(missing / "policy.json"),
    ]
    for key in replay.SUBJECT_KEYS:
        command.extend(
            [
                "--" + key.removesuffix("_sha256").replace("_", "-") + "-file",
                str(missing / f"{key}.bin"),
            ]
        )
    for role in replay.REQUIRED_ROLES:
        command.extend(
            [
                "--" + role.replace("_", "-") + "-evidence-file",
                str(missing / f"{role}.json"),
            ]
        )
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    report = json.loads(completed.stdout)
    assert completed.returncode == 2
    assert "NO_L2D_PRODUCTION_TRUST_ANCHOR_ENROLLED" in report["error"]
    assert report["one_time_acceptance_proven"] is False
    assert report["production_backend_authorized"] is False


def test_placeholder_bundle_is_schema_valid_but_operationally_rejected() -> None:
    bundle = replay.ROOT / replay.BUNDLE_PLACEHOLDER_RELATIVE
    policy = replay.ROOT / replay.POLICY_PLACEHOLDER_RELATIVE
    missing_files = {key: bundle for key in replay.SUBJECT_KEYS}
    evidence_files = {role: bundle for role in replay.REQUIRED_ROLES}
    with pytest.raises(replay.L2DReplayError, match="NO_L2D_PRODUCTION"):
        replay.verify_replay_role_bundle(
            bundle,
            policy,
            subject_files=missing_files,
            role_evidence_files=evidence_files,
        )
