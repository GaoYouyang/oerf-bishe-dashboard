#!/usr/bin/env python3
"""Verify an N5-D5 L2-D0 replay-prefix and role-evidence capsule.

The verifier recomputes a caller-supplied RFC 9162-style Merkle prefix from
genesis, checks a registry-pinned floor checkpoint, verifies one sequencer and
two checkpoint-monitor signatures, parses the L2-B authorization to derive a
domain-separated semantic nonce digest, and rehashes role-evidence files.  It
has no signing path and accepts no caller-selected production trust registry.

A passing report proves uniqueness only inside the supplied witnessed prefix.
Without an online atomic consume service plus cross-observer gossip it cannot
prove global one-time acceptance, fork freedom, or operational independence.
It never authorizes production execution or a scientific claim.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timedelta, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import site_tools.n5_d5_l2c_external_witness as base


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_RELATIVE = "data_templates/n5_d5_l2d_trust_anchor_registry.json"
REGISTRY_SCHEMA_RELATIVE = (
    "data_templates/n5_d5_l2d_trust_anchor_registry.schema.json"
)
POLICY_SCHEMA_RELATIVE = (
    "data_templates/n5_d5_l2d_replay_role_policy.schema.json"
)
BUNDLE_SCHEMA_RELATIVE = (
    "data_templates/n5_d5_l2d_replay_role_bundle.schema.json"
)
POLICY_PLACEHOLDER_RELATIVE = (
    "data_templates/n5_d5_l2d_replay_role_policy.placeholder.json"
)
AUTHORIZATION_SCHEMA_RELATIVE = (
    "data_templates/n5_d5_l2b_describe_authorization.schema.json"
)
BUNDLE_PLACEHOLDER_RELATIVE = (
    "data_templates/n5_d5_l2d_replay_role_bundle.placeholder.json"
)
COMPILED_REGISTRY_SHA256 = (
    "5459afd7f71fe1668d28405f5b49d57f08d1bfca04d785ab28f78f15ffbae1b7"
)
PREDICATE_TYPE = (
    "https://gaoyouyang.github.io/oerf-bishe-dashboard/"
    "attestations/n5-d5-l2d-replay-role/v1"
)
REQUIRED_ROLES = (
    "acceptance_sequencer",
    "checkpoint_monitor_a",
    "checkpoint_monitor_b",
)
SUBJECT_KEYS = (
    "authorization_sha256",
    "l2c_bundle_sha256",
    "l2c_verification_report_sha256",
    "challenge_commitment_sha256",
    "nonce_commitment_sha256",
)
CLAIM_KEYS = base.CLAIM_KEYS
EMPTY_MERKLE_ROOT_SHA256 = hashlib.sha256(b"").hexdigest()
ALLOWED_CLOCK_SKEW = timedelta(seconds=30)
MAX_POLICY_LIFETIME = timedelta(days=370)
PLACEHOLDER_MARKERS = ("replace_", "REPLACE_ME")
ROLE_SIGNATURE_DOMAINS = {
    "acceptance_sequencer": b"OERF:N5-D5:L2-D:ACCEPTANCE-SEQUENCER:V1\x00",
    "checkpoint_monitor_a": b"OERF:N5-D5:L2-D:CHECKPOINT-MONITOR-A:V1\x00",
    "checkpoint_monitor_b": b"OERF:N5-D5:L2-D:CHECKPOINT-MONITOR-B:V1\x00",
}
NONCE_SEMANTIC_DOMAIN = b"OERF:N5-D5:L2-B:ONE-TIME-NONCE:V1\x00"
NONCE_COMMITMENT_DOMAIN_LABEL = "OERF:N5-D5:L2-B:ONE-TIME-NONCE:V1"
REQUIRED_LIMITATIONS = (
    "A recomputed caller-supplied prefix proves uniqueness only inside that supplied signed prefix; external completeness is not proven.",
    "Checkpoint signatures do not rule out a split-view or fork without online gossip across independent observers.",
    "Distinct keys and operator-domain labels do not prove that signer services or human operators are independent.",
    "The bound L2-C report is treated as unauthenticated bytes; its status is not interpreted and its verifier is not re-executed here.",
    "A same-UID verifier and registry cannot issue an authoritative production authorization credential.",
    "This capsule never authorizes real BOST, reconstruction, model training, superiority, generalization, or publication claims.",
)


class L2DReplayError(RuntimeError):
    """Fail-closed replay-prefix or role-evidence verification error."""


def canonical_json(value: object) -> str:
    return base.canonical_json(value)


def sha256_bytes(value: bytes) -> str:
    return base.sha256_bytes(value)


def semantic_nonce_sha256(one_time_nonce: str) -> str:
    """Hash the parsed nonce value, not its containing file serialization."""

    return sha256_bytes(NONCE_SEMANTIC_DOMAIN + one_time_nonce.encode("ascii"))


def merkle_leaf_sha256(body: dict[str, Any]) -> str:
    """RFC 9162 leaf hash: SHA-256(0x00 || canonical leaf bytes)."""

    return sha256_bytes(b"\x00" + canonical_json(body).encode("utf-8"))


def _largest_power_of_two_less_than(value: int) -> int:
    if value <= 1:
        raise ValueError("value must exceed one")
    return 1 << ((value - 1).bit_length() - 1)


def merkle_root_from_leaf_hashes(leaf_hashes: Sequence[str]) -> str:
    """Recompute RFC 9162 MTH from already domain-separated leaf hashes."""

    if not leaf_hashes:
        return EMPTY_MERKLE_ROOT_SHA256
    for value in leaf_hashes:
        if len(value) != 64:
            raise L2DReplayError("ledger leaf digest has invalid length")
        try:
            bytes.fromhex(value)
        except ValueError as exc:
            raise L2DReplayError("ledger leaf digest is not hexadecimal") from exc
    if len(leaf_hashes) == 1:
        return leaf_hashes[0]
    split = _largest_power_of_two_less_than(len(leaf_hashes))
    left = bytes.fromhex(merkle_root_from_leaf_hashes(leaf_hashes[:split]))
    right = bytes.fromhex(merkle_root_from_leaf_hashes(leaf_hashes[split:]))
    return sha256_bytes(b"\x01" + left + right)


def role_signed_payload_bytes(bundle: dict[str, Any], role: str) -> bytes:
    if role not in REQUIRED_ROLES:
        raise L2DReplayError(f"unsupported signature role: {role}")
    common = {
        "schema_version": bundle["schema_version"],
        "record_kind": bundle["record_kind"],
        "predicate_type": bundle["predicate_type"],
        "run_id": bundle["run_id"],
        "issued_at_utc": bundle["issued_at_utc"],
        "expires_at_utc": bundle["expires_at_utc"],
        "trust_context": bundle["trust_context"],
        "subject": bundle["subject"],
        "previous_checkpoint": bundle["previous_checkpoint"],
        "checkpoint": bundle["checkpoint"],
        "terminal_status": bundle["terminal_status"],
        "claim_authorizations": bundle["claim_authorizations"],
        "limitations": bundle["limitations"],
    }
    if role == "acceptance_sequencer":
        scoped = {
            **common,
            "payload_scope": "ACCEPTANCE_SUBJECT_AND_CHECKPOINT",
            "acceptance": bundle["acceptance"],
        }
    else:
        scoped = {
            **common,
            "payload_scope": "CHECKPOINT_PREFIX_AND_ROLE_EVIDENCE",
            "witnessed_prefix": bundle["witnessed_prefix"],
            "role_evidence": bundle["role_evidence"],
        }
    return ROLE_SIGNATURE_DOMAINS[role] + canonical_json(scoped).encode("utf-8")


def _all_claims_false(value: object) -> bool:
    return base._all_claims_false(value)


def _snapshot_json(path: Path) -> tuple[dict[str, Any], str]:
    payload, digest = base._snapshot_regular_file(path)
    return base._load_json_bytes(payload), digest


def _validate(schema_relative: str, value: object, label: str) -> None:
    schema = base._load_schema(ROOT / schema_relative)
    try:
        base._validate_schema(schema, value, label)
    except base.L2CWitnessError as exc:
        raise L2DReplayError(str(exc)) from exc


def _parse_utc(value: str, label: str) -> datetime:
    try:
        return base._parse_utc(value, label)
    except base.L2CWitnessError as exc:
        raise L2DReplayError(str(exc)) from exc


def _decode_public_key(value: str) -> bytes:
    try:
        raw = base64.b64decode(value, validate=True)
    except Exception as exc:
        raise L2DReplayError("trusted Ed25519 public key is not valid base64") from exc
    if len(raw) != 32:
        raise L2DReplayError("trusted Ed25519 public key has wrong length")
    if raw == b"\x00" * 32:
        raise L2DReplayError("all-zero trusted Ed25519 public key is forbidden")
    return raw


def _verify_registry_and_policy(
    policy: dict[str, Any],
    *,
    policy_sha256: str,
    registry: dict[str, Any],
    now: datetime,
) -> dict[str, dict[str, Any]]:
    if not _all_claims_false(registry.get("claim_authorizations")):
        raise L2DReplayError("replay trust registry authorizes a scientific claim")
    if not _all_claims_false(policy.get("claim_authorizations")):
        raise L2DReplayError("replay policy authorizes a scientific claim")
    valid_from = _parse_utc(policy["valid_from_utc"], "policy valid_from_utc")
    valid_until = _parse_utc(policy["valid_until_utc"], "policy valid_until_utc")
    if valid_until <= valid_from:
        raise L2DReplayError("replay policy validity interval is empty")
    if valid_until - valid_from > MAX_POLICY_LIFETIME:
        raise L2DReplayError("replay policy validity interval is too long")
    if now + ALLOWED_CLOCK_SKEW < valid_from or now - ALLOWED_CLOCK_SKEW > valid_until:
        raise L2DReplayError("replay policy is not currently valid")

    matching_pins = [
        item
        for item in registry["active_policy_pins"]
        if item["policy_id"] == policy["policy_id"]
        and item["policy_epoch"] == policy["policy_epoch"]
    ]
    if len(matching_pins) != 1:
        raise L2DReplayError("replay policy is not uniquely enrolled")
    pin = matching_pins[0]
    if pin["policy_sha256"] != policy_sha256:
        raise L2DReplayError("replay policy digest does not match registry pin")
    if pin["predicate_type"] != policy["expected_predicate_type"]:
        raise L2DReplayError("replay predicate type does not match registry pin")
    if tuple(pin["required_signature_roles"]) != REQUIRED_ROLES:
        raise L2DReplayError("registry role set is not the required three-role set")
    if pin["log_epoch"] != policy["log_epoch"]:
        raise L2DReplayError("policy log epoch differs from registry pin")
    if pin["floor_checkpoint"] != policy["floor_checkpoint"]:
        raise L2DReplayError("policy floor checkpoint differs from registry pin")
    if pin["enrollment_review_sha256"] == "0" * 64:
        raise L2DReplayError("all-zero enrollment review digest is forbidden")

    trusted: dict[str, dict[str, Any]] = {}
    roles: set[str] = set()
    raw_keys: set[bytes] = set()
    operator_domains: set[str] = set()
    service_identities: set[str] = set()
    for entry in policy["trusted_keys"]:
        role = entry["role"]
        if role in roles:
            raise L2DReplayError("replay policy duplicates a signature role")
        roles.add(role)
        key_id = entry["key_id"]
        if key_id in trusted:
            raise L2DReplayError("replay policy duplicates a key ID")
        raw_key = _decode_public_key(entry["public_key_base64"])
        if raw_key in raw_keys:
            raise L2DReplayError("one Ed25519 key is reused across replay roles")
        raw_keys.add(raw_key)
        domain = entry["operator_domain_id"]
        if domain in operator_domains:
            raise L2DReplayError("replay policy reuses an operator-domain label")
        operator_domains.add(domain)
        identity = entry["service_identity_uri"]
        if identity in service_identities:
            raise L2DReplayError("replay policy reuses a service identity")
        service_identities.add(identity)
        trusted[key_id] = {**entry, "raw_public_key": raw_key}
    if roles != set(REQUIRED_ROLES) or len(trusted) != 3:
        raise L2DReplayError("replay policy does not contain exactly three roles")
    if policy["signature_threshold"] != 3:
        raise L2DReplayError("replay signature threshold is not three")
    return trusted


def _verify_bundle_times(
    bundle: dict[str, Any], policy: dict[str, Any], *, now: datetime
) -> None:
    issued = _parse_utc(bundle["issued_at_utc"], "bundle issued_at_utc")
    expires = _parse_utc(bundle["expires_at_utc"], "bundle expires_at_utc")
    if expires <= issued:
        raise L2DReplayError("replay bundle validity interval is empty")
    if (expires - issued).total_seconds() > policy["max_bundle_lifetime_seconds"]:
        raise L2DReplayError("replay bundle lifetime exceeds policy")
    if issued > now + ALLOWED_CLOCK_SKEW:
        raise L2DReplayError("replay bundle is issued in the future")
    if expires < now - ALLOWED_CLOCK_SKEW:
        raise L2DReplayError("replay bundle is expired")
    policy_start = _parse_utc(policy["valid_from_utc"], "policy valid_from_utc")
    policy_end = _parse_utc(policy["valid_until_utc"], "policy valid_until_utc")
    if issued < policy_start or expires > policy_end:
        raise L2DReplayError("replay bundle validity is outside the policy window")

    previous_time = _parse_utc(
        bundle["previous_checkpoint"]["issued_at_utc"],
        "previous checkpoint issued_at_utc",
    )
    acceptance_time = _parse_utc(
        bundle["acceptance"]["recorded_at_utc"], "acceptance recorded_at_utc"
    )
    checkpoint_time = _parse_utc(
        bundle["checkpoint"]["issued_at_utc"], "checkpoint issued_at_utc"
    )
    if not (previous_time <= acceptance_time <= checkpoint_time):
        raise L2DReplayError("checkpoint and acceptance timestamps are not ordered")
    if acceptance_time < issued - ALLOWED_CLOCK_SKEW:
        raise L2DReplayError("acceptance predates replay bundle issuance")
    if checkpoint_time > expires + ALLOWED_CLOCK_SKEW:
        raise L2DReplayError("checkpoint postdates replay bundle expiry")
    if checkpoint_time > now + ALLOWED_CLOCK_SKEW:
        raise L2DReplayError("checkpoint is issued in the future")
    if now - checkpoint_time > timedelta(seconds=policy["max_checkpoint_age_seconds"]):
        raise L2DReplayError("checkpoint is older than the policy freshness window")
    if bundle["checkpoint"]["maximum_merge_delay_seconds"] > policy["max_merge_delay_seconds"]:
        raise L2DReplayError("checkpoint merge delay exceeds replay policy")
    if checkpoint_time - acceptance_time > timedelta(
        seconds=bundle["checkpoint"]["maximum_merge_delay_seconds"]
    ):
        raise L2DReplayError(
            "actual acceptance-to-checkpoint delay exceeds signed checkpoint MMD"
        )


def _snapshot_subject_files(
    subject_files: dict[str, Path]
) -> tuple[dict[str, str], dict[str, bytes]]:
    if set(subject_files) != set(SUBJECT_KEYS):
        raise L2DReplayError("subject file mapping is incomplete or has extra keys")
    digests: dict[str, str] = {}
    payloads: dict[str, bytes] = {}
    for key in SUBJECT_KEYS:
        try:
            payload, digest = base._snapshot_regular_file(subject_files[key])
        except base.L2CWitnessError as exc:
            raise L2DReplayError(str(exc)) from exc
        digests[key] = digest
        payloads[key] = payload
    return digests, payloads


def _bind_untrusted_l2c_inputs(
    payloads: dict[str, bytes], digests: dict[str, str]
) -> str:
    try:
        l2c_bundle = base._load_json_bytes(payloads["l2c_bundle_sha256"])
        l2c_report = base._load_json_bytes(
            payloads["l2c_verification_report_sha256"]
        )
    except base.L2CWitnessError as exc:
        raise L2DReplayError(str(exc)) from exc
    if not _all_claims_false(l2c_bundle.get("claim_authorizations")):
        raise L2DReplayError("bound L2-C bundle authorizes a scientific claim")
    if not _all_claims_false(l2c_report.get("claim_authorizations")):
        raise L2DReplayError("bound L2-C report authorizes a scientific claim")
    required_false = (
        "role_operational_independence_proven",
        "replay_ledger_checked",
        "one_time_acceptance_proven",
        "production_backend_authorized",
        "formal_replay_authorized",
        "model_training_authorized",
    )
    if any(l2c_report.get(key) is True for key in required_false):
        raise L2DReplayError("bound L2-C report overstates authorization or replay proof")
    if l2c_report.get("bundle_file_sha256") != digests["l2c_bundle_sha256"]:
        raise L2DReplayError("bound L2-C report references different bundle bytes")
    event_chain = l2c_bundle.get("event_chain")
    if not isinstance(event_chain, dict):
        raise L2DReplayError("bound L2-C bundle lacks an event chain")
    final_event = event_chain.get("final_event_sha256")
    if not isinstance(final_event, str) or len(final_event) != 64:
        raise L2DReplayError("bound L2-C final event digest is invalid")
    if l2c_report.get("final_event_sha256") != final_event:
        raise L2DReplayError("bound L2-C report and bundle disagree on final event")
    return final_event


def _verify_subject(
    bundle: dict[str, Any],
    *,
    policy_sha256: str,
    subject_files: dict[str, Path],
) -> tuple[dict[str, str], str, str, str]:
    digests, payloads = _snapshot_subject_files(subject_files)
    expected = {**digests, "trust_policy_sha256": policy_sha256}
    if bundle["subject"] != expected:
        raise L2DReplayError("replay subject digests do not match snapshotted files")
    final_event = _bind_untrusted_l2c_inputs(payloads, digests)
    try:
        authorization = base._load_json_bytes(payloads["authorization_sha256"])
    except base.L2CWitnessError as exc:
        raise L2DReplayError(str(exc)) from exc
    _validate(AUTHORIZATION_SCHEMA_RELATIVE, authorization, "L2-B authorization")
    if not _all_claims_false(authorization.get("claim_authorizations")):
        raise L2DReplayError("bound authorization authorizes a scientific claim")
    nonce_digest = semantic_nonce_sha256(authorization["one_time_nonce"])
    try:
        nonce_commitment = base._load_json_bytes(payloads["nonce_commitment_sha256"])
    except base.L2CWitnessError as exc:
        raise L2DReplayError(str(exc)) from exc
    expected_commitment = {
        "schema_version": "n5-d5-l2d-semantic-nonce-commitment-1.0",
        "authorization_id": authorization["authorization_id"],
        "domain": NONCE_COMMITMENT_DOMAIN_LABEL,
        "semantic_nonce_sha256": nonce_digest,
    }
    if nonce_commitment != expected_commitment:
        raise L2DReplayError("nonce commitment does not match parsed authorization nonce")
    acceptance = bundle["acceptance"]
    if acceptance["authorization_sha256"] != expected["authorization_sha256"]:
        raise L2DReplayError("acceptance does not bind authorization bytes")
    if acceptance["l2c_bundle_sha256"] != expected["l2c_bundle_sha256"]:
        raise L2DReplayError("acceptance does not bind L2-C bundle bytes")
    if acceptance["l2c_verification_report_sha256"] != expected[
        "l2c_verification_report_sha256"
    ]:
        raise L2DReplayError("acceptance does not bind L2-C report bytes")
    if acceptance["authorization_id"] != authorization["authorization_id"]:
        raise L2DReplayError("acceptance authorization ID differs from parsed authorization")
    if acceptance["nonce_sha256"] != nonce_digest:
        raise L2DReplayError("acceptance nonce differs from parsed authorization nonce")
    if acceptance["l2c_final_event_sha256"] != final_event:
        raise L2DReplayError("acceptance does not bind L2-C final event")
    if acceptance["run_id"] != bundle["run_id"]:
        raise L2DReplayError("acceptance run ID differs from capsule run ID")
    return expected, final_event, nonce_digest, authorization["authorization_id"]


def _expected_trust_context(
    *,
    registry: dict[str, Any],
    registry_sha256: str,
    policy: dict[str, Any],
    policy_sha256: str,
    challenge_commitment_sha256: str,
) -> dict[str, Any]:
    return {
        "registry_id": registry["registry_id"],
        "registry_sha256": registry_sha256,
        "policy_id": policy["policy_id"],
        "policy_epoch": policy["policy_epoch"],
        "policy_sha256": policy_sha256,
        "predicate_type": policy["expected_predicate_type"],
        "ledger_namespace": policy["ledger_namespace"],
        "log_id": policy["log_id"],
        "log_epoch": policy["log_epoch"],
        "challenge_commitment_sha256": challenge_commitment_sha256,
    }


def _verify_prefix(
    bundle: dict[str, Any], policy: dict[str, Any]
) -> dict[str, Any]:
    prefix = bundle["witnessed_prefix"]
    entries = prefix["entries"]
    if len(entries) > policy["max_witnessed_prefix_entries"]:
        raise L2DReplayError("witnessed prefix exceeds policy entry bound")
    leaf_hashes: list[str] = []
    for expected_index, entry in enumerate(entries):
        if entry["index"] != expected_index:
            raise L2DReplayError("witnessed prefix indices are not contiguous from zero")
        body = entry["body"]
        if body["ledger_namespace"] != policy["ledger_namespace"]:
            raise L2DReplayError("ledger entry uses an unexpected namespace")
        computed_leaf = merkle_leaf_sha256(body)
        if entry["leaf_sha256"] != computed_leaf:
            raise L2DReplayError("ledger leaf digest does not match canonical body")
        leaf_hashes.append(computed_leaf)

    previous = bundle["previous_checkpoint"]
    current = bundle["checkpoint"]
    if previous["log_id"] != policy["log_id"] or current["log_id"] != policy["log_id"]:
        raise L2DReplayError("checkpoint log ID differs from replay policy")
    anchor = policy["floor_checkpoint"]
    if previous["tree_size"] != anchor["tree_size"] or previous[
        "root_sha256"
    ] != anchor["root_sha256"]:
        raise L2DReplayError("previous checkpoint does not match registry floor")
    if previous["tree_size"] > len(entries):
        raise L2DReplayError("previous checkpoint exceeds witnessed prefix")
    previous_root = merkle_root_from_leaf_hashes(
        leaf_hashes[: previous["tree_size"]]
    )
    if previous_root != previous["root_sha256"]:
        raise L2DReplayError("witnessed prefix does not reproduce previous checkpoint")
    if current["tree_size"] != len(entries):
        raise L2DReplayError("current checkpoint size differs from witnessed prefix")
    if current["tree_size"] <= previous["tree_size"]:
        raise L2DReplayError("current checkpoint does not extend registry floor")
    current_root = merkle_root_from_leaf_hashes(leaf_hashes)
    if current_root != current["root_sha256"]:
        raise L2DReplayError("witnessed prefix does not reproduce current checkpoint")

    acceptance_ids = [entry["body"]["acceptance_id"] for entry in entries]
    authorization_ids = [entry["body"]["authorization_id"] for entry in entries]
    nonce_digests = [entry["body"]["nonce_sha256"] for entry in entries]
    authorization_digests = [
        entry["body"]["authorization_sha256"] for entry in entries
    ]
    for label, values in (
        ("acceptance ID", acceptance_ids),
        ("authorization ID", authorization_ids),
        ("claimed nonce digest", nonce_digests),
        ("authorization digest", authorization_digests),
    ):
        if len(values) != len(set(values)):
            raise L2DReplayError(
                f"witnessed prefix contains a duplicate {label}"
            )

    target = bundle["acceptance"]
    exact_count = sum(entry["body"] == target for entry in entries)
    acceptance_id_count = sum(
        entry["body"]["acceptance_id"] == target["acceptance_id"]
        for entry in entries
    )
    nonce_count = sum(
        entry["body"]["ledger_namespace"] == target["ledger_namespace"]
        and entry["body"]["nonce_sha256"] == target["nonce_sha256"]
        for entry in entries
    )
    authorization_count = sum(
        entry["body"]["ledger_namespace"] == target["ledger_namespace"]
        and entry["body"]["authorization_sha256"]
        == target["authorization_sha256"]
        for entry in entries
    )
    if exact_count != 1 or acceptance_id_count != 1:
        raise L2DReplayError("target acceptance is not unique in witnessed prefix")
    if nonce_count != 1:
        raise L2DReplayError("target nonce is not unique in witnessed prefix")
    if authorization_count != 1:
        raise L2DReplayError("target authorization is not unique in witnessed prefix")
    return {
        "entry_count": len(entries),
        "previous_tree_size": previous["tree_size"],
        "current_tree_size": current["tree_size"],
        "current_root_sha256": current_root,
        "target_acceptance_id": target["acceptance_id"],
        "target_nonce_occurrences": nonce_count,
        "all_acceptance_ids_unique": True,
        "all_authorization_ids_unique": True,
        "all_claimed_nonce_digest_values_unique": True,
        "all_authorization_digests_unique": True,
    }


def _verify_role_evidence(
    bundle: dict[str, Any],
    trusted: dict[str, dict[str, Any]],
    *,
    role_evidence_files: dict[str, Path],
) -> dict[str, str]:
    if set(role_evidence_files) != set(REQUIRED_ROLES):
        raise L2DReplayError("role-evidence file mapping is incomplete or has extra roles")
    evidence_by_role: dict[str, dict[str, Any]] = {}
    for item in bundle["role_evidence"]:
        role = item["role"]
        if role in evidence_by_role:
            raise L2DReplayError("role evidence duplicates one signature role")
        evidence_by_role[role] = item
    if set(evidence_by_role) != set(REQUIRED_ROLES):
        raise L2DReplayError("role evidence does not cover all required roles")

    digests: dict[str, str] = {}
    seen_digests: set[str] = set()
    for role in REQUIRED_ROLES:
        item = evidence_by_role[role]
        trusted_entry = next(
            (entry for entry in trusted.values() if entry["role"] == role), None
        )
        if trusted_entry is None:
            raise L2DReplayError("role evidence has no trusted policy entry")
        for field in (
            "key_id",
            "operator_domain_id",
            "service_identity_uri",
            "credential_issuer_uri",
        ):
            if item[field] != trusted_entry[field]:
                raise L2DReplayError(f"role evidence {field} differs from policy")
        try:
            _payload, digest = base._snapshot_regular_file(role_evidence_files[role])
        except base.L2CWitnessError as exc:
            raise L2DReplayError(str(exc)) from exc
        if digest != item["evidence_artifact_sha256"]:
            raise L2DReplayError("role evidence digest does not match private file")
        if digest in seen_digests:
            raise L2DReplayError("one evidence artifact is reused across roles")
        seen_digests.add(digest)
        digests[role] = digest
    return digests


def _verify_signatures(
    bundle: dict[str, Any], trusted: dict[str, dict[str, Any]]
) -> tuple[list[str], dict[str, str]]:
    seen_roles: set[str] = set()
    seen_keys: set[str] = set()
    payload_digests: dict[str, str] = {}
    for signature in bundle["signatures"]:
        role = signature["role"]
        key_id = signature["key_id"]
        if role in seen_roles or key_id in seen_keys:
            raise L2DReplayError("replay signatures duplicate a role or key")
        seen_roles.add(role)
        seen_keys.add(key_id)
        trusted_entry = trusted.get(key_id)
        if trusted_entry is None or trusted_entry["role"] != role:
            raise L2DReplayError("replay signature uses an untrusted role/key pair")
        expected_scope = (
            "ACCEPTANCE_SUBJECT_AND_CHECKPOINT"
            if role == "acceptance_sequencer"
            else "CHECKPOINT_PREFIX_AND_ROLE_EVIDENCE"
        )
        if signature["payload_scope"] != expected_scope:
            raise L2DReplayError("replay signature payload scope is wrong")
        message = role_signed_payload_bytes(bundle, role)
        digest = sha256_bytes(message)
        if signature["signed_payload_sha256"] != digest:
            raise L2DReplayError("replay signed-payload digest is wrong")
        try:
            raw_signature = base64.b64decode(
                signature["signature_base64"], validate=True
            )
        except Exception as exc:
            raise L2DReplayError("replay signature is not valid base64") from exc
        if len(raw_signature) != 64:
            raise L2DReplayError("replay signature has wrong length")
        try:
            Ed25519PublicKey.from_public_bytes(
                trusted_entry["raw_public_key"]
            ).verify(raw_signature, message)
        except (InvalidSignature, ValueError) as exc:
            raise L2DReplayError("replay Ed25519 signature verification failed") from exc
        payload_digests[role] = digest
    if seen_roles != set(REQUIRED_ROLES):
        raise L2DReplayError("all three replay signature roles are required")
    return sorted(seen_roles), payload_digests


def _verify_with_registry(
    bundle_path: Path,
    trust_policy_path: Path,
    *,
    subject_files: dict[str, Path],
    role_evidence_files: dict[str, Path],
    trust_anchor_registry_path: Path,
) -> dict[str, Any]:
    production_registry = Path(os.path.abspath(trust_anchor_registry_path)) == Path(
        os.path.abspath(ROOT / REGISTRY_RELATIVE)
    )
    registry, registry_sha256 = _snapshot_json(trust_anchor_registry_path)
    if production_registry and registry_sha256 != COMPILED_REGISTRY_SHA256:
        raise L2DReplayError("L2D_VERIFIER_CONTROLLED_REGISTRY_BYTES_CHANGED")
    _validate(REGISTRY_SCHEMA_RELATIVE, registry, "L2-D trust registry")
    if not registry["production_trust_anchor_enrolled"]:
        raise L2DReplayError("NO_L2D_PRODUCTION_TRUST_ANCHOR_ENROLLED")

    policy, policy_sha256 = _snapshot_json(trust_policy_path)
    bundle, bundle_sha256 = _snapshot_json(bundle_path)
    _validate(POLICY_SCHEMA_RELATIVE, policy, "L2-D replay policy")
    _validate(BUNDLE_SCHEMA_RELATIVE, bundle, "L2-D replay bundle")
    if any(marker in canonical_json(policy) for marker in PLACEHOLDER_MARKERS):
        raise L2DReplayError("L2-D replay policy still contains placeholder markers")
    if any(marker in canonical_json(bundle) for marker in PLACEHOLDER_MARKERS):
        raise L2DReplayError("L2-D replay bundle still contains placeholder markers")
    if not _all_claims_false(bundle.get("claim_authorizations")):
        raise L2DReplayError("L2-D replay bundle authorizes a scientific claim")
    if not all(item in bundle["limitations"] for item in REQUIRED_LIMITATIONS):
        raise L2DReplayError("L2-D replay bundle weakens a required limitation")
    if bundle["predicate_type"] != policy["expected_predicate_type"]:
        raise L2DReplayError("L2-D bundle predicate differs from replay policy")
    if bundle["acceptance"]["ledger_namespace"] != policy["ledger_namespace"]:
        raise L2DReplayError("L2-D acceptance namespace differs from policy")

    now = datetime.now(timezone.utc)
    trusted = _verify_registry_and_policy(
        policy,
        policy_sha256=policy_sha256,
        registry=registry,
        now=now,
    )
    _verify_bundle_times(bundle, policy, now=now)
    subject, final_event, semantic_nonce_digest, authorization_id = _verify_subject(
        bundle, policy_sha256=policy_sha256, subject_files=subject_files
    )
    expected_trust_context = _expected_trust_context(
        registry=registry,
        registry_sha256=registry_sha256,
        policy=policy,
        policy_sha256=policy_sha256,
        challenge_commitment_sha256=subject["challenge_commitment_sha256"],
    )
    if bundle["trust_context"] != expected_trust_context:
        raise L2DReplayError("signed trust context does not match registry and policy")
    prefix = _verify_prefix(bundle, policy)
    role_evidence_digests = _verify_role_evidence(
        bundle, trusted, role_evidence_files=role_evidence_files
    )
    roles, payload_digests = _verify_signatures(bundle, trusted)

    return {
        "schema_version": "n5-d5-l2d-replay-role-verification-report-1.0",
        "status": (
            "L2D_PREFIX_UNIQUENESS_AND_ROLE_EVIDENCE_STRUCTURALLY_VERIFIED_"
            "ONLINE_ATOMICITY_GOSSIP_AND_INDEPENDENCE_REQUIRED"
            if production_registry
            else "L2D_DEVELOPMENT_REGISTRY_PREFIX_AND_ROLE_EVIDENCE_VERIFIED"
        ),
        "bundle_file_sha256": bundle_sha256,
        "trust_policy_file_sha256": policy_sha256,
        "trust_anchor_registry_file_sha256": registry_sha256,
        "verifier_controlled_registry_used": production_registry,
        "enrolled_policy_digest_matched": True,
        "static_registry_floor_checkpoint_matched": True,
        "anti_rollback_protection_proven": False,
        "caller_supplied_prefix_declared_complete_from_genesis": True,
        "prefix_completeness_external_to_bundle_proven": False,
        "supplied_prefix_recomputed_from_genesis_index_zero": True,
        "rfc9162_style_merkle_root_recomputed": True,
        "witnessed_prefix_entry_count": prefix["entry_count"],
        "previous_tree_size": prefix["previous_tree_size"],
        "current_tree_size": prefix["current_tree_size"],
        "current_root_sha256": prefix["current_root_sha256"],
        "target_acceptance_id": prefix["target_acceptance_id"],
        "target_semantic_nonce_digest_occurrences_in_supplied_prefix": prefix[
            "target_nonce_occurrences"
        ],
        "target_authorization_id": authorization_id,
        "target_semantic_nonce_sha256": semantic_nonce_digest,
        "target_semantic_nonce_unique_in_supplied_signed_prefix": True,
        "all_acceptance_ids_unique_in_supplied_signed_prefix": prefix[
            "all_acceptance_ids_unique"
        ],
        "all_authorization_ids_unique_in_supplied_signed_prefix": prefix[
            "all_authorization_ids_unique"
        ],
        "all_claimed_nonce_digest_values_unique_in_supplied_signed_prefix": prefix[
            "all_claimed_nonce_digest_values_unique"
        ],
        "all_authorization_digests_unique_in_supplied_signed_prefix": prefix[
            "all_authorization_digests_unique"
        ],
        "authorization_schema_validated": True,
        "semantic_nonce_derived_from_parsed_authorization": True,
        "nonce_commitment_semantics_validated": True,
        "l2c_bundle_and_report_bytes_rehashed": True,
        "l2c_final_event_bound": final_event,
        "l2c_report_bytes_bound_without_authentication": True,
        "l2c_report_authenticity_proven": False,
        "l2c_report_status_interpreted": False,
        "l2c_verifier_execution_reperformed": False,
        "role_evidence_files_rehashed": True,
        "role_evidence_file_sha256": role_evidence_digests,
        "trusted_signature_count": len(roles),
        "trusted_signature_roles": roles,
        "role_scoped_signed_payload_sha256": payload_digests,
        "role_key_separation_verified": True,
        "policy_operator_domain_labels_distinct": True,
        "policy_service_identity_uris_distinct": True,
        "role_operational_independence_proven": False,
        "verifier_binary_integrity_proven": False,
        "same_uid_trust_root_replacement_excluded": False,
        "authoritative_production_authorization_credential_issued": False,
        "online_atomic_consume_observed": False,
        "latest_checkpoint_queried_online": False,
        "cross_observer_gossip_performed": False,
        "global_log_view_consistency_proven": False,
        "log_fork_freedom_proven": False,
        "ledger_service_atomic_first_write_proven": False,
        "ledger_state_machine_transitions_verified": False,
        "replay_ledger_prefix_checked": True,
        "one_time_acceptance_proven": False,
        "one_time_acceptance_scope": (
            "SEMANTIC_NONCE_UNIQUE_ONLY_WITHIN_CALLER_SUPPLIED_SIGNED_PREFIX"
        ),
        "production_backend_authorized": False,
        "formal_replay_authorized": False,
        "model_training_authorized": False,
        "claim_authorizations": {key: False for key in CLAIM_KEYS},
        "next_gate": (
            "online shared-state atomic consume, persisted prior checkpoint, "
            "independent witness gossip, and human review of operator separation"
        ),
        "limitations": list(REQUIRED_LIMITATIONS),
        "subject": subject,
    }


def verify_replay_role_bundle(
    bundle_path: Path,
    trust_policy_path: Path,
    *,
    subject_files: dict[str, Path],
    role_evidence_files: dict[str, Path],
) -> dict[str, Any]:
    """Production API: use only the verifier-controlled registry."""

    return _verify_with_registry(
        bundle_path,
        trust_policy_path,
        subject_files=subject_files,
        role_evidence_files=role_evidence_files,
        trust_anchor_registry_path=ROOT / REGISTRY_RELATIVE,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--trust-policy", type=Path, required=True)
    for key in SUBJECT_KEYS:
        stem = key.removesuffix("_sha256").replace("_", "-")
        parser.add_argument("--" + stem + "-file", type=Path, required=True)
    for role in REQUIRED_ROLES:
        parser.add_argument(
            "--" + role.replace("_", "-") + "-evidence-file",
            type=Path,
            required=True,
        )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subject_files = {
        key: getattr(args, key.removesuffix("_sha256") + "_file")
        for key in SUBJECT_KEYS
    }
    role_evidence_files = {
        role: getattr(args, role + "_evidence_file") for role in REQUIRED_ROLES
    }
    try:
        report = verify_replay_role_bundle(
            args.bundle,
            args.trust_policy,
            subject_files=subject_files,
            role_evidence_files=role_evidence_files,
        )
    except Exception as exc:
        report = {
            "schema_version": "n5-d5-l2d-replay-role-verification-report-1.0",
            "status": "L2D_FAIL_CLOSED",
            "error": f"{type(exc).__name__}:{exc}",
            "one_time_acceptance_proven": False,
            "role_operational_independence_proven": False,
            "production_backend_authorized": False,
            "formal_replay_authorized": False,
            "model_training_authorized": False,
            "claim_authorizations": {key: False for key in CLAIM_KEYS},
        }
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 2
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _production_api_has_no_registry_parameter() -> bool:
    """Tiny introspection helper used by tests and reviewers."""

    return "trust_anchor_registry_path" not in inspect.signature(
        verify_replay_role_bundle
    ).parameters


if __name__ == "__main__":
    raise SystemExit(main())
