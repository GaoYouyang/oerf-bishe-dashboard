#!/usr/bin/env python3
"""Verify an externally witnessed N5-D5 L2-C describe-only capsule.

The verifier accepts a verifier-registry-pinned trust policy, two distinct
Ed25519 key roles, a closed event hash chain, and declared external protocol cost.
It contains no private-key or signing path.  A successful result authenticates
who signed which bytes and checks the declared structure; it does not prove that
the signed host-capability assertions are physically true and never authorizes
production execution, reconstruction, training, or a scientific claim by itself.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[1]
POLICY_SCHEMA_RELATIVE = "data_templates/n5_d5_l2c_trust_policy.schema.json"
TRUST_ANCHOR_REGISTRY_SCHEMA_RELATIVE = (
    "data_templates/n5_d5_l2c_trust_anchor_registry.schema.json"
)
TRUST_ANCHOR_REGISTRY_RELATIVE = (
    "data_templates/n5_d5_l2c_trust_anchor_registry.json"
)
COMPILED_TRUST_ANCHOR_REGISTRY_SHA256 = (
    "a11b2770d74d8b0541caa4dfd6f71e689825b5f67c90c6bec4551d6cf60cce92"
)
BUNDLE_SCHEMA_RELATIVE = (
    "data_templates/n5_d5_l2c_external_witness_bundle.schema.json"
)
POLICY_PLACEHOLDER_RELATIVE = (
    "data_templates/n5_d5_l2c_trust_policy.placeholder.json"
)
BUNDLE_PLACEHOLDER_RELATIVE = (
    "data_templates/n5_d5_l2c_external_witness_bundle.placeholder.json"
)
HOST_SIGNATURE_DOMAIN = b"OERF:N5-D5:L2-C:HOST-CAPABILITY-WITNESS:V1\x00"
COST_SIGNATURE_DOMAIN = b"OERF:N5-D5:L2-C:RUN-COST-OBSERVER:V1\x00"
EVENT_DOMAIN = b"OERF:N5-D5:L2-C:EVENT:V1\x00"
MAX_CONTROL_FILE_BYTES = 4 * 1024 * 1024
MAX_JSON_DEPTH = 24
MAX_JSON_ITEMS = 2_048
MAX_JSON_STRING = 32_768
MAX_BUNDLE_LIFETIME = timedelta(minutes=15)
ALLOWED_CLOCK_SKEW = timedelta(seconds=30)
ZERO_SHA256 = "0" * 64
PLACEHOLDER_MARKERS = ("replace_", "REPLACE_ME")
REQUIRED_ROLES = ("host_capability_witness", "run_cost_observer")
CAPABILITY_CODES = (
    "PROCESS_EXEC_REPLACEMENT_DENIED",
    "PRIVATE_INPUT_ROOT_EXTERNAL_MUTATION_DENIED",
    "DURABLE_NONCE_LEDGER_ROOT_PROTECTED",
    "OUTPUT_ROOT_EXTERNAL_MUTATION_DENIED",
)
EXPECTED_EVENT_TYPES = (
    "RUN_CREATED",
    "HOST_CAPABILITY_CAPTURE_STARTED",
    "HOST_CAPABILITY_CAPTURE_FINISHED",
    "AUTHORIZATION_BOUND",
    "AUTHORIZATION_CONSUMED",
    "PROCESS_STARTED",
    "DESCRIBE_SENT",
    "DESCRIBE_RECEIVED",
    "DESCRIBE_SENT",
    "DESCRIBE_RECEIVED",
    "PROCESS_EXITED",
    "OUTPUT_BOUND",
    "COST_OBSERVATION_FINALIZED",
    "RUN_SEALED",
)
SUBJECT_KEYS = (
    "authorization_sha256",
    "replay_plan_sha256",
    "foundation_report_sha256",
    "adapter_entrypoint_sha256",
    "runner_source_sha256",
    "challenge_commitment_sha256",
    "output_manifest_sha256",
)
CLAIM_KEYS = (
    "real_bost_interface",
    "physical_forward_correctness",
    "derivative_correctness",
    "reconstruction",
    "algorithm_superiority",
    "generalization",
    "publication",
)
REQUIRED_LIMITATIONS = (
    "Ed25519 signatures authenticate statements but do not prove the statements are physically true.",
    "An event hash chain detects post-hoc mutation but cannot prove that every real event was observed.",
    "Describe-only cost is protocol telemetry and cannot establish reconstruction or model efficiency.",
    "This bundle never authorizes real BOST, reconstruction, superiority, generalization, or publication claims.",
)


class L2CWitnessError(RuntimeError):
    """Fail-closed external-witness verification error."""


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise L2CWitnessError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise L2CWitnessError(f"non-finite JSON number: {value}")


def _bounded_json_value(value: object, *, depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        raise L2CWitnessError("JSON nesting exceeds the control-file bound")
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise L2CWitnessError("non-finite JSON number")
        return
    if isinstance(value, str):
        if len(value) > MAX_JSON_STRING:
            raise L2CWitnessError("JSON string exceeds the control-file bound")
        return
    if isinstance(value, list):
        if len(value) > MAX_JSON_ITEMS:
            raise L2CWitnessError("JSON array exceeds the control-file bound")
        for item in value:
            _bounded_json_value(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > MAX_JSON_ITEMS:
            raise L2CWitnessError("JSON object exceeds the control-file bound")
        for key, item in value.items():
            if not isinstance(key, str):
                raise L2CWitnessError("JSON object key is not a string")
            _bounded_json_value(key, depth=depth + 1)
            _bounded_json_value(item, depth=depth + 1)
        return
    raise L2CWitnessError(f"unsupported JSON value type: {type(value).__name__}")


def _load_json_bytes(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_no_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise L2CWitnessError(f"invalid UTF-8 JSON: {exc}") from exc
    _bounded_json_value(value)
    if not isinstance(value, dict):
        raise L2CWitnessError("control file is not a JSON object")
    return value


def _snapshot_regular_file(path: Path) -> tuple[bytes, str]:
    lexical = Path(os.path.abspath(path))
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        file_fd = os.open(lexical, flags)
    except OSError as exc:
        raise L2CWitnessError(f"control file cannot be opened safely: {exc}") from exc
    try:
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise L2CWitnessError("control file is not regular")
        if before.st_size > MAX_CONTROL_FILE_BYTES:
            raise L2CWitnessError("control file exceeds the byte bound")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(file_fd, min(1024 * 1024, MAX_CONTROL_FILE_BYTES + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_CONTROL_FILE_BYTES:
                raise L2CWitnessError("control file exceeds the byte bound")
            chunks.append(chunk)
        after = os.fstat(file_fd)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise L2CWitnessError("control file changed while being read")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise L2CWitnessError("control file size changed while being read")
        return payload, sha256_bytes(payload)
    finally:
        os.close(file_fd)


def _load_schema(path: Path) -> dict[str, Any]:
    payload, _ = _snapshot_regular_file(path)
    return _load_json_bytes(payload)


def _validate_schema(schema: dict[str, Any], value: object, label: str) -> None:
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(value), key=lambda item: list(item.path))
    if errors:
        rendered = []
        for error in errors[:12]:
            path = ".".join(str(part) for part in error.absolute_path) or "$"
            rendered.append(f"{path}: {error.message}")
        raise L2CWitnessError(f"{label} schema failed: " + "; ".join(rendered))


def _parse_utc(value: str, label: str) -> datetime:
    if not value.endswith("Z"):
        raise L2CWitnessError(f"{label} must use UTC Z notation")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise L2CWitnessError(f"{label} is not a valid UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise L2CWitnessError(f"{label} is not UTC")
    return parsed.astimezone(timezone.utc)


def _all_claims_false(value: object) -> bool:
    return isinstance(value, dict) and tuple(sorted(value)) == tuple(
        sorted(CLAIM_KEYS)
    ) and all(value[key] is False for key in CLAIM_KEYS)


def _decode_base64(value: str, expected_bytes: int, label: str) -> bytes:
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise L2CWitnessError(f"{label} is not strict base64") from exc
    if len(decoded) != expected_bytes:
        raise L2CWitnessError(f"{label} has the wrong byte length")
    if decoded == b"\x00" * expected_bytes:
        raise L2CWitnessError(f"{label} is the all-zero placeholder")
    return decoded


def event_sha256(event_without_hash: dict[str, Any]) -> str:
    if "event_sha256" in event_without_hash:
        raise L2CWitnessError("event hash input still contains event_sha256")
    return sha256_bytes(
        EVENT_DOMAIN + canonical_json(event_without_hash).encode("utf-8")
    )


def role_signed_payload_bytes(bundle: dict[str, Any], role: str) -> bytes:
    if not isinstance(bundle.get("signatures"), list):
        raise L2CWitnessError("bundle signatures are missing")
    common = {
        key: copy.deepcopy(bundle[key])
        for key in (
            "schema_version",
            "record_kind",
            "predicate_type",
            "run_id",
            "issued_at_utc",
            "expires_at_utc",
            "subject",
            "claim_authorizations",
            "limitations",
        )
    }
    if role == "host_capability_witness":
        common.update(
            {
                "payload_scope": "HOST_CAPABILITY_SUBJECT_AND_FINAL_SEAL",
                "host_capability_statement": copy.deepcopy(
                    bundle["host_capability_statement"]
                ),
                "final_event_sha256": bundle["event_chain"][
                    "final_event_sha256"
                ],
                "terminal_status": bundle["terminal_status"],
            }
        )
        domain = HOST_SIGNATURE_DOMAIN
    elif role == "run_cost_observer":
        common.update(
            {
                "payload_scope": "RUN_EVENT_CHAIN_AND_DESCRIBE_TELEMETRY",
                "event_chain": copy.deepcopy(bundle["event_chain"]),
                "observed_cost": copy.deepcopy(bundle["observed_cost"]),
                "terminal_status": bundle["terminal_status"],
            }
        )
        domain = COST_SIGNATURE_DOMAIN
    else:
        raise L2CWitnessError("unknown signature role")
    return domain + canonical_json(common).encode("utf-8")


def _verify_policy(
    policy: dict[str, Any],
    *,
    policy_sha256: str,
    registry: dict[str, Any],
    now: datetime,
) -> dict[str, dict[str, Any]]:
    if not registry["production_trust_anchor_enrolled"]:
        raise L2CWitnessError("NO_PRODUCTION_TRUST_ANCHOR_ENROLLED")
    matching_pins = [
        pin
        for pin in registry["active_policy_pins"]
        if pin["policy_id"] == policy["policy_id"]
        and pin["policy_epoch"] == policy["policy_epoch"]
        and pin["policy_sha256"] == policy_sha256
        and pin["predicate_type"] == policy["expected_predicate_type"]
        and tuple(pin["required_signature_roles"]) == REQUIRED_ROLES
    ]
    if len(matching_pins) != 1:
        raise L2CWitnessError(
            "trust policy is not enrolled in the verifier-controlled registry"
        )
    if matching_pins[0]["enrollment_review_sha256"] == ZERO_SHA256:
        raise L2CWitnessError("trust-policy enrollment review digest is all-zero")
    if any(marker in canonical_json(policy) for marker in PLACEHOLDER_MARKERS):
        raise L2CWitnessError("trust policy still contains placeholder markers")
    if not _all_claims_false(policy.get("claim_authorizations")):
        raise L2CWitnessError("trust policy attempts to authorize a scientific claim")
    valid_from = _parse_utc(policy["valid_from_utc"], "policy valid_from_utc")
    valid_until = _parse_utc(policy["valid_until_utc"], "policy valid_until_utc")
    if valid_from >= valid_until:
        raise L2CWitnessError("trust policy validity interval is empty")
    if now < valid_from - ALLOWED_CLOCK_SKEW or now > valid_until + ALLOWED_CLOCK_SKEW:
        raise L2CWitnessError("trust policy is not valid at verification time")
    if tuple(policy["required_signature_roles"]) != REQUIRED_ROLES:
        raise L2CWitnessError("trust policy signature roles drifted")
    if policy["signature_threshold"] != len(REQUIRED_ROLES):
        raise L2CWitnessError("trust policy threshold does not require both roles")

    trusted: dict[str, dict[str, Any]] = {}
    seen_roles: set[str] = set()
    raw_keys: set[bytes] = set()
    for entry in policy["trusted_keys"]:
        key_id = entry["key_id"]
        if key_id in trusted:
            raise L2CWitnessError("trust policy contains duplicate key IDs")
        raw = _decode_base64(entry["public_key_base64"], 32, "trusted public key")
        if raw in raw_keys:
            raise L2CWitnessError("trust policy reuses one public key across identities")
        raw_keys.add(raw)
        trusted[key_id] = {**entry, "raw_public_key": raw}
        seen_roles.add(entry["role"])
    if seen_roles != set(REQUIRED_ROLES):
        raise L2CWitnessError("trust policy lacks one required independent role")
    return trusted


def _verify_subject(
    subject: dict[str, Any],
    *,
    policy_sha256: str,
    computed_subject: dict[str, str],
) -> None:
    if tuple(sorted(computed_subject)) != tuple(sorted(SUBJECT_KEYS)):
        raise L2CWitnessError("subject files must contain the seven pinned artifacts")
    if subject["trust_policy_sha256"] != policy_sha256:
        raise L2CWitnessError("bundle subject is not bound to the trust-policy bytes")
    for key in SUBJECT_KEYS:
        computed = computed_subject[key]
        if computed == ZERO_SHA256:
            raise L2CWitnessError(f"subject artifact digest is all-zero: {key}")
        if subject[key] != computed:
            raise L2CWitnessError(f"bundle subject mismatch: {key}")
    if any(value == ZERO_SHA256 for value in subject.values()):
        raise L2CWitnessError("bundle subject contains an all-zero digest")


def _snapshot_subject_artifacts(
    subject_files: dict[str, Path],
) -> dict[str, str]:
    if tuple(sorted(subject_files)) != tuple(sorted(SUBJECT_KEYS)):
        raise L2CWitnessError("subject_files must contain exactly seven artifacts")
    lexical_paths = [Path(os.path.abspath(subject_files[key])) for key in SUBJECT_KEYS]
    if len(set(lexical_paths)) != len(lexical_paths):
        raise L2CWitnessError("one subject file is reused for multiple artifact roles")
    return {
        key: _snapshot_regular_file(subject_files[key])[1]
        for key in SUBJECT_KEYS
    }


def _snapshot_evidence_materials(evidence_files: Sequence[Path]) -> set[str]:
    if not evidence_files:
        raise L2CWitnessError("private capability and observer evidence files are required")
    lexical_paths = [Path(os.path.abspath(path)) for path in evidence_files]
    if len(set(lexical_paths)) != len(lexical_paths):
        raise L2CWitnessError("evidence file paths contain duplicates")
    return {_snapshot_regular_file(path)[1] for path in evidence_files}


def _verify_times(
    bundle: dict[str, Any], policy: dict[str, Any], *, now: datetime
) -> None:
    issued = _parse_utc(bundle["issued_at_utc"], "bundle issued_at_utc")
    expires = _parse_utc(bundle["expires_at_utc"], "bundle expires_at_utc")
    policy_from = _parse_utc(policy["valid_from_utc"], "policy valid_from_utc")
    policy_until = _parse_utc(policy["valid_until_utc"], "policy valid_until_utc")
    if issued >= expires:
        raise L2CWitnessError("bundle validity interval is empty")
    if expires - issued > MAX_BUNDLE_LIFETIME:
        raise L2CWitnessError("bundle lifetime exceeds fifteen minutes")
    if issued < policy_from or expires > policy_until:
        raise L2CWitnessError("bundle validity lies outside the trust policy")
    if now < issued - ALLOWED_CLOCK_SKEW:
        raise L2CWitnessError("bundle is issued too far in the future")
    if now > expires + ALLOWED_CLOCK_SKEW:
        raise L2CWitnessError("bundle has expired")


def _verify_capability_statement(
    statement: dict[str, Any], *, evidence_digests: set[str]
) -> str:
    evidence = statement["evidence"]
    codes = [entry["capability_code"] for entry in evidence]
    if len(codes) != len(set(codes)):
        raise L2CWitnessError("capability evidence contains duplicate codes")
    if set(codes) != set(CAPABILITY_CODES):
        raise L2CWitnessError("capability evidence is incomplete")
    for entry in evidence:
        if entry["artifact_sha256"] == ZERO_SHA256:
            raise L2CWitnessError("capability evidence uses an all-zero artifact digest")
        if entry["measurement_tool_sha256"] == ZERO_SHA256:
            raise L2CWitnessError("capability evidence uses an all-zero tool digest")
        if entry["artifact_sha256"] not in evidence_digests:
            raise L2CWitnessError("capability evidence artifact bytes are unavailable")
        if entry["measurement_tool_sha256"] not in evidence_digests:
            raise L2CWitnessError("capability measurement-tool bytes are unavailable")
        if any(marker in entry["measurement_method"] for marker in PLACEHOLDER_MARKERS):
            raise L2CWitnessError("capability evidence still contains placeholders")
    return sha256_bytes(canonical_json(statement).encode("utf-8"))


def _verify_cost(cost: dict[str, Any], *, evidence_digests: set[str]) -> str:
    memory_status = cost["memory_observation_status"]
    peak_rss = cost["describe_only_peak_rss_bytes"]
    if memory_status == "OBSERVED_PROCESS_TREE" and peak_rss is None:
        raise L2CWitnessError("observed memory status lacks peak RSS")
    if memory_status == "NOT_RELIABLY_OBSERVED" and peak_rss is not None:
        raise L2CWitnessError("unreliable memory status must not publish pseudo-precision")
    if cost["observer_tool_sha256"] == ZERO_SHA256:
        raise L2CWitnessError("cost observer tool digest is all-zero")
    if cost["observer_tool_sha256"] not in evidence_digests:
        raise L2CWitnessError("cost observer tool bytes are unavailable")
    return sha256_bytes(canonical_json(cost).encode("utf-8"))


def _event_expected_request(sequence: int) -> tuple[int | None, str | None]:
    if sequence in (6, 7):
        return 1, "describe"
    if sequence in (8, 9):
        return 2, "describe"
    return None, None


def _verify_event_chain(
    chain: dict[str, Any],
    *,
    subject: dict[str, Any],
    capability_sha256: str,
    cost_sha256: str,
) -> str:
    events = chain["events"]
    if chain["event_count"] != len(events):
        raise L2CWitnessError("event count does not match the event list")
    previous = ZERO_SHA256
    last_monotonic = -1
    for sequence, event in enumerate(events):
        if event["sequence"] != sequence:
            raise L2CWitnessError("event sequence is not contiguous")
        if event["event_type"] != EXPECTED_EVENT_TYPES[sequence]:
            raise L2CWitnessError("event type order drifted")
        request_index, operation = _event_expected_request(sequence)
        if event["request_index"] != request_index or event["operation"] != operation:
            raise L2CWitnessError("describe event request binding drifted")
        if event["monotonic_ns"] <= last_monotonic:
            raise L2CWitnessError("event monotonic clock is not strictly increasing")
        last_monotonic = event["monotonic_ns"]
        if event["payload_sha256"] == ZERO_SHA256:
            raise L2CWitnessError("event payload uses an all-zero digest")
        if event["previous_event_sha256"] != previous:
            raise L2CWitnessError("event previous hash does not match the chain")
        unsigned_event = dict(event)
        claimed = unsigned_event.pop("event_sha256")
        computed = event_sha256(unsigned_event)
        if claimed != computed:
            raise L2CWitnessError("event hash does not reproduce")
        previous = claimed

    if chain["final_event_sha256"] != previous:
        raise L2CWitnessError("final event hash does not seal the chain")
    if events[0]["payload_sha256"] != sha256_bytes(
        canonical_json(subject).encode("utf-8")
    ):
        raise L2CWitnessError("RUN_CREATED is not bound to the subject")
    if events[2]["payload_sha256"] != capability_sha256:
        raise L2CWitnessError("capability-finalized event is not bound to the statement")
    if events[3]["payload_sha256"] != subject["authorization_sha256"]:
        raise L2CWitnessError("authorization-bound event has the wrong digest")
    if events[4]["payload_sha256"] != subject["authorization_sha256"]:
        raise L2CWitnessError("authorization-consumed event has the wrong digest")
    if events[5]["payload_sha256"] != subject["adapter_entrypoint_sha256"]:
        raise L2CWitnessError("process-started event is not bound to the adapter")
    if events[11]["payload_sha256"] != subject["output_manifest_sha256"]:
        raise L2CWitnessError("output-bound event has the wrong manifest digest")
    if events[12]["payload_sha256"] != cost_sha256:
        raise L2CWitnessError("cost-finalized event is not bound to observed cost")
    seal_material = {
        "subject_sha256": sha256_bytes(canonical_json(subject).encode("utf-8")),
        "capability_statement_sha256": capability_sha256,
        "observed_cost_sha256": cost_sha256,
        "previous_event_sha256": events[12]["event_sha256"],
    }
    if events[13]["payload_sha256"] != sha256_bytes(
        canonical_json(seal_material).encode("utf-8")
    ):
        raise L2CWitnessError("RUN_SEALED payload does not bind the capsule")
    return previous


def _verify_signatures(
    bundle: dict[str, Any], trusted: dict[str, dict[str, Any]]
) -> tuple[list[str], dict[str, str]]:
    accepted_roles: list[str] = []
    payload_digests: dict[str, str] = {}
    seen_keys: set[str] = set()
    seen_roles: set[str] = set()
    for signature in bundle["signatures"]:
        key_id = signature["key_id"]
        if key_id in seen_keys:
            raise L2CWitnessError("bundle contains duplicate signer key IDs")
        seen_keys.add(key_id)
        trusted_entry = trusted.get(key_id)
        if trusted_entry is None:
            raise L2CWitnessError("bundle contains an untrusted signer")
        role = signature["role"]
        if role != trusted_entry["role"]:
            raise L2CWitnessError("signer role does not match the trust policy")
        if role in seen_roles:
            raise L2CWitnessError("one signature role is duplicated")
        seen_roles.add(role)
        expected_scope = (
            "HOST_CAPABILITY_SUBJECT_AND_FINAL_SEAL"
            if role == "host_capability_witness"
            else "RUN_EVENT_CHAIN_AND_DESCRIBE_TELEMETRY"
        )
        if signature["payload_scope"] != expected_scope:
            raise L2CWitnessError("signature payload scope does not match its role")
        message = role_signed_payload_bytes(bundle, role)
        payload_sha256 = sha256_bytes(message)
        if signature["signed_payload_sha256"] != payload_sha256:
            raise L2CWitnessError("signature payload digest does not match the bundle")
        raw_signature = _decode_base64(
            signature["signature_base64"], 64, "Ed25519 signature"
        )
        try:
            Ed25519PublicKey.from_public_bytes(
                trusted_entry["raw_public_key"]
            ).verify(raw_signature, message)
        except (InvalidSignature, ValueError) as exc:
            raise L2CWitnessError("Ed25519 signature verification failed") from exc
        accepted_roles.append(role)
        payload_digests[role] = payload_sha256
    if set(accepted_roles) != set(REQUIRED_ROLES):
        raise L2CWitnessError("both independent signature roles are required")
    return sorted(accepted_roles), payload_digests


def _verify_external_witness_bundle_with_registry(
    bundle_path: Path,
    trust_policy_path: Path,
    *,
    subject_files: dict[str, Path],
    evidence_files: Sequence[Path],
    trust_anchor_registry_path: Path,
) -> dict[str, Any]:
    production_registry = (
        Path(os.path.abspath(trust_anchor_registry_path))
        == Path(os.path.abspath(ROOT / TRUST_ANCHOR_REGISTRY_RELATIVE))
    )
    registry_payload, registry_file_sha256 = _snapshot_regular_file(
        trust_anchor_registry_path
    )
    if (
        production_registry
        and registry_file_sha256 != COMPILED_TRUST_ANCHOR_REGISTRY_SHA256
    ):
        raise L2CWitnessError("VERIFIER_CONTROLLED_REGISTRY_BYTES_CHANGED")
    registry = _load_json_bytes(registry_payload)
    registry_schema = _load_schema(ROOT / TRUST_ANCHOR_REGISTRY_SCHEMA_RELATIVE)
    _validate_schema(registry_schema, registry, "trust-anchor registry")
    if not _all_claims_false(registry.get("claim_authorizations")):
        raise L2CWitnessError("trust-anchor registry authorizes a scientific claim")
    if not registry["production_trust_anchor_enrolled"]:
        raise L2CWitnessError("NO_PRODUCTION_TRUST_ANCHOR_ENROLLED")

    bundle_payload, bundle_file_sha256 = _snapshot_regular_file(bundle_path)
    policy_payload, policy_file_sha256 = _snapshot_regular_file(trust_policy_path)
    bundle = _load_json_bytes(bundle_payload)
    policy = _load_json_bytes(policy_payload)
    policy_schema = _load_schema(ROOT / POLICY_SCHEMA_RELATIVE)
    bundle_schema = _load_schema(ROOT / BUNDLE_SCHEMA_RELATIVE)
    _validate_schema(policy_schema, policy, "trust policy")
    _validate_schema(bundle_schema, bundle, "witness bundle")
    if any(marker in canonical_json(bundle) for marker in PLACEHOLDER_MARKERS):
        raise L2CWitnessError("witness bundle still contains placeholder markers")
    if not _all_claims_false(bundle.get("claim_authorizations")):
        raise L2CWitnessError("witness bundle attempts to authorize a scientific claim")
    if not all(item in bundle["limitations"] for item in REQUIRED_LIMITATIONS):
        raise L2CWitnessError("witness bundle weakens a required limitation")

    now = datetime.now(timezone.utc)
    trusted = _verify_policy(
        policy,
        policy_sha256=policy_file_sha256,
        registry=registry,
        now=now,
    )
    if bundle["predicate_type"] != policy["expected_predicate_type"]:
        raise L2CWitnessError("bundle predicate type does not match the trust policy")
    _verify_times(bundle, policy, now=now)
    computed_subject = _snapshot_subject_artifacts(subject_files)
    evidence_digests = _snapshot_evidence_materials(evidence_files)
    _verify_subject(
        bundle["subject"],
        policy_sha256=policy_file_sha256,
        computed_subject=computed_subject,
    )
    capability_sha256 = _verify_capability_statement(
        bundle["host_capability_statement"], evidence_digests=evidence_digests
    )
    cost_sha256 = _verify_cost(
        bundle["observed_cost"], evidence_digests=evidence_digests
    )
    final_event_sha256 = _verify_event_chain(
        bundle["event_chain"],
        subject=bundle["subject"],
        capability_sha256=capability_sha256,
        cost_sha256=cost_sha256,
    )
    accepted_roles, signed_payload_sha256_by_role = _verify_signatures(
        bundle, trusted
    )
    return {
        "schema_version": "n5-d5-l2c-external-witness-verification-report-1.0",
        "status": (
            "L2C_SIGNED_STATEMENTS_STRUCTURALLY_VERIFIED_"
            "REPLAY_AND_EVIDENCE_REVIEW_REQUIRED"
            if production_registry
            else "L2C_DEVELOPMENT_REGISTRY_SIGNED_STATEMENTS_STRUCTURALLY_VERIFIED"
        ),
        "bundle_file_sha256": bundle_file_sha256,
        "trust_policy_file_sha256": policy_file_sha256,
        "trust_anchor_registry_file_sha256": registry_file_sha256,
        "verifier_controlled_registry_used": production_registry,
        "enrolled_policy_digest_matched": True,
        "predicate_type_matched": True,
        "subject_artifact_bytes_rehashed": True,
        "subject_digests_matched_to_files": True,
        "private_evidence_bytes_rehashed": True,
        "private_evidence_digest_count": len(evidence_digests),
        "trusted_signature_count": len(accepted_roles),
        "trusted_signature_roles": accepted_roles,
        "role_scoped_signed_payload_sha256": signed_payload_sha256_by_role,
        "role_key_separation_verified": True,
        "role_operational_independence_proven": False,
        "event_chain_verified": True,
        "final_event_sha256": final_event_sha256,
        "host_capability_claims_signed": True,
        "host_capability_truth_independently_proven": False,
        "external_describe_telemetry_statement_structurally_verified": True,
        "external_describe_telemetry_truth_independently_proven": False,
        "cost_namespace": "DESCRIBE_ONLY_PROTOCOL_TELEMETRY",
        "eligible_for_reconstruction_or_model_cost_comparison": False,
        "replay_ledger_checked": False,
        "one_time_acceptance_proven": False,
        "backend_capability_attestation_signature_gate_passed": True,
        "backend_capability_attestation_blocker_automatically_cleared": False,
        "production_backend_authorized": False,
        "formal_replay_authorized": False,
        "model_training_authorized": False,
        "claim_authorizations": {key: False for key in CLAIM_KEYS},
        "next_gate": (
            "independent review of private capability evidence plus enforced "
            "backend integration; then a separate one-time production authorization"
        ),
        "limitations": list(REQUIRED_LIMITATIONS),
    }


def verify_external_witness_bundle(
    bundle_path: Path,
    trust_policy_path: Path,
    *,
    subject_files: dict[str, Path],
    evidence_files: Sequence[Path],
) -> dict[str, Any]:
    """Use the fixed registry; the caller cannot nominate a new trust root."""

    return _verify_external_witness_bundle_with_registry(
        bundle_path,
        trust_policy_path,
        subject_files=subject_files,
        evidence_files=evidence_files,
        trust_anchor_registry_path=ROOT / TRUST_ANCHOR_REGISTRY_RELATIVE,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--trust-policy", type=Path, required=True)
    for key in SUBJECT_KEYS:
        stem = key.removesuffix("_sha256").replace("_", "-")
        parser.add_argument("--" + stem + "-file", type=Path, required=True)
    parser.add_argument(
        "--evidence-file", type=Path, action="append", required=True
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subject_files = {
        key: getattr(args, key.removesuffix("_sha256") + "_file")
        for key in SUBJECT_KEYS
    }
    try:
        report = verify_external_witness_bundle(
            args.bundle,
            args.trust_policy,
            subject_files=subject_files,
            evidence_files=args.evidence_file,
        )
    except Exception as exc:
        report = {
            "schema_version": "n5-d5-l2c-external-witness-verification-report-1.0",
            "status": "L2C_FAIL_CLOSED",
            "error": f"{type(exc).__name__}:{exc}",
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


if __name__ == "__main__":
    raise SystemExit(main())
