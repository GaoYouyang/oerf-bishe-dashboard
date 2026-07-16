"""Protocol locks and hashing firewalls for v5h GC-RIO."""

from __future__ import annotations

import copy
import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


SPLITS = ("train", "validation", "design_lock")
PREDICTOR_KEYS = frozenset(
    {
        "source_operator",
        "target_operator",
        "source_residual",
        "source_sigma",
        "target_sigma",
        "base_field",
        "support",
        "analytic_correction",
        "conditioning_target_operator",
    }
)
FORBIDDEN_PAYLOAD_TERMS = frozenset(
    {"label", "labels", "observation", "observations", "truth", "clean"}
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash JSON data using a stable, whitespace-independent encoding."""

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _array_bytes(value: Any) -> tuple[str, tuple[int, ...], bytes]:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().contiguous().numpy()
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("object arrays cannot be hashed as state_dict values")
    array = np.ascontiguousarray(array)
    return array.dtype.str, tuple(array.shape), array.tobytes(order="C")


def sha256_state_dict(state_dict: Mapping[str, Any]) -> str:
    """Hash tensor contents plus key, dtype, and shape in canonical key order."""

    if not isinstance(state_dict, Mapping):
        raise TypeError("state_dict must be a mapping")
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        if not isinstance(key, str):
            raise TypeError("state_dict keys must be strings")
        dtype, shape, raw = _array_bytes(state_dict[key])
        metadata = _canonical_json_bytes({"key": key, "dtype": dtype, "shape": shape})
        digest.update(struct.pack(">Q", len(metadata)))
        digest.update(metadata)
        digest.update(struct.pack(">Q", len(raw)))
        digest.update(raw)
    return digest.hexdigest()


def _split_families(config: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    splits = config.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(SPLITS):
        raise ValueError("config.splits must contain exactly train, validation, and design_lock")
    result: dict[str, tuple[str, ...]] = {}
    for split in SPLITS:
        entry = splits[split]
        if not isinstance(entry, Mapping):
            raise ValueError(f"split {split} must be a mapping")
        families = entry.get("families")
        if not isinstance(families, Sequence) or isinstance(families, (str, bytes)):
            raise ValueError(f"split {split} must define a family sequence")
        result[split] = tuple(families)
        if not result[split] or any(not isinstance(name, str) or not name for name in result[split]):
            raise ValueError(f"split {split} must contain nonempty family names")
        if len(set(result[split])) != len(result[split]):
            raise ValueError(f"split {split} contains duplicate families")
    return result


def validate_full_protocol(config: Mapping[str, Any]) -> None:
    """Validate all pre-freeze split, view, and selection protocol locks."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    families = _split_families(config)
    family_owners = {
        family: split for split in SPLITS for family in families[split]
    }
    if len(family_owners) != sum(len(families[split]) for split in SPLITS):
        raise ValueError("families must be disjoint across all three splits")

    rigs = config.get("rigs")
    if not isinstance(rigs, Sequence) or isinstance(rigs, (str, bytes)):
        raise ValueError("config.rigs must be a sequence")
    rig_ids: set[str] = set()
    rig_counts = {split: 0 for split in SPLITS}
    for rig in rigs:
        if not isinstance(rig, Mapping):
            raise ValueError("each rig must be a mapping")
        rig_id, split = rig.get("id"), rig.get("split")
        if not isinstance(rig_id, str) or not rig_id or rig_id in rig_ids:
            raise ValueError("rig ids must be nonempty and unique across all three splits")
        if split not in SPLITS:
            raise ValueError(f"rig {rig_id} has an invalid split")
        rig_ids.add(rig_id)
        rig_counts[split] += 1
        source = tuple(rig.get("source_indices", ()))
        target = tuple(rig.get("target_indices", ()))
        reserved = tuple(rig.get("reserved_indices", ()))
        if (len(source), len(target), len(reserved)) != (4, 2, 1):
            raise ValueError(f"rig {rig_id} must use a 4/2/1 view partition")
        if set(source) | set(target) | set(reserved) != set(range(7)):
            raise ValueError(f"rig {rig_id} must partition exactly views 0 through 6")
        if len(source + target + reserved) != len(set(source + target + reserved)):
            raise ValueError(f"rig {rig_id} view partitions overlap")
    if any(rig_counts[split] == 0 for split in SPLITS):
        raise ValueError("rigs must populate train, validation, and design_lock separately")
    if rig_counts["design_lock"] < 2:
        raise ValueError("at least two design_lock rigs are required")

    gate = config.get("development_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("config.development_gate must be a mapping")
    if gate.get("checkpoint_selection_split") != "validation":
        raise ValueError("checkpoint selection must use validation only")
    if gate.get("truth_field_forbidden_for_training_and_selection") is not True:
        raise ValueError("truth fields must be forbidden for training and selection")
    if gate.get("design_lock_forbidden_before_freeze") is not True:
        raise ValueError("design_lock must be forbidden before freeze")


def make_development_config(full_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a buildable config containing only train and validation assets."""

    validate_full_protocol(full_config)
    development = copy.deepcopy(dict(full_config))
    development["splits"] = {
        split: development["splits"][split] for split in ("train", "validation")
    }
    # The dataset builder uses this top-level value as a missing-split fallback.
    development["families"] = []
    development["rigs"] = [
        rig for rig in development["rigs"] if rig["split"] in {"train", "validation"}
    ]
    if any(rig.get("split") == "design_lock" for rig in development["rigs"]):
        raise RuntimeError("design_lock rig escaped into development config")
    return development


def _key_terms(key: Any) -> set[str]:
    return set(str(key).lower().replace("-", "_").split("_"))


def _reject_forbidden_keys(value: Any, *, location: str) -> None:
    if isinstance(value, Mapping):
        for key in value:
            if _key_terms(key) & FORBIDDEN_PAYLOAD_TERMS:
                raise ValueError(f"forbidden label/observation key at {location}.{key}")
            child = value[key]
            _reject_forbidden_keys(child, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden_keys(child, location=f"{location}[{index}]")


def _reject_design_lock_data(value: Any, *, location: str) -> None:
    if isinstance(value, Mapping):
        for key in value:
            normalized = str(key).lower().replace("-", "_")
            terms = set(normalized.split("_"))
            if terms & {"label", "labels", "observation", "observations"} or normalized in {
                "truth_field",
                "clean_observation",
            }:
                raise ValueError(f"forbidden design_lock label key at {location}.{key}")
            _reject_design_lock_data(value[key], location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_design_lock_data(child, location=f"{location}[{index}]")


def _reject_design_lock_labels(config: Mapping[str, Any]) -> None:
    splits = config.get("splits", {})
    if isinstance(splits, Mapping) and "design_lock" in splits:
        _reject_design_lock_data(
            splits["design_lock"], location="config.splits.design_lock"
        )
    rigs = config.get("rigs", ())
    if isinstance(rigs, Sequence) and not isinstance(rigs, (str, bytes)):
        for index, rig in enumerate(rigs):
            if isinstance(rig, Mapping) and rig.get("split") == "design_lock":
                _reject_design_lock_data(rig, location=f"config.rigs[{index}]")


def hash_prediction_payload(
    payload: Mapping[str, Any], *, predictor_keys: frozenset[str] = PREDICTOR_KEYS
) -> str:
    """Hash predictor-only arrays while rejecting all scoring-side material."""

    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("prediction payload must be a nonempty mapping")
    keys = set(payload)
    forbidden = keys - predictor_keys
    if forbidden:
        raise ValueError(f"prediction payload contains non-predictor keys: {sorted(forbidden)}")
    _reject_forbidden_keys(payload, location="prediction_payload")
    return sha256_state_dict(payload)


def create_freeze_manifest(
    full_config: Mapping[str, Any],
    *,
    code_files: Mapping[str, str | Path],
    checkpoint_path: str | Path,
    validation_selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Commit protocol, code, checkpoint, and validation-only selection evidence."""

    validate_full_protocol(full_config)
    # Refuse label-bearing lock material before canonicalizing the full config.
    _reject_design_lock_labels(full_config)
    if not isinstance(validation_selection, Mapping):
        raise TypeError("validation_selection must be a mapping")
    if validation_selection.get("split") != "validation":
        raise ValueError("freeze selection record must explicitly name validation split")
    _reject_forbidden_keys(validation_selection, location="validation_selection")
    if not code_files:
        raise ValueError("at least one code file is required")
    return {
        "schema": "v5h-gc-rio-freeze-manifest-1",
        "full_config_sha256": sha256_json(full_config),
        "code_file_sha256": {
            name: sha256_file(path) for name, path in sorted(code_files.items())
        },
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "validation_selection": copy.deepcopy(dict(validation_selection)),
        "validation_selection_sha256": sha256_json(validation_selection),
    }


# Readable aliases for callers that use noun-first naming.
development_config = make_development_config
freeze_manifest = create_freeze_manifest
prediction_payload_sha256 = hash_prediction_payload
