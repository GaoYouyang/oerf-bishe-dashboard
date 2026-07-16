from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from .protocol import (
    PREDICTOR_KEYS,
    create_freeze_manifest,
    hash_prediction_payload,
    make_development_config,
    sha256_file,
    sha256_json,
    sha256_state_dict,
    validate_full_protocol,
)


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "v5h_gc_rio_development.json"


def full_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_sha256_helpers_are_deterministic_and_tamper_sensitive(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"checkpoint-a")
    first = sha256_file(artifact)
    artifact.write_bytes(b"checkpoint-b")
    assert sha256_file(artifact) != first

    assert sha256_json({"b": 2, "a": 1}) == sha256_json({"a": 1, "b": 2})
    assert sha256_json({"a": 2}) != sha256_json({"a": 1})

    left = {"weight": np.arange(6, dtype=np.float32).reshape(2, 3), "bias": np.zeros(2)}
    right = {"bias": np.zeros(2), "weight": left["weight"].copy()}
    assert sha256_state_dict(left) == sha256_state_dict(right)
    right["weight"][0, 0] = 9
    assert sha256_state_dict(left) != sha256_state_dict(right)


def test_development_config_excludes_design_lock_without_mutating_full_config() -> None:
    config = full_config()
    before = copy.deepcopy(config)
    development = make_development_config(config)

    assert set(development["splits"]) == {"train", "validation"}
    assert development["families"] == []
    assert {rig["split"] for rig in development["rigs"]} == {"train", "validation"}
    assert all("design-lock" not in rig["id"] for rig in development["rigs"])
    assert config == before


def test_full_protocol_accepts_preregistered_config() -> None:
    validate_full_protocol(full_config())


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda c: c["splits"]["validation"].update(families=c["splits"]["train"]["families"]), "families must be disjoint"),
        (lambda c: c["rigs"].__setitem__(1, {**c["rigs"][1], "id": c["rigs"][0]["id"]}), "rig ids"),
        (lambda c: c["rigs"][0].update(source_indices=[0, 1, 2]), "4/2/1"),
        (lambda c: c.update(rigs=[r for r in c["rigs"] if r["id"] != "design-lock-rig-b"]), "at least two"),
        (lambda c: c["development_gate"].update(checkpoint_selection_split="train"), "validation only"),
        (lambda c: c["development_gate"].update(truth_field_forbidden_for_training_and_selection=False), "truth fields"),
        (lambda c: c["development_gate"].update(design_lock_forbidden_before_freeze=False), "design_lock"),
    ],
)
def test_full_protocol_rejects_lock_tampering(mutator, message: str) -> None:
    config = full_config()
    mutator(config)
    with pytest.raises(ValueError, match=message):
        validate_full_protocol(config)


def predictor_payload() -> dict[str, np.ndarray]:
    return {key: np.asarray([index], dtype=np.float32) for index, key in enumerate(sorted(PREDICTOR_KEYS))}


def test_prediction_hash_accepts_only_predictors_and_detects_tampering() -> None:
    payload = predictor_payload()
    digest = hash_prediction_payload(payload)
    payload["source_residual"][0] += 1
    assert hash_prediction_payload(payload) != digest

    for forbidden in ("target_observation", "truth_field", "clean_target_residual", "target_residual_label"):
        poisoned = predictor_payload()
        poisoned[forbidden] = np.zeros(1)
        with pytest.raises(ValueError, match="non-predictor keys"):
            hash_prediction_payload(poisoned)


def test_freeze_manifest_hashes_only_validation_selection(tmp_path: Path) -> None:
    code = tmp_path / "train.py"
    checkpoint = tmp_path / "best.pt"
    code.write_text("print('train')\n", encoding="utf-8")
    checkpoint.write_bytes(b"selected checkpoint")
    selection = {"split": "validation", "epoch": 17, "metric": 0.125}

    manifest = create_freeze_manifest(
        full_config(),
        code_files={"train": code},
        checkpoint_path=checkpoint,
        validation_selection=selection,
    )
    assert manifest["full_config_sha256"] == sha256_json(full_config())
    assert manifest["code_file_sha256"] == {"train": sha256_file(code)}
    assert manifest["checkpoint_sha256"] == sha256_file(checkpoint)
    assert manifest["validation_selection"] == selection


@pytest.mark.parametrize(
    "selection",
    [
        {"split": "train", "metric": 0.1},
        {"split": "design_lock", "metric": 0.1},
        {"split": "validation", "target_observation": [1.0]},
        {"split": "validation", "truth_field": [1.0]},
        {"split": "validation", "clean_labels": [1.0]},
    ],
)
def test_freeze_manifest_rejects_nonvalidation_or_label_material(
    tmp_path: Path, selection: dict
) -> None:
    code = tmp_path / "code.py"
    checkpoint = tmp_path / "checkpoint.pt"
    code.write_bytes(b"code")
    checkpoint.write_bytes(b"checkpoint")
    with pytest.raises(ValueError):
        create_freeze_manifest(
            full_config(),
            code_files={"code": code},
            checkpoint_path=checkpoint,
            validation_selection=selection,
        )


def test_freeze_manifest_refuses_design_lock_labels_before_hashing(tmp_path: Path) -> None:
    config = full_config()
    config["splits"]["design_lock"]["labels"] = ["must-not-be-hashed"]
    code = tmp_path / "code.py"
    checkpoint = tmp_path / "checkpoint.pt"
    code.write_bytes(b"code")
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="forbidden design_lock label key"):
        create_freeze_manifest(
            config,
            code_files={"code": code},
            checkpoint_path=checkpoint,
            validation_selection={"split": "validation", "metric": 0.1},
        )
