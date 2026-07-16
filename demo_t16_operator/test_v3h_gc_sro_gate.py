from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from demo_t16_operator.own_algorithm_models import (
    AcquisitionSetConditioner,
    GeometryConditionedSpectralResidualOperator,
)
from demo_t16_operator.variable_geometry import (
    assign_geometry_partitions,
    build_geometry_manifest,
    enumerate_budget_masks,
    geometry_entropy_bits,
    reference_mask_id,
)


ROOT = Path(__file__).resolve().parent


def synthetic_inputs() -> torch.Tensor:
    batch, channels, depth, height, width = 4, 42, 8, 16, 16
    x = torch.randn(batch, channels, depth, height, width)
    x[:, 1] = 1.0
    x[:, 2] = 6.0 / 9.0
    angles = torch.deg2rad(torch.arange(0, 180, 20, dtype=torch.float32))
    masks = torch.tensor(
        [
            [1, 0, 1, 0, 1, 1, 1, 0, 1],
            [1, 1, 1, 0, 0, 0, 1, 1, 1],
            [0, 1, 1, 0, 1, 1, 1, 1, 0],
            [1, 1, 0, 0, 1, 1, 0, 1, 1],
        ],
        dtype=torch.float32,
    )
    for index in range(batch):
        for camera in range(9):
            x[index, 3 + camera] = masks[index, camera]
            x[index, 24 + camera] = masks[index, camera] * torch.sin(angles[camera])
            x[index, 33 + camera] = masks[index, camera] * torch.cos(angles[camera])
    return x


def make_gc_sro() -> GeometryConditionedSpectralResidualOperator:
    base = nn.Conv3d(42, 1, kernel_size=1)
    return GeometryConditionedSpectralResidualOperator(
        base_operator=base,
        view_count=9,
        mask_channel_start=3,
        angle_sin_channel_start=24,
        angle_cos_channel_start=33,
        coordinate_channels=(12, 13, 14),
        descriptor_hidden=12,
        descriptor_embedding=8,
        adapter_hidden=6,
        spectral_modes=(4, 6, 6),
        freeze_base=True,
    )


def test_v3h_config_locks_geometry_and_claim_boundaries() -> None:
    config = json.loads(
        (ROOT / "configs" / "v3h_gc_sro_geometry_gate.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["partition_counts"] == {
        "train": 16,
        "validation": 4,
        "geometry_ood": 4,
        "stress": 4,
    }
    assert config["gc_sro"]["descriptor_modes"] == [
        "geometry",
        "mask_only",
        "static",
        "shuffled",
    ]
    assert config["claims_boundary"]["superiority_tested"] is False
    assert config["claims_boundary"]["blind_final_opened"] is False


def test_variable_geometry_manifest_has_28_disjoint_masks() -> None:
    angles = np.arange(0, 180, 20, dtype=np.float32)
    forward = np.random.default_rng(3).normal(size=(9, 16, 256)).astype(np.float32)
    masks = enumerate_budget_masks(9, 6, 3)
    assert len(masks) == 28
    assert all(int(mask.sum()) == 6 and mask[3] == 0 for mask in masks)
    rows, _ = build_geometry_manifest(angles, forward, 6, 3, 180.0)
    reference = reference_mask_id([1, 0, 1, 0, 1, 1, 1, 0, 1])
    assigned = assign_geometry_partitions(
        rows,
        reference,
        partition_seed=20261201,
        counts={"train": 16, "validation": 4, "geometry_ood": 4, "stress": 4},
    )
    counts = {
        partition: sum(row["partition"] == partition for row in assigned)
        for partition in ("train", "validation", "geometry_ood", "stress")
    }
    assert counts == {"train": 16, "validation": 4, "geometry_ood": 4, "stress": 4}
    assert next(row for row in assigned if row["geometry_id"] == reference)[
        "partition"
    ] == "train"
    assert geometry_entropy_bits(np.repeat(masks[0][None], 12, axis=0)) == 0.0
    assert geometry_entropy_bits(np.stack(masks)) > 4.0


def test_acquisition_conditioner_is_joint_permutation_invariant() -> None:
    x = synthetic_inputs()
    conditioner = AcquisitionSetConditioner(
        view_count=9,
        mask_channel_start=3,
        angle_sin_channel_start=24,
        angle_cos_channel_start=33,
        output_features=6,
    )
    masks, sin, cos = conditioner.components(x)
    permutation = torch.tensor([8, 2, 5, 0, 7, 1, 6, 4, 3])
    embedding, modulation = conditioner.encode_components(masks, sin, cos)
    permuted_embedding, permuted_modulation = conditioner.encode_components(
        masks[:, permutation], sin[:, permutation], cos[:, permutation]
    )
    assert torch.allclose(embedding, permuted_embedding, atol=1e-6)
    assert torch.allclose(modulation, permuted_modulation, atol=1e-6)


def test_gc_sro_exact_fallback_and_descriptor_controls() -> None:
    torch.manual_seed(7)
    x = synthetic_inputs()
    model = make_gc_sro()
    model.eval()
    with torch.no_grad():
        base = model.base_operator(x)
        output = model(x)
        geometry_embedding, _ = model.descriptor_embedding(x, mode="geometry")
        mask_embedding, _ = model.descriptor_embedding(x, mode="mask_only")
        static_embedding, _ = model.descriptor_embedding(x, mode="static")
        shuffled_embedding, _ = model.descriptor_embedding(x, mode="shuffled")
    assert torch.equal(base, output)
    assert not torch.allclose(geometry_embedding, mask_embedding)
    assert not torch.allclose(geometry_embedding, static_embedding)
    assert not torch.allclose(geometry_embedding, shuffled_embedding)
    assert all(not parameter.requires_grad for parameter in model.base_operator.parameters())


def test_gc_sro_learns_correction_without_base_drift() -> None:
    torch.manual_seed(11)
    x = synthetic_inputs()
    target = torch.randn(4, 1, 8, 16, 16)
    model = make_gc_sro()
    frozen = {
        name: value.detach().clone() for name, value in model.base_operator.state_dict().items()
    }
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.0)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        torch.mean((model(x) - target) ** 2).backward()
        optimizer.step()
    with torch.no_grad():
        correction, _ = model.correction(x)
    assert float(torch.linalg.vector_norm(correction)) > 0.0
    assert max(
        float(torch.max(torch.abs(value - frozen[name])))
        for name, value in model.base_operator.state_dict().items()
    ) == 0.0


def test_gc_sro_accepts_explicit_deranged_descriptor_components() -> None:
    torch.manual_seed(17)
    x = synthetic_inputs()
    model = make_gc_sro()
    correct = model.conditioner.components(x)
    wrong = tuple(torch.roll(value, shifts=1, dims=0) for value in correct)
    with torch.no_grad():
        correct_embedding, _ = model.descriptor_embedding(
            x, descriptor_components=correct
        )
        wrong_embedding, _ = model.descriptor_embedding(
            x, descriptor_components=wrong
        )
        base = model.base_operator(x)
        wrong_output = model(x, descriptor_components=wrong)
    assert not torch.allclose(correct_embedding, wrong_embedding)
    assert torch.equal(base, wrong_output)
