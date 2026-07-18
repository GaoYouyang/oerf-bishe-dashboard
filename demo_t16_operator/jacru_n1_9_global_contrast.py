"""Frozen global-K camera-contrast frames for the JACRU N1.9 screen.

The two frames differ only in whether camera contrasts modulate the visible
warm residual or the deployment-visible damping estimate.  Both retain the
same global solver responses ``A P A^T d`` and ``A P A^T r`` and therefore use
exactly two forward and two adjoint setup calls.
"""

from __future__ import annotations

import math
from typing import Callable

import torch

from .jacru_n1_8_camera_ray_hybrid import (
    HybridMeasurementBasis,
    _finite_observation,
    _normal_probes,
    _orthonormalize,
)
from .jacru_synthetic_fixture import JACRUGeometry


Tensor = torch.Tensor
LinearMap = Callable[[Tensor], Tensor]
SUPPORTED_CONTRAST_BASIS_KINDS = (
    "residual_contrast_global_k6_total",
    "damping_contrast_global_k6_total",
)


def three_camera_helmert_contrasts(geometry: JACRUGeometry) -> Tensor:
    """Return the frozen centered orthonormal contrasts for three cameras."""

    if geometry.camera_count != 3:
        raise ValueError("N1.9 is frozen for exactly three cameras")
    labels = geometry.camera_index.to(torch.int64)
    expected = torch.arange(3, dtype=torch.int64)
    if not torch.equal(torch.unique(labels, sorted=True), expected):
        raise ValueError("camera_index labels must be contiguous from zero")
    ray_counts = torch.bincount(labels, minlength=3)
    if not bool(torch.all(ray_counts == ray_counts[0])):
        raise ValueError(
            "unweighted N1.9 contrasts require equal valid ray counts per camera"
        )
    contrasts = torch.tensor(
        (
            (1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0), 0.0),
            (1.0 / math.sqrt(6.0), 1.0 / math.sqrt(6.0), -2.0 / math.sqrt(6.0)),
        ),
        dtype=torch.float64,
    )
    if not torch.allclose(
        contrasts @ contrasts.T,
        torch.eye(2, dtype=torch.float64),
        atol=1e-14,
        rtol=0.0,
    ) or not torch.allclose(
        torch.sum(contrasts, dim=1),
        torch.zeros(2, dtype=torch.float64),
        atol=1e-14,
        rtol=0.0,
    ):
        raise RuntimeError("frozen camera contrasts lost orthonormality or centering")
    return contrasts


def camera_contrast_vectors(
    anchor: Tensor,
    *,
    geometry: JACRUGeometry,
    quantity_name: str,
) -> tuple[tuple[str, Tensor], tuple[str, Tensor]]:
    """Apply the two frozen camera contrasts to one visible observation."""

    observation = _finite_observation(anchor, name=quantity_name)
    if observation.shape != (geometry.ray_count, 2):
        raise ValueError(f"{quantity_name} must have shape [ray,2]")
    contrasts = three_camera_helmert_contrasts(geometry)
    labels = geometry.camera_index.to(torch.int64)
    first = observation * contrasts[0, labels, None]
    second = observation * contrasts[1, labels, None]
    return (
        (f"camera_contrast_1_{quantity_name}", first),
        (f"camera_contrast_2_{quantity_name}", second),
    )


def build_global_contrast_basis(
    *,
    kind: str,
    damping: Tensor,
    warm_residual: Tensor,
    forward: LinearMap,
    adjoint: LinearMap,
    support: Tensor,
    geometry: JACRUGeometry,
    dependence_tolerance: float = 1e-10,
) -> HybridMeasurementBasis:
    """Build one frozen six-vector contrast/global-K basis without truth."""

    basis_kind = str(kind)
    if basis_kind not in SUPPORTED_CONTRAST_BASIS_KINDS:
        raise ValueError(f"unsupported N1.9 basis kind: {basis_kind}")
    anchor = _finite_observation(damping, name="damping")
    residual = _finite_observation(warm_residual, name="warm_residual")
    expected = (geometry.ray_count, 2)
    if anchor.shape != expected or residual.shape != expected:
        raise ValueError(f"damping and warm_residual must have shape {expected}")
    contrast_anchor = (
        residual
        if basis_kind == "residual_contrast_global_k6_total"
        else anchor
    )
    contrast_name = (
        "warm_residual"
        if basis_kind == "residual_contrast_global_k6_total"
        else "damping"
    )
    named: list[tuple[str, Tensor]] = [
        ("damping", anchor),
        ("warm_residual", residual),
    ]
    named.extend(
        camera_contrast_vectors(
            contrast_anchor,
            geometry=geometry,
            quantity_name=contrast_name,
        )
    )
    probes = _normal_probes(
        seeds=(("damping", anchor), ("warm_residual", residual)),
        forward=forward,
        adjoint=adjoint,
        support=support,
    )
    named.extend(probes)
    return _orthonormalize(
        kind=basis_kind,
        named_vectors=named,
        observation_shape=tuple(anchor.shape),
        dependence_tolerance=dependence_tolerance,
        setup_forward_calls=len(probes),
        setup_adjoint_calls=len(probes),
        fit_mode_count=0,
    )


__all__ = [
    "SUPPORTED_CONTRAST_BASIS_KINDS",
    "build_global_contrast_basis",
    "camera_contrast_vectors",
    "three_camera_helmert_contrasts",
]
