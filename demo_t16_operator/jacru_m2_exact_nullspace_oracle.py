"""Exact dense null-space oracle for at-most-12^3 toy diagnostics.

This module is intentionally independent of the training and reconstruction
pipeline.  It builds (or accepts) a dense linear matrix and uses a full CPU
float64 SVD to split a caller-provided correction into numerical row-space and
null-space components.  It never consumes a reference field or an evaluation
score.

This is an oracle for small headroom studies, not a deployable reconstruction
algorithm.  In particular, the kernel of a finite-aperture, discretized
approximate inverse operator ``A`` is not necessarily the null space of the
true optical forward process.  A correction that is invisible to this ``A``
can still be visible to an independent renderer or a real experiment.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Literal

import torch


Tensor = torch.Tensor
TOY_VOXEL_LIMIT = 12**3


@dataclass(frozen=True)
class ExactDenseNullspaceResult:
    """Numerical row/null decomposition and its internal consistency checks.

    All tensor fields are detached CPU float64 tensors.  Numerical rank uses
    the strict rule ``singular_value > rank_tolerance``, where
    ``rank_tolerance = max(rank_atol, rank_rtol * largest_singular_value)``.

    ``internal_projection_residual`` is the maximum of the normalized
    decomposition, null-space observability, row-space membership, and
    row/null orthogonality residuals reported alongside it.
    """

    row_space_correction: Tensor
    null_space_correction: Tensor
    rank: int
    singular_values: Tensor
    rank_tolerance: float
    rank_relative_tolerance: float
    rank_absolute_tolerance: float
    internal_projection_residual: float
    decomposition_residual: float
    nullspace_residual: float
    rowspace_residual: float
    orthogonality_residual: float
    measurement_count: int
    active_voxel_count: int


@dataclass(frozen=True)
class ExactDenseAffineProjectionResult:
    """Nearest dense-toy field after projection toward one observation."""

    projected_field: Tensor
    removed_row_space_correction: Tensor
    target_residual: Tensor
    rank: int
    measurement_count: int
    active_voxel_count: int
    initial_target_defect_norm: float
    final_target_residual_norm: float
    relative_target_residual: float
    normal_equation_residual: float
    rowspace_residual: float
    internal_projection_residual: float


@dataclass(frozen=True)
class ExactDenseNullspaceProjector:
    """Reusable numerical row-space projector for one frozen toy geometry."""

    dense_active_matrix: Tensor
    support_mask: Tensor
    left_basis: Tensor
    row_basis: Tensor
    rank: int
    singular_values: Tensor
    rank_tolerance: float
    rank_relative_tolerance: float
    rank_absolute_tolerance: float

    def project(self, correction: Tensor) -> ExactDenseNullspaceResult:
        """Split one correction without repeating the geometry SVD."""

        correction_cpu = _validated_correction(
            correction,
            support_shape=tuple(self.support_mask.shape),
            support_mask=self.support_mask,
        )
        active_correction = correction_cpu.masked_select(self.support_mask)
        active_count = active_correction.numel()
        if self.rank == 0:
            row_active = torch.zeros_like(active_correction)
            null_active = active_correction.clone()
        else:
            row_active = self.row_basis.mT @ (self.row_basis @ active_correction)
            null_active = active_correction - row_active

        if self.rank:
            row_reprojection = self.row_basis.mT @ (self.row_basis @ row_active)
        else:
            row_reprojection = torch.zeros_like(row_active)
        largest = float(self.singular_values[0]) if self.singular_values.numel() else 0.0
        eps = torch.finfo(torch.float64).eps
        correction_norm = float(torch.linalg.vector_norm(active_correction))
        row_norm = float(torch.linalg.vector_norm(row_active))
        null_norm = float(torch.linalg.vector_norm(null_active))
        decomposition_residual = float(
            torch.linalg.vector_norm(active_correction - row_active - null_active)
        ) / max(correction_norm, eps)
        nullspace_residual = float(
            torch.linalg.vector_norm(self.dense_active_matrix @ null_active)
        ) / max(largest * correction_norm, eps)
        rowspace_residual = float(
            torch.linalg.vector_norm(row_active - row_reprojection)
        ) / max(row_norm, eps)
        orthogonality_residual = abs(float(torch.dot(row_active, null_active))) / max(
            row_norm * null_norm,
            eps,
        )
        internal_projection_residual = max(
            decomposition_residual,
            nullspace_residual,
            rowspace_residual,
            orthogonality_residual,
        )
        row_full = torch.zeros_like(correction_cpu)
        null_full = torch.zeros_like(correction_cpu)
        row_full.masked_scatter_(self.support_mask, row_active)
        null_full.masked_scatter_(self.support_mask, null_active)
        return ExactDenseNullspaceResult(
            row_space_correction=row_full,
            null_space_correction=null_full,
            rank=self.rank,
            singular_values=self.singular_values.detach().clone(),
            rank_tolerance=self.rank_tolerance,
            rank_relative_tolerance=self.rank_relative_tolerance,
            rank_absolute_tolerance=self.rank_absolute_tolerance,
            internal_projection_residual=internal_projection_residual,
            decomposition_residual=decomposition_residual,
            nullspace_residual=nullspace_residual,
            rowspace_residual=rowspace_residual,
            orthogonality_residual=orthogonality_residual,
            measurement_count=self.dense_active_matrix.shape[0],
            active_voxel_count=active_count,
        )

    def project_field_to_observation(
        self,
        *,
        field: Tensor,
        observation: Tensor,
    ) -> ExactDenseAffineProjectionResult:
        """Return the exact dense affine least-squares projection.

        The operation never reads field truth.  If part of the observation is
        outside ``range(A)``, that unreachable residual remains and the output
        satisfies the least-squares normal equation in float64 precision.
        """

        field_cpu = _validated_correction(
            field,
            support_shape=tuple(self.support_mask.shape),
            support_mask=self.support_mask,
        )
        if not isinstance(observation, Tensor) or observation.numel() < 1:
            raise ValueError("observation must be one nonempty tensor")
        if observation.is_complex() or not observation.dtype.is_floating_point:
            raise TypeError("observation must use a real floating dtype")
        target = observation.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        if target.numel() != self.dense_active_matrix.shape[0]:
            raise ValueError("observation size must match the dense matrix rows")
        if not bool(torch.all(torch.isfinite(target))):
            raise ValueError("observation must contain only finite values")

        active_field = field_cpu.masked_select(self.support_mask)
        defect = self.dense_active_matrix @ active_field - target
        if self.rank:
            coefficients = self.left_basis @ defect
            scaled = coefficients / self.singular_values[: self.rank]
            removed_active = self.row_basis.mT @ scaled
        else:
            removed_active = torch.zeros_like(active_field)
        projected_active = active_field - removed_active
        residual = self.dense_active_matrix @ projected_active - target

        if self.rank:
            row_reprojection = self.row_basis.mT @ (
                self.row_basis @ removed_active
            )
        else:
            row_reprojection = torch.zeros_like(removed_active)
        eps = torch.finfo(torch.float64).eps
        largest = float(self.singular_values[0]) if self.singular_values.numel() else 0.0
        defect_norm = float(torch.linalg.vector_norm(defect))
        residual_norm = float(torch.linalg.vector_norm(residual))
        removed_norm = float(torch.linalg.vector_norm(removed_active))
        normal = self.dense_active_matrix.mT @ residual
        normal_equation_residual = float(torch.linalg.vector_norm(normal)) / max(
            largest * max(residual_norm, defect_norm),
            eps,
        )
        rowspace_residual = float(
            torch.linalg.vector_norm(removed_active - row_reprojection)
        ) / max(removed_norm, eps)
        internal = max(normal_equation_residual, rowspace_residual)

        projected_full = torch.zeros_like(field_cpu)
        removed_full = torch.zeros_like(field_cpu)
        projected_full.masked_scatter_(self.support_mask, projected_active)
        removed_full.masked_scatter_(self.support_mask, removed_active)
        return ExactDenseAffineProjectionResult(
            projected_field=projected_full,
            removed_row_space_correction=removed_full,
            target_residual=residual,
            rank=self.rank,
            measurement_count=self.dense_active_matrix.shape[0],
            active_voxel_count=active_field.numel(),
            initial_target_defect_norm=defect_norm,
            final_target_residual_norm=residual_norm,
            relative_target_residual=residual_norm / max(defect_norm, eps),
            normal_equation_residual=normal_equation_residual,
            rowspace_residual=rowspace_residual,
            internal_projection_residual=internal,
        )


def _validated_support(support: Tensor) -> tuple[Tensor, Tensor]:
    if not isinstance(support, Tensor) or support.ndim != 3:
        raise ValueError("support must be one three-dimensional tensor")
    if support.is_complex():
        raise TypeError("support must be real")
    if support.numel() > TOY_VOXEL_LIMIT:
        raise ValueError("dense oracle is limited to at most 12^3 voxels")
    support_cpu = support.detach().cpu()
    if support_cpu.dtype.is_floating_point and not bool(
        torch.all(torch.isfinite(support_cpu))
    ):
        raise ValueError("support must contain only finite values")
    if not bool(torch.all((support_cpu == 0) | (support_cpu == 1))):
        raise ValueError("support must be strictly binary")
    mask = support_cpu.to(dtype=torch.bool)
    if not bool(torch.any(mask)):
        raise ValueError("support must retain at least one voxel")
    return support_cpu, mask


def _validated_correction(
    correction: Tensor,
    *,
    support_shape: tuple[int, ...],
    support_mask: Tensor,
) -> Tensor:
    if not isinstance(correction, Tensor) or correction.ndim != 3:
        raise ValueError("correction must be one three-dimensional tensor")
    if tuple(correction.shape) != support_shape:
        raise ValueError("correction must match support shape")
    if correction.is_complex() or not correction.dtype.is_floating_point:
        raise TypeError("correction must use a real floating dtype")
    correction_cpu = correction.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.all(torch.isfinite(correction_cpu))):
        raise ValueError("correction must contain only finite values")
    if bool(torch.any(correction_cpu.masked_select(~support_mask) != 0.0)):
        raise ValueError("correction must be exactly zero outside binary support")
    return correction_cpu


def _validated_dense_matrix(
    dense_matrix: Tensor,
    *,
    support_mask: Tensor,
) -> Tensor:
    if not isinstance(dense_matrix, Tensor) or dense_matrix.ndim != 2:
        raise ValueError("dense_matrix must be one two-dimensional tensor")
    if dense_matrix.is_complex() or not dense_matrix.dtype.is_floating_point:
        raise TypeError("dense_matrix must use a real floating dtype")
    matrix = dense_matrix.detach().to(device="cpu", dtype=torch.float64)
    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("dense_matrix must have at least one row and one column")
    if not bool(torch.all(torch.isfinite(matrix))):
        raise ValueError("dense_matrix must contain only finite values")

    full_count = support_mask.numel()
    active_count = int(torch.count_nonzero(support_mask))
    if matrix.shape[1] == full_count:
        return matrix[:, support_mask.reshape(-1)]
    if matrix.shape[1] == active_count:
        return matrix
    raise ValueError(
        "dense_matrix columns must equal either full or active voxel count"
    )


def _resolve_forward(
    *,
    forward: Callable[[Tensor], Tensor] | None,
    operator: object | None,
) -> Callable[[Tensor], Tensor]:
    if (forward is None) == (operator is None):
        raise ValueError("provide exactly one of forward or operator")
    if forward is not None:
        if not callable(forward):
            raise TypeError("forward must be callable")
        return forward
    candidate = getattr(operator, "forward", None)
    if not callable(candidate):
        raise TypeError("operator must expose a callable forward method")
    return candidate


def _validated_forward_output(
    value: Tensor,
    *,
    expected_shape: tuple[int, ...] | None,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError("linear forward must return a tensor")
    if value.is_complex() or not value.dtype.is_floating_point:
        raise TypeError("linear forward output must use a real floating dtype")
    output = value.detach().to(device="cpu", dtype=torch.float64)
    if output.numel() < 1:
        raise ValueError("linear forward output cannot be empty")
    if expected_shape is not None and tuple(output.shape) != expected_shape:
        raise ValueError("linear forward output shape changed between basis calls")
    if not bool(torch.all(torch.isfinite(output))):
        raise ValueError("linear forward output must contain only finite values")
    return output


def assemble_dense_operator_matrix(
    *,
    support: Tensor,
    forward: Callable[[Tensor], Tensor] | None = None,
    operator: object | None = None,
    forward_input_layout: Literal["spatial", "batch_channel"] = "spatial",
    forward_dtype: torch.dtype = torch.float64,
    forward_device: torch.device | str | None = None,
    zero_atol: float | None = None,
) -> Tensor:
    """Assemble active-support columns of a caller-provided linear map.

    ``forward_input_layout='spatial'`` supplies each basis field as ``[z,y,x]``;
    ``'batch_channel'`` supplies ``[1,1,z,y,x]`` for reconstruction operators.
    The callable is evaluated once at zero to reject an affine offset and once
    per active support voxel.  The returned matrix is detached CPU float64.

    This routine assumes the caller's map is linear; the zero check rejects an
    obvious affine map but is not a proof of linearity.
    """

    _, support_mask = _validated_support(support)
    linear_forward = _resolve_forward(forward=forward, operator=operator)
    if forward_input_layout not in {"spatial", "batch_channel"}:
        raise ValueError("unknown forward_input_layout")
    if not isinstance(forward_dtype, torch.dtype):
        raise TypeError("forward_dtype must be a torch dtype")
    if not forward_dtype.is_floating_point or forward_dtype.is_complex:
        raise TypeError("forward_dtype must be a real floating dtype")
    device = torch.device(forward_device or support.device)

    if zero_atol is None:
        zero_tolerance = 32.0 * torch.finfo(forward_dtype).eps
    else:
        zero_tolerance = float(zero_atol)
        if not math.isfinite(zero_tolerance) or zero_tolerance < 0.0:
            raise ValueError("zero_atol must be finite and non-negative")

    def encoded(field: Tensor) -> Tensor:
        if forward_input_layout == "spatial":
            return field
        return field[None, None]

    field_shape = tuple(support_mask.shape)
    with torch.no_grad():
        zero_field = torch.zeros(
            field_shape,
            dtype=forward_dtype,
            device=device,
        )
        zero_output = _validated_forward_output(
            linear_forward(encoded(zero_field)),
            expected_shape=None,
        )
        if float(torch.max(torch.abs(zero_output))) > zero_tolerance:
            raise ValueError("linear forward must map zero to zero")

        columns: list[Tensor] = []
        for flat_index in torch.nonzero(
            support_mask.reshape(-1),
            as_tuple=False,
        ).reshape(-1):
            basis = torch.zeros(
                field_shape,
                dtype=forward_dtype,
                device=device,
            )
            basis.reshape(-1)[int(flat_index)] = 1.0
            output = _validated_forward_output(
                linear_forward(encoded(basis)),
                expected_shape=tuple(zero_output.shape),
            )
            columns.append(output.reshape(-1))

    return torch.stack(columns, dim=1).contiguous()


def build_exact_dense_nullspace_projector(
    *,
    support: Tensor,
    dense_matrix: Tensor | None = None,
    forward: Callable[[Tensor], Tensor] | None = None,
    operator: object | None = None,
    forward_input_layout: Literal["spatial", "batch_channel"] = "spatial",
    forward_dtype: torch.dtype = torch.float64,
    forward_device: torch.device | str | None = None,
    zero_atol: float | None = None,
    rank_rtol: float | None = None,
    rank_atol: float = 0.0,
) -> ExactDenseNullspaceProjector:
    """Factor one frozen toy geometry for repeated truth-free projections."""

    _, support_mask = _validated_support(support)
    source_count = int(dense_matrix is not None) + int(forward is not None) + int(
        operator is not None
    )
    if source_count != 1:
        raise ValueError("provide exactly one matrix source")
    if dense_matrix is None:
        assembled = assemble_dense_operator_matrix(
            support=support,
            forward=forward,
            operator=operator,
            forward_input_layout=forward_input_layout,
            forward_dtype=forward_dtype,
            forward_device=forward_device,
            zero_atol=zero_atol,
        )
    else:
        assembled = dense_matrix
    matrix = _validated_dense_matrix(assembled, support_mask=support_mask)
    if rank_rtol is None:
        relative_tolerance = max(matrix.shape) * torch.finfo(torch.float64).eps
    else:
        relative_tolerance = float(rank_rtol)
        if not math.isfinite(relative_tolerance) or relative_tolerance < 0.0:
            raise ValueError("rank_rtol must be finite and non-negative")
    absolute_tolerance = float(rank_atol)
    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
        raise ValueError("rank_atol must be finite and non-negative")
    left_vectors, singular_values, right_vectors_h = torch.linalg.svd(
        matrix,
        full_matrices=False,
    )
    largest = float(singular_values[0]) if singular_values.numel() else 0.0
    rank_tolerance = max(absolute_tolerance, relative_tolerance * largest)
    rank = int(torch.count_nonzero(singular_values > rank_tolerance))
    row_basis = right_vectors_h[:rank].detach().clone()
    return ExactDenseNullspaceProjector(
        dense_active_matrix=matrix.detach().clone(),
        support_mask=support_mask.detach().clone(),
        left_basis=left_vectors[:, :rank].mT.detach().clone(),
        row_basis=row_basis,
        rank=rank,
        singular_values=singular_values.detach().clone(),
        rank_tolerance=rank_tolerance,
        rank_relative_tolerance=relative_tolerance,
        rank_absolute_tolerance=absolute_tolerance,
    )


def exact_dense_nullspace_oracle(
    *,
    correction: Tensor,
    support: Tensor,
    dense_matrix: Tensor | None = None,
    forward: Callable[[Tensor], Tensor] | None = None,
    operator: object | None = None,
    forward_input_layout: Literal["spatial", "batch_channel"] = "spatial",
    forward_dtype: torch.dtype = torch.float64,
    forward_device: torch.device | str | None = None,
    zero_atol: float | None = None,
    rank_rtol: float | None = None,
    rank_atol: float = 0.0,
) -> ExactDenseNullspaceResult:
    """Return the exact SVD oracle split for one support-limited correction.

    Supply exactly one matrix source: ``dense_matrix``, ``forward``, or an
    ``operator`` exposing ``forward``.  A dense matrix may contain one column
    per full voxel or one column per active support voxel.  In the former case
    inactive columns are removed before the SVD.

    The default relative rank tolerance is
    ``max(measurements, active_voxels) * eps(float64)``.  Singular values equal
    to the final tolerance are treated as numerical null-space values.
    """

    _, support_mask = _validated_support(support)
    correction_cpu = _validated_correction(
        correction,
        support_shape=tuple(support_mask.shape),
        support_mask=support_mask,
    )
    source_count = int(dense_matrix is not None) + int(forward is not None) + int(
        operator is not None
    )
    if source_count != 1:
        raise ValueError("provide exactly one matrix source")
    if dense_matrix is None:
        assembled = assemble_dense_operator_matrix(
            support=support,
            forward=forward,
            operator=operator,
            forward_input_layout=forward_input_layout,
            forward_dtype=forward_dtype,
            forward_device=forward_device,
            zero_atol=zero_atol,
        )
    else:
        assembled = dense_matrix
    matrix = _validated_dense_matrix(assembled, support_mask=support_mask)

    if rank_rtol is None:
        relative_tolerance = max(matrix.shape) * torch.finfo(torch.float64).eps
    else:
        relative_tolerance = float(rank_rtol)
        if not math.isfinite(relative_tolerance) or relative_tolerance < 0.0:
            raise ValueError("rank_rtol must be finite and non-negative")
    absolute_tolerance = float(rank_atol)
    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
        raise ValueError("rank_atol must be finite and non-negative")

    _, singular_values, right_vectors_h = torch.linalg.svd(
        matrix,
        full_matrices=True,
    )
    largest_singular_value = (
        float(singular_values[0]) if singular_values.numel() else 0.0
    )
    rank_tolerance = max(
        absolute_tolerance,
        relative_tolerance * largest_singular_value,
    )
    rank = int(torch.count_nonzero(singular_values > rank_tolerance))

    active_correction = correction_cpu.masked_select(support_mask)
    active_count = active_correction.numel()
    if rank == 0:
        row_active = torch.zeros_like(active_correction)
        null_active = active_correction.clone()
    elif rank == active_count:
        row_active = active_correction.clone()
        null_active = torch.zeros_like(active_correction)
    else:
        row_basis = right_vectors_h[:rank]
        row_active = row_basis.mT @ (row_basis @ active_correction)
        null_active = active_correction - row_active

    if rank:
        row_basis = right_vectors_h[:rank]
        row_reprojection = row_basis.mT @ (row_basis @ row_active)
    else:
        row_reprojection = torch.zeros_like(row_active)

    eps = torch.finfo(torch.float64).eps
    correction_norm = float(torch.linalg.vector_norm(active_correction))
    row_norm = float(torch.linalg.vector_norm(row_active))
    null_norm = float(torch.linalg.vector_norm(null_active))
    decomposition_residual = float(
        torch.linalg.vector_norm(active_correction - row_active - null_active)
    ) / max(correction_norm, eps)
    nullspace_residual = float(
        torch.linalg.vector_norm(matrix @ null_active)
    ) / max(largest_singular_value * correction_norm, eps)
    rowspace_residual = float(
        torch.linalg.vector_norm(row_active - row_reprojection)
    ) / max(row_norm, eps)
    orthogonality_residual = abs(float(torch.dot(row_active, null_active))) / max(
        row_norm * null_norm,
        eps,
    )
    internal_projection_residual = max(
        decomposition_residual,
        nullspace_residual,
        rowspace_residual,
        orthogonality_residual,
    )

    row_full = torch.zeros_like(correction_cpu)
    null_full = torch.zeros_like(correction_cpu)
    row_full.masked_scatter_(support_mask, row_active)
    null_full.masked_scatter_(support_mask, null_active)

    return ExactDenseNullspaceResult(
        row_space_correction=row_full,
        null_space_correction=null_full,
        rank=rank,
        singular_values=singular_values.detach().clone(),
        rank_tolerance=rank_tolerance,
        rank_relative_tolerance=relative_tolerance,
        rank_absolute_tolerance=absolute_tolerance,
        internal_projection_residual=internal_projection_residual,
        decomposition_residual=decomposition_residual,
        nullspace_residual=nullspace_residual,
        rowspace_residual=rowspace_residual,
        orthogonality_residual=orthogonality_residual,
        measurement_count=matrix.shape[0],
        active_voxel_count=active_count,
    )


__all__ = [
    "ExactDenseAffineProjectionResult",
    "ExactDenseNullspaceResult",
    "ExactDenseNullspaceProjector",
    "TOY_VOXEL_LIMIT",
    "assemble_dense_operator_matrix",
    "build_exact_dense_nullspace_projector",
    "exact_dense_nullspace_oracle",
]
