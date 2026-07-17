"""Exact coordinate embedding and tiny zero-coupling reduction for PSU-B0.

The coordinate gauge is deliberately restrictive: a support is a finite,
strictly binary ``[Z,Y,X]`` mask, so its embedding ``E`` consists only of
columns of the identity.  The dense matrices exposed here are audit objects;
``embed_active`` and ``restrict_active`` use the same stable flat indices.

The reduction ledger is a small CPU/float64 reference utility.  It removes
only exactly zero rows and columns.  No tolerance or epsilon participates in
the activity decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


ACTIVE_COORDINATES_SCHEMA = "psu-b0-active-coordinates-1.0"


def _finite_cpu_f64(value: Any, *, name: str) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if tensor.is_complex():
        raise ValueError(f"{name} must be real")
    try:
        finite = torch.isfinite(tensor)
    except RuntimeError as error:
        raise ValueError(f"{name} must be numeric") from error
    if not bool(torch.all(finite)):
        raise ValueError(f"{name} must contain only finite values")
    converted = tensor.to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.all(torch.isfinite(converted))):
        raise ValueError(f"{name} must remain finite in float64")
    return converted


def _index_vector(mask: torch.Tensor) -> torch.Tensor:
    return torch.nonzero(mask, as_tuple=False).flatten().to(torch.int64)


@dataclass(frozen=True)
class CoordinateSupportGauge:
    """Strict coordinate support with explicit ``E`` and ``E.T`` matrices."""

    support: torch.Tensor
    active_indices: torch.Tensor
    E: torch.Tensor

    @classmethod
    def from_support(cls, support: Any) -> "CoordinateSupportGauge":
        mask = _finite_cpu_f64(support, name="support")
        if mask.ndim != 3 or any(int(size) < 1 for size in mask.shape):
            raise ValueError("support must have non-empty shape [Z,Y,X]")
        if not bool(torch.all((mask == 0.0) | (mask == 1.0))):
            raise ValueError("support must be strictly binary coordinate support")

        active = _index_vector(mask.reshape(-1) == 1.0)
        if active.numel() == 0:
            raise ValueError("coordinate support must contain an active coordinate")

        embedding = torch.zeros(
            (mask.numel(), active.numel()),
            dtype=torch.float64,
            device="cpu",
        )
        embedding[active, torch.arange(active.numel(), dtype=torch.int64)] = 1.0
        return cls(
            support=mask,
            active_indices=active.contiguous(),
            E=embedding,
        )

    def __post_init__(self) -> None:
        support = _finite_cpu_f64(self.support, name="support")
        active = torch.as_tensor(self.active_indices, dtype=torch.int64, device="cpu")
        embedding = _finite_cpu_f64(self.E, name="E")
        if support.ndim != 3 or any(int(size) < 1 for size in support.shape):
            raise ValueError("support must have non-empty shape [Z,Y,X]")
        if not bool(torch.all((support == 0.0) | (support == 1.0))):
            raise ValueError("support must be strictly binary coordinate support")
        expected = _index_vector(support.reshape(-1) == 1.0)
        if expected.numel() == 0:
            raise ValueError("coordinate support must contain an active coordinate")
        if active.ndim != 1 or not torch.equal(active, expected):
            raise ValueError("active_indices must be the stable support coordinates")
        expected_E = torch.zeros(
            (support.numel(), expected.numel()), dtype=torch.float64
        )
        expected_E[expected, torch.arange(expected.numel(), dtype=torch.int64)] = 1.0
        if embedding.shape != expected_E.shape or not torch.equal(embedding, expected_E):
            raise ValueError("E must be the exact coordinate embedding")
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "active_indices", active.contiguous())
        object.__setattr__(self, "E", embedding)

    @property
    def ET(self) -> torch.Tensor:
        """Return the exact dense transpose of ``E``."""

        return self.E.T

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return tuple(int(size) for size in self.support.shape)

    @property
    def full_primal_count(self) -> int:
        return int(self.support.numel())

    @property
    def n_active(self) -> int:
        return int(self.active_indices.numel())

    def embed_active(self, active: Any) -> torch.Tensor:
        """Apply ``E`` to ``[B,n_active]`` and return ``[B,1,Z,Y,X]``."""

        values = _finite_cpu_f64(active, name="active")
        if values.ndim != 2 or values.shape[1] != self.n_active:
            raise ValueError("active must have shape [B,n_active]")
        if values.shape[0] < 1:
            raise ValueError("active batch must be non-empty")
        full = values @ self.ET
        return full.reshape(values.shape[0], 1, *self.grid_shape)

    def restrict_active(self, full: Any) -> torch.Tensor:
        """Apply ``E.T`` to ``[B,1,Z,Y,X]`` and return ``[B,n_active]``."""

        values = _finite_cpu_f64(full, name="full")
        expected = (1, *self.grid_shape)
        if values.ndim != 5 or tuple(values.shape[1:]) != expected:
            raise ValueError("full must have shape [B,1,Z,Y,X]")
        if values.shape[0] < 1:
            raise ValueError("full batch must be non-empty")
        return values.flatten(1) @ self.E


def build_coordinate_support_gauge(support: Any) -> CoordinateSupportGauge:
    """Build a validated coordinate gauge from a strict binary support."""

    return CoordinateSupportGauge.from_support(support)


@dataclass(frozen=True)
class ZeroCouplingReductionLedger:
    """Auditable exact-zero reduction of ``0.5 * ||K x - target||^2``.

    The reduced variable is a correction around ``fixed_full``.  Therefore
    ``recover_full_primal(x_active)`` returns ``fixed_full + E x_active`` and
    the reduced target is ``target - K @ fixed_full``.
    """

    K: torch.Tensor
    target: torch.Tensor
    fixed_full: torch.Tensor
    original_data_indices: torch.Tensor
    active_data_indices: torch.Tensor
    deleted_data_indices: torch.Tensor
    original_primal_indices: torch.Tensor
    active_primal_indices: torch.Tensor
    deleted_primal_indices: torch.Tensor
    fixed_data_offset: torch.Tensor
    target_shifted: torch.Tensor
    K_active: torch.Tensor
    target_active: torch.Tensor
    deleted_target_shifted: torch.Tensor
    deleted_data_objective_constant: torch.Tensor

    @property
    def data_count(self) -> int:
        return int(self.K.shape[0])

    @property
    def full_primal_count(self) -> int:
        return int(self.K.shape[1])

    @property
    def active_primal_count(self) -> int:
        return int(self.active_primal_indices.numel())

    def recover_full_primal(self, active_primal: Any) -> torch.Tensor:
        """Scatter active corrections into one vector or a batch of vectors."""

        active = _finite_cpu_f64(active_primal, name="active_primal")
        if active.ndim == 1:
            if active.shape != (self.active_primal_count,):
                raise ValueError("active_primal has the wrong size")
            full = self.fixed_full.clone()
            full[self.active_primal_indices] += active
            return full
        if active.ndim == 2:
            if active.shape[0] < 1 or active.shape[1] != self.active_primal_count:
                raise ValueError("active_primal must have shape [B,n_active]")
            full = self.fixed_full.expand(active.shape[0], -1).clone()
            full[:, self.active_primal_indices] += active
            return full
        raise ValueError("active_primal must be one- or two-dimensional")

    def reduced_objective(self, active_primal: Any) -> torch.Tensor:
        """Evaluate the reduced quadratic plus the deleted-row constant."""

        active = _finite_cpu_f64(active_primal, name="active_primal")
        if active.ndim == 1:
            if active.shape != (self.active_primal_count,):
                raise ValueError("active_primal has the wrong size")
            residual = self.K_active @ active - self.target_active
            return 0.5 * torch.sum(residual.square()) + self.deleted_data_objective_constant
        if active.ndim == 2:
            if active.shape[0] < 1 or active.shape[1] != self.active_primal_count:
                raise ValueError("active_primal must have shape [B,n_active]")
            residual = active @ self.K_active.T - self.target_active
            return 0.5 * torch.sum(residual.square(), dim=1) + self.deleted_data_objective_constant
        raise ValueError("active_primal must be one- or two-dimensional")


def reduce_zero_coupling_system(
    K: Any,
    target: Any,
    *,
    fixed_full: Any | None = None,
) -> ZeroCouplingReductionLedger:
    """Delete exactly zero data rows and primal columns from a tiny dense map."""

    matrix = _finite_cpu_f64(K, name="K")
    rhs = _finite_cpu_f64(target, name="target")
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("K must be a non-empty matrix")
    if rhs.ndim != 1 or rhs.shape != (matrix.shape[0],):
        raise ValueError("target must have shape [data_count]")
    if fixed_full is None:
        fixed = torch.zeros(matrix.shape[1], dtype=torch.float64)
    else:
        fixed = _finite_cpu_f64(fixed_full, name="fixed_full")
        if fixed.ndim != 1 or fixed.shape != (matrix.shape[1],):
            raise ValueError("fixed_full must have shape [full_primal_count]")

    active_data_mask = torch.any(matrix != 0.0, dim=1)
    active_primal_mask = torch.any(matrix != 0.0, dim=0)
    active_data = _index_vector(active_data_mask)
    active_primal = _index_vector(active_primal_mask)
    if active_data.numel() == 0 or active_primal.numel() == 0:
        raise ValueError("zero-coupling reduction produced an empty active system")

    deleted_data = _index_vector(~active_data_mask)
    deleted_primal = _index_vector(~active_primal_mask)
    fixed_data_offset = matrix @ fixed
    shifted = rhs - fixed_data_offset
    reduced_matrix = matrix.index_select(0, active_data).index_select(1, active_primal)
    reduced_target = shifted.index_select(0, active_data)
    deleted_target = shifted.index_select(0, deleted_data)
    constant = 0.5 * torch.sum(deleted_target.square())

    return ZeroCouplingReductionLedger(
        K=matrix,
        target=rhs,
        fixed_full=fixed,
        original_data_indices=torch.arange(matrix.shape[0], dtype=torch.int64),
        active_data_indices=active_data,
        deleted_data_indices=deleted_data,
        original_primal_indices=torch.arange(matrix.shape[1], dtype=torch.int64),
        active_primal_indices=active_primal,
        deleted_primal_indices=deleted_primal,
        fixed_data_offset=fixed_data_offset,
        target_shifted=shifted,
        K_active=reduced_matrix,
        target_active=reduced_target,
        deleted_target_shifted=deleted_target,
        deleted_data_objective_constant=constant,
    )
