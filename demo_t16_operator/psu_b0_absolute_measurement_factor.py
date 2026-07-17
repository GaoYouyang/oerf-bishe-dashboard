"""Exact signed and absolute camera-whitening factor for PSU-B0 BOST.

The physical operator first projects every sampled gradient into detector
``u/v`` components and then applies a signed detector-whitening matrix.  For
the elementwise PDHG majorizer, the absolute value must be taken after those
two signed maps have been composed::

    W_v = measurement_scale * H_v * R_v * Q_v
    M_v = abs(W_v)

Using ``abs(H_v) * abs(R_v * Q_v)`` would be a different, looser envelope.
This module stores the exact composed per-view kernel and exposes auditable
forward/transpose calls for both the signed and absolute maps.  It does not
construct a solver or make a performance claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn


ABSOLUTE_MEASUREMENT_FACTOR_SCHEMA = (
    "psu-b0-exact-absolute-measurement-factor-1.0"
)
WHITENING_SUPPORT_CONTRACT_SCHEMA = (
    "psu-b0-view-local-independent-whitening-contract-1.0"
)

_CROSS_VIEW_METADATA_FIELDS = (
    "cross_view_covariance",
    "cross_view_coupling",
)
_CROSS_VIEW_SUPPORT_FIELDS = (
    "cross_view_covariance_supported",
    "cross_view_coupling_supported",
)
_VIEW_LOCAL_METADATA_FIELDS = (
    "whitening_block_scope",
    "independent_whitening_blocks",
)
_COVARIANCE_BLOCK_ID_FIELD = "covariance_block_ids"
_WHITENING_CONTRACT_FIELDS = (
    *_CROSS_VIEW_METADATA_FIELDS,
    *_CROSS_VIEW_SUPPORT_FIELDS,
    *_VIEW_LOCAL_METADATA_FIELDS,
    _COVARIANCE_BLOCK_ID_FIELD,
)


def _declared_whitening_metadata(whitening: nn.Module) -> list[tuple[str, Any]]:
    declarations: list[tuple[str, Any]] = []
    for container_name in (
        "whitening_metadata",
        "whitening_contract_metadata",
        "contract_metadata",
        "metadata",
    ):
        container = getattr(whitening, container_name, None)
        if callable(container):
            container = container()
        if container is None:
            continue
        if not isinstance(container, Mapping):
            if container_name in {
                "whitening_metadata",
                "whitening_contract_metadata",
            }:
                raise TypeError(f"{container_name} must be a mapping when declared")
            continue
        for field in _WHITENING_CONTRACT_FIELDS:
            if field in container:
                declarations.append((field, container[field]))

    for field in _WHITENING_CONTRACT_FIELDS:
        if hasattr(whitening, field):
            declarations.append((field, getattr(whitening, field)))
    return declarations


def _validate_view_local_whitening_contract(
    whitening: nn.Module,
    *,
    view_count: int,
) -> None:
    """Reject metadata incompatible with independent per-view blocks."""

    for field, value in _declared_whitening_metadata(whitening):
        if field in (*_CROSS_VIEW_METADATA_FIELDS, *_CROSS_VIEW_SUPPORT_FIELDS):
            if not isinstance(value, bool):
                raise ValueError(f"{field} metadata must be boolean")
            if value:
                raise ValueError(
                    f"{field}=True is unsupported; whitening must be view-local"
                )
            continue

        if field == "whitening_block_scope":
            if value != "view_local":
                raise ValueError(
                    "whitening_block_scope must be exactly 'view_local'"
                )
            continue

        if field == "independent_whitening_blocks":
            if not isinstance(value, bool) or not value:
                raise ValueError(
                    "independent_whitening_blocks must be exactly True"
                )
            continue

        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ValueError("covariance_block_ids must contain one id per view")
        block_ids = tuple(value)
        if len(block_ids) != view_count:
            raise ValueError("covariance_block_ids must contain one id per view")
        try:
            unique_count = len(set(block_ids))
        except TypeError as error:
            raise ValueError(
                "covariance_block_ids must be hashable view-local ids"
            ) from error
        if unique_count != view_count:
            raise ValueError(
                "covariance_block_ids must be unique per view; shared blocks imply "
                "cross-view covariance"
            )


def _finite_tensor(
    value: Any,
    *,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if torch.any(~torch.isfinite(tensor)):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


@dataclass(frozen=True)
class FactorShape:
    view_count: int
    rays_per_view: int
    sample_count: int

    @property
    def ray_count(self) -> int:
        return self.view_count * self.rays_per_view

    @property
    def detector_dimension_per_view(self) -> int:
        return 2 * self.rays_per_view


class ExactAbsoluteMeasurementFactor(nn.Module):
    """Compose ``H``, camera projection, and ray scale before taking ``abs``.

    Sampled gradients use shape ``[batch,3,ray,sample]``.  Detector values use
    shape ``[batch,ray,2]``.  Whitening is block diagonal across views, while
    each view block may couple every detector ray and both ``u/v`` components.
    ``scale_by_view`` is positive and may vary by batch item.
    """

    def __init__(
        self,
        whitening: nn.Module,
        *,
        projection_u_xyz: Any,
        projection_v_xyz: Any,
        ray_scale: Any,
        sample_count: int,
        measurement_scale: float = 1.0,
    ) -> None:
        super().__init__()
        matrix = getattr(whitening, "matrix", None)
        scales = getattr(whitening, "scale_by_view", None)
        view_count = int(getattr(whitening, "view_count", -1))
        rays_per_view = int(getattr(whitening, "rays_per_view", -1))
        if not isinstance(matrix, torch.Tensor) or not isinstance(
            scales, torch.Tensor
        ):
            raise TypeError("whitening must expose tensor matrix and scale_by_view")
        if view_count < 1 or rays_per_view < 1:
            raise ValueError("whitening view and ray counts must be positive")
        _validate_view_local_whitening_contract(
            whitening,
            view_count=view_count,
        )
        sample_count = int(sample_count)
        if sample_count < 1:
            raise ValueError("sample_count must be positive")
        measurement_scale = float(measurement_scale)
        if not math.isfinite(measurement_scale) or measurement_scale <= 0.0:
            raise ValueError("measurement_scale must be finite and positive")

        shape = FactorShape(view_count, rays_per_view, sample_count)
        dtype = matrix.dtype
        device = matrix.device
        expected_detector = shape.detector_dimension_per_view
        if matrix.shape != (view_count, expected_detector, expected_detector):
            raise ValueError("whitening matrix shape does not match declared views")
        if scales.ndim != 2 or scales.shape[1] != view_count:
            raise ValueError("scale_by_view must have shape [batch,view]")
        if torch.any(~torch.isfinite(scales)) or torch.any(scales <= 0.0):
            raise ValueError("scale_by_view must be finite and positive")

        projection_u = _finite_tensor(
            projection_u_xyz,
            name="projection_u_xyz",
            dtype=dtype,
            device=device,
        )
        projection_v = _finite_tensor(
            projection_v_xyz,
            name="projection_v_xyz",
            dtype=dtype,
            device=device,
        )
        if projection_u.shape != (shape.ray_count, 3) or projection_v.shape != (
            shape.ray_count,
            3,
        ):
            raise ValueError("projection vectors must have shape [ray,3]")
        ray_scale_tensor = _finite_tensor(
            ray_scale,
            name="ray_scale",
            dtype=dtype,
            device=device,
        ).reshape(-1)
        if ray_scale_tensor.shape != (shape.ray_count,):
            raise ValueError("ray_scale must contain one value per ray")

        projection = torch.stack((projection_u, projection_v), dim=1).reshape(
            view_count,
            rays_per_view,
            2,
            3,
        )
        matrix_by_ray = matrix.detach().reshape(
            view_count,
            expected_detector,
            rays_per_view,
            2,
        )
        signed_kernel = torch.einsum(
            "vork,vrkc->vorc",
            matrix_by_ray,
            projection,
        )
        signed_kernel = signed_kernel * ray_scale_tensor.reshape(
            view_count,
            1,
            rays_per_view,
            1,
        )
        signed_kernel = signed_kernel * measurement_scale

        self.shape = shape
        self.measurement_scale = measurement_scale
        self.register_buffer("signed_kernel", signed_kernel.contiguous())
        self.register_buffer("absolute_kernel", signed_kernel.abs().contiguous())
        self.register_buffer("scale_by_view", scales.detach().clone())
        self.signed_forward_calls = 0
        self.signed_transpose_calls = 0
        self.absolute_forward_calls = 0
        self.absolute_transpose_calls = 0

    @property
    def view_count(self) -> int:
        return self.shape.view_count

    @property
    def rays_per_view(self) -> int:
        return self.shape.rays_per_view

    @property
    def ray_count(self) -> int:
        return self.shape.ray_count

    @property
    def sample_count(self) -> int:
        return self.shape.sample_count

    @property
    def contract_metadata(self) -> dict[str, Any]:
        """Describe the production whitening boundary enforced at construction."""

        return {
            "schema_version": WHITENING_SUPPORT_CONTRACT_SCHEMA,
            "whitening_block_scope": "view_local",
            "independent_whitening_blocks": True,
            "cross_view_covariance_supported": False,
            "cross_view_coupling_supported": False,
            "covariance_block_ids": tuple(range(self.view_count)),
        }

    def reset_call_counts(self) -> None:
        self.signed_forward_calls = 0
        self.signed_transpose_calls = 0
        self.absolute_forward_calls = 0
        self.absolute_transpose_calls = 0

    def call_report(self) -> dict[str, int]:
        return {
            "signed_forward_calls": int(self.signed_forward_calls),
            "signed_transpose_calls": int(self.signed_transpose_calls),
            "absolute_forward_calls": int(self.absolute_forward_calls),
            "absolute_transpose_calls": int(self.absolute_transpose_calls),
        }

    def _expanded_scale(self, batch_size: int) -> torch.Tensor:
        if self.scale_by_view.shape[0] not in {1, int(batch_size)}:
            raise ValueError("scale_by_view batch must be one or match the input")
        return self.scale_by_view.expand(int(batch_size), -1)

    def _canonical_sampled(self, sampled_gradient: torch.Tensor) -> torch.Tensor:
        values = sampled_gradient.to(self.signed_kernel)
        expected = (3, self.ray_count, self.sample_count)
        if values.ndim != 4 or tuple(values.shape[1:]) != expected:
            raise ValueError(
                "sampled_gradient must have shape [batch,3,ray,sample]"
            )
        if torch.any(~torch.isfinite(values)):
            raise ValueError("sampled_gradient must be finite")
        return values.reshape(
            len(values),
            3,
            self.view_count,
            self.rays_per_view,
            self.sample_count,
        )

    def _canonical_detector(self, detector_values: torch.Tensor) -> torch.Tensor:
        values = detector_values.to(self.signed_kernel)
        if values.ndim != 3 or tuple(values.shape[1:]) != (
            self.ray_count,
            2,
        ):
            raise ValueError("detector_values must have shape [batch,ray,2]")
        if torch.any(~torch.isfinite(values)):
            raise ValueError("detector_values must be finite")
        return values.reshape(
            len(values),
            self.view_count,
            2 * self.rays_per_view,
        )

    def _factor_forward(
        self,
        sampled_gradient: torch.Tensor,
        *,
        kernel: torch.Tensor,
    ) -> torch.Tensor:
        sampled = self._canonical_sampled(sampled_gradient).sum(dim=-1)
        canonical = torch.einsum("vorc,bcvr->bvo", kernel, sampled)
        canonical = canonical / self._expanded_scale(len(sampled))[:, :, None]
        return canonical.reshape(len(sampled), self.ray_count, 2)

    def _factor_transpose(
        self,
        detector_values: torch.Tensor,
        *,
        kernel: torch.Tensor,
    ) -> torch.Tensor:
        detector = self._canonical_detector(detector_values)
        detector = detector / self._expanded_scale(len(detector))[:, :, None]
        sampled_sum = torch.einsum("vorc,bvo->bcvr", kernel, detector)
        sampled = sampled_sum[:, :, :, :, None].expand(
            -1,
            -1,
            -1,
            -1,
            self.sample_count,
        )
        return sampled.reshape(
            len(detector),
            3,
            self.ray_count,
            self.sample_count,
        )

    def signed_forward(self, sampled_gradient: torch.Tensor) -> torch.Tensor:
        self.signed_forward_calls += 1
        return self._factor_forward(sampled_gradient, kernel=self.signed_kernel)

    def signed_transpose(self, detector_values: torch.Tensor) -> torch.Tensor:
        self.signed_transpose_calls += 1
        return self._factor_transpose(detector_values, kernel=self.signed_kernel)

    def absolute_forward(self, sampled_gradient: torch.Tensor) -> torch.Tensor:
        self.absolute_forward_calls += 1
        return self._factor_forward(sampled_gradient, kernel=self.absolute_kernel)

    def absolute_transpose(self, detector_values: torch.Tensor) -> torch.Tensor:
        self.absolute_transpose_calls += 1
        return self._factor_transpose(detector_values, kernel=self.absolute_kernel)

    def one_pass_sums(
        self,
        *,
        batch_index: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return row/column sums using one absolute call in each direction."""

        batch_size = int(self.scale_by_view.shape[0])
        batch_index = int(batch_index)
        if not 0 <= batch_index < batch_size:
            raise IndexError("batch_index is outside scale_by_view")
        sampled_ones = torch.ones(
            (batch_size, 3, self.ray_count, self.sample_count),
            dtype=self.signed_kernel.dtype,
            device=self.signed_kernel.device,
        )
        detector_ones = torch.ones(
            (batch_size, self.ray_count, 2),
            dtype=self.signed_kernel.dtype,
            device=self.signed_kernel.device,
        )
        row_sums = self.absolute_forward(sampled_ones)[batch_index]
        column_sums = self.absolute_transpose(detector_ones)[batch_index]
        return row_sums, column_sums

    def dense_block(
        self,
        view_index: int,
        *,
        batch_index: int = 0,
        absolute: bool = False,
    ) -> torch.Tensor:
        """Materialize one view block for tiny-oracle audits only."""

        view_index = int(view_index)
        batch_index = int(batch_index)
        if not 0 <= view_index < self.view_count:
            raise IndexError("view_index is outside the factor")
        if not 0 <= batch_index < int(self.scale_by_view.shape[0]):
            raise IndexError("batch_index is outside scale_by_view")
        kernel = self.absolute_kernel if absolute else self.signed_kernel
        block = kernel[view_index] / self.scale_by_view[batch_index, view_index]
        return block.permute(0, 2, 1)[:, :, :, None].expand(
            -1,
            -1,
            -1,
            self.sample_count,
        ).reshape(
            2 * self.rays_per_view,
            3 * self.rays_per_view * self.sample_count,
        )


__all__ = [
    "ABSOLUTE_MEASUREMENT_FACTOR_SCHEMA",
    "ExactAbsoluteMeasurementFactor",
    "FactorShape",
    "WHITENING_SUPPORT_CONTRACT_SCHEMA",
]
