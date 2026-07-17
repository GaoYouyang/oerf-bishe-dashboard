"""Matrix-free signed/absolute factor pipeline for PSU-B0 majorizer setup.

The production maps use coordinate scatter/gather rather than the dense audit
embedding stored by :class:`CoordinateSupportGauge`.  This module only builds
factor maps, exact-zero activity masks, and diagonal metric candidates; it does
not construct or run an optimization algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import torch

from .psu_b0_absolute_measurement_factor import ExactAbsoluteMeasurementFactor
from .psu_b0_active_coordinates import CoordinateSupportGauge
from .psu_b0_primal_dual import ForwardNeumannRegularizationOperator
from .psu_b0_reconstruction_interface import (
    PSUB0VoxelGradientOperator,
    absolute_finite_difference_gradient,
    absolute_finite_difference_gradient_adjoint,
    finite_difference_gradient,
    finite_difference_gradient_adjoint,
)


FACTOR_MAJORIZER_PIPELINE_SCHEMA = "psu-b0-factor-majorizer-pipeline-1.0"


@dataclass(frozen=True)
class FactorPipelineCallLedger:
    signed_data_forward_calls: int = 0
    signed_data_transpose_calls: int = 0
    absolute_data_forward_calls: int = 0
    absolute_data_transpose_calls: int = 0
    signed_tv_forward_calls: int = 0
    signed_tv_transpose_calls: int = 0
    absolute_tv_forward_calls: int = 0
    absolute_tv_transpose_calls: int = 0


@dataclass(frozen=True)
class FactorPipelineFreezeToken:
    """Lightweight mutation token for every setup-critical tensor."""

    tensor_records: tuple[tuple[Any, ...], ...]
    measurement_shape: tuple[int, int, int]
    measurement_scale: float
    grid_shape: tuple[int, int, int]
    grid_minimum_xyz: tuple[float, float, float]
    grid_maximum_xyz: tuple[float, float, float]
    voxel_spacing_xyz: tuple[float, float, float]
    regularization_spacing_xyz: tuple[float, float, float]
    content_sha256: str


@dataclass(frozen=True)
class FactorMajorizerSetupFreezeToken:
    """Mutation token for the factor token and every derived setup tensor."""

    factor_freeze_token: FactorPipelineFreezeToken
    tensor_records: tuple[tuple[Any, ...], ...]
    eta: float
    batch_index: int
    content_sha256: str


def _ledger_delta(
    after: FactorPipelineCallLedger,
    before: FactorPipelineCallLedger,
) -> FactorPipelineCallLedger:
    return FactorPipelineCallLedger(
        **{
            name: getattr(after, name) - getattr(before, name)
            for name in FactorPipelineCallLedger.__dataclass_fields__
        }
    )


def _single_call_ledger(name: str) -> FactorPipelineCallLedger:
    if name not in FactorPipelineCallLedger.__dataclass_fields__:
        raise KeyError(f"unknown factor ledger field: {name}")
    return FactorPipelineCallLedger(**{name: 1})


class PSUB0FactorPipeline:
    """Device-aware maps on the declared coordinate-support active space."""

    _LEDGER_FIELDS = tuple(FactorPipelineCallLedger.__dataclass_fields__)

    def __init__(
        self,
        *,
        gauge: CoordinateSupportGauge,
        voxel_operator: PSUB0VoxelGradientOperator,
        measurement_factor: ExactAbsoluteMeasurementFactor,
        regularization_operator: ForwardNeumannRegularizationOperator,
    ) -> None:
        _validate_factor_contract(
            gauge=gauge,
            voxel_operator=voxel_operator,
            measurement_factor=measurement_factor,
            regularization_operator=regularization_operator,
        )
        self.gauge = gauge
        self.voxel_operator = voxel_operator
        self.measurement_factor = measurement_factor
        self.regularization_operator = regularization_operator
        self.device = voxel_operator.sample_weights.device
        self.dtype = voxel_operator.sample_weights.dtype
        self.grid_shape = tuple(int(size) for size in voxel_operator.grid_shape)
        self.n_active = gauge.n_active
        self._active_indices = gauge.active_indices.to(
            device=self.device,
            dtype=torch.int64,
        )
        self._calls = {name: 0 for name in self._LEDGER_FIELDS}

    @property
    def view_count(self) -> int:
        return self.measurement_factor.view_count

    @property
    def rays_per_view(self) -> int:
        return self.measurement_factor.rays_per_view

    @property
    def ray_count(self) -> int:
        return self.measurement_factor.ray_count

    @property
    def sample_count(self) -> int:
        return self.measurement_factor.sample_count

    def reset_call_ledger(self) -> None:
        for name in self._calls:
            self._calls[name] = 0

    def call_ledger(self) -> FactorPipelineCallLedger:
        return FactorPipelineCallLedger(**self._calls)

    def physical_call_ledger(self) -> FactorPipelineCallLedger:
        """Read sealed implementation counters without overridable reports."""

        measurement = self.measurement_factor
        regularization = self.regularization_operator
        counters = (
            measurement.signed_forward_calls,
            measurement.signed_transpose_calls,
            measurement.absolute_forward_calls,
            measurement.absolute_transpose_calls,
            regularization.gradient_calls,
            regularization.gradient_adjoint_calls,
            regularization.absolute_forward_calls,
            regularization.absolute_adjoint_calls,
        )
        if any(not isinstance(value, int) or value < 0 for value in counters):
            raise TypeError("physical factor counters must be nonnegative integers")
        return FactorPipelineCallLedger(
            signed_data_forward_calls=measurement.signed_forward_calls,
            signed_data_transpose_calls=measurement.signed_transpose_calls,
            absolute_data_forward_calls=measurement.absolute_forward_calls,
            absolute_data_transpose_calls=measurement.absolute_transpose_calls,
            signed_tv_forward_calls=regularization.gradient_calls,
            signed_tv_transpose_calls=regularization.gradient_adjoint_calls,
            absolute_tv_forward_calls=regularization.absolute_forward_calls,
            absolute_tv_transpose_calls=regularization.absolute_adjoint_calls,
        )

    def factor_freeze_token(self) -> FactorPipelineFreezeToken:
        tensors = (
            ("gauge.support", self.gauge.support),
            ("gauge.active_indices", self.gauge.active_indices),
            ("voxel.sample_indices", self.voxel_operator.sample_indices),
            ("voxel.sample_weights", self.voxel_operator.sample_weights),
            ("voxel.sample_valid", self.voxel_operator.sample_valid),
            ("voxel.projection_u", self.voxel_operator.projection_u),
            ("voxel.projection_v", self.voxel_operator.projection_v),
            ("voxel.ray_scale", self.voxel_operator.ray_scale),
            ("voxel.support", self.voxel_operator.support),
            ("measurement.signed_kernel", self.measurement_factor.signed_kernel),
            (
                "measurement.absolute_kernel",
                self.measurement_factor.absolute_kernel,
            ),
            ("measurement.scale_by_view", self.measurement_factor.scale_by_view),
            ("pipeline.active_indices", self._active_indices),
        )
        records: list[tuple[Any, ...]] = []
        digest = hashlib.sha256()
        for name, tensor in tensors:
            shape = tuple(int(size) for size in tensor.shape)
            dtype = str(tensor.dtype)
            device = str(tensor.device)
            records.append(
                (
                    name,
                    id(tensor),
                    int(tensor.data_ptr()),
                    int(tensor._version),
                    shape,
                    dtype,
                    device,
                )
            )
            digest.update(name.encode("utf-8"))
            digest.update(repr((shape, dtype, device)).encode("ascii"))
            contiguous_cpu = tensor.detach().contiguous().cpu()
            digest.update(
                contiguous_cpu.view(torch.uint8).numpy().tobytes(order="C")
            )
        return FactorPipelineFreezeToken(
            tensor_records=tuple(records),
            measurement_shape=(
                self.measurement_factor.view_count,
                self.measurement_factor.rays_per_view,
                self.measurement_factor.sample_count,
            ),
            measurement_scale=float(self.measurement_factor.measurement_scale),
            grid_shape=self.grid_shape,
            grid_minimum_xyz=tuple(
                float(value) for value in self.voxel_operator.grid_minimum_xyz
            ),
            grid_maximum_xyz=tuple(
                float(value) for value in self.voxel_operator.grid_maximum_xyz
            ),
            voxel_spacing_xyz=tuple(
                float(value) for value in self.voxel_operator.spacing_xyz
            ),
            regularization_spacing_xyz=tuple(
                float(value)
                for value in self.regularization_operator.spacing_xyz
            ),
            content_sha256=digest.hexdigest(),
        )

    def _increment(self, name: str) -> None:
        self._calls[name] += 1

    def _audit_physical_call(self, name: str, operation: Any) -> Any:
        before = self.physical_call_ledger()
        result = operation()
        after = self.physical_call_ledger()
        if _ledger_delta(after, before) != _single_call_ledger(name):
            raise AssertionError(
                f"physical factor call ledger mismatch for {name}"
            )
        self._increment(name)
        return result

    def _require_tensor(self, value: Any, *, name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.device != self.device:
            raise ValueError(f"{name} device must match the factor pipeline")
        if value.dtype != self.dtype:
            raise ValueError(f"{name} dtype must match the factor pipeline")
        if torch.any(~torch.isfinite(value)):
            raise ValueError(f"{name} must contain only finite values")
        return value

    def _active(self, active: Any) -> torch.Tensor:
        values = self._require_tensor(active, name="active")
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != self.n_active:
            raise ValueError("active must have non-empty shape [batch,n_active]")
        return values

    def _full(self, full: Any) -> torch.Tensor:
        values = self._require_tensor(full, name="full")
        if values.ndim != 5 or tuple(values.shape[1:]) != (1, *self.grid_shape):
            raise ValueError("full must have shape [batch,1,z,y,x]")
        if values.shape[0] < 1:
            raise ValueError("full batch must be non-empty")
        return values

    def _detector(self, detector: Any) -> torch.Tensor:
        values = self._require_tensor(detector, name="detector")
        if values.ndim != 3 or tuple(values.shape[1:]) != (self.ray_count, 2):
            raise ValueError("detector must have shape [batch,ray,2]")
        if values.shape[0] < 1:
            raise ValueError("detector batch must be non-empty")
        return values

    def _tv_dual(self, tv_dual: Any) -> torch.Tensor:
        values = self._require_tensor(tv_dual, name="tv_dual")
        if values.ndim != 5 or tuple(values.shape[1:]) != (3, *self.grid_shape):
            raise ValueError("tv_dual must have shape [batch,3,z,y,x]")
        if values.shape[0] < 1:
            raise ValueError("tv_dual batch must be non-empty")
        return values

    def embed_active(self, active: Any) -> torch.Tensor:
        """Scatter support-active values without applying the dense ``E``."""

        values = self._active(active)
        flat = values.new_zeros((values.shape[0], self.gauge.full_primal_count))
        flat.index_copy_(1, self._active_indices, values)
        return flat.reshape(values.shape[0], 1, *self.grid_shape)

    def restrict_active(self, full: Any) -> torch.Tensor:
        """Gather support-active values without applying the dense ``E.T``."""

        values = self._full(full)
        return values.flatten(1).index_select(1, self._active_indices)

    def _sampled_gradient(self, active: Any, *, absolute: bool) -> torch.Tensor:
        full = self.embed_active(active)[:, 0] * self.voxel_operator.support
        gradient_function = (
            absolute_finite_difference_gradient
            if absolute
            else finite_difference_gradient
        )
        gradient = gradient_function(
            full,
            spacing_xyz=self.voxel_operator.spacing_xyz,
        )
        return self.voxel_operator.trilinear_interpolation(gradient)

    def _data_transpose(self, detector: Any, *, absolute: bool) -> torch.Tensor:
        values = self._detector(detector)
        sampled = (
            self.measurement_factor.absolute_transpose(values)
            if absolute
            else self.measurement_factor.signed_transpose(values)
        )
        gradient = self.voxel_operator.trilinear_interpolation_adjoint(sampled)
        gradient_adjoint = (
            absolute_finite_difference_gradient_adjoint
            if absolute
            else finite_difference_gradient_adjoint
        )
        full = gradient_adjoint(
            gradient,
            spacing_xyz=self.voxel_operator.spacing_xyz,
        )
        full = (full * self.voxel_operator.support)[:, None]
        return self.restrict_active(full)

    def signed_data_forward(self, active: Any) -> torch.Tensor:
        return self._audit_physical_call(
            "signed_data_forward_calls",
            lambda: self.measurement_factor.signed_forward(
                self._sampled_gradient(active, absolute=False)
            ),
        )

    def signed_data_transpose(self, detector: Any) -> torch.Tensor:
        return self._audit_physical_call(
            "signed_data_transpose_calls",
            lambda: self._data_transpose(detector, absolute=False),
        )

    def absolute_data_forward(self, active: Any) -> torch.Tensor:
        return self._audit_physical_call(
            "absolute_data_forward_calls",
            lambda: self.measurement_factor.absolute_forward(
                self._sampled_gradient(active, absolute=True)
            ),
        )

    def absolute_data_transpose(self, detector: Any) -> torch.Tensor:
        return self._audit_physical_call(
            "absolute_data_transpose_calls",
            lambda: self._data_transpose(detector, absolute=True),
        )

    def signed_tv_forward(self, active: Any) -> torch.Tensor:
        return self._audit_physical_call(
            "signed_tv_forward_calls",
            lambda: self.regularization_operator(
                self.embed_active(active)[:, 0]
            ),
        )

    def signed_tv_transpose(self, tv_dual: Any) -> torch.Tensor:
        full = self._audit_physical_call(
            "signed_tv_transpose_calls",
            lambda: self.regularization_operator.adjoint(
                self._tv_dual(tv_dual)
            ),
        )
        return self.restrict_active(full[:, None])

    def absolute_tv_forward(self, active: Any) -> torch.Tensor:
        return self._audit_physical_call(
            "absolute_tv_forward_calls",
            lambda: self.regularization_operator.absolute_forward(
                self.embed_active(active)[:, 0]
            ),
        )

    def absolute_tv_transpose(self, tv_dual: Any) -> torch.Tensor:
        full = self._audit_physical_call(
            "absolute_tv_transpose_calls",
            lambda: self.regularization_operator.absolute_adjoint(
                self._tv_dual(tv_dual)
            ),
        )
        return self.restrict_active(full[:, None])


@dataclass(frozen=True)
class PSUB0FactorMajorizerSetup:
    """One-pass absolute sums, exact activity masks, and shared step sizes."""

    pipeline: PSUB0FactorPipeline
    eta: float
    batch_index: int
    data_row_sums: torch.Tensor
    data_column_sums: torch.Tensor
    tv_row_sums: torch.Tensor
    tv_column_sums: torch.Tensor
    total_column_sums: torch.Tensor
    data_row_mask: torch.Tensor
    tv_row_mask: torch.Tensor
    tv_site_mask: torch.Tensor
    active_primal_mask: torch.Tensor
    active_primal_indices: torch.Tensor
    rho_data_by_view: torch.Tensor
    sigma_data_by_view: torch.Tensor
    rho_tv_by_site: torch.Tensor
    sigma_tv_by_site: torch.Tensor
    tau: torch.Tensor
    setup_call_ledger: FactorPipelineCallLedger
    setup_physical_call_ledger: FactorPipelineCallLedger
    factor_freeze_token: FactorPipelineFreezeToken
    setup_freeze_token: FactorMajorizerSetupFreezeToken

    @property
    def active_primal_count(self) -> int:
        return int(self.active_primal_indices.numel())

    @property
    def data_sigma_rows(self) -> torch.Tensor:
        shared = self.sigma_data_by_view[:, None, None].expand(
            self.pipeline.view_count,
            self.pipeline.rays_per_view,
            2,
        )
        return torch.where(self.data_row_mask, shared, torch.zeros_like(shared))

    @property
    def tv_sigma_rows(self) -> torch.Tensor:
        shared = self.sigma_tv_by_site.unsqueeze(0).expand(
            3,
            *self.pipeline.grid_shape,
        )
        active_sites = self.tv_site_mask.unsqueeze(0).expand_as(shared)
        return torch.where(active_sites, shared, torch.zeros_like(shared))


@dataclass(frozen=True)
class PSUB0FactorPDHGState:
    """One single-instance state on the exact-zero-reduced primal space."""

    x: torch.Tensor
    x_bar: torch.Tensor
    data_dual: torch.Tensor
    tv_dual: torch.Tensor


@dataclass(frozen=True)
class PSUB0DeletedDataLedger:
    """Runtime record for exact-zero data rows removed from the recurrence."""

    target_shape: tuple[int, int, int]
    active_flat_indices: torch.Tensor
    deleted_flat_indices: torch.Tensor
    deleted_target_values: torch.Tensor
    objective_constant: torch.Tensor


def _build_setup_freeze_token(
    *,
    factor_freeze_token: FactorPipelineFreezeToken,
    eta: float,
    batch_index: int,
    data_row_sums: torch.Tensor,
    data_column_sums: torch.Tensor,
    tv_row_sums: torch.Tensor,
    tv_column_sums: torch.Tensor,
    total_column_sums: torch.Tensor,
    data_row_mask: torch.Tensor,
    tv_row_mask: torch.Tensor,
    tv_site_mask: torch.Tensor,
    active_primal_mask: torch.Tensor,
    active_primal_indices: torch.Tensor,
    rho_data_by_view: torch.Tensor,
    sigma_data_by_view: torch.Tensor,
    rho_tv_by_site: torch.Tensor,
    sigma_tv_by_site: torch.Tensor,
    tau: torch.Tensor,
) -> FactorMajorizerSetupFreezeToken:
    named_tensors = (
        ("setup.data_row_sums", data_row_sums),
        ("setup.data_column_sums", data_column_sums),
        ("setup.tv_row_sums", tv_row_sums),
        ("setup.tv_column_sums", tv_column_sums),
        ("setup.total_column_sums", total_column_sums),
        ("setup.data_row_mask", data_row_mask),
        ("setup.tv_row_mask", tv_row_mask),
        ("setup.tv_site_mask", tv_site_mask),
        ("setup.active_primal_mask", active_primal_mask),
        ("setup.active_primal_indices", active_primal_indices),
        ("setup.rho_data_by_view", rho_data_by_view),
        ("setup.sigma_data_by_view", sigma_data_by_view),
        ("setup.rho_tv_by_site", rho_tv_by_site),
        ("setup.sigma_tv_by_site", sigma_tv_by_site),
        ("setup.tau", tau),
    )
    records: list[tuple[Any, ...]] = []
    digest = hashlib.sha256()
    digest.update(factor_freeze_token.content_sha256.encode("ascii"))
    digest.update(repr((float(eta), int(batch_index))).encode("ascii"))
    for name, tensor in named_tensors:
        shape = tuple(int(size) for size in tensor.shape)
        dtype = str(tensor.dtype)
        device = str(tensor.device)
        records.append(
            (
                name,
                id(tensor),
                int(tensor.data_ptr()),
                int(tensor._version),
                shape,
                dtype,
                device,
            )
        )
        digest.update(name.encode("utf-8"))
        digest.update(repr((shape, dtype, device)).encode("ascii"))
        contiguous_cpu = tensor.detach().contiguous().cpu()
        digest.update(contiguous_cpu.view(torch.uint8).numpy().tobytes(order="C"))
    return FactorMajorizerSetupFreezeToken(
        factor_freeze_token=factor_freeze_token,
        tensor_records=tuple(records),
        eta=float(eta),
        batch_index=int(batch_index),
        content_sha256=digest.hexdigest(),
    )


def _current_setup_freeze_token(
    setup: PSUB0FactorMajorizerSetup,
) -> FactorMajorizerSetupFreezeToken:
    return _build_setup_freeze_token(
        factor_freeze_token=setup.pipeline.factor_freeze_token(),
        eta=setup.eta,
        batch_index=setup.batch_index,
        data_row_sums=setup.data_row_sums,
        data_column_sums=setup.data_column_sums,
        tv_row_sums=setup.tv_row_sums,
        tv_column_sums=setup.tv_column_sums,
        total_column_sums=setup.total_column_sums,
        data_row_mask=setup.data_row_mask,
        tv_row_mask=setup.tv_row_mask,
        tv_site_mask=setup.tv_site_mask,
        active_primal_mask=setup.active_primal_mask,
        active_primal_indices=setup.active_primal_indices,
        rho_data_by_view=setup.rho_data_by_view,
        sigma_data_by_view=setup.sigma_data_by_view,
        rho_tv_by_site=setup.rho_tv_by_site,
        sigma_tv_by_site=setup.sigma_tv_by_site,
        tau=setup.tau,
    )


def _solver_tensor(
    setup: PSUB0FactorMajorizerSetup,
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
) -> torch.Tensor:
    tensor = setup.pipeline._require_tensor(value, name=name)
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {list(shape)}")
    return tensor


def _require_single_instance(setup: PSUB0FactorMajorizerSetup) -> None:
    batch_size = int(setup.pipeline.measurement_factor.scale_by_view.shape[0])
    if batch_size != 1 or setup.batch_index != 0:
        raise ValueError(
            "factor PDHG currently requires one frozen measurement-scale instance"
        )
    if setup.pipeline.factor_freeze_token() != setup.factor_freeze_token:
        raise RuntimeError("factor state changed after majorizer setup")
    if _current_setup_freeze_token(setup) != setup.setup_freeze_token:
        raise RuntimeError("derived majorizer state changed after setup")


def _expand_reduced_primal(
    setup: PSUB0FactorMajorizerSetup,
    reduced: torch.Tensor,
) -> torch.Tensor:
    full = reduced.new_zeros(setup.pipeline.n_active)
    full.index_copy_(0, setup.active_primal_indices, reduced)
    return full


def _restrict_reduced_primal(
    setup: PSUB0FactorMajorizerSetup,
    active: torch.Tensor,
) -> torch.Tensor:
    return active.index_select(0, setup.active_primal_indices)


def _validate_state(
    setup: PSUB0FactorMajorizerSetup,
    state: PSUB0FactorPDHGState,
) -> PSUB0FactorPDHGState:
    if not isinstance(state, PSUB0FactorPDHGState):
        raise TypeError("state must be a PSUB0FactorPDHGState")
    reduced_shape = (setup.active_primal_count,)
    data_shape = (
        setup.pipeline.view_count,
        setup.pipeline.rays_per_view,
        2,
    )
    tv_shape = (3, *setup.pipeline.grid_shape)
    return PSUB0FactorPDHGState(
        x=_solver_tensor(setup, state.x, name="state.x", shape=reduced_shape),
        x_bar=_solver_tensor(
            setup,
            state.x_bar,
            name="state.x_bar",
            shape=reduced_shape,
        ),
        data_dual=_solver_tensor(
            setup,
            state.data_dual,
            name="state.data_dual",
            shape=data_shape,
        ),
        tv_dual=_solver_tensor(
            setup,
            state.tv_dual,
            name="state.tv_dual",
            shape=tv_shape,
        ),
    )


def initial_factor_pdhg_state(
    setup: PSUB0FactorMajorizerSetup,
) -> PSUB0FactorPDHGState:
    """Return the declared zero state without touching any factor map."""

    _require_single_instance(setup)
    template = setup.tau
    return PSUB0FactorPDHGState(
        x=torch.zeros_like(template),
        x_bar=torch.zeros_like(template),
        data_dual=template.new_zeros(
            (
                setup.pipeline.view_count,
                setup.pipeline.rays_per_view,
                2,
            )
        ),
        tv_dual=template.new_zeros((3, *setup.pipeline.grid_shape)),
    )


def factor_pdhg_step(
    setup: PSUB0FactorMajorizerSetup,
    state: PSUB0FactorPDHGState,
    target: Any,
    *,
    regularization_weight: float = 0.0,
    penalty: str = "tv",
    huber_delta: float = 0.4,
    theta: float = 1.0,
) -> PSUB0FactorPDHGState:
    """Perform one camera-block/site-shared production-factor recurrence."""

    _require_single_instance(setup)
    current = _validate_state(setup, state)
    target_values = _solver_tensor(
        setup,
        target,
        name="target",
        shape=(
            setup.pipeline.view_count,
            setup.pipeline.rays_per_view,
            2,
        ),
    )
    weight = float(regularization_weight)
    delta = float(huber_delta)
    relaxation = float(theta)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("regularization_weight must be finite and nonnegative")
    if penalty not in {"tv", "huber"}:
        raise ValueError("penalty must be 'tv' or 'huber'")
    if penalty == "huber" and (not math.isfinite(delta) or delta <= 0.0):
        raise ValueError("huber_delta must be finite and positive")
    if not math.isfinite(relaxation) or not 0.0 <= relaxation <= 1.0:
        raise ValueError("theta must lie in [0,1]")

    active_x_bar = _expand_reduced_primal(setup, current.x_bar)
    data_values = setup.pipeline.signed_data_forward(active_x_bar[None])[0].reshape(
        setup.pipeline.view_count,
        setup.pipeline.rays_per_view,
        2,
    )
    sigma_data = setup.data_sigma_rows
    data_candidate = (
        current.data_dual + sigma_data * (data_values - target_values)
    ) / (1.0 + sigma_data)
    data_dual = torch.where(
        setup.data_row_mask,
        data_candidate,
        torch.zeros_like(data_candidate),
    )

    if weight == 0.0 or not bool(torch.any(setup.tv_site_mask)):
        tv_dual = torch.zeros_like(current.tv_dual)
    else:
        tv_values = setup.pipeline.signed_tv_forward(active_x_bar[None])[0]
        sigma_tv = setup.tv_sigma_rows
        candidate = current.tv_dual + sigma_tv * tv_values
        if penalty == "huber":
            candidate = candidate / (1.0 + sigma_tv * delta / weight)
        norms = torch.linalg.vector_norm(candidate, dim=0)
        factors = torch.clamp(norms / weight, min=1.0)
        projected = candidate / factors[None]
        tv_dual = torch.where(
            setup.tv_site_mask[None],
            projected,
            torch.zeros_like(projected),
        )

    gradient_active = setup.pipeline.signed_data_transpose(
        data_dual.reshape(1, setup.pipeline.ray_count, 2)
    )[0]
    if weight != 0.0 and bool(torch.any(setup.tv_site_mask)):
        gradient_active = gradient_active + setup.pipeline.signed_tv_transpose(
            tv_dual[None]
        )[0]
    gradient = _restrict_reduced_primal(setup, gradient_active)
    next_x = current.x - setup.tau * gradient
    next_x_bar = next_x + relaxation * (next_x - current.x)
    return PSUB0FactorPDHGState(next_x, next_x_bar, data_dual, tv_dual)


def run_factor_pdhg(
    setup: PSUB0FactorMajorizerSetup,
    target: Any,
    *,
    iterations: int = 1,
    regularization_weight: float = 0.0,
    penalty: str = "tv",
    huber_delta: float = 0.4,
    theta: float = 1.0,
    initial: PSUB0FactorPDHGState | None = None,
) -> tuple[PSUB0FactorPDHGState, ...]:
    """Run a fixed short recurrence for Gate-A dense-oracle comparison."""

    steps = int(iterations)
    if steps < 1:
        raise ValueError("iterations must be positive")
    current = initial_factor_pdhg_state(setup) if initial is None else initial
    states: list[PSUB0FactorPDHGState] = []
    for _ in range(steps):
        current = factor_pdhg_step(
            setup,
            current,
            target,
            regularization_weight=regularization_weight,
            penalty=penalty,
            huber_delta=huber_delta,
            theta=theta,
        )
        states.append(current)
    return tuple(states)


def build_deleted_data_ledger(
    setup: PSUB0FactorMajorizerSetup,
    target: Any,
) -> PSUB0DeletedDataLedger:
    """Materialize the exact constant contributed by deleted zero rows."""

    _require_single_instance(setup)
    target_values = _solver_tensor(
        setup,
        target,
        name="target",
        shape=(
            setup.pipeline.view_count,
            setup.pipeline.rays_per_view,
            2,
        ),
    )
    flat_mask = setup.data_row_mask.reshape(-1)
    active_indices = torch.nonzero(flat_mask, as_tuple=False).flatten()
    deleted_indices = torch.nonzero(~flat_mask, as_tuple=False).flatten()
    deleted_targets = target_values.reshape(-1).index_select(
        0,
        deleted_indices,
    )
    return PSUB0DeletedDataLedger(
        target_shape=tuple(int(size) for size in target_values.shape),
        active_flat_indices=active_indices.detach().clone(),
        deleted_flat_indices=deleted_indices.detach().clone(),
        deleted_target_values=deleted_targets.detach().clone(),
        objective_constant=0.5 * torch.sum(deleted_targets.square()),
    )


def factor_pdhg_objective(
    setup: PSUB0FactorMajorizerSetup,
    state: PSUB0FactorPDHGState,
    target: Any,
    *,
    regularization_weight: float = 0.0,
    penalty: str = "tv",
    huber_delta: float = 0.4,
) -> torch.Tensor:
    """Evaluate the full target objective, including deleted zero data rows."""

    _require_single_instance(setup)
    current = _validate_state(setup, state)
    target_values = _solver_tensor(
        setup,
        target,
        name="target",
        shape=(
            setup.pipeline.view_count,
            setup.pipeline.rays_per_view,
            2,
        ),
    )
    weight = float(regularization_weight)
    delta = float(huber_delta)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("regularization_weight must be finite and nonnegative")
    if penalty not in {"tv", "huber"}:
        raise ValueError("penalty must be 'tv' or 'huber'")
    if penalty == "huber" and (not math.isfinite(delta) or delta <= 0.0):
        raise ValueError("huber_delta must be finite and positive")

    deleted_ledger = build_deleted_data_ledger(setup, target_values)
    active = _expand_reduced_primal(setup, current.x)
    prediction = setup.pipeline.signed_data_forward(active[None])[0].reshape_as(
        target_values
    )
    flat_residual = prediction.reshape(-1) - target_values.reshape(-1)
    active_residual = flat_residual.index_select(
        0,
        deleted_ledger.active_flat_indices,
    )
    value = (
        0.5 * torch.sum(active_residual.square())
        + deleted_ledger.objective_constant
    )
    if weight:
        gradient = setup.pipeline.signed_tv_forward(active[None])[0]
        magnitudes = torch.linalg.vector_norm(gradient, dim=0)
        if penalty == "huber":
            edge = torch.where(
                magnitudes <= delta,
                0.5 * magnitudes.square() / delta,
                magnitudes - 0.5 * delta,
            )
        else:
            edge = magnitudes
        value = value + weight * torch.sum(edge)
    return value


def _validate_factor_contract(
    *,
    gauge: CoordinateSupportGauge,
    voxel_operator: PSUB0VoxelGradientOperator,
    measurement_factor: ExactAbsoluteMeasurementFactor,
    regularization_operator: ForwardNeumannRegularizationOperator,
) -> None:
    if not isinstance(gauge, CoordinateSupportGauge):
        raise TypeError("gauge must be a CoordinateSupportGauge")
    if not isinstance(voxel_operator, PSUB0VoxelGradientOperator):
        raise TypeError("voxel_operator must be a PSUB0VoxelGradientOperator")
    if type(measurement_factor) is not ExactAbsoluteMeasurementFactor:
        raise TypeError(
            "measurement_factor must use the sealed "
            "ExactAbsoluteMeasurementFactor implementation"
        )
    if type(regularization_operator) is not ForwardNeumannRegularizationOperator:
        raise TypeError(
            "regularization_operator must use the sealed "
            "ForwardNeumannRegularizationOperator implementation"
        )
    sealed_methods = (
        (
            measurement_factor,
            ExactAbsoluteMeasurementFactor,
            (
                "signed_forward",
                "signed_transpose",
                "absolute_forward",
                "absolute_transpose",
            ),
        ),
        (
            regularization_operator,
            ForwardNeumannRegularizationOperator,
            ("__call__", "adjoint", "absolute_forward", "absolute_adjoint"),
        ),
    )
    for instance, owner, names in sealed_methods:
        for name in names:
            bound = getattr(instance, name)
            if (
                getattr(bound, "__self__", None) is not instance
                or getattr(bound, "__func__", None) is not getattr(owner, name)
            ):
                raise TypeError(f"{name} must use the sealed implementation")

    shape = tuple(int(size) for size in voxel_operator.grid_shape)
    if gauge.grid_shape != shape:
        raise ValueError("gauge and voxel_operator grid shapes must match")
    dtype = voxel_operator.sample_weights.dtype
    device = voxel_operator.sample_weights.device
    floating_buffers = (
        voxel_operator.sample_weights,
        voxel_operator.projection_u,
        voxel_operator.projection_v,
        voxel_operator.ray_scale,
        voxel_operator.support,
        measurement_factor.signed_kernel,
        measurement_factor.absolute_kernel,
        measurement_factor.scale_by_view,
    )
    if any(tensor.dtype != dtype for tensor in floating_buffers):
        raise ValueError("all factor floating tensors must have one dtype")
    if any(tensor.device != device for tensor in floating_buffers):
        raise ValueError("all factor tensors must share one device")
    if (
        voxel_operator.sample_indices.device != device
        or voxel_operator.sample_valid.device != device
    ):
        raise ValueError("trilinear index and validity tensors must share the device")
    if voxel_operator.sample_indices.dtype != torch.int64:
        raise ValueError("trilinear sample indices must use int64")
    if voxel_operator.sample_valid.dtype != torch.bool:
        raise ValueError("trilinear sample validity must use bool")
    if any(torch.any(~torch.isfinite(tensor)) for tensor in floating_buffers):
        raise ValueError("all factor tensors must be finite")
    if not dtype.is_floating_point:
        raise ValueError("factor pipeline dtype must be floating point")

    declared_support = gauge.support.to(device=device, dtype=dtype)
    if not torch.equal(voxel_operator.support, declared_support):
        raise ValueError("voxel_operator support must exactly match the gauge")
    expected_indices = (voxel_operator.ray_count, voxel_operator.sample_count, 8)
    if tuple(voxel_operator.sample_indices.shape) != expected_indices:
        raise ValueError("trilinear sample index shape is inconsistent")
    if tuple(voxel_operator.sample_weights.shape) != expected_indices:
        raise ValueError("trilinear sample weight shape is inconsistent")
    if tuple(voxel_operator.sample_valid.shape) != expected_indices[:2]:
        raise ValueError("trilinear sample validity shape is inconsistent")
    if measurement_factor.sample_count != voxel_operator.sample_count:
        raise ValueError("measurement and trilinear sample counts must match")
    if measurement_factor.ray_count != voxel_operator.ray_count:
        raise ValueError("measurement and trilinear ray counts must match")
    if (
        measurement_factor.view_count * measurement_factor.rays_per_view
        != voxel_operator.ray_count
    ):
        raise ValueError("measurement view/ray declaration is inconsistent")
    if tuple(measurement_factor.signed_kernel.shape) != (
        measurement_factor.view_count,
        2 * measurement_factor.rays_per_view,
        measurement_factor.rays_per_view,
        3,
    ):
        raise ValueError("measurement factor kernel shape is inconsistent")
    if tuple(float(value) for value in regularization_operator.spacing_xyz) != tuple(
        float(value) for value in voxel_operator.spacing_xyz
    ):
        raise ValueError("data and TV spacing_xyz must match")


def _nonnegative_finite(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if torch.any(~torch.isfinite(value)):
        raise ValueError(f"{name} must be finite")
    if torch.any(value < 0.0):
        raise ValueError(f"{name} must be elementwise nonnegative")
    return value


def build_psu_b0_factor_majorizer_pipeline(
    *,
    gauge: CoordinateSupportGauge,
    voxel_operator: PSUB0VoxelGradientOperator,
    measurement_factor: ExactAbsoluteMeasurementFactor,
    regularization_operator: ForwardNeumannRegularizationOperator,
    eta: float = 0.8,
    batch_index: int = 0,
) -> PSUB0FactorMajorizerSetup:
    """Build one sample's exact-zero factor metric without targets or truth."""

    eta = float(eta)
    if not math.isfinite(eta) or not 0.0 < eta < 1.0:
        raise ValueError("eta must lie strictly in (0,1)")
    pipeline = PSUB0FactorPipeline(
        gauge=gauge,
        voxel_operator=voxel_operator,
        measurement_factor=measurement_factor,
        regularization_operator=regularization_operator,
    )
    batch_size = int(measurement_factor.scale_by_view.shape[0])
    batch_index = int(batch_index)
    if batch_size != 1 or batch_index != 0:
        raise ValueError(
            "factor majorizer setup requires exactly one frozen "
            "measurement-scale instance and batch_index=0"
        )
    physical_before = pipeline.physical_call_ledger()

    data_active_ones = torch.ones(
        (batch_size, pipeline.n_active),
        dtype=pipeline.dtype,
        device=pipeline.device,
    )
    data_dual_ones = torch.ones(
        (batch_size, pipeline.ray_count, 2),
        dtype=pipeline.dtype,
        device=pipeline.device,
    )
    tv_active_ones = torch.ones(
        (1, pipeline.n_active),
        dtype=pipeline.dtype,
        device=pipeline.device,
    )
    tv_dual_ones = torch.ones(
        (1, 3, *pipeline.grid_shape),
        dtype=pipeline.dtype,
        device=pipeline.device,
    )

    data_row_sums = _nonnegative_finite(
        pipeline.absolute_data_forward(data_active_ones)[batch_index],
        name="data row sums",
    )
    data_column_sums = _nonnegative_finite(
        pipeline.absolute_data_transpose(data_dual_ones)[batch_index],
        name="data column sums",
    )
    tv_row_sums = _nonnegative_finite(
        pipeline.absolute_tv_forward(tv_active_ones)[0],
        name="TV row sums",
    )
    tv_column_sums = _nonnegative_finite(
        pipeline.absolute_tv_transpose(tv_dual_ones)[0],
        name="TV column sums",
    )

    data_rows_by_view = data_row_sums.reshape(
        pipeline.view_count,
        pipeline.rays_per_view,
        2,
    )
    data_row_mask = data_rows_by_view > 0.0
    tv_row_mask = tv_row_sums > 0.0
    tv_rows_by_site = torch.movedim(tv_row_sums, 0, -1)
    tv_site_mask = torch.any(tv_rows_by_site > 0.0, dim=-1)
    total_column_sums = data_column_sums + tv_column_sums
    active_primal_mask = total_column_sums > 0.0
    active_primal_indices = torch.nonzero(
        active_primal_mask,
        as_tuple=False,
    ).flatten()
    if active_primal_indices.numel() == 0:
        raise ValueError("no active primal columns remain after exact-zero masking")

    rho_data_by_view = data_rows_by_view.amax(dim=(1, 2))
    sigma_data_by_view = torch.zeros_like(rho_data_by_view)
    active_views = rho_data_by_view > 0.0
    sigma_data_by_view[active_views] = eta / rho_data_by_view[active_views]

    rho_tv_by_site = tv_rows_by_site.amax(dim=-1)
    sigma_tv_by_site = torch.zeros_like(rho_tv_by_site)
    sigma_tv_by_site[tv_site_mask] = eta / rho_tv_by_site[tv_site_mask]
    tau = eta / total_column_sums.index_select(0, active_primal_indices)

    metric_values = (rho_data_by_view, sigma_data_by_view, rho_tv_by_site, sigma_tv_by_site, tau)
    if any(torch.any(~torch.isfinite(value)) for value in metric_values):
        raise ValueError("factor metric must not contain inf or NaN")
    if torch.any(tau <= 0.0):
        raise ValueError("tau must be strictly positive on active primal columns")

    ledger = pipeline.call_ledger()
    physical_ledger = _ledger_delta(
        pipeline.physical_call_ledger(),
        physical_before,
    )
    expected_ledger = FactorPipelineCallLedger(
        absolute_data_forward_calls=1,
        absolute_data_transpose_calls=1,
        absolute_tv_forward_calls=1,
        absolute_tv_transpose_calls=1,
    )
    if ledger != expected_ledger:
        raise AssertionError("one-pass factor setup call ledger is inconsistent")
    if physical_ledger != expected_ledger:
        raise AssertionError(
            "one-pass physical factor setup call ledger is inconsistent"
        )

    factor_freeze_token = pipeline.factor_freeze_token()
    setup_freeze_token = _build_setup_freeze_token(
        factor_freeze_token=factor_freeze_token,
        eta=eta,
        batch_index=batch_index,
        data_row_sums=data_row_sums,
        data_column_sums=data_column_sums,
        tv_row_sums=tv_row_sums,
        tv_column_sums=tv_column_sums,
        total_column_sums=total_column_sums,
        data_row_mask=data_row_mask,
        tv_row_mask=tv_row_mask,
        tv_site_mask=tv_site_mask,
        active_primal_mask=active_primal_mask,
        active_primal_indices=active_primal_indices,
        rho_data_by_view=rho_data_by_view,
        sigma_data_by_view=sigma_data_by_view,
        rho_tv_by_site=rho_tv_by_site,
        sigma_tv_by_site=sigma_tv_by_site,
        tau=tau,
    )
    return PSUB0FactorMajorizerSetup(
        pipeline=pipeline,
        eta=eta,
        batch_index=batch_index,
        data_row_sums=data_row_sums,
        data_column_sums=data_column_sums,
        tv_row_sums=tv_row_sums,
        tv_column_sums=tv_column_sums,
        total_column_sums=total_column_sums,
        data_row_mask=data_row_mask,
        tv_row_mask=tv_row_mask,
        tv_site_mask=tv_site_mask,
        active_primal_mask=active_primal_mask,
        active_primal_indices=active_primal_indices,
        rho_data_by_view=rho_data_by_view,
        sigma_data_by_view=sigma_data_by_view,
        rho_tv_by_site=rho_tv_by_site,
        sigma_tv_by_site=sigma_tv_by_site,
        tau=tau,
        setup_call_ledger=ledger,
        setup_physical_call_ledger=physical_ledger,
        factor_freeze_token=factor_freeze_token,
        setup_freeze_token=setup_freeze_token,
    )


__all__ = [
    "FACTOR_MAJORIZER_PIPELINE_SCHEMA",
    "FactorPipelineCallLedger",
    "FactorPipelineFreezeToken",
    "FactorMajorizerSetupFreezeToken",
    "PSUB0DeletedDataLedger",
    "PSUB0FactorMajorizerSetup",
    "PSUB0FactorPDHGState",
    "PSUB0FactorPipeline",
    "build_psu_b0_factor_majorizer_pipeline",
    "build_deleted_data_ledger",
    "factor_pdhg_objective",
    "factor_pdhg_step",
    "initial_factor_pdhg_state",
    "run_factor_pdhg",
]
