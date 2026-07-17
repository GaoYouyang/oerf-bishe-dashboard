"""True data-only scalar, view-block, and voxel-factor PDHG metrics.

Gate A attests the production factor maps.  Gate B must isolate the metric,
so all three candidates below are built from the same absolute data factor
``M >= |A|`` and execute exactly ``1F + 1A^T`` per iteration.  No TV row,
column, norm, spacing, or runtime call participates in setup or recurrence.

The scalar candidate uses one global dual step and one global primal step.
The view-block candidate keeps one dual step per camera view and the global
primal step.  The voxel-factor candidate additionally uses one primal step
per active data-coupled voxel.  This staged ablation prevents a skipped TV
operator from being mistaken for a conditioning improvement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Literal

import torch
from torch import nn

from .psu_b0_absolute_measurement_factor import ExactAbsoluteMeasurementFactor
from .psu_b0_active_coordinates import build_coordinate_support_gauge
from .psu_b0_factor_majorizer_pipeline import (
    FactorPipelineCallLedger,
    FactorPipelineFreezeToken,
    PSUB0FactorPDHGState,
    PSUB0FactorPipeline,
)
from .psu_b0_primal_dual import ForwardNeumannRegularizationOperator
from .psu_b0_reconstruction_interface import PSUB0VoxelGradientOperator


GATE_B_DATA_ONLY_SCHEMA = "psu-b0-gate-b-true-data-only-metric-1.0"
GateBMetricMode = Literal["scalar", "view_block", "voxel_factor"]


class SingleSampleViewLocalWhitening(nn.Module):
    """Freeze one row of an existing independent per-view whitener."""

    def __init__(self, source: nn.Module, sample_index: int) -> None:
        super().__init__()
        matrix = getattr(source, "matrix", None)
        scales = getattr(source, "scale_by_view", None)
        means = getattr(source, "calibration_mean", None)
        view_count = int(getattr(source, "view_count", -1))
        rays_per_view = int(getattr(source, "rays_per_view", -1))
        if not isinstance(matrix, torch.Tensor):
            raise TypeError("source whitening must expose tensor matrix")
        if not isinstance(scales, torch.Tensor):
            raise TypeError("source whitening must expose tensor scale_by_view")
        if not isinstance(means, torch.Tensor):
            raise TypeError("source whitening must expose tensor calibration_mean")
        index = int(sample_index)
        if view_count < 1 or rays_per_view < 1:
            raise ValueError("source whitening dimensions must be positive")
        detector_dimension = 2 * rays_per_view
        if matrix.shape != (view_count, detector_dimension, detector_dimension):
            raise ValueError("source whitening matrix shape is inconsistent")
        if scales.ndim != 2 or scales.shape[1] != view_count:
            raise ValueError("source scale_by_view shape is inconsistent")
        if means.shape == (view_count, rays_per_view, 2):
            means = means.reshape(view_count, detector_dimension)
        elif means.shape != (view_count, detector_dimension):
            raise ValueError("source calibration mean shape is inconsistent")
        if index < 0 or index >= scales.shape[0]:
            raise IndexError("sample_index is outside source scale_by_view")

        self.view_count = view_count
        self.rays_per_view = rays_per_view
        self.measurement_dimension_per_view = detector_dimension
        self.whitening_block_scope = "view_local"
        self.independent_whitening_blocks = True
        self.covariance_block_ids = tuple(range(view_count))
        self.register_buffer("matrix", matrix.detach().clone())
        self.register_buffer(
            "scale_by_view",
            scales[index : index + 1].detach().clone(),
        )
        self.register_buffer("calibration_mean", means.detach().clone())

    def _canonical(self, values: torch.Tensor) -> torch.Tensor:
        expected_rays = self.view_count * self.rays_per_view
        if values.ndim != 3 or values.shape[1:] != (expected_rays, 2):
            raise ValueError("detector values must have shape [1,view*rays,2]")
        if len(values) != 1:
            raise ValueError("single-sample whitening requires batch size one")
        if values.device != self.matrix.device or values.dtype != self.matrix.dtype:
            raise ValueError("detector values must match whitening dtype and device")
        if torch.any(~torch.isfinite(values)):
            raise ValueError("detector values must be finite")
        return values.reshape(1, self.view_count, self.measurement_dimension_per_view)

    def center_observation(self, values: torch.Tensor) -> torch.Tensor:
        canonical = self._canonical(values)
        centered = canonical - self.scale_by_view[:, :, None] * self.calibration_mean[None]
        return centered.reshape_as(values)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        canonical = self._canonical(values)
        whitened = torch.einsum("vij,bvj->bvi", self.matrix, canonical)
        whitened = whitened / self.scale_by_view[:, :, None]
        return whitened.reshape_as(values)

    def transpose(self, values: torch.Tensor) -> torch.Tensor:
        canonical = self._canonical(values)
        scaled = canonical / self.scale_by_view[:, :, None]
        output = torch.einsum("vji,bvj->bvi", self.matrix, scaled)
        return output.reshape_as(values)

    def prepare_observation(self, values: torch.Tensor) -> torch.Tensor:
        return self(self.center_observation(values))


def _tensor_digest(digest: Any, name: str, tensor: torch.Tensor) -> None:
    values = tensor.detach().contiguous().cpu()
    digest.update(name.encode("utf-8"))
    digest.update(repr((tuple(values.shape), str(values.dtype))).encode("ascii"))
    digest.update(values.view(torch.uint8).numpy().tobytes(order="C"))


@dataclass(frozen=True)
class GateBDataOnlySetup:
    """One-pass A-only majorizer with an immutable content fingerprint."""

    pipeline: PSUB0FactorPipeline
    eta: float
    data_row_sums: torch.Tensor
    data_column_sums: torch.Tensor
    data_row_mask: torch.Tensor
    active_primal_indices: torch.Tensor
    rho_data_by_view: torch.Tensor
    sigma_data_by_view: torch.Tensor
    tau_voxel_factor: torch.Tensor
    setup_call_ledger: FactorPipelineCallLedger
    setup_physical_call_ledger: FactorPipelineCallLedger
    factor_freeze_token: FactorPipelineFreezeToken
    content_sha256: str

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


@dataclass(frozen=True)
class GateBMetricDescription:
    """Frozen A-only step tensors for one ablation level."""

    mode: GateBMetricMode
    tau: torch.Tensor
    sigma_data_by_view: torch.Tensor
    voxel_tau_minimum: float
    voxel_tau_maximum: float

    def validate(self, setup: GateBDataOnlySetup) -> None:
        if self.mode not in {"scalar", "view_block", "voxel_factor"}:
            raise ValueError("unsupported Gate-B metric mode")
        if self.tau.shape != setup.tau_voxel_factor.shape:
            raise ValueError("metric tau shape does not match A-only setup")
        if self.sigma_data_by_view.shape != setup.sigma_data_by_view.shape:
            raise ValueError("metric sigma shape does not match A-only setup")
        if self.tau.device != setup.pipeline.device or self.tau.dtype != setup.pipeline.dtype:
            raise ValueError("metric tau must match setup dtype and device")
        if self.sigma_data_by_view.device != setup.pipeline.device:
            raise ValueError("metric sigma must match setup device")
        if torch.any(~torch.isfinite(self.tau)) or torch.any(self.tau <= 0.0):
            raise ValueError("metric tau must be finite and positive")
        if torch.any(~torch.isfinite(self.sigma_data_by_view)):
            raise ValueError("metric sigma must be finite")
        if torch.any(self.sigma_data_by_view < 0.0):
            raise ValueError("metric sigma must be nonnegative")
        if torch.any(self.tau > setup.tau_voxel_factor):
            raise ValueError("ablation tau must not exceed voxel-factor tau")
        if torch.any(self.sigma_data_by_view > setup.sigma_data_by_view):
            raise ValueError("ablation sigma must not exceed view-factor sigma")


def _setup_content_sha256(setup: GateBDataOnlySetup) -> str:
    digest = hashlib.sha256()
    digest.update(setup.factor_freeze_token.content_sha256.encode("ascii"))
    digest.update(repr(float(setup.eta)).encode("ascii"))
    for name in (
        "data_row_sums",
        "data_column_sums",
        "data_row_mask",
        "active_primal_indices",
        "rho_data_by_view",
        "sigma_data_by_view",
        "tau_voxel_factor",
    ):
        _tensor_digest(digest, name, getattr(setup, name))
    return digest.hexdigest()


def _validate_setup(setup: GateBDataOnlySetup) -> None:
    if not isinstance(setup, GateBDataOnlySetup):
        raise TypeError("setup must be a GateBDataOnlySetup")
    if setup.pipeline.factor_freeze_token() != setup.factor_freeze_token:
        raise RuntimeError("factor state changed after A-only setup")
    if _setup_content_sha256(setup) != setup.content_sha256:
        raise RuntimeError("derived A-only metric changed after setup")


def build_single_sample_factor_setup(
    *,
    voxel_operator: PSUB0VoxelGradientOperator,
    source_whitening: nn.Module,
    sample_index: int,
    measurement_scale: float,
    eta: float,
) -> GateBDataOnlySetup:
    """Build one target-free A-only setup using exactly two absolute calls."""

    if not isinstance(voxel_operator, PSUB0VoxelGradientOperator):
        raise TypeError("voxel_operator must be a PSUB0VoxelGradientOperator")
    eta_value = float(eta)
    if not math.isfinite(eta_value) or not 0.0 < eta_value < 1.0:
        raise ValueError("eta must lie strictly in (0,1)")
    whitening = SingleSampleViewLocalWhitening(source_whitening, sample_index)
    measurement = ExactAbsoluteMeasurementFactor(
        whitening,
        projection_u_xyz=voxel_operator.projection_u,
        projection_v_xyz=voxel_operator.projection_v,
        ray_scale=voxel_operator.ray_scale,
        sample_count=voxel_operator.sample_count,
        measurement_scale=float(measurement_scale),
    )
    gauge = build_coordinate_support_gauge(voxel_operator.support.detach().cpu())
    regularization = ForwardNeumannRegularizationOperator(
        tuple(float(value) for value in voxel_operator.spacing_xyz)
    )
    pipeline = PSUB0FactorPipeline(
        gauge=gauge,
        voxel_operator=voxel_operator,
        measurement_factor=measurement,
        regularization_operator=regularization,
    )
    physical_before = pipeline.physical_call_ledger()
    active_ones = torch.ones(
        (1, pipeline.n_active),
        dtype=pipeline.dtype,
        device=pipeline.device,
    )
    dual_ones = torch.ones(
        (1, pipeline.ray_count, 2),
        dtype=pipeline.dtype,
        device=pipeline.device,
    )
    data_row_sums = pipeline.absolute_data_forward(active_ones)[0]
    data_column_sums = pipeline.absolute_data_transpose(dual_ones)[0]
    if torch.any(~torch.isfinite(data_row_sums)) or torch.any(data_row_sums < 0.0):
        raise ValueError("A-only row sums must be finite and nonnegative")
    if torch.any(~torch.isfinite(data_column_sums)) or torch.any(data_column_sums < 0.0):
        raise ValueError("A-only column sums must be finite and nonnegative")
    data_row_mask = data_row_sums.reshape(
        pipeline.view_count,
        pipeline.rays_per_view,
        2,
    ) > 0.0
    active_primal_indices = torch.nonzero(
        data_column_sums > 0.0,
        as_tuple=False,
    ).flatten()
    if active_primal_indices.numel() == 0:
        raise ValueError("A-only setup has no data-coupled primal columns")
    rho_data_by_view = data_row_sums.reshape(
        pipeline.view_count,
        pipeline.rays_per_view,
        2,
    ).amax(dim=(1, 2))
    sigma_data_by_view = torch.zeros_like(rho_data_by_view)
    active_views = rho_data_by_view > 0.0
    sigma_data_by_view[active_views] = eta_value / rho_data_by_view[active_views]
    active_columns = data_column_sums.index_select(0, active_primal_indices)
    tau_voxel_factor = eta_value / active_columns

    expected = FactorPipelineCallLedger(
        absolute_data_forward_calls=1,
        absolute_data_transpose_calls=1,
    )
    logical = pipeline.call_ledger()
    physical_after = pipeline.physical_call_ledger()
    physical = FactorPipelineCallLedger(
        **{
            name: int(getattr(physical_after, name) - getattr(physical_before, name))
            for name in FactorPipelineCallLedger.__dataclass_fields__
        }
    )
    if logical != expected or physical != expected:
        raise AssertionError("A-only setup must use exactly |A| and |A|^T once")
    factor_token = pipeline.factor_freeze_token()
    provisional = GateBDataOnlySetup(
        pipeline=pipeline,
        eta=eta_value,
        data_row_sums=data_row_sums,
        data_column_sums=data_column_sums,
        data_row_mask=data_row_mask,
        active_primal_indices=active_primal_indices,
        rho_data_by_view=rho_data_by_view,
        sigma_data_by_view=sigma_data_by_view,
        tau_voxel_factor=tau_voxel_factor,
        setup_call_ledger=logical,
        setup_physical_call_ledger=physical,
        factor_freeze_token=factor_token,
        content_sha256="",
    )
    return GateBDataOnlySetup(
        **{
            **provisional.__dict__,
            "content_sha256": _setup_content_sha256(provisional),
        }
    )


def describe_gate_b_metric(
    setup: GateBDataOnlySetup,
    mode: GateBMetricMode,
) -> GateBMetricDescription:
    """Return one level of the A-only scalar-to-factor ablation."""

    _validate_setup(setup)
    tau_minimum = float(setup.tau_voxel_factor.detach().amin().cpu())
    tau_maximum = float(setup.tau_voxel_factor.detach().amax().cpu())
    if mode == "voxel_factor":
        tau = setup.tau_voxel_factor.detach().clone()
        sigma = setup.sigma_data_by_view.detach().clone()
    elif mode == "view_block":
        tau = torch.full_like(setup.tau_voxel_factor, tau_minimum)
        sigma = setup.sigma_data_by_view.detach().clone()
    elif mode == "scalar":
        tau = torch.full_like(setup.tau_voxel_factor, tau_minimum)
        positive = setup.sigma_data_by_view[setup.sigma_data_by_view > 0.0]
        if positive.numel() == 0:
            raise ValueError("A-only scalar metric has no active dual row")
        scalar_sigma = torch.min(positive)
        sigma = torch.where(
            setup.sigma_data_by_view > 0.0,
            scalar_sigma,
            torch.zeros_like(setup.sigma_data_by_view),
        )
    else:
        raise ValueError("mode must be scalar, view_block, or voxel_factor")
    description = GateBMetricDescription(
        mode=mode,
        tau=tau,
        sigma_data_by_view=sigma,
        voxel_tau_minimum=tau_minimum,
        voxel_tau_maximum=tau_maximum,
    )
    description.validate(setup)
    return description


def _validated_target(setup: GateBDataOnlySetup, target: Any) -> torch.Tensor:
    if not isinstance(target, torch.Tensor):
        raise TypeError("target must be a torch.Tensor")
    expected = (setup.pipeline.view_count, setup.pipeline.rays_per_view, 2)
    if target.shape != expected:
        raise ValueError(f"target must have shape {expected}")
    if target.device != setup.pipeline.device or target.dtype != setup.pipeline.dtype:
        raise ValueError("target must match setup dtype and device")
    if torch.any(~torch.isfinite(target)):
        raise ValueError("target must be finite")
    return target


def initial_gate_b_state(setup: GateBDataOnlySetup) -> PSUB0FactorPDHGState:
    _validate_setup(setup)
    return _initial_gate_b_state_unchecked(setup)


def _initial_gate_b_state_unchecked(
    setup: GateBDataOnlySetup,
) -> PSUB0FactorPDHGState:
    template = setup.tau_voxel_factor
    return PSUB0FactorPDHGState(
        x=torch.zeros_like(template),
        x_bar=torch.zeros_like(template),
        data_dual=template.new_zeros(
            (setup.pipeline.view_count, setup.pipeline.rays_per_view, 2)
        ),
        tv_dual=template.new_zeros((3, *setup.pipeline.grid_shape)),
    )


def _expanded_active(setup: GateBDataOnlySetup, reduced: torch.Tensor) -> torch.Tensor:
    active = reduced.new_zeros(setup.pipeline.n_active)
    active.index_copy_(0, setup.active_primal_indices, reduced)
    return active


def factor_state_to_volume(
    setup: GateBDataOnlySetup,
    state: PSUB0FactorPDHGState,
) -> torch.Tensor:
    """Decode one reduced A-only state to ``[1,1,Z,Y,X]``."""

    _validate_setup(setup)
    if not isinstance(state, PSUB0FactorPDHGState):
        raise TypeError("state must be a PSUB0FactorPDHGState")
    if state.x.shape != (setup.active_primal_count,):
        raise ValueError("state primal shape does not match A-only setup")
    return setup.pipeline.embed_active(_expanded_active(setup, state.x)[None])


def _data_only_step(
    setup: GateBDataOnlySetup,
    state: PSUB0FactorPDHGState,
    target: torch.Tensor,
    metric: GateBMetricDescription,
    *,
    theta: float,
) -> PSUB0FactorPDHGState:
    relaxation = float(theta)
    if not math.isfinite(relaxation) or not 0.0 <= relaxation <= 1.0:
        raise ValueError("theta must lie in [0,1]")
    expected_primal = (setup.active_primal_count,)
    expected_data = (setup.pipeline.view_count, setup.pipeline.rays_per_view, 2)
    if state.x.shape != expected_primal or state.x_bar.shape != expected_primal:
        raise ValueError("state primal shape does not match setup")
    if state.data_dual.shape != expected_data:
        raise ValueError("state data dual shape does not match setup")

    active_x_bar = _expanded_active(setup, state.x_bar)
    data_values = setup.pipeline.signed_data_forward(active_x_bar[None])[0].reshape(
        expected_data
    )
    sigma_rows = metric.sigma_data_by_view[:, None, None].expand(expected_data)
    sigma_rows = torch.where(
        setup.data_row_mask,
        sigma_rows,
        torch.zeros_like(sigma_rows),
    )
    candidate = (
        state.data_dual + sigma_rows * (data_values - target)
    ) / (1.0 + sigma_rows)
    data_dual = torch.where(
        setup.data_row_mask,
        candidate,
        torch.zeros_like(candidate),
    )
    gradient_active = setup.pipeline.signed_data_transpose(
        data_dual.reshape(1, setup.pipeline.ray_count, 2)
    )[0]
    gradient = gradient_active.index_select(0, setup.active_primal_indices)
    next_x = state.x - metric.tau * gradient
    next_x_bar = next_x + relaxation * (next_x - state.x)
    return PSUB0FactorPDHGState(
        x=next_x,
        x_bar=next_x_bar,
        data_dual=data_dual,
        tv_dual=torch.zeros_like(state.tv_dual),
    )


def run_gate_b_data_only_trajectory(
    setup: GateBDataOnlySetup,
    target: Any,
    *,
    checkpoints: Sequence[int],
    mode: GateBMetricMode,
    theta: float = 1.0,
) -> Mapping[int, PSUB0FactorPDHGState]:
    """Run one maximum-depth A-only trajectory and return checkpoints."""

    target_values = _validated_target(setup, target)
    ordered = tuple(sorted({int(value) for value in checkpoints}))
    if not ordered or ordered[0] < 1:
        raise ValueError("checkpoints must contain positive iterations")
    metric = describe_gate_b_metric(setup, mode)
    current = _initial_gate_b_state_unchecked(setup)
    states: list[PSUB0FactorPDHGState] = []
    for _ in range(ordered[-1]):
        current = _data_only_step(
            setup,
            current,
            target_values,
            metric,
            theta=float(theta),
        )
        states.append(current)
    return {iteration: states[iteration - 1] for iteration in ordered}


__all__ = [
    "GATE_B_DATA_ONLY_SCHEMA",
    "GateBDataOnlySetup",
    "GateBMetricDescription",
    "GateBMetricMode",
    "SingleSampleViewLocalWhitening",
    "build_single_sample_factor_setup",
    "describe_gate_b_metric",
    "factor_state_to_volume",
    "initial_gate_b_state",
    "run_gate_b_data_only_trajectory",
]
