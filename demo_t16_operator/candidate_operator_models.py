"""Small, falsifiable operator-learning prototypes for BOST inverse problems.

These modules are research scaffolds, not claimed novel methods.  They expose
the physical interfaces that a future full implementation must preserve:
camera/ray geometry, measurement covariance, explicit forward/adjoint calls,
data-proximal null-space variation, and causal low-rank 4D decoding.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _batched_operator(operator: torch.Tensor, batch: int) -> torch.Tensor:
    if operator.ndim == 2:
        return operator.unsqueeze(0).expand(batch, -1, -1)
    if operator.ndim != 3 or operator.shape[0] != batch:
        raise ValueError("operator must have shape [ray, voxel] or [batch, ray, voxel]")
    return operator


def apply_linear_operator(field: torch.Tensor, operator: torch.Tensor) -> torch.Tensor:
    """Apply a dense toy forward operator to `[batch, voxel]` fields."""

    if field.ndim != 2:
        raise ValueError("field must have shape [batch, voxel]")
    matrix = _batched_operator(operator, field.shape[0])
    return torch.einsum("bmv,bv->bm", matrix, field)


def apply_linear_adjoint(residual: torch.Tensor, operator: torch.Tensor) -> torch.Tensor:
    """Apply the exact transpose of :func:`apply_linear_operator`."""

    if residual.ndim != 2:
        raise ValueError("residual must have shape [batch, ray]")
    matrix = _batched_operator(operator, residual.shape[0])
    return torch.einsum("bmv,bm->bv", matrix, residual)


def _expand_batch_feature(value: torch.Tensor, batch: int, width: int, name: str) -> torch.Tensor:
    if value.ndim == 1 and value.shape[0] == width:
        return value.unsqueeze(0).expand(batch, -1)
    if value.ndim == 2 and value.shape == (batch, width):
        return value
    raise ValueError(f"{name} must have shape [{width}] or [batch, {width}]")


class PermutationInvariantRayEncoder(nn.Module):
    """Encode an unordered active-ray set with masked first and second moments."""

    def __init__(self, geometry_features: int, hidden_features: int = 32, output_features: int = 24):
        super().__init__()
        token_features = int(geometry_features) + 3
        hidden = int(hidden_features)
        output = int(output_features)
        self.token = nn.Sequential(
            nn.Linear(token_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.aggregate = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, output),
        )

    def forward(
        self,
        observation: torch.Tensor,
        residual: torch.Tensor,
        geometry: torch.Tensor,
        active_mask: torch.Tensor,
        local_precision: torch.Tensor,
    ) -> torch.Tensor:
        if geometry.ndim != 3 or geometry.shape[:2] != observation.shape:
            raise ValueError("geometry must have shape [batch, ray, feature]")
        if active_mask.shape != observation.shape or local_precision.shape != observation.shape:
            raise ValueError("mask and local_precision must match observation")
        count = active_mask.sum(dim=1, keepdim=True)
        if torch.any(count < 1):
            raise ValueError("each sample needs at least one active ray")
        token_input = torch.cat(
            [
                geometry,
                observation.unsqueeze(-1),
                residual.unsqueeze(-1),
                torch.log(local_precision.clamp_min(1e-12)).unsqueeze(-1),
            ],
            dim=-1,
        )
        encoded = self.token(token_input)
        weight = active_mask.unsqueeze(-1)
        mean = (weight * encoded).sum(dim=1) / count
        variance = (weight * (encoded - mean[:, None]) ** 2).sum(dim=1) / count
        return self.aggregate(torch.cat([mean, torch.sqrt(variance.clamp_min(1e-12))], dim=-1))


class CovarianceGeometryUnrolledOperator(nn.Module):
    """Data-consistent unrolling with geometry and covariance conditioning.

    The final learned correction layer is initialized to zero.  Before
    training, this model is therefore exactly a projected, prewhitened gradient
    iteration.  That deterministic control must be measured before attributing
    any gain to the neural block.
    """

    def __init__(
        self,
        voxel_count: int,
        geometry_features: int,
        voxel_features: int = 0,
        iterations: int = 6,
        context_features: int = 24,
        hidden_features: int = 48,
        initial_step: float = 0.1,
        learned_step: bool = False,
        correction_scale: float = 0.1,
        nonnegative: bool = True,
    ):
        super().__init__()
        if iterations < 1 or initial_step <= 0:
            raise ValueError("iterations and initial_step must be positive")
        self.voxel_count = int(voxel_count)
        self.voxel_features = int(voxel_features)
        self.iterations = int(iterations)
        self.correction_scale = float(correction_scale)
        self.nonnegative = bool(nonnegative)
        self.ray_encoder = PermutationInvariantRayEncoder(
            geometry_features=geometry_features,
            output_features=context_features,
        )
        self.proximal = nn.Sequential(
            nn.Linear(3 + self.voxel_features + int(context_features), int(hidden_features)),
            nn.GELU(),
            nn.Linear(int(hidden_features), int(hidden_features)),
            nn.GELU(),
            nn.Linear(int(hidden_features), 1),
        )
        nn.init.zeros_(self.proximal[-1].weight)
        nn.init.zeros_(self.proximal[-1].bias)
        raw = math.log(math.expm1(float(initial_step)))
        if learned_step:
            self.raw_step = nn.Parameter(torch.full((self.iterations,), raw))
        else:
            self.register_buffer("raw_step", torch.full((self.iterations,), raw))

    def _weighted_residual(
        self,
        residual: torch.Tensor,
        active_mask: torch.Tensor,
        noise_std: torch.Tensor | None,
        whitener: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, rays = residual.shape
        mask = _expand_batch_feature(active_mask, batch, rays, "active_mask")
        masked = mask * residual
        if whitener is not None and noise_std is not None:
            raise ValueError("pass either noise_std or whitener, not both")
        if whitener is not None:
            if whitener.ndim == 2:
                matrix = whitener.unsqueeze(0).expand(batch, -1, -1)
            elif whitener.ndim == 3 and whitener.shape[0] == batch:
                matrix = whitener
            else:
                raise ValueError("whitener must have shape [ray, ray] or [batch, ray, ray]")
            if matrix.shape[1:] != (rays, rays):
                raise ValueError("whitener ray dimensions do not match observation")
            white = mask * torch.einsum("bij,bj->bi", matrix, masked)
            weighted = mask * torch.einsum("bji,bj->bi", matrix, white)
            precision_matrix = torch.einsum("bji,bjk->bik", matrix, matrix)
            local_precision = mask * torch.diagonal(precision_matrix, dim1=1, dim2=2)
        else:
            if noise_std is None:
                std = torch.ones_like(residual)
            else:
                std = _expand_batch_feature(noise_std, batch, rays, "noise_std")
            local_precision = mask / std.clamp_min(1e-8).square()
            white = mask * residual / std.clamp_min(1e-8)
            weighted = local_precision * residual
        return weighted, white, local_precision.clamp_min(1e-12)

    def forward(
        self,
        observation: torch.Tensor,
        operator: torch.Tensor,
        geometry: torch.Tensor,
        active_mask: torch.Tensor,
        support: torch.Tensor,
        voxel_features: torch.Tensor | None = None,
        correction_budget: torch.Tensor | None = None,
        initial: torch.Tensor | None = None,
        noise_std: torch.Tensor | None = None,
        whitener: torch.Tensor | None = None,
        return_history: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if observation.ndim != 2:
            raise ValueError("observation must have shape [batch, ray]")
        batch, rays = observation.shape
        matrix = _batched_operator(operator, batch)
        if matrix.shape[1] != rays or matrix.shape[2] != self.voxel_count:
            raise ValueError("operator dimensions do not match model")
        if geometry.shape[:2] != observation.shape:
            raise ValueError("geometry ray dimensions do not match observation")
        mask = _expand_batch_feature(active_mask, batch, rays, "active_mask")
        support_batch = _expand_batch_feature(support, batch, self.voxel_count, "support")
        if self.voxel_features:
            if voxel_features is None:
                raise ValueError("voxel_features are required by this model")
            if voxel_features.ndim == 2 and voxel_features.shape == (
                self.voxel_count,
                self.voxel_features,
            ):
                voxel_feature_batch = voxel_features.unsqueeze(0).expand(batch, -1, -1)
            elif voxel_features.ndim == 3 and voxel_features.shape == (
                batch,
                self.voxel_count,
                self.voxel_features,
            ):
                voxel_feature_batch = voxel_features
            else:
                raise ValueError("voxel_features do not match model dimensions")
        else:
            if voxel_features is not None and voxel_features.shape[-1] != 0:
                raise ValueError("model was constructed without voxel features")
            voxel_feature_batch = observation.new_empty((batch, self.voxel_count, 0))
        if correction_budget is None:
            budget = observation.new_ones((batch, 1))
        elif correction_budget.ndim == 1 and correction_budget.shape[0] == batch:
            budget = correction_budget[:, None]
        elif correction_budget.shape == (batch, 1):
            budget = correction_budget
        else:
            raise ValueError("correction_budget must have shape [batch] or [batch, 1]")
        if torch.any((budget < 0) | (budget > 1)):
            raise ValueError("correction_budget must lie in [0, 1]")
        current = (
            torch.zeros((batch, self.voxel_count), device=observation.device, dtype=observation.dtype)
            if initial is None
            else initial
        )
        if current.shape != (batch, self.voxel_count):
            raise ValueError("initial field dimensions do not match model")
        states = [current]
        residual_norms = []
        steps = F.softplus(self.raw_step)
        for iteration in range(self.iterations):
            residual = apply_linear_operator(current, matrix) - observation
            weighted, white, local_precision = self._weighted_residual(
                residual, mask, noise_std, whitener
            )
            gradient = apply_linear_adjoint(weighted, matrix)
            context = self.ray_encoder(
                observation=observation,
                residual=residual,
                geometry=geometry,
                active_mask=mask,
                local_precision=local_precision,
            )
            voxel_context = context[:, None].expand(-1, self.voxel_count, -1)
            features = torch.cat(
                [
                    current.unsqueeze(-1),
                    gradient.unsqueeze(-1),
                    support_batch.unsqueeze(-1),
                    voxel_feature_batch,
                    voxel_context,
                ],
                dim=-1,
            )
            correction = self.proximal(features).squeeze(-1)
            current = current - steps[iteration] * gradient
            current = current + self.correction_scale * budget * support_batch * correction
            current = support_batch * current
            if self.nonnegative:
                current = current.clamp_min(0.0)
            residual_norms.append(torch.sqrt(torch.mean(white.square(), dim=1)))
            states.append(current)
        if not return_history:
            return current
        return current, {
            "states": torch.stack(states, dim=1),
            "whitened_residual_rms": torch.stack(residual_norms, dim=1),
            "steps": steps,
            "forward_calls": torch.tensor(self.iterations),
            "adjoint_calls": torch.tensor(self.iterations),
        }


class DataProximalNullspaceSampler(nn.Module):
    """Generate approximate null-space alternatives around a stable visible field.

    This dense implementation is intended for small mechanism tests.  A full
    BOST implementation must replace the matrix solve with an iterative method
    or an operator-aware low-rank approximation.
    """

    def __init__(
        self,
        voxel_count: int,
        latent_features: int = 8,
        hidden_features: int = 64,
        proposal_scale: float = 0.1,
        damping: float = 1e-6,
    ):
        super().__init__()
        self.voxel_count = int(voxel_count)
        self.latent_features = int(latent_features)
        self.proposal_scale = float(proposal_scale)
        self.damping = float(damping)
        self.proposal = nn.Sequential(
            nn.Linear(self.voxel_count + self.latent_features, int(hidden_features)),
            nn.GELU(),
            nn.Linear(int(hidden_features), self.voxel_count),
        )

    def _row_space_solve(
        self, operator: torch.Tensor, rhs: torch.Tensor
    ) -> torch.Tensor:
        batch, rays, _ = operator.shape
        gram = torch.einsum("bmv,bnv->bmn", operator, operator)
        eye = torch.eye(rays, device=operator.device, dtype=operator.dtype)[None]
        return torch.linalg.solve(gram + self.damping * eye, rhs.unsqueeze(-1)).squeeze(-1)

    def data_project(
        self, field: torch.Tensor, observation: torch.Tensor, operator: torch.Tensor
    ) -> torch.Tensor:
        matrix = _batched_operator(operator, field.shape[0])
        residual = observation - apply_linear_operator(field, matrix)
        dual = self._row_space_solve(matrix, residual)
        return field + apply_linear_adjoint(dual, matrix)

    def null_project(self, proposal: torch.Tensor, operator: torch.Tensor) -> torch.Tensor:
        matrix = _batched_operator(operator, proposal.shape[0])
        measured = apply_linear_operator(proposal, matrix)
        dual = self._row_space_solve(matrix, measured)
        return proposal - apply_linear_adjoint(dual, matrix)

    def forward(
        self,
        visible_field: torch.Tensor,
        observation: torch.Tensor,
        operator: torch.Tensor,
        latent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if latent.ndim != 3 or latent.shape[0] != visible_field.shape[0]:
            raise ValueError("latent must have shape [batch, sample, feature]")
        if latent.shape[2] != self.latent_features:
            raise ValueError("latent feature count does not match model")
        batch, sample_count, _ = latent.shape
        matrix = _batched_operator(operator, batch)
        visible = self.data_project(visible_field, observation, matrix)
        repeated = visible[:, None].expand(-1, sample_count, -1)
        raw = self.proposal(torch.cat([repeated, latent], dim=-1))
        flat_raw = raw.reshape(batch * sample_count, self.voxel_count)
        flat_matrix = matrix[:, None].expand(-1, sample_count, -1, -1).reshape(
            batch * sample_count, matrix.shape[1], matrix.shape[2]
        )
        null = self.null_project(flat_raw, flat_matrix).reshape_as(raw)
        samples = repeated + self.proposal_scale * null
        return {
            "visible": visible,
            "samples": samples,
            "mean": samples.mean(dim=1),
            "std": samples.std(dim=1, unbiased=False),
        }


class AdaptiveRankInnovation4DOperator(nn.Module):
    """Causal low-rank 4D decoder with an explicit sparse innovation branch."""

    def __init__(
        self,
        measurement_features: int,
        voxel_count: int,
        rank: int = 12,
        hidden_features: int = 48,
        coordinate_features: int = 3,
    ):
        super().__init__()
        hidden = int(hidden_features)
        self.voxel_count = int(voxel_count)
        self.rank = int(rank)
        self.measurement_encoder = nn.Sequential(
            nn.Linear(int(measurement_features), hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.temporal = nn.GRU(hidden, hidden, batch_first=True)
        self.coefficients = nn.Linear(hidden, self.rank)
        self.spatial_basis = nn.Sequential(
            nn.Linear(int(coordinate_features), hidden),
            nn.GELU(),
            nn.Linear(hidden, self.rank),
        )
        self.innovation = nn.Linear(hidden, self.voxel_count)
        self.innovation_gate = nn.Linear(2 * hidden, 1)
        nn.init.zeros_(self.innovation.weight)
        nn.init.zeros_(self.innovation.bias)
        nn.init.constant_(self.innovation_gate.bias, -3.0)

    def forward(
        self,
        measurement_sequence: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if measurement_sequence.ndim != 3:
            raise ValueError("measurement_sequence must have shape [batch, time, feature]")
        if coordinates.ndim != 2 or coordinates.shape[0] != self.voxel_count:
            raise ValueError("coordinates must have shape [voxel, coordinate_feature]")
        encoded = self.measurement_encoder(measurement_sequence)
        state, _ = self.temporal(encoded)
        coefficients = self.coefficients(state)
        basis = self.spatial_basis(coordinates)
        low_rank = torch.einsum("btr,vr->btv", coefficients, basis) / math.sqrt(self.rank)
        previous = torch.cat([torch.zeros_like(state[:, :1]), state[:, :-1]], dim=1)
        gate = torch.sigmoid(self.innovation_gate(torch.cat([state, state - previous], dim=-1)))
        innovation = gate * self.innovation(state)
        return {
            "field": low_rank + innovation,
            "low_rank": low_rank,
            "innovation": innovation,
            "innovation_gate": gate,
            "coefficients": coefficients,
        }
