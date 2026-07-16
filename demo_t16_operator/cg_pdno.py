"""Covariance- and geometry-conditioned primal descent neural operator.

This is an engineering prototype, not a superiority claim.  Every stage uses
the declared forward/adjoint pair.  A small controller may only change a
bounded normalized step and proximal strength.  The zero-initialized proximal
starts exactly as a deterministic prewhitened projected-gradient solver.
"""

from __future__ import annotations

import math

import torch
from torch import nn

try:
    from .measurement_contract import (
        BOSTBatch,
        DepthSeparableLinearBOST,
        geometry_noise_features,
    )
except ImportError:
    from measurement_contract import BOSTBatch, DepthSeparableLinearBOST, geometry_noise_features


class ZeroInitializedProximal3D(nn.Module):
    def __init__(self, width: int = 12):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv3d(3, int(width), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(int(width), int(width), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(int(width), 1, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class BoundedStepController(nn.Module):
    def __init__(
        self,
        feature_count: int,
        *,
        hidden: int = 16,
        step_min: float = 0.2,
        step_max: float = 1.8,
        initial_step: float = 1.0,
    ):
        super().__init__()
        if not 0 < step_min < initial_step < step_max < 2:
            raise ValueError("step bounds must satisfy 0 < min < initial < max < 2")
        self.step_min = float(step_min)
        self.step_max = float(step_max)
        self.initial_step = float(initial_step)
        self.network = nn.Sequential(
            nn.Linear(int(feature_count), int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), 2),
        )
        final = self.network[-1]
        nn.init.zeros_(final.weight)
        probability = (float(initial_step) - self.step_min) / (self.step_max - self.step_min)
        step_bias = math.log(probability / (1.0 - probability))
        with torch.no_grad():
            final.bias[0] = step_bias
            final.bias[1] = -2.0

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.network(features)
        step = self.step_min + (self.step_max - self.step_min) * torch.sigmoid(raw[:, 0])
        proximal_scale = torch.sigmoid(raw[:, 1])
        return step, proximal_scale


def residual_features(residual: torch.Tensor, batch: BOSTBatch) -> torch.Tensor:
    white = batch.whitened(residual)
    camera_rms = torch.sqrt(torch.mean(white * white, dim=(1, 3)).clamp_min(1e-18))
    active = batch.view_mask > 0.5
    active_count = active.sum(dim=1).clamp_min(1)
    pooled = torch.sqrt(
        torch.sum(torch.where(active, camera_rms * camera_rms, torch.zeros_like(camera_rms)), dim=1)
        / active_count
    )
    camera_max = torch.max(
        torch.where(active, camera_rms, torch.full_like(camera_rms, -torch.inf)), dim=1
    ).values
    camera_cv = torch.sqrt(
        torch.sum(
            torch.where(active, (camera_rms - pooled[:, None]) ** 2, torch.zeros_like(camera_rms)),
            dim=1,
        )
        / active_count
    ) / pooled.clamp_min(1e-12)
    return torch.stack(
        [torch.log1p(pooled), torch.log1p(camera_max), torch.log1p(camera_cv)], dim=1
    )


class CGPDNO(nn.Module):
    """Fixed-depth unrolled inverse operator with a conservative fallback."""

    def __init__(
        self,
        *,
        stages: int = 6,
        proximal_width: int = 12,
        controller_hidden: int = 16,
        step_min: float = 0.2,
        step_max: float = 1.8,
        initial_step: float = 1.0,
    ):
        super().__init__()
        if stages < 1:
            raise ValueError("stages must be positive")
        self.stages = int(stages)
        self.proximal = ZeroInitializedProximal3D(proximal_width)
        self.controller = BoundedStepController(
            8 + 3 + 1,
            hidden=controller_hidden,
            step_min=step_min,
            step_max=step_max,
            initial_step=initial_step,
        )

    def forward(
        self,
        batch: BOSTBatch,
        operator: DepthSeparableLinearBOST,
        warm_start: torch.Tensor,
        lipschitz: torch.Tensor,
        trust_budget: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch.validate()
        support = batch.expanded_support().to(warm_start)
        if warm_start.shape != support.shape:
            raise ValueError("warm_start and support shapes disagree")
        if lipschitz.shape != (len(batch.geometry_ids),) or torch.any(lipschitz <= 0):
            raise ValueError("lipschitz must contain one positive value per sample")
        if trust_budget is None:
            trust = torch.ones(
                len(batch.geometry_ids), dtype=warm_start.dtype, device=warm_start.device
            )
        else:
            trust = trust_budget.to(device=warm_start.device, dtype=warm_start.dtype)
            if trust.shape != (len(batch.geometry_ids),):
                raise ValueError("trust_budget must contain one value per sample")
            if torch.any((trust < 0.0) | (trust > 1.0)):
                raise ValueError("trust_budget must lie in [0, 1]")
        geometry = geometry_noise_features(batch).to(warm_start)
        current = torch.clamp(warm_start, min=0.0) * support
        previous_update = torch.zeros_like(current)
        iterates = [current]
        steps = []
        proximal_scales = []
        discrepancies = []
        for stage in range(self.stages):
            gradient, residual = operator.weighted_gradient(current, batch)
            normalized_gradient = gradient / lipschitz[:, None, None, None, None]
            diagnostics = residual_features(residual, batch)
            stage_fraction = torch.full(
                (len(batch.geometry_ids), 1),
                (stage + 1) / self.stages,
                dtype=current.dtype,
                device=current.device,
            )
            controller_input = torch.cat([geometry, diagnostics, stage_fraction], dim=1)
            learned_step, learned_proximal_scale = self.controller(controller_input)
            fallback_step = torch.full_like(learned_step, self.controller.initial_step)
            step = fallback_step + trust * (learned_step - fallback_step)
            proximal_scale = trust * learned_proximal_scale
            proposal = current + step[:, None, None, None, None] * normalized_gradient
            correction = self.proximal(
                torch.cat([proposal, normalized_gradient, previous_update], dim=1)
            )
            candidate = torch.clamp(
                proposal + proximal_scale[:, None, None, None, None] * correction,
                min=0.0,
            ) * support
            previous_update = candidate - current
            current = candidate
            iterates.append(current)
            steps.append(step)
            proximal_scales.append(proximal_scale)
            discrepancies.append(diagnostics[:, 0])
        return {
            "prediction": current,
            "iterates": torch.stack(iterates),
            "normalized_step": torch.stack(steps),
            "proximal_scale": torch.stack(proximal_scales),
            "log_discrepancy": torch.stack(discrepancies),
            "trust_budget": trust,
            "forward_calls": torch.as_tensor(self.stages, device=current.device),
            "adjoint_calls": torch.as_tensor(self.stages, device=current.device),
        }

    def deterministic_fallback(
        self,
        batch: BOSTBatch,
        operator: DepthSeparableLinearBOST,
        warm_start: torch.Tensor,
        lipschitz: torch.Tensor,
    ) -> torch.Tensor:
        """Run the declared initial step without either learned component."""

        support = batch.expanded_support().to(warm_start)
        current = torch.clamp(warm_start, min=0.0) * support
        with torch.no_grad():
            step = torch.full(
                (len(batch.geometry_ids),),
                self.controller.initial_step,
                dtype=current.dtype,
                device=current.device,
            )
            for _ in range(self.stages):
                gradient, _ = operator.weighted_gradient(current, batch)
                current = torch.clamp(
                    current + step[:, None, None, None, None]
                    * gradient
                    / lipschitz[:, None, None, None, None],
                    min=0.0,
                ) * support
        return current


class GeometryConditionedCorrection3D(nn.Module):
    """One bounded field-space correction conditioned on the acquisition."""

    def __init__(self, feature_count: int, width: int = 12):
        super().__init__()
        width = int(width)
        self.input = nn.Conv3d(4, width, kernel_size=3, padding=1)
        self.middle = nn.Conv3d(width, width, kernel_size=3, padding=1)
        self.output = nn.Conv3d(width, 1, kernel_size=3, padding=1)
        self.conditioner = nn.Sequential(
            nn.Linear(int(feature_count), width),
            nn.GELU(),
            nn.Linear(width, 2 * width),
        )
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.zeros_(self.conditioner[-1].bias)

    def forward(self, values: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.gelu(self.input(values))
        scale, shift = self.conditioner(features).chunk(2, dim=1)
        hidden = hidden * (1.0 + 0.25 * torch.tanh(scale)[:, :, None, None, None])
        hidden = hidden + 0.25 * shift[:, :, None, None, None]
        hidden = torch.nn.functional.gelu(self.middle(hidden))
        return self.output(hidden)


def _bounded_correction(
    correction: torch.Tensor,
    reference: torch.Tensor,
    maximum_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    correction_norm = torch.linalg.vector_norm(correction.flatten(1), dim=1)
    reference_norm = torch.linalg.vector_norm(reference.flatten(1), dim=1).clamp_min(1e-12)
    ratio = correction_norm / reference_norm
    scale = torch.clamp(float(maximum_ratio) / ratio.clamp_min(1e-12), max=1.0)
    return correction * scale[:, None, None, None, None], ratio


def descent_certificate(
    gradient_toward_data: torch.Tensor,
    delta: torch.Tensor,
    lipschitz_upper: torch.Tensor,
    *,
    safety: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a no-extra-forward-call safe scale and quadratic upper bound.

    ``gradient_toward_data`` is ``A^T C^-1 (y-Ax)``. For the weighted
    least-squares data term, the change along ``alpha * delta`` is bounded by
    ``-alpha <g,delta> + alpha^2 L ||delta||^2 / 2`` when ``L`` is a true
    upper bound on the normal-operator norm.
    """

    if not 0.0 < safety < 1.0:
        raise ValueError("safety must lie in (0, 1)")
    if gradient_toward_data.shape != delta.shape:
        raise ValueError("gradient and delta shapes disagree")
    if lipschitz_upper.shape != (len(delta),) or torch.any(lipschitz_upper <= 0):
        raise ValueError("lipschitz_upper must contain one positive value per sample")
    descent = torch.sum(gradient_toward_data * delta, dim=(1, 2, 3, 4))
    squared_norm = torch.sum(delta * delta, dim=(1, 2, 3, 4))
    curvature_upper = lipschitz_upper * squared_norm
    maximum = 2.0 * float(safety) * descent / curvature_upper.clamp_min(1e-18)
    alpha = torch.clamp(maximum, min=0.0, max=1.0)
    upper_change = -alpha * descent + 0.5 * alpha.square() * curvature_upper
    return alpha, upper_change, descent


class BaseCorrectionCGPDNO(nn.Module):
    """Shared deterministic base followed by one certified learned direction.

    With a zero correction this class is exactly a ``stages``-step fixed
    prewhitened projected-gradient solver. The candidate and fallback both use
    ``stages`` forward and ``stages`` adjoint calls: the first ``stages - 1``
    calls build the shared base and the final call supplies the gradient used
    by either the deterministic or learned final direction.
    """

    def __init__(
        self,
        *,
        stages: int = 4,
        correction_width: int = 12,
        maximum_correction_ratio: float = 1.0,
        certificate_safety: float = 0.9,
        certificate_temperature: float = 8.0,
        minimum_certificate_improvement: float = 0.0,
    ):
        super().__init__()
        if stages < 2:
            raise ValueError("BaseCorrectionCGPDNO requires at least two stages")
        if maximum_correction_ratio <= 0:
            raise ValueError("maximum_correction_ratio must be positive")
        if certificate_temperature <= 0:
            raise ValueError("certificate_temperature must be positive")
        if minimum_certificate_improvement < 0:
            raise ValueError("minimum_certificate_improvement cannot be negative")
        self.stages = int(stages)
        self.maximum_correction_ratio = float(maximum_correction_ratio)
        self.certificate_safety = float(certificate_safety)
        self.certificate_temperature = float(certificate_temperature)
        self.minimum_certificate_improvement = float(minimum_certificate_improvement)
        self.correction = GeometryConditionedCorrection3D(8 + 3, correction_width)

    @staticmethod
    def _project(values: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        return torch.clamp(values, min=0.0) * support

    def _shared_base(
        self,
        batch: BOSTBatch,
        operator: object,
        warm_start: torch.Tensor,
        lipschitz_upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        support = batch.expanded_support().to(warm_start)
        current = self._project(warm_start, support)
        previous_update = torch.zeros_like(current)
        for _ in range(self.stages - 1):
            gradient, _ = operator.weighted_gradient(current, batch)
            proposal = current + gradient / lipschitz_upper[:, None, None, None, None]
            candidate = self._project(proposal, support)
            previous_update = candidate - current
            current = candidate
        return current, previous_update

    def forward(
        self,
        batch: BOSTBatch,
        operator: object,
        warm_start: torch.Tensor,
        lipschitz_upper: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch.validate()
        support = batch.expanded_support().to(warm_start)
        if warm_start.shape != support.shape:
            raise ValueError("warm_start and support shapes disagree")
        if lipschitz_upper.shape != (len(batch.geometry_ids),):
            raise ValueError("lipschitz_upper must contain one value per sample")
        base, previous_update = self._shared_base(
            batch, operator, warm_start, lipschitz_upper
        )
        gradient, residual = operator.weighted_gradient(base, batch)
        normalized_gradient = gradient / lipschitz_upper[:, None, None, None, None]
        features = torch.cat(
            [geometry_noise_features(batch).to(base), residual_features(residual, batch)],
            dim=1,
        )
        raw_correction = self.correction(
            torch.cat([base, normalized_gradient, previous_update, support], dim=1),
            features,
        )
        bounded_correction, raw_ratio = _bounded_correction(
            raw_correction, normalized_gradient, self.maximum_correction_ratio
        )

        fallback = self._project(base + normalized_gradient, support)
        fallback_delta = fallback - base
        fallback_alpha, fallback_bound, _ = descent_certificate(
            gradient,
            fallback_delta,
            lipschitz_upper,
            safety=self.certificate_safety,
        )
        fallback = base + fallback_alpha[:, None, None, None, None] * fallback_delta

        proposed = self._project(base + normalized_gradient + bounded_correction, support)
        proposed_delta = proposed - base
        candidate_alpha, candidate_bound, candidate_descent = descent_certificate(
            gradient,
            proposed_delta,
            lipschitz_upper,
            safety=self.certificate_safety,
        )
        candidate = base + candidate_alpha[:, None, None, None, None] * proposed_delta

        required = self.minimum_certificate_improvement * fallback_bound.abs()
        score_scale = fallback_bound.abs().clamp_min(1e-12)
        score = (fallback_bound - candidate_bound - required) / score_scale
        soft_gate = torch.sigmoid(self.certificate_temperature * score)
        hard_gate = (candidate_bound + required <= fallback_bound).to(base.dtype)
        gate = hard_gate + soft_gate - soft_gate.detach()
        prediction = fallback + gate[:, None, None, None, None] * (candidate - fallback)
        prediction = self._project(prediction, support)
        return {
            "prediction": prediction,
            "shared_base": base,
            "deterministic_fallback": fallback,
            "candidate": candidate,
            "acceptance_gate": hard_gate,
            "acceptance_soft": soft_gate,
            "candidate_alpha": candidate_alpha,
            "fallback_alpha": fallback_alpha,
            "candidate_bound": candidate_bound,
            "fallback_bound": fallback_bound,
            "candidate_descent": candidate_descent,
            "raw_correction_ratio": raw_ratio,
            "forward_calls": torch.as_tensor(self.stages, device=base.device),
            "adjoint_calls": torch.as_tensor(self.stages, device=base.device),
        }

    @torch.no_grad()
    def deterministic_fallback(
        self,
        batch: BOSTBatch,
        operator: object,
        warm_start: torch.Tensor,
        lipschitz_upper: torch.Tensor,
    ) -> torch.Tensor:
        base, _ = self._shared_base(batch, operator, warm_start, lipschitz_upper)
        support = batch.expanded_support().to(warm_start)
        gradient, _ = operator.weighted_gradient(base, batch)
        proposal = self._project(
            base + gradient / lipschitz_upper[:, None, None, None, None], support
        )
        delta = proposal - base
        alpha, _, _ = descent_certificate(
            gradient, delta, lipschitz_upper, safety=self.certificate_safety
        )
        return base + alpha[:, None, None, None, None] * delta


def _projected_bb_step(
    current: torch.Tensor,
    gradient: torch.Tensor,
    previous_field: torch.Tensor | None,
    previous_gradient: torch.Tensor | None,
    lipschitz_upper: torch.Tensor,
    *,
    normalized_min: float,
    normalized_max: float,
) -> torch.Tensor:
    fallback = 1.0 / lipschitz_upper
    if previous_field is None or previous_gradient is None:
        return fallback
    displacement = current - previous_field
    gradient_change = previous_gradient - gradient
    numerator = torch.sum(displacement * displacement, dim=(1, 2, 3, 4))
    denominator = torch.sum(displacement * gradient_change, dim=(1, 2, 3, 4))
    bb = numerator / denominator.clamp_min(1e-18)
    step = torch.where(denominator > 1e-12, bb, fallback)
    normalized = torch.clamp(
        step * lipschitz_upper,
        min=float(normalized_min),
        max=float(normalized_max),
    )
    return normalized / lipschitz_upper


class PBBBaseCorrectionCGPDNO(nn.Module):
    """Projected-BB base plus one bounded data-descent correction.

    This second candidate exists because the first independent audit showed
    that a four-call projected-BB baseline dominated the fixed-step base. The
    first ``stages - 1`` PBB iterates are shared. The final PBB gradient and
    step cost the last ``F/F^T`` call; the network may alter that field-space
    direction, but the Lipschitz certificate scales it back to a descent step
    relative to the shared base. This is not a guarantee that the learned
    candidate outperforms the deterministic final PBB fallback.
    """

    def __init__(
        self,
        *,
        stages: int = 4,
        correction_width: int = 12,
        maximum_correction_ratio: float = 0.8,
        certificate_safety: float = 0.95,
        certificate_temperature: float = 8.0,
        bb_normalized_step_min: float = 0.2,
        bb_normalized_step_max: float = 1.8,
        use_saturation_gate: bool = False,
    ):
        super().__init__()
        if stages < 2:
            raise ValueError("PBBBaseCorrectionCGPDNO requires at least two stages")
        if not 0 < bb_normalized_step_min <= bb_normalized_step_max < 2:
            raise ValueError("BB normalized step bounds must lie in (0, 2)")
        if bb_normalized_step_max >= 2.0 * certificate_safety:
            raise ValueError("BB maximum must be below twice the certificate safety")
        self.stages = int(stages)
        self.maximum_correction_ratio = float(maximum_correction_ratio)
        self.certificate_safety = float(certificate_safety)
        self.certificate_temperature = float(certificate_temperature)
        self.bb_normalized_step_min = float(bb_normalized_step_min)
        self.bb_normalized_step_max = float(bb_normalized_step_max)
        self.use_saturation_gate = bool(use_saturation_gate)
        self.correction = GeometryConditionedCorrection3D(8 + 3, correction_width)

    @staticmethod
    def _project(values: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        return torch.clamp(values, min=0.0) * support

    def _shared_base(
        self,
        batch: BOSTBatch,
        operator: object,
        warm_start: torch.Tensor,
        lipschitz_upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        support = batch.expanded_support().to(warm_start)
        current = self._project(warm_start, support)
        previous_field = None
        previous_gradient = None
        previous_update = torch.zeros_like(current)
        for _ in range(self.stages - 1):
            gradient, _ = operator.weighted_gradient(current, batch)
            step = _projected_bb_step(
                current,
                gradient,
                previous_field,
                previous_gradient,
                lipschitz_upper,
                normalized_min=self.bb_normalized_step_min,
                normalized_max=self.bb_normalized_step_max,
            )
            candidate = self._project(
                current + step[:, None, None, None, None] * gradient, support
            )
            previous_field, previous_gradient = current, gradient
            previous_update = candidate - current
            current = candidate
        if previous_field is None or previous_gradient is None:
            raise RuntimeError("PBB shared base did not initialize its history")
        return current, previous_field, previous_gradient

    def prepare_shared_state(
        self,
        batch: BOSTBatch,
        operator: object,
        warm_start: torch.Tensor,
        lipschitz_upper: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch.validate()
        support = batch.expanded_support().to(warm_start)
        if warm_start.shape != support.shape:
            raise ValueError("warm_start and support shapes disagree")
        base, previous_field, previous_gradient = self._shared_base(
            batch, operator, warm_start, lipschitz_upper
        )
        gradient, residual = operator.weighted_gradient(base, batch)
        step = _projected_bb_step(
            base,
            gradient,
            previous_field,
            previous_gradient,
            lipschitz_upper,
            normalized_min=self.bb_normalized_step_min,
            normalized_max=self.bb_normalized_step_max,
        )
        pbb_direction = step[:, None, None, None, None] * gradient
        fallback = self._project(base + pbb_direction, support)
        fallback_delta = fallback - base
        _, fallback_bound, _ = descent_certificate(
            gradient,
            fallback_delta,
            lipschitz_upper,
            safety=self.certificate_safety,
        )

        features = torch.cat(
            [geometry_noise_features(batch).to(base), residual_features(residual, batch)],
            dim=1,
        )
        return {
            "base": base,
            "support": support,
            "gradient": gradient,
            "pbb_direction": pbb_direction,
            "previous_update": base - previous_field,
            "features": features,
            "fallback": fallback,
            "fallback_bound": fallback_bound,
        }

    def correct_from_shared_state(
        self,
        state: dict[str, torch.Tensor],
        lipschitz_upper: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        base = state["base"]
        support = state["support"]
        gradient = state["gradient"]
        pbb_direction = state["pbb_direction"]
        previous_update = state["previous_update"]
        features = state["features"]
        fallback = state["fallback"]
        fallback_bound = state["fallback_bound"]
        raw_correction = self.correction(
            torch.cat([base, pbb_direction, previous_update, support], dim=1),
            features,
        )
        bounded_correction, raw_ratio = _bounded_correction(
            raw_correction, pbb_direction, self.maximum_correction_ratio
        )
        proposed = self._project(base + pbb_direction + bounded_correction, support)
        proposed_delta = proposed - base
        alpha, candidate_bound, candidate_descent = descent_certificate(
            gradient,
            proposed_delta,
            lipschitz_upper,
            safety=self.certificate_safety,
        )
        candidate = base + alpha[:, None, None, None, None] * proposed_delta
        scale = fallback_bound.abs().clamp_min(1e-12)
        soft_gate = torch.sigmoid(
            self.certificate_temperature * (-candidate_bound / scale)
        )
        descent_gate = (candidate_descent > 0) & (candidate_bound <= 1e-10)
        saturation_gate = raw_ratio <= self.maximum_correction_ratio
        deploy_gate = descent_gate & (
            saturation_gate if self.use_saturation_gate else torch.ones_like(saturation_gate)
        )
        hard_gate = (descent_gate if self.training else deploy_gate).to(base.dtype)
        gate = hard_gate + soft_gate - soft_gate.detach()
        prediction = fallback + gate[:, None, None, None, None] * (candidate - fallback)
        prediction = self._project(prediction, support)
        return {
            "prediction": prediction,
            "shared_base": base,
            "deterministic_fallback": fallback,
            "candidate": candidate,
            "acceptance_gate": hard_gate,
            "descent_gate": descent_gate.to(base.dtype),
            "saturation_gate": saturation_gate.to(base.dtype),
            "acceptance_soft": soft_gate,
            "candidate_alpha": alpha,
            "fallback_alpha": torch.ones_like(alpha),
            "candidate_bound": candidate_bound,
            "fallback_bound": fallback_bound,
            "candidate_descent": candidate_descent,
            "raw_correction_ratio": raw_ratio,
        }

    def forward(
        self,
        batch: BOSTBatch,
        operator: object,
        warm_start: torch.Tensor,
        lipschitz_upper: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        state = self.prepare_shared_state(batch, operator, warm_start, lipschitz_upper)
        output = self.correct_from_shared_state(state, lipschitz_upper)
        output.update(
            {
                "forward_calls": torch.as_tensor(
                    self.stages, device=state["base"].device
                ),
                "adjoint_calls": torch.as_tensor(
                    self.stages, device=state["base"].device
                ),
            }
        )
        return output

    @torch.no_grad()
    def deterministic_fallback(
        self,
        batch: BOSTBatch,
        operator: object,
        warm_start: torch.Tensor,
        lipschitz_upper: torch.Tensor,
    ) -> torch.Tensor:
        return self.prepare_shared_state(
            batch, operator, warm_start, lipschitz_upper
        )["fallback"]
