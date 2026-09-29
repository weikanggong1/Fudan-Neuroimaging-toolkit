"""Continuous Watson stick/tortuosity/isotropic NODDI fit in PyTorch."""

from __future__ import annotations

import numpy as np
import torch


def _even_legendre(x, *, derivative=False):
    polynomials = [torch.ones_like(x), x]
    gradients = [torch.zeros_like(x), torch.ones_like(x)]
    for order in range(1, 12):
        polynomials.append(
            ((2 * order + 1) * x * polynomials[-1] - order * polynomials[-2])
            / (order + 1)
        )
        gradients.append(
            ((2 * order + 1) * (polynomials[-2] + x * gradients[-1])
             - order * gradients[-2]) / (order + 1)
        )
    selected = gradients if derivative else polynomials
    return torch.stack(selected[::2], dim=-1)


class ClassicNODDIModel:
    """The continuous version of AMICO's Watson stick/tortuosity signal."""

    def __init__(self, bvals, bvecs, *, d_par, d_iso, device):
        dtype = torch.float64
        self.b = torch.as_tensor(bvals, device=device, dtype=dtype)
        self.g = torch.as_tensor(bvecs, device=device, dtype=dtype)
        self.d_par = float(d_par)
        self.iso = torch.exp(-self.b * float(d_iso))
        nodes, weights = np.polynomial.legendre.leggauss(64)
        self.nodes = torch.as_tensor(nodes, device=device, dtype=dtype)
        self.weights = torch.as_tensor(weights, device=device, dtype=dtype)
        self.node2 = self.nodes.square()
        self.legendre_nodes = _even_legendre(self.nodes)
        attenuation = torch.exp(
            -self.b[:, None] * self.d_par * self.node2[None]
        )
        moments = (attenuation * self.weights) @ self.legendre_nodes
        order = torch.arange(0, 13, 2, device=device, dtype=dtype)
        self.stick_coefficients = moments * ((2 * order + 1) / 2)

    def evaluate(self, parameters, directions, *, jacobian=False):
        volume_fraction, dispersion, free_water, baseline = parameters.unbind(dim=1)
        concentration = torch.reciprocal(torch.tan(dispersion * (torch.pi / 2)))
        weights = torch.exp(
            concentration[:, None] * (self.node2[None] - 1)
        ) * self.weights
        weights = weights / weights.sum(dim=1, keepdim=True)
        watson = weights @ self.legendre_nodes
        second = weights @ self.node2
        cosine = directions @ self.g.T
        legendre = _even_legendre(cosine)
        stick = torch.einsum("ml,vl,vml->vm", self.stick_coefficients, watson, legendre)
        radial_micro = self.d_par * (1 - volume_fraction)
        difference = self.d_par * volume_fraction
        axial = radial_micro + difference * second
        radial = radial_micro + difference * (1 - second) / 2
        tissue_extra = torch.exp(
            -self.b[None] * (radial[:, None] + (axial - radial)[:, None] * cosine.square())
        )
        tissue = volume_fraction[:, None] * stick + (1 - volume_fraction[:, None]) * tissue_extra
        normalized = (1 - free_water[:, None]) * tissue + free_water[:, None] * self.iso[None]
        prediction = baseline[:, None] * normalized
        if not jacobian:
            return prediction

        concentration_derivative = -(torch.pi / 2) * (1 + concentration.square())
        watson_derivative = (
            (weights * self.node2[None]) @ self.legendre_nodes
            - second[:, None] * watson
        ) * concentration_derivative[:, None]
        second_derivative = (
            weights @ self.node2.square() - second.square()
        ) * concentration_derivative
        stick_dispersion = torch.einsum(
            "ml,vl,vml->vm", self.stick_coefficients, watson_derivative, legendre
        )
        stick_direction = torch.einsum(
            "ml,vl,vml->vm", self.stick_coefficients, watson,
            _even_legendre(cosine, derivative=True),
        )
        axial_fraction = self.d_par * (second - 1)
        radial_fraction = -self.d_par * (1 + second) / 2
        axial_dispersion = difference * second_derivative
        radial_dispersion = -difference * second_derivative / 2

        def extra_derivative(axial_change, radial_change):
            apparent_change = radial_change[:, None] + (
                axial_change - radial_change
            )[:, None] * cosine.square()
            return -self.b[None] * tissue_extra * apparent_change

        volume_jacobian = (1 - free_water[:, None]) * (
            stick - tissue_extra + (1 - volume_fraction[:, None])
            * extra_derivative(axial_fraction, radial_fraction)
        )
        dispersion_jacobian = (1 - free_water[:, None]) * (
            volume_fraction[:, None] * stick_dispersion
            + (1 - volume_fraction[:, None])
            * extra_derivative(axial_dispersion, radial_dispersion)
        )
        direction_jacobian = (1 - free_water[:, None]) * (
            volume_fraction[:, None] * stick_direction
            - (1 - volume_fraction[:, None]) * tissue_extra * self.b[None]
            * 2 * (axial - radial)[:, None] * cosine
        )
        reference_axis = torch.zeros_like(directions)
        reference_axis[:, 2] = 1
        near_pole = directions[:, 2].abs() > 0.9
        reference_axis[near_pole, 0] = 1
        reference_axis[near_pole, 2] = 0
        tangent_one = torch.nn.functional.normalize(
            torch.linalg.cross(directions, reference_axis), dim=1
        )
        tangent_two = torch.linalg.cross(directions, tangent_one)
        jacobian_columns = torch.stack(
            (baseline[:, None] * volume_jacobian,
             baseline[:, None] * dispersion_jacobian,
             baseline[:, None] * (self.iso[None] - tissue),
             normalized,
             baseline[:, None] * direction_jacobian * (tangent_one @ self.g.T),
             baseline[:, None] * direction_jacobian * (tangent_two @ self.g.T)),
            dim=2,
        )
        return prediction, jacobian_columns, tangent_one, tangent_two


def fit_classic_noddi(
    signal, bvals, bvecs, b0_indices, estimates, directions, *,
    d_par, d_iso, device, batch_size=1024, maximum_iterations=30,
):
    """Refine AMICO initial values with the Toolbox's b0-based noise scale."""
    model = ClassicNODDIModel(bvals, bvecs, d_par=d_par, d_iso=d_iso, device=device)
    fitted = np.empty_like(estimates, dtype=np.float32)
    fitted_directions = np.empty_like(directions, dtype=np.float32)
    fitted_rmse = np.empty(len(signal), dtype=np.float32)
    improved = 0
    noise_scales = np.empty(len(signal), dtype=np.float32)
    for start in range(0, len(signal), batch_size):
        stop = min(start + batch_size, len(signal))
        observed = torch.as_tensor(signal[start:stop], device=device, dtype=torch.float64)
        initial = torch.as_tensor(
            estimates[start:stop], device=device, dtype=torch.float64
        ).clone()
        parameters = torch.cat((initial, torch.ones_like(initial[:, :1])), dim=1)
        parameters[:, 0].clamp_(0.001, 0.999)
        parameters[:, 1].clamp_(0.005, 0.995)
        parameters[:, 2].clamp_(0, 0.999)
        vectors = torch.nn.functional.normalize(
            torch.as_tensor(directions[start:stop], device=device, dtype=torch.float64),
            dim=1,
        )
        b0_signal = observed[:, b0_indices]
        sigma = (
            torch.maximum(
                b0_signal.std(dim=1, unbiased=False),
                0.02 * b0_signal.mean(dim=1),
            ) / 100
        ).clamp_min(1e-8)[:, None]
        damping = torch.full((stop - start,), 0.01, device=device, dtype=torch.float64)
        identity = torch.eye(5, device=device, dtype=torch.float64)

        def objective(prediction):
            argument = observed * prediction / sigma.square()
            log_bessel = torch.log(torch.special.i0e(argument)) + argument
            return (prediction.square() / (2 * sigma.square()) - log_bessel).sum(1)

        current = objective(model.evaluate(parameters, vectors))
        for _ in range(maximum_iterations):
            prediction, jacobian, tangent_one, tangent_two = model.evaluate(
                parameters, vectors, jacobian=True
            )
            jacobian = jacobian[:, :, (0, 1, 2, 4, 5)]
            argument = observed * prediction / sigma.square()
            ratio = torch.special.i1e(argument) / torch.special.i0e(argument)
            residual = observed * ratio - prediction
            gram = jacobian.transpose(1, 2) @ jacobian
            right = jacobian.transpose(1, 2) @ residual.unsqueeze(-1)
            step = torch.linalg.solve(
                gram + damping[:, None, None] * identity, right
            ).squeeze(-1)
            step[:, :3].clamp_(-0.15, 0.15)
            step[:, 3:].clamp_(-0.3, 0.3)
            candidate = parameters.clone()
            candidate[:, :3] += step[:, :3]
            candidate[:, 0].clamp_(0.001, 0.999)
            candidate[:, 1].clamp_(0.005, 0.995)
            candidate[:, 2].clamp_(0, 0.999)
            candidate_vectors = torch.nn.functional.normalize(
                vectors + step[:, 3:4] * tangent_one
                + step[:, 4:5] * tangent_two, dim=1
            )
            candidate_cost = objective(model.evaluate(candidate, candidate_vectors))
            accept = candidate_cost < current
            improved += int(accept.sum().item())
            parameters = torch.where(accept[:, None], candidate, parameters)
            vectors = torch.where(accept[:, None], candidate_vectors, vectors)
            current = torch.where(accept, candidate_cost, current)
            damping = torch.where(accept, damping / 2, damping * 4).clamp(1e-7, 1e6)
        prediction = model.evaluate(parameters, vectors)
        fitted[start:stop] = parameters[:, :3].cpu().numpy().astype(np.float32)
        noise_scales[start:stop] = sigma[:, 0].cpu().numpy().astype(np.float32)
        fitted_directions[start:stop] = vectors.cpu().numpy().astype(np.float32)
        fitted_rmse[start:stop] = torch.sqrt(
            (prediction - observed).square().mean(1)
        ).cpu().numpy().astype(np.float32)
    return fitted, fitted_directions, fitted_rmse, {
        "accepted_updates": improved,
        "rician_sigma_median": float(np.median(noise_scales)),
    }
