"""Independent GPU Bayesian segmentation with a FreeSurfer GEMS atlas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .._dmri import configure_device
from .atlas import GEMSAtlas
from .deformation import ashburner_prior
from .gaussian import (GaussianParameters, gaussian_log_likelihood,
                       initialise_gaussians, label_posterior, update_gaussians)
from .rasterize import BlockIndex, build_block_index, rasterize_priors


@dataclass
class TorchGEMSResult:
    labels: torch.Tensor
    posterior: torch.Tensor
    priors: torch.Tensor
    vertices: torch.Tensor
    gaussian_parameters: GaussianParameters
    objective_history: list[float]
    min_jacobian: float

    def mask(self, label_id: int) -> torch.Tensor:
        return self.labels == int(label_id)


class TorchGEMS:
    """GEMS-style generative segmentation on CPU or CUDA.

    The implementation uses FreeSurfer's tetrahedral alpha atlas representation
    and exact Ashburner/KVL deformation-prior cost. Gaussian parameter EM and
    image likelihood are native PyTorch. Mesh deformation is optional because a
    caller may supply an already subject-aligned atlas.
    """

    def __init__(self, atlas: GEMSAtlas, *, device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float32, block_size: int = 8):
        self.atlas = atlas
        self.device = configure_device(device)
        self.dtype = dtype
        self.block_size = int(block_size)

    def _tensors(self, vertices=None):
        a = self.atlas
        v = torch.as_tensor(a.vertices if vertices is None else vertices,
                            device=self.device, dtype=self.dtype)
        ref = torch.as_tensor(a.reference_vertices, device=self.device, dtype=self.dtype)
        tet = torch.as_tensor(a.tetrahedra, device=self.device, dtype=torch.long)
        alpha = torch.as_tensor(a.alphas, device=self.device, dtype=self.dtype)
        movable = torch.as_tensor(a.can_move, device=self.device, dtype=self.dtype)
        label_ids = torch.as_tensor(a.label_ids, device=self.device, dtype=torch.long)
        return v, ref, tet, alpha, movable, label_ids

    def __call__(
        self,
        image: torch.Tensor | np.ndarray,
        *,
        label_classes: torch.Tensor | np.ndarray | None = None,
        em_iterations: int = 8,
        deform_iterations: int = 0,
        deform_lr: float = 0.05,
        deform_optimizer: str = "adam",
        deform_em_interval: int = 1,
        deformation_weight: float = 1.0,
        mean_hyper: torch.Tensor | None = None,
        n_hyper: torch.Tensor | None = None,
        background_channel: int | None = 0,
        mask_to_atlas: bool = False,
        fit_alpha_stages: Sequence[tuple[torch.Tensor, int]] | None = None,
    ) -> TorchGEMSResult:
        image = torch.as_tensor(image, device=self.device, dtype=self.dtype)
        if image.ndim not in (3, 4):
            raise ValueError("image must be [X,Y,Z] or [M,X,Y,Z]")
        shape = tuple(image.shape[-3:])
        vertices, reference, tetra, alphas, movable, label_ids = self._tensors()
        if label_classes is None:
            label_classes = torch.arange(self.atlas.n_labels, device=self.device)
        else:
            label_classes = torch.as_tensor(label_classes, device=self.device, dtype=torch.long)
        if label_classes.shape != (self.atlas.n_labels,):
            raise ValueError("label_classes must have one class index per anatomical label")
        _, label_classes = torch.unique(label_classes, sorted=True, return_inverse=True)
        n_classes = int(label_classes.max()) + 1
        class_ids = torch.arange(n_classes, device=self.device)
        class_alphas = torch.zeros((len(vertices), n_classes), device=self.device, dtype=self.dtype)
        class_alphas.index_add_(1, label_classes, alphas)
        class_background = (None if background_channel is None else
                            int(label_classes[int(background_channel)]))

        stages = ([(class_alphas, int(deform_iterations))] if fit_alpha_stages is None else
                  [(torch.as_tensor(a, device=self.device, dtype=self.dtype), int(n))
                   for a, n in fit_alpha_stages])
        if any(a.shape != class_alphas.shape or n < 0 for a, n in stages):
            raise ValueError("each fit alpha stage must be [vertices, Gaussian classes] with nonnegative iterations")
        total_steps = sum(n for _, n in stages)
        if deform_em_interval < 1:
            raise ValueError("deform_em_interval must be positive")
        index = build_block_index(vertices.detach().cpu().numpy(), self.atlas.tetrahedra,
                                  shape, self.block_size, margin=max(1.0, total_steps * deform_lr))
        if mask_to_atlas:
            with torch.no_grad():
                occupancy = torch.ones((len(vertices), 1), device=self.device, dtype=self.dtype)
                _, working_mask = rasterize_priors(vertices, tetra, occupancy, shape,
                                                    block_index=index, background_channel=None)
            image = image.clone()
            image[..., ~working_mask] = 0
        valid = (image.isfinite().all(0) & (image.abs().sum(0) > 0)
                 if image.ndim == 4 else image.isfinite() & (image != 0))

        def infer(current_vertices, params=None, n_em=em_iterations):
            priors, _ = rasterize_priors(current_vertices, tetra, class_alphas, shape,
                                          block_index=index, background_channel=class_background)
            if params is None:
                params = initialise_gaussians(image, priors, class_ids,
                                               mean_hyper=mean_hyper, n_hyper=n_hyper)
            posterior = priors
            nll = torch.tensor(float("inf"), device=self.device, dtype=self.dtype)
            for _ in range(max(1, int(n_em))):
                ll = gaussian_log_likelihood(image, params)
                posterior, nll = label_posterior(priors, ll, class_ids, valid)
                params = update_gaussians(image, posterior, mean_hyper=mean_hyper,
                                          n_hyper=n_hyper)
            ll = gaussian_log_likelihood(image, params)
            posterior, nll = label_posterior(priors, ll, class_ids, valid)
            return priors, posterior, params, nll

        history: list[float] = []
        priors, posterior, params, nll = infer(vertices)
        history.append(float(nll.detach()))

        if total_steps:
            vertices = vertices.clone().detach().requires_grad_(True)
            for class_alphas, iterations in stages:
                with torch.no_grad():
                    priors, posterior, params, nll = infer(vertices, params=params,
                                                           n_em=em_iterations)
                if deform_optimizer == "adam":
                    optimizer = torch.optim.Adam([vertices], lr=float(deform_lr))
                elif deform_optimizer == "lbfgs":
                    optimizer = torch.optim.LBFGS([vertices], lr=float(deform_lr),
                                                  max_iter=1, history_size=12,
                                                  line_search_fn="strong_wolfe")
                else:
                    raise ValueError("deform_optimizer must be adam or lbfgs")
                for step in range(iterations):
                    def closure():
                        optimizer.zero_grad(set_to_none=True)
                        priors, _ = rasterize_priors(vertices, tetra, class_alphas, shape,
                                                      block_index=index,
                                                      background_channel=class_background)
                        _, data_cost = label_posterior(
                            priors, gaussian_log_likelihood(image, params), class_ids, valid)
                        prior_cost, _ = ashburner_prior(vertices, reference, tetra,
                                                         self.atlas.stiffness)
                        objective = data_cost + float(deformation_weight) * prior_cost
                        objective.backward()
                        if vertices.grad is not None:
                            vertices.grad.mul_(movable)
                        return objective

                    if deform_optimizer == "adam":
                        objective = closure()
                        optimizer.step()
                    else:
                        objective = optimizer.step(closure)
                    history.append(float(objective.detach()))
                    # Update Gaussian parameters after the accepted geometry step.
                    if (step + 1) % deform_em_interval == 0 or step + 1 == iterations:
                        with torch.no_grad():
                            priors, posterior, params, nll = infer(vertices, params=params,
                                                                   n_em=em_iterations)
            vertices = vertices.detach()

        class_alphas = torch.zeros((len(vertices), n_classes), device=self.device, dtype=self.dtype)
        class_alphas.index_add_(1, label_classes, alphas)
        _, _, params, _ = infer(vertices, params=params, n_em=1)
        priors, _ = rasterize_priors(vertices, tetra, alphas, shape,
                                      block_index=index, background_channel=background_channel)
        posterior, _ = label_posterior(priors, gaussian_log_likelihood(image, params), label_classes)
        _, jac = ashburner_prior(vertices, reference, tetra, self.atlas.stiffness)
        hard = torch.where(valid, label_ids[posterior.argmax(0)], label_ids[0])
        return TorchGEMSResult(hard, posterior, priors, vertices, params, history,
                               float(jac.min().detach()) if jac.numel() else float("nan"))
