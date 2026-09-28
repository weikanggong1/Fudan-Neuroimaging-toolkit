"""Independent GPU Bayesian segmentation with a FreeSurfer GEMS atlas."""

from __future__ import annotations

from dataclasses import dataclass

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
        deformation_weight: float = 1.0,
        mean_hyper: torch.Tensor | None = None,
        n_hyper: torch.Tensor | None = None,
        background_channel: int | None = 0,
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

        index = build_block_index(vertices.detach().cpu().numpy(), self.atlas.tetrahedra,
                                  shape, self.block_size, margin=max(1.0, deform_iterations * deform_lr))

        def infer(current_vertices, params=None, n_em=em_iterations):
            priors, _ = rasterize_priors(current_vertices, tetra, alphas, shape,
                                          block_index=index, background_channel=background_channel)
            if params is None:
                params = initialise_gaussians(image, priors, label_classes)
            posterior = priors
            nll = torch.tensor(float("inf"), device=self.device, dtype=self.dtype)
            for _ in range(max(1, int(n_em))):
                ll = gaussian_log_likelihood(image, params)
                posterior, nll = label_posterior(priors, ll, label_classes)
                class_resp = torch.zeros((int(label_classes.max()) + 1, *shape),
                                         device=self.device, dtype=self.dtype)
                class_resp.index_add_(0, label_classes, posterior)
                params = update_gaussians(image, class_resp, mean_hyper=mean_hyper,
                                          n_hyper=n_hyper)
            ll = gaussian_log_likelihood(image, params)
            posterior, nll = label_posterior(priors, ll, label_classes)
            return priors, posterior, params, nll

        history: list[float] = []
        priors, posterior, params, nll = infer(vertices)
        history.append(float(nll.detach()))

        if deform_iterations:
            vertices = vertices.clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([vertices], lr=float(deform_lr))
            for _ in range(int(deform_iterations)):
                optimizer.zero_grad(set_to_none=True)
                priors, posterior, _, data_cost = infer(vertices, params=params, n_em=1)
                prior_cost, jac = ashburner_prior(vertices, reference, tetra, self.atlas.stiffness)
                objective = data_cost + float(deformation_weight) * prior_cost
                objective.backward()
                if vertices.grad is not None:
                    vertices.grad.mul_(movable)
                optimizer.step()
                history.append(float(objective.detach()))
                # Update Gaussian parameters after the accepted geometry step.
                with torch.no_grad():
                    priors, posterior, params, nll = infer(vertices, params=params, n_em=1)
            vertices = vertices.detach()

        priors, posterior, params, _ = infer(vertices, params=params, n_em=1)
        _, jac = ashburner_prior(vertices, reference, tetra, self.atlas.stiffness)
        hard = label_ids[posterior.argmax(0)]
        return TorchGEMSResult(hard, posterior, priors, vertices, params, history,
                               float(jac.min().detach()) if jac.numel() else float("nan"))
