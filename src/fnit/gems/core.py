"""Independent GPU Bayesian segmentation with a FreeSurfer GEMS atlas."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from time import monotonic
from typing import Sequence

import numpy as np
from scipy import ndimage
import torch

from .._dmri import configure_device
from .atlas import GEMSAtlas
from .deformation import (ashburner_prior, prepare_current_geometry, prepare_deformation_reference,
                          prepare_vertex_reduction, sliding_boundary_projectors)
from .gaussian import (GaussianParameters, gaussian_log_likelihood,
                       initialise_gaussians, label_posterior, update_gaussians,
                       _class_posterior_from_log_prior)
from .optim import CachedLBFGS, PrecisionLBFGS
from .rasterize import (BlockIndex, build_block_index, compact_data_cost, rasterize_priors,
                       rasterize_priors_compact)
from .precision import parallel_precision_enabled


logger = logging.getLogger(__name__)


def _cpu_em_log_prior(priors, image, params):
    # EM iterations share these priors until the next mesh rasterization.
    # Leave differentiable calls and every CUDA call on their original path.
    if (priors.device.type != "cpu" or priors.requires_grad or image.requires_grad
            or params.means.requires_grad or params.covariances.requires_grad):
        return None
    return torch.log(priors.clamp_min(torch.finfo(priors.dtype).tiny))


@dataclass
class TorchGEMSResult:
    labels: torch.Tensor | None
    posterior: torch.Tensor | None
    priors: torch.Tensor | None
    vertices: torch.Tensor
    gaussian_parameters: GaussianParameters
    objective_history: list[float]
    min_jacobian: float
    affine: np.ndarray | None = None
    highres_labels: object | None = None
    optimization_stats: dict | None = None

    def mask(self, label_id: int) -> torch.Tensor:
        if self.labels is None:
            raise ValueError("labels were not materialized for this intermediate fit")
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
        # Parallel region workers keep the process-wide TF32 policy fixed while
        # their independent CUDA streams overlap.
        self.device = configure_device(
            device, configure_precision=not parallel_precision_enabled())
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
        atlas_mask_erosion: int = 0,
        fit_alpha_stages: Sequence[tuple[torch.Tensor, int]] | None = None,
        fixed_gaussians: GaussianParameters | None = None,
        initial_gaussians: GaussianParameters | None = None,
        index_refresh_interval: int | None = None,
        index_margin: float | None = None,
        adaptive_index: bool = False,
        relative_cost_stop: float | None = None,
        outer_iterations: int = 1,
        em_relative_cost_stop: float | None = None,
        outer_relative_cost_stop: float | None = None,
        boundary_transform: np.ndarray | torch.Tensor | None = None,
        compact: bool | None = None,
        reuse_geometry: bool = True,
        analytic_prior: bool = True,
        cache_mesh_evaluations: bool = True,
        deformation_stop: float = 1e-10,
        cost_stop_patience: int = 1,
        fused_data_cost_enabled: bool = True,
        materialize_outputs: bool = True,
        mesh_sampling_stride: int = 1,
        owner_hint_enabled: bool = False,
        owner_hint_refresh_interval: int = 8,
        owner_hint_tolerance: float = 2e-4,
        double_data_cost_accumulation: bool = False,
        stable_mesh_fitting: bool = False,
        precise_mesh_matrices: bool = False,
        mesh_line_search: str = "strong_wolfe",
    ) -> TorchGEMSResult:
        image = torch.as_tensor(image, device=self.device, dtype=self.dtype)
        if image.ndim not in (3, 4):
            raise ValueError("image must be [X,Y,Z] or [M,X,Y,Z]")
        shape = tuple(image.shape[-3:])
        vertices, reference, tetra, alphas, movable, label_ids = self._tensors()
        reference_geometry = prepare_deformation_reference(reference, tetra)
        projection = (None if boundary_transform is None else sliding_boundary_projectors(
            movable.bool(), torch.as_tensor(boundary_transform, device=self.device, dtype=self.dtype)))
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
        if outer_iterations < 1:
            raise ValueError("outer_iterations must be positive")
        if not np.isfinite(deformation_stop) or deformation_stop < 0:
            raise ValueError("deformation_stop must be finite and nonnegative")
        if not isinstance(cost_stop_patience, int) or cost_stop_patience < 1:
            raise ValueError("cost_stop_patience must be a positive integer")
        if not isinstance(mesh_sampling_stride, int) or mesh_sampling_stride < 1:
            raise ValueError("mesh_sampling_stride must be a positive integer")
        if not isinstance(owner_hint_refresh_interval, int) or owner_hint_refresh_interval < 1:
            raise ValueError("owner_hint_refresh_interval must be a positive integer")
        if not np.isfinite(owner_hint_tolerance) or owner_hint_tolerance <= 0:
            raise ValueError("owner_hint_tolerance must be finite and positive")
        if mesh_line_search not in ("strong_wolfe", "backtracking"):
            raise ValueError("mesh_line_search must be strong_wolfe or backtracking")
        if mesh_line_search == "backtracking" and (
                deform_optimizer != "lbfgs" or not cache_mesh_evaluations or not stable_mesh_fitting):
            raise ValueError("backtracking requires stable cached L-BFGS mesh fitting")
        margin = (max(1.0, total_steps * deform_lr) if index_margin is None else
                  float(index_margin))
        index = build_block_index(vertices.detach().cpu().numpy(), self.atlas.tetrahedra,
                                  shape, self.block_size, margin=margin)
        index_anchor = vertices.detach().clone()
        index_rebuilds = 0
        if mask_to_atlas:
            with torch.no_grad():
                occupancy = torch.ones((len(vertices), 1), device=self.device, dtype=self.dtype)
                _, working_mask = rasterize_priors(vertices, tetra, occupancy, shape,
                                                    block_index=index, background_channel=None)
                if atlas_mask_erosion:
                    radius = int(atlas_mask_erosion)
                    coords = np.arange(-radius, radius + 1)
                    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
                    sphere = x*x + y*y + z*z <= radius*radius
                    eroded = ndimage.binary_erosion(working_mask.cpu().numpy(),
                                                    structure=sphere, border_value=1)
                    working_mask = torch.as_tensor(eroded, device=self.device)
            image = image.clone()
            image[..., ~working_mask] = 0
        valid = (image.isfinite().all(0) & (image.abs().sum(0) > 0)
                 if image.ndim == 4 else image.isfinite() & (image != 0))
        # Compact computation is independent of how EM is stopped. In particular,
        # synthetic-label fits use fixed Gaussians and a sparse valid mask.
        compact_em = image.ndim == 3 and compact is not False
        if compact is True and image.ndim != 3:
            raise ValueError("compact=True requires a single 3-D image")
        if double_data_cost_accumulation and not compact_em:
            raise ValueError("double data cost accumulation requires compact single-image fitting")
        if stable_mesh_fitting and (not compact_em or not reuse_geometry):
            raise ValueError("stable mesh fitting requires compact single-image fitting and shared geometry")
        double_data_cost_accumulation = double_data_cost_accumulation or stable_mesh_fitting
        vertex_reduction = (prepare_vertex_reduction(tetra.reshape(-1), len(vertices))
                            if stable_mesh_fitting else None)
        em_image = image[valid].reshape(-1, 1, 1) if compact_em else image
        mesh_valid = valid
        mesh_selection = None
        sampling_scale = 1.0
        if mesh_sampling_stride > 1:
            if not compact_em:
                raise ValueError("mesh sampling requires compact single-image fitting")
            x = torch.arange(shape[0], device=self.device)[:, None, None]
            y = torch.arange(shape[1], device=self.device)[None, :, None]
            z = torch.arange(shape[2], device=self.device)[None, None, :]
            # Fixed spatial quadrature changes only the mesh data integral.
            # EM and final dense anatomical outputs retain all valid voxels.
            quadrature = (x + 3 * y + 5 * z) % mesh_sampling_stride == 0
            selection = quadrature[valid]
            sampled_count = int(selection.sum())
            if sampled_count:
                mesh_selection = selection
                mesh_valid = valid & quadrature
                sampling_scale = selection.numel() / sampled_count

        def refresh_index(current_vertices):
            nonlocal index, index_anchor, index_rebuilds
            if adaptive_index and (current_vertices.detach() - index_anchor).abs().amax().item() > margin / 2:
                index = build_block_index(current_vertices.detach().cpu().numpy(),
                                          self.atlas.tetrahedra, shape,
                                          self.block_size, margin=margin)
                index_anchor = current_vertices.detach().clone()
                index_rebuilds += 1

        def infer(current_vertices, params=None, n_em=em_iterations):
            refresh_index(current_vertices)
            if compact_em:
                priors, _ = rasterize_priors_compact(
                    current_vertices, tetra, class_alphas, shape, valid_mask=valid,
                    block_index=index, background_channel=class_background)
                em_priors = priors.reshape(n_classes, -1, 1, 1)
            else:
                priors, _ = rasterize_priors(current_vertices, tetra, class_alphas, shape,
                                              block_index=index, background_channel=class_background)
                em_priors = priors
            if fixed_gaussians is not None:
                params = fixed_gaussians
            elif params is None:
                params = initialise_gaussians(em_image, em_priors, class_ids,
                                               mean_hyper=mean_hyper, n_hyper=n_hyper)
            cpu_log_prior = _cpu_em_log_prior(em_priors, em_image, params)

            def em_posterior(likelihood):
                if cpu_log_prior is not None:
                    return _class_posterior_from_log_prior(
                        cpu_log_prior, likelihood, None if compact_em else valid)
                return label_posterior(em_priors, likelihood, class_ids,
                                       None if compact_em else valid)

            posterior = priors
            nll = torch.tensor(float("inf"), device=self.device, dtype=self.dtype)
            for _ in range(max(0, int(n_em))):
                ll = gaussian_log_likelihood(em_image, params)
                posterior, nll = em_posterior(ll)
                em_cost = nll
                if em_relative_cost_stop is not None and mean_hyper is not None and n_hyper is not None:
                    variance = params.covariances[:, 0, 0]
                    hyper_mass = n_hyper.to(variance).flatten()
                    hyper_mean = mean_hyper.to(variance).flatten()
                    em_cost = em_cost + (0.5 * (2 * torch.pi * variance).log()
                        - 0.5 * hyper_mass.log()
                        + 0.5 * hyper_mass * (params.means[:, 0] - hyper_mean).square()
                        / variance).sum()
                if em_relative_cost_stop is not None and _ > 0:
                    change = (previous_em_cost - em_cost) / em_cost.abs().clamp_min(1)
                    if change.item() < em_relative_cost_stop:
                        break
                previous_em_cost = em_cost.detach()
                if fixed_gaussians is None:
                    params = update_gaussians(em_image, posterior, mean_hyper=mean_hyper,
                                              n_hyper=n_hyper)
            ll = gaussian_log_likelihood(em_image, params)
            posterior, nll = em_posterior(ll)
            return priors, posterior, params, nll

        history_tensors: list[torch.Tensor] = []
        mesh_evaluations = mesh_steps = cache_hits = fused_evaluations = 0
        line_search_restarts = line_search_recovered = 0
        owner_hint_evaluations = owner_full_evaluations = owner_hint_points = 0
        owner_hint_reused = []
        if em_relative_cost_stop is not None:
            class_alphas = stages[0][0]
        em_started = monotonic()
        priors, posterior, params, nll = infer(vertices, params=initial_gaussians)
        if compact_em:
            logger.info("GEMS initial EM: %.2f s, %d valid voxels", monotonic() - em_started,
                        em_image.numel())
        history_tensors.append(nll.detach())

        if total_steps:
            vertices = vertices.clone().detach().requires_grad_(True)
            global_step = 0
            for class_alphas, iterations in stages:
                previous_outer_cost = None
                for outer in range(outer_iterations):
                    em_started = monotonic()
                    with torch.no_grad():
                        priors, posterior, params, nll = infer(vertices, params=params,
                                                               n_em=em_iterations)
                    # Likelihood stays fixed while optimizing the mesh geometry.
                    likelihood = gaussian_log_likelihood(em_image, params).detach()
                    if compact_em:
                        likelihood = likelihood.reshape(n_classes, -1)
                        logger.info("GEMS outer %d/%d EM: %.2f s", outer + 1, outer_iterations,
                                    monotonic() - em_started)
                    mesh_likelihood = (likelihood[:, mesh_selection]
                                       if mesh_selection is not None else likelihood)
                    mesh_started = monotonic()
                    evaluations = 0
                    have_moved = False
                    cost_stalls = deformation_stalls = 0
                    previous_mesh_cost = None
                    if deform_optimizer == "adam":
                        optimizer = torch.optim.Adam([vertices], lr=float(deform_lr))
                    elif deform_optimizer == "lbfgs":
                        if mesh_line_search == "backtracking":
                            from .optim import CachedArmijoLBFGS
                            optimizer_type = CachedArmijoLBFGS
                        else:
                            optimizer_type = CachedLBFGS if cache_mesh_evaluations else PrecisionLBFGS
                        optimizer = optimizer_type([vertices], lr=float(deform_lr),
                                                      max_iter=1, history_size=12,
                                                      tolerance_grad=1e-10, tolerance_change=1e-10,
                                                      line_search_fn="strong_wolfe",
                                                      double_precision_state=stable_mesh_fitting)
                    else:
                        raise ValueError("deform_optimizer must be adam or lbfgs")
                    for step in range(iterations):
                        previous_vertices = (vertices.detach().clone()
                                             if stable_mesh_fitting or projection is not None
                                             or deformation_stop > 1e-10 else None)
                        def mesh_objective():
                            nonlocal evaluations, fused_evaluations
                            nonlocal owner_hint_evaluations, owner_full_evaluations, owner_hint_points
                            evaluations += 1
                            refresh_index(vertices)
                            optimizer.zero_grad(set_to_none=True)
                            geometry = (prepare_current_geometry(
                                        vertices, tetra, deterministic_gradient=stable_mesh_fitting,
                                        vertex_reduction=vertex_reduction)
                                        if reuse_geometry else None)
                            if compact_em:
                                hint_stats = {} if owner_hint_enabled else None
                                use_owner_hints = (owner_hint_enabled and
                                    (evaluations - 1) % owner_hint_refresh_interval != 0)
                                data_cost = (compact_data_cost(
                                    vertices, tetra, class_alphas, shape, valid_mask=mesh_valid,
                                    block_index=index, background_channel=class_background,
                                    current_geometry=geometry, likelihood=mesh_likelihood,
                                    cache_owner_hints=owner_hint_enabled,
                                    owner_hints=use_owner_hints,
                                    hint_tolerance=owner_hint_tolerance,
                                    hint_stats=hint_stats,
                                    double_accumulation=double_data_cost_accumulation,
                                    deterministic_gradient=stable_mesh_fitting)
                                    if fused_data_cost_enabled else None)
                                if data_cost is None:
                                    priors, _ = rasterize_priors_compact(
                                        vertices, tetra, class_alphas, shape, valid_mask=mesh_valid,
                                        block_index=index, background_channel=class_background,
                                        current_geometry=geometry)
                                    joint = priors.clamp_min(torch.finfo(priors.dtype).tiny).log() + mesh_likelihood
                                    data_cost = -joint.logsumexp(dim=0).sum(
                                        dtype=torch.float64 if double_data_cost_accumulation else joint.dtype)
                                else:
                                    fused_evaluations += 1
                                    if owner_hint_enabled:
                                        owner_hint_evaluations += int(use_owner_hints)
                                        owner_full_evaluations += int(not use_owner_hints)
                                        owner_hint_points += hint_stats["evaluated_points"]
                                        owner_hint_reused.append(hint_stats["reused_points"])
                                data_cost = data_cost * sampling_scale
                            else:
                                priors, _ = rasterize_priors(vertices, tetra, class_alphas, shape,
                                                              block_index=index,
                                                              background_channel=class_background,
                                                              current_geometry=geometry)
                                _, data_cost = label_posterior(
                                    priors, likelihood, class_ids, valid)
                            prior_cost, jacobian = ashburner_prior(vertices, reference, tetra,
                                self.atlas.stiffness, reference_geometry=reference_geometry,
                                current_geometry=geometry, analytic_gradient=analytic_prior,
                                double_accumulation=stable_mesh_fitting)
                            objective = data_cost + float(deformation_weight) * prior_cost
                            if mesh_line_search == "backtracking":
                                # Reject inverted trial meshes independently of
                                # the finite penalty's numerical magnitude.
                                objective = torch.where(
                                    torch.isfinite(jacobian).all() & (jacobian > 0).all(),
                                    objective, objective.new_tensor(float("inf")))
                            objective.backward()
                            if vertices.grad is not None:
                                if projection is None:
                                    vertices.grad.mul_(movable)
                                else:
                                    vertices.grad.copy_(torch.bmm(projection, vertices.grad[..., None]).squeeze(-1))
                            return objective

                        def closure():
                            if (not precise_mesh_matrices
                                    or vertices.device.type != "cuda"
                                    or parallel_precision_enabled()):
                                return mesh_objective()
                            # Tiny geometric products need consistent FP32 cost
                            # and derivatives. Restore the caller's TF32 policy
                            # before returning to EM or any other model.
                            previous_tf32 = torch.backends.cuda.matmul.allow_tf32
                            try:
                                torch.backends.cuda.matmul.allow_tf32 = False
                                return mesh_objective()
                            finally:
                                torch.backends.cuda.matmul.allow_tf32 = previous_tf32

                        if deform_optimizer == "adam":
                            objective = closure()
                            optimizer.step()
                        else:
                            if isinstance(optimizer, CachedLBFGS):
                                objective = optimizer.step(closure, cache_key=(index_rebuilds, id(likelihood), id(class_alphas)))
                                if optimizer.accepted_objective is not None:
                                    objective = optimizer.accepted_objective
                            else:
                                objective = optimizer.step(closure)
                            if stable_mesh_fitting and mesh_line_search == "strong_wolfe" and previous_vertices is not None:
                                # A rejected Wolfe direction is not convergence.
                                # Retry once from the same accepted position with
                                # a fresh, normalized steepest-descent direction.
                                stalled = torch.equal(vertices.detach(), previous_vertices)
                                accepted_gradient = (optimizer._cached[1]
                                    if isinstance(optimizer, CachedLBFGS) and optimizer._cached is not None
                                    else optimizer.state[vertices].get("prev_flat_grad"))
                                if accepted_gradient is None:
                                    accepted_gradient = optimizer._gather_flat_grad()
                                if stalled and accepted_gradient.abs().max().item() > 1e-10:
                                    line_search_restarts += 1
                                    optimizer.reset_history()
                                    if isinstance(optimizer, CachedLBFGS):
                                        objective = optimizer.step(closure, cache_key=(
                                            index_rebuilds, id(likelihood), id(class_alphas)))
                                        if optimizer.accepted_objective is not None:
                                            objective = optimizer.accepted_objective
                                    else:
                                        objective = optimizer.step(closure)
                                    line_search_recovered += int(not torch.equal(
                                        vertices.detach(), previous_vertices))
                        global_step += 1
                        mesh_steps += 1
                        if index_refresh_interval and global_step % index_refresh_interval == 0:
                            index = build_block_index(vertices.detach().cpu().numpy(),
                                                      self.atlas.tetrahedra, shape,
                                                      self.block_size, margin=margin)
                            index_anchor = vertices.detach().clone()
                            index_rebuilds += 1
                        history_tensors.append(objective.detach())
                        if previous_vertices is not None:
                            maximal_deformation = torch.linalg.vector_norm(
                                vertices.detach() - previous_vertices, dim=1).amax().item()
                            have_moved |= maximal_deformation > 0
                            deformation_stalls = deformation_stalls + 1 if maximal_deformation <= deformation_stop else 0
                            if maximal_deformation == 0 or deformation_stalls >= cost_stop_patience:
                                break
                        # Update Gaussian parameters after the accepted geometry step.
                        objective_changed = False
                        if outer_iterations == 1 and em_relative_cost_stop is None and fixed_gaussians is None and ((step + 1) % deform_em_interval == 0
                                                      or step + 1 == iterations):
                            with torch.no_grad():
                                priors, posterior, params, nll = infer(vertices, params=params,
                                                                       n_em=em_iterations)
                            likelihood = gaussian_log_likelihood(em_image, params).detach()
                            if compact_em:
                                likelihood = likelihood.reshape(n_classes, -1)
                            mesh_likelihood = (likelihood[:, mesh_selection]
                                               if mesh_selection is not None else likelihood)
                            if isinstance(optimizer, CachedLBFGS):
                                optimizer.invalidate_cache()
                            objective_changed = True
                            previous_mesh_cost = None
                            cost_stalls = 0
                        if relative_cost_stop is not None and previous_mesh_cost is not None and not objective_changed:
                            relative_change = ((previous_mesh_cost - objective).abs() /
                                               objective.abs().clamp_min(1)).item()
                            cost_stalls = cost_stalls + 1 if relative_change < relative_cost_stop else 0
                            if cost_stalls >= cost_stop_patience:
                                break
                        if not objective_changed:
                            previous_mesh_cost = objective.detach()
                    mesh_evaluations += evaluations
                    if isinstance(optimizer, CachedLBFGS):
                        cache_hits += optimizer.cache_hits
                    if compact_em:
                        logger.info("GEMS outer %d/%d mesh: %.2f s, %d evaluations", outer + 1,
                                    outer_iterations, monotonic() - mesh_started, evaluations)
                    if projection is not None and not have_moved:
                        break
                    if outer_relative_cost_stop is not None and iterations:
                        current_outer_cost = history_tensors[-1]
                        if previous_outer_cost is not None:
                            relative_change = ((previous_outer_cost - current_outer_cost).abs() /
                                               current_outer_cost.abs().clamp_min(1)).item()
                            if relative_change < outer_relative_cost_stop:
                                break
                        previous_outer_cost = current_outer_cost
            vertices = vertices.detach()

        class_alphas = torch.zeros((len(vertices), n_classes), device=self.device, dtype=self.dtype)
        class_alphas.index_add_(1, label_classes, alphas)
        if em_relative_cost_stop is not None:
            refresh_index(vertices)
        else:
            _, _, params, _ = infer(vertices, params=params, n_em=1)
        if materialize_outputs:
            priors, _ = rasterize_priors(vertices, tetra, alphas, shape,
                                          block_index=index, background_channel=background_channel)
            posterior, _ = label_posterior(priors, gaussian_log_likelihood(image, params), label_classes)
            posterior = torch.where(valid[None], posterior, priors)
            hard = torch.where(valid, label_ids[posterior.argmax(0)], label_ids[0])
        else:
            priors = posterior = hard = None
        _, jac = ashburner_prior(vertices, reference, tetra, self.atlas.stiffness,
                                reference_geometry=reference_geometry)
        history = torch.stack(history_tensors).cpu().tolist()
        return TorchGEMSResult(hard, posterior, priors, vertices, params, history,
                               float(jac.min().detach()) if jac.numel() else float("nan"),
                               optimization_stats={"compact": compact_em,
                                   "objective_history": history,
                                   "mesh_evaluations": mesh_evaluations, "mesh_steps": mesh_steps,
                                   "line_search_restarts": line_search_restarts,
                                   "line_search_recovered": line_search_recovered,
                                   "accepted_cache_hits": cache_hits, "index_rebuilds": index_rebuilds,
                                   "shared_geometry": reuse_geometry, "analytic_prior": analytic_prior,
                                   "fused_data_evaluations": fused_evaluations,
                                   "materialized_outputs": bool(materialize_outputs),
                                   "mesh_sampling_stride": int(mesh_sampling_stride),
                                   "mesh_sampling_scale": sampling_scale,
                                   "owner_hint_enabled": bool(owner_hint_enabled),
                                   "owner_hint_refresh_interval": int(owner_hint_refresh_interval),
                                   "owner_hint_tolerance": float(owner_hint_tolerance),
                                   "owner_hint_evaluations": owner_hint_evaluations,
                                   "owner_full_evaluations": owner_full_evaluations,
                                   "owner_hint_evaluated_points": owner_hint_points,
                                   "owner_hint_reused_points": int(torch.stack(owner_hint_reused).sum())
                                       if owner_hint_reused else 0,
                                   "data_cost_accumulation": "float64" if double_data_cost_accumulation
                                       else str(self.dtype),
                                   "stable_mesh_fitting": bool(stable_mesh_fitting),
                                   "precise_mesh_matrices": bool(precise_mesh_matrices),
                                   "mesh_line_search": mesh_line_search if deform_optimizer == "lbfgs" else None,
                                   "prior_cost_accumulation": "float64" if stable_mesh_fitting
                                       else str(self.dtype),
                                   "optimizer_state_precision": "float64" if stable_mesh_fitting
                                       and deform_optimizer == "lbfgs" else str(self.dtype)})
