"""Brainstem-specific mesh fit to an aseg/SynthSeg coarse segmentation."""

from __future__ import annotations

from dataclasses import replace
from time import monotonic

import numpy as np
import nibabel as nib
from scipy import ndimage
import torch

from .atlas import GEMSAtlas
from .deformation import (ashburner_prior, prepare_current_geometry,
                          prepare_deformation_reference, prepare_vertex_reduction)
from .optim import CachedLBFGS
from .rasterize import (build_block_index, rasterize_priors,
                       rasterize_priors_compact)
from .precision import parallel_precision_enabled


def brainstem_gaussian_hyperparameters(
    image: nib.spatialimages.SpatialImage,
    coarse_labels: np.ndarray,
    label_classes: np.ndarray,
    atlas_label_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate brainstem Gaussian priors from eroded native-space tissue labels."""
    class_masks = {
        0: [16, 28, 60], 1: [4, 43, 14, 15],
        2: [3, 42, 17, 53, 18, 54], 3: [11, 50],
        4: [26, 58], 5: [13, 52], 6: [12, 51],
        7: [10, 49], 8: [31, 63], 9: [2, 41],
        10: [0], 11: [7, 46], 12: [8, 47],
    }
    classes = np.asarray(label_classes, dtype=np.int64)
    labels = np.asarray(atlas_label_ids, dtype=np.int64)
    if classes.shape != labels.shape:
        raise ValueError("label_classes and atlas_label_ids must have equal length")
    data = np.asarray(image.dataobj, dtype=np.float32)
    if data.shape != coarse_labels.shape:
        raise ValueError("image and coarse_labels must have equal shape")
    voxel_volume = abs(np.linalg.det(image.affine[:3, :3]))
    ball = ndimage.generate_binary_structure(3, 1)
    means = np.zeros(int(classes.max()) + 1, dtype=np.float32)
    counts = np.zeros_like(means)
    for group in range(len(means)):
        if group not in class_masks:
            continue
        mask = ndimage.binary_erosion(np.isin(coarse_labels, class_masks[group]),
                                      structure=ball, border_value=1)
        values = data[mask & np.isfinite(data) & (data > 0)]
        if values.size:
            means[group] = np.median(values)
            counts[group] = 10 + 0.1 * values.size / voxel_volume
        else:
            means[group] = 55
            counts[group] = 10
    return means, counts


def make_brainstem_working_image(
    image: nib.spatialimages.SpatialImage,
    coarse_labels: np.ndarray,
    resolution_mm: float = 0.5,
) -> tuple[nib.Nifti1Image, np.ndarray, dict]:
    """Crop around the coarse brainstem and resample T1 and labels to isotropic voxels."""
    points = np.argwhere(np.isin(coarse_labels, [16, 28, 60]))
    if not len(points):
        raise ValueError("coarse segmentation has no brainstem or ventral diencephalon")
    voxel_sizes = np.linalg.norm(image.affine[:3, :3], axis=0)
    margin = int(np.rint(15 / voxel_sizes.mean()))
    low = np.maximum(points.min(0) - margin, 0)
    high = np.minimum(points.max(0) + margin + 1, coarse_labels.shape)
    step = resolution_mm / voxel_sizes
    shape = tuple(np.ceil((high - low) / step - 1e-5).astype(int))
    origin = low + ((high - low) - np.asarray(shape) * step) / 2
    scale = np.diag(step)
    data = ndimage.affine_transform(
        np.asarray(image.dataobj, dtype=np.float32), scale, offset=origin,
        output_shape=shape, order=3, mode="nearest").astype(np.float32)
    labels = ndimage.affine_transform(
        coarse_labels, scale, offset=origin, output_shape=shape, order=0, mode="nearest")
    data[ndimage.distance_transform_edt(labels == 0) > 5 / resolution_mm] = 0
    transform = np.eye(4)
    transform[:3, :3] = scale
    transform[:3, 3] = origin
    working = nib.Nifti1Image(data, image.affine @ transform)
    return working, labels, {"resolution_mm": resolution_mm,
                             "crop_start_native": low.tolist(),
                             "crop_stop_native": high.tolist(),
                             "grid_origin_native_voxel": origin.tolist(),
                             "working_shape": [int(v) for v in shape]}


def fit_brainstem_segmentation(
    atlas: GEMSAtlas,
    coarse_labels: np.ndarray,
    *,
    device: str | torch.device,
    iterations: int = 40,
    fit_alphas: np.ndarray | None = None,
    optimizer_name: str = "adam",
    stable_mesh_fitting: bool = False,
) -> tuple[GEMSAtlas, dict]:
    """Fit the reference mesh to coarse brainstem and surrounding labels.

    The first class joins brainstem, ventral diencephalon, cerebellum and fourth
    ventricle, as in the reference synthetic-image stage. The target labels
    come from an existing coarse segmentation on the T1 voxel grid.

    Stable fitting fixes the shared gradient reduction order and promotes only
    scalar objective sums and optional L-BFGS state to FP64. Image values,
    vertices, priors and Adam state stay FP32. Accurate geometric products are
    local to this synthetic stage; the caller's TF32 policy is restored.
    """
    if not isinstance(stable_mesh_fitting, bool):
        raise ValueError("stable_mesh_fitting must be a bool")
    arguments = dict(device=device, iterations=iterations, fit_alphas=fit_alphas,
                     optimizer_name=optimizer_name, stable_mesh_fitting=stable_mesh_fitting)
    if (not stable_mesh_fitting or torch.device(device).type != "cuda"
            or parallel_precision_enabled()):
        return _fit_brainstem_segmentation(atlas, coarse_labels, **arguments)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        return _fit_brainstem_segmentation(atlas, coarse_labels, **arguments)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def _fit_brainstem_segmentation(
    atlas: GEMSAtlas,
    coarse_labels: np.ndarray,
    *,
    device: str | torch.device,
    iterations: int,
    fit_alphas: np.ndarray | None,
    optimizer_name: str,
    stable_mesh_fitting: bool,
) -> tuple[GEMSAtlas, dict]:
    """Execute the unchanged configured synthetic objective and step schedule."""
    start = monotonic()
    atlas = replace(atlas, vertices=atlas.reference_vertices.copy(), stiffness=0.05)
    low = np.maximum(np.floor(atlas.vertices.min(0)).astype(int) - 8, 0)
    high = np.minimum(np.ceil(atlas.vertices.max(0)).astype(int) + 9, coarse_labels.shape)
    crop = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
    shift = np.eye(4)
    shift[:3, 3] = -low
    local = atlas.transformed(shift, transform_reference=True)
    shape = coarse_labels[crop].shape
    vertices = torch.nn.Parameter(torch.as_tensor(local.vertices, device=device, dtype=torch.float32).clone())
    reference = torch.as_tensor(local.reference_vertices, device=device, dtype=torch.float32)
    tetrahedra = torch.as_tensor(local.tetrahedra, device=device, dtype=torch.long)
    group = np.isin(local.label_ids, [175, 174, 178, 173, 28, 7, 8, 15])
    if fit_alphas is None:
        fit_alphas = np.stack((local.alphas[:, group].sum(1),
                               local.alphas[:, ~group].sum(1)), axis=1)
    if fit_alphas.shape != (len(local.vertices), 2):
        raise ValueError("brainstem fit_alphas must be [vertices, 2]")
    alphas = torch.as_tensor(fit_alphas, device=device, dtype=torch.float32)
    index = build_block_index(local.vertices, local.tetrahedra, shape, block_size=12, margin=10)
    with torch.no_grad():
        _, covered = rasterize_priors(vertices, tetrahedra, alphas, shape,
                                      block_index=index, background_channel=1)
    xyz = np.stack(np.meshgrid(*[np.arange(-5, 6)] * 3, indexing="ij"), axis=-1)
    ball = np.square(xyz).sum(-1) <= 25
    interior = ndimage.binary_erosion(covered.cpu().numpy(), structure=ball, border_value=1)
    mask = torch.as_tensor(interior, device=device)
    target = torch.as_tensor(np.isin(coarse_labels[crop], [16, 7, 8, 15, 28, 46, 47, 60]),
                             device=device)
    # The reference objective only evaluates the eroded atlas mask. Keep that
    # mask fixed and rasterize its points directly instead of allocating the
    # full crop for every trial in the line search.
    expected = target[mask]
    reference_geometry = prepare_deformation_reference(reference, tetrahedra)
    vertex_reduction = (prepare_vertex_reduction(tetrahedra.reshape(-1), len(vertices))
                        if stable_mesh_fitting else None)
    movable = torch.as_tensor(local.can_move, device=device)
    mesh_evaluations = 0
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam([vertices], lr=0.1)
    elif optimizer_name == "lbfgs":
        state_options = {"double_precision_state": True} if stable_mesh_fitting else {}
        optimizer = CachedLBFGS([vertices], lr=0.8, max_iter=1,
                                history_size=12, line_search_fn="strong_wolfe", **state_options)
    else:
        raise ValueError("optimizer_name must be adam or lbfgs")
    for _ in range(int(iterations)):
        def closure():
            nonlocal mesh_evaluations
            mesh_evaluations += 1
            optimizer.zero_grad(set_to_none=True)
            geometry = prepare_current_geometry(
                vertices, tetrahedra, deterministic_gradient=stable_mesh_fitting,
                vertex_reduction=vertex_reduction)
            priors, _ = rasterize_priors_compact(
                vertices, tetrahedra, alphas, shape, valid_mask=mask,
                block_index=index, background_channel=1, current_geometry=geometry)
            p = priors[0].clamp(1e-5, 1 - 1e-5)
            selected = torch.where(expected, torch.log(p), torch.log1p(-p))
            data_cost = -selected.sum(dtype=torch.float64 if stable_mesh_fitting else selected.dtype)
            regularizer, _ = ashburner_prior(
                vertices, reference, tetrahedra, atlas.stiffness,
                reference_geometry=reference_geometry, current_geometry=geometry,
                analytic_gradient=True, double_accumulation=stable_mesh_fitting)
            objective = data_cost + regularizer
            objective.backward()
            vertices.grad.mul_(movable)
            return objective

        if optimizer_name == "adam":
            closure()
            optimizer.step()
        else:
            optimizer.step(closure)
    with torch.no_grad():
        geometry = prepare_current_geometry(vertices, tetrahedra)
        priors, _ = rasterize_priors_compact(
            vertices, tetrahedra, alphas, shape, valid_mask=mask,
            block_index=index, background_channel=1, current_geometry=geometry)
        predicted = priors[0] > 0.5
        dice = 2 * (predicted & expected).sum() / (predicted.sum() + expected.sum()).clamp_min(1)
        _, jacobian = ashburner_prior(
            vertices, reference, tetrahedra, atlas.stiffness,
            reference_geometry=reference_geometry, current_geometry=geometry,
            analytic_gradient=True)
        displacement = (vertices - reference).norm(dim=1).max()
    fitted = atlas.with_vertices(vertices.detach().cpu().numpy() + low)
    return fitted, {"mask_dice": float(dice), "min_jacobian": float(jacobian.min()),
                    "max_displacement_voxels": float(displacement), "seconds": monotonic() - start,
                    "mesh_solver": {"compact": True, "shared_geometry": True,
                                    "analytic_prior": True,
                                    "stable_mesh_fitting": stable_mesh_fitting,
                                    "ordered_shared_vertex_gradient": stable_mesh_fitting,
                                    "ordered_point_to_tetrahedron_gradient": stable_mesh_fitting,
                                    "data_cost_accumulation": "float64" if stable_mesh_fitting else "float32",
                                    "prior_cost_accumulation": "float64" if stable_mesh_fitting else "float32",
                                    "precise_mesh_matrices": stable_mesh_fitting,
                                    "matrix_precision_scope": "custom synthetic stage; caller TF32 restored",
                                    "optimizer_state_precision": "float64" if stable_mesh_fitting
                                    and optimizer_name == "lbfgs" else "float32",
                                    "optimizer": optimizer_name,
                                    "line_search": "strong_wolfe" if optimizer_name == "lbfgs" else None,
                                    "alpha_policy": "fixed configured arrays",
                                    "global_deterministic_algorithms_changed": False,
                                    "cublas_workspace_config_required": False,
                                    "mesh_evaluations": mesh_evaluations,
                                    "mesh_steps": int(iterations),
                                    "accepted_cache_hits": getattr(optimizer, "cache_hits", 0),
                                    "valid_voxels": int(mask.sum())}}
