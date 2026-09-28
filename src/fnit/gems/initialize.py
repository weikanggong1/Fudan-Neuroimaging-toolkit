"""Subject-space initialization for GEMS atlas meshes."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .atlas import GEMSAtlas
from .rasterize import rasterize_priors


def estimate_mask_affine(
    atlas_image,
    target_image,
    target_labels: np.ndarray,
    target_label_ids: tuple[int, ...] | list[int],
    *,
    device: str | torch.device = "cpu",
    max_iterations: int = 100,
) -> tuple[np.ndarray, float]:
    """Align an atlas intensity mask to selected coarse labels in target voxels.

    The atlas image header supplies orientation. A center-aligned affine is
    refined by differentiable trilinear resampling and soft Dice. This replaces
    the unstable global label-centroid fit for structure-specific atlas packs.
    """
    source = np.asarray(atlas_image.dataobj, dtype=np.float32)
    source = np.clip(source, 0, None)
    if source.ndim != 3 or not np.any(source):
        raise ValueError("atlas_image must contain a nonempty 3-D mask")
    source /= source.max()
    target = np.isin(target_labels, target_label_ids)
    source_points = np.argwhere(source > 0.5)
    target_points = np.argwhere(target)
    if len(source_points) < 4 or len(target_points) < 4:
        raise ValueError("atlas mask and target labels must each contain at least four voxels")
    source_center = source_points.mean(0)
    target_center = target_points.mean(0)
    margin = np.ceil(15 / np.linalg.norm(target_image.affine[:3, :3], axis=0)).astype(int)
    lower = np.maximum(target_points.min(0) - margin, 0)
    upper = np.minimum(target_points.max(0) + margin + 1, target.shape)
    region = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
    coordinates = np.stack(np.meshgrid(
        *[np.arange(s.start, s.stop) for s in region], indexing="ij"), -1)

    original = np.linalg.inv(target_image.affine) @ atlas_image.affine
    inverse_linear = np.linalg.inv(original[:3, :3])
    dtype = torch.float32
    xyz = torch.as_tensor(coordinates - target_center, device=device, dtype=dtype)
    reference = torch.as_tensor(target[region], device=device, dtype=dtype)
    template = torch.as_tensor(source[None, None], device=device, dtype=dtype)
    source_center_t = torch.as_tensor(source_center, device=device, dtype=dtype)
    source_extent = torch.as_tensor(np.asarray(source.shape) - 1, device=device, dtype=dtype)
    initial_linear = torch.as_tensor(inverse_linear, device=device, dtype=dtype)
    linear = torch.nn.Parameter(initial_linear.clone())
    shift = torch.nn.Parameter(torch.zeros(3, device=device, dtype=dtype))

    def soft_dice():
        locations = xyz @ linear.T + source_center_t + shift
        grid = (2 * locations / source_extent - 1)[..., [2, 1, 0]][None]
        moved = F.grid_sample(template, grid, align_corners=True)[0, 0]
        return (2 * (moved * reference).sum() + 1) / (moved.sum() + reference.sum() + 1)

    optimizer = torch.optim.LBFGS([linear, shift], lr=0.8,
                                   max_iter=int(max_iterations), line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        score = soft_dice()
        loss = 1 - score + 1e-4 * (linear - initial_linear).square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        score = float(soft_dice())
        inverse = np.eye(4)
        inverse[:3, :3] = linear.detach().cpu().numpy()
        inverse[:3, 3] = (source_center + shift.detach().cpu().numpy()
                           - inverse[:3, :3] @ target_center)
    return np.linalg.inv(inverse), score


def _centroids(labels: np.ndarray, ids) -> dict[int, np.ndarray]:
    result = {}
    for label in ids:
        xyz = np.argwhere(labels == int(label))
        if len(xyz):
            result[int(label)] = xyz.mean(0)
    return result


def estimate_label_centroid_affine(
    atlas: GEMSAtlas,
    target_labels: np.ndarray,
    *,
    device: str | torch.device = "cpu",
    min_shared_labels: int = 4,
    atlas_to_target_labels: dict[int, int] | None = None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Estimate atlas-voxel -> target-voxel affine from shared coarse labels.

    This is an FNIT initialization, not a claim of numerical equivalence to
    FreeSurfer's structure-specific alignment.  It lets the generic GEMS engine
    start directly from a SynthSeg/aseg-like native-space segmentation.
    """
    lo = np.floor(atlas.vertices.min(0)).astype(int)
    hi = np.ceil(atlas.vertices.max(0)).astype(int)
    shift = np.minimum(lo, 0)
    shape = tuple((hi - shift + 2).tolist())
    moved = atlas.transformed(np.array([
        [1,0,0,-shift[0]], [0,1,0,-shift[1]], [0,0,1,-shift[2]], [0,0,0,1]
    ], dtype=float), transform_reference=True)
    v = torch.as_tensor(moved.vertices, device=device, dtype=torch.float32)
    t = torch.as_tensor(moved.tetrahedra, device=device, dtype=torch.long)
    a = torch.as_tensor(moved.alphas, device=device, dtype=torch.float32)
    priors, _ = rasterize_priors(v, t, a, shape)
    hard = moved.label_ids[priors.argmax(0).cpu().numpy()]

    atlas_cent_raw = _centroids(hard, moved.label_ids)
    mapping = atlas_to_target_labels or {}
    atlas_cent: dict[int, np.ndarray] = {}
    for label, centroid in atlas_cent_raw.items():
        target_label = int(mapping.get(int(label), int(label)))
        if target_label not in atlas_cent:
            atlas_cent[target_label] = centroid
    target_cent = _centroids(np.asarray(target_labels), np.unique(target_labels))
    shared = sorted((set(atlas_cent) & set(target_cent)) - {0})
    # Fine subregion IDs are absent from a 33-class SynthSeg target by design;
    # the intersection naturally selects coarse surrounding anatomy.
    if len(shared) < min_shared_labels:
        raise ValueError(
            f"Only {len(shared)} shared atlas/target labels ({shared}); "
            f"need at least {min_shared_labels} or provide atlas_to_native_voxel.npy"
        )
    x = np.stack([atlas_cent[k] for k in shared])
    y = np.stack([target_cent[k] for k in shared])
    design = np.concatenate((x, np.ones((len(x), 1))), axis=1)
    coeff, *_ = np.linalg.lstsq(design, y, rcond=None)
    affine_moved = np.eye(4)
    affine_moved[:3, :] = coeff.T
    unshift = np.eye(4)
    unshift[:3, 3] = -shift
    # moved = unshift @ original; target = affine_moved @ moved
    return affine_moved @ unshift, tuple(shared)
