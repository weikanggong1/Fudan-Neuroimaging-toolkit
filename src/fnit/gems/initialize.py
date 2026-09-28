"""Subject-space initialization for GEMS atlas meshes."""

from __future__ import annotations

import numpy as np
import torch

from .atlas import GEMSAtlas
from .rasterize import rasterize_priors


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
