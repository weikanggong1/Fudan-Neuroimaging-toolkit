"""MRtrix3 SIFT2 FMLS, ACT mask, track mapping and optimization on torch.

The source-derived components follow MRtrix3 3.0.3-103-g026e850d and are
covered by MPL-2.0; see ``THIRD_PARTY_NOTICES.md``. Tracking and FOD fitting
are separate stages. Same-track real-data validation is recorded in
``validation/connectome/ds004666``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .sift2_fixels import _directions, segment_fod_fixels
from .sift2_mapping import map_streamlines_to_fixels
from .sift2_optimizer import optimize_sift2_fixels
from .sift2_proc_mask import processing_mask_from_5tt


@torch.inference_mode()
def estimate_sift2_weights(
    paths: Sequence[torch.Tensor],
    wm_sh: torch.Tensor,
    fod_affine: torch.Tensor,
    five_tissue: torch.Tensor,
    five_tissue_affine: torch.Tensor,
    *,
    step_size_mm: float,
    processing_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return MRtrix-style SIFT2 weights for ordered world-mm streamlines.

    ``paths`` is a sequence of float32 ``[Pi,3]`` tensors in TCK order and
    RAS millimetres. ``wm_sh`` is normalized float32 FOD ``[X,Y,Z,45]``;
    ``fod_affine`` maps voxel centers to RAS mm. ``five_tissue`` is float32
    ``[A,B,C,5]`` on its own ``five_tissue_affine`` grid in cGM/sGM/WM/CSF/
    pathology order. All tensors share one CPU/CUDA device. ``step_size_mm``
    is the tracking step from the TCK header or ``tckgen`` configuration.
    The optional float32 processing mask ``[X,Y,Z]`` allows same-mask stage
    comparison; otherwise it is computed from 5TT by ACT supersampling.

    Returns float64 ``[len(paths)]`` per-track factors on the input device,
    including unassigned tracks. FMLS uses the packaged MRtrix 1281-direction
    sphere and true FOD lobes, precise Hermite voxel traversal uses 8-bit
    length quantization, and the default nonlinear optimization estimates
    proportionality coefficient mu internally. Equivalent command:
    ``tcksift2 tracks.tck wm_fod_norm.mif weights.txt -act 5tt_dwi.mif``.
    Fixed official TCK/FOD/5TT outputs should be compared by track index;
    independent probabilistic tractograms require distributional comparison.
    """
    if wm_sh.ndim != 4 or wm_sh.shape[-1] != 45 or wm_sh.dtype != torch.float32:
        raise ValueError("normalized WM FOD must be float32 [X,Y,Z,45]")
    if fod_affine.shape != (4, 4) or five_tissue_affine.shape != (4, 4):
        raise ValueError("FOD and 5TT affines must be 4x4")
    if five_tissue.ndim != 4 or five_tissue.shape[-1] != 5:
        raise ValueError("five_tissue must have shape [A,B,C,5]")
    if step_size_mm <= 0:
        raise ValueError("step_size_mm must be positive")
    device = wm_sh.device
    if any(t.device != device for t in (fod_affine, five_tissue, five_tissue_affine)):
        raise ValueError("FOD and 5TT tensors must share a device")
    if not paths:
        return torch.empty(0, device=device, dtype=torch.float64)
    if processing_mask is None:
        processing_mask = processing_mask_from_5tt(
            wm_sh, fod_affine, five_tissue, five_tissue_affine,
        )
    if processing_mask.shape != wm_sh.shape[:3] or processing_mask.device != device:
        raise ValueError("processing mask must match the FOD grid and device")
    fixels = segment_fod_fixels(wm_sh, processing_mask)
    if not len(fixels.voxel_ids):
        raise ValueError("FOD/ACT mask contains no fixels")
    directions = _directions(device)[0]
    mapped = map_streamlines_to_fixels(
        paths, fod_affine, wm_sh.shape[:3], fixels.voxel_ids,
        fixels.first_fixel_index, fixels.count, fixels.lookup_table,
        directions, step_size_mm=step_size_mm,
        n_fixels=len(fixels.fixel_integrals),
    )
    pm = torch.zeros_like(fixels.fixel_integrals)
    offsets = torch.arange(int(fixels.count.max()), device=device)
    available = offsets[None, :] < fixels.count.long()[:, None]
    indices = fixels.first_fixel_index[:, None] + offsets[None, :]
    voxel_values = processing_mask.reshape(-1)[fixels.voxel_ids.long()].to(pm.dtype)
    pm[indices[available]] = voxel_values[:, None].expand_as(indices)[available]
    solution = optimize_sift2_fixels(
        mapped.track_index, mapped.fixel_index, mapped.length_mm,
        fixels.fixel_integrals, pm, len(paths),
    )
    return solution.weights
