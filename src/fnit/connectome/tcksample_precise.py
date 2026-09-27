"""MRtrix ``tcksample -precise -stat_tck mean`` on PyTorch CPU or GPU.

The precise voxel crossing is shared with SIFT2 mapping. MRtrix source:
https://github.com/MRtrix3/mrtrix3/blob/026e850d/cmd/tcksample.cpp
(MPL-2.0; see THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

from typing import Sequence

import torch

from .sift2_mapping import _upsample_tracks, _voxelise_precise


@torch.inference_mode()
def sample_streamline_mean_precise(
    streamlines: Sequence[torch.Tensor], scalar_image: torch.Tensor, affine: torch.Tensor,
) -> torch.Tensor:
    """Return float32 per-track length-weighted image mean [T] in TCK order.

    Each streamline is float32 world-mm coordinates [Ni,3], Ni>=2, on one
    device. ``scalar_image`` is float32 [X,Y,Z]; ``affine`` is float32 or
    float64 [4,4] mapping voxel centers to world mm. All inputs share a
    device; output stays on that device. Tracks with no image voxel visit
    produce NaN, as in MRtrix. Image values are sampled at precise voxel
    passages, without trilinear interpolation or streamline upsampling.

    Equivalent original command: ``tcksample -precise -stat_tck mean
    tracks_10000.tck fa_corrected.mif streamline_mean_fa.txt -nthreads 8``.
    """
    if not streamlines or scalar_image.ndim != 3 or affine.shape != (4, 4):
        raise ValueError("expected nonempty streamlines, scalar image [X,Y,Z], and affine [4,4]")
    device = scalar_image.device
    if scalar_image.dtype != torch.float32 or any(
        path.device != device or path.dtype != torch.float32 or path.ndim != 2
        or path.shape[1] != 3 or len(path) < 2 for path in streamlines
    ) or affine.device != device:
        raise ValueError("paths and image must be float32, well shaped and share one device")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    points, track, starts = _upsample_tracks(streamlines, 1)
    voxel, track, length, _ = _voxelise_precise(points, track, starts, affine, scalar_image.shape)
    value = scalar_image[voxel[:, 0], voxel[:, 1], voxel[:, 2]].double()
    numerator = torch.zeros(len(streamlines), device=device, dtype=torch.float64).index_add_(
        0, track, length * value,
    )
    denominator = torch.zeros_like(numerator).index_add_(0, track, length)
    return (numerator / denominator).float()
