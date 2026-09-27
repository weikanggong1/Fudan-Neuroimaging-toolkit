"""Six-neighbour MRtrix maskfilter morphology for connectome brain masks."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..synthseg_parc.postprocess import largest_connected_component
from .response import _mrtrix_optimal_threshold


@torch.inference_mode()
def maskfilter_six_connected(
    mask: torch.Tensor, operation: str, passes: int = 2,
) -> torch.Tensor:
    """Apply MRtrix ``maskfilter`` six-neighbour erosion or dilation.

    Input is a bool 3D tensor ``[X,Y,Z]`` on CPU/CUDA; ``operation`` is
    ``"erode"`` or ``"dilate"``; ``passes`` is a non-negative integer.
    Output is a bool tensor of the same shape and device. Each pass includes
    the central voxel and its six face neighbours, with zero outside the
    image. The UKB-connectomics script uses two passes for the FOD dilation
    and mtnormalise erosion. Equivalent commands are ``maskfilter mask.mif
    dilate dilated.mif -npass 2`` and ``maskfilter mask.mif erode eroded.mif
    -npass 2`` (MRtrix3 3.0.3). No floating-point arithmetic is used.
    """
    if mask.ndim != 3 or mask.dtype != torch.bool:
        raise ValueError("mask must be bool [X,Y,Z]")
    if operation not in ("erode", "dilate") or passes < 0:
        raise ValueError("operation must be erode or dilate and passes nonnegative")
    current = mask
    for _ in range(passes):
        updated = current.clone()
        for axis in range(3):
            for shift in (-1, 1):
                neighbour = current.roll(shift, dims=axis)
                neighbour.select(axis, 0 if shift == 1 else -1).zero_()
                if operation == "erode":
                    updated &= neighbour
                else:
                    updated |= neighbour
        current = updated
    return current


@torch.inference_mode()
def dwi2mask_legacy(
    signal: torch.Tensor,
    bvalues: torch.Tensor,
    shell_bvalues: torch.Tensor,
) -> torch.Tensor:
    """Reproduce the default mask of pinned MRtrix ``dwi2mask legacy``.

    ``signal`` is corrected float32 DWI ``[X,Y,Z,N]`` on CPU/CUDA;
    ``bvalues`` is its MRtrix gradient-table b-value column ``[N]``;
    ``shell_bvalues`` is the ordered MRtrix shell mean ``[S]``. Returns a
    same-device bool ``[X,Y,Z]`` mask. The path is per-shell mean after
    nonnegative clipping, per-volume Ridgway correlation threshold, union,
    3x3x3 median, largest six-connected component, interior fill, and
    MRtrix two-scale bridge cleaning. The final mask excludes voxels with
    no positive DWI sample. Reference command: ``dwi2mask legacy dwi.mif
    mask.mif`` (pinned UKB-connectomics MRtrix commit eeab681).
    """
    if signal.ndim != 4 or signal.dtype != torch.float32:
        raise ValueError("signal must be float32 [X,Y,Z,N]")
    if signal.device.type == "cuda":
        torch.backends.cudnn.allow_tf32 = True
    bvalues = torch.as_tensor(bvalues, device=signal.device, dtype=torch.float64)
    shells = torch.as_tensor(shell_bvalues, device=signal.device, dtype=torch.float64)
    if bvalues.shape != (signal.shape[-1],) or shells.ndim != 1 or shells.numel() < 2:
        raise ValueError("bvalues/shell_bvalues do not match the DWI")
    if not bool(torch.isfinite(signal).all()):
        raise ValueError("DWI must be finite")
    assignment = (bvalues[:, None] - shells[None, :]).abs().argmin(dim=1)
    union = torch.zeros(signal.shape[:3], dtype=torch.bool, device=signal.device)
    for shell in range(shells.numel()):
        indices = assignment == shell
        if not bool(indices.any()):
            raise ValueError(f"shell {shell} has no DWI volumes")
        trace = signal[..., indices].clamp_min(0).mean(dim=-1)
        threshold = _mrtrix_optimal_threshold(trace, torch.ones_like(union))
        union |= trace > threshold

    kernel = torch.ones((1, 1, 3, 3, 3), device=signal.device)
    count = F.conv3d(union[None, None].float(), kernel, padding=1)[0, 0]
    available = F.conv3d(torch.ones_like(union, dtype=torch.float32)[None, None],
                         kernel, padding=1)[0, 0]
    median = count * 2 >= available
    current = largest_connected_component(median)
    current = ~largest_connected_component(~current)
    while True:
        prior = current
        work = current
        for scale in (2, 1):
            eroded = maskfilter_six_connected(work, "erode", scale)
            islands = eroded & ~largest_connected_component(eroded)
            bridge = maskfilter_six_connected(islands, "dilate", scale + 1)
            work = work & ~bridge
        current = largest_connected_component(work)
        if torch.equal(current, prior):
            return current & signal.amax(dim=-1).gt(0)
