"""MRtrix 026e850d SIFT2 processing mask from an ACT five-tissue image."""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.inference_mode()
def processing_mask_from_5tt(
    wm_sh: torch.Tensor,
    fod_affine: torch.Tensor,
    five_tt: torch.Tensor,
    five_tt_affine: torch.Tensor,
    *,
    batch_voxels: int = 2048,
) -> torch.Tensor:
    """Return the ACT SIFT2 processing mask as float32 ``[X,Y,Z]``.

    ``wm_sh`` is float32 WM FOD SH data on the DWI grid. ``five_tt`` has
    cGM, sGM, WM, CSF, pathology fractions in its last dimension. Both
    affines map voxel indices to scanner millimetres. Inputs share a device.
    ``batch_voxels`` bounds peak memory for the 10x10x10 subvoxel sampling.

    This follows ``tcksift2 tracks.tck wm_fod.mif weights.txt -act 5tt.mif``:
    5TT samples are trilinearly interpolated, ACT-classified, and the WM
    class fraction is squared. Equal shapes use direct WM fractions only when
    affines also agree, preventing a same-shape registration mistake. See
    ``validation/connectome/ds004666/sift2_proc_mask_real_fod.json``.
    """
    if wm_sh.ndim != 4 or five_tt.ndim != 4 or five_tt.shape[-1] != 5:
        raise ValueError("expected WM SH [X,Y,Z,C] and 5TT [X,Y,Z,5]")
    if wm_sh.device != five_tt.device or batch_voxels < 1:
        raise ValueError("WM FOD and 5TT must share a device; batch_voxels must be positive")
    if fod_affine.shape != (4, 4) or five_tt_affine.shape != (4, 4):
        raise ValueError("affines must be 4x4")
    same_space = tuple(wm_sh.shape[:3]) == tuple(five_tt.shape[:3]) and torch.allclose(
        fod_affine.to(device=wm_sh.device, dtype=torch.float64),
        five_tt_affine.to(device=wm_sh.device, dtype=torch.float64),
        rtol=0.0, atol=1e-4,
    )
    if same_space:
        wm = five_tt[..., 2]
        return torch.where(torch.isfinite(wm), wm.square(), 0.0).to(torch.float32)

    device = wm_sh.device
    shape = wm_sh.shape[:3]
    out = torch.zeros(shape, device=device, dtype=torch.float32)
    voxels = torch.nonzero(torch.isfinite(wm_sh[..., 0]) & (wm_sh[..., 0] != 0))
    if not len(voxels):
        return out
    volume = five_tt.permute(3, 2, 1, 0).unsqueeze(0).contiguous().to(torch.float32)
    fod_to_tt = torch.linalg.inv(five_tt_affine.to(device=device, dtype=torch.float32)) @ fod_affine.to(device=device, dtype=torch.float32)
    axis = (torch.arange(10, device=device, dtype=torch.float32) - 4.5) / 10.0
    subvoxels = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(1000, 3)
    tt_size = torch.tensor(five_tt.shape[:3], device=device, dtype=torch.float32)
    for voxel_batch in voxels.split(batch_voxels):
        position = voxel_batch.to(torch.float32)[:, None, :] + subvoxels[None, :, :]
        mapped = position @ fod_to_tt[:3, :3].T + fod_to_tt[:3, 3]
        in_bounds = ((mapped >= 0) & (mapped <= tt_size - 1)).all(dim=-1)
        grid = (mapped * (2.0 / (tt_size - 1)) - 1.0).reshape(1, 1, 1, -1, 3)
        samples = F.grid_sample(volume, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        samples = samples.reshape(5, len(voxel_batch), 1000).permute(1, 2, 0).clamp(0.0, 1.0)
        valid = in_bounds & torch.isfinite(samples).all(dim=-1) & (samples.sum(dim=-1) >= 0.5)
        wm = samples[..., 2]
        wm_class = valid & (wm > samples[..., 0]) & (wm > samples[..., 1]) & (wm > samples[..., 3]) & (wm > samples[..., 4])
        total = valid.sum(dim=-1)
        fraction = torch.where(total > 500, wm_class.sum(dim=-1) / total.clamp_min(1), 0.0)
        out[voxel_batch[:, 0], voxel_batch[:, 1], voxel_batch[:, 2]] = fraction.square()
    return out
