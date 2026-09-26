"""Single-subject GPU diffusion-to-connectome path from corrected DWI and T1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from .assignment import build_connectomes
from .dti import fit_tensor_fa
from .fod import fit_three_tissue_csd
from .response import estimate_three_tissue_response
from .sift2 import estimate_sift2_weights
from .tracking import Tractogram, probabilistic_tractography


_WM_LABELS = {2, 41, 7, 46, 16}
_CSF_LABELS = {4, 5, 14, 15, 24, 43, 44}


def _image(path: str | Path, device: torch.device):
    image = nib.load(str(path))
    data = torch.from_numpy(np.asarray(image.get_fdata(dtype=np.float32))).to(device)
    affine = torch.as_tensor(image.affine, device=device, dtype=torch.float32)
    return data, affine


def _gradients(bvals_path: str | Path, bvecs_path: str | Path,
               n_volumes: int, affine: torch.Tensor, device: torch.device):
    bvals = torch.as_tensor(np.loadtxt(bvals_path).reshape(-1),
                            device=device, dtype=torch.float32)
    bvecs_array = np.loadtxt(bvecs_path)
    if bvecs_array.shape == (3, n_volumes):
        bvecs_array = bvecs_array.T
    if bvals.shape != (n_volumes,) or bvecs_array.shape != (n_volumes, 3):
        raise ValueError("bvals/bvecs must match the DWI volume count")
    bvecs = torch.as_tensor(bvecs_array, device=device, dtype=torch.float32)
    # MRtrix mrconvert -fslgrad flips FSL x for positive-determinant images
    # before mapping image-axis gradients to RAS world directions.
    if bool(torch.linalg.det(affine[:3, :3]) > 0):
        bvecs[:, 0] = -bvecs[:, 0]
    u, _, vh = torch.linalg.svd(affine[:3, :3])
    rotation = u @ vh
    bvecs = bvecs @ rotation.T
    nonzero = bvals >= 50
    bvecs[nonzero] = torch.nn.functional.normalize(bvecs[nonzero], dim=-1)
    bvecs[~nonzero] = 0
    if bool((torch.linalg.vector_norm(bvecs[nonzero], dim=-1) < 0.9).any()):
        raise ValueError("non-b0 bvecs must be nonzero")
    return bvals, bvecs


def _resample_labels(data: torch.Tensor, source_affine: torch.Tensor,
                     target_shape: tuple[int, int, int], target_affine: torch.Tensor,
                     target_to_source_world: torch.Tensor):
    """Nearest-neighbour label sampling on a target grid; all coordinates are RAS mm."""
    device = data.device
    ranges = [torch.arange(size, device=device) for size in target_shape]
    voxel = torch.stack(torch.meshgrid(*ranges, indexing="ij"), -1).reshape(-1, 3).to(torch.float32)
    target_world = voxel @ target_affine[:3, :3].T + target_affine[:3, 3]
    source_world = target_world @ target_to_source_world[:3, :3].T + target_to_source_world[:3, 3]
    inverse = torch.linalg.inv(source_affine)
    source_voxel = (source_world @ inverse[:3, :3].T + inverse[:3, 3]).round().long()
    inside = torch.ones(len(source_voxel), dtype=torch.bool, device=device)
    for axis, size in enumerate(data.shape):
        inside &= (source_voxel[:, axis] >= 0) & (source_voxel[:, axis] < size)
    safe = torch.stack([source_voxel[:, axis].clamp(0, size - 1)
                        for axis, size in enumerate(data.shape)], -1)
    sampled = data[safe[:, 0], safe[:, 1], safe[:, 2]]
    return torch.where(inside, sampled, torch.zeros_like(sampled)).reshape(target_shape)


def _tissue_labels(segmentation: torch.Tensor):
    labels = torch.zeros_like(segmentation, dtype=torch.int16)
    foreground = segmentation > 0
    labels[foreground] = 1
    for value in _WM_LABELS:
        labels[segmentation == value] = 2
    for value in _CSF_LABELS:
        labels[segmentation == value] = 3
    return labels


def _registration(b0: torch.Tensor, dwi_affine: torch.Tensor, t1_path: str | Path,
                  device: torch.device):
    # Reuse the package's GPU registration implementation. Its current
    # 12-DOF correlation-ratio path differs from the UKB script's 6-DOF NMI.
    import surfa as sf
    from ..flirt import TorchFLIRT
    b0_cpu = b0.detach().cpu().numpy()
    geometry = sf.ImageGeometry(shape=b0_cpu.shape, vox2world=dwi_affine.cpu().numpy())
    moving = sf.Volume(b0_cpu, geometry=geometry)
    result = TorchFLIRT(device=str(device))(moving, sf.load_volume(str(t1_path)))
    return torch.as_tensor(result.moving_to_fixed_world, device=device, dtype=torch.float32)


@dataclass
class ConnectomeResult:
    """One subject's computed intermediates and four region matrices."""

    matrices: dict[str, torch.Tensor]
    region_labels: tuple[int, ...]
    atlas: torch.Tensor
    tissues: torch.Tensor
    wm_sh: torch.Tensor
    fa: torch.Tensor
    tractogram: Tractogram
    sift2_weights: torch.Tensor
    dwi_affine: torch.Tensor
    atlas_affine: torch.Tensor
    dwi_to_t1_world: torch.Tensor


class UKBConnectome:
    """Compute a structural connectome from corrected DWI and paired T1.

    The DWI, bvals and eddy-rotated bvecs are expected to be preprocessed, as
    in UKB-connectomics. GPU calculations use float32 with TF32 enabled.
    """

    def __init__(self, device: str = "cuda:0", *, synthseg_weights=None):
        self.device = torch.device(device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.synthseg_weights = synthseg_weights

    @torch.inference_mode()
    def __call__(
        self,
        dwi: str | Path,
        bvals: str | Path,
        bvecs: str | Path,
        t1: str | Path,
        *,
        atlas_dwi: str | Path | None = None,
        t1_segmentation: str | Path | None = None,
        dwi_to_t1_world: np.ndarray | torch.Tensor | None = None,
        n_seeds: int = 10_000_000,
        seed: int = 0,
    ) -> ConnectomeResult:
        dwi_data, dwi_affine = _image(dwi, self.device)
        if dwi_data.ndim != 4:
            raise ValueError("DWI must have shape [X,Y,Z,N]")
        bvals_data, bvecs_data = _gradients(
            bvals, bvecs, dwi_data.shape[-1], dwi_affine, self.device
        )
        b0 = dwi_data[..., bvals_data < 50].mean(-1)
        if t1_segmentation is None:
            from ..synthseg_parc import SynthSeg
            segmentation = SynthSeg(weights=self.synthseg_weights, device=str(self.device))(
                t1, keep_geometry=True
            ).segmentation
            seg_data = torch.as_tensor(np.asarray(segmentation.data).copy(),
                                       device=self.device, dtype=torch.int32)
            seg_affine = torch.as_tensor(segmentation.geom.vox2world.matrix,
                                         device=self.device, dtype=torch.float32)
        else:
            seg_data, seg_affine = _image(t1_segmentation, self.device)
            seg_data = seg_data.to(torch.int32)
            if seg_data.ndim != 3:
                raise ValueError("T1 segmentation must be a 3D label image")
        if dwi_to_t1_world is None:
            transform = _registration(b0, dwi_affine, t1, self.device)
        else:
            transform = torch.as_tensor(dwi_to_t1_world, device=self.device,
                                         dtype=torch.float32)
            if transform.shape != (4, 4):
                raise ValueError("dwi_to_t1_world must be 4x4")
        native_seg = _resample_labels(
            seg_data, seg_affine, tuple(dwi_data.shape[:3]), dwi_affine, transform
        )
        tissues = _tissue_labels(native_seg)
        if atlas_dwi is None:
            # The default is a compact SynthSeg structure atlas. Explicit
            # UKB cortical+Tian atlases can be supplied in the DWI grid.
            atlas_source = torch.where(tissues == 1, native_seg, 0)
            region_labels = tuple(int(value) for value in
                                  torch.unique(atlas_source[atlas_source > 0]).tolist())
            atlas = torch.zeros_like(native_seg, dtype=torch.int32)
            atlas_affine = dwi_affine
            for index, label in enumerate(region_labels, 1):
                atlas[atlas_source == label] = index
        else:
            atlas_data, atlas_affine = _image(atlas_dwi, self.device)
            if atlas_data.ndim != 3:
                raise ValueError("atlas_dwi must be a 3D image in DWI world space")
            if not bool(torch.isfinite(atlas_data).all()) or not bool(
                torch.equal(atlas_data, atlas_data.round())
            ) or bool((atlas_data < 0).any()):
                raise ValueError("atlas_dwi must contain finite non-negative integer labels")
            atlas = atlas_data.to(torch.int32)
            region_labels = tuple(range(1, int(atlas.max()) + 1))
        if not bool((atlas > 0).any()):
            raise ValueError("no atlas regions overlap the DWI grid")
        fa = fit_tensor_fa(dwi_data, bvals_data, bvecs_data, mask=tissues > 0)
        shell_bvals, wm_response, gm_response, csf_response = estimate_three_tissue_response(
            dwi_data, bvals_data, bvecs_data, tissues, fa, lmax=4
        )
        wm_sh, _, _ = fit_three_tissue_csd(
            dwi_data, bvals_data, bvecs_data,
            shell_bvals, wm_response, gm_response, csf_response,
            mask=tissues > 0, lmax=4,
        )
        tractogram = probabilistic_tractography(
            wm_sh, dwi_affine, tissues, n_seeds=n_seeds, fa=fa, seed=seed, lmax=4
        )
        sift2_weights = estimate_sift2_weights(
            tractogram.paths, wm_sh, dwi_affine, lmax=4
        )
        matrices = build_connectomes(
            tractogram.endpoints, atlas, atlas_affine,
            weights=sift2_weights, lengths=tractogram.lengths_mm,
            fa=tractogram.mean_fa,
        )
        return ConnectomeResult(
            matrices, region_labels, atlas, tissues, wm_sh, fa,
            tractogram, sift2_weights, dwi_affine, atlas_affine, transform
        )
