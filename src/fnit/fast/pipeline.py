"""Image-facing API for PyTorch T1 tissue segmentation."""

from dataclasses import dataclass
import os

import nibabel as nib
import numpy as np
import torch

from ..synthstrip.geometry import Volume, load_volume
from .algorithm import FASTConfig, segment_t1


@dataclass
class FASTResult:
    """Three-tissue partial volumes and bias-correction products."""

    pve_csf: Volume
    pve_gm: Volume
    pve_wm: Volume
    hard_segmentation: Volume
    pve_segmentation: Volume
    mixel_type: Volume
    bias_field: Volume
    restored: Volume
    tissue_means: tuple[float, float, float]
    tissue_variances: tuple[float, float, float]


class _NiftiFASTVolume(Volume):
    """Keep NIfTI spatial/unit metadata when writing FAST products."""

    def __init__(self, data, affine, header):
        super().__init__(data, affine)
        self._header = header.copy()

    def save(self, path):
        if str(path).lower().endswith((".nii", ".nii.gz")):
            header = self._header.copy()
            header.set_data_dtype(self.data.dtype)
            nib.save(nib.Nifti1Image(self.data, self.affine, header=header), str(path))
        else:
            super().save(path)


def _load_volume(value, name):
    if isinstance(value, (str, os.PathLike)):
        value = nib.load(str(value))
    if isinstance(value, nib.spatialimages.SpatialImage):
        volume = load_volume(value)
        if isinstance(value, nib.Nifti1Image):
            volume._nifti_header = value.header.copy()
        return volume
    if isinstance(value, Volume):
        return value
    # Existing in-memory Surfa callers remain usable without importing Surfa.
    if (hasattr(value, "data") and hasattr(value, "new") and
            hasattr(getattr(value, "geom", None), "vox2world")):
        return value
    raise TypeError(f"{name} must be a path or image volume")


def _single_frame(volume, name):
    data = np.asarray(volume.data)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{name} must contain one 3D frame")
    return data


class TorchFAST:
    """Reusable single-channel T1 HMRF-EM tissue estimator.

    The input should already be brain-extracted. If ``mask`` is omitted,
    positive input voxels define the brain. All image outputs preserve the
    input shape and voxel-to-world geometry.
    """

    def __init__(self, device="cpu", threads=None, *, init_iterations=15,
                 bias_iterations=4, fixed_iterations=4, bias_fwhm_mm=20.0,
                 init_mrf=0.02, mrf=0.1, mixel_mrf=0.3, pve_steps=100,
                 mean_field_iterations=5, pve_chunk_size=8):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if threads is not None:
            if not isinstance(threads, int) or threads < 1:
                raise ValueError("threads must be a positive integer")
            torch.set_num_threads(threads)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.config = FASTConfig(
            init_iterations=init_iterations,
            bias_iterations=bias_iterations,
            fixed_iterations=fixed_iterations,
            bias_fwhm_mm=bias_fwhm_mm,
            init_mrf=init_mrf,
            mrf=mrf,
            mixel_mrf=mixel_mrf,
            pve_steps=pve_steps,
            mean_field_iterations=mean_field_iterations,
            pve_chunk_size=pve_chunk_size,
        )

    @torch.inference_mode()
    def __call__(self, image, mask=None):
        image = _load_volume(image, "image")
        data = _single_frame(image, "image")
        affine = np.asarray(image.geom.vox2world.matrix, dtype=float)
        voxel_size = tuple(np.linalg.norm(affine[:3, :3], axis=0))

        mask_data = None
        if mask is not None:
            mask = _load_volume(mask, "mask")
            mask_data = _single_frame(mask, "mask")
            mask_affine = np.asarray(mask.geom.vox2world.matrix, dtype=float)
            if mask_data.shape != data.shape or not np.allclose(mask_affine, affine,
                                                                 atol=1e-5, rtol=0):
                raise ValueError("mask must use the same shape and geometry as image")

        tensor = torch.from_numpy(np.asarray(data, dtype=np.float32).copy()).to(self.device)
        mask_tensor = None
        if mask_data is not None:
            mask_tensor = torch.from_numpy(np.asarray(mask_data > 0).copy()).to(self.device)
        result = segment_t1(tensor, mask_tensor, voxel_size, self.config)

        def volume(value, dtype):
            array = value.detach().cpu().numpy().astype(dtype, copy=False)
            header = getattr(image, "_nifti_header", None)
            if header is not None:
                return _NiftiFASTVolume(array, affine, header)
            return image.new(array)

        means = tuple(float(value) for value in result.tissue_means.cpu())
        variances = tuple(float(value) for value in result.tissue_variances.cpu())
        return FASTResult(
            pve_csf=volume(result.pve[0], np.float32),
            pve_gm=volume(result.pve[1], np.float32),
            pve_wm=volume(result.pve[2], np.float32),
            hard_segmentation=volume(result.hard_segmentation, np.int32),
            pve_segmentation=volume(result.pve_segmentation, np.int32),
            mixel_type=volume(result.mixel_type, np.int32),
            bias_field=volume(result.bias_field, np.float32),
            restored=volume(result.restored, np.float32),
            tissue_means=means,
            tissue_variances=variances,
        )


__all__ = ["FASTResult", "TorchFAST"]
