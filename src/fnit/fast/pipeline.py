"""Image-facing API for PyTorch T1 tissue segmentation."""

from dataclasses import dataclass
import numpy as np
import torch

from .._nib import FNITNifti1Image, load_image, new_image
from .algorithm import FASTConfig, segment_t1


@dataclass
class FASTResult:
    """Three-tissue partial volumes and bias-correction products."""

    pve_csf: FNITNifti1Image
    pve_gm: FNITNifti1Image
    pve_wm: FNITNifti1Image
    hard_segmentation: FNITNifti1Image
    pve_segmentation: FNITNifti1Image
    mixel_type: FNITNifti1Image
    bias_field: FNITNifti1Image
    restored: FNITNifti1Image
    tissue_means: tuple[float, float, float]
    tissue_variances: tuple[float, float, float]


def _load_volume(value, name):
    return load_image(value, name)


def _single_frame(volume, name):
    data = np.asanyarray(volume.dataobj)
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
                 mean_field_iterations=5, pve_chunk_size=8, execution="tensor"):
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
            execution=execution,
        )

    @torch.inference_mode()
    def __call__(self, image, mask=None):
        image = _load_volume(image, "image")
        data = _single_frame(image, "image")
        affine = np.asarray(image.affine, dtype=float)
        voxel_size = (tuple(image.header.get_zooms()[:3]) if self.config.execution == "fsl"
                      else tuple(np.linalg.norm(affine[:3, :3], axis=0)))

        mask_data = None
        if mask is not None:
            mask = _load_volume(mask, "mask")
            mask_data = _single_frame(mask, "mask")
            mask_affine = np.asarray(mask.affine, dtype=float)
            if mask_data.shape != data.shape or not np.allclose(mask_affine, affine,
                                                                 atol=1e-5, rtol=0):
                raise ValueError("mask must use the same shape and geometry as image")

        tensor = torch.from_numpy(np.asarray(data, dtype=np.float32).copy()).to(self.device)
        mask_tensor = None
        if mask_data is not None:
            mask_tensor = torch.from_numpy(np.asarray(mask_data > 0).copy()).to(self.device)
        # NEWIMAGE scans an internal radiological array. A neurological file is
        # reversed in x on read and reversed back on write; this changes the
        # direction of the ordered MRF updates but not the public image grid.
        internal_x_flip = self.config.execution == "fsl" and np.linalg.det(affine[:3, :3]) > 0
        if internal_x_flip:
            tensor = tensor.flip(0)
            if mask_tensor is not None:
                mask_tensor = mask_tensor.flip(0)
        result = segment_t1(tensor, mask_tensor, voxel_size, self.config)

        def volume(value, dtype):
            if internal_x_flip:
                value = value.flip(-3)
            array = value.detach().cpu().numpy().astype(dtype, copy=False)
            return new_image(array, image)

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
