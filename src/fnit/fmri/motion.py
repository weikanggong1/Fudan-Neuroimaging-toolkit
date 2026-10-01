"""MCFLIRT source-derived BOLD motion estimation using PyTorch.

The matrices map each moving frame to the reference in FSL scaled-mm axes.
This compatibility wrapper keeps its original reference-to-moving parameter
convention. TorchMCFLIRT.parameters exposes the source MCFLIRT .par convention.
"""

from dataclasses import dataclass
import os

import nibabel as nib
import numpy as np

from ..applywarp.core import _fsl_voxel_matrix
from ..flirt.core import fsl_parameters_from_affine


@dataclass
class MotionResult:
    parameters: np.ndarray
    fsl_matrices: np.ndarray
    corrected: nib.Nifti1Image | None
    reference: nib.Nifti1Image


def _load_image(value):
    image = nib.load(str(value)) if isinstance(value, (str, bytes, os.PathLike)) else value
    if not isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)):
        raise TypeError("input and reference must be NIfTI images or paths")
    return image


def estimate_motion(
    input_bold,
    reference,
    *,
    mask=None,
    device=None,
    batch_size=16,
    iterations=(1, 1, 1),
    resample=False,
):
    """按 MCFLIRT 2111.0 的 8/4/4 mm 路径估计六自由度运动。

    ``mask`` 仅核对网格；MCFLIRT 默认 cost 不使用脑掩膜。``iterations``
    为三个阶段的坐标轮回次数，官方默认各一次。
    ``batch_size`` 保留调用兼容，
    时间帧的初值传播按原程序顺序执行。
    """
    from ..mcflirt import TorchMCFLIRT
    input_image = _load_image(input_bold)
    reference_image = _load_image(reference)
    if batch_size < 1 or len(iterations) != 3 or any(value < 0 for value in iterations):
        raise ValueError("batch_size must be positive and iterations must have three nonnegative counts")
    if mask is not None:
        mask_image = _load_image(mask)
        if mask_image.shape != reference_image.shape or not np.allclose(
            mask_image.affine, reference_image.affine, atol=1e-4
        ):
            raise ValueError("mask must use the reference voxel grid")
    stage_iterations = tuple(iterations)
    result = TorchMCFLIRT(device=device).run(
        input_image, reference_image, stage_iterations=stage_iterations, resample=resample)
    # Existing callers may inspect the original pull-parameter convention.
    # Derive it from the newly fitted MCFLIRT matrices instead of fitting a
    # second motion model. The public MCFLIRT class returns source .par rows.
    sizes = np.asarray(input_image.header.get_zooms()[:3], dtype=np.float64)
    voxel_mm = np.diag([*sizes, 1.0])
    centre = (np.asarray(reference_image.shape, dtype=np.float64) - 1) / 2 * sizes
    output_parameters = []
    for matrix in result.matrices:
        pull = (voxel_mm @ np.linalg.inv(_fsl_voxel_matrix(input_image)) @
                np.linalg.inv(matrix) @ _fsl_voxel_matrix(reference_image) @ np.linalg.inv(voxel_mm))
        rotation = pull[:3, :3]
        angles = (np.arctan2(rotation[2, 1], rotation[2, 2]),
                  np.arctan2(-rotation[2, 0], np.hypot(rotation[2, 1], rotation[2, 2])),
                  np.arctan2(rotation[1, 0], rotation[0, 0]))
        translation = pull[:3, 3] - centre + rotation @ centre
        output_parameters.append(np.r_[angles, translation])
    return MotionResult(np.stack(output_parameters), result.matrices, result.corrected, result.reference)


def matrices_to_mcflirt_parameters(matrices, reference):
    """Convert FLIRT matrices to MCFLIRT rotation/radian and translation/mm rows.

    Translations use the reference image intensity-weighted centre, matching
    MCFLIRT's ``-plots`` convention. This was checked against same-run FSL
    matrices and `.par` on the real 490-frame UKB example.
    """
    image = _load_image(reference)
    if image.ndim != 3:
        raise ValueError("reference must be 3D")
    data = np.asarray(image.dataobj, dtype=np.float64)
    total = data.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("reference must have a positive finite intensity sum")
    from ..flirt.core import _centre_of_gravity, _flip_to_radiological
    import torch
    radiological = _flip_to_radiological(data.astype(np.float32), image.affine)
    sizes = tuple(float(value) for value in image.header.get_zooms()[:3])
    centre = _centre_of_gravity(torch.as_tensor(radiological), np.diag([*sizes, 1.0]))
    matrices = np.asarray(matrices, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
        raise ValueError("matrices must be a stack of 4x4 FLIRT transforms")
    return np.stack([fsl_parameters_from_affine(matrix, centre)[:6]
                     for matrix in matrices])
