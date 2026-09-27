"""Public single-subject function for PyTorch MMORF registration."""

from __future__ import annotations

from .core import MMORFConfig, MMORFResult, TorchMMORF


def run_mmorf(
    moving_scalar,
    reference_scalar,
    moving_tensor,
    reference_tensor,
    *,
    output_dir,
    moving_scalar_affine=None,
    moving_tensor_affine=None,
    reference_tensor_affine=None,
    device=None,
    config=None,
    overwrite=False,
) -> MMORFResult:
    """Register one scalar/tensor pair and write the complete result.

    Inputs are a moving 3D scalar image, the 3D common-space scalar reference,
    a moving tensor image, and a common-space tensor reference. Tensor images
    are 4D NIfTI images whose final dimension contains the six FSL components
    ``Dxx, Dxy, Dxz, Dyy, Dyz, Dzz``. Images may be NIfTI paths or nibabel
    images. Each optional affine is a path or array containing a finite 4x4
    FSL scaled-mm matrix from that input image to ``reference_scalar``.

    ``output_dir`` receives ``mmorf_warp.nii.gz`` (three-frame relative pull
    displacement in millimetres along the reference image axes),
    ``mmorf_jacobian.nii.gz``,
    ``mmorf_warped_scalar.nii.gz``, ``mmorf_warped_tensor.nii.gz`` and
    ``mmorf_report.json``. The returned :class:`MMORFResult` contains the same
    four NIfTI images in memory plus the report dictionary.
    """
    model = TorchMMORF(
        device=device,
        config=MMORFConfig() if config is None else config,
    )
    return model.run(
        moving_scalar,
        reference_scalar,
        moving_tensor,
        reference_tensor,
        moving_scalar_affine=moving_scalar_affine,
        moving_tensor_affine=moving_tensor_affine,
        reference_tensor_affine=reference_tensor_affine,
        output_dir=output_dir,
        overwrite=overwrite,
    )


__all__ = ["run_mmorf"]
