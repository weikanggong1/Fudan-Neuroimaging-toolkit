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
    reference_scalar_affine=None,
    moving_tensor_affine=None,
    reference_tensor_affine=None,
    scalar_weights=None,
    auto_linear=True,
    device=None,
    config=None,
    overwrite=False,
) -> MMORFResult:
    """Register ordered scalar pairs plus a tensor pair and write the result.

    Scalar inputs are one 3D NIfTI image/path or equal-length ordered sequences
    of moving and reference images. The first reference defines warp space.
    Tensor inputs are a moving image and a common-space reference. Tensor images
    are 4D NIfTI images whose final dimension contains the six FSL components
    ``Dxx, Dxy, Dxz, Dyy, Dyz, Dzz``. Each optional affine is a finite 4x4
    FSL scaled-mm matrix (or path) from that input to the first reference.
    Scalar affines follow the input pair order. Missing matrices are estimated
    by TorchFLIRT when ``auto_linear=True``. Tensor registration uses FA made
    from the six input channels. ``scalar_weights`` is one non-negative weight
    per scalar pair, defaulting to ``1 / number_of_pairs``.

    ``output_dir`` receives ``mmorf_warp.nii.gz`` (three-frame relative pull
    displacement in millimetres along the reference image axes),
    ``mmorf_jacobian.nii.gz``,
    ``mmorf_warped_scalar.nii.gz``, ``mmorf_warped_tensor.nii.gz`` and
    ``mmorf_report.json``. The returned :class:`MMORFResult` contains the same
    four NIfTI images in memory plus ``warped_scalars`` and the report dict.
    Further scalar outputs use ``mmorf_warped_scalar_2.nii.gz`` and so on.
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
        reference_scalar_affine=reference_scalar_affine,
        moving_tensor_affine=moving_tensor_affine,
        reference_tensor_affine=reference_tensor_affine,
        scalar_weights=scalar_weights,
        auto_linear=auto_linear,
        output_dir=output_dir,
        overwrite=overwrite,
    )


__all__ = ["run_mmorf"]
