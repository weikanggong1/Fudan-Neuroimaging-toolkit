import json

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.topup import (
    FSL_TOPUP_CUBIC_SPLINE_COEFFICIENTS,
    TOPUPConfig,
    TorchTOPUP,
    make_topup_coefficient_image,
    make_topup_jacobian_image,
    prepare_ukb_topup,
)
from fnit.fnirt.spline import fsl_control_shape
from fnit.topup.core import (
    _cubic_spline_coefficients,
    _gaussian_blur,
    _grid,
    _sample,
    _sample_cubic,
)


def test_topup_coefficient_header_contract():
    shape = (12, 10, 8)
    spacing = (2, 2, 2)
    coefficients = np.zeros(fsl_control_shape(shape, spacing), np.float32)
    image = make_topup_coefficient_image(
        coefficients, shape, (2.0, 2.0, 2.0), spacing
    )
    assert int(image.header["intent_code"]) == FSL_TOPUP_CUBIC_SPLINE_COEFFICIENTS
    assert tuple(image.shape) == fsl_control_shape(shape, spacing)
    assert tuple(image.header["pixdim"][1:4]) == spacing
    offsets = tuple(
        int(round(float(image.header[name])))
        for name in ("qoffset_x", "qoffset_y", "qoffset_z")
    )
    assert offsets == shape


def test_topup_coefficient_header_survives_nifti_roundtrip(tmp_path):
    shape = (12, 10, 8)
    spacing = (2, 2, 2)
    coefficients = np.zeros(fsl_control_shape(shape, spacing), np.float32)
    image = make_topup_coefficient_image(
        coefficients, shape, (2.0, 2.0, 2.0), spacing
    )
    path = tmp_path / "fieldcoef.nii.gz"
    nib.save(image, path)
    header = nib.load(path).header
    assert int(header["intent_code"]) == FSL_TOPUP_CUBIC_SPLINE_COEFFICIENTS
    assert tuple(header["pixdim"][1:4]) == spacing
    assert int(header["qform_code"]) == 1
    assert int(header["sform_code"]) == 0
    assert tuple(float(header[name]) for name in (
        "qoffset_x", "qoffset_y", "qoffset_z"
    )) == shape


def test_topup_jacobian_header_matches_fsl_analyze_geometry(tmp_path):
    reference = nib.Nifti1Image(
        np.zeros((12, 10, 8), np.float32),
        np.diag((-2.0, 2.0, 2.0, 1.0)),
    )
    image = make_topup_jacobian_image(np.ones((12, 10, 8)), reference)
    path = tmp_path / "jacobian.nii.gz"
    nib.save(image, path)
    loaded = nib.load(path)
    assert int(loaded.header["qform_code"]) == 0
    assert int(loaded.header["sform_code"]) == 0
    assert tuple(loaded.header["pixdim"][1:4]) == (2.0, 2.0, 2.0)
    assert tuple(loaded.header["pixdim"][4:8]) == (1.0, 1.0, 1.0, 1.0)


def test_ukb_preparation_selects_b0_and_writes_acqparams(tmp_path):
    raw = tmp_path / "raw"
    output = tmp_path / "fieldmap"
    raw.mkdir()
    affine = np.diag((2, 2, 2, 1))
    base = np.arange(8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
    ap = np.stack((base, np.zeros_like(base), base * 1.01), axis=3)
    pa = np.stack((base[::-1], base[::-1] * 1.01), axis=3)
    for stem, data, direction in (("AP", ap, "j-"), ("PA", pa, "j")):
        nib.save(nib.Nifti1Image(data, affine), raw / f"{stem}.nii.gz")
        np.savetxt(raw / f"{stem}.bval", np.arange(data.shape[3])[None] * 1000)
        (raw / f"{stem}.json").write_text(
            json.dumps(
                {
                    "PhaseEncodingDirection": direction,
                    "EffectiveEchoSpacing": 0.00123456,
                }
            )
        )
    prepared = prepare_ukb_topup(raw, output)
    assert nib.load(prepared["imain"]).shape == (8, 8, 8, 2)
    np.testing.assert_allclose(
        np.loadtxt(prepared["datain"]),
        ((0, -1, 0, 0.0086), (0, 1, 0, 0.0086)),
    )


def test_topup_rejects_unsupported_config_length():
    with pytest.raises(ValueError, match="nine"):
        TOPUPConfig(warp_resolution_mm=(4,))


def test_topup_cuda_flag_is_checked(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        TorchTOPUP("cuda:0")


def test_identity_grid_sampling_preserves_xyz_array():
    volume = torch.arange(5 * 6 * 7, dtype=torch.float32).reshape(5, 6, 7)
    sampled, valid = _sample(
        volume, _grid(volume.shape, device=volume.device, dtype=volume.dtype)
    )
    torch.testing.assert_close(sampled, volume)
    assert bool(valid.all())


def test_fsl_rigid_pull_uses_translation_then_rotation_order():
    from fnit.topup.core import _rigid_coordinates

    grid = _grid((5, 6, 7), device="cpu", dtype=torch.float32)
    parameters = torch.tensor([2.0, 0, 0, 0, 0, 0])
    pulled = _rigid_coordinates(grid, parameters, (2.0, 2.0, 2.0))
    torch.testing.assert_close(pulled[0], grid[0] - 1)
    torch.testing.assert_close(pulled[1:], grid[1:])


def test_gaussian_blur_preserves_non_cubic_shape():
    images = torch.zeros((2, 12, 10, 8), dtype=torch.float32)
    blurred = _gaussian_blur(images, 4.0, (2.0, 2.0, 2.0))
    assert blurred.shape == images.shape


def test_cubic_sampler_reconstructs_integer_grid():
    volume = torch.arange(6 * 8 * 10, dtype=torch.float32).reshape(6, 8, 10)
    coefficients = _cubic_spline_coefficients(volume[None])[0]
    sampled, valid = _sample_cubic(
        coefficients,
        _grid(volume.shape, device=volume.device, dtype=volume.dtype),
        phase_encode_axis=1,
    )
    torch.testing.assert_close(sampled, volume, atol=5e-4, rtol=1e-6)
    assert bool(valid.all())
